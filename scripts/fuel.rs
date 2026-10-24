#!/usr/bin/env -S cargo -Zscript
---
[package]
edition = "2024"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
wasmi = { version = "2", features = ["simd"] }

# cargo-script builds the dev profile.
[profile.dev]
opt-level = 3
debug = false
debug-assertions = false
overflow-checks = false

[lints.rust]
rust_2018_idioms = { level = "deny", priority = -1 }
unused_qualifications = "deny"

[lints.clippy]
pedantic = { level = "warn", priority = -1 }
missing_assert_message = "deny"
wildcard_enum_match_arm = "deny"
---
//! Count instructions executed inside the WebAssembly module.
//!
//! ```console
//! FEATURES=bench scripts/build-wasm.sh
//! cargo -Zscript scripts/fuel.rs [path/to/johari.bench.wasm] [seeds=8] [nItems=127] [nActive=100] [checkpoints=20,60,120,180] [markNoise=1] [careless=0.075]
//! ```

use serde::Deserialize;
use std::env::args;
use std::error::Error;
use std::fmt::{Display, Write as _};
use std::fs::read;
use std::iter::zip;
use std::ops::Add;
use std::process::Command;
use std::thread::scope;
use wasmi::{Config, Engine, Linker, Module, Store, TypedFunc};

type Fallible<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// A session's log.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Log {
    n_items: usize,
    tie_cost: f64,
    marks: Vec<(usize, Option<i32>)>,
    pairs: Vec<(usize, usize, Option<i32>)>,
}

/// The output of `node web/replay.mjs log`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Simulated {
    wasm: String,
    workload: Workload,
    review_level: f64,
    phases: Vec<String>,
    logs: Vec<Log>,
}

/// The simulated sessions' workload.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Workload {
    seeds: usize,
    checkpoints: Vec<usize>,
    n_items: usize,
    n_active: usize,
    mark_noise: f64,
    careless: f64,
}

/// The simulated sessions of the workload given by `args`.
fn simulated_sessions(args: impl IntoIterator<Item = String>) -> Fallible<Simulated> {
    let output = Command::new("node")
        .args(["web/replay.mjs", "log"])
        .args(args)
        .output()
        .map_err(|error| format!("running node for the simulated sessions: {error}"))?;
    if !output.status.success() {
        return Err(format!("web/replay.mjs log failed: {}", String::from_utf8_lossy(&output.stderr).trim_end()).into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// A module instance with a session and exports called by the harness.
struct Host {
    store: Store<()>,
    session: i32,
    mark: TypedFunc<(i32, i32, i32), i32>,
    propose: TypedFunc<i32, i32>,
    record: TypedFunc<(i32, i32, i32, i32), i32>,
    review: TypedFunc<(i32, f64), i32>,
    bench: Option<TypedFunc<(i32, i32, i32), f64>>,
}

impl Host {
    const TANK: u64 = u64::MAX / 4;

    /// A new instance of `module` with a session over `n_items` at the given tie cost.
    fn new(engine: &Engine, module: &Module, n_items: usize, tie_cost: f64) -> Fallible<Self> {
        let mut store = Store::new(engine, ());
        store.set_fuel(Self::TANK)?;
        let instance = Linker::new(engine).instantiate_and_start(&mut store, module)?;
        let new = instance.get_typed_func(&store, "johari_new")?;
        let session = new.call(&mut store, (i32::try_from(n_items)?, tie_cost))?;
        Ok(Self {
            mark: instance.get_typed_func(&store, "johari_mark")?,
            propose: instance.get_typed_func(&store, "johari_propose")?,
            record: instance.get_typed_func(&store, "johari_record")?,
            review: instance.get_typed_func(&store, "johari_review")?,
            bench: instance.get_typed_func(&store, "johari_bench").ok(),
            store,
            session,
        })
    }

    /// The fuel consumed by `f`.
    fn fuel(&mut self, f: impl FnOnce(&mut Self) -> Fallible<()>) -> Fallible<u64> {
        self.store.set_fuel(Self::TANK)?;
        f(self)?;
        Ok(Self::TANK - self.store.get_fuel()?)
    }

    fn mark(&mut self, item: usize, bucket: Option<i32>) -> Fallible<()> {
        expect_ok(self.mark.call(&mut self.store, (self.session, i32::try_from(item)?, bucket.unwrap_or(-1)))?)
    }

    fn propose(&mut self) -> Fallible<()> {
        expect_ok(self.propose.call(&mut self.store, self.session)?)
    }

    fn record(&mut self, a: usize, b: usize, outcome: Option<i32>) -> Fallible<()> {
        let pair = (i32::try_from(a)?, i32::try_from(b)?);
        expect_ok(self.record.call(&mut self.store, (self.session, pair.0, pair.1, outcome.unwrap_or(-1)))?)
    }

    fn review(&mut self, level: f64) -> Fallible<()> {
        expect_ok(self.review.call(&mut self.store, (self.session, level))?)
    }

    fn bench(&mut self, phase: i32, iters: i32) -> Fallible<()> {
        let run = self.bench.as_ref().ok_or("this module was built without --features bench")?;
        run.call(&mut self.store, (self.session, phase, iters))?;
        Ok(())
    }
}

fn expect_ok(code: i32) -> Fallible<()> {
    if code == 0 { Ok(()) } else { Err(format!("the module returned error {code}").into()) }
}

/// A row of the report.
fn row(report: &mut String, name: &str, cells: impl IntoIterator<Item = impl Display>) -> Fallible<()> {
    write!(report, "{name:<40}")?;
    for cell in cells {
        write!(report, "{cell:>14}")?;
    }
    Ok(writeln!(report)?)
}

struct Counts {
    calls: [Vec<u64>; 3],
    phases: Vec<Vec<u64>>,
    first_proposals: u64,
    proposals: u64,
}

impl Add for Counts {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        for (sums, counts) in zip(&mut self.calls, other.calls).chain(zip(&mut self.phases, other.phases)) {
            for (sum, count) in zip(sums, counts) {
                *sum += count;
            }
        }
        self.first_proposals += other.first_proposals;
        self.proposals += other.proposals;
        self
    }
}

fn replay(engine: &Engine, module: &Module, log: &Log, checkpoints: &[usize], review_level: f64, n_phases: usize) -> Fallible<Counts> {
    let mut host = Host::new(engine, module, log.n_items, log.tie_cost)?;
    let mut phases = vec![vec![0; checkpoints.len()]; n_phases];
    let mut calls = std::array::from_fn(|_| vec![0; checkpoints.len()]);
    let (mut first_proposals, mut proposals) = (0, 0);
    for &(item, bucket) in &log.marks {
        host.mark(item, bucket)?;
    }
    for answered in 0..=log.pairs.len() {
        let propose = host.fuel(Host::propose)?;
        if let Some(c) = checkpoints.iter().position(|&checkpoint| checkpoint == answered) {
            let cached = host.fuel(Host::propose)?;
            let review = host.fuel(|h| h.review(review_level))?;
            for (counts, count) in zip(&mut calls, [propose, cached, review]) {
                counts[c] += count;
            }
            // Each phase prepares its inputs before looping through iterations, so the difference is one iteration.
            for (phase, counts) in zip(0.., &mut phases) {
                let one = host.fuel(|h| h.bench(phase, 1))?;
                let two = host.fuel(|h| h.bench(phase, 2))?;
                counts[c] += two - one;
            }
        }
        if let Some(&(a, b, outcome)) = log.pairs.get(answered) {
            if answered == 0 {
                first_proposals += propose;
            }
            proposals += propose;
            host.record(a, b, outcome)?;
        }
    }
    Ok(Counts { calls, phases, first_proposals, proposals })
}

fn main() -> Fallible<()> {
    let Simulated { wasm, workload, review_level, phases: names, logs } = simulated_sessions(args().skip(1))?;
    let Workload { seeds, checkpoints, n_items, n_active, mark_noise, careless } = workload;
    let answers = *checkpoints.iter().max().ok_or("no checkpoints")?;

    let mut config = Config::default();
    config.consume_fuel(true).wasm_simd(true);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, read(&wasm)?)?;
    let counts = scope(|scope| {
        let (engine, module, checkpoints, n_phases) = (&engine, &module, &checkpoints, names.len());
        let threads: Vec<_> =
            logs.iter().map(|log| scope.spawn(move || replay(engine, module, log, checkpoints, review_level, n_phases))).collect();
        zip(1.., threads)
            .map(|(seed, thread)| thread.join().unwrap_or_else(|_| Err(format!("the replay of seed {seed} panicked").into())))
            .collect::<Fallible<Vec<_>>>()
    })?;
    let Counts { calls, phases, first_proposals, proposals } = counts.into_iter().reduce(Counts::add).ok_or("no sessions")?;

    let mut report = String::new();
    writeln!(report, "wasm: {wasm}")?;
    writeln!(report, "n_items={n_items} n_active={n_active} mark noise {mark_noise} careless {careless} seeds 1..={seeds}")?;
    row(&mut report, "phase", checkpoints.iter().map(|checkpoint| format!("@{checkpoint}")))?;
    for (name, counts) in zip(&names, &phases) {
        row(&mut report, name, counts)?;
    }
    writeln!(report, "call")?;
    let review = format!("johari_review, level {review_level}, fit cached");
    let labels = ["johari_propose, fit pending", "johari_propose, fit and scorer cached", &review];
    for (label, counts) in zip(labels, &calls) {
        row(&mut report, label, counts)?;
    }
    writeln!(report, "session (summed over the seeds)")?;
    row(&mut report, "first proposal (no answers yet)", [first_proposals])?;
    row(&mut report, &format!("every proposal through {answers} answers"), [proposals])?;
    print!("{report}");
    Ok(())
}
