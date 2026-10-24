//! Data read by harness logic.

use super::{Fit, Session};
use crate::observations::{Observations, PairSet};
use crate::posterior::{Bench, Posterior, fit, fit_diagnostics as fit_diag};
use crate::rank_stats::{Level, PairWeights, rank_intervals};
use crate::ranking::{RankedList, TieCost};
use crate::rng::Rng;
use crate::selection::Scorer;
use core::hint::black_box;

/// Data read by a benchmark phase.
struct Inputs<'a> {
    post: &'a Posterior,
    obs: &'a Observations,
    shown: &'a PairSet,
    tie_cost: TieCost,
}

type Body = fn(&Inputs<'_>, u32) -> f64;

const LEVEL: Level = Level::new(0.8).unwrap();

const PHASES: [(&str, Body); 9] = [
    // W.
    ("pair weights", |c, iters| repeat(iters, || f64::from(PairWeights::new(c.post).get(0, 0)))),
    // W and Σ·M·Σ.
    ("scorer", |c, iters| {
        repeat(iters, || {
            black_box(Scorer::new(c.post, c.tie_cost));
            0.
        })
    }),
    // The scorer and the scan.
    ("whole proposal", |c, iters| {
        repeat(iters, || {
            let scorer = Scorer::new(c.post, c.tie_cost);
            f64::from(scorer.propose(c.post, c.shown, &mut Rng::new(0)).0.items()[0])
        })
    }),
    // The first sweep of a fit.
    ("EP sweep", |c, iters| {
        let sweep = Bench::new(c.obs);
        repeat(iters, || sweep.sweep())
    }),
    // Λ, its factor, Λ⁻¹'s selected entries, μ, and the sites' marginals.
    ("q from the sites", |c, iters| {
        let sweep = Bench::new(c.obs);
        repeat(iters, || sweep.moments())
    }),
    // The full inverse as computed by `Posterior::new`.
    ("covariance from the factor", |c, iters| repeat(iters, || c.post.chol_precision.inverse().get(0, 0))),
    ("whole fit", |c, iters| repeat(iters, || fit(c.obs).mean_theta()[0])),
    // W and B, then the final list's order and tie groups.
    ("list", |c, iters| {
        repeat(iters, || {
            let list = RankedList::new(c.post, c.tie_cost);
            f64::from(list.group_sizes[0]) + f64::from(list.order[0])
        })
    }),
    // Each item's interval on rank at `LEVEL`.
    ("rank intervals", |c, iters| {
        repeat(iters, || {
            black_box(rank_intervals(c.post, LEVEL));
            0.
        })
    }),
];

/// Run the given phase `iters` times over the session's current posterior. Returns the sum of the runs' results.
///
/// # Panics
///
/// Panics if the given phase index is out of bounds or the session is exhausted (no pairs left to score).
#[must_use]
pub fn run(s: &mut Session, phase: u32, iters: u32) -> f64 {
    let body = PHASES[phase as usize].1;
    let n_active = s.active().len();
    let shown = s.shown_pairs(n_active).expect("a pair is left to score");
    let Fit { posterior: post, obs, tie_cost, .. } = s.fit();
    body(&Inputs { post, obs, shown: &shown, tie_cost: *tie_cost }, iters)
}

/// The name of the given phase, or `None` if the phase index is out of bounds.
#[must_use]
pub fn phase_name(phase: u32) -> Option<&'static str> {
    PHASES.get(phase as usize).map(|&(name, _)| name)
}

/// Fit diagnostics for the session's current log. Returns the number of sweeps and whether the fit converged.
#[must_use]
pub fn fit_diagnostics(s: &Session) -> (usize, bool) {
    fit_diag(&s.observations())
}

/// The sum of `once`'s results over `iters` runs.
fn repeat(iters: u32, mut once: impl FnMut() -> f64) -> f64 {
    (0..iters).map(|_| once()).sum()
}
