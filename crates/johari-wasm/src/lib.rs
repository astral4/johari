//! WebAssembly C-ABI wrapper over `johari-core`.

#![no_std]
#![feature(sync_unsafe_cell)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::arch::wasm32::unreachable;
use core::cell::SyncUnsafeCell;
use core::panic::PanicInfo;
use johari_core::session::{Bucket, Level, Outcome, RosterSize, Session, SessionError, TieCost, Undone};
use talc::wasm::{WasmDynamicTalc, new_wasm_dynamic_allocator};

#[cfg(feature = "bench")]
use johari_core::session::bench::{fit_diagnostics, phase_name, run};

#[global_allocator]
static TALC: WasmDynamicTalc = new_wasm_dynamic_allocator();

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    unreachable()
}

const ERR_BAD_POINTER: i32 = -1;
const ERR_BAD_ARGUMENT: i32 = -2;
const ERR_BAD_BUCKET: i32 = -3;
const ERR_BAD_OUTCOME: i32 = -4;
const ERR_ITEM_OUT_OF_RANGE: i32 = -5;
const ERR_ALREADY_MARKED: i32 = -6;
const ERR_BUCKET_STAGE_CLOSED: i32 = -7;
const ERR_PAIR_NOT_DISTINCT: i32 = -8;
const ERR_ITEM_NOT_ACTIVE: i32 = -9;
const ERR_PAIR_ALREADY_SHOWN: i32 = -10;

const BUCKETS: [Bucket; 3] = [Bucket::Meh, Bucket::Fine, Bucket::Love];

const OUTCOMES: [Outcome; 3] = [Outcome::First, Outcome::Second, Outcome::Same];

fn decode<T: Copy>(code: i32, values: [T; 3]) -> Result<Option<T>, ()> {
    if code == -1 {
        return Ok(None);
    }
    usize::try_from(code).ok().and_then(|k| values.get(k)).map(|&value| Some(value)).ok_or(())
}

const fn code(result: Result<(), SessionError>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(SessionError::ItemOutOfRange) => ERR_ITEM_OUT_OF_RANGE,
        Err(SessionError::AlreadyMarked) => ERR_ALREADY_MARKED,
        Err(SessionError::BucketStageClosed) => ERR_BUCKET_STAGE_CLOSED,
        Err(SessionError::PairNotDistinct) => ERR_PAIR_NOT_DISTINCT,
        Err(SessionError::ItemNotActive) => ERR_ITEM_NOT_ACTIVE,
        Err(SessionError::PairAlreadyShown) => ERR_PAIR_ALREADY_SHOWN,
    }
}

const fn undone_code(undone: Undone) -> i32 {
    match undone {
        Undone::Nothing => 0,
        Undone::Mark => 1,
        Undone::Pair => 2,
    }
}

fn to_u32(n: usize) -> u32 {
    u32::try_from(n).expect("usize is 32 bits on wasm32")
}

/// The result record.
static RECORD: SyncUnsafeCell<[u64; 3]> = SyncUnsafeCell::new([0; 3]);

/// The block of a list result.
static BLOCK: SyncUnsafeCell<Vec<u32>> = SyncUnsafeCell::new(Vec::new());

/// Write `values` into `RECORD`.
fn put<T, const N: usize>(values: [T; N]) {
    const { assert!(size_of::<[T; N]>() <= size_of::<[u64; 3]>() && align_of::<T>() <= align_of::<u64>(), "the record holds the values") };
    // SAFETY: The record holds aligned `[T; N]` and there is no live reference to it.
    unsafe { RECORD.get().cast::<[T; N]>().write(values) }
}

/// Concatenate `parts` and write the result into `BLOCK`.
fn give(parts: &[&[u32]]) {
    // SAFETY: No other reference to the block is live.
    let block = unsafe { &mut *BLOCK.get() };
    block.clear();
    for part in parts {
        block.extend_from_slice(part);
    }
    put([to_u32(block.as_ptr().addr()), to_u32(block.len())]);
}

/// The address of the result record.
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_result() -> *const u8 {
    RECORD.get().cast()
}

/// A new session over a roster of `roster_size` items using the given tie cost.
/// Returns `None` if `roster_size` exceeds [`RosterSize::MAX`] or `tie_cost` is not in (0, 1].
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_new(roster_size: u32, tie_cost: f64) -> Option<Box<Session>> {
    let (roster_size, tie_cost) = (RosterSize::new(roster_size)?, TieCost::new(tie_cost)?);
    Some(Box::new(Session::new(roster_size, tie_cost)))
}

/// Free a session.
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_free_session(session: Option<Box<Session>>) {
    drop(session);
}

/// Record a bucket mark for a roster item. -1 = don't know; 0 = meh; 1 = fine; 2 = love.
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_mark(session: Option<&mut Session>, item: u32, bucket: i32) -> i32 {
    let Some(s) = session else { return ERR_BAD_POINTER };
    let Ok(bucket) = decode(bucket, BUCKETS) else { return ERR_BAD_BUCKET };
    code(s.mark(item, bucket))
}

/// Record the answer to a shown pair. -1 = skip; 0 = pick `a`; 1 = pick `b`; 2 = same.
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_record(session: Option<&mut Session>, a: u32, b: u32, outcome: i32) -> i32 {
    let Some(s) = session else { return ERR_BAD_POINTER };
    let Ok(outcome) = decode(outcome, OUTCOMES) else { return ERR_BAD_OUTCOME };
    code(s.record([a, b], outcome))
}

/// Retract the most recent data action.
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_undo(session: Option<&mut Session>) -> i32 {
    let Some(s) = session else { return ERR_BAD_POINTER };
    undone_code(s.undo())
}

/// Fit the log if needed and choose the next pair.
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_propose(session: Option<&mut Session>) -> i32 {
    let Some(s) = session else { return ERR_BAD_POINTER };
    let Some(p) = s.propose() else { return 1 };
    put([f64::from(p.shown[0]), f64::from(p.shown[1]), p.value]);
    0
}

/// Fit the log if needed and compute the final list. The list includes the order, tie groups at the session's tie cost,
/// and rank intervals at level `level`.
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_review(session: Option<&mut Session>, level: f64) -> i32 {
    let Some(s) = session else { return ERR_BAD_POINTER };
    let Some(level) = Level::new(level) else { return ERR_BAD_ARGUMENT };
    let review = s.review(level);
    let list = &review.list;
    let head = [to_u32(list.order.len()), to_u32(list.group_sizes.len())];
    give(&[&head, &list.order, &list.group_sizes, &review.rank_lo, &review.rank_hi]);
    0
}

/// Fit diagnostics for the session's current log. Returns the number of sweeps and whether the fit converged.
#[cfg(feature = "bench")]
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_bench_fit_diagnostics(session: Option<&mut Session>) -> i32 {
    let Some(s) = session else { return ERR_BAD_POINTER };
    let (sweeps, converged) = fit_diagnostics(s);
    put([to_u32(sweeps), u32::from(converged)]);
    0
}

/// Time one phase of a step.
#[cfg(feature = "bench")]
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_bench(session: Option<&mut Session>, phase: u32, iters: u32) -> f64 {
    let Some(s) = session else { return f64::NAN };
    run(s, phase, iters)
}

/// Write the name of phase `phase` of [`johari_bench`] into the result record.
#[cfg(feature = "bench")]
#[unsafe(no_mangle)]
#[optimize(size)]
pub extern "C" fn johari_bench_phase(phase: u32) -> i32 {
    let Some(name) = phase_name(phase) else { return 1 };
    put([to_u32(name.as_ptr().addr()), to_u32(name.len())]);
    0
}
