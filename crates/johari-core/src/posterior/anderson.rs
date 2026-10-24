//! Anderson acceleration.

use crate::linalg::{dot, sub_scaled};
use alloc::vec::Vec;
use core::f64::math::sqrt;
use core::iter::zip;

/// The number of differences between past iterations used for fitting extrapolations.
const DEPTH: usize = 6;
const PROGRESS: f64 = 0.5;
const DAMPING: f64 = 0.5;
const RIDGE: f64 = 1e-10;
const REFRESH: f64 = f64::EPSILON / RIDGE;

/// Anderson acceleration of the iteration x ← g(x) toward a fixed point of g. With the residual f = g(x) - x and its weighting u = W·f,
/// the next x is g(x) - Σᵢ γᵢ·Δgᵢ over the last `DEPTH` differences. W is chosen per point by the caller such that a residual's size
/// tracks how far its point is from the fixed point. γ is the ridge least squares fitting u by the Δu.
/// Note that the steps move by f; only γ and the progress rule read u.
pub(super) struct Anderson {
    state: State,
    /// The history's last point, [g(x), u], where the next differences start.
    last: Option<[Vec<f64>; 2]>,
    /// The recent differences [Δg, Δu], oldest first.
    diffs: Vec<[Vec<f64>; 2]>,
    /// Δuᵢ·Δuⱼ over `diffs` in the lower triangle.
    gram: [[f64; DEPTH]; DEPTH],
    /// Δuᵢ·u over `diffs` at the last point.
    rhs: [f64; DEPTH],
    /// The largest norm since `rhs` was last computed whole.
    anchor: f64,
}

#[derive(Clone, Copy)]
enum State {
    Extrapolating {
        /// The last residual that counted as progress.
        best: f64,
        /// The number of steps since `best` was achieved.
        since: usize,
    },
    BackingOff {
        /// The largest residual since backing off began.
        peak: f64,
        /// The number of damped steps taken.
        taken: usize,
    },
}

impl State {
    /// The next state from a point whose residual's norm is `norm`.
    fn next(self, norm: f64) -> Self {
        match self {
            Self::Extrapolating { best, .. } if norm < PROGRESS * best => Self::Extrapolating { best: norm, since: 0 },
            Self::Extrapolating { best, since } if since + 1 < DEPTH => Self::Extrapolating { best, since: since + 1 },
            Self::Extrapolating { .. } => Self::BackingOff { peak: norm, taken: 1 },
            Self::BackingOff { peak, taken } => {
                let peak = peak.max(norm);
                if norm < PROGRESS * peak && taken >= DEPTH {
                    Self::Extrapolating { best: norm, since: 0 }
                } else {
                    Self::BackingOff { peak, taken: taken + 1 }
                }
            }
        }
    }
}

impl Anderson {
    pub(super) fn new() -> Self {
        let state = State::Extrapolating { best: f64::INFINITY, since: 0 };
        Self { state, last: None, diffs: Vec::new(), gram: [[0.; DEPTH]; DEPTH], rhs: [0.; DEPTH], anchor: 0. }
    }

    /// Step from `x`, whose residual is `f` = g(x) - x, weighted as `u` = W·f. Returns the extrapolation g(x) - Σᵢ γᵢ·Δgᵢ;
    /// i.e. Σⱼ αⱼ·g(xⱼ) over the stored iterates, with the weights α summing to 1 and minimizing ‖Σⱼ αⱼ·uⱼ‖.
    /// Returns the damped step x + `DAMPING`·f instead on the first step, on every step while backing off, and if γ is not finite.
    pub(super) fn step(&mut self, mut x: Vec<f64>, mut f: Vec<f64>, u: Vec<f64>) -> Vec<f64> {
        debug_assert_eq!(f.len(), u.len(), "u weights f");
        let norm = sqrt(dot(&u, &u));
        self.state = self.state.next(norm);
        // x becomes the damped step, which an extrapolation replaces, and f becomes g(x).
        for (x, f) in zip(&mut x, &mut f) {
            let g = *x + *f;
            *x += DAMPING * *f;
            *f = g;
        }
        let g = f;
        if let Some(last) = self.last.take() {
            self.record(last, &g, &u, norm);
        }
        if matches!(self.state, State::Extrapolating { .. })
            && let Some(gamma) = self.gamma()
        {
            x.copy_from_slice(&g);
            for (&gamma, [dg, _]) in zip(&gamma, &self.diffs) {
                sub_scaled(&mut x, gamma, dg);
            }
        }
        self.last = Some([g, u]);
        x
    }

    /// Add the differences of `g` and `u` from the last point [g, u] to the history, dropping the oldest beyond `DEPTH`.
    /// Also brings `rhs` to `u` of norm `norm`.
    fn record(&mut self, mut diff: [Vec<f64>; 2], g: &[f64], u: &[f64], norm: f64) {
        for (d, new) in zip(&mut diff, [g, u]) {
            for (d, x) in zip(d, new) {
                *d = x - *d;
            }
        }
        if self.diffs.len() == DEPTH {
            self.diffs.remove(0);
            self.gram.copy_within(1.., 0);
            for row in &mut self.gram {
                row.copy_within(1.., 0);
            }
            self.rhs.copy_within(1.., 0);
        }
        let (m, [_, du]) = (self.diffs.len(), &diff);
        // u moved by Δu, so each earlier Δuᵢ·u moved by Δuᵢ·Δu.
        for (entry, rhs, [_, earlier]) in zip!(&mut self.gram[m], &mut self.rhs, &self.diffs) {
            *entry = dot(du, earlier);
            *rhs += *entry;
        }
        self.gram[m][m] = dot(du, du);
        self.diffs.push(diff);
        self.anchor = self.anchor.max(norm);
        let start = if norm < REFRESH * self.anchor {
            self.anchor = norm;
            0
        } else {
            m
        };
        for (rhs, [_, du]) in zip(&mut self.rhs[start..], &self.diffs[start..]) {
            *rhs = dot(du, u);
        }
    }

    /// Returns γ for the last point, or `None` when there are no stored differences or γ is not finite.
    fn gamma(&self) -> Option<[f64; DEPTH]> {
        if self.diffs.is_empty() {
            return None;
        }
        // The normal equations ΔUᵀΔU·γ = ΔUᵀu.
        let mut gamma = self.rhs;
        solve_gram(&self.gram, &mut gamma[..self.diffs.len()]);
        gamma.iter().all(|g| g.is_finite()).then_some(gamma)
    }
}

/// Solve G·γ = `b` in place for the Gram matrix G of the first `b.len()` differences.
#[inline(never)]
#[optimize(size)]
fn solve_gram(gram: &[[f64; DEPTH]; DEPTH], b: &mut [f64]) {
    let m = b.len();
    let mut l = [[0.; DEPTH]; DEPTH];
    for i in 0..m {
        for j in 0..i {
            l[i][j] = (gram[i][j] - dot(&l[i][..j], &l[j][..j])) / l[j][j];
        }
        let v = &l[i][..i];
        l[i][i] = sqrt(gram[i][i] * (1. + RIDGE) - dot(v, v));
    }
    // L·y = b, then Lᵀ·γ = y, each in place.
    for (i, row) in l[..m].iter().enumerate() {
        b[i] = (b[i] - dot(&row[..i], &b[..i])) / row[i];
    }
    for (i, row) in l[..m].iter().enumerate().rev() {
        let gamma = b[i] / row[i];
        b[i] = gamma;
        sub_scaled(&mut b[..i], gamma, &row[..i]);
    }
}

#[cfg(test)]
mod tests {
    use super::{Anderson, DEPTH, PROGRESS, State};
    use crate::linalg::dot;
    use crate::testing::Lcg;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::iter::{chain, zip};
    use libm::{exp, exp2, sqrt};

    #[test]
    fn extrapolations_solve_affine_maps() {
        let mut rng = Lcg(5);
        for (trial, n) in (0..8000).map(|trial| (trial, 2 + trial % 4)) {
            let linear: Vec<_> = (0..n * n).map(|_| rng.sym(1.)).collect();
            let norm = linear.chunks(n).map(|row| row.iter().map(|x| x.abs()).sum()).fold(0., f64::max);
            let linear: Vec<_> = linear.iter().map(|x| 0.95 * x / norm).collect();
            let offset: Vec<_> = (0..n).map(|_| rng.sym(1.)).collect();
            let weights: Vec<_> = (0..n).map(|_| exp(rng.sym(3.))).collect();

            let g = |x: &[f64]| zip(linear.chunks(n), &offset).map(|(row, b)| dot(row, x) + b).collect::<Vec<_>>();
            let residual = |x: &[f64]| zip(g(x), x).map(|(g, x)| (g - x).abs()).fold(0., f64::max);

            let steps = 2 * n + 6;
            for weighted in [false, true] {
                let mut anderson = Anderson::new();
                let mut x = vec![0.; n];
                for step in 0.. {
                    let size = residual(&x);
                    if size <= 1e-9 {
                        break;
                    }
                    assert!(step < steps, "trial {trial}, {n} dimensions, weighted {weighted}: residual {size} after {steps} steps");
                    let f: Vec<_> = zip(g(&x), &x).map(|(g, x)| g - x).collect();
                    let u = if weighted { zip(&f, &weights).map(|(f, w)| f * w).collect() } else { f.clone() };
                    x = anderson.step(x, f, u);
                }
            }
        }
    }

    #[test]
    fn stalled_runs_back_off() {
        const { assert!(DEPTH == 6 && PROGRESS == 0.5) };

        let mut rng = Lcg(11);
        let early = ([2., 4., 8., 3.5, 3., 2.5, 2.4, 2.], DEPTH);
        let late = ([2., 4., 8., 16., 12., 9., 8.5, 7., 6.], DEPTH + 1);
        for (back_off, damped_steps) in [(&early.0[..], early.1), (&late.0[..], late.1)] {
            let scales: Vec<_> = chain([100.], [1.; DEPTH]).chain(back_off.iter().copied()).collect();
            for weighted in [true, false] {
                let mut anderson = Anderson::new();
                let mut x = vec![0.; 16];
                let mut extrapolated = Vec::new();

                let unit = |rng: &mut Lcg| {
                    let f: Vec<_> = (0..8).map(|_| rng.sym(1.)).collect();
                    let norm = sqrt(dot(&f, &f));
                    f.into_iter().map(move |f| f / norm)
                };

                for (k, &scale) in scales.iter().enumerate() {
                    let quartering = 1e15 * exp2(-2. * f64::from(u32::try_from(k).expect("a few steps")));
                    let f: Vec<_> = unit(&mut rng).map(|f| scale * f).chain(unit(&mut rng).map(|f| quartering * f)).collect();
                    let mut u = f.clone();
                    if weighted {
                        u[8..].fill(0.);
                    }
                    x = anderson.step(x, f, u);
                    extrapolated.push(matches!(anderson.state, State::Extrapolating { .. }));
                }

                if weighted {
                    let b_start = 1 + DEPTH;
                    let b_end = 1 + DEPTH + damped_steps;
                    assert!(extrapolated[1..b_start].iter().all(|&e| e), "extrapolations until the stall: {extrapolated:?}");
                    assert!(extrapolated[b_start..b_end].iter().all(|&e| !e), "then damped steps: {extrapolated:?}");
                    assert!(extrapolated[b_end..].iter().all(|&e| e), "then extrapolations again: {extrapolated:?}");
                } else {
                    assert!(extrapolated[1..].iter().all(|&e| e), "unweighted, no back-off: {extrapolated:?}");
                }
            }
        }
    }
}
