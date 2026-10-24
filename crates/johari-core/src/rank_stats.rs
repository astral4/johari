//! Posterior rank statistics.

use crate::lanes::{from_memory, max_lanes, padded, sqrt_lanes};
use crate::linalg::Matrix;
use crate::posterior::Posterior;
use crate::quadrature::{Rule, gauss_hermite};
use crate::special::{Lane as _, ndtr_in_place, to_f32};
use alloc::vec;
use alloc::vec::Vec;
use core::f64::math::sqrt;
use core::iter::zip;
use core::simd::f32x4;
use core::simd::num::SimdFloat as _;

const ZERO: f32x4 = f32x4::splat(0.);
const ONE: f32x4 = f32x4::splat(1.);

/// Nodes of a Gauss-Hermite rule over an item's θ. Used by [`PairWeights`] and [`PairTerms`].
const THETA_NODES: usize = 5;
const THETA_RULE: Rule<THETA_NODES> = gauss_hermite();

/// E[w(N)] for a 0-based rank N of mean `mean` and variance `var` to second order in N's spread.
/// Let w(k) = 1/(k + 2) be the weight of a pair whose better item has rank k. Then:
///
/// ```text
/// E[w(N)] ≈ w(m) + ½·w''(m)·v = r + v·r³
/// r = w(m) = 1/(m + 2)
/// ```
fn expected_position_weight(mean: f32x4, var: f32x4) -> f32x4 {
    let r = ONE / (mean + f32x4::splat(2.));
    r + var * r * r * r
}

/// The node weight `weight` times E[w(Rᵢ - 1ⱼ) | x], at a node x where item j is above i with probability `above`
/// and Rᵢ has mean `count` and variance `spread`.
fn weighted_epw(above: f32x4, count: f32x4, spread: f32x4, weight: f32x4) -> f32x4 {
    weight * expected_position_weight(count - above, spread - above * (ONE - above))
}

/// Each pair's expected position weight W[i, j] = E[w(min(Rᵢ, Rⱼ))] under the posterior; i.e. the weight w of the true rank of
/// the pair's better item averaged over the posterior. Ranks are 0-based (0 = best), and Rᵢ counts the items above i.
///
/// ```text
/// H[i, j] = E[w(Rᵢ)·1{θᵢ > θⱼ}]
/// W[i, j] = H[i, j] + H[j, i]
/// ```
pub(crate) struct PairWeights(Matrix<f32>);

impl PairWeights {
    /// The weights over `posterior`.
    pub(crate) fn new(posterior: &Posterior) -> Self {
        // Rᵢ is computed as the sum of the other items' indicators 1ⱼ = 1{θⱼ > θᵢ}, which are counted as independent given θᵢ.
        let roots = posterior.twins.roots();
        let mut rows = ItemRows::new(posterior);
        let cols = rows.lines.mu.len();
        let mut halves = Matrix::zero(roots.len(), cols);
        for (c, &root) in roots.iter().enumerate() {
            rows.fill(posterior, root);
            let half = halves.row_mut(c).as_chunks_mut().0;
            for (node, &[count, spread], &weight) in zip!(rows.above.chunks_exact(cols), &rows.moments, &THETA_RULE.weights) {
                let [count, spread, weight] = [count, spread, weight].map(f32x4::splat_f64);
                for (half, &above) in zip(half.iter_mut(), node.as_chunks().0) {
                    let above = f32x4::from_array(above);
                    *half = ((ONE - above) * weighted_epw(above, count, spread, weight) + f32x4::from_array(*half)).to_array();
                }
            }
        }
        Self(halves)
    }

    /// W[i, j] for items i ≠ j of classes `c` and `d`.
    pub(crate) fn get(&self, c: usize, d: usize) -> f32 {
        self.0.get(c, d) + self.0.get(d, c)
    }
}

/// The final list's pair terms; i.e. each pair's W and the parts of it in which either item is noticeably above the other,
///
/// ```text
/// B[i, j] = E[w(min(Rᵢ, Rⱼ))·1{θᵢ - θⱼ > δ}]
/// ```
///
/// so that W[i, j] - B[i, j] - B[j, i] is the weight of the two being within δ. Entry (i, j) is [B[i, j], B[j, i], W[i, j]].
pub(crate) struct PairTerms(Matrix<[f32; 3]>);

impl PairTerms {
    /// The terms over `posterior`.
    pub(crate) fn new(posterior: &Posterior) -> Self {
        // δ is fixed at its posterior median instead of being averaged over its posterior.
        let roots = posterior.twins.roots();
        let class = posterior.twins.class();
        let mut rows = ItemRows::new(posterior);
        let cols = rows.lines.mu.len();
        let nodes = THETA_RULE.nodes.map(|x| f32x4::splat_f64(-x));
        let delta = f32x4::splat_f64(posterior.median_delta());
        let mut halves = Matrix::zero(roots.len(), cols);
        let mut beyond = Matrix::zero(roots.len(), cols);
        let mut events = vec![0.; THETA_NODES * cols];
        let mut offset = vec![0.; cols];
        for (c, &root) in roots.iter().enumerate() {
            rows.fill(posterior, root);
            let lines = &rows.lines;
            for (offset, &intercept, &inv_c) in zip!(offset.as_chunks_mut().0, lines.intercept.as_chunks().0, lines.inv_c.as_chunks().0) {
                *offset = (-(delta * f32x4::from_array(inv_c) + f32x4::from_array(intercept))).to_array();
            }
            probits(&mut events, &nodes, &lines.slope, &offset);
            let half_row = halves.row_mut(c).as_chunks_mut().0;
            let sums = beyond.row_mut(c).as_chunks_mut().0;
            for (node, above_row, &[count, spread], &weight) in
                zip!(events.chunks_exact(cols), rows.above.chunks_exact(cols), &rows.moments, &THETA_RULE.weights)
            {
                let [count, spread, weight] = [count, spread, weight].map(f32x4::splat_f64);
                for (half, sum, &event, &above) in zip!(half_row.iter_mut(), sums.iter_mut(), node.as_chunks().0, above_row.as_chunks().0) {
                    let above = f32x4::from_array(above);
                    let expected = weighted_epw(above, count, spread, weight);
                    *half = ((ONE - above) * expected + f32x4::from_array(*half)).to_array();
                    *sum = (f32x4::from_array(event) * expected + f32x4::from_array(*sum)).to_array();
                }
            }
        }
        let weights = PairWeights(halves);
        Self::from_fn(class.len(), |i, j| {
            let [c, d] = [class[i], class[j]];
            [beyond.get(c, d), beyond.get(d, c), weights.get(c, d)]
        })
    }

    /// The terms over `n` items whose pair i < j has the entry `terms(i, j)` = [B[i, j], B[j, i], W[i, j]].
    pub(crate) fn from_fn(n: usize, mut terms: impl FnMut(usize, usize) -> [f32; 3]) -> Self {
        let mut rows = Matrix::zero(n, n);
        for i in 0..n {
            for j in i + 1..n {
                let [forward, back, weight] = terms(i, j);
                *rows.get_mut(i, j) = [forward, back, weight];
                *rows.get_mut(j, i) = [back, forward, weight];
            }
        }
        Self(rows)
    }

    /// The number of items.
    pub(crate) const fn n_items(&self) -> usize {
        self.0.rows()
    }

    /// Row i: [B[i, j], B[j, i], W[i, j]] at j; 0 at i.
    pub(crate) fn row(&self, i: usize) -> &[[f32; 3]] {
        self.0.row(i)
    }

    /// Margin of rounding error between entries of B.
    #[expect(clippy::cast_precision_loss, reason = "item counts are small")]
    pub(crate) fn rounding(&self, size: f64) -> f64 {
        4. * self.n_items() as f64 * f64::from(f32::EPSILON) * size
    }
}

/// Each node's row of Φ(slope·x + intercept) at each node x of `nodes` into `rows`.
#[inline(never)]
fn probits(rows: &mut [f32], nodes: &[f32x4], slope: &[f32], intercept: &[f32]) {
    for (node, &x) in zip(rows.chunks_exact_mut(slope.len()), nodes) {
        for (arg, &slope, &intercept) in zip!(node.as_chunks_mut().0, slope.as_chunks().0, intercept.as_chunks().0) {
            *arg = (f32x4::from_array(slope) * x + f32x4::from_array(intercept)).to_array();
        }
    }
    ndtr_in_place(rows.as_chunks_mut().0, None);
}

/// One root i's lines at a time over a column per class of twins.
struct Lines {
    /// Each column's class size.
    sizes: Vec<f32>,
    /// The items' class sizes besides the root i that each column counts for.
    others: Vec<f32>,
    /// Each μⱼ.
    mu: Vec<f32>,
    /// Each Σⱼⱼ.
    var: Vec<f32>,
    /// Row i of Σ.
    cross: Vec<f32>,
    /// Each 1/cⱼ.
    inv_c: Vec<f32>,
    /// Each aⱼ's slope and intercept in the node x.
    slope: Vec<f32>,
    intercept: Vec<f32>,
}

impl Lines {
    /// The lines over the posterior's classes of twins.
    #[optimize(size)]
    #[expect(clippy::cast_possible_truncation, reason = "the lanes run in f32")]
    fn new(posterior: &Posterior) -> Self {
        let cols = padded(posterior.twins.roots().len());
        let mut sizes = vec![0.; cols];
        let mut mu = vec![0.; cols];
        let mut var = vec![1.; cols];
        for (&item, &size, to, mu, var) in zip!(posterior.twins.lasts(), posterior.twins.sizes(), &mut sizes, &mut mu, &mut var) {
            (*to, *mu, *var) = (size as f32, posterior.mean_theta()[item] as f32, posterior.var[item] as f32);
        }

        Self {
            others: sizes.clone(),
            sizes,
            mu,
            var,
            cross: vec![0.; cols],
            inv_c: vec![0.; cols],
            slope: vec![0.; cols],
            intercept: vec![0.; cols],
        }
    }

    /// Root i's Σ row, class sizes besides i, and each aⱼ's 1/cⱼ, slope and intercept.
    #[inline(never)]
    #[expect(clippy::cast_possible_truncation, reason = "the lanes run in f32")]
    fn fill(&mut self, posterior: &Posterior, i: usize) {
        const EPS: f32x4 = f32x4::splat(f32::MIN_POSITIVE);

        let row = posterior.covariance.row(i);
        for (to, &item) in zip(&mut self.cross, posterior.twins.lasts()) {
            *to = row[item] as f32;
        }
        self.others.copy_from_slice(&self.sizes);
        self.others[posterior.twins.class()[i]] -= 1.;

        let mu_i = f32x4::splat(posterior.mean_theta()[i] as f32);
        let var_i = f32x4::splat(posterior.var[i] as f32);
        let inv_var = ONE / var_i;
        let inv_sd = ONE / sqrt_lanes(var_i);
        let line = zip(self.slope.as_chunks_mut().0, self.intercept.as_chunks_mut().0);
        let lanes = zip!(self.cross.as_chunks().0, self.var.as_chunks().0, self.mu.as_chunks().0, self.inv_c.as_chunks_mut().0, line);
        for (&cross, &var, &mu, inv_c, (slope, intercept)) in lanes {
            let [cross, var, mu] = [cross, var, mu].map(f32x4::from_array);
            let scale = ONE / sqrt_lanes(max_lanes(var - cross * cross * inv_var, EPS));
            *inv_c = scale.to_array();
            *slope = ((cross - var_i) * inv_sd * scale).to_array();
            *intercept = ((mu - mu_i) * scale).to_array();
        }
    }
}

/// One root's rows at a time over a column per class of twins, at each node of [`THETA_RULE`].
struct ItemRows {
    /// The root's lines.
    lines: Lines,
    /// The rule's nodes.
    nodes: Vec<f32x4>,
    /// Each node's row of pⱼ = Φ(aⱼ).
    above: Vec<f32>,
    /// Each node's [m, v].
    moments: Vec<[f64; 2]>,
}

impl ItemRows {
    /// The rows over the posterior's classes of twins.
    #[optimize(size)]
    fn new(posterior: &Posterior) -> Self {
        let lines = Lines::new(posterior);
        let rows = THETA_NODES * lines.mu.len();
        Self {
            lines,
            nodes: THETA_RULE.nodes.iter().copied().map(f32x4::splat_f64).collect(),
            above: vec![0.; rows],
            moments: vec![[0.; 2]; THETA_NODES],
        }
    }

    /// Root i's rows.
    #[inline(never)]
    fn fill(&mut self, posterior: &Posterior, i: usize) {
        self.lines.fill(posterior, i);
        let lines = &self.lines;
        probits(&mut self.above, &self.nodes, &lines.slope, &lines.intercept);

        let cols = lines.mu.len();
        for (moment, node) in zip(&mut self.moments, self.above.chunks_exact(cols)) {
            let mut count = ZERO;
            let mut spread = ZERO;
            for (&above, &others) in zip(node.as_chunks().0, lines.others.as_chunks().0) {
                let [above, others] = [above, others].map(f32x4::from_array);
                let all = others * above;
                count += all;
                spread = all * (ONE - above) + spread;
            }
            *moment = [f64::from(count.reduce_sum()), f64::from(spread.reduce_sum())];
        }
    }
}

/// A review's interval level.
#[derive(Clone, Copy)]
pub(crate) struct Level(f64);

impl Level {
    #[must_use]
    pub(crate) const fn new(x: f64) -> Option<Self> {
        if x >= 0. && x <= 1. { Some(Self(x)) } else { None }
    }

    pub(crate) const fn get(self) -> f64 {
        self.0
    }
}

/// Nodes of a Gauss-Hermite rule over an item's θ. Used by [`rank_intervals`].
const REVIEW_NODES: usize = 24;
const REVIEW_RULE: Rule<REVIEW_NODES> = gauss_hermite();
static REVIEW_NODES_F32: &[f32; REVIEW_NODES] = &to_f32(REVIEW_RULE.nodes);

const _: () = assert!(REVIEW_NODES.is_multiple_of(f32x4::LEN), "the nodes fill whole lanes");

/// The lanes over [`REVIEW_RULE`]'s nodes.
const NODE_LANES: usize = REVIEW_NODES / f32x4::LEN;

/// A value at each node of [`REVIEW_RULE`], in lanes.
type AtNodes = [[f32; f32x4::LEN]; NODE_LANES];

/// The floor on each node's V(x).
const MIN_RANK_VAR: f64 = 1e-6;

/// Each item's central credible interval on rank at level `level` as `[lo, hi]` from its distribution of rank under the posterior.
#[inline(never)]
#[optimize(size)]
#[expect(clippy::cast_possible_truncation, reason = "roster size fits in u32")]
pub(crate) fn rank_intervals(posterior: &Posterior, level: Level) -> Vec<[u32; 2]> {
    let n = posterior.n_items();

    let tail = 0.5 * (1. - level.get());
    if tail <= 0. {
        return vec![[0, n.saturating_sub(1) as u32]; n];
    }

    let moments = rank_moments(posterior);
    #[expect(clippy::chunks_exact_to_as_chunks, reason = "slices keep loops rolled for size")]
    let intervals: Vec<_> = moments
        .chunks_exact(REVIEW_NODES)
        .map(|moments| {
            let mut mixture = Mixture::default();
            for (center, scale, &[count, spread]) in zip!(mixture.centers.as_flattened_mut(), mixture.scales.as_flattened_mut(), moments) {
                *center = (count - 0.5) as f32;
                *scale = (1. / sqrt(spread.max(MIN_RANK_VAR))) as f32;
            }
            [mixture.quantile(tail, n), mixture.quantile(1. - tail, n)]
        })
        .collect();
    posterior.twins.class().iter().map(|&c| intervals[c]).collect()
}

/// The weights of [`REVIEW_RULE`] in f32.
static MIXTURE_WEIGHTS: &[f32; REVIEW_NODES] = &to_f32(REVIEW_RULE.weights);

/// An item's distribution of rank. P(Rᵢ ≤ r) = Σₜ wₜ·Φ((r - cₜ)·sₜ) over [`REVIEW_RULE`],
/// with wₜ in [`MIXTURE_WEIGHTS`], cₜ in `centers`, and sₜ in `scales`.
#[derive(Default)]
struct Mixture {
    centers: AtNodes,
    scales: AtNodes,
}

impl Mixture {
    /// P(Rᵢ ≤ r).
    #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
    fn cdf(&self, r: usize) -> f32 {
        let mut args = self.centers;
        for (arg, &scale) in zip(&mut args, &self.scales) {
            *arg = ((f32x4::splat(r as f32) - f32x4::from_array(*arg)) * f32x4::from_array(scale)).to_array();
        }
        ndtr_in_place(&mut args, None);
        let weights = from_memory(&MIXTURE_WEIGHTS).as_chunks().0;
        zip(&args, weights).fold(ZERO, |sum, (&p, &w)| f32x4::from_array(p) * f32x4::from_array(w) + sum).reduce_sum()
    }

    /// The smallest rank r < `n` such that P(Rᵢ ≤ r) ≥ `p`.
    #[inline(never)]
    #[optimize(size)]
    #[expect(clippy::cast_possible_truncation, reason = "ranks are small")]
    fn quantile(&self, p: f64, n: usize) -> u32 {
        let mut lo = 0;
        let mut hi = n.saturating_sub(1);
        while lo < hi {
            let mid = lo.midpoint(hi);
            if f64::from(self.cdf(mid)) >= p {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo as u32
    }
}

const BATCH: usize = 16;

/// Each class' [m(x), V(x)] at each node of [`REVIEW_RULE`] for the review's rank intervals.
#[inline(never)]
#[expect(clippy::cast_possible_truncation, reason = "the lanes run in f32")]
fn rank_moments(posterior: &Posterior) -> Vec<[f64; 2]> {
    // Given θᵢ at a node, θⱼ and θₗ have the conditional covariance Σ₍ⱼₗ|ᵢ₎ = Σⱼₗ - ΣⱼᵢΣᵢₗ/Σᵢᵢ and correlation ρⱼₗ = Σ₍ⱼₗ|ᵢ₎/(cⱼcₗ),
    // so both are above i with probability Φ₂(aⱼ, aₗ; ρⱼₗ) = Φ(aⱼ)Φ(aₗ) + ρⱼₗ·φ(aⱼ)φ(aₗ) to first order in ρ.
    let roots = posterior.twins.roots();
    let class = posterior.twins.class();
    let mut lines = Lines::new(posterior);
    let nodes = from_memory(&REVIEW_NODES_F32).as_chunks().0;
    let mut above = vec![AtNodes::default(); roots.len()];
    let mut densities = vec![AtNodes::default(); roots.len()];
    let mut moments = vec![[0.; 2]; REVIEW_NODES * roots.len()];
    for (batch, moments) in zip(roots.chunks(BATCH), moments.chunks_mut(REVIEW_NODES * BATCH)) {
        let mut columns = Matrix::zero(posterior.dim(), REVIEW_NODES * batch.len());
        for (b, &i, moments) in zip!(0.., batch, moments.as_chunks_mut::<REVIEW_NODES>().0) {
            lines.fill(posterior, i);
            for (args, &slope, &intercept) in zip!(&mut above, &lines.slope, &lines.intercept) {
                let [slope, intercept] = [slope, intercept].map(f32x4::splat);
                for (arg, &x) in zip(args, nodes) {
                    *arg = (slope * f32x4::from_array(x) + intercept).to_array();
                }
            }
            ndtr_in_place(above.as_flattened_mut(), Some(densities.as_flattened_mut()));

            let [mut count, mut spread, mut own, mut squares] = [[ZERO; NODE_LANES]; _];
            for (&lanes, &others) in zip(&above, &lines.others) {
                let others = f32x4::splat(others);
                for (count, spread, above) in zip!(&mut count, &mut spread, lanes.map(f32x4::from_array)) {
                    let all = others * above;
                    *count += all;
                    *spread = all * (ONE - above) + *spread;
                }
            }
            for (lanes, &cross, &inv_c, &others) in zip!(&mut densities, &lines.cross, &lines.inv_c, &lines.others) {
                let [cross, inv_c, others] = [cross, inv_c, others].map(f32x4::splat);
                for (own, squares, lane) in zip!(&mut own, &mut squares, lanes) {
                    let phi = f32x4::from_array(*lane);
                    let u = phi * inv_c;
                    *own += others * cross * u;
                    *squares += others * phi * phi;
                    *lane = u.to_array();
                }
            }
            let var_i = f32x4::splat(posterior.var[i] as f32);
            let at_i = own.map(|own| (-own / var_i).to_array());
            let [count, spread, squares] = [count, spread, squares].map(|sums| sums.map(f32x4::to_array));
            for (moment, &count, &spread, &squares) in zip!(moments, count.as_flattened(), spread.as_flattened(), squares.as_flattened()) {
                *moment = [f64::from(count), f64::from(spread) - f64::from(squares)];
            }
            for (j, &d) in class.iter().enumerate() {
                let row = &mut columns.row_mut(j)[REVIEW_NODES * b..][..REVIEW_NODES];
                row.copy_from_slice(if j == i { at_i.as_flattened() } else { densities[d].as_flattened() });
            }
        }
        for (moment, form) in zip(moments, posterior.chol_precision.inverse_quadratic_forms(columns)) {
            moment[1] += f64::from(form);
        }
    }
    moments
}

#[cfg(test)]
mod tests {
    use super::{Level, rank_intervals};
    use crate::observations::{Bucket, Observations};
    use crate::posterior::fit;
    use crate::testing::{Lcg, random_log};
    use alloc::vec;
    use alloc::vec::Vec;
    use core::iter::{chain, zip};

    #[test]
    fn intervals_nest_and_span_every_rank_at_level_one() {
        let mut lcg = Lcg(13);
        let random = [(2, 1), (7, 12), (20, 40), (70, 100)].map(|(n, answers)| random_log(&mut lcg, n, answers));
        let untouched =
            (2..=64).step_by(2).flat_map(|n| Bucket::ALL.map(|bucket| Observations { buckets: vec![bucket; n], pairs: Vec::new() }));
        for obs in chain(random, untouched) {
            let (n, posterior) = (obs.buckets.len(), fit(&obs));
            let levels = [0., 0.5, 0.8, 0.95, 0.999, 1.].map(|level| rank_intervals(&posterior, Level::new(level).expect("in [0, 1]")));
            for (narrow, wide) in zip(&levels, &levels[1..]) {
                assert!(zip(narrow, wide).all(|(a, b)| b[0] <= a[0] && a[0] <= a[1] && a[1] <= b[1]), "n {n}: nested intervals");
            }
            #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
            let every = [0, n as u32 - 1];
            assert!(levels[5].iter().all(|&interval| interval == every), "n {n}: level 1 spans every rank");
        }
    }
}
