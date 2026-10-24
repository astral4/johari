//! Adaptive pair selection and scoring.
//!
//! ## The value of an answer
//!
//! Each pair h = (i, j) has a gap gₕ = θᵢ - θⱼ ~ N(mₕ, σₕ²) under q. The list claims i is above j, j is above i, or they are tied.
//! The claim with the lowest expected cost depends on mₕ and changes at a boundary b:
//!
//! - Listing i above j costs (1 - τ) - P(gₕ > δ) more (per unit of the pair's position weight) than tying them, so the two claims switch
//!   where P(gₕ > δ) = 1 - τ, at mₕ = δ + z·σₕ with z = Φ⁻¹(1 - τ) (see [`tie_boundary`]). Likewise for j above i at mₕ = -(δ + z·σₕ).
//! - If δ + z·σₕ ≤ 0, then tying is never cheapest and the two orders switch at mₕ = 0.
//!
//! As a function of mₕ, the expected cost of the cheapest claim has a kink at each boundary b, where its slope jumps by some Δ
//! that depends on b. Evidence that removes R of gₕ's variance moves mₕ by a random amount with mean 0 and variance R
//! (q's mean is a martingale). If that move is Gaussian and we treat the claims' costs as linear near b,
//! then it lowers the expected cost by Δ·sqrt(R)·ψ(d/sqrt(R)), where d = mₕ - b and ψ(x) = φ(x) - |x|·Φ(-|x|).
//!
//! An answer's move takes one of three values, so it isn't Gaussian and the formula can't price an answer directly.
//! However, the formula does hold for completely settling the pair. With R = σₕ², mₕ moves to gₕ by a Gaussian,
//! so settling is worth Δ·σₕ·ψ(d/σₕ). Each answer is credited with a share of that in proportion to the variance it removes,
//! Δ·ψ(d/σₕ)/σₕ per unit, so the shares of the answers that settle the pair add up to its whole value in any order (Aumann-Shapley).
//!
//! Summing over the pair's boundaries gives Dₕ (see below), the value of a unit of variance removed from gₕ. An answer to the pair (x, y),
//! whose gap is g = θₓ - θᵧ, removes κ·cov(gₕ, g)² of gₕ's variance in expectation, where κ is the answer's Fisher information
//! about the mean of g (see [`answer_information`]). Summed over every pair h, the answer's value is a quadratic form:
//!
//! ```text
//! value(x, y) = κ · Σₕ Dₕ·cov(gₕ, g)²  =  κ · (eₓ - eᵧ)ᵀ·Σ·M·Σ·(eₓ - eᵧ)
//! M           = Σₕ Dₕ·(eᵢ - eⱼ)(eᵢ - eⱼ)ᵀ
//! Dₕ          = wₕ · Σ_b Δ_b·ψ((mₕ - b)/σₕ)/σₕ
//!
//! b = ±(δ + z·σₕ),  Δ_b = φ(z)/σₕ           if δ + z·σₕ > 0
//! b = 0,            Δ_b = 2φ(δ/σₕ)/σₕ       otherwise
//! ```
//!
//! where Σ is the items' posterior covariance and wₕ is the pair's expected position weight.
//!
//! ## Approximations
//!
//! - δ is fixed at its posterior median exp(E[log δ]).
//! - Each claim's expected cost is wₕ times the gap's probabilities, as if the weight and the gap were independent.
//! - Each claim's expected cost is linear in mₕ near each boundary.

use crate::gaussian::Normal;
use crate::lanes::max_lanes;
use crate::linalg::Matrix;
use crate::model::{ANSWER_NOISE_VAR, Careless, answer_likelihood, sharpness};
use crate::observations::{AnswerKind, Pair, PairSet};
use crate::posterior::Posterior;
use crate::rank_stats::PairWeights;
use crate::ranking::TieCost;
use crate::rng::Rng;
use crate::special::{exp, ndtr_lanes, ndtri};
use alloc::vec::Vec;
use core::f64::consts::FRAC_1_SQRT_2PI;
use core::iter::zip;
use core::simd::f64x2;
use core::simd::num::SimdFloat as _;

/// Values within this fraction of the best are considered tied.
const TIE_TOLERANCE: f64 = 1e-5;

/// State for computing how much an answer is expected to reduce the expected loss of the final list.
pub(crate) struct Scorer {
    /// Entry (c, d) is the quadratic form (eₓ - eᵧ)ᵀ·Σ·M·Σ·(eₓ - eᵧ) for x in class c and y in class d of the posterior's twins.
    /// In other words, it is the weighted variance removed by an answer to (x, y) per unit of κ.
    removed: Matrix,
}

impl Scorer {
    /// The scorer for the posterior at the given tie cost.
    pub(crate) fn new(posterior: &Posterior, tie_cost: TieCost) -> Self {
        let weights = PairWeights::new(posterior);
        let delta = posterior.median_delta();
        let z = tie_boundary(tie_cost);
        // φ(z) = Δ_b·σₕ at either tie boundary.
        let tie_jump = FRAC_1_SQRT_2PI * exp(-0.5 * z * z);

        let roots = posterior.twins.roots();
        let class = posterior.twins.class();
        let size = posterior.twins.sizes();
        let last = posterior.twins.lasts();
        let n_classes = roots.len();

        // M is the Laplacian matrix of the complete graph on the items with edge weights Dₕ.
        let mut between = Matrix::zero(n_classes, n_classes);
        for (c, &root) in roots.iter().enumerate() {
            for (d, &other) in zip(c.., &roots[c..]) {
                let partner = if d == c { last[c] } else { other };
                if partner == root {
                    continue;
                }

                let gap = posterior.gap(root, partner);
                let (mean, sd) = (gap.mean, gap.sd());
                let boundary = delta + z * sd;
                let inv_sd = 1. / sd;

                let x = f64x2::from_array(if boundary > 0. { [mean - boundary, mean + boundary] } else { [delta, mean] });
                let x = -(x * f64x2::splat(inv_sd)).abs();
                let [tail, density] = ndtr_lanes(x);
                let [first, second] = max_lanes(x * tail + density, f64x2::splat(0.)).to_array();
                let kinks = if boundary > 0. { tie_jump * (first + second) } else { 2. * density[0] * second };
                let weight = weights.get(c, d);
                #[expect(clippy::cast_possible_truncation, reason = "the solves run in f32")]
                let edge = (f64::from(weight) * kinks * inv_sd * inv_sd) as f32;
                *between.get_mut(c, d) = -edge;
                *between.get_mut(d, c) = -edge;
            }
        }

        // M̃'s columns at the roots.
        let mut product = Matrix::zero(posterior.dim(), n_classes);
        for (j, &d) in class.iter().enumerate() {
            product.row_mut(j).copy_from_slice(between.row(d));
        }
        for (c, &root) in roots.iter().enumerate() {
            let row = product.row_mut(root);
            #[expect(clippy::cast_possible_truncation, reason = "the solves run in f32")]
            let degree = zip(size, row.iter()).map(|(&s, &x)| s * f64::from(x)).sum::<f64>() as f32;
            row[c] -= degree;
        }
        posterior.chol_precision.solve_matrix(&mut product);

        // The item rows now hold C = Σ·M at the roots' columns. We turn them into columns of Cᵀ so the column of a root r is row r of C.
        for (c, &root) in roots.iter().enumerate() {
            for (d, &other) in zip(c + 1.., &roots[c + 1..]) {
                let (upper, lower) = (product.get(root, d), product.get(other, c));
                *product.get_mut(root, d) = lower;
                *product.get_mut(other, c) = upper;
            }
        }
        for (j, &d, root) in zip!(0.., class, posterior.twins.item_roots()) {
            if j != root {
                let (row, root_row) = product.two_rows(j, root);
                let own = row[d];
                row.copy_from_slice(root_row);
                row[d] = own;
            }
        }
        for row in class.len()..posterior.dim() {
            product.row_mut(row).fill(0.);
        }
        posterior.chol_precision.solve_matrix(&mut product);

        // The item rows now hold S = Σ·M·Σ at the roots' columns.
        // For classes c and d, (eₓ - eᵧ)ᵀ·S·(eₓ - eᵧ) takes the two roots as x and y.
        let own: Vec<_> = roots.iter().enumerate().map(|(c, &root)| f64::from(product.get(root, c))).collect();
        let mut removed = Matrix::zero(n_classes, n_classes);
        for (c, &root) in roots.iter().enumerate() {
            for (value, &own_d, &cross) in zip!(removed.row_mut(c), &own, &product.row(root)[..n_classes]) {
                *value = (own[c] + own_d - 2. * f64::from(cross)).max(0.);
            }
            *removed.get_mut(c, c) = (2. * (own[c] - f64::from(product.get(last[c], c)))).max(0.);
        }
        Self { removed }
    }

    /// The unshown pair with the most valuable answer, plus its value. `posterior` must be the same as the one used in [`Scorer::new`].
    ///
    /// # Panics
    ///
    /// Panics if every pair has been shown.
    #[must_use]
    pub(crate) fn propose(&self, posterior: &Posterior, shown: &PairSet, rng: &mut Rng) -> (Pair, f64) {
        let ties = self.best_pairs(posterior, shown);
        rng.choice(&ties).expect("an unshown pair with a value")
    }

    /// The unshown pairs whose values are within [`TIE_TOLERANCE`] of the best, plus their values.
    /// `posterior` must be the same as the one used in [`Scorer::new`].
    fn best_pairs(&self, posterior: &Posterior, shown: &PairSet) -> Vec<(Pair, f64)> {
        debug_assert_eq!(self.removed.rows(), posterior.twins.roots().len(), "the values are over this posterior's classes");

        let n = posterior.n_items();
        let delta = posterior.median_delta();
        let class = posterior.twins.class();
        // The best value so far.
        let mut best = f64::NEG_INFINITY;
        // The minimum value tied with the best.
        let mut floor = f64::NEG_INFINITY;
        // The pairs at or above the floor.
        let mut ties = Vec::new();
        let bound = information_bound(posterior.careless);
        for x in 0..n {
            let removed_x = self.removed.row(class[x]);
            #[expect(clippy::cast_possible_truncation, reason = "item counts are far below u32::MAX")]
            for (y, gap, &class_y) in zip!(x + 1.., posterior.gaps_after(x), &class[x + 1..]) {
                let pair = Pair::new(x as u32, y as u32);
                if shown.contains(pair) {
                    continue;
                }
                // (eₓ - eᵧ)ᵀ·Σ·M·Σ·(eₓ - eᵧ). The value is κ times this.
                let removed = removed_x[class_y];
                // Most pairs can be ruled out without evaluating the probits. Since κ ≤ B·s² for B = `bound`,
                // a pair with B·s²·removed < floor can't tie the best. Both sides are multiplied by 1/s² = 1/β² + var.
                if bound * removed < floor * (ANSWER_NOISE_VAR + gap.var) {
                    continue;
                }
                let value = answer_information(gap, delta, posterior.careless) * removed;
                // A new best raises the floor, so drop the ties now below it.
                if value > best {
                    (best, floor) = (value, value - TIE_TOLERANCE * value.abs());
                    ties.retain(|&(_, other)| other >= floor);
                }
                if value >= floor {
                    ties.push((pair, value));
                }
            }
        }
        ties
    }
}

/// z = Φ⁻¹(1 - τ) = -Φ⁻¹(τ) for the given tie cost τ.
fn tie_boundary(tie_cost: TieCost) -> f64 {
    -ndtri(tie_cost.get())
}

/// F, the maximum share of information in the perceived gap kept by a careful answer. In other words, κ/s² ≤ F for every gap and δ.
/// κ is the answer's Fisher information about the mean m of its gap, and s is its sharpness at the gap's variance.
/// [`information_bound`] extends this to careless answers.
///
/// The perceived gap is m + W/s for W ~ N(0, 1). If this could be seen directly, it would give s² of information about m.
/// However, the answer only indicates which of three ranges W falls in, so the question is how much of s²
/// a three-way split can keep at best. With x = s·m and u = s·δ:
///
/// ```text
/// answer      | range of W | P                   | ∂P/∂x
/// ------------|------------|---------------------|--------------------
/// pick first  | W > u - x  | Φ(x - u)            | φ(x - u)
/// pick second | W < -u - x | Φ(-x - u)           | -φ(x + u)
/// same        | in between | Φ(x + u) - Φ(x - u) | φ(x + u) - φ(x - u)
/// ```
///
/// Since w·φ(w) = -φ′(w), each ∂P/∂x is P·E[W | answer], so:
///
/// ```text
/// κ/s² = Σ (∂P/∂x)²/P
///      = Σ P·E[W | answer]²
///      = Var(E[W | answer])
///      = 1 - E[Var(W | answer)]
/// ```
///
/// That is, κ/s² is the share of W's variance explained by the answer's range. E[Var(W | answer)] is the mean squared error of
/// replacing W by its range's mean, so it is at least the error D₃ of the best three-level quantizer of N(0, 1) (Max, 1960).
/// For a log-concave density, the best quantizer is unique (Fleischer, 1964), with thresholds ±t where φ(t) = 2t·Φ(-t), t ≈ 0.612.
/// So, κ/s² ≤ 1 - D₃ = 2·φ(t)²/Φ(-t) = 4t·φ(t) ≈ 0.8098, with equality at m = 0 and s·δ = t.
const INFORMATION_BOUND: f64 = 0.81;

/// B = c·F/(1 + 2r), an upper bound on κ/s² under the careless mixture `careless`, where c is its careful share,
/// F is [`INFORMATION_BOUND`], and r is the smaller of its two odds.
///
/// Carelessness only loses information. The mixture maps each answer's careful probability P to c·(P + o),
/// where its odds o ≥ r don't depend on m, so each of κ's terms is at most
///
/// ```text
/// (c·∂P/∂m)²/(c·(P + o))  ≤  c·s²·(∂P/∂x)²/(P + r)
/// ```
///
/// with x = s·m. So it is enough that (1 + 2r)·Σ (∂P/∂x)²/(P + r) ≤ F over the three answers for every r ≥ 0.
/// Since (1 + 2r)/(P + r) = 2 + (1 - 2P)/(P + r):
///
/// ```text
/// (1 + 2r)·Σ (∂P/∂x)²/(P + r)  =  2·Σ (∂P/∂x)²  +  h(r),    h(r) = Σ (∂P/∂x)²·(1 - 2P)/(P + r)
/// ```
///
/// The first term doesn't depend on r, and h(r) is at most the larger of h(0) and h(∞) = 0 (see below).
/// So the sum is at most the larger of its two limits, both at most 1 - D₃ ≤ F:
///
/// - At r = 0, it is κ/s² without carelessness, at most 1 - D₃ (see [`INFORMATION_BOUND`]).
/// - As r → ∞, it tends to 2·Σ (∂P/∂x)². The three ∂P/∂x are a = φ(x - u), -b = -φ(x + u) and b - a,
///   and a² + b² + (a - b)² ≤ 2·max(a, b)² ≤ 2·φ(0)², so the limit is at most 4·φ(0)² = 2/π ≈ 0.64.
///
/// A term of h with P < ½ is positive and falls as r rises, and a term with P > ½ is negative and rises toward 0. The answers'
/// probabilities sum to 1, so at most one has P > ½. If none does, h(r) ≤ h(0). Otherwise, let Q be that answer's probability:
///
/// ```text
/// (Q + r)²·h′(r)  =  (∂Q/∂x)²·(2Q - 1)  -  Σ (∂P/∂x)²·(1 - 2P)·((Q + r)/(P + r))²    over the other two answers
/// ```
///
/// Each ratio (Q + r)/(P + r) falls as r rises, since P < Q, so the right side rises. Then h′ changes sign at most once,
/// from - to +, and h(r) ≤ max(h(0), h(∞)).
fn information_bound(careless: Careless) -> f64 {
    let r = f64::min(careless.odds(AnswerKind::Pick), careless.odds(AnswerKind::Same));
    careless.careful() * INFORMATION_BOUND / (1. + 2. * r)
}

/// κ, the answer's expected Fisher information about the mean m of its gap N(`mean`, `var`),
/// from the answer's likelihood averaged over the gap under the careless mixture `careless`. Over the three answers y:
///
/// ```text
/// κ = Σ_y (∂P_y/∂m)² / P_y
/// ```
///
/// In expectation, the answer removes var²·κ of the gap's variance:
///
/// ```text
/// E[Var(g | y)] = var + var²·Σ_y P_y·∂²log P_y/∂m²  =  var + var²·(Σ_y ∂²P_y/∂m² - κ)  =  var - var²·κ
/// ```
///
/// The answers' probabilities sum to 1, so their second derivatives sum to 0. The same sum gives "same" from the two picks,
/// so κ takes only one probit per pick.
fn answer_information(gap: Normal, delta: f64, careless: Careless) -> f64 {
    let Normal { mean, var } = gap;
    let s = sharpness(ANSWER_NOISE_VAR, var);
    let picks = answer_likelihood(f64x2::from_array([mean, -mean]), s, f64x2::splat(delta), AnswerKind::Pick, careless);
    let ([first, second], [slope_first, slope_second]) = (picks.value.to_array(), picks.d1.to_array());
    let (same, slope_same) = (1. - first - second, slope_second - slope_first);
    slope_first * slope_first / first + slope_second * slope_second / second + slope_same * slope_same / same
}

#[cfg(test)]
mod tests {
    use super::{INFORMATION_BOUND, Scorer, TIE_TOLERANCE, answer_information, information_bound, tie_boundary};
    use crate::gaussian::Normal;
    use crate::model::{ANSWER_NOISE_VAR, Careless, sharpness};
    use crate::observations::{Bucket, Observations, Pair, PairObs, PairSet, n_pairs};
    use crate::posterior::{Posterior, fit};
    use crate::rank_stats::PairWeights;
    use crate::ranking::TieCost;
    use crate::rng::Rng;
    use crate::special::{exp, ndtr_lanes};
    use crate::testing::{Lcg, TIE_COST, ndtr, normal_density, random_log, twin_log};
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f64::consts::FRAC_1_SQRT_2PI;
    use core::iter::{chain, zip};
    use core::simd::f64x2;

    /// Every pair over `n` items.
    fn all_pairs(n: usize) -> impl Iterator<Item = Pair> {
        #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
        let n = n as u32;
        (0..n).flat_map(move |i| (i + 1..n).map(move |j| Pair::new(i, j)))
    }

    /// The value of asking `pair`.
    fn pair_value(posterior: &Posterior, weights: &PairWeights, tie_z: f64, pair: Pair) -> f64 {
        let phi = |x| FRAC_1_SQRT_2PI * exp(-0.5 * x * x);
        let psi = |x: f64| {
            let [tail, density] = ndtr_lanes(f64x2::splat(-x.abs()));
            (density[0] - x.abs() * tail[0]).max(0.)
        };
        let cov = |i, j| posterior.covariance.get(i, j);

        let n_items = posterior.n_items();
        let delta = posterior.median_delta();
        let class = posterior.twins.class();
        let [x, y] = pair.indices();
        let kappa = answer_information(posterior.gap(x, y), delta, posterior.careless);
        let mut sum = 0.;
        for i in 0..n_items {
            for j in i + 1..n_items {
                let gap = posterior.gap(i, j);
                let (mean, sd) = (gap.mean, gap.sd());
                let boundary = delta + tie_z * sd;
                let boundaries = if boundary > 0. {
                    vec![(boundary, phi(tie_z) / sd), (-boundary, phi(tie_z) / sd)]
                } else {
                    vec![(0., 2. * phi(delta / sd) / sd)]
                };
                let weight = weights.get(class[i], class[j]);
                let density = boundaries.iter().map(|&(at, jump)| jump * psi((mean - at) / sd) / sd).sum::<f64>() * f64::from(weight);
                let shared = cov(i, x) - cov(i, y) - cov(j, x) + cov(j, y);
                sum += density * kappa * shared * shared;
            }
        }
        sum
    }

    /// Whether the scorer's `got` is `want` within half of [`TIE_TOLERANCE`] of `best`, the best value at stake.
    ///
    /// If every computed value is within e of its exact value and the best exact value is B, then the best pair's computed value is
    /// at least B - e, and the ties' floor is at most (B + e)·(1 - T) for T = `TIE_TOLERANCE`. So, the best pair is a tie
    /// when e·(2 - T) ≤ T·B, and then every tie's exact value is at least (B - e)·(1 - T) - e ≥ B·(1 - 2T).
    fn within_rounding(got: f64, want: f64, best: f64) -> bool {
        (got - want).abs() <= 0.5 * TIE_TOLERANCE * best
    }

    #[test]
    fn proposals_cover_every_pair() {
        for n in 0..10 {
            let obs = Observations { pairs: Vec::new(), buckets: vec![Bucket::Fine; n] };
            let posterior = fit(&obs);
            let scorer = Scorer::new(&posterior, TIE_COST);
            let mut shown = PairSet::new(n);
            for count in 0..n_pairs(n) {
                let (pair, _) = scorer.propose(&posterior, &shown, &mut Rng::new(count));
                assert!(!shown.contains(pair), "n = {n}: an unshown pair");
                shown.insert(pair);
            }
            assert!(all_pairs(n).all(|pair| shown.contains(pair)), "n = {n}: every pair was shown");
        }
    }

    #[test]
    fn ties_hold_the_best_unshown_pair() {
        let z = tie_boundary(TIE_COST);
        let mut lcg = Lcg(41);
        for trial in 0..10 {
            let obs = random_log(&mut lcg, 12, 10 + 4 * trial);
            let posterior = fit(&obs);
            let mut shown = PairSet::new(12);
            for answer in &obs.pairs {
                shown.insert(match *answer {
                    PairObs::Pick { above, below } => Pair::new(above, below),
                    PairObs::Same(pair) => pair,
                });
            }
            let weights = PairWeights::new(&posterior);
            let scorer = Scorer::new(&posterior, TIE_COST);
            let value = |pair| pair_value(&posterior, &weights, z, pair);
            let (best_pair, best) = all_pairs(12)
                .filter(|&pair| !shown.contains(pair))
                .map(|pair| (pair, value(pair)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("an unshown pair");
            let ties = scorer.best_pairs(&posterior, &shown);
            assert!(ties.iter().any(|&(pair, _)| pair == best_pair), "trial {trial}: the best unshown pair is a tie");
            for &(pair, reported) in &ties {
                let [x, y] = pair.indices();
                assert!(!shown.contains(pair), "trial {trial}: ({x}, {y}) is unshown");
                let got = value(pair);
                assert!(got >= best * (1. - 2. * TIE_TOLERANCE), "trial {trial}: ({x}, {y}) is worth {got} against the best {best}");
                assert!(within_rounding(reported, got, best), "trial {trial}: ({x}, {y}) reported {reported} against its value {got}");
            }
        }
    }

    #[test]
    fn information_within_bound() {
        let mut lo = 0.;
        let mut hi = 2.;
        for _ in 0..100 {
            let mid = f64::midpoint(lo, hi);
            if normal_density(mid) > 2. * mid * ndtr(-mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let u = f64::midpoint(lo, hi);
        let most = 4. * u * normal_density(u);
        let bound = (100. * most * (1. + 1e-4)).ceil() / 100.;
        assert_eq!(INFORMATION_BOUND.to_bits(), bound.to_bits());
        let var = 1.;
        let s = sharpness(ANSWER_NOISE_VAR, var);
        let mixtures = [[-20., -20.], [-1.5, -6.], [-6., -1.5], [-1., -1.]].map(Careless::from_log_ratios);
        for careless in chain(mixtures, [Careless::prior_median()]) {
            let limit = information_bound(careless) * most * (1. + 1e-4) / INFORMATION_BOUND;
            for i in -600..=600 {
                for j in 1..=240 {
                    let m = f64::from(i) * 0.02 / s;
                    let delta = f64::from(j) * 0.05 / s;
                    let kappa = answer_information(Normal { mean: m, var }, delta, careless) / (s * s);
                    assert!(kappa <= limit, "κ/s² is {kappa} at s·m = {}, s·δ = {}, rates {:?}", s * m, s * delta, careless.rates());
                }
            }
        }
    }

    #[test]
    fn value_is_each_pairs_weighted_removed_variance() {
        let mut rng = Lcg(6);
        let cases = [(3, 9, 0.8), (7, 21, 0.8), (12, 36, 0.8), (7, 21, 0.3), (7, 21, 1.), (12, 3, 0.8), (9, 0, 0.3)];
        let logs = cases.map(|(n, answers, cost)| (random_log(&mut rng, n, answers), cost));
        for (obs, cost) in chain(logs, [(twin_log(), 0.8)]) {
            let n = obs.buckets.len();
            let tie_cost = TieCost::new(cost).expect("in (0, 1]");
            let posterior = fit(&obs);
            let scorer = Scorer::new(&posterior, tie_cost);
            let weights = PairWeights::new(&posterior);
            let z = tie_boundary(tie_cost);
            let wants: Vec<_> = all_pairs(n).map(|pair| pair_value(&posterior, &weights, z, pair)).collect();
            let best = wants.iter().copied().fold(0., f64::max);
            for (pair, &want) in zip(all_pairs(n), &wants) {
                let mut others = PairSet::new(n);
                for other in all_pairs(n).filter(|&other| other != pair) {
                    others.insert(other);
                }
                let scanned = scorer.best_pairs(&posterior, &others);
                let [x, y] = pair.indices();
                assert!(scanned.len() == 1 && scanned[0].0 == pair, "n {n}: ({x}, {y}) is the one unshown pair");
                let value = scanned[0].1;
                assert!(within_rounding(value, want, best), "n {n}, tie cost {cost}, pair ({x}, {y}): {value} against {want}");
            }
        }
    }
}
