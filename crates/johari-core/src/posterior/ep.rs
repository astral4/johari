//! Posterior fitting via expectation propagation (EP).
//!
//! The posterior q = N(μ, Λ⁻¹) is the prior times one Gaussian site per observation, each over its row's variables (x, y).
//! In natural parameters, a product of Gaussians is a sum, so q's precision and shift are the prior's plus every site's
//! pulled back through its row. Each site is matched through its cavity (q's marginal over the row with the site divided out)
//! and its tilted distribution (the cavity times the observation's likelihood).
//!
//! In exact EP, the new site is the Gaussian with the tilted mean and covariance, divided by the cavity. However, the likelihoods
//! aren't necessarily log-concave in (x, y). So, the tilted distribution can be wider than the cavity, and then that site's precision T
//! has a negative eigenvalue and widens q. Sites are instead kept proper (T ⪰ 0). A match takes the proper T that brings cavity × site
//! closest to the tilted distribution (see [`proper_site`]), and its shift still gives cavity × site the tilted mean.
//! An Anderson extrapolation can push a site out of T ⪰ 0, so after each step every site's T is clamped to its positive part.

use super::Posterior;
use super::anderson::Anderson;
use crate::gaussian::{Natural, Natural2, Normal2, Regression, Sym2};
use crate::jet::{Jet, Slopes};
use crate::linalg::{Cholesky, Placed, Structure, Symmetric, View, sub_scaled};
use crate::model::{ANSWER_NOISE_VAR, Careless, Extra, MODEL, MarkProbits, N_EXTRA, Prior, answer_band, reads_cutpoint_gap, sharpness};
use crate::observations::{AnswerKind, Bucket, N_BUCKETS, Observations, PairObs};
use crate::quadrature::{Rule, gauss_hermite};
use crate::special::{exp, exp_lanes};
use crate::twins::Twins;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::array::from_fn as array_from_fn;
use core::f64::consts::FRAC_1_SQRT_2;
use core::f64::math::sqrt;
use core::iter::{chain, repeat_n, zip};
use core::ops::Add;
use core::simd::f64x2;
use core::simd::num::SimdFloat;

/// A site as a Gaussian over its row's variables (x, y) defined by its natural parameters (precision T and shift h).
type Site = Natural2;
type SiteParams = [f64; 5];

fn sites_of(x: &[f64]) -> &[SiteParams] {
    let (sites, rest) = x.as_chunks();
    debug_assert!(rest.is_empty(), "whole sites");
    sites
}

fn sites_of_mut(x: &mut [f64]) -> &mut [SiteParams] {
    let (sites, rest) = x.as_chunks_mut();
    debug_assert!(rest.is_empty(), "whole sites");
    sites
}

/// Parameters of a site.
fn flatten(site: &Site) -> SiteParams {
    [site.precision.xx, site.precision.xy, site.precision.yy, site.shift[0], site.shift[1]]
}

/// The site with parameters `x`.
fn unflatten(x: SiteParams) -> Site {
    Site { precision: Sym2 { xx: x[0], xy: x[1], yy: x[2] }, shift: [x[3], x[4]] }
}

/// Each site of the point `x` with its precision T replaced by its positive part so sites are proper.
fn make_proper(x: &mut [f64]) {
    for site in sites_of_mut(x) {
        let Site { precision, shift } = unflatten(*site);
        *site = flatten(&Site { precision: precision.positive_part(), shift });
    }
}

/// An observation's row as the linear map A = ∂(x, y)/∂φ to the two variables (x, y) that its likelihood depends on.
/// x is the difference of two coordinates of φ, and y is a third coordinate.
///
/// ```text
/// row     x = φ[plus] - φ[minus]    y = φ[y]
/// pair    θᵢ - θⱼ                   log δ           an answer depends on its items only through their gap
/// mark    c₁ - θ                    log(c₂ - c₁)    a mark depends on its item and c₁ only through their offset
/// ```
///
/// A is 2 × dim. A[0, plus] = 1, A[0, minus] = -1, and A[1, y] = 1. All other entries are 0, so only those three indices are stored.
///
/// As the likelihood depends on φ only through A·φ, the observation's EP site is a Gaussian over (x, y) rather than over all of φ,
/// and A converts between the two. Once [placed](Self::placed), [`PlacedRow::push_forward`] takes q to the site's marginal, N(A·μ, A·Σ·Aᵀ)
/// (i.e. q's marginal over the row), and [`PlacedRow::pull_back`] adds the site (T, h) to q's natural parameters as Aᵀ·T·A and Aᵀ·h.
#[derive(Clone, Copy)]
struct RowJacobian {
    plus: usize,
    minus: usize,
    y: usize,
}

impl RowJacobian {
    /// The row of an answer involving items `i` and `j` of `n_items`. x = θᵢ - θⱼ and y = log δ.
    const fn pair(n_items: usize, i: usize, j: usize) -> Self {
        Self { plus: i, minus: j, y: Extra::LogDelta.index(n_items) }
    }

    /// The row of a mark involving `item` of `n_items`. x = c₁ - θ and y = log(c₂ - c₁).
    const fn mark(n_items: usize, item: usize) -> Self {
        Self { plus: Extra::C1.index(n_items), minus: item, y: Extra::LogGap.index(n_items) }
    }

    /// The entries (row, column) of a symmetric matrix over φ that Aᵀ·T·A can make nonzero.
    /// These are also the only entries of Σ read by A·Σ·Aᵀ.
    const fn entries(&self) -> [[usize; 2]; 6] {
        let Self { plus: p, minus: m, y: k } = *self;
        [[p, p], [m, m], [k, k], [p, m], [p, k], [m, k]]
    }

    /// The row placed along `structure`.
    fn placed(&self, structure: &Structure) -> PlacedRow {
        PlacedRow(self.entries().map(|[i, j]| structure.position(i, j)))
    }
}

/// A row placed along Λ's structure.
#[derive(Clone, Copy)]
struct PlacedRow([usize; 6]);

impl PlacedRow {
    /// The distribution N(A·μ, A·Σ·Aᵀ) of (x, y) = A·φ for φ ~ N(`mu`, Σ), with Σ along the structure.
    fn push_forward(&self, mu: &View<Placed>, cov: &View<Symmetric>) -> Normal2 {
        let [p, m, k, pm, pk, mk] = self.0;
        Normal2 { mean: [mu[p] - mu[m], mu[k]], cov: Sym2 { xx: cov[p] + cov[m] - 2. * cov[pm], xy: cov[pk] - cov[mk], yy: cov[k] } }
    }

    /// Multiply a Gaussian over φ (with natural parameters `precision` and `shift`) by `site`,
    /// a Gaussian over (x, y) with natural parameters (T, h). Adds Aᵀ·T·A to `precision` and Aᵀ·h to `shift`.
    fn pull_back(&self, site: &Natural2, precision: &mut View<Symmetric>, shift: &mut View<Placed>) {
        let [p, m, k, pm, pk, mk] = self.0;
        let Natural2 { precision: t, shift: [hx, hy] } = *site;
        precision[p] += t.xx;
        precision[pm] -= t.xx;
        precision[m] += t.xx;
        precision[pk] += t.xy;
        precision[mk] -= t.xy;
        precision[k] += t.yy;
        shift[p] += hx;
        shift[m] -= hx;
        shift[k] += hy;
    }
}

/// Pairs of nodes of a rule over a site's second variable.
const MAX_PAIRS: usize = 4;

/// A pair of nodes (-x, x) of a Gauss-Hermite rule over a site's second variable, plus their weight.
type NodePair = ([f64; 2], f64);

/// The `H` pairs of a Gauss-Hermite `rule` of `K` = 2·`H` nodes, in order of increasing magnitude.
const fn pairs<const K: usize, const H: usize>(rule: &Rule<K>) -> [NodePair; H] {
    assert!(K == 2 * H, "two nodes per pair");
    let mut pairs = [([0.; 2], 0.); H];
    let mut k = 0;
    while k < H {
        pairs[k] = ([rule.nodes[H - 1 - k], rule.nodes[H + k]], rule.weights[H + k]);
        k += 1;
    }
    pairs
}

/// Nodes and weights of `rule` for y ~ N(`mean`, `sd`²).
fn nodes_at(rule: &[NodePair], mean: f64, sd: f64) -> impl Iterator<Item = (f64x2, f64x2)> {
    rule.iter().map(move |&(z, weight)| (f64x2::splat(mean) + f64x2::splat(sd) * f64x2::from_array(z), f64x2::splat(weight)))
}

/// The Gauss-Hermite rule for a site on x and y.
static RULE: [NodePair; MAX_PAIRS] = pairs(&gauss_hermite::<{ 2 * MAX_PAIRS }>());

/// The rule for a site on x alone.
static X_ALONE: [NodePair; 1] = [([0.; 2], 0.5)];

/// Likelihood of an observation.
#[derive(Clone, Copy)]
enum Likelihood {
    Answer(AnswerKind),
    Mark(Bucket),
}

impl Likelihood {
    /// Whether the likelihood depends on its row's second variable, y.
    fn reads_y(self) -> bool {
        match self {
            Self::Answer(_) => true,
            Self::Mark(bucket) => reads_cutpoint_gap(bucket),
        }
    }

    /// The site's cavity from q's marginal with the site divided out. Returns `None` if rounding makes the cavity improper.
    fn cavity(self, site: &Site, marginal: Normal2) -> Option<Cavity> {
        let (natural, regression, rule) = if self.reads_y() {
            let natural = marginal.natural()? - *site;
            (natural, Regression::new(natural.normal()?), &RULE[..])
        } else {
            let natural = marginal.x().natural() - site.x_part();
            if !natural.is_proper() {
                return None;
            }
            (Natural2::of_x(natural), Regression::of_x(natural.normal()), &X_ALONE[..])
        };
        let mut widths = [f64x2::splat(0.); MAX_PAIRS];
        for (width, (y, _)) in zip(&mut widths, nodes_at(rule, regression.mean[1], regression.y_sd)) {
            *width = exp_lanes(y);
        }
        Some(Cavity { natural, regression, rule, widths })
    }

    /// The tilted sums at an answer's cavity with its careful probability as the likelihood. Returns `None` for a mark.
    fn careful_sums(self, cavity: &Cavity) -> Option<MixtureSums> {
        match self {
            Self::Answer(kind) => {
                let s = sharpness(ANSWER_NOISE_VAR, cavity.regression.spread);
                Some(cavity.tilted_sums(|m, width| answer_band(m, s, width, kind)))
            }
            Self::Mark(_) => None,
        }
    }

    /// The proper site that brings cavity × site closest to the tilted distribution (the cavity times the likelihood).
    /// Returns `None` if the tilted covariance is not positive definite.
    fn matched(self, cavity: &Cavity, careful: Option<&MixtureSums>, estimates: Estimates) -> Option<Site> {
        let tilted = match self {
            // An answer's likelihood is (1 - εₚ - εₛ)·(B + oₖ), where B is the careful probability and oₖ is its odds.
            // oₖ is constant in the gap, so its part of the tilted mixture is the cavity itself of weight oₖ.
            Self::Answer(kind) => *careful? + MixtureSums::gaussian(cavity.regression.normal(), estimates.careless.odds(kind)),
            Self::Mark(bucket) => {
                let s = sharpness(estimates.s_b * estimates.s_b, cavity.regression.spread);
                let normalizer = |m, width| MarkProbits::new(m, width, s, bucket).likelihood();
                if !self.reads_y() {
                    return cavity.site_on_x(normalizer);
                }
                cavity.tilted_sums(normalizer)
            }
        };
        cavity.site_matching(tilted)
    }
}

/// x's mean and variance under the tilted ℓ(x)·N(x; m, v) / Z, from the jet `z` in m of Z(m) = E[ℓ(x)] over x ~ N(m, v).
///
/// ```text
/// E[x]   = m + v·∂ln Z/∂m
/// Var[x] = v + v²·∂²ln Z/∂m²
/// ```
fn tilted_moments(z: Jet<f64x2>, m: f64x2, v: f64x2) -> (f64x2, f64x2) {
    let ln_z = z.log_slopes();
    (m + v * ln_z.d1, v + v * v * ln_z.d2)
}

/// Weighted sums over a mixture.
#[derive(Clone, Copy, Default)]
struct MixtureSums {
    total: f64x2,
    x: f64x2,
    y: f64x2,
    xx: f64x2,
    xy: f64x2,
    yy: f64x2,
}

impl MixtureSums {
    /// Add the component at `y`, whose x is N(m, v) tilted by a likelihood ℓ, for m = `x_mean` and v = `x_var`.
    /// `z` is the jet in m of the component's weight W = w·E[ℓ(x)], where w is its node's rule weight.
    /// By the tilted mean and variance (see [`tilted_moments`]), the tilted component has:
    ///
    /// ```text
    /// W·E[x]  = W·m + v·W′
    /// W·E[x²] = W·(v + m²) + 2·m·v·W′ + v²·W″
    /// ```
    ///
    /// These are linear in W's jet, and W is linear in ℓ, so the sums for ℓ₁ + ℓ₂ are those for ℓ₁ plus those for ℓ₂.
    fn add_jet(&mut self, y: f64x2, x_mean: f64x2, x_var: f64x2, z: Jet<f64x2>) {
        let Jet { value: z0, d1: z1, d2: z2 } = z;
        let x_sum = z0 * x_mean + x_var * z1;
        self.total += z0;
        self.x += x_sum;
        self.y += z0 * y;
        self.xx += z0 * (x_var + x_mean * x_mean) + x_var * (f64x2::splat(2.) * x_mean * z1 + x_var * z2);
        self.xy += y * x_sum;
        self.yy += z0 * y * y;
    }

    /// The sums of the Gaussian component `normal` of weight `weight` in the first lane.
    fn gaussian(normal: Normal2, weight: f64) -> Self {
        let Normal2 { mean: [mx, my], cov } = normal;
        let lane = |x| f64x2::from_array([weight * x, 0.]);
        Self {
            total: lane(1.),
            x: lane(mx),
            y: lane(my),
            xx: lane(cov.xx + mx * mx),
            xy: lane(cov.xy + mx * my),
            yy: lane(cov.yy + my * my),
        }
    }

    /// The mixture's mean and covariance.
    fn normal(self) -> Normal2 {
        let [total, ex, ey, exx, exy, eyy] = [self.total, self.x, self.y, self.xx, self.xy, self.yy].map(SimdFloat::reduce_sum);
        let (ex, ey) = (ex / total, ey / total);
        Normal2 { mean: [ex, ey], cov: Sym2 { xx: exx / total - ex * ex, xy: exy / total - ex * ey, yy: eyy / total - ey * ey } }
    }
}

impl Add for MixtureSums {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            total: self.total + other.total,
            x: self.x + other.x,
            y: self.y + other.y,
            xx: self.xx + other.xx,
            xy: self.xy + other.xy,
            yy: self.yy + other.yy,
        }
    }
}

/// A pair of nodes of a rule over y: (weight, y, eʸ, E[x | y]).
type Node = (f64x2, f64x2, f64x2, f64x2);

/// A site's cavity over its row's variables.
struct Cavity {
    natural: Natural2,
    regression: Regression,
    /// The rule over y.
    rule: &'static [NodePair],
    /// eʸ at the rule's nodes over y, two at a time.
    widths: [f64x2; MAX_PAIRS],
}

impl Cavity {
    /// The tilted mixture's sums from `normalizer`(m, eʸ), the jet in m of Z = E[ℓ(x, y)] over x ~ N(m, Var(x | y)).
    /// Each lane holds one of the nodes over y.
    fn tilted_sums(&self, mut normalizer: impl FnMut(f64x2, f64x2) -> Jet<f64x2>) -> MixtureSums {
        let v = f64x2::splat(self.regression.spread);
        let mut sums = MixtureSums::default();
        for (weight, y, width, m) in self.nodes() {
            sums.add_jet(y, m, v, normalizer(m, width).scaled(weight));
        }
        sums
    }

    /// The proper site that brings cavity × site closest to the tilted mixture with the sums `tilted`.
    /// Returns `None` if the tilted covariance is not positive definite.
    ///
    /// 1. The sums give the tilted mean m̂ and covariance V̂ over (x, y). The likelihood's floor keeps their total positive.
    /// 2. [`proper_site`] gives the site's precision T from V̂ and the cavity's precision Λc.
    /// 3. The site's shift h = (Λc + T)·m̂ - hc gives cavity × site the tilted mean. (Under exact EP's T, this would be V̂⁻¹·m̂ - hc.)
    fn site_matching(&self, tilted: MixtureSums) -> Option<Site> {
        let cavity = self.natural;
        let tilted = tilted.normal();
        let precision = proper_site(cavity.precision, tilted.cov)?;
        let shift = (cavity.precision + precision) * tilted.mean;
        Some(Site { precision, shift: [shift[0] - cavity.shift[0], shift[1] - cavity.shift[1]] })
    }

    /// The proper site on x alone that brings cavity × site closest to the tilted distribution. `normalizer`(m, eʸ) is the jet in m
    /// of Z = E[ℓ(x)] over x ~ N(m, v) at the cavity's one node, which [`tilted_moments`] turns into the tilted mean and variance.
    /// Returns `None` if the tilted variance is not finite or not positive.
    fn site_on_x(&self, mut normalizer: impl FnMut(f64x2, f64x2) -> Jet<f64x2>) -> Option<Site> {
        let (m, v) = (f64x2::splat(self.regression.mean[0]), f64x2::splat(self.regression.spread));
        let (mean, var) = tilted_moments(normalizer(m, self.widths[0]), m, v);
        let cavity = self.natural.x_part();
        let precision = proper_site_on_x(cavity.precision, var[0])?;
        Some(Site::of_x(Natural { precision, shift: (cavity.precision + precision) * mean[0] - cavity.shift }))
    }

    /// The rule's nodes over the second variable. Every node shares Var(x | y), the regression's spread.
    fn nodes(&self) -> impl Iterator<Item = Node> + '_ {
        let Regression { mean, y_sd, slope, .. } = self.regression;
        zip(nodes_at(self.rule, mean[1], y_sd), &self.widths).map(move |((y, weight), &width)| {
            let m = f64x2::splat(mean[0]) + f64x2::splat(slope) * (y - f64x2::splat(mean[1]));
            (weight, y, width, m)
        })
    }
}

/// The precision T of the proper site (T ⪰ 0) that brings cavity × site closest to the tilted distribution,
/// given the cavity's precision Λc (`cavity`) and the tilted covariance V̂ (`tilted`). Returns `None` if V̂ is not positive definite.
///
/// "Closest" means the minimum KL(tilted ‖ q) for q = N(m̂, (Λc + T)⁻¹) (i.e. cavity × site) whose shift puts q's mean
/// at the tilted mean m̂. Over all sites, the minimum is at exact EP's T = V̂⁻¹ - Λc, where q has the tilted covariance.
/// However, that T has a negative eigenvalue exactly when the tilted distribution is wider than the cavity in some direction,
/// and a proper T can never make q wider than the cavity.
///
/// The constraint is simplest in coordinates where the cavity is standard normal. Let Λc = R·Rᵀ (see [`Sym2::cholesky`]) and
/// z = Rᵀ·(x, y). In z, the cavity's precision is I, the tilted covariance is W = Rᵀ·V̂·R, and exact EP's T is W⁻¹ - I.
/// The problem then splits along W's eigenvectors: along one where the tilted variance is w (against the cavity's 1),
/// the best proper T is max(1/w - 1, 0). Together, those make the positive part (W⁻¹ - I)₊ (see [`Sym2::positive_part`]),
/// so T = R·(W⁻¹ - I)₊·Rᵀ back in (x, y). q's covariance in z is W with its eigenvalues capped at 1; i.e. the tilted covariance
/// along the eigenvectors where it is narrower than the cavity, and the cavity's along the rest.
///
/// The positive part of V̂⁻¹ - Λc taken directly in (x, y) is also proper but generally not the closest.
fn proper_site(cavity: Sym2, tilted: Sym2) -> Option<Sym2> {
    let r = cavity.cholesky()?;
    let exact = r.on_columns(tilted).inverse()? - Sym2::IDENTITY;
    Some(r.on_rows(exact.positive_part()))
}

/// [`proper_site`] on one axis. Returns exact EP's site precision 1/V̂ - Λc, or 0 if the precision is negative.
/// Returns `None` if V̂ is not finite or not positive.
fn proper_site_on_x(cavity: f64, tilted: f64) -> Option<f64> {
    (tilted.is_finite() && tilted > 0.).then(|| (1. / tilted - cavity).max(0.))
}

/// The slopes ∂ ln Ẑ/∂ν and ∂² ln Ẑ/∂ν² of the mark `bucket` at its cavity,
/// where ν = `s_b`² and Ẑ = E\[Z\] is the tilted normalizer as a function of ν.
fn scale_slopes(bucket: Bucket, cavity: &Cavity, s_b: f64) -> Slopes<f64> {
    let s = sharpness(s_b * s_b, cavity.regression.spread);
    // Ẑ is a weighted sum over the cavity's nodes, so its jet in ν is the same sum of each node's.
    cavity
        .nodes()
        .fold(Jet::default(), |sum, (weight, _, width, m)| sum + MarkProbits::new(m, width, s, bucket).likelihood_in_noise().scaled(weight))
        .map(SimdFloat::reduce_sum)
        .log_slopes()
}

/// Item `item`'s mark's likelihood and row over φ.
fn mark(obs: &Observations, item: usize) -> (Likelihood, RowJacobian) {
    (Likelihood::Mark(obs.buckets[item]), RowJacobian::mark(obs.buckets.len(), item))
}

/// The answer `pair`'s likelihood and row over φ, over `n_items`.
fn answer(n_items: usize, pair: PairObs) -> (Likelihood, RowJacobian) {
    let [i, j] = pair.indices();
    (Likelihood::Answer(pair.kind()), RowJacobian::pair(n_items, i, j))
}

/// q's moments from a set of sites.
struct Moments {
    mu: Placed,
    /// Λ's factor.
    chol: Cholesky,
    /// Every site's marginal under q.
    marginals: Vec<Normal2>,
}

impl Moments {
    /// How far `sites` are from their matches `targets` at q. The first sites are the marks' factors',
    /// each shared by its class' number of marks in `mark_counts`.
    fn residual(&self, sites: &[f64], targets: Vec<SiteParams>, mark_counts: &[f64]) -> Residual {
        let mut f = targets.into_flattened();
        sub_scaled(&mut f, 1., sites);
        let mut u: Vec<_> = zip(&self.marginals, sites_of(&f)).map(|(&marginal, &delta)| fisher_weight(marginal, delta)).collect();
        let largest = u.iter().map(|site| site.iter().map(|x| x * x).sum()).fold(0., f64::max);
        for (site, &count) in zip(&mut u, mark_counts) {
            if count > 1. {
                let weight = sqrt(count);
                for x in site {
                    *x *= weight;
                }
            }
        }
        Residual { f, u: u.into_flattened(), change: sqrt(largest) }
    }
}

/// Weights a site's change `delta` so that it measures how far the change would move q (in units of q's own spread)
/// from q's marginal over the site's row.
///
/// The raw residual sizes a site's change by its parameters, which grow with the site's precision and with its mean's distance from 0,
/// regardless of what the change does to q. The size of a change should depend only on q before and after,
/// not on how the sites or the row variables are parametrized. The KL divergence between q before and after has this property,
/// and the Fisher metric of q is its second-order approximation.
///
/// Each site's change (δT, δh) is rewritten in coordinates where q's marginal over the site's row r is standard normal.
/// Let the marginal be N(m, L·Lᵀ) (see [`Sym2::semidefinite_cholesky`]) and r = m + L·z.
/// Up to a constant, the change adds δhᵀ·r - ½·rᵀ·δT·r = bᵀ·z - ½·zᵀ·S·z to log q, where:
///
/// ```text
/// S = Lᵀ·δT·L
/// b = Lᵀ·(δh - δT·m)
/// ```
///
/// The change takes z from N(0, I) to N((I + S)⁻¹·b, (I + S)⁻¹). So, ‖b‖ is the change in q's mean over the row
/// in Mahalanobis distance (to first order), and S is the change in q's precision over the row (relative to q's own precision).
/// The change's squared norm in the Fisher metric is the variance under q of what it adds to log q.
/// Under z ~ N(0, I), the linear and quadratic terms are uncorrelated, and Var(zᵀ·S·z) = 2·tr(S²), so:
///
/// ```text
/// Var(bᵀ·z - ½·zᵀ·S·z) = ‖b‖² + ½·tr(S²) = b.x² + b.y² + ½·S.xx² + S.xy² + ½·S.yy²
/// ```
///
/// The site's part of the weighted residual is (S.xx / sqrt(2), S.xy, S.yy / sqrt(2), b.x, b.y).
/// The vector is linear in the change, as needed for Anderson acceleration. Its squared norm is the variance above.
fn fisher_weight(Normal2 { mean, cov }: Normal2, delta: SiteParams) -> SiteParams {
    let (Site { precision: dt, shift: dh }, l) = (unflatten(delta), cov.semidefinite_cholesky());
    let dm = dt * mean;
    let (s, [bx, by]) = (l.on_columns(dt), l.transpose_mul([dh[0] - dm[0], dh[1] - dm[1]]));
    [FRAC_1_SQRT_2 * s.xx, s.xy, FRAC_1_SQRT_2 * s.yy, bx, by]
}

/// How far a point's sites are from their matches at its q.
struct Residual {
    /// The residual.
    f: Vec<f64>,
    /// `f` weighted so that each site's norm is in the Fisher metric of q (see [`fisher_weight`]),
    /// times the square root of the number of observations that share the site.
    u: Vec<f64>,
    /// The largest Fisher norm of one site's change.
    change: f64,
}

/// Parameters that a fit estimates with a point rather than in q due to not being coordinates of φ.
/// The estimates approximately maximize the log evidence plus the log prior over coordinates in which the prior is normal.
#[derive(Clone, Copy)]
struct Estimates {
    /// λ = ln `s_b`, the coordinate the mark noise steps in, where its prior is normal.
    lambda: f64,
    /// `s_b` = e^λ, cached for the matches.
    s_b: f64,
    careless: Careless,
}

impl Estimates {
    fn new(lambda: f64, careless: Careless) -> Self {
        Self { lambda, s_b: exp(lambda), careless }
    }

    /// The estimates at their priors' medians.
    fn prior_medians() -> Self {
        Self::new(MODEL.mark_noise_prior().mean(), Careless::prior_median())
    }
}

/// A fit is considered converged once no site's match would change q by this much in units of q's uncertainty (standard deviation).
const TOL: f64 = 1e-4;

/// The number of sweeps before a fit is stopped and considered unconverged.
const SWEEPS: usize = 300;

const TRUST_REGION: f64 = 0.5;

/// A fit in progress.
struct Problem {
    /// Each factor's likelihood.
    likelihoods: Vec<Likelihood>,
    /// Each factor's row placed along Λ's structure.
    rows: Vec<PlacedRow>,
    /// The number of marks in each class of twins, in the order of the marks' factors.
    mark_counts: Vec<f64>,
    /// Each other twin's mark's row with its class' factor.
    twin_rows: Vec<(usize, PlacedRow)>,
    /// The structure of Λ.
    structure: Rc<Structure>,
    /// The prior's natural parameters.
    prior_precision: Symmetric,
    prior_shift: Placed,
}

/// Every site's cavity from its marginal for the factors' `likelihoods`, or `None` where rounding makes it improper.
fn cavities(likelihoods: &[Likelihood], sites: &[SiteParams], marginals: &[Normal2]) -> Vec<Option<Cavity>> {
    zip!(likelihoods, sites, marginals).map(|(likelihood, site, &marginal)| likelihood.cavity(&unflatten(*site), marginal)).collect()
}

/// Each answer's careful sums at its cavity in `cavities` for the answers' likelihoods `answers`, or `None` where it has no cavity.
fn careful_sums(answers: &[Likelihood], cavities: &[Option<Cavity>]) -> Vec<Option<MixtureSums>> {
    zip(answers, cavities).map(|(likelihood, cavity)| cavity.as_ref().and_then(|cavity| likelihood.careful_sums(cavity))).collect()
}

/// Every site matched from its cavity at the estimates `estimates` for the factors' `likelihoods`.
fn matched(
    likelihoods: &[Likelihood],
    sites: &[SiteParams],
    cavities: &[Option<Cavity>],
    careful: &[Option<MixtureSums>],
    estimates: Estimates,
) -> Vec<SiteParams> {
    let careful = chain(repeat_n(None, likelihoods.len() - careful.len()), careful.iter().map(Option::as_ref));
    zip!(likelihoods, sites, cavities, careful)
        .map(|(likelihood, &site, cavity, careful)| {
            cavity
                .as_ref()
                .and_then(|c| likelihood.matched(c, careful, estimates))
                .filter(|new| new.is_finite())
                .map_or(site, |new| flatten(&new))
        })
        .collect()
}

/// q under the marks alone.
struct MarksQ {
    /// The distribution of c.
    cutpoints: Normal2,
    /// Each bucket's E\[θ\].
    theta_mean: [f64; N_BUCKETS],
    /// Each bucket's Var(θ | c).
    spread: [f64; N_BUCKETS],
    /// Each bucket's slopes of E[θ | c] in c.
    slopes: [[f64; 2]; N_BUCKETS],
}

impl MarksQ {
    /// q under `count[b]` marks in each bucket b, each with that bucket's site `sites[b]`.
    /// Returns `None` if rounding makes c's precision not positive definite.
    fn new(sites: &[SiteParams; N_BUCKETS], count: &[f64; N_BUCKETS]) -> Option<Self> {
        const THETA: Natural = MODEL.utility_prior().natural();
        const C1: Natural = MODEL.prior(Extra::C1).natural();
        const GAP: Natural = MODEL.prior(Extra::LogGap).natural();

        let mut c = Natural2 { precision: Sym2 { xx: C1.precision, xy: 0., yy: GAP.precision }, shift: [C1.shift, GAP.shift] };
        let mut spread = [0.; N_BUCKETS];
        let mut slopes = [[0.; 2]; N_BUCKETS];
        let mut theta_shift = [0.; N_BUCKETS];
        for (b, &site, &n) in zip!(0.., sites, count) {
            let Site { precision: t, shift: h } = unflatten(site);
            spread[b] = 1. / (THETA.precision + t.xx);
            slopes[b] = [t.xx * spread[b], t.xy * spread[b]];
            theta_shift[b] = THETA.shift - h[0];
            c.precision = c.precision + Sym2 { xx: n * t.xx, xy: n * t.xy, yy: n * t.yy } - Sym2::outer(n / spread[b], slopes[b]);
            c.shift = [0, 1].map(|k| c.shift[k] + n * (h[k] + slopes[b][k] * theta_shift[b]));
        }
        let cutpoints = c.normal()?;
        let mean = cutpoints.mean;
        let theta_mean = array_from_fn(|b| spread[b] * theta_shift[b] + slopes[b][0] * mean[0] + slopes[b][1] * mean[1]);
        Some(Self { cutpoints, theta_mean, spread, slopes })
    }

    /// The marginal of a mark in `bucket` over its row (c₁ - θ, log(c₂ - c₁)).
    fn mark(&self, bucket: Bucket) -> Normal2 {
        let b = bucket as usize;
        let Normal2 { mean, cov } = self.cutpoints;
        let a = [1. - self.slopes[b][0], -self.slopes[b][1]];
        let wa = cov * a;
        let var = a[0] * wa[0] + a[1] * wa[1] + self.spread[b];
        Normal2 { mean: [mean[0] - self.theta_mean[b], mean[1]], cov: Sym2 { xx: var, xy: wa[1], yy: cov.yy } }
    }

    /// The marginal of an answer about items in buckets `i` and `j` over its row (θᵢ - θⱼ, log δ).
    fn answer(&self, i: Bucket, j: Bucket) -> Normal2 {
        const DELTA: Prior = MODEL.prior(Extra::LogDelta);

        let (i, j) = (i as usize, j as usize);
        let d = [self.slopes[i][0] - self.slopes[j][0], self.slopes[i][1] - self.slopes[j][1]];
        let wd = self.cutpoints.cov * d;
        let var = d[0] * wd[0] + d[1] * wd[1] + self.spread[i] + self.spread[j];
        Normal2 { mean: [self.theta_mean[i] - self.theta_mean[j], DELTA.mean()], cov: Sym2 { xx: var, xy: 0., yy: DELTA.variance() } }
    }
}

/// The number of passes for the marks' own EP fit.
const MARK_PASSES: usize = 8;

/// Starting sites for a fit.
struct Start {
    marks: [SiteParams; N_BUCKETS],
    answers: [[[SiteParams; N_BUCKETS]; N_BUCKETS]; AnswerKind::ALL.len()],
}

impl Start {
    /// The site of a mark in `bucket`.
    const fn mark(&self, bucket: Bucket) -> SiteParams {
        self.marks[bucket as usize]
    }

    /// The site of an answer of kind `kind` about items in buckets `[i, j]`.
    const fn answer(&self, kind: AnswerKind, [i, j]: [Bucket; 2]) -> SiteParams {
        self.answers[kind as usize][i as usize][j as usize]
    }
}

/// The starting sites for a fit for items in `buckets`.
#[inline(never)]
#[optimize(size)]
fn marks_first(buckets: &[Bucket]) -> Start {
    let estimates = Estimates::prior_medians();
    let mut count = [0.; N_BUCKETS];
    for &bucket in buckets {
        count[bucket as usize] += 1.;
    }
    let marks = Bucket::ALL.map(Likelihood::Mark);
    let mut sites = [SiteParams::default(); N_BUCKETS];
    let mut q = MarksQ::new(&sites, &count).expect("the prior's cutpoints are proper");
    for _ in 0..MARK_PASSES {
        let cavities = cavities(&marks, &sites, &Bucket::ALL.map(|bucket| q.mark(bucket)));
        let next_sites = matched(&marks, &sites, &cavities, &[], estimates).try_into().expect("a site per bucket");
        let Some(next_q) = MarksQ::new(&next_sites, &count) else { break };
        sites = next_sites;
        q = next_q;
    }

    let classes = AnswerKind::ALL.map(|kind| Bucket::ALL.map(|i| Bucket::ALL.map(|j| (kind, [i, j]))));
    let classes = classes.as_flattened().as_flattened();
    let answers: Vec<_> = classes.iter().map(|&(kind, _)| Likelihood::Answer(kind)).collect();
    let marginals: Vec<_> = classes.iter().map(|&(_, [i, j])| q.answer(i, j)).collect();
    let zero = vec![SiteParams::default(); classes.len()];
    let cavities = cavities(&answers, &zero, &marginals);
    let matches = matched(&answers, &zero, &cavities, &careful_sums(&answers, &cavities), estimates);
    Start { marks: sites, answers: matches.as_chunks::<N_BUCKETS>().0.as_chunks::<N_BUCKETS>().0.try_into().expect("a table per kind") }
}

impl Problem {
    #[inline(never)]
    #[optimize(size)]
    fn new(obs: &Observations, twins: &Twins) -> (Self, Vec<SiteParams>) {
        let n = obs.buckets.len();
        let structure = Rc::new(Structure::ordered(n + N_EXTRA, N_EXTRA, obs.pairs.iter().map(|pair| pair.indices())));
        let roots = twins.roots();
        let factors = || chain(roots.iter().map(|&root| mark(obs, root)), obs.pairs.iter().map(|&pair| answer(n, pair)));
        let likelihoods = factors().map(|(likelihood, _)| likelihood).collect();
        let rows = factors().map(|(_, row)| row.placed(&structure)).collect();

        let first = marks_first(&obs.buckets);
        let marks = roots.iter().map(|&root| first.mark(obs.buckets[root]));
        let answers = obs.pairs.iter().map(|&pair| first.answer(pair.kind(), pair.indices().map(|item| obs.buckets[item])));
        let start = chain(marks, answers).collect();
        let mark_counts = twins.sizes().to_vec();
        let mut twin_rows = Vec::new();
        for (item, &class, root) in zip!(0.., twins.class(), twins.item_roots()) {
            if root != item {
                debug_assert!(obs.buckets[root] == obs.buckets[item], "a class is in one bucket");
                twin_rows.push((class, RowJacobian::mark(n, item).placed(&structure)));
            }
        }

        let mut prior_precision = Symmetric::zero(&structure);
        let mut prior_shift = Placed::zero(&structure);
        for (coordinate, prior) in MODEL.priors(n).enumerate() {
            let at = structure.position(coordinate, coordinate);
            let Natural { precision, shift } = prior.natural();
            prior_precision[at] = precision;
            prior_shift[at] = shift;
        }

        (Self { likelihoods, rows, mark_counts, twin_rows, structure, prior_precision, prior_shift }, start)
    }

    /// The marks' likelihoods.
    fn marks(&self) -> &[Likelihood] {
        &self.likelihoods[..self.mark_counts.len()]
    }

    /// The answers' likelihoods.
    fn answers(&self) -> &[Likelihood] {
        &self.likelihoods[self.mark_counts.len()..]
    }

    /// q from the prior and `sites`. Returns `None` if:
    /// - Λ is not finite or not positive definite
    /// - μ is not finite
    /// - Λ⁻¹'s selected entries are not finite or its diagonal is not positive
    fn moments(&self, sites: &[SiteParams]) -> Option<Moments> {
        let mut precision = self.prior_precision.clone();
        let mut shift = self.prior_shift.clone();
        for (row, site) in zip(&self.rows, sites) {
            row.pull_back(&unflatten(*site), &mut precision, &mut shift);
        }
        for &(f, row) in &self.twin_rows {
            row.pull_back(&unflatten(sites[f]), &mut precision, &mut shift);
        }
        let chol = Cholesky::factor(Rc::clone(&self.structure), precision)?;
        let covariance = chol.selected_inverse();
        chol.solve_placed(&mut shift);
        let (diagonal, below) = covariance.split_at(shift.len());
        if chain(shift.iter(), below).all(|x| x.is_finite()) && diagonal.iter().all(|&v| v.is_finite() && v > 0.) {
            Some(Moments { marginals: self.marginals(&shift, &covariance), mu: shift, chol })
        } else {
            None
        }
    }

    /// Every site's marginal under q = N(`mu`, `covariance`).
    fn marginals(&self, mu: &View<Placed>, covariance: &View<Symmetric>) -> Vec<Normal2> {
        self.rows.iter().map(|row| row.push_forward(mu, covariance)).collect()
    }

    /// λ after one step from the estimates' toward its estimate.
    fn scale_step(&self, cavities: &[Option<Cavity>], Estimates { lambda, s_b, .. }: Estimates) -> f64 {
        const PRIOR: Prior = MODEL.mark_noise_prior();
        const PRECISION: f64 = 1. / PRIOR.variance();

        let mut evidence = Slopes::default();
        for (&likelihood, &count, cavity) in zip!(self.marks(), &self.mark_counts, cavities) {
            if let (Likelihood::Mark(bucket), Some(cavity)) = (likelihood, cavity) {
                evidence = evidence + scale_slopes(bucket, cavity, s_b).scaled(count);
            }
        }
        let nu = s_b * s_b;
        // The slopes of ν = e^{2λ} in λ.
        let nu_slopes = Slopes { d1: 2. * nu, d2: 4. * nu };
        let objective = evidence.chain(nu_slopes) + PRIOR.log_density_slopes(lambda);
        let step = objective.d1 / f64::max(-objective.d2, PRECISION);
        lambda + step.clamp(-TRUST_REGION, TRUST_REGION)
    }

    /// The careless mixture after one step from `careless` toward its estimate, from the answers' careful sums `careful`.
    ///
    /// The answers' part of the evidence is Z = Πᵢ Ẑᵢ, where Ẑᵢ is answer i's likelihood averaged over its cavity.
    /// Its careful probability averaged over its cavity is the total of its careful sums, which gives its responsibility aᵢ.
    /// With eᵢ being the unit vector of its kind, over the n answers:
    ///
    /// ```text
    /// ∂ln Z/∂η    = Σᵢ aᵢ·eᵢ - n·ε
    /// -∂²ln Z/∂η² = n·(diag(ε) - ε·εᵀ) - Σᵢ aᵢ·(1 - aᵢ)·eᵢ·eᵢᵀ
    /// ```
    fn careless_step(&self, careful: &[Option<MixtureSums>], careless: Careless) -> Careless {
        const PRIOR: Prior = MODEL.careless_prior();
        const PRECISION: f64 = 1. / PRIOR.variance();

        let mut n = 0.;
        let mut expected = [0.; 2];
        let mut lost = [0.; 2];
        for (&likelihood, careful) in zip(self.answers(), careful) {
            if let (Likelihood::Answer(kind), Some(careful)) = (likelihood, careful) {
                let (a, k) = (careless.responsibility(kind, careful.total.reduce_sum()), kind as usize);
                n += 1.;
                expected[k] += a;
                lost[k] += a * (1. - a);
            }
        }

        let eta = careless.log_ratios();
        let rates = careless.rates();
        let gradient = [0, 1].map(|k| PRIOR.log_density_slopes(eta[k]).d1 + expected[k] - n * rates[k]);
        let [p, s] = rates;
        let curvature = Sym2 { xx: n * p * (1. - p) - lost[0], xy: -n * p * s, yy: n * s * (1. - s) - lost[1] };
        let information = Sym2 { xx: PRECISION, xy: 0., yy: PRECISION } + curvature.positive_part();
        let step = information.inverse().expect("the prior's precision keeps the information positive definite") * gradient;
        Careless::from_log_ratios([0, 1].map(|k| eta[k] + step[k].clamp(-TRUST_REGION, TRUST_REGION)))
    }

    /// Perform one sweep at the point (`sites`, `q`). Returns the stepped estimates and the matches.
    #[inline(never)]
    fn sweep(&self, sites: &[SiteParams], q: &Moments, estimates: Estimates) -> (Estimates, Vec<SiteParams>) {
        let cavities = cavities(&self.likelihoods, sites, &q.marginals);
        let careful = careful_sums(self.answers(), &cavities[self.mark_counts.len()..]);
        let estimates = Estimates::new(self.scale_step(&cavities, estimates), self.careless_step(&careful, estimates.careless));
        (estimates, matched(&self.likelihoods, sites, &cavities, &careful, estimates))
    }

    /// Sweeps from the sites `start` until q settles. Returns the fit's point and the number of sweeps taken.
    fn settle(&self, start: Vec<SiteParams>) -> (Point, usize) {
        let mut anderson = Anderson::new();
        let mut estimates = Estimates::prior_medians();
        let mut sites = start.into_flattened();
        let mut best: Option<Point> = None;
        let mut sweeps = 0;
        while let Some(q) = self.moments(sites_of(&sites)) {
            sweeps += 1;
            let (next, targets) = self.sweep(sites_of(&sites), &q, estimates);
            estimates = next;
            let Residual { f, u, change } = q.residual(&sites, targets, &self.mark_counts);
            if best.as_ref().is_none_or(|best| change < best.change) {
                best = Some(Point { q, careless: estimates.careless, change });
            }
            if change < TOL || sweeps == SWEEPS {
                break;
            }
            sites = anderson.step(sites, f, u);
            make_proper(&mut sites);
        }
        (best.expect("the start's q is finite"), sweeps)
    }
}

/// A point that a fit sweeps through.
struct Point {
    q: Moments,
    careless: Careless,
    change: f64,
}

/// The posterior approximation of `obs`.
pub(crate) fn fit(obs: &Observations) -> Posterior {
    let twins = Twins::of(obs);
    let (problem, start) = Problem::new(obs, &twins);
    let (Point { q, careless, .. }, _) = problem.settle(start);
    Posterior::new(q.chol.unplace(&q.mu), q.chol, twins, careless)
}

#[cfg(feature = "bench")]
pub(crate) fn fit_diagnostics(obs: &Observations) -> (usize, bool) {
    let (problem, start) = Problem::new(obs, &Twins::of(obs));
    let (best, sweeps) = problem.settle(start);
    (sweeps, best.change < TOL)
}

#[cfg(feature = "bench")]
pub(crate) struct Bench {
    problem: Problem,
    sites: Vec<SiteParams>,
    q: Moments,
    estimates: Estimates,
    targets: Vec<SiteParams>,
}

#[cfg(feature = "bench")]
impl Bench {
    pub(crate) fn new(obs: &Observations) -> Self {
        let ((problem, sites), estimates) = (Problem::new(obs, &Twins::of(obs)), Estimates::prior_medians());
        let q = problem.moments(&sites).expect("the start's q is finite");
        let targets = problem.sweep(&sites, &q, estimates).1;
        Self { problem, sites, q, estimates, targets }
    }

    /// The sweep's estimates and matches as a checksum.
    pub(crate) fn sweep(&self) -> f64 {
        let (Estimates { s_b, careless, .. }, targets) = self.problem.sweep(&self.sites, &self.q, self.estimates);
        s_b + careless.rates()[0] + targets.iter().map(|&site| unflatten(site).precision.xx).sum::<f64>()
    }

    /// q from the sweep's matches as a checksum.
    pub(crate) fn moments(&self) -> f64 {
        self.problem.moments(&self.targets).map_or(f64::NAN, |q| q.mu[0])
    }
}

#[cfg(test)]
mod tests {
    use super::{Likelihood, MixtureSums, Site, fisher_weight, proper_site, unflatten};
    use crate::gaussian::{Normal2, Sym2};
    use crate::jet::Jet;
    use crate::linalg::dot;
    use crate::observations::AnswerKind;
    use crate::testing::{Lcg, close, entries, entries_within, is_psd, random_spd};
    use core::iter::zip;
    use core::simd::f64x2;
    use core::simd::num::SimdFloat;
    use libm::{exp, log};

    #[test]
    fn fisher_metric_weighting() {
        let mut rng = Lcg(5);
        for trial in 0..3000 {
            let cov = if trial % 4 == 3 { Sym2::outer(1., [rng.sym(2.), rng.sym(2.)]) } else { random_spd(&mut rng) };
            let marginal = Normal2 { mean: [rng.sym(4.), rng.sym(4.)], cov };
            let change = [rng.sym(2.), rng.sym(2.), rng.sym(2.), rng.sym(2.), rng.sym(2.)];
            let u = fisher_weight(marginal, change);
            let got = dot(&u, &u);

            let Site { precision: dt, shift: dh } = unflatten(change);
            // A = V·δT.
            let a = [
                [cov.xx * dt.xx + cov.xy * dt.xy, cov.xx * dt.xy + cov.xy * dt.yy],
                [cov.xy * dt.xx + cov.yy * dt.xy, cov.xy * dt.xy + cov.yy * dt.yy],
            ];
            // tr(A²).
            let trace = a[0][0] * a[0][0] + 2. * a[0][1] * a[1][0] + a[1][1] * a[1][1];
            let dm = dt * marginal.mean;
            let r = [dh[0] - dm[0], dh[1] - dm[1]];
            let vr = cov * r;
            let want = 0.5 * trace + dot(&r, &vr);

            assert!(close(got, want, 1e-12), "trial {trial}: |u|² = {got} vs. target = {want}");
        }
    }

    /// -ln|P| + tr(P·V̂); i.e. twice the KL divergence from N(m̂, V̂) to N(m̂, P⁻¹), up to a constant.
    fn kl(p: Sym2, v: Sym2) -> f64 {
        -log(p.det()) + p.xx * v.xx + 2. * p.xy * v.xy + p.yy * v.yy
    }

    #[test]
    fn proper_sites_are_the_constrained_kl_minimum() {
        let mut rng = Lcg(7);
        for _ in 0..3000 {
            let (cavity, tilted) = (random_spd(&mut rng), random_spd(&mut rng));
            let p = cavity + proper_site(cavity, tilted).expect("proper tilted covariance");
            assert!(is_psd(p - cavity), "P ⪰ Λc");
            let wider = !is_psd(cavity.inverse().expect("proper") - tilted);
            if !wider {
                let exact = tilted.inverse().expect("proper");
                let [got, want] = [p, exact].map(entries);
                assert!(zip(got, want).all(|(got, want)| close(got, want, 1e-9)), "V̂⁻¹ when nothing widens: {got:?} against {want:?}");
            }
            for _ in 0..20 {
                let x = rng.sym(1.);
                let y = rng.sym(1.);
                let s = rng.sym(0.3).abs();
                let other = p + Sym2::outer(s, [x, y]);
                let site = p - cavity;
                let shrink = cavity + Sym2 { xx: 0.9 * site.xx, xy: 0.9 * site.xy, yy: 0.9 * site.yy };
                for alternative in [other, shrink] {
                    assert!(
                        kl(p, tilted) <= kl(alternative, tilted) + 1e-9 * (1. + kl(p, tilted).abs()),
                        "a proper alternative does better"
                    );
                }
            }
        }
    }

    #[test]
    fn near_diagonal_problems_split_by_axis() {
        let mut rng = Lcg(17);
        let near = |got, want, scale| entries_within(got, want, 1e-9 * scale);
        let cross = |rng: &mut Lcg, scale| {
            if rng.below(4) == 0 { 0. } else { rng.sym(1.) * exp(rng.sym(10.)) * 1e-16 * scale }
        };
        for _ in 0..2000 {
            let diagonal = [rng.sym(3.) * exp(rng.sym(2.)), rng.sym(3.) * exp(rng.sym(2.))];
            let scale = diagonal[0].abs() + diagonal[1].abs();
            let part = Sym2 { xx: diagonal[0], xy: cross(&mut rng, scale), yy: diagonal[1] }.positive_part();
            let want = Sym2 { xx: diagonal[0].max(0.), xy: 0., yy: diagonal[1].max(0.) };
            assert!(near(part, want, scale), "positive part of {diagonal:?}: {:?}", entries(part));

            let cavity = Sym2 { xx: exp(rng.sym(3.)), xy: 0., yy: exp(rng.sym(3.)) };
            let tilted = Sym2 { xx: exp(rng.sym(3.)), xy: 0., yy: exp(rng.sym(3.)) };
            let tilted = Sym2 { xy: cross(&mut rng, tilted.xx + tilted.yy), ..tilted };
            let t = proper_site(cavity, tilted).expect("proper tilted covariance");
            let want = Sym2 { xx: (1. / tilted.xx - cavity.xx).max(0.), xy: 0., yy: (1. / tilted.yy - cavity.yy).max(0.) };
            let scale = cavity.xx + cavity.yy + 1. / tilted.xx + 1. / tilted.yy;
            let [shown_cavity, shown_tilted, got, wanted] = [cavity, tilted, t, want].map(entries);
            assert!(near(t, want, scale), "cavity {shown_cavity:?}, tilted {shown_tilted:?}: {got:?} vs. {wanted:?}");
        }
    }

    #[test]
    fn constant_likelihood_tilted_cavity() {
        // The rule's sums for a constant likelihood are the cavity's own moments at that weight.
        let mut rng = Lcg(19);
        let zero = Site { precision: Sym2 { xx: 0., xy: 0., yy: 0. }, shift: [0.; 2] };
        for trial in 0..3000 {
            let marginal = Normal2 { mean: [rng.sym(4.), rng.sym(4.)], cov: random_spd(&mut rng) };
            let cavity = Likelihood::Answer(AnswerKind::Pick).cavity(&zero, marginal).expect("a proper cavity");
            let weight = exp(rng.sym(3.));
            let constant = Jet { value: f64x2::splat(weight), d1: f64x2::splat(0.), d2: f64x2::splat(0.) };
            let sums = [cavity.tilted_sums(|_, _| constant), MixtureSums::gaussian(cavity.regression.normal(), weight)];
            let [summed, closed] = sums.map(|s| [s.total, s.x, s.y, s.xx, s.xy, s.yy].map(SimdFloat::reduce_sum));
            assert!(zip(summed, closed).all(|(got, want)| close(got, want, 1e-12)), "trial {trial}: {summed:?} vs. {closed:?}");
        }
    }
}
