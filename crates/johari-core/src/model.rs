//! The statistical model.

use crate::gaussian::{Natural, Normal};
use crate::jet::{Jet, Slopes};
use crate::observations::{AnswerKind, Bucket, N_BUCKETS};
use crate::special::{exp, ndtr_lanes};
use core::f64::consts::LN_2;
use core::f64::math::sqrt;
use core::iter::repeat_n;
use core::simd::f64x2;

/// The number of shared parameters appended to θ: c₁, log(c₂ - c₁), log δ.
pub(crate) const N_EXTRA: usize = Extra::ALL.len();

/// A shared parameter.
#[derive(Clone, Copy)]
pub(crate) enum Extra {
    /// c₁.
    C1 = 0,
    /// log(c₂ - c₁).
    LogGap = 1,
    /// log δ.
    LogDelta = 2,
}

impl Extra {
    const ALL: [Self; 3] = [Self::C1, Self::LogGap, Self::LogDelta];

    /// The parameter's index in φ over `n_items` items, which holds θ and then the shared parameters in `ALL`'s order.
    pub(crate) const fn index(self, n_items: usize) -> usize {
        n_items + self as usize
    }
}

const _: () = {
    let mut k = 0;
    while k < N_EXTRA {
        assert!(Extra::ALL[k] as usize == k, "Extra::ALL must be in φ's order");
        k += 1;
    }
};

/// Normal-distribution model prior.
#[derive(Clone, Copy)]
pub(crate) struct Prior {
    /// Mean.
    loc: f64,
    /// Standard deviation.
    scale: f64,
}

impl Prior {
    /// The natural parameters.
    pub(crate) const fn natural(self) -> Natural {
        Normal { mean: self.loc, var: self.variance() }.natural()
    }

    /// The mean.
    pub(crate) const fn mean(&self) -> f64 {
        self.loc
    }

    /// The variance.
    pub(crate) const fn variance(&self) -> f64 {
        self.scale * self.scale
    }

    /// The slopes of the log density at `x`.
    pub(crate) fn log_density_slopes(&self, x: f64) -> Slopes<f64> {
        Slopes { d1: -(x - self.loc) / self.variance(), d2: -1. / self.variance() }
    }
}

/// Fixed constants of the observation model and priors.
///
/// The utility scale is arbitrary, but the pair channel fixes it since a perceived gap carries noise with sd 1/β. Scaling θ by c
/// scales σ₀ and `c1`'s mean and sd by c, shifts the means of `log_gap`, `log_delta` and `log_s_b` by log c, and divides β by c.
pub(crate) struct ModelConfig {
    /// Prior sd σ₀ of the utilities, θᵢ ~ N(0, σ₀²).
    sigma0: f64,
    /// Pair sharpness β; i.e. the inverse sd of the noise on a perceived gap.
    beta_pair: f64,
    /// Prior of log δ, where δ is the just-noticeable difference (indifference threshold).
    /// The sd guards against an early run of "same" answers pushing up δ.
    log_delta: Prior,
    /// Prior of each careless log-ratio ηₖ.
    careless: Prior,
    /// Prior of the log bucket noise scale.
    log_s_b: Prior,
    /// Prior of the lower cutpoint c₁.
    c1: Prior,
    /// Prior of log(c₂ - c₁), where c₂ - c₁ is the cutpoint gap.
    log_gap: Prior,
    /// Weight of a uniformly random bucket mixed into each mark. Guards against mis-keying.
    bucket_slip: f64,
}

impl ModelConfig {
    /// The prior of each coordinate of φ over `n_items` items in φ's order.
    pub(crate) fn priors(&self, n_items: usize) -> impl Iterator<Item = Prior> + use<> {
        repeat_n(self.utility_prior(), n_items).chain(Extra::ALL.map(|extra| self.prior(extra)))
    }

    /// The prior of log `s_b`, the log bucket noise scale.
    pub(crate) const fn mark_noise_prior(&self) -> Prior {
        self.log_s_b
    }

    /// The prior of each careless log-ratio ηₖ.
    pub(crate) const fn careless_prior(&self) -> Prior {
        self.careless
    }

    /// The prior of an item's utility θᵢ, the same for every item.
    pub(crate) const fn utility_prior(&self) -> Prior {
        Prior { loc: 0., scale: self.sigma0 }
    }

    /// The prior of a shared parameter.
    pub(crate) const fn prior(&self, extra: Extra) -> Prior {
        match extra {
            Extra::C1 => self.c1,
            Extra::LogGap => self.log_gap,
            Extra::LogDelta => self.log_delta,
        }
    }
}

const LN_2_25: f64 = 0.810_930_216_216_328_8;

/// ln(3/74), or ln(0.0375/0.925).
const LN_3_OVER_74: f64 = -3.205_452_804_536_06;

pub(crate) const MODEL: ModelConfig = ModelConfig {
    sigma0: 3.75,
    beta_pair: 1.25,
    log_delta: Prior { loc: 0., scale: 0.7 },
    careless: Prior { loc: LN_3_OVER_74, scale: 1. },
    log_s_b: Prior { loc: LN_2, scale: 0.5 },
    c1: Prior { loc: -1.125, scale: 2.25 },
    log_gap: Prior { loc: LN_2_25, scale: 0.75 },
    bucket_slip: 0.02,
};

const _: () = {
    let priors = [MODEL.utility_prior(), MODEL.c1, MODEL.log_gap, MODEL.log_delta, MODEL.log_s_b, MODEL.careless];
    let mut k = 0;
    while k < priors.len() {
        assert!(priors[k].natural().is_proper(), "every prior's precision must be finite and positive");
        k += 1;
    }
};

/// The careless mixture of answers. An answer is a careless pick with probability εₚ, of either item evenly,
/// a careless "same" with probability εₛ, and careful with probability 1 - εₚ - εₛ.
///
/// The fit estimates the rates through their log-ratios to careful answering, ηₖ = ln(εₖ/(1 - εₚ - εₛ)). So, εₖ = e^ηₖ/(1 + e^ηₚ + e^ηₛ).
#[derive(Clone, Copy)]
pub(crate) struct Careless {
    /// [ηₚ, ηₛ].
    log_ratios: [f64; 2],
    /// The careful rate 1 - εₚ - εₛ.
    careful: f64,
    /// [½·e^ηₚ, e^ηₛ].
    odds: [f64; 2],
}

impl Careless {
    /// The mixture at the log-ratios [ηₚ, ηₛ].
    pub(crate) fn from_log_ratios(log_ratios: [f64; 2]) -> Self {
        let [pick, same] = log_ratios.map(exp);
        Self { log_ratios, careful: 1. / (1. + pick + same), odds: [0.5 * pick, same] }
    }

    /// The mixture with both log-ratios at their prior's median.
    pub(crate) fn prior_median() -> Self {
        let median = MODEL.careless.mean();
        Self::from_log_ratios([median, median])
    }

    /// The log-ratios [ηₚ, ηₛ].
    pub(crate) const fn log_ratios(self) -> [f64; 2] {
        self.log_ratios
    }

    /// The careful rate 1 - εₚ - εₛ.
    pub(crate) const fn careful(self) -> f64 {
        self.careful
    }

    /// The odds oₖ of an answer of kind `kind`. If the answer's careful probability is B,
    /// then its expected likelihood is (1 - εₚ - εₛ)·(B + oₖ), so the odds that it was careless are oₖ/B.
    pub(crate) const fn odds(self, kind: AnswerKind) -> f64 {
        self.odds[kind as usize]
    }

    /// The probability of giving of an answer of kind `kind` carelessly.
    const fn floor(self, kind: AnswerKind) -> f64 {
        self.careful * self.odds(kind)
    }

    /// The rates [εₚ, εₛ].
    pub(crate) const fn rates(self) -> [f64; 2] {
        [2. * self.floor(AnswerKind::Pick), self.floor(AnswerKind::Same)]
    }

    /// The probability a = oₖ/(B + oₖ) that an answer of kind k = `kind` was careless, where B = `band` is its careful probability
    /// and oₖ is its odds. This gives the slopes in η = [ηₚ, ηₛ] of ln Ẑ, for the answer's expected likelihood Ẑ = (1 - εₚ - εₛ)·(B + oₖ).
    /// With ε = [εₚ, εₛ] and eₖ the unit vector of kind k:
    ///
    /// ```text
    /// ln Ẑ        = ln(B + oₖ) - ln(1 + e^ηₚ + e^ηₛ)
    /// ∂ln Ẑ/∂η    = a·eₖ - ε
    /// -∂²ln Ẑ/∂η² = diag(ε) - ε·εᵀ - a·(1 - a)·eₖ·eₖᵀ
    /// ```
    pub(crate) fn responsibility(self, kind: AnswerKind, band: f64) -> f64 {
        let odds = self.odds(kind);
        odds / (band + odds)
    }
}

const _: () = assert!(MODEL.bucket_slip > 0. && MODEL.bucket_slip <= 1., "every mark's likelihood is positive");

/// The variance that an answer's noise adds to the perceived gap.
pub(crate) const ANSWER_NOISE_VAR: f64 = 1. / (MODEL.beta_pair * MODEL.beta_pair);

/// The sharpness of a probit in the mean of a quantity of variance `var` read through Gaussian noise of variance `noise_var`.
pub(crate) fn sharpness(noise_var: f64, var: f64) -> f64 {
    1. / sqrt(noise_var + var)
}

/// The probability of an answer of kind `kind` with its first two derivatives in the gap's mean, P, dP/dm and d²P/dm²,
/// for a gap N(`mean`, v) and just-noticeable difference `delta`, given the probits' sharpness `s` at v (see below).
///
/// A respondent perceives the gap g plus Gaussian noise with sd 1/β, picking the first above the just-noticeable difference δ,
/// the second below -δ, and "same" in between:
///
/// ```text
/// g       = θ_above - θ_below
/// P(pick) = Φ(β·(g - δ))
/// P(same) = Φ(β·(g + δ)) - Φ(β·(g - δ))
/// ```
///
/// which, with the mirrored pick Φ(-β·(g + δ)), sum to 1. Then the careless mixture `careless` mixes in careless picks
/// at rate εₚ, of either item evenly, and careless "same" answers at rate εₛ:
///
/// ```text
/// P_obs(pick) = (1 - εₚ - εₛ)·P(pick) + εₚ/2
/// P_obs(same) = (1 - εₚ - εₛ)·P(same) + εₛ
/// ```
///
/// so no answer is less likely than its careless floor. Averaged over g ~ N(m, v), each probit is the same probit of m
/// at sharpness s = β / sqrt(1 + β²·v), so s gives the expected likelihood as a function of m.
pub(crate) fn answer_likelihood(mean: f64x2, s: f64, delta: f64x2, kind: AnswerKind, careless: Careless) -> Jet<f64x2> {
    answer_band(mean, s, delta, kind).affine(f64x2::splat(careless.careful), f64x2::splat(careless.floor(kind)))
}

/// The careful probability of an answer of kind `kind` with its first two derivatives in the gap's mean,
/// P(pick) or P(same) of [`answer_likelihood`] before the careless mixture.
pub(crate) fn answer_band(mean: f64x2, s: f64, delta: f64x2, kind: AnswerKind) -> Jet<f64x2> {
    let s = f64x2::splat(s);
    let pick = s * (mean - delta);
    let (upper, lower) = match kind {
        AnswerKind::Pick => (pick, None),
        AnswerKind::Same => (s * (mean + delta), Some(pick)),
    };
    Band::new(s, upper, lower).jet(Probit::jet_in_mean)
}

#[derive(Clone, Copy)]
struct Band {
    upper: Probit,
    /// `None` for a band open below.
    lower: Option<Probit>,
}

impl Band {
    /// A band from probits at `lower` and `upper`, both with slope `slope`.
    fn new(slope: f64x2, upper: f64x2, lower: Option<f64x2>) -> Self {
        Self { upper: Probit::at(slope, upper), lower: lower.map(|w| Probit::at(slope, w)) }
    }

    /// The band's probability with the derivatives that `jet` gives each probit.
    fn jet(self, jet: impl Fn(Probit) -> Jet<f64x2>) -> Jet<f64x2> {
        let upper = jet(self.upper);
        self.lower.map_or(upper, |lower| upper - jet(lower))
    }
}

/// A probit Φ(w) of a mean m. The argument w = σ·s·(m + k) is m shifted by a constant k, scaled by the sharpness
/// s = 1/sqrt(ν + v) of a noise variance ν and the variance v about m (see [`sharpness`]), and signed by σ = ±1.
#[derive(Clone, Copy)]
struct Probit {
    w: f64x2,
    slope: f64x2,
    cdf: f64x2,
    pdf: f64x2,
}

impl Probit {
    fn at(slope: f64x2, w: f64x2) -> Self {
        let [cdf, pdf] = ndtr_lanes(w);
        Self { w, slope, cdf, pdf }
    }

    /// Φ(w) with its first two derivatives in m. With Φ′(w) = φ(w) and Φ″(w) = -w·φ(w):
    ///
    /// ```text
    /// ∂Φ(w)/∂m   = σ·s·φ(w)
    /// ∂²Φ(w)/∂m² = -s²·w·φ(w)
    /// ```
    fn jet_in_mean(self) -> Jet<f64x2> {
        let Self { w, slope, cdf, pdf } = self;
        Jet { value: cdf, d1: slope * pdf, d2: slope * slope * (-w * pdf) }
    }

    /// Φ(w) with its first two derivatives in ν. Since ν enters w only through s, w ∝ (ν + v)^(-1/2),
    /// so ∂w/∂ν = -½·w·s² and ∂²w/∂ν² = ¾·w·s⁴. By the chain rule:
    ///
    /// ```text
    /// ∂Φ(w)/∂ν   = φ(w)·∂w/∂ν                     = -½·s²·w·φ(w)
    /// ∂²Φ(w)/∂ν² = φ(w)·∂²w/∂ν² - w·φ(w)·(∂w/∂ν)² = ¼·s⁴·(3w - w³)·φ(w)
    /// ```
    ///
    /// Equivalently, ν adds to the variance about m, so by the heat equation these are ½·∂²Φ(w)/∂m² and ¼·∂⁴Φ(w)/∂m⁴, respectively.
    fn jet_in_noise(self) -> Jet<f64x2> {
        let Self { w, slope, cdf, pdf } = self;
        let s2 = slope * slope;
        Jet {
            value: cdf,
            d1: f64x2::splat(-0.5) * s2 * w * pdf,
            d2: f64x2::splat(0.25) * s2 * s2 * (f64x2::splat(3.) * w - w * w * w) * pdf,
        }
    }
}

/// The band of a mark's likelihood at two offsets c₁ - θ, from which follow its probability and its derivatives
/// in the offset's mean and in the squared noise scale.
///
/// A bucket mark is modeled by an ordinal probit with cutpoints c₁ < c₂ and noise scale `s_b` under a uniform slip:
///
/// ```text
/// P(meh)   = Φ(z₁)
/// P(fine)  = Φ(z₂) - Φ(z₁)
/// P(love)  = 1 - Φ(z₂) = Φ(-z₂)
/// zₖ       = (cₖ - θ) / s_b
/// P_obs(b) = (1 - slip)·P(b) + slip / 3
/// ```
///
/// so no mark is less likely than slip / 3. Averaged over the offset ~ N(m, v), each probit is the same probit of m
/// at the noise scale widened by v in variance, at sharpness s = 1/sqrt(`s_b`² + v),
/// which gives the expected likelihood as a function of m and of ν = `s_b`².
#[derive(Clone, Copy)]
pub(crate) struct MarkProbits(Band);

impl MarkProbits {
    /// The band of the mark `bucket` at offset means `mean`, cutpoint gaps `width`, and sharpness `s`.
    pub(crate) fn new(mean: f64x2, width: f64x2, s: f64, bucket: Bucket) -> Self {
        let s = f64x2::splat(s);
        let (z1, z2) = (s * mean, s * (mean + width));
        let (slope, upper, lower) = match bucket {
            Bucket::Meh => (s, z1, None),
            Bucket::Fine => (s, z2, Some(z1)),
            Bucket::Love => (-s, -z2, None),
        };
        Self(Band::new(slope, upper, lower))
    }

    /// The mark's probability with its first two derivatives in the offset's mean.
    pub(crate) fn likelihood(self) -> Jet<f64x2> {
        self.combined(Probit::jet_in_mean)
    }

    /// The mark's probability with its first two derivatives in ν = `s_b`².
    pub(crate) fn likelihood_in_noise(self) -> Jet<f64x2> {
        self.combined(Probit::jet_in_noise)
    }

    /// The mark's probability with the derivatives that `jet` gives each probit.
    fn combined(self, jet: impl Fn(Probit) -> Jet<f64x2>) -> Jet<f64x2> {
        #[expect(clippy::cast_precision_loss, reason = "N_BUCKETS is 3")]
        let floor = MODEL.bucket_slip / N_BUCKETS as f64;
        self.0.jet(jet).affine(f64x2::splat(1. - MODEL.bucket_slip), f64x2::splat(floor))
    }
}

/// Whether the likelihood of the mark `bucket` reads the cutpoint gap.
pub(crate) const fn reads_cutpoint_gap(bucket: Bucket) -> bool {
    !matches!(bucket, Bucket::Meh)
}

#[cfg(test)]
mod tests {
    use super::{ANSWER_NOISE_VAR, Careless, MarkProbits, answer_likelihood, sharpness};
    use crate::jet::Jet;
    use crate::observations::{AnswerKind, Bucket};
    use crate::testing::{Lcg, close};
    use core::simd::f64x2;
    use libm::{exp, log};

    /// Whether `jets` sum to the jet of 1.
    fn sum_to_one(jets: [Jet<f64x2>; 3]) -> bool {
        let Jet { value, d1, d2 } = jets[0] + jets[1] + jets[2];
        (0..2).all(|k| close(value[k], 1., 1e-12) && close(d1[k], 0., 1e-12) && close(d2[k], 0., 1e-12))
    }

    /// Whether `jet`'s first derivative is positive.
    fn increasing(jet: Jet<f64x2>) -> bool {
        jet.d1.to_array().iter().all(|&d| d > 0.)
    }

    #[test]
    fn outcome_jets_sum_to_one() {
        let mut rng = Lcg(13);
        for _ in 0..2000 {
            let mean = f64x2::from_array([rng.sym(4.), rng.sym(4.)]);
            let width = f64x2::from_array([exp(rng.sym(1.)), exp(rng.sym(1.))]);
            let nu = exp(rng.sym(1.5));
            let v = rng.sym(1.).abs();
            let marks = Bucket::ALL.map(|bucket| MarkProbits::new(mean, width, sharpness(nu, v), bucket));
            for jet in [MarkProbits::likelihood, MarkProbits::likelihood_in_noise] {
                assert!(sum_to_one(marks.map(jet)), "marks at {mean:?}, {width:?}, ν = {nu}, v = {v}");
            }
            let [meh, _, love] = marks.map(MarkProbits::likelihood);
            assert!(increasing(meh) && increasing(Jet { d1: -love.d1, ..love }), "marks at {mean:?}, {width:?}, ν = {nu}, v = {v}");

            let s = sharpness(ANSWER_NOISE_VAR, v);
            let careless = Careless::from_log_ratios([rng.sym(4.) - 3., rng.sym(4.) - 3.]);
            let first = answer_likelihood(mean, s, width, AnswerKind::Pick, careless);
            let second = {
                let x = answer_likelihood(-mean, s, width, AnswerKind::Pick, careless);
                Jet { d1: -x.d1, ..x }
            };
            let same = answer_likelihood(mean, s, width, AnswerKind::Same, careless);
            assert!(sum_to_one([first, second, same]) && increasing(first), "answers at {mean:?}, δ = {width:?}, v = {v}");
        }
    }

    #[test]
    fn careless_rates_follow_log_ratios() {
        let mut rng = Lcg(5);
        for _ in 0..1000 {
            let ratios = [rng.sym(6.) - 3., rng.sym(6.) - 3.];
            let careless = Careless::from_log_ratios(ratios);
            let [pick, same] = ratios.map(exp);
            let total = 1. + pick + same;
            let ([p, s], careful) = (careless.rates(), careless.careful());
            assert!(close(p, pick / total, 1e-8) && close(s, same / total, 1e-8), "{ratios:?} gave rates {p}, {s}");
            assert!(close(careful, 1. / total, 1e-8), "{ratios:?} gave the careful rate {careful}");
            assert!(p > 0. && s > 0. && p + s < 1., "rates {p}, {s}");
        }
    }

    #[test]
    fn careless_responsibility_slopes() {
        let mut rng = Lcg(9);
        let (h, h2) = (1e-6, 1e-4);
        for _ in 0..1000 {
            let band = rng.unit();
            let ratios = [rng.sym(4.) - 3., rng.sym(4.) - 3.];
            for kind in AnswerKind::ALL {
                // [cₚ, cₛ] and eₖ.
                let (shares, e) = match kind {
                    AnswerKind::Pick => ([0.5, 0.], [1., 0.]),
                    AnswerKind::Same => ([0., 1.], [0., 1.]),
                };
                let ln_z = |[pick, same]: [f64; 2]| {
                    let (pick, same) = (exp(pick), exp(same));
                    log((band + shares[0] * pick + shares[1] * same) / (1. + pick + same))
                };
                let careless = Careless::from_log_ratios(ratios);
                let z = careless.careful() * (band + careless.odds(kind));
                assert!(close(z, exp(ln_z(ratios)), 1e-8), "B = {band}, η = {ratios:?}: Ẑ = {z} against {}", exp(ln_z(ratios)));
                let a = careless.responsibility(kind, band);
                let rates = careless.rates();
                // ln Ẑ with each x of (k, x) in `steps` added to ηₖ.
                let at = |steps: &[(usize, _)]| {
                    let mut eta = ratios;
                    for &(k, x) in steps {
                        eta[k] += x;
                    }
                    ln_z(eta)
                };
                for k in 0..2 {
                    let slope = (at(&[(k, h)]) - at(&[(k, -h)])) / (2. * h);
                    let want = a * e[k] - rates[k];
                    assert!(close(want, slope, 1e-6), "B = {band}, η = {ratios:?}, slope {k}: {want} against {slope}");
                    for l in 0..2 {
                        let [pp, pm, mp, mm] = [(h2, h2), (h2, -h2), (-h2, h2), (-h2, -h2)].map(|(x, y)| at(&[(k, x), (l, y)]));
                        let curvature = -(pp - pm - mp + mm) / (4. * h2 * h2);
                        let want = f64::from(k == l) * rates[k] - rates[k] * rates[l] - a * (1. - a) * e[k] * e[l];
                        assert!(close(want, curvature, 1e-5), "B = {band}, η = {ratios:?}, curvature {k}{l}: {want} against {curvature}");
                    }
                }
            }
        }
    }
}
