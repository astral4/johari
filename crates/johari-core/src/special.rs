//! Special functions.

use crate::lanes::{from_memory, max_lanes, min_lanes};
use core::f32::consts::{FRAC_1_SQRT_2 as FRAC_1_SQRT_2_F, FRAC_1_SQRT_2PI as FRAC_1_SQRT_2PI_F, LOG2_E as LOG2_E_F};
use core::f64::consts::{FRAC_1_SQRT_2, FRAC_1_SQRT_2PI, LOG2_E};
use core::ops::{Add, Div, Mul, Neg, Sub};
use core::simd::cmp::SimdPartialOrd;
use core::simd::num::{SimdFloat, SimdInt as _, SimdUint as _};
use core::simd::{Select, Simd, SimdElement, f32x4, f64x2, i32x4, i64x2};

fn cubic<V: Lane>(c: &[V], x: V, x2: V) -> V {
    (c[3] * x + c[2]) * x2 + (c[1] * x + c[0])
}

/// erfcx(x)/t for t = 1/(1 + x/4) as a polynomial R in 2t - 1, constant term first. R(1) = erfcx(0) = 1.
const ERFCX_POLY: [f64; 7] = [
    0.274_558_097_078_154_4,
    0.237_749_891_729_283_85,
    0.223_690_813_644_200_24,
    0.056_980_207_499_385_12,
    0.200_426_905_162_369_88,
    -0.059_411_867_229_450_19,
    0.066_005_952_116_056_74,
];

/// eʳ ≈ 1 + r + r²·P(r) for |r| ≤ 0.3466 (½·ln 2 rounded up), constant term first.
const EXP_POLY: [f64; 5] =
    [0.499_999_934_487_195_3, 0.166_665_206_455_532_18, 0.041_668_387_884_155_174, 0.008_368_715_199_617_204, 0.001_381_460_204_213_447];

/// A vector of floats for a lane kernel.
pub(crate) trait Lane:
    'static
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + SimdFloat
    + SimdPartialOrd<Mask: Select<Self>>
{
    const FLUSHES: bool;

    /// `x` rounded to the lanes' precision.
    fn splat_f64(x: f64) -> Self;

    /// 2ᵏ, for the integer k held in the low bits of `shifted`. `shifted` is the sum of k and `shift`.
    fn pow2(shifted: Self, shift: Self) -> Self;

    /// The kernel table.
    fn table() -> &'static KernelTable<Self>;
}

impl Lane for f32x4 {
    const FLUSHES: bool = true;

    #[expect(clippy::cast_possible_truncation, reason = "the lanes are f32")]
    fn splat_f64(x: f64) -> Self {
        Self::splat(x as f32)
    }

    fn pow2(shifted: Self, shift: Self) -> Self {
        let k = shifted.to_bits().cast::<i32>() - shift.to_bits().cast();
        Self::from_bits(((k + i32x4::splat(127)) << 23).cast())
    }

    fn table() -> &'static KernelTable<Self> {
        from_memory(&KERNELS_F32)
    }
}

impl Lane for f64x2 {
    const FLUSHES: bool = false;

    fn splat_f64(x: f64) -> Self {
        Self::splat(x)
    }

    fn pow2(shifted: Self, shift: Self) -> Self {
        let k = shifted.to_bits().cast::<i64>() - shift.to_bits().cast();
        Self::from_bits(((k + i64x2::splat(1023)) << 52).cast())
    }

    fn table() -> &'static KernelTable<Self> {
        from_memory(&KERNELS_F64)
    }
}

/// A table of lane kernels.
pub(crate) struct KernelTable<V> {
    zero: V,
    half: V,
    one: V,
    /// 1/sqrt(2π).
    density: V,
    /// The smallest x such that eˣ is normal. Exponents below this threshold are clamped.
    floor: V,
    log2e: V,
    /// 1.5 * 2^(mantissa width). Rounds x·log₂e to an integer.
    shift: V,
    /// ln 2 in two parts.
    ln2: [V; 2],
    /// P; i.e. [`EXP_POLY`] in the table's precision.
    exp_poly: [V; 5],
    /// 1/root(4, 2).
    t_scale: V,
    /// R; i.e. [`ERFCX_POLY`] in the table's precision.
    erfcx_poly: [V; 7],
}

/// The kernels in f64.
static KERNELS_F64: &KernelTable<f64x2> = &KernelTable {
    zero: 0.,
    half: 0.5,
    one: 1.,
    density: FRAC_1_SQRT_2PI,
    floor: -708.,
    log2e: LOG2_E,
    shift: 6_755_399_441_055_744.,
    ln2: [0.693_147_180_369_123_8, 1.908_214_929_270_587_7e-10],
    exp_poly: EXP_POLY,
    t_scale: 0.25 * FRAC_1_SQRT_2,
    erfcx_poly: ERFCX_POLY,
}
.splat();

/// The kernels in f32.
static KERNELS_F32: &KernelTable<f32x4> = &KernelTable {
    zero: 0.,
    half: 0.5,
    one: 1.,
    density: FRAC_1_SQRT_2PI_F,
    floor: -40.5,
    log2e: LOG2_E_F,
    shift: 12_582_912.,
    ln2: [0.693_359_4, -2.121_944_4e-4],
    exp_poly: to_f32(EXP_POLY),
    t_scale: 0.25 * FRAC_1_SQRT_2_F,
    erfcx_poly: to_f32(ERFCX_POLY),
}
.splat();

impl<T: SimdElement> KernelTable<T> {
    /// Each constant in every lane.
    const fn splat<const L: usize>(self) -> KernelTable<Simd<T, L>> {
        KernelTable {
            zero: Simd::splat(self.zero),
            half: Simd::splat(self.half),
            one: Simd::splat(self.one),
            density: Simd::splat(self.density),
            floor: Simd::splat(self.floor),
            log2e: Simd::splat(self.log2e),
            shift: Simd::splat(self.shift),
            ln2: splat_each(self.ln2),
            exp_poly: splat_each(self.exp_poly),
            t_scale: Simd::splat(self.t_scale),
            erfcx_poly: splat_each(self.erfcx_poly),
        }
    }
}

/// A vector of x in every lane for each x in `xs`.
const fn splat_each<T: SimdElement, const L: usize, const N: usize>(xs: [T; N]) -> [Simd<T, L>; N] {
    let mut out = [Simd::splat(xs[0]); N];
    let mut k = 1;
    while k < N {
        out[k] = Simd::splat(xs[k]);
        k += 1;
    }
    out
}

/// Φ(x) and φ(x) per lane.
pub(crate) fn ndtr_lanes<V: Lane>(x: V) -> [V; 2] {
    // Let e = e^(-x²/2), t = 1/(1 + |x|/root(4, 2)), and R = `ERFCX_POLY`. Then, the smaller tail is:
    //
    // Φ(-|x|) = ½·erfc(|x|/sqrt(2))
    //         = ½·erfcx(|x|/sqrt(2))·e
    //         = ½·t·R(2t - 1)·e
    let c = V::table();
    let exponent = -(c.half * x * x);
    let gauss = exp_short(exponent);
    let gauss = if V::FLUSHES { c.floor.simd_gt(exponent).select(V::splat_f64(0.), gauss) } else { gauss };
    let t = c.one / (c.t_scale * x.abs() + c.one);
    let y = t + t - c.one;
    let y2 = y * y;
    let poly = &c.erfcx_poly;
    let erfcx = (poly[6] * y2 + (poly[5] * y + poly[4])) * y2 * y2 + cubic(poly, y, y2);
    let tail = c.half * t * erfcx * gauss;
    [x.simd_lt(c.zero).select(tail, c.one - tail), c.density * gauss]
}

/// Writes Φ(x) into `xs`, per lane. Also writes φ(x) into `densities` if `densities` is `Some(_)`.
#[inline(never)]
pub(crate) fn ndtr_in_place(xs: &mut [[f32; f32x4::LEN]], mut densities: Option<&mut [[f32; f32x4::LEN]]>) {
    for (k, x) in xs.iter_mut().enumerate() {
        let [cdf, pdf] = ndtr_lanes(f32x4::from_array(*x));
        *x = cdf.to_array();
        if let Some(densities) = densities.as_deref_mut() {
            densities[k] = pdf.to_array();
        }
    }
}

/// eˣ per lane.
fn exp_short<V: Lane>(x: V) -> V {
    let c = V::table();
    let x = max_lanes(c.floor, x);
    let shifted = x * c.log2e + c.shift;
    let k = shifted - c.shift;
    let r = x - k * c.ln2[0] - k * c.ln2[1];
    let (series, r2) = (&c.exp_poly, r * r);
    let poly = series[4] * (r2 * r2) + cubic(series, r, r2);
    (poly * r2 + r + c.one) * V::pow2(shifted, c.shift)
}

/// eˣ per lane.
pub(crate) fn exp_lanes(x: f64x2) -> f64x2 {
    // 2ᵏ overflows to ∞ at k = 1024, and past it wraps into the sign: k stops at 1024.
    exp_short(min_lanes(f64x2::splat(710.), x))
}

/// eˣ.
#[inline(never)]
pub(crate) fn exp(x: f64) -> f64 {
    exp_lanes(f64x2::splat(x))[0]
}

/// The constants `c` rounded to f32.
#[expect(clippy::cast_possible_truncation)]
pub(crate) const fn to_f32<const N: usize>(c: [f64; N]) -> [f32; N] {
    let mut out = [0.; N];
    let mut k = 0;
    while k < N {
        out[k] = c[k] as f32;
        k += 1;
    }
    out
}

/// The integer whose order is [`f64::total_cmp`]'s.
const fn total_order_key(x: f64) -> i64 {
    let bits = x.to_bits().cast_signed();
    bits ^ ((bits >> 63).cast_unsigned() >> 1).cast_signed()
}

/// The float whose [`total_order_key`] is `key`.
const fn from_total_order_key(key: i64) -> f64 {
    f64::from_bits((key ^ ((key >> 63).cast_unsigned() >> 1).cast_signed()).cast_unsigned())
}

/// Φ⁻¹(p).
pub(crate) fn ndtri(p: f64) -> f64 {
    if p <= 0. {
        return f64::NEG_INFINITY;
    }
    if p >= 1. {
        return f64::INFINITY;
    }
    let cdf = |x| ndtr_lanes(f64x2::splat(x))[0][0];
    let (mut lo, mut hi) = (total_order_key(-40.), total_order_key(40.));
    while lo.abs_diff(hi) > 1 {
        let mid = lo.midpoint(hi);
        if cdf(from_total_order_key(mid)) < p { lo = mid } else { hi = mid }
    }
    from_total_order_key(hi)
}

#[cfg(test)]
mod tests {
    use super::{exp_lanes, from_total_order_key, ndtr_lanes, ndtri, total_order_key};
    use crate::testing::{Lcg, ndtr, normal_density};
    use core::array::from_fn as array_from_fn;
    use core::iter::{chain, zip};
    use core::simd::{f32x4, f64x2};
    use libm::exp;

    #[test]
    fn ndtri_is_where_phi_crosses_p() {
        let cdf = |x| ndtr_lanes(f64x2::splat(x))[0][0];

        let mut rng = Lcg(2);
        let ends = [f64::MIN_POSITIVE, 1e-300, 1e-12, 0.2, 0.5, 0.8, 1. - 1e-12, 1. - f64::EPSILON / 2.];
        for p in chain(ends, (0..2000).map(|_| 0.5 + rng.sym(0.5))).filter(|&p| p > 0.) {
            let x = ndtri(p);
            assert!(cdf(x) >= p && cdf(x.next_down()) < p, "p {p}: x {x}");
            assert!((ndtr(x) - p).abs() <= 1.1e-8, "p {p}: Φ({x}) = {}", ndtr(x));
        }
        for p in [0., -0., -1., f64::NEG_INFINITY] {
            assert_eq!(ndtri(p), f64::NEG_INFINITY, "p {p}");
        }
        for p in [1., 2., f64::INFINITY] {
            assert_eq!(ndtri(p), f64::INFINITY, "p {p}");
        }
    }

    #[test]
    fn ndtr_f64_matches_libm() {
        let mut rng = Lcg(37);
        for k in 0..400_000 {
            let reach = if k % 4 == 0 { 40. } else { 9. };
            let w = f64x2::from_array([rng.sym(reach), rng.sym(reach)]);
            let ([cdf, pdf], [mirror, _]) = (ndtr_lanes(w), ndtr_lanes(-w));
            for lane in 0..2 {
                let x = w[lane];
                let density = normal_density(x);
                assert!((cdf[lane] - ndtr(x)).abs() <= 1.1e-8, "Φ({x}) = {} against {}", cdf[lane], ndtr(x));
                assert!((pdf[lane] - density).abs() <= 1.1e-9, "φ({x}) = {} against {density}", pdf[lane]);
                assert!((cdf[lane] + mirror[lane] - 1.).abs() <= f64::EPSILON, "Φ({x}) + Φ(-{x}) = 1");
            }
        }
        for w in [-40., f64::NEG_INFINITY, 40., f64::INFINITY] {
            let [cdf, pdf] = ndtr_lanes(f64x2::splat(w)).map(|v| v[0]);
            assert!((cdf - if w < 0. { 0. } else { 1. }).abs() <= 1e-300, "Φ({w}) = {cdf}");
            assert!(pdf <= 1e-300, "φ = {pdf}");
        }
        for x in [-f64::MIN_POSITIVE, 0., f64::MIN_POSITIVE] {
            assert_eq!(ndtr_lanes(f64x2::splat(x))[0][0].to_bits(), 0.5f64.to_bits(), "Φ({x}) = ½");
        }
        let [cdf, pdf] = ndtr_lanes(f64x2::splat(f64::NAN));
        assert!(cdf[0].is_nan() && pdf[0].is_nan(), "NaN at NaN");
    }

    #[test]
    #[expect(clippy::cast_possible_truncation, reason = "the f32 kernel runs in f32")]
    fn ndtr_f32_matches_libm() {
        let mut rng = Lcg(29);
        for _ in 0..50_000 {
            let x = array_from_fn(|_| rng.sym(14.) as f32);
            let got = ndtr_lanes(f32x4::from_array(x))[0].to_array();
            for (got, x) in zip(got, x) {
                let want = ndtr(f64::from(x));
                assert!((f64::from(got) - want).abs() <= 3.3e-7, "Φ({x}) = {got} against {want}");
            }
        }
        for x in [-f32::MIN_POSITIVE, 0., f32::MIN_POSITIVE] {
            assert!(ndtr_lanes(f32x4::splat(x))[0][0].to_bits() == 0.5f32.to_bits(), "Φ({x}) = ½");
        }
    }

    #[test]
    #[expect(clippy::cast_possible_truncation, reason = "the f32 kernel runs in f32")]
    fn ndtr_f32_is_zero_or_normal() {
        let check = |x| {
            let [cdf, pdf] = ndtr_lanes(f32x4::splat(x)).map(|v| v[0]);
            let smaller = ndtr_lanes(f32x4::splat(-x.abs()))[0][0];
            let past = x.abs() > 9.;
            let normal = !past && pdf >= 1e-18 && smaller >= 1e-19;
            let end: f32 = if x < 0. { 0. } else { 1. };
            let flushed = past && pdf.to_bits() == 0 && smaller.to_bits() == 0 && cdf.to_bits() == end.to_bits();
            assert!(normal || flushed, "x {x}: Φ {cdf}, smaller tail {smaller}, φ {pdf}");
        };

        for x in (8.99f32.to_bits()..=9.01f32.to_bits()).map(f32::from_bits) {
            check(x);
            check(-x);
        }

        let mut rng = Lcg(31);
        for _ in 0..50_000 {
            check(exp(rng.sym(88.)) as f32 * if rng.unit() < 0.5 { -1. } else { 1. });
        }
        let [cdf, pdf] = ndtr_lanes(f32x4::from_array([f32::NEG_INFINITY, f32::INFINITY, f32::NAN, 0.]));
        let ends = [cdf[0], cdf[1], pdf[0], pdf[1]].map(f32::to_bits);
        assert!(ends == [0., 1., 0., 0.].map(f32::to_bits), "0 and 1 at the infinities");
        assert!(cdf[2].is_nan() && pdf[2].is_nan(), "NaN at NaN");
    }

    #[test]
    fn exp_f64_matches_libm() {
        let mut rng = Lcg(23);
        for _ in 0..20_000 {
            let x = f64x2::from_array([rng.sym(708.), 700. + rng.sym(9.)]);
            let got = exp_lanes(x);
            for lane in 0..2 {
                let (got, want) = (got[lane], exp(x[lane]));
                assert!((got - want).abs() <= 3.1e-9 * want, "e^{} = {got} against {want}", x[lane]);
            }
        }
        let [near, far] = exp_lanes(f64x2::from_array([709.43, 709.44])).to_array();
        assert!(near.is_finite() && far == f64::INFINITY && exp_lanes(f64x2::splat(1e300))[0] == f64::INFINITY, "∞ from 709.44");
        assert!(exp_lanes(f64x2::splat(f64::NAN))[0].is_nan(), "NaN at NaN");
    }

    #[test]
    fn total_order_key_orders_as_total_cmp() {
        const EPS: f64 = f64::from_bits(1);

        let xs =
            [f64::NEG_INFINITY, -1., -f64::MIN_POSITIVE, -EPS, -0., 0., EPS, f64::MIN_POSITIVE, 1., f64::INFINITY, f64::NAN, -f64::NAN];
        for a in xs {
            assert_eq!(from_total_order_key(total_order_key(a)).to_bits(), a.to_bits(), "the transform is its own inverse");
            for b in xs {
                assert_eq!(total_order_key(a).cmp(&total_order_key(b)), a.total_cmp(&b), "the key orders as total_cmp");
            }
        }
    }
}
