//! Testing utilities.

use crate::gaussian::Sym2;
use crate::observations::{Bucket, Observations, Pair, PairObs};
use crate::ranking::TieCost;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{FRAC_1_SQRT_2, FRAC_1_SQRT_2PI};
use core::iter::zip;
use libm::{erfc, exp};

pub(crate) const TIE_COST: TieCost = TieCost::new(0.8).unwrap();

/// φ(x).
pub(crate) fn normal_density(x: f64) -> f64 {
    FRAC_1_SQRT_2PI * exp(-0.5 * x * x)
}

/// Φ(x).
pub(crate) fn ndtr(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// PRNG for testing.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniformly sample an integer in [0, `n`).
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 33) % n
    }

    /// Uniformly sample a float in [0, 1).
    #[expect(clippy::cast_precision_loss, reason = "2⁵³ and every integer below it are exact in f64")]
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniformly sample a float in [-`scale`, `scale`).
    pub(crate) fn sym(&mut self, scale: f64) -> f64 {
        (2. * self.unit() - 1.) * scale
    }

    /// Uniformly sample an element of `items`.
    #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
    pub(crate) fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }

    /// Uniformly shuffle `items`.
    #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for k in (1..items.len()).rev() {
            items.swap(k, self.below(k as u64 + 1) as usize);
        }
    }

    /// Uniformly sample an ordered pair of distinct items of `n`.
    #[expect(clippy::cast_possible_truncation, reason = "test sizes are tiny")]
    fn pair(&mut self, n: usize) -> (u32, u32) {
        let a = self.below(n as u64) as u32;
        (a, (a + 1 + self.below(n as u64 - 1) as u32) % n as u32)
    }
}

/// A random 2 × 2 symmetric positive-definite matrix.
pub(crate) fn random_spd(rng: &mut Lcg) -> Sym2 {
    let (u, v, w) = (rng.sym(2.), rng.sym(2.), rng.sym(2.));
    Sym2 { xx: u * u + 0.05, xy: u * v, yy: v * v + w * w + 0.05 }
}

/// A random log over `n` items with `answers` answered pairs.
pub(crate) fn random_log(rng: &mut Lcg, n: usize, answers: usize) -> Observations {
    let theta: Vec<_> = (0..n).map(|_| rng.sym(4.)).collect();
    let pairs = (0..answers)
        .map(|_| {
            let (a, b) = rng.pair(n);
            answer(a, b, theta[a as usize] - theta[b as usize] + rng.sym(1.))
        })
        .collect();
    let buckets = theta.iter().copied().map(bucket_of).collect();
    Observations { buckets, pairs }
}

/// The answer to the pair (a, b) whose utility gap is `gap`, including noise.
fn answer(a: u32, b: u32, gap: f64) -> PairObs {
    if gap > 0.6 {
        PairObs::Pick { above: a, below: b }
    } else if gap < -0.6 {
        PairObs::Pick { above: b, below: a }
    } else {
        PairObs::Same(Pair::new(a, b))
    }
}

/// The bucket for a utility `t`.
fn bucket_of(t: f64) -> Bucket {
    Bucket::ALL[usize::from(t >= -1.5) + usize::from(t >= 1.5)]
}

/// A log over ten items with various kinds of twins.
pub(crate) fn twin_log() -> Observations {
    use Bucket::{Fine, Meh};
    let buckets = vec![Fine, Fine, Fine, Fine, Fine, Fine, Fine, Meh, Fine, Fine];
    let pick = |above, below| PairObs::Pick { above, below };
    let pairs = vec![pick(0, 1), pick(0, 2), pick(0, 3), pick(0, 4), PairObs::Same(Pair::new(3, 4)), pick(5, 6), pick(0, 7)];
    Observations { buckets, pairs }
}

/// Whether a symmetric 2 × 2 matrix is positive semidefinite within rounding.
pub(crate) fn is_psd(m: Sym2) -> bool {
    let Sym2 { xx: a, yy: c, .. } = m;
    a >= -1e-12 && c >= -1e-12 && m.det() >= -1e-12 * (1. + a * a + c * c)
}

/// Whether `got` is within `rel` of `want` relative to 1 + |`want`|.
pub(crate) fn close(got: f64, want: f64, rel: f64) -> bool {
    (got - want).abs() <= rel * (1. + want.abs())
}

/// The entries [xx, xy, yy] of `s`.
pub(crate) const fn entries(s: Sym2) -> [f64; 3] {
    [s.xx, s.xy, s.yy]
}

/// Whether all corresponding entries of `a` and `b` are within `tol` of each other.
pub(crate) fn entries_within(a: Sym2, b: Sym2, tol: f64) -> bool {
    zip(entries(a), entries(b)).all(|(a, b)| (a - b).abs() <= tol)
}
