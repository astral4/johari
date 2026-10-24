//! Testing utilities.

use crate::gaussian::Sym2;
use core::f64::consts::{FRAC_1_SQRT_2, FRAC_1_SQRT_2PI};
use core::iter::zip;
use libm::{erfc, exp};

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
}

/// A random 2 × 2 symmetric positive-definite matrix.
pub(crate) fn random_spd(rng: &mut Lcg) -> Sym2 {
    let (u, v, w) = (rng.sym(2.), rng.sym(2.), rng.sym(2.));
    Sym2 { xx: u * u + 0.05, xy: u * v, yy: v * v + w * w + 0.05 }
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
