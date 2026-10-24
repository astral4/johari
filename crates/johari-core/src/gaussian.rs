//! 1D and 2D Gaussians.

use core::f64::math::sqrt;
use core::ops::{Add, Mul, Sub};

/// A univariate Gaussian defined by its moments.
#[derive(Clone, Copy)]
pub(crate) struct Normal {
    pub(crate) mean: f64,
    pub(crate) var: f64,
}

impl Normal {
    /// The standard deviation.
    pub(crate) fn sd(self) -> f64 {
        sqrt(self.var)
    }

    /// The natural parameters.
    pub(crate) const fn natural(self) -> Natural {
        Natural { precision: 1. / self.var, shift: self.mean / self.var }
    }
}

/// A univariate Gaussian defined by its natural parameters.
#[derive(Clone, Copy)]
pub(crate) struct Natural {
    pub(crate) precision: f64,
    pub(crate) shift: f64,
}

impl Natural {
    /// Whether this is a proper Gaussian.
    pub(crate) const fn is_proper(self) -> bool {
        self.precision.is_finite() && self.precision > 0.
    }

    /// The moments.
    pub(crate) fn normal(self) -> Normal {
        Normal { mean: self.shift / self.precision, var: 1. / self.precision }
    }
}

impl Sub for Natural {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self { precision: self.precision - other.precision, shift: self.shift - other.shift }
    }
}

/// A symmetric 2 × 2 matrix representing a bivariate Gaussian's precision or covariance.
#[derive(Clone, Copy, Default)]
pub(crate) struct Sym2 {
    pub(crate) xx: f64,
    pub(crate) xy: f64,
    pub(crate) yy: f64,
}

impl Sym2 {
    pub(crate) const IDENTITY: Self = Self { xx: 1., xy: 0., yy: 1. };

    /// The determinant.
    pub(crate) const fn det(self) -> f64 {
        self.xx * self.yy - self.xy * self.xy
    }

    /// The inverse. Returns `None` if the matrix is not positive definite.
    pub(crate) fn inverse(self) -> Option<Self> {
        let Self { xx, xy, yy } = self;
        let det = self.det();
        (xx > 0. && det > 0.).then(|| Self { xx: yy / det, xy: -xy / det, yy: xx / det })
    }

    /// The lower Cholesky factor of a positive definite matrix. Returns `None` unless both pivots are finite and positive.
    pub(crate) fn cholesky(self) -> Option<Lower2> {
        let l00 = sqrt(self.xx);
        let l10 = self.xy / l00;
        let l11 = sqrt(self.yy - l10 * l10);
        (l00 > 0. && l00 < f64::INFINITY && l11 > 0. && l11 < f64::INFINITY).then_some(Lower2 { l00, l10, l11 })
    }

    /// The lower Cholesky factor of a positive semidefinite matrix.
    pub(crate) fn semidefinite_cholesky(self) -> Lower2 {
        let l00 = sqrt(self.xx.max(0.));
        let l10 = if l00 > 0. { self.xy / l00 } else { 0. };
        let l11 = sqrt((self.yy - l10 * l10).max(0.));
        Lower2 { l00, l10, l11 }
    }

    /// The positive part; i.e. the matrix with its negative eigenvalues set to zero.
    pub(crate) fn positive_part(self) -> Self {
        let Self { xx, xy, yy } = self;
        if self.det() >= 0. {
            return if xx + yy >= 0. { self } else { Self::default() };
        }
        // The larger eigenvalue.
        let top = xx.midpoint(yy) + sqrt(0.25 * (xx - yy) * (xx - yy) + xy * xy);
        let [x, y] = if xx >= yy { [top - yy, xy] } else { [xy, top - xx] };
        Self::outer(top.max(0.) / (x * x + y * y), [x, y])
    }

    /// s·v·vᵀ for the vector v = `[x, y]`.
    pub(crate) const fn outer(s: f64, [x, y]: [f64; 2]) -> Self {
        Self { xx: s * x * x, xy: s * x * y, yy: s * y * y }
    }
}

impl Add for Sym2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { xx: self.xx + other.xx, xy: self.xy + other.xy, yy: self.yy + other.yy }
    }
}

impl Sub for Sym2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self { xx: self.xx - other.xx, xy: self.xy - other.xy, yy: self.yy - other.yy }
    }
}

impl Mul<[f64; 2]> for Sym2 {
    type Output = [f64; 2];

    fn mul(self, [x, y]: [f64; 2]) -> [f64; 2] {
        [self.xx * x + self.xy * y, self.xy * x + self.yy * y]
    }
}

/// A lower triangular 2 × 2 matrix L.
#[derive(Clone, Copy)]
pub(crate) struct Lower2 {
    l00: f64,
    l10: f64,
    l11: f64,
}

impl Lower2 {
    /// Lᵀ·M·L.
    pub(crate) fn on_columns(self, m: Sym2) -> Sym2 {
        let Self { l00, l10, l11 } = self;
        let c0 = m * [l00, l10];
        Sym2 { xx: l00 * c0[0] + l10 * c0[1], xy: l11 * c0[1], yy: l11 * (m.yy * l11) }
    }

    /// L·M·Lᵀ.
    pub(crate) fn on_rows(self, m: Sym2) -> Sym2 {
        let Self { l00, l10, l11 } = self;
        let r1 = m * [l10, l11];
        Sym2 { xx: l00 * (m.xx * l00), xy: l00 * r1[0], yy: l10 * r1[0] + l11 * r1[1] }
    }

    /// Lᵀ·v.
    pub(crate) fn transpose_mul(self, [x, y]: [f64; 2]) -> [f64; 2] {
        [self.l00 * x + self.l10 * y, self.l11 * y]
    }
}

/// A bivariate Gaussian over (x, y) defined by its moments.
#[derive(Clone, Copy)]
pub(crate) struct Normal2 {
    pub(crate) mean: [f64; 2],
    pub(crate) cov: Sym2,
}

impl Normal2 {
    /// x's marginal.
    pub(crate) fn x(self) -> Normal {
        Normal { mean: self.mean[0], var: self.cov.xx }
    }

    /// The natural parameters. Returns `None` if the covariance is not positive definite.
    pub(crate) fn natural(self) -> Option<Natural2> {
        let precision = self.cov.inverse()?;
        Some(Natural2 { precision, shift: precision * self.mean })
    }
}

/// A bivariate Gaussian over (x, y) defined by its natural parameters.
#[derive(Clone, Copy)]
pub(crate) struct Natural2 {
    pub(crate) precision: Sym2,
    pub(crate) shift: [f64; 2],
}

impl Natural2 {
    /// Converts a Gaussian over x into a Gaussian over (x, y) that leaves y free.
    pub(crate) fn of_x(x: Natural) -> Self {
        Self { precision: Sym2 { xx: x.precision, xy: 0., yy: 0. }, shift: [x.shift, 0.] }
    }

    /// The parameters Λₓₓ and hₓ.
    pub(crate) fn x_part(self) -> Natural {
        Natural { precision: self.precision.xx, shift: self.shift[0] }
    }

    /// Whether every parameter is finite.
    pub(crate) const fn is_finite(self) -> bool {
        let Self { precision: Sym2 { xx, xy, yy }, shift: [hx, hy] } = self;
        xx.is_finite() && xy.is_finite() && yy.is_finite() && hx.is_finite() && hy.is_finite()
    }

    /// The moments. Returns `None` if the precision is not positive definite.
    pub(crate) fn normal(self) -> Option<Normal2> {
        let cov = self.precision.inverse()?;
        Some(Normal2 { mean: cov * self.shift, cov })
    }
}

impl Sub for Natural2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self { precision: self.precision - other.precision, shift: [self.shift[0] - other.shift[0], self.shift[1] - other.shift[1]] }
    }
}

/// A bivariate Gaussian over (x, y) defined by y's marginal and x's regression on y.
/// y ~ N(`mean[1]`, `y_sd`²) and, given y, x ~ N(`mean[0]` + `slope`·(y - `mean[1]`), `spread`).
#[derive(Clone, Copy)]
pub(crate) struct Regression {
    pub(crate) mean: [f64; 2],
    pub(crate) y_sd: f64,
    /// Cov(x, y) / Var y.
    pub(crate) slope: f64,
    /// Var(x | y) = Var x - Cov(x, y)² / Var y.
    pub(crate) spread: f64,
}

impl Regression {
    pub(crate) fn new(normal: Normal2) -> Self {
        let Normal2 { mean, cov } = normal;
        let slope = cov.xy / cov.yy;
        Self { mean, y_sd: sqrt(cov.yy), slope, spread: cov.xx - slope * cov.xy }
    }

    /// The mean and covariance.
    pub(crate) fn normal(self) -> Normal2 {
        let Self { mean, y_sd, slope, spread } = self;
        let vy = y_sd * y_sd;
        Normal2 { mean, cov: Sym2 { xx: spread + slope * slope * vy, xy: slope * vy, yy: vy } }
    }

    /// Converts a Gaussian over x into a Gaussian over (x, y) that leaves y free.
    pub(crate) fn of_x(x: Normal) -> Self {
        Self { mean: [x.mean, 0.], y_sd: 0., slope: 0., spread: x.var }
    }
}
