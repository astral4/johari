//! The posterior approximation q = N(μ, Λ⁻¹).

use crate::gaussian::Normal;
use crate::linalg::{Cholesky, Matrix};
use crate::model::{Careless, Extra, N_EXTRA};
use crate::special::exp;
use crate::twins::Twins;
use alloc::vec::Vec;

mod anderson;
mod ep;

pub(crate) use ep::fit;
#[cfg(feature = "bench")]
pub(crate) use ep::{Bench, fit_diagnostics};

/// The posterior approximation q = N(μ, Λ⁻¹) of an observation log.
pub(crate) struct Posterior {
    /// μ.
    pub(crate) mu: Vec<f64>,
    /// Λ's sparse Cholesky factor.
    pub(crate) chol_precision: Cholesky,
    /// Λ⁻¹.
    pub(crate) covariance: Matrix,
    /// θ's variances; i.e. Λ⁻¹'s diagonal over the items.
    pub(crate) var: Vec<f64>,
    /// The classes of twins of the log.
    pub(crate) twins: Twins,
    /// The fit's point estimate of the careless mixture.
    pub(crate) careless: Careless,
}

impl Posterior {
    /// q from the fit's μ and careless mixture, Λ's Cholesky factor, and the log's `twins`.
    fn new(mu: Vec<f64>, chol_precision: Cholesky, twins: Twins, careless: Careless) -> Self {
        debug_assert!(mu.len() >= N_EXTRA, "φ holds the shared parameters");
        debug_assert_eq!(twins.class().len(), mu.len() - N_EXTRA, "a class for each item");
        let covariance = chol_precision.inverse();
        let var = (0..mu.len() - N_EXTRA).map(|i| covariance.get(i, i)).collect();
        Self { mu, chol_precision, covariance, var, twins, careless }
    }

    /// The length of φ.
    pub(crate) fn dim(&self) -> usize {
        self.mu.len()
    }

    /// The number of items.
    pub(crate) fn n_items(&self) -> usize {
        self.mu.len() - N_EXTRA
    }

    /// The mean of θ.
    pub(crate) fn mean_theta(&self) -> &[f64] {
        &self.mu[..self.n_items()]
    }

    /// The median of δ.
    pub(crate) fn median_delta(&self) -> f64 {
        exp(self.mu[Extra::LogDelta.index(self.n_items())])
    }

    /// The gap θᵢ - θⱼ under q.
    pub(crate) fn gap(&self, i: usize, j: usize) -> Normal {
        let mu = self.mean_theta();
        let var = &self.var[..];
        difference([mu[i], mu[j]], [var[i], var[j]], self.covariance.get(i, j))
    }

    /// The gap g = θᵢ - θⱼ under q to each later item j in turn.
    pub(crate) fn gaps_after(&self, i: usize) -> impl Iterator<Item = Normal> + '_ {
        let mu = self.mean_theta();
        let var = &self.var[..];
        let n = self.n_items();
        let row = &self.covariance.row(i)[i + 1..n];
        zip!(&mu[i + 1..], &var[i + 1..], row).map(move |(&mu_j, &var_j, &cov_ij)| difference([mu[i], mu_j], [var[i], var_j], cov_ij))
    }
}

/// θᵢ - θⱼ from their means `mu`, variances `var` and covariance `cov`.
fn difference(mu: [f64; 2], var: [f64; 2], cov: f64) -> Normal {
    Normal { mean: mu[0] - mu[1], var: var[0] + var[1] - 2. * cov }
}
