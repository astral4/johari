"""Log posterior, damped-Newton MAP, and Laplace approximation.

The parameter vector is φ = (θ₁ … θₙ, logit ε, c₁, log(c₂ - c₁), log δ), with every coordinate unconstrained.
The approximation is q = N(μ, H⁻¹), where μ is the MAP and H is the exact Hessian of -log p(φ | data) at μ.
"""

import enum
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy.linalg

from johari.config import (
    EXTRA_C1,
    EXTRA_EPS_LOGIT,
    EXTRA_LOG_DELTA,
    EXTRA_LOG_GAP,
    N_EXTRA,
    ModelConfig,
)
from johari.likelihood import bucket_logprob, pair_outcome_logprobs
from johari.observations import ObsData, Observation, pack_observations

_OBS_BUFFER = 512
"""Pair-buffer size for :func:`pack_observations`."""


class Params[A: (jax.Array, npt.NDArray[np.float64])](NamedTuple):
    """Named views over the last axis of a parameter vector."""

    theta: A
    eps_logit: A
    c1: A
    log_gap: A
    log_delta: A


def unpack[A: (jax.Array, npt.NDArray[np.float64])](phi: A) -> Params[A]:
    """Split a parameter vector or a matrix of them into named views."""
    n = phi.shape[-1] - N_EXTRA
    return Params(
        theta=phi[..., :n],
        eps_logit=phi[..., n + EXTRA_EPS_LOGIT],
        c1=phi[..., n + EXTRA_C1],
        log_gap=phi[..., n + EXTRA_LOG_GAP],
        log_delta=phi[..., n + EXTRA_LOG_DELTA],
    )


def init_phi(cfg: ModelConfig, n_items: int) -> npt.NDArray[np.float64]:
    """Return the prior mode (the MAP with no observations)."""
    phi = np.zeros(cfg.dim(n_items))
    phi[n_items + EXTRA_EPS_LOGIT] = np.log(cfg.eps_a / cfg.eps_b)
    phi[n_items + EXTRA_C1] = cfg.c1_loc
    phi[n_items + EXTRA_LOG_GAP] = cfg.log_gap_loc
    phi[n_items + EXTRA_LOG_DELTA] = cfg.delta_loc
    return phi


def log_prior(phi: jax.Array, cfg: ModelConfig) -> jax.Array:
    """Log prior density of φ.

        log p(φ) = Σᵢ log N(θᵢ; 0, σ₀²)
                 + log N(c₁; c1_loc, c1_scale²) + log N(log(c₂ - c₁); log_gap_loc, log_gap_scale²)
                 + log N(log δ; delta_loc, delta_scale²)
                 + a·log ε + b·log(1 - ε) - log B(a, b)

    The last line is the Beta(a, b) density on ε times the Jacobian ε(1 - ε) of the logit map.
    """
    p = unpack(phi)
    norm = jax.scipy.stats.norm
    beta_prior = (
        cfg.eps_a * jax.nn.log_sigmoid(p.eps_logit)
        + cfg.eps_b * jax.nn.log_sigmoid(-p.eps_logit)
        - jax.scipy.special.betaln(cfg.eps_a, cfg.eps_b)
    )
    return (
        jnp.sum(norm.logpdf(p.theta, 0.0, cfg.sigma0))
        + beta_prior
        + norm.logpdf(p.c1, cfg.c1_loc, cfg.c1_scale)
        + norm.logpdf(p.log_gap, cfg.log_gap_loc, cfg.log_gap_scale)
        + norm.logpdf(p.log_delta, cfg.delta_loc, cfg.delta_scale)
    )


def log_likelihood(phi: jax.Array, data: ObsData, cfg: ModelConfig) -> jax.Array:
    """Log likelihood Σ_pairs log P_obs(y | θ, ε, δ) + Σ_marks log P_obs(b | θ, c₁, c₂) over the packed rows.

    Padded rows are valid likelihood inputs in their own right (see :func:`johari.observations.pack_observations`),
    so every row's log-probability and gradient are finite and the masks only select relevance.
    """
    p = unpack(phi)
    pair_logp = pair_outcome_logprobs(p.theta[data.pair_items], p.eps_logit, jnp.exp(p.log_delta), cfg)
    lp_pair = jnp.take_along_axis(pair_logp, data.pair_outcome[:, None], axis=-1)[:, 0]
    pair_ll = jnp.sum(jnp.where(data.pair_mask, lp_pair, 0.0))
    c2 = p.c1 + jnp.exp(p.log_gap)
    lp_bucket = bucket_logprob(p.theta[data.bucket_item], p.c1, c2, data.bucket_val, cfg)
    bucket_ll = jnp.sum(jnp.where(data.bucket_mask, lp_bucket, 0.0))
    return pair_ll + bucket_ll


def neg_log_posterior(phi: jax.Array, data: ObsData, cfg: ModelConfig) -> jax.Array:
    """-(log prior + log likelihood), which is -log p(φ | data) up to a constant."""
    return -(log_prior(phi, cfg) + log_likelihood(phi, data, cfg))


_ROW_EPS, _ROW_DELTA = 2, 3
"""Slots of ``eps_logit`` and ``log_delta`` in a pair row."""


def _pair_row_nll(row: jax.Array, outcome: jax.Array, live: jax.Array, cfg: ModelConfig) -> jax.Array:
    """Negative log likelihood of one pair row over ``[theta_a, theta_b, eps_logit, log_delta]``."""
    logp = pair_outcome_logprobs(row[:_ROW_EPS], row[_ROW_EPS], jnp.exp(row[_ROW_DELTA]), cfg)
    return -jnp.where(live, logp[outcome], 0.0)


def _pair_slots(data: ObsData, n_items: int) -> jax.Array:
    """Per-pair-row parameter indices in :func:`_pair_row_nll`'s order."""
    items = data.pair_items
    shared = [jnp.full_like(items[:, 0], n_items + k) for k in (EXTRA_EPS_LOGIT, EXTRA_LOG_DELTA)]
    return jnp.stack([items[:, 0], items[:, 1], *shared], axis=-1)


def _bucket_row_nll(row: jax.Array, bucket: jax.Array, live: jax.Array, cfg: ModelConfig) -> jax.Array:
    """Negative log likelihood of one bucket row over ``[theta_i, c1, log_gap]``."""
    lp = bucket_logprob(row[0], row[1], row[1] + jnp.exp(row[2]), bucket, cfg)
    return -jnp.where(live, lp, 0.0)


def _bucket_slots(data: ObsData, n_items: int) -> jax.Array:
    """Per-bucket-row parameter indices in :func:`_bucket_row_nll`'s order."""
    item = data.bucket_item
    shared = [jnp.full_like(item, n_items + k) for k in (EXTRA_C1, EXTRA_LOG_GAP)]
    return jnp.stack([item, *shared], axis=-1)


_RowNll = Callable[[jax.Array, jax.Array, jax.Array, ModelConfig], jax.Array]


def _add_row_blocks(
    hessian: jax.Array,
    phi: jax.Array,
    row_nll: _RowNll,
    slots: jax.Array,
    value: jax.Array,
    live: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:
    """Scatter-add every row's Hessian block onto the coordinates touched by that row."""
    blocks = jax.vmap(jax.hessian(row_nll), in_axes=(0, 0, 0, None))(phi[slots], value, live, cfg)
    return hessian.at[slots[:, :, None], slots[:, None, :]].add(blocks)


@functools.partial(jax.jit, static_argnames="cfg")
def hessian(phi: jax.Array, data: ObsData, cfg: ModelConfig) -> jax.Array:
    """Hessian of -log p(φ | data) at ``phi``, assembled from per-row blocks."""
    n = phi.shape[0] - N_EXTRA
    hessian = -jax.hessian(log_prior)(phi, cfg)
    hessian = _add_row_blocks(hessian, phi, _pair_row_nll, _pair_slots(data, n), data.pair_outcome, data.pair_mask, cfg)
    return _add_row_blocks(
        hessian,
        phi,
        _bucket_row_nll,
        _bucket_slots(data, n),
        data.bucket_val,
        data.bucket_mask,
        cfg,
    )


_ValueGradFn = Callable[[jax.Array, ObsData, ModelConfig], tuple[jax.Array, jax.Array]]

_value_and_grad: _ValueGradFn = jax.jit(jax.value_and_grad(neg_log_posterior), static_argnames="cfg")


@dataclass(frozen=True)
class LaplacePosterior:
    """MAP plus Laplace precision for one observation log."""

    mu: npt.NDArray[np.float64]
    chol_precision: npt.NDArray[np.float64]
    """Lower-triangular L with L·Lᵀ = H (jittered if needed), the precision of q."""
    cfg: ModelConfig
    converged: bool
    """Whether max|∇f| < tol at ``mu``, for the tolerance given to :func:`fit_laplace`."""

    @property
    def params(self) -> Params[npt.NDArray[np.float64]]:
        """The MAP as named views."""
        return unpack(self.mu)

    @property
    def theta(self) -> npt.NDArray[np.float64]:
        """Posterior-mode utilities."""
        return self.params.theta

    def covariance(self) -> npt.NDArray[np.float64]:
        """Posterior covariance H⁻¹ = L⁻ᵀ·L⁻¹, from the precision's Cholesky factor."""
        identity = np.eye(self.mu.shape[0])
        half = scipy.linalg.solve_triangular(self.chol_precision, identity, lower=True)
        return half.T @ half

    def sample(self, rng: np.random.Generator, count: int) -> npt.NDArray[np.float64]:
        """Draw ``count`` samples μ + L⁻ᵀ·z, z ~ N(0, I), from q = N(μ, H⁻¹)."""
        z = rng.standard_normal((count, self.mu.shape[0]))
        offsets = scipy.linalg.solve_triangular(self.chol_precision.T, z.T, lower=False).T
        return self.mu + offsets


def _evaluate(phi: npt.NDArray[np.float64], data: ObsData, cfg: ModelConfig) -> tuple[float, npt.NDArray[np.float64]]:
    value, grad = _value_and_grad(jnp.asarray(phi), data, cfg)
    return float(value), np.asarray(grad)


def _newton_step(hessian: npt.NDArray[np.float64], damping: float, grad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """Solve (H + λI)·step = -∇f for the damped Newton step. Return None if H + λI is not positive definite."""
    damped = hessian + damping * np.eye(hessian.shape[0])
    try:
        chol = np.linalg.cholesky(damped)
    except np.linalg.LinAlgError:
        return None
    return scipy.linalg.cho_solve((chol, True), -grad)


def _chol_with_jitter(hessian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the Hessian's Cholesky factor, adding escalating jitter if needed."""
    if not np.all(np.isfinite(hessian)):
        msg = "Hessian at the MAP is not finite"
        raise FloatingPointError(msg)
    jitters = [0.0, *(1e-10 * 10.0**i for i in range(13))]
    for jitter in jitters:
        try:
            return np.linalg.cholesky(hessian + jitter * np.eye(hessian.shape[0]))
        except np.linalg.LinAlgError:
            continue
    msg = f"Hessian at the MAP is not positive definite even with jitter {jitters[-1]:g}"
    raise np.linalg.LinAlgError(msg)


def _line_search(
    phi: npt.NDArray[np.float64],
    value: float,
    grad: npt.NDArray[np.float64],
    step: npt.NDArray[np.float64],
    data: ObsData,
    cfg: ModelConfig,
) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]] | None:
    """Backtracking line search (Armijo 1966) along ``step``. Return the accepted ``(phi, value, grad)``, or None.

    Accepts the first t ∈ {1, 2⁻¹, …, 2⁻¹¹} with a finite gradient and f(φ + t·step) ≤ f(φ) + 10⁻⁴·t·∇f·step.
    """
    slope = float(grad @ step)
    t = 1.0
    for _ in range(12):
        candidate = phi + t * step
        new_value, new_grad = _evaluate(candidate, data, cfg)
        finite = np.isfinite(new_value) and bool(np.all(np.isfinite(new_grad)))
        if finite and new_value <= value + 1e-4 * t * slope:
            return candidate, new_value, new_grad
        t *= 0.5
    return None


def fit_laplace(
    data: ObsData,
    cfg: ModelConfig,
    n_items: int,
    phi0: npt.NDArray[np.float64] | None = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> LaplacePosterior:
    """Fit the MAP by damped Newton and return the Laplace approximation.

    ``phi0`` warm-starts the optimization. If it is not provided, then the prior mode is used.
    """
    phi = init_phi(cfg, n_items) if phi0 is None else np.asarray(phi0, dtype=np.float64).copy()
    value, grad = _evaluate(phi, data, cfg)
    if not (np.isfinite(value) and np.all(np.isfinite(grad))):
        msg = "objective is not finite at the starting point phi0"
        raise FloatingPointError(msg)
    damping = 1e-6
    for _ in range(max_iter):
        if np.max(np.abs(grad)) < tol:
            break
        curvature = np.asarray(hessian(jnp.asarray(phi), data, cfg))
        accepted = None
        for _ in range(25):
            step = _newton_step(curvature, damping, grad)
            accepted = None if step is None else _line_search(phi, value, grad, step, data, cfg)
            if accepted is not None:
                damping = max(damping * 0.3, 1e-10)
                break
            damping *= 10.0
        if accepted is None:
            break
        phi, value, grad = accepted
    curvature = np.asarray(hessian(jnp.asarray(phi), data, cfg))
    return LaplacePosterior(
        mu=phi,
        chol_precision=_chol_with_jitter(curvature),
        cfg=cfg,
        converged=bool(np.max(np.abs(grad)) < tol),
    )


class FitStatus(enum.Enum):
    """Result of a :class:`PosteriorCache` refit."""

    OK = "ok"
    NOT_CONVERGED = "not converged"
    """Newton stopped short of tolerance, so the rank intervals should be treated with care."""
    STALE = "stale"
    """The refit failed numerically and the previous posterior stands."""


def _status_of(posterior: LaplacePosterior) -> FitStatus:
    return FitStatus.OK if posterior.converged else FitStatus.NOT_CONVERGED


@dataclass
class PosteriorCache:
    """Memoized, warm-started refit for the refit-per-answer loop."""

    last: LaplacePosterior | None = None
    last_input: tuple[list[Observation], int, ModelConfig] | None = None
    """The ``(observations, n_items, cfg)`` that produced ``last``."""
    min_buffer: int = _OBS_BUFFER
    """Pair-buffer size handed to :func:`pack_observations`."""

    def fit(self, observations: list[Observation], n_items: int, cfg: ModelConfig) -> tuple[LaplacePosterior, FitStatus]:
        """Refit from an observation log."""
        if self.last is not None and (observations, n_items, cfg) == self.last_input:
            return self.last, _status_of(self.last)
        data = pack_observations(observations, n_items, min_buffer=self.min_buffer)
        previous = self.last if self.last is not None and self.last.mu.shape[0] == cfg.dim(n_items) else None
        try:
            self.last = fit_laplace(data, cfg, n_items, None if previous is None else previous.mu)
        except FloatingPointError, np.linalg.LinAlgError:
            if previous is None:
                raise
            return previous, FitStatus.STALE
        self.last_input = (list(observations), n_items, cfg)
        return self.last, _status_of(self.last)
