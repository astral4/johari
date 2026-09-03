"""Pair and bucket likelihoods.

A pair is a Rao-Kupper (1967) threshold comparison under a careless mixture (:func:`pair_outcome_logprobs`).
A bucket mark is an ordinal probit under a uniform slip (:func:`bucket_logprob`). Every formula is shift-invariant in θ
and evaluated in log space after subtracting the larger utility, so no term underflows for any finite spread.
"""

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from johari.config import ModelConfig

TIE_OUTCOME = 2
"""Outcome index of "about the same"."""

_MIN_DELTA = 1e-300


def pair_outcome_logprobs(
    theta_pair: ArrayLike,
    eps: ArrayLike,
    delta: ArrayLike,
    cfg: ModelConfig,
) -> jax.Array:
    """Log-probabilities of the three pair outcomes ``[A, B, same]``.

    Let w = exp(β·θ), t = exp(β·δ), and δ be the just-noticeable difference in utility units:

        P(A)    = w_A / (w_A + t·w_B)
        P(B)    = w_B / (w_B + t·w_A)
        P(same) = (t² - 1)·w_A·w_B / ((w_A + t·w_B)·(w_B + t·w_A))

    which sum to one exactly; then the careless mixture at rate ε,

        P_obs(pick) = (1 - ε)·P(pick) + ε / 2,    P_obs(same) = (1 - ε)·P(same).

    ``theta_pair`` is ``(..., 2)`` in display order; ``eps`` and ``delta`` broadcast against its batch dimensions.
    """
    theta_pair, eps = jnp.asarray(theta_pair), jnp.asarray(eps)
    scale = cfg.beta_pair
    delta = jnp.maximum(jnp.asarray(delta), _MIN_DELTA) * scale
    z = scale * (theta_pair - jnp.max(theta_pair, axis=-1)[..., None])
    log_wa, log_wb = z[..., 0], z[..., 1]
    log_da = jnp.logaddexp(log_wa, delta + log_wb)
    log_db = jnp.logaddexp(log_wb, delta + log_wa)
    log_pa = log_wa - log_da
    log_pb = log_wb - log_db
    # log(t² - 1) = 2·delta + log(1 - exp(-2·delta)).
    log_t2m1 = 2.0 * delta + _log1mexp(-2.0 * delta)
    log_same = log_t2m1 + log_wa + log_wb - log_da - log_db
    log_picks = jnp.logaddexp(
        jnp.log1p(-eps)[..., None] + jnp.stack([log_pa, log_pb], axis=-1),
        jnp.log(eps)[..., None] - math.log(2),
    )
    log_tie = jnp.log1p(-eps) + log_same
    return jnp.concatenate([log_picks, log_tie[..., None]], axis=-1)


_LOG_HALF = math.log(0.5)


def _log1mexp(x: jax.Array) -> jax.Array:
    """log(1 - exp(x)) for x < 0 (Mächler 2012)."""
    return jnp.where(
        x < _LOG_HALF,
        jnp.log1p(-jnp.exp(jnp.minimum(x, _LOG_HALF))),
        jnp.log(-jnp.expm1(jnp.maximum(x, _LOG_HALF))),
    )


def _log_cdf_diff(lo: jax.Array, hi: jax.Array) -> jax.Array:
    """log(Φ(hi) - Φ(lo)) for hi > lo."""
    flip = lo + hi > 0.0
    a = jnp.where(flip, -hi, lo)
    b = jnp.where(flip, -lo, hi)
    log_a: jax.Array = jax.scipy.special.log_ndtr(a)
    log_b: jax.Array = jax.scipy.special.log_ndtr(b)
    return log_b + _log1mexp(log_a - log_b)


def bucket_logprob(
    theta_item: ArrayLike,
    c1: ArrayLike,
    c2: ArrayLike,
    bucket: ArrayLike,
    cfg: ModelConfig,
) -> jax.Array:
    """Log-probability of a bucket mark b ∈ {0 = meh, 1 = fine, 2 = love}.

    An ordinal probit with cutpoints c₁ < c₂ and scale s_b, under a uniform slip:

        P(0) = Φ(z₁)
        P(1) = Φ(z₂) - Φ(z₁)
        P(2) = 1 - Φ(z₂)
        zₖ = (cₖ - θ) / s_b
        P_obs(b) = (1 - slip)·P(b) + slip / 3

    so no mark scores below log(slip / 3). At slip = 0, this is the plain probit.
    """
    theta_item, c1, c2 = jnp.asarray(theta_item), jnp.asarray(c1), jnp.asarray(c2)
    z1 = (c1 - theta_item) / cfg.s_b
    z2 = (c2 - theta_item) / cfg.s_b
    logps = jnp.stack(
        [
            jax.scipy.stats.norm.logcdf(z1),
            _log_cdf_diff(z1, z2),
            jax.scipy.stats.norm.logcdf(-z2),
        ],
        axis=-1,
    )
    if cfg.bucket_slip > 0.0:
        n_buckets = logps.shape[-1]
        logps = jnp.logaddexp(math.log1p(-cfg.bucket_slip) + logps, math.log(cfg.bucket_slip / n_buckets))
    picked = jnp.take_along_axis(logps, jnp.asarray(bucket)[..., None], axis=-1)
    return jnp.squeeze(picked, axis=-1)
