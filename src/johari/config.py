"""Model constants and defaults.

Population-level constants are in :class:`ModelConfig`.
Per-session latent parameters are in the vector fitted by :mod:`johari.posterior`.

Symbol reference:

    θ       theta       utilities, one per active item
    ε       eps         careless rate, fitted as ``eps_logit``
    c₁, c₂  c1, c2      bucket cutpoints, fitted as ``c1`` and ``log_gap`` = log(c₂ - c₁)
    δ       delta       just-noticeable difference, fitted as ``log_delta``
    σ₀      sigma0      prior sd of θ
    β       beta_pair   pair sharpness
    s_b     s_b         bucket noise scale
    Φ                   the standard normal CDF
    q, μ, H             the Laplace posterior N(μ, H⁻¹) (:mod:`johari.posterior`)
"""

import math
from dataclasses import dataclass

N_BUCKETS = 3
"""Bucket marks: 0 = meh, 1 = fine, 2 = love."""

N_EXTRA = 4
"""Non-utility parameters appended to θ: logit ε, c₁, log(c₂ - c₁), log δ."""
EXTRA_EPS_LOGIT, EXTRA_C1, EXTRA_LOG_GAP, EXTRA_LOG_DELTA = range(N_EXTRA)
"""Offsets of the extra parameters after the ``n_items`` utilities."""

DEFAULT_STOP_THRESHOLD = 0.03
"""Price of one more pair, in loss units. Auto-stop ends the pairs stage once
:data:`~johari.session.AUTO_STOP_STREAK` consecutive proposals predict a loss reduction below this.
At or below zero, auto-stop is disabled and the session can only be manually ended.

This is denominated in whatever :class:`~johari.selection.ELRPolicy` reports
and priced at ``SelectionConfig.price_beta_pair`` over ``s_eig`` draws.
There's also the winner's-curse inflation of a maximum over ``cand_cap`` candidates.
If any of those change, this value should be readjusted."""


@dataclass(frozen=True)
class ModelConfig:
    """Fixed constants of the observation model and priors.

    Utilities are in units of the pair channel's noise at β = 1.
    Scaling θ by c scales σ₀, s_b, c1_loc and c1_scale by c, shifts log_gap_loc and delta_loc by log c, and divides β by c.
    Fixing the cutpoint and log δ priors means σ₀ and s_b are identified separately.
    """

    sigma0: float = 3.75
    """Prior sd σ₀ of the utilities, θᵢ ~ N(0, σ₀²)."""

    beta_pair: float = 3.0
    """Pair sharpness β for the pair kernel, which receives β·θ and β·δ.
    The policy prices candidates at ``SelectionConfig.price_beta_pair``, not at this."""
    delta_loc: float = 0.0
    """Prior mean of log δ, the just-noticeable difference.
    In other words, this is the Rao-Kupper threshold inside which a pair is "about the same"."""
    delta_scale: float = 0.7
    """Prior sd of log δ. This guards against an early run of "same" answers pushing up δ
    and causing the entire list to be declared tied. """
    eps_a: float = 2.0
    """Careless-rate prior ε ~ Beta(a, b), shape a; the mean a / (a + b) is 0.05."""
    eps_b: float = 38.0
    """Careless-rate prior shape b."""

    s_b: float = 1.5
    """Bucket noise scale s_b, in utility units."""
    c1_loc: float = -1.125
    """Prior mean of the lower cutpoint c₁. This should be kept central."""
    c1_scale: float = 2.25
    """Prior sd of c₁."""
    log_gap_loc: float = math.log(2.25)
    """Prior mean of log(c₂ - c₁), the cutpoint gap."""
    log_gap_scale: float = 0.75
    """Prior sd of log(c₂ - c₁)."""
    bucket_slip: float = 0.02
    """Weight of a uniformly random bucket mixed into each mark. Guards against mis-keying.
    P_obs(b) = (1 - slip)·P_probit(b) + slip / 3, so no mark scores below log(slip / 3). Zero is the plain ordinal probit."""

    def __post_init__(self) -> None:
        """Reject invalid parameters."""
        if not 0.0 <= self.bucket_slip < 1.0:
            msg = f"bucket_slip must be in [0, 1), got {self.bucket_slip}"
            raise ValueError(msg)
        for name in ("sigma0", "beta_pair", "delta_scale", "eps_a", "eps_b", "s_b", "c1_scale", "log_gap_scale"):
            if getattr(self, name) <= 0.0:
                msg = f"{name} must be positive, got {getattr(self, name)}"
                raise ValueError(msg)

    def dim(self, n_items: int) -> int:
        """Length of the parameter vector for ``n_items`` active characters."""
        return n_items + N_EXTRA
