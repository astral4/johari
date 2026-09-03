"""Posterior rank statistics and output ordering.

The output list (:func:`final_order`) is the weighted feedback-arc-set local minimizer (:func:`weighted_fas_order`)
initialized from the expected-rank order, with :func:`tie_groups` drawn over it as equal ranks.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.special

from johari.config import EXTRA_LOG_DELTA, N_EXTRA

if TYPE_CHECKING:
    from collections.abc import Sequence

QUAD_NODES = 7
"""Gauss-Hermite nodes for the log δ integral in :func:`pairwise_beyond_gaussian`. More nodes = less error for a fitted posterior."""
_MIN_VAR = 1e-12


def _n_items(phi: npt.NDArray[np.float64]) -> int:
    """Utilities in a parameter vector, a matrix of them, or a covariance over them."""
    return phi.shape[-1] - N_EXTRA


def rank_weight(
    pos: npt.NDArray[np.int64] | npt.NDArray[np.float64] | int,
) -> npt.NDArray[np.float64]:
    """Return the weight of an inversion whose upper item sits at 0-based ``pos``."""
    return 1.0 / (np.asarray(pos, dtype=np.float64) + 2.0)


@dataclass(frozen=True)
class RankSummary:
    """Per-item rank statistics. All ranks are 0-based (0 = best)."""

    expected_rank: npt.NDArray[np.float64]
    rank_lo: npt.NDArray[np.float64]
    """Lower end of the central credible interval on rank."""
    rank_hi: npt.NDArray[np.float64]
    """Upper end of the central credible interval on rank."""
    order: npt.NDArray[np.int64]
    """Item indices best-to-worst by expected rank."""

    @property
    def width(self) -> npt.NDArray[np.float64]:
        """Width of the credible interval on rank, per item."""
        return self.rank_hi - self.rank_lo


def rank_summary(samples: npt.NDArray[np.float64], *, interval: float = 0.8) -> RankSummary:
    """Summarize rank distributions from posterior samples of the full vector."""
    n_items = _n_items(samples)
    theta = samples[:, :n_items]
    ranks = np.empty(theta.shape, dtype=np.int64)
    np.put_along_axis(ranks, np.argsort(-theta, axis=1), np.arange(n_items), axis=1)
    tail = (1.0 - interval) / 2.0
    expected_rank = ranks.mean(axis=0)
    rank_lo, rank_hi = np.quantile(ranks, [tail, 1.0 - tail], axis=0)
    return RankSummary(
        expected_rank=expected_rank,
        rank_lo=rank_lo,
        rank_hi=rank_hi,
        order=np.argsort(expected_rank, kind="stable"),
    )


def _gap_and_var(mean: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean and variance of every gap θᵢ - θⱼ under q. In other words, this is (μᵢ - μⱼ, covᵢᵢ + covⱼⱼ - 2·covᵢⱼ)."""
    n_items = _n_items(mean)
    theta = mean[:n_items]
    block = cov[:n_items, :n_items]
    gap = theta[:, None] - theta[None, :]
    var = np.maximum(np.add.outer(np.diag(block), np.diag(block)) - 2.0 * block, _MIN_VAR)
    return gap, var


def pairwise_beyond_gaussian(
    mean: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    *,
    nodes: int = QUAD_NODES,
) -> npt.NDArray[np.float64]:
    """P[i, j] = P(θᵢ - θⱼ > δ) in closed form under the Gaussian posterior.

    In other words, this is the probability that the respondent would notice i above j.
    1 - P[i, j] - P[j, i] is the probability that the two are within the just-noticeable difference.
    With g = θᵢ - θⱼ and u = log δ jointly Gaussian under q:

        g | u ~ N(E[g] + (Cov(g, u) / Var u)·(u - E[u]),  Var g - Cov(g, u)² / Var u)
        P[i, j] = ∫ Φ((E[g | u] - exp(u)) / sd(g | u)) dN(u; E[u], Var u)

    The integral is computed deterministically via ``nodes``-point Gauss-Hermite quadrature.
    """
    n_items = _n_items(mean)
    gap, var = _gap_and_var(mean, cov)
    index = n_items + EXTRA_LOG_DELTA
    cross = cov[:n_items, index][:, None] - cov[:n_items, index][None, :]
    var_delta = float(cov[index, index])
    slope = cross / var_delta
    resid = np.maximum(var - slope**2 * var_delta, _MIN_VAR)
    z, w = np.polynomial.hermite_e.hermegauss(nodes)
    out = np.zeros_like(gap)
    sd_delta = np.sqrt(var_delta)
    slope_sd = slope * sd_delta
    inv_sd = 1.0 / np.sqrt(resid)
    for node, weight in zip(z, w / np.sqrt(2.0 * np.pi), strict=True):
        conditioned = gap + slope_sd * node
        delta = np.exp(mean[index] + sd_delta * node)
        out += weight * scipy.special.ndtr((conditioned - delta) * inv_sd)
    np.fill_diagonal(out, 0.0)
    return out


def expected_rank_gaussian(mean: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """E[rankᵢ] = Σⱼ P(θⱼ > θᵢ) = Σⱼ Φ((μⱼ - μᵢ) / sd(θⱼ - θᵢ)) in closed form under the Gaussian posterior.

    This is exact by linearity of expectation; ranks are 0-based, so ``np.argsort`` of the result is the expected-rank order.
    However, rank quantiles still need draws (see :func:`rank_summary`).
    """
    gap, var = _gap_and_var(mean, cov)
    above = scipy.special.ndtr(-gap / np.sqrt(var))
    np.fill_diagonal(above, 0.0)
    return above.sum(axis=1)


def expected_order(mean: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Return the expected-rank order (best first, ties by index) under the Gaussian posterior."""
    return np.argsort(expected_rank_gaussian(mean, cov), kind="stable")


def pair_risk(p_ij: npt.NDArray[np.float64], p_ji: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the Bayes risk of the best strict call on a pair, per unit position weight. In other words, this is min(pᵢⱼ, pⱼᵢ).

    With pᵢⱼ = P(θᵢ - θⱼ > δ) and pⱼᵢ its mirror, listing i above j costs pⱼᵢ and the reverse costs pᵢⱼ.
    So, the cost of a believed tie is nearly free, which lets the policy stop asking about it.
    """
    return np.minimum(p_ij, p_ji)


def weighted_fas_order(
    p_above: npt.NDArray[np.float64],
    *,
    init_order: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.int64]:
    """Locally minimize the top-weighted Bayes loss via adjacent-swap passes.

        L(order) = Σₖ w(k) · Σ_{m > k} P[order[m], order[k]],   w(k) = 1 / (k + 2)

    with P = ``p_above``. The cost of each inversion is the probability that it is one, weighted at the upper item's position
    (see :func:`rank_weight`). The exact minimizer is a weighted minimum feedback-arc-set problem, which is NP-hard in general,
    but local search from the expected-rank order quickly converges at this size and can only improve on its initializer.
    """
    n = p_above.shape[0]
    order = np.argsort(-p_above.sum(axis=1), kind="stable") if init_order is None else init_order.copy()
    weights = rank_weight(np.arange(n))
    # A swap of positions ``pos`` and ``pos + 1`` also reweights both items' pairs with everything below them,
    # so the delta reads the tail sums Σ_{m ≥ pos + 2} P[order[m], ·]. These are fixed for a whole pass
    # since a swap at ``pos`` leaves everything from ``pos + 2`` downward as the pass found it.
    tail = np.zeros((n + 1, n), dtype=np.float64)  # tail[n], the empty sum, is never written
    for _ in range(n):
        # tail[k] sums the rows of ``p_above`` for the items at positions k and below.
        np.cumsum(p_above[order[::-1]], axis=0, out=tail[n - 1 :: -1])
        improved = False
        for pos in range(n - 1):
            upper, lower = order[pos], order[pos + 1]
            s_upper = tail[pos + 2, upper]
            s_lower = tail[pos + 2, lower]
            before = weights[pos] * (p_above[lower, upper] + s_upper) + weights[pos + 1] * s_lower
            after = weights[pos] * (p_above[upper, lower] + s_lower) + weights[pos + 1] * s_upper
            if after < before - 1e-12:
                order[pos], order[pos + 1] = lower, upper
                improved = True
        if not improved:
            break
    return order


TIE_BELIEF = 0.7
"""Two items are shown as tied when the posterior puts at least this much mass on their gap being inside the just-noticeable difference."""


def tie_groups(
    order: npt.NDArray[np.int64] | Sequence[int],
    p_beyond: npt.NDArray[np.float64],
    *,
    belief: float = TIE_BELIEF,
) -> list[list[int]]:
    """Cut an order into tie groups (i.e. consecutive cliques of believed ties).

    i and j are believed tied when 1 - p_beyond[i, j] - p_beyond[j, i] ≥ ``belief``.
    A group grows by the next item only if it is believed tied with every member.
    """
    items = [int(i) for i in order]
    groups: list[list[int]] = []
    for item in items:
        if groups and all(1.0 - p_beyond[member, item] - p_beyond[item, member] >= belief for member in groups[-1]):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def final_order(mean: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], list[list[int]]]:
    """Return the output list under the Gaussian posterior: ``(order, tie groups)``.

    ``order`` is item indices best first, and the groups are consecutive runs of it.
    """
    p_beyond = pairwise_beyond_gaussian(mean, cov)
    order = weighted_fas_order(p_beyond, init_order=expected_order(mean, cov))
    return order, tie_groups(order, p_beyond)
