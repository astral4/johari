"""Adaptive pair selection logic including candidates, the scorer, and the policy.

Every question is a pair. :class:`ELRPolicy` maximizes the one-step expected reduction
of a pairwise surrogate of the top-weighted Bayes risk,

    L(q) = Σ_{i < j} w(min(posᵢ, posⱼ)) · min(pᵢⱼ, pⱼᵢ),    pᵢⱼ = P_q(θᵢ - θⱼ > δ),

over a candidate pipeline. The pipeline involves μ-sorted windows with jitter, uncertainty-led pairs, probes, and a few random pairs,
padded with random fill to a fixed, JIT-friendly count. Random pairs are legitimate candidates, so the padding isn't wasted.

When the eligible space holds fewer distinct pairs than the cap, the list cycles duplicates,
which score identically and cannot perturb the argmax.
"""

import functools
import math
from dataclasses import dataclass, field, replace
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from johari.config import ModelConfig
from johari.likelihood import pair_outcome_logprobs
from johari.posterior import LaplacePosterior, unpack
from johari.ranking import RankSummary, pair_risk, rank_summary, rank_weight


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for candidate generation and the scorer."""

    s_eig: int = 128
    """Posterior draws reweighted by the scorer."""
    cand_cap: int = 448
    """Candidates scored per step (padded to exactly this count)."""
    window_jitter: int = 2
    """Extra pairs per mu-adjacent pair, each keeping one of its items and swapping the other for one from its rank-overlap neighborhood."""
    n_uncertain: int = 5
    """Anchors for uncertainty-led pairs."""
    n_probes: int = 3
    """Anchors for probe pairs (an unresolved item against a confidently placed neighbor)."""
    n_explore: int = 8
    """Purely random pairs per step."""
    uncertainty_horizon: float = 60.0
    """Uncertainty anchors must have expected rank at or above this."""
    pair_budget: int = 496
    """Pairs priced by the scorer, chosen per step by current risk contribution
    (see :func:`contribution_pairs`) and capped at ``C(n, 2)``."""
    price_beta_pair: float = 1.0
    """Pair sharpness for the scorer to price candidates at."""

    def __post_init__(self) -> None:
        """Reject an empty draw set, candidate list, or pair set."""
        for name in ("s_eig", "cand_cap", "pair_budget"):
            if getattr(self, name) < 1:
                msg = f"{name} must be at least 1, got {getattr(self, name)}"
                raise ValueError(msg)


@dataclass(frozen=True)
class SessionConstraints:
    """Pairs already shown this session."""

    seen_sets: frozenset[frozenset[int]] = field(default_factory=frozenset)

    def allows(self, items: frozenset[int]) -> bool:
        """Check a candidate pair against the never-repeat rule."""
        return items not in self.seen_sets

    def unshown(self, n_items: int) -> int:
        """Return the number of distinct pairs of ``n_items`` items not yet shown."""
        return math.comb(n_items, 2) - len(self.seen_sets)

    def is_exhausted(self, n_items: int) -> bool:
        """Whether every distinct pair of ``n_items`` items has been shown."""
        return self.unshown(n_items) <= 0


def _window_candidates(
    ranked: npt.NDArray[np.int64],
    summary: RankSummary,
    jitter: int,
    rng: np.random.Generator,
) -> list[npt.NDArray[np.int64]]:
    """Return mu-adjacent pairs, each with ``jitter`` variants that keep one of its items and swap the other for a neighbor.

    The neighborhood is every other item whose rank interval overlaps the pair's.
    """
    out: list[npt.NDArray[np.int64]] = []
    n = ranked.shape[0]
    lo_ranked, hi_ranked = summary.rank_lo[ranked], summary.rank_hi[ranked]

    window_lo = np.minimum(lo_ranked[:-1], lo_ranked[1:])[:, None]
    window_hi = np.maximum(hi_ranked[:-1], hi_ranked[1:])[:, None]
    outside_of = (summary.rank_hi[None, :] >= window_lo) & (summary.rank_lo[None, :] <= window_hi)
    starts = np.arange(n - 1)
    outside_of[starts, ranked[:-1]] = False
    outside_of[starts, ranked[1:]] = False

    for start in range(n - 1):
        window = ranked[start : start + 2]
        out.append(window.copy())
        outside = np.flatnonzero(outside_of[start])
        if outside.size == 0:
            continue
        for _ in range(jitter):
            variant = window.copy()
            variant[rng.integers(2)] = rng.choice(outside)
            out.append(variant)
    return out


def _uncertainty_candidates(
    ranked: npt.NDArray[np.int64],
    summary: RankSummary,
    sel: SelectionConfig,
    rng: np.random.Generator,
) -> list[npt.NDArray[np.int64]]:
    """Pairs built around the widest rank intervals with non-trivial weight."""
    out: list[npt.NDArray[np.int64]] = []
    near_top = ranked[summary.expected_rank[ranked] <= sel.uncertainty_horizon]
    for anchor in _widest(near_top, summary, sel.n_uncertain):
        overlapping = ranked[(summary.rank_hi[ranked] >= summary.rank_lo[anchor]) & (summary.rank_lo[ranked] <= summary.rank_hi[anchor])]
        pool = np.setdiff1d(overlapping, np.array([anchor]))
        if pool.size == 0:
            pool = np.setdiff1d(ranked, np.array([anchor]))
        out.append(np.array([anchor, rng.choice(pool)]))
    return out


def _probe_candidates(
    ranked: npt.NDArray[np.int64],
    summary: RankSummary,
    sel: SelectionConfig,
) -> list[npt.NDArray[np.int64]]:
    """Probes; i.e. the least resolved items, each against the most confidently placed one near it."""
    out: list[npt.NDArray[np.int64]] = []
    for anchor in _widest(ranked, summary, sel.n_probes):
        others = np.setdiff1d(ranked, np.array([anchor]))
        if others.size == 0:
            continue
        expected = summary.expected_rank[others]
        inside = others[(expected >= summary.rank_lo[anchor]) & (expected <= summary.rank_hi[anchor])]
        pool = inside if inside.size else others
        near = np.abs(summary.expected_rank[pool] - summary.expected_rank[anchor])
        resolved_first = pool[np.lexsort((near, summary.width[pool]))]
        out.append(np.array([anchor, resolved_first[0]]))
    return out


def _widest(items: npt.NDArray[np.int64], summary: RankSummary, count: int) -> npt.NDArray[np.int64]:
    """Pick the ``count`` widest-interval items of ``items``, widest first. Ties keep order."""
    return items[np.argsort(-summary.width[items], kind="stable")[:count]]


def _accept(
    candidate: npt.NDArray[np.int64],
    batch_seen: set[frozenset[int]],
    constraints: SessionConstraints,
) -> frozenset[int] | None:
    """Return the candidate's key if it passes constraints and is new to this batch."""
    key = frozenset(int(i) for i in candidate)
    if key in batch_seen or not constraints.allows(key):
        return None
    return key


_ENUM_MARGIN = 4
"""Enumerate the whole pair space when it is within this factor of ``cand_cap``."""


def _fill_by_enumeration(
    chosen: list[npt.NDArray[np.int64]],
    batch_seen: set[frozenset[int]],
    n_items: int,
    sel: SelectionConfig,
    constraints: SessionConstraints,
    rng: np.random.Generator,
) -> None:
    """Fill from the fully enumerated pair space, unseen pairs first.

    Already-shown pairs are only touched if no unshown pair exists at all.
    Tiers are shuffled before truncation. Otherwise, ``combinations`` order would bias candidates toward low item indices.
    """
    allowed: list[tuple[int, ...]] = []
    seen: list[tuple[int, ...]] = []
    for combo in combinations(range(n_items), 2):
        key = frozenset(combo)
        if key in batch_seen:
            continue
        (allowed if constraints.allows(key) else seen).append(combo)
    tiers = [allowed]
    if not chosen and not allowed:
        tiers.append(seen)  # every eligible pair has been shown
    for tier in tiers:
        rng.shuffle(tier)
        for combo in tier:
            if len(chosen) == sel.cand_cap:
                return
            batch_seen.add(frozenset(combo))
            chosen.append(np.array(combo, dtype=np.int64))


def _fill_by_sampling(
    chosen: list[npt.NDArray[np.int64]],
    batch_seen: set[frozenset[int]],
    n_items: int,
    sel: SelectionConfig,
    constraints: SessionConstraints,
    rng: np.random.Generator,
) -> None:
    """Fill by rejection sampling over unshown pairs.

    This is only reached while more unshown pairs remain than the cap, so acceptance stays high.
    """
    budget = 25 * sel.cand_cap
    while len(chosen) < sel.cand_cap and budget > 0:
        candidate = np.sort(rng.choice(n_items, size=2, replace=False))
        key = _accept(candidate, batch_seen, constraints)
        if key is None:
            budget -= 1
            continue
        batch_seen.add(key)
        chosen.append(candidate)


def generate_candidates(
    summary: RankSummary,
    sel: SelectionConfig,
    constraints: SessionConstraints,
    rng: np.random.Generator,
) -> npt.NDArray[np.int32]:
    """Generate exactly ``sel.cand_cap`` candidate pairs.

    When the space holds fewer than ``cand_cap`` distinct pairs, the list is padded by cycling duplicates.
    """
    n_items = summary.expected_rank.shape[0]
    ranked = summary.order

    windows = _window_candidates(ranked, summary, sel.window_jitter, rng)
    uncertain = _uncertainty_candidates(ranked, summary, sel, rng)
    probes = _probe_candidates(ranked, summary, sel)
    explore = [rng.choice(n_items, size=2, replace=False) for _ in range(sel.n_explore)]

    batch_seen: set[frozenset[int]] = set()
    chosen: list[npt.NDArray[np.int64]] = []
    for candidate in [*uncertain, *probes, *explore, *windows]:
        key = _accept(candidate, batch_seen, constraints)
        if key is None:
            continue
        batch_seen.add(key)
        chosen.append(np.sort(candidate))
        if len(chosen) == sel.cand_cap:
            break

    if len(chosen) < sel.cand_cap:
        space = math.comb(n_items, 2)
        if space <= _ENUM_MARGIN * sel.cand_cap or constraints.unshown(n_items) <= sel.cand_cap:
            _fill_by_enumeration(chosen, batch_seen, n_items, sel, constraints, rng)
        else:
            _fill_by_sampling(chosen, batch_seen, n_items, sel, constraints, rng)
    if not chosen:
        # This is unreachable in practice because the enumeration tier always yields at least one pair.
        # However, this guarantees that the padding loop below terminates.
        chosen.append(np.sort(rng.choice(n_items, size=2, replace=False)))
    n_distinct = len(chosen)
    for i in range(sel.cand_cap - n_distinct):
        chosen.append(chosen[i % n_distinct])
    return np.stack(chosen).astype(np.int32)


_COUNT_BLOCK = 16
"""Draws per block in :func:`_above_counts`."""


@jax.jit
def _above_counts(theta: jax.Array, delta: jax.Array) -> jax.Array:
    """Nᵢⱼ = #{draws s : θᵢ - θⱼ > δ in draw s}, as an ``(n, n)`` count."""
    n_draws, n = theta.shape

    def add_block(total: jax.Array, block: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
        block_theta, block_delta = block
        above = block_theta[:, :, None] - block_theta[:, None, :] > block_delta[:, None, None]
        return total + jnp.sum(above.astype(jnp.int16), axis=0).astype(jnp.int32), None

    whole = n_draws - n_draws % _COUNT_BLOCK
    blocks = (
        theta[:whole].reshape(-1, _COUNT_BLOCK, n),
        delta[:whole].reshape(-1, _COUNT_BLOCK),
    )
    total, _ = jax.lax.scan(add_block, jnp.zeros((n, n), jnp.int32), blocks)
    if whole < n_draws:
        total, _ = add_block(total, (theta[whole:], delta[whole:]))
    return total


def contribution_pairs(
    theta: jax.Array | npt.NDArray[np.float64],
    delta: jax.Array | npt.NDArray[np.float64],
    summary: RankSummary,
    budget: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64], float]:
    """Select the ``budget`` pairs with the largest term in the surrogate risk L(q).

    Each pair's term is cᵢⱼ = w(min(posᵢ, posⱼ)) · min(pᵢⱼ, pⱼᵢ), with pᵢⱼ = P(θᵢ - θⱼ > δ) counted over the draws,
    each draw with its own δ, and the position weight (see :func:`~johari.ranking.rank_weight`) frozen at the current expected-rank order.

    Returns ``(ii, jj, weights, risk_now)``: the kept pairs, their position weights, and Σ cᵢⱼ over them
    (which is the baseline that the scorer subtracts from).
    """
    if budget < 1:
        msg = f"budget must be at least 1, got {budget}"
        raise ValueError(msg)
    n = theta.shape[1]
    pos = np.empty(n, dtype=np.int64)
    pos[summary.order] = np.arange(n)
    ii, jj = np.triu_indices(n, k=1)
    n_draws = theta.shape[0]
    counts = np.asarray(_above_counts(theta, delta))
    p_ij = counts[ii, jj] / n_draws
    p_ji = counts[jj, ii] / n_draws
    weights = rank_weight(np.minimum(pos[ii], pos[jj]))
    contribution = weights * pair_risk(p_ij, p_ji)
    if budget < contribution.size:
        keep = np.argpartition(-contribution, budget - 1)[:budget]
        ii, jj, weights, contribution = ii[keep], jj[keep], weights[keep], contribution[keep]
    return ii.astype(np.int64), jj.astype(np.int64), weights, float(contribution.sum())


@functools.partial(jax.jit, static_argnames="cfg")
def _elr_batch(
    theta: jax.Array,
    eps_logit: jax.Array,
    delta: jax.Array,
    candidates: jax.Array,
    ii: jax.Array,
    jj: jax.Array,
    weights: jax.Array,
    cfg: ModelConfig,
) -> jax.Array:
    """Compute E_y[L(q | y)], the expected surrogate risk after one more answer, per candidate.

    Reweighting the draws by the answer's likelihood is the one-observation Bayes update of the ensemble,
    so no refitting is needed to price outcomes. Per candidate, with s over draws and p over the priced pairs ``(ii, jj)``:

        w_s(y)   = p(y | φ_s) / Σ_s' p(y | φ_s')       weight of draw s given answer y
        P(y)     = mean_s p(y | φ_s)                   predictive probability of y
        pᵢⱼ(y)   = Σ_s w_s(y) · 1[θᵢ - θⱼ > δ]_s       P(θᵢ - θⱼ > δ | y); pⱼᵢ is its mirror
        L(q | y) = Σ_p weights_p · min(pᵢⱼ(y), pⱼᵢ(y))
        return     Σ_y P(y) · L(q | y)

    The indicators are formed here so XLA fuses them into the reweighting.
    """
    gap = theta[:, ii] - theta[:, jj]
    ind_ij = (gap > delta[:, None]).astype(theta.dtype)
    ind_ji = (gap < -delta[:, None]).astype(theta.dtype)
    logp = pair_outcome_logprobs(theta[:, candidates], eps_logit[:, None], delta[:, None], cfg)
    p = jnp.exp(jnp.minimum(logp, 0.0))
    p_bar = p.mean(axis=0)
    w = p / jnp.maximum(p.sum(axis=0), 1e-300)
    p_ij = jnp.einsum("scy,sp->cyp", w, ind_ij)
    p_ji = jnp.einsum("scy,sp->cyp", w, ind_ji)
    best = jnp.minimum(p_ij, p_ji)
    risk = jnp.einsum("cyp,p->cy", best, weights)
    return jnp.sum(p_bar * risk, axis=1)


def score_candidates_elr(
    samples: npt.NDArray[np.float64],
    summary: RankSummary,
    candidates: npt.NDArray[np.int32],
    cfg: ModelConfig,
    sel: SelectionConfig,
) -> npt.NDArray[np.float64]:
    """Estimate ELR = L(q) - E_y[L(q | y)] per candidate. In other words, this is the one-step expected reduction of the surrogate risk.

    L(q) is summed over the :func:`contribution_pairs` set and E_y priced by :func:`_elr_batch`. The reweighted probabilities
    are exact martingales in-sample, so every score is nonnegative by Jensen's inequality. ``summary`` must come from ``samples``.
    """
    p = unpack(jnp.asarray(samples))
    delta = jnp.exp(p.log_delta)
    ii, jj, weights, risk_now = contribution_pairs(p.theta, delta, summary, sel.pair_budget)
    scored = _elr_batch(p.theta, p.eps_logit, delta, candidates, ii, jj, weights, cfg)
    return risk_now - np.asarray(scored)


def price_shape(cfg: ModelConfig, sel: SelectionConfig) -> ModelConfig:
    """Return the shape for scorer pricing."""
    return replace(cfg, beta_pair=sel.price_beta_pair)


def sample_and_generate(
    posterior: LaplacePosterior,
    constraints: SessionConstraints,
    rng: np.random.Generator,
    sel: SelectionConfig,
) -> tuple[npt.NDArray[np.float64], RankSummary, npt.NDArray[np.int32]]:
    """Return posterior draws, a rank summary, and candidates."""
    samples = posterior.sample(rng, sel.s_eig)
    summary = rank_summary(samples)
    candidates = generate_candidates(summary, sel, constraints, rng)
    return samples, summary, candidates


@dataclass(frozen=True)
class ELRPolicy:
    """Decision-aware pair selection based on maximizing expected loss reduction.

    Scores each candidate via :func:`score_candidates_elr` and shows the best.
    """

    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def propose(
        self,
        posterior: LaplacePosterior,
        constraints: SessionConstraints,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.int64], float]:
        """Return the highest-scoring candidate and its predicted loss reduction."""
        sel = self.selection
        samples, summary, candidates = sample_and_generate(posterior, constraints, rng, sel)
        elr = score_candidates_elr(samples, summary, candidates, price_shape(posterior.cfg, sel), sel)
        best = int(np.argmax(elr))
        return candidates[best].astype(np.int64), float(elr[best])
