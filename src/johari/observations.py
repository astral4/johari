"""Observation types and packing into padded device arrays."""

from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from johari.config import N_BUCKETS
from johari.likelihood import TIE_OUTCOME

PAIR_ITEMS = 2


def _check_pair(items: tuple[int, int]) -> None:
    if len(items) != PAIR_ITEMS or len(set(items)) != PAIR_ITEMS:
        msg = f"expected {PAIR_ITEMS} distinct items, got {items}"
        raise ValueError(msg)


@dataclass(frozen=True)
class ScreenPick:
    """A pair answered with a preference for ``picked``, one of ``items``."""

    items: tuple[int, int]
    """The two items in display order."""
    picked: int

    def __post_init__(self) -> None:
        """Reject a malformed pair."""
        _check_pair(self.items)
        if self.picked not in self.items:
            msg = f"pick {self.picked} is not on the pair {self.items}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScreenTie:
    """A pair answered "about the same"."""

    items: tuple[int, int]

    def __post_init__(self) -> None:
        """Reject a malformed pair."""
        _check_pair(self.items)


@dataclass(frozen=True)
class BucketMark:
    """A bucket mark. 0 = meh, 1 = fine, 2 = love."""

    item: int
    bucket: int


Observation = ScreenPick | ScreenTie | BucketMark


class ObsData(NamedTuple):
    """Padded device arrays over which the log posterior is evaluated."""

    pair_items: jax.Array
    """``(buffer, 2)`` int32; item indices per pair row in display order (0-padded)."""
    pair_outcome: jax.Array
    """``(buffer,)`` int32; the outcome column of :func:`~johari.likelihood.pair_outcome_logprobs`.
    This is the picked display position or :data:`~johari.likelihood.TIE_OUTCOME` (0 on padded rows)."""
    pair_mask: jax.Array
    """``(buffer,)`` bool; which pair rows are real."""
    bucket_item: jax.Array
    """``(n_items,)`` int32; item index per bucket row (0-padded)."""
    bucket_val: jax.Array
    """``(n_items,)`` int32 bucket value per row."""
    bucket_mask: jax.Array
    """``(n_items,)`` bool; which bucket rows are real."""


def _pair_outcome(row: int, obs: ScreenPick | ScreenTie, n_items: int) -> int:
    """Validate one pair against the roster and return its outcome column."""
    if any(not 0 <= item < n_items for item in obs.items):
        msg = f"pair row {row}: items outside [0, {n_items}): {obs.items}"
        raise ValueError(msg)
    if isinstance(obs, ScreenTie):
        return TIE_OUTCOME
    return obs.items.index(obs.picked)


def _check_bucket_row(row: int, mark: BucketMark, n_items: int) -> None:
    if not 0 <= mark.item < n_items:
        msg = f"bucket row {row}: item {mark.item} outside [0, {n_items})"
        raise ValueError(msg)
    if mark.bucket not in range(N_BUCKETS):
        msg = f"bucket row {row}: bucket {mark.bucket} not in [0, {N_BUCKETS})"
        raise ValueError(msg)


def pack_observations(
    observations: list[Observation],
    n_items: int,
    *,
    min_buffer: int = 16,
) -> ObsData:
    """Pack an observation log into padded arrays for the log posterior.

    The pair buffer is the smallest power-of-two multiple of ``min_buffer`` that fits the log.

    Every row of every returned array, padded or real, is a valid likelihood input on its own. So, the log likelihood and its derivatives
    are finite at any parameter vector whose spread stays within exp range, unlike masking with ``jnp.where``.

    Raises
    ------
    ValueError
        If item indices fall outside ``[0, n_items)`` or an item carries more than one bucket mark.
    """
    pairs = [o for o in observations if not isinstance(o, BucketMark)]
    buckets = [o for o in observations if isinstance(o, BucketMark)]
    counts = Counter(mark.item for mark in buckets)
    duplicated = sorted(item for item, count in counts.items() if count > 1)
    if duplicated:
        msg = f"multiple bucket marks for items {duplicated}; at most one per item"
        raise ValueError(msg)

    buffer = min_buffer
    while buffer < len(pairs):
        buffer *= 2
    p_items = np.zeros((buffer, 2), dtype=np.int32)
    p_outcome = np.zeros(buffer, dtype=np.int32)
    p_mask = np.zeros(buffer, dtype=bool)
    for row, obs in enumerate(pairs):
        p_items[row] = obs.items
        p_outcome[row] = _pair_outcome(row, obs, n_items)
        p_mask[row] = True

    b_item = np.zeros(n_items, dtype=np.int32)
    b_val = np.zeros(n_items, dtype=np.int32)
    b_mask = np.zeros(n_items, dtype=bool)
    for row, mark in enumerate(buckets):
        _check_bucket_row(row, mark, n_items)
        b_item[row] = mark.item
        b_val[row] = mark.bucket
        b_mask[row] = True

    return ObsData(
        pair_items=jnp.asarray(p_items),
        pair_outcome=jnp.asarray(p_outcome),
        pair_mask=jnp.asarray(p_mask),
        bucket_item=jnp.asarray(b_item),
        bucket_val=jnp.asarray(b_val),
        bucket_mask=jnp.asarray(b_mask),
    )
