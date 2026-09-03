"""In-memory session state."""

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from johari.config import DEFAULT_STOP_THRESHOLD, ModelConfig
from johari.observations import BucketMark, Observation, ScreenPick, ScreenTie
from johari.selection import SelectionConfig, SessionConstraints

if TYPE_CHECKING:
    from johari.roster import Character, Roster

AUTO_STOP_STREAK = 3
"""Consecutive proposals priced below the stop threshold before ending the pairs stage."""


@dataclass(frozen=True)
class PairRecord:
    """The response for a pair in the pairs stage."""

    shown: tuple[int, int]
    answer: ScreenPick | ScreenTie | None
    """The value is ``None`` for a skip."""
    price: float
    """The predicted loss reduction that the pair was proposed at."""

    def __post_init__(self) -> None:
        """Reject an answer about a different pair than the one shown."""
        if self.answer is not None and self.answer.items != self.shown:
            msg = f"answer {self.answer.items} is not about the pair shown {self.shown}"
            raise ValueError(msg)


class Undone(enum.Enum):
    """The item retracted by :meth:`Session.undo`."""

    PAIR = "pair"
    MARK = "mark"
    NOTHING = "nothing"


class Stream(enum.IntEnum):
    """Tags separating the session's random streams."""

    PROPOSAL = 0
    REVIEW = 1
    DISPLAY = 2


@dataclass
class Session:
    """Session state."""

    roster: Roster
    cfg: ModelConfig = field(default_factory=ModelConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    seed: int = 0
    stop_threshold: float = DEFAULT_STOP_THRESHOLD
    marks: list[tuple[int, int | None]] = field(default_factory=list)
    """``(roster ID, bucket)`` in the order marked. ``None`` corresponds to "don't know"."""
    pairs: list[PairRecord] = field(default_factory=list)
    """Pairs shown in order, skips included."""

    def next_unmarked(self) -> int | None:
        """Roster ID of the next character without a mark, in roster order."""
        handled = {char for char, _ in self.marks}
        return next((i for i in range(len(self.roster)) if i not in handled), None)

    def mark(self, char: int, bucket: int | None) -> None:
        """Record a bucket (or "don't know") for a character."""
        if self.pairs:
            msg = "the bucket pass is closed once pairs have been shown"
            raise ValueError(msg)
        if any(c == char for c, _ in self.marks):
            msg = f"character {char} is already marked"
            raise ValueError(msg)
        self.marks.append((char, bucket))

    @property
    def active(self) -> list[int]:
        """Roster IDs of the bucketed characters in roster order."""
        return sorted(char for char, bucket in self.marks if bucket is not None)

    @property
    def excluded(self) -> list[int]:
        """Roster IDs marked "don't know"."""
        return sorted(char for char, bucket in self.marks if bucket is None)

    def character(self, local: int) -> Character:
        """Return the character at a session-local index."""
        return self.roster.characters[self.active[local]]

    @property
    def screens_answered(self) -> int:
        """Pairs answered with a pick or a tie (skips excluded)."""
        return sum(1 for record in self.pairs if record.answer is not None)

    def record(self, shown: tuple[int, int], answer: ScreenPick | ScreenTie | None, price: float) -> None:
        """Append one shown pair with its answer (``None`` for a skip) and its price."""
        self.pairs.append(PairRecord(shown, answer, price))

    def undo(self) -> Undone:
        """Retract the most recent data action."""
        if self.pairs:
            self.pairs.pop()
            return Undone.PAIR
        if self.marks:
            self.marks.pop()
            return Undone.MARK
        return Undone.NOTHING

    def constraints(self) -> SessionConstraints:
        """Return the never-repeat state (every pair shown so far, skips included)."""
        return SessionConstraints({frozenset(record.shown) for record in self.pairs})

    def exhausted(self) -> bool:
        """Whether every distinct pair of active characters has been shown."""
        return self.constraints().is_exhausted(len(self.active))

    def should_stop(self, utility: float) -> bool:
        """Auto-stop verdict for a proposal priced at ``utility``."""
        threshold = self.stop_threshold
        if threshold <= 0.0 or not utility < threshold:
            return False
        run = 1
        for record in reversed(self.pairs):
            if not record.price < threshold:
                break
            run += 1
        return run >= AUTO_STOP_STREAK

    def rng(self, stream: Stream) -> np.random.Generator:
        """Return a generator determined by the seed, the stream, and the pairs shown."""
        return np.random.default_rng([self.seed, int(stream), len(self.pairs)])

    def observations(self) -> list[Observation]:
        """Model observations over local indices. Returns the bucket marks in roster order, then the pairs."""
        local = {char: i for i, char in enumerate(self.active)}
        marks = [BucketMark(local[c], b) for c, b in sorted(self.marks) if b is not None]
        answers = [r.answer for r in self.pairs if r.answer is not None]
        return [*marks, *answers]
