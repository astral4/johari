"""Interactive TUI for sessions."""

import argparse
import enum
import math
import os
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from johari.config import DEFAULT_STOP_THRESHOLD
from johari.observations import PAIR_ITEMS, ScreenPick, ScreenTie
from johari.posterior import FitStatus, LaplacePosterior, PosteriorCache
from johari.ranking import expected_order, final_order, rank_summary
from johari.roster import load_roster
from johari.selection import ELRPolicy
from johari.session import Session, Stream, Undone

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from johari.roster import Roster

_BUCKET_KEYS = {"l": 2, "f": 1, "m": 0}
_BUCKET_HELP = (
    "  l = love · f = fine · m = meh · x = don't know them (excludes from ranking)",
    "  u = undo · q = quit (discards the session) · ? = show this guide again",
)
_PAIR_HELP = (
    "  t = about the same · s = skip · u = undo",
    "  d = done (see your list) · q = quit (discards the session) · ? = show this guide again",
)
_REVIEW_HELP = ("  done = save and exit · b = back to pairs for more questions · ? = show this guide again",)
_PAIR_PROMPT = "1 / 2 / t (about the same) › "
_REVIEW_PROMPT = "done (save and exit) / b (back to pairs) › "
_QUIT_PROMPT = "Discard session and quit? [y / n] › "
_SAVE_PROMPT = "Save result to text file? [y / n] › "


class Next(enum.Enum):
    """Where the state machine goes after a stage returns."""

    BUCKETS = "buckets"
    PAIRS = "pairs"
    REVIEW = "review"
    DONE = "done"
    QUIT = "quit"


class Console:
    """Terminal I/O."""

    def __init__(
        self,
        reader: Callable[[str], str] | None = None,
        writer: Callable[[str], None] | None = None,
    ) -> None:
        """Establish the I/O endpoints, defaulting to real stdin and stdout."""
        self._reader: Callable[[str], str] | None = reader
        self._writer: Callable[[str], None] = writer if writer is not None else _stdout_line
        self._raw: bool = reader is None and sys.stdin.isatty()

    def say(self, text: str = "") -> None:
        """Print one line."""
        self._writer(text)

    def say_all(self, lines: Sequence[str]) -> None:
        """Print several lines."""
        for line in lines:
            self.say(line)

    def ask(self, prompt: str) -> str:
        """Read one line (stripped). Raises ``EOFError`` at the end of input."""
        if self._reader is not None:
            return self._reader(prompt).strip()
        return input(prompt).strip()

    def ask_key(self, prompt: str, allowed: Sequence[str]) -> str:
        """Read one key from ``allowed``. Other keys are ignored. Raises ``EOFError`` at the end of input."""
        if self._raw:
            return _ask_raw_key(prompt, allowed)
        while True:
            key = self.ask(prompt)[:1].lower()
            if key in allowed:
                return key


def _stdout_line(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _ask_raw_key(prompt: str, allowed: Sequence[str]) -> str:
    """Show ``prompt`` and read one key from ``allowed`` in cbreak mode, ignoring other keys."""
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd, termios.TCSANOW)
    except termios.error as error:
        raise EOFError from error
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while True:
            try:
                byte = os.read(fd, 1)
            except OSError as error:
                raise EOFError from error
            if not byte:
                raise EOFError
            key = chr(byte[0]).lower()
            if key in allowed:
                sys.stdout.write(key + "\n")
                sys.stdout.flush()
                return key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _confirm(console: Console, prompt: str) -> bool:
    return console.ask_key(prompt, ("y", "n")) == "y"


def _quit_if_confirmed(console: Console) -> Next | None:
    """Handle the ``q`` key."""
    return Next.QUIT if _confirm(console, _QUIT_PROMPT) else None


def display_interval(lo: float, hi: float) -> tuple[int, int]:
    """Round a 0-based rank quantile pair outward to a 1-based interval."""
    return math.floor(lo) + 1, math.ceil(hi) + 1


def ranks_of(groups: Sequence[int]) -> list[tuple[int, bool]]:
    """Return competition ranks for consecutive group sizes: ``(rank, tied)`` per listed item."""
    out: list[tuple[int, bool]] = []
    position = 1
    for size in groups:
        out.extend((position, size > 1) for _ in range(size))
        position += size
    return out


def _rank_label(rank: int, *, tied: bool) -> str:
    """Format a string with ``rank`` right-aligned, marked ``=`` inside a tie group and ``.`` otherwise."""
    return f"{rank:>3}{'=' if tied else '.'}"


def _refit(session: Session, console: Console, cache: PosteriorCache) -> LaplacePosterior:
    """Refit from the session's observations, saying what the respondent should know."""
    posterior, status = cache.fit(session.observations(), len(session.active), session.cfg)
    if status is FitStatus.NOT_CONVERGED:
        console.say("  (warning: the model fit did not converge; treat rank intervals with care)")
    elif status is FitStatus.STALE:
        console.say("  (warning: the model could not be refitted; your last answer is kept but not yet reflected)")
    return posterior


def _bucket_stage(session: Session, console: Console, *, resume: bool = False) -> Next:
    """Run the bucket stage. Returns PAIRS or QUIT."""
    total = len(session.roster)
    if not resume:
        console.say("\nBuckets: how do you tier characters?")
        console.say_all(_BUCKET_HELP)
    while (char_id := session.next_unmarked()) is not None:
        character = session.roster.characters[char_id]
        prompt = f"[{len(session.marks) + 1:>3}/{total}] {character.name} ({character.game}) › "
        key = console.ask_key(prompt, ("l", "f", "m", "x", "u", "q", "?"))
        verdict = _bucket_key(session, console, char_id, key)
        if verdict is not None:
            return verdict
    return Next.PAIRS


def _bucket_key(session: Session, console: Console, char_id: int, key: str) -> Next | None:
    """Apply one bucket-pass key. Returns None to continue the pass."""
    if key == "?":
        console.say_all(_BUCKET_HELP)
    elif key == "q":
        return _quit_if_confirmed(console)
    elif key == "u":
        if session.undo() is Undone.NOTHING:
            console.say("  (nothing to undo)")
    elif key == "x":
        session.mark(char_id, None)
    else:
        session.mark(char_id, _BUCKET_KEYS[key])
    return None


def _status_lines(posterior: LaplacePosterior, session: Session) -> list[str]:
    """Build the pair counter and the model's current top by closed-form expected rank."""
    order = expected_order(posterior.mu, posterior.covariance())
    top_names = [session.character(i).name for i in order[:8]]
    return [
        f"— pair {session.screens_answered + 1}",
        "  current top: " + " · ".join(top_names),
    ]


def _pairs_stage(session: Session, console: Console, cache: PosteriorCache, *, auto_stop: bool, resume: bool = False) -> Next:
    """Run the pairs stage. Returns REVIEW, BUCKETS (cross-boundary undo), or QUIT."""
    policy = ELRPolicy(session.selection)
    if not resume:
        console.say("\nPairs: which of the two characters do you prefer?")
        console.say_all(_PAIR_HELP)
    elif not auto_stop:
        console.say("\n(Back to pairs. Press d to recompile rankings and show the new result.)")
    while True:
        if session.exhausted():
            console.say("\nEvery pair has been asked. Here is the list!")
            return Next.REVIEW
        posterior = _refit(session, console, cache)
        items, utility = policy.propose(posterior, session.constraints(), session.rng(Stream.PROPOSAL))
        if auto_stop and session.should_stop(utility):
            console.say("\nThat's enough to build a list. Here it is!")
            return Next.REVIEW
        display = items[session.rng(Stream.DISPLAY).permutation(2)]
        shown = (int(display[0]), int(display[1]))
        for line in _status_lines(posterior, session):
            console.say(line)
        for position, local in enumerate(shown, start=1):
            character = session.character(local)
            console.say(f"  {position}. {character.name} ({character.game})")
        verdict = _pair_prompt(session, console, shown, utility)
        if verdict is not Next.PAIRS:
            return verdict


_ANSWER_KEYS = ("1", "2", "t", "s")


def _answer_of(line: str, shown: tuple[int, int]) -> ScreenPick | ScreenTie | None:
    """Translate an answer key into its corresponding observation."""
    if line == "t":
        return ScreenTie(shown)
    if line == "s":
        return None
    return ScreenPick(shown, shown[int(line) - 1])


def _pair_prompt(session: Session, console: Console, shown: tuple[int, int], price: float) -> Next:
    """Read one pair's answer."""
    while True:
        line = console.ask(_PAIR_PROMPT).lower()
        if line in _ANSWER_KEYS:
            session.record(shown, _answer_of(line, shown), price)
            return Next.PAIRS
        verdict = _pair_command(session, console, line)
        if verdict is not None:
            return verdict


def _pair_command(session: Session, console: Console, line: str) -> Next | None:
    """Handle a non-answer key. Returns None to ask the same pair again."""
    if line == "q":
        return _quit_if_confirmed(console)
    if line == "d":
        return Next.REVIEW
    if line == "u":
        if session.undo() is Undone.PAIR:
            return Next.PAIRS
        console.say("  (back to the bucket stage)")
        return Next.BUCKETS
    _help_or_hint(console, line, _PAIR_HELP, "  (answer one of 1/2/t/s/u/d/q; ? explains the options)")
    return None


def _help_or_hint(console: Console, line: str, help_lines: Sequence[str], hint: str) -> None:
    """Answer a key that doesn't correspond to a selection. ``?`` shows ``help_lines``, and anything else gets ``hint``."""
    if line == "?":
        console.say_all(help_lines)
    else:
        console.say(hint)


REVIEW_SAMPLES = 512
"""Posterior draws behind the review's rank intervals."""


@dataclass(frozen=True)
class _ReviewList:
    """The model's list at review entry in roster IDs."""

    order: list[int]
    """Strict order; best first."""
    ranks: list[tuple[int, bool]]
    """Competition rank and tie flag per listed item, parallel to ``order``."""
    interval: list[tuple[int, int]]
    """1-based 80% rank interval per listed item, parallel to ``order``."""
    unranked: list[int]
    """Characters marked "don't know"."""


def _build_review(session: Session, posterior: LaplacePosterior) -> _ReviewList:
    """Build the weak order from closed-form pairwise probabilities and compute intervals from draws."""
    active = session.active
    order, groups = final_order(posterior.mu, posterior.covariance())
    summary = rank_summary(posterior.sample(session.rng(Stream.REVIEW), REVIEW_SAMPLES))
    listed = [int(i) for i in order]
    return _ReviewList(
        order=[active[i] for i in listed],
        ranks=ranks_of([len(g) for g in groups]),
        interval=[display_interval(float(summary.rank_lo[i]), float(summary.rank_hi[i])) for i in listed],
        unranked=session.excluded,
    )


def _render_review(console: Console, session: Session, review: _ReviewList) -> None:
    """Print the list and the not-ranked section."""
    roster = session.roster
    console.say(f"\n{len(review.order)} ranked (80% rank interval):")
    for (rank, tied), char_id, (lo, hi) in zip(review.ranks, review.order, review.interval, strict=True):
        console.say(f"  {_rank_label(rank, tied=tied)} {roster.characters[char_id].name} ({lo}-{hi})")
    if review.unranked:
        console.say("  — not ranked —")
        for char_id in review.unranked:
            console.say(f"      {roster.characters[char_id].name}")


def _results_path(directory: Path) -> Path:
    """Return the first unused ``results-N.txt`` in ``directory``."""
    n = 1
    while (path := directory / f"results-{n}.txt").exists():
        n += 1
    return path


def _export(session: Session, review: _ReviewList) -> Path:
    """Write the list to a fresh ``results-N.txt`` in the working directory."""
    roster = session.roster
    lines = [f"{roster.franchise} personal ranking", ""]
    for char_id, (rank, tied) in zip(review.order, review.ranks, strict=True):
        lines.append(f"{_rank_label(rank, tied=tied)} {roster.characters[char_id].name}")
    if review.unranked:
        lines += ["", "Not ranked:"]
        lines += [f"  - {roster.characters[c].name}" for c in review.unranked]
    path = _results_path(Path())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save(session: Session, console: Console, review: _ReviewList) -> Next | None:
    """Export and finish."""
    try:
        path = _export(session, review)
    except OSError as error:
        console.say(f"  (could not save: {error})")
        return None
    console.say(f"Saved to {path}")
    return Next.DONE


def _review_stage(session: Session, console: Console, cache: PosteriorCache, *, resume: bool = False) -> Next:
    """Show the list and take commands. Returns PAIRS or DONE."""
    review = _build_review(session, _refit(session, console, cache))
    _render_review(console, session, review)
    if not resume:
        console.say_all(_REVIEW_HELP)
    while True:
        command = console.ask(_REVIEW_PROMPT).lower()
        if command == "done":
            verdict = _save(session, console, review) if _confirm(console, _SAVE_PROMPT) else Next.DONE
        elif command == "b":
            verdict = _leave_for_pairs(session, console)
        else:
            _help_or_hint(console, command, _REVIEW_HELP, "  (answer done or b; ? explains the options)")
            continue
        if verdict is not None:
            return verdict


def _leave_for_pairs(session: Session, console: Console) -> Next | None:
    """Return to the pairs stage for more questions. Returns None when none are left."""
    if session.exhausted():
        console.say("  (every pair has been asked)")
        return None
    return Next.PAIRS


def _roster(text: str) -> Roster:
    """Parse ``--roster``."""
    try:
        return load_roster(Path(text))
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _seed(text: str) -> int:
    """Parse ``--seed``."""
    if not text.isdecimal():
        msg = f"the seed must be a non-negative integer, not {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(text)


def _threshold(text: str) -> float:
    """Parse ``--stop-threshold``."""
    value = float(text)
    if math.isnan(value):
        msg = "the stop threshold must be a number, not NaN"
        raise argparse.ArgumentTypeError(msg)
    return value


def run(argv: Sequence[str] | None = None, console: Console | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=_roster, default=None, help="roster JSON data (default: Windows-era Touhou)")
    parser.add_argument("--seed", type=_seed, default=0, help="session seed, a non-negative integer (default: 0)")
    parser.add_argument(
        "--stop-threshold",
        type=_threshold,
        default=DEFAULT_STOP_THRESHOLD,
        help=f"loss-reduction price below which the session auto-stops (default: {DEFAULT_STOP_THRESHOLD}); "
        + "at or below 0, only d ends the pairs stage",
    )
    args = parser.parse_args(argv)
    active_console = console if console is not None else Console()
    roster = args.roster if args.roster is not None else load_roster()
    session = Session(roster=roster, seed=args.seed, stop_threshold=args.stop_threshold)
    active_console.say(f"New session for {roster.franchise} ({len(roster)} characters).")
    try:
        return _run_stages(session, active_console)
    except KeyboardInterrupt, EOFError:
        active_console.say("\nInterrupted; nothing was saved.")
        return 130


def _run_stages(session: Session, console: Console) -> int:
    """Drive the stages."""
    cache = PosteriorCache()
    stage = Next.BUCKETS
    entered_buckets = False
    entered_pairs = False
    reviewed = False
    while True:
        if stage is Next.BUCKETS:
            stage = _bucket_stage(session, console, resume=entered_buckets)
            entered_buckets = True
            if stage is Next.PAIRS and len(session.active) < PAIR_ITEMS:
                console.say(f"Only {len(session.active)} characters bucketed; at least {PAIR_ITEMS} are needed for pairs.")
                return 1
        elif stage is Next.PAIRS:
            stage = _pairs_stage(session, console, cache, auto_stop=not reviewed, resume=entered_pairs)
            entered_pairs = True
        elif stage is Next.REVIEW:
            stage = _review_stage(session, console, cache, resume=reviewed)
            reviewed = True
        elif stage is Next.QUIT:
            console.say("Session discarded.")
            return 0
        else:
            return 0


if __name__ == "__main__":
    sys.exit(run())
