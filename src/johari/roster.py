"""Roster loading from JSON."""

import json
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Character:
    """One rankable character."""

    name: str
    game: str
    """Debut-game tag (e.g. ``TH06``) shown alongside the name."""


@dataclass(frozen=True)
class Roster:
    """An ordered character list."""

    franchise: str
    characters: tuple[Character, ...]

    def __len__(self) -> int:
        """Return the roster size."""
        return len(self.characters)


def _parse(payload: object, source: str) -> Roster:
    if not isinstance(payload, dict):
        msg = f"{source}: roster must be a JSON object"
        raise ValueError(msg)

    franchise = payload.get("franchise", "")
    entries = payload.get("characters")
    if not isinstance(franchise, str) or not isinstance(entries, list) or not entries:
        msg = f"{source}: expected a 'franchise' string and a non-empty 'characters' list"
        raise ValueError(msg)

    characters: list[Character] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            msg = f"{source}: characters[{position}] is not an object"
            raise ValueError(msg)

        name, game = entry.get("name"), entry.get("game")
        if not isinstance(name, str) or not name or not isinstance(game, str):
            msg = f"{source}: characters[{position}] needs 'name' and 'game' strings"
            raise ValueError(msg)

        characters.append(Character(name=name, game=game))

    names = [c.name for c in characters]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        msg = f"{source}: duplicate character names {duplicates}"
        raise ValueError(msg)

    return Roster(franchise=franchise, characters=tuple(characters))


def load_roster(path: Path | None = None) -> Roster:
    """Load a roster file, defaulting to the packaged roster.

    Raises ``OSError`` when the file cannot be read and ``ValueError`` when
    its content is not a roster, invalid JSON included.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file content is not a valid roster.
    """
    if path is None:
        text = (resources.files("johari") / "data" / "touhou_windows.json").read_text(encoding="utf-8")
        source = "packaged touhou_windows.json"
    else:
        text = path.read_text(encoding="utf-8")
        source = str(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"{source}: not valid JSON ({error})"
        raise ValueError(msg) from error
    return _parse(payload, source)
