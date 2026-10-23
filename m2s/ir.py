"""The shared note format (IR).

Every stage reads and writes these records. Stages add fields; none remove
them. Times are in seconds until the notation stage fills the score-time
fields (measure, beat, qdur, voice, staff, spelling).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Note:
    id: str
    onset: float
    offset: float
    pitch: int
    velocity: int = 80
    confidence: float = 1.0
    cents: float = 0.0
    keep: bool = True
    # which models saw this note, and how strongly (model name -> score)
    support: dict[str, float] = field(default_factory=dict)
    # who plays it: {"hand": "L"/"R", "finger": 1-5} | {"string": 1-6, "fret": n} | {"limb": "RH"}
    effector: dict[str, Any] = field(default_factory=dict)
    player: int = 0
    tags: list[str] = field(default_factory=list)
    # pitch curve for bends / slides / vibrato: list of (seconds, cents relative to pitch)
    bend: list[tuple[float, float]] = field(default_factory=list)
    # score time, filled by notate
    measure: int | None = None
    beat: float | None = None
    qdur: float | None = None
    value: float | None = None     # written value in quarters (set by the Rust notation stage)
    voice: int | None = None
    staff: int | None = None
    spelling: str | None = None
    provenance: str = ""

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Note":
        d = dict(d)
        d["bend"] = [tuple(x) for x in d.get("bend", [])]
        return Note(**d)


@dataclass
class Track:
    """One part: an instrument played by one or more players."""

    name: str
    preset: str
    notes: list[Note] = field(default_factory=list)
    # sustain pedal intervals (seconds), piano only
    pedal: list[tuple[float, float]] = field(default_factory=list)
    stem: str = "mix"
    players: int = 1

    def kept(self) -> list[Note]:
        return [n for n in self.notes if n.keep]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Track":
        d = dict(d)
        d["notes"] = [Note.from_dict(n) for n in d.get("notes", [])]
        d["pedal"] = [tuple(p) for p in d.get("pedal", [])]
        return Track(**d)


@dataclass
class Piece:
    tracks: list[Track] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def track(self, name: str) -> Track:
        for t in self.tracks:
            if t.name == name:
                return t
        raise KeyError(name)

    def all_notes(self) -> Iterable[tuple[Track, Note]]:
        for t in self.tracks:
            for n in t.notes:
                yield t, n

    def note_index(self) -> dict[str, tuple[Track, Note]]:
        return {n.id: (t, n) for t, n in self.all_notes()}

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "tracks": [asdict(t) for t in self.tracks]}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Piece":
        return Piece(tracks=[Track.from_dict(t) for t in d.get("tracks", [])], meta=d.get("meta", {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @staticmethod
    def load(path: str | Path) -> "Piece":
        return Piece.from_dict(json.loads(Path(path).read_text()))


def assign_ids(notes: list[Note], prefix: str) -> None:
    """Stable ids ordered by time then pitch, e.g. ``pno-000123``."""
    notes.sort(key=lambda n: (n.onset, n.pitch))
    for i, n in enumerate(notes):
        n.id = f"{prefix}-{i:06d}"
