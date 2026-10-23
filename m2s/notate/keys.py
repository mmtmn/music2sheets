"""Key estimation from notes (Krumhansl-Kessler profiles, duration-weighted)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from m2s.ir import Note

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# tonic pitch class -> line-of-fifths position of the conventional tonic spelling
MAJOR_TONIC_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
MINOR_TONIC_FIFTHS = {9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 8: 8, 3: -3, 2: 2, 7: 1, 0: 0, 5: -1, 10: -2}


@dataclass
class Key:
    tonic_pc: int
    mode: str        # "major" | "minor"
    score: float

    @property
    def tonic_fifths(self) -> int:
        return (MAJOR_TONIC_FIFTHS if self.mode == "major" else MINOR_TONIC_FIFTHS)[self.tonic_pc]

    @property
    def sharps(self) -> int:
        """Key signature: + sharps / - flats."""
        return self.tonic_fifths - (0 if self.mode == "major" else 3)

    @property
    def centre(self) -> float:
        """Centre of the key on the line of fifths (minor leans sharp for its raised 6th and 7th)."""
        return self.tonic_fifths + (2.0 if self.mode == "major" else -0.5)

    def name(self) -> str:
        from m2s.notate.spell import fifths_name

        return f"{fifths_name(self.tonic_fifths)} {self.mode}"


def histogram(notes: list[Note], bass_weight: float = 0.5) -> np.ndarray:
    h = np.zeros(12)
    if not notes:
        return h
    lo = np.percentile([n.pitch for n in notes], 15)
    for n in notes:
        w = min(n.duration, 2.0) + 0.05
        if n.pitch <= lo:
            w *= 1 + bass_weight
        h[n.pitch % 12] += w
    return h


def estimate_key(notes: list[Note], bass: list[Note] | None = None) -> Key:
    h = histogram(notes)
    best = Key(0, "major", -2.0)
    for pc in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = float(np.corrcoef(h, np.roll(prof, pc))[0, 1]) if h.sum() else 0.0
            if r > best.score:
                best = Key(pc, mode, r)
    return _relative_by_bass(best, bass or _low_notes(notes))


def _low_notes(notes: list[Note]) -> list[Note]:
    if not notes:
        return []
    lo = np.percentile([n.pitch for n in notes], 20)
    return [n for n in notes if n.pitch <= lo]


def _relative_by_bass(k: Key, bass: list[Note]) -> Key:
    """Relative major and minor share their notes; the bass decides which tonic is home:
    more time on it, and the first and last bass notes."""
    if not bass:
        return k
    rel = Key((k.tonic_pc + 9) % 12, "minor", k.score) if k.mode == "major" else \
        Key((k.tonic_pc + 3) % 12, "major", k.score)
    order = sorted(bass, key=lambda n: n.onset)

    def weight(pc: int) -> float:
        w = sum(min(n.duration, 2.0) for n in bass if n.pitch % 12 == pc)
        w += 2.0 * (order[0].pitch % 12 == pc) + 3.0 * (order[-1].pitch % 12 == pc)
        return w

    return rel if weight(rel.tonic_pc) > 1.25 * weight(k.tonic_pc) else k
