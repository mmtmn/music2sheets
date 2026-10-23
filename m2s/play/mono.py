"""Monophonic instruments that breathe (voice, saxophone): one note at a time, range, breaths."""

from __future__ import annotations

from m2s.ir import Track
from m2s.play.piano import Violation
from m2s.play.preset import load_preset


class MonoDecoder:
    def __init__(self, preset: str):
        self.p = load_preset(preset)

    def decode(self, track: Track) -> list[Violation]:
        for n in track.notes:
            n.keep = n.confidence >= 0.5
        notes = sorted((n for n in track.notes if n.keep), key=lambda n: n.onset)
        # one note at a time: an overlap keeps the more confident note and trims the other
        for a, b in zip(notes, notes[1:]):
            if b.onset < a.offset - 0.02:
                if abs(b.onset - a.onset) < 0.05 and b.confidence > a.confidence:
                    a.keep = False
                else:
                    a.offset = b.onset
        notes = [n for n in notes if n.keep]
        out: list[Violation] = []
        lo, hi = self.p["range"]
        for n in notes:
            if not lo <= n.pitch <= hi:
                out.append(Violation("range", n.onset, f"pitch {n.pitch} outside {lo}-{hi}", [n.id]))
        if "voice_types" in self.p and notes:
            ps = sorted(n.pitch for n in notes)
            core = ps[len(ps) // 10: max(len(ps) * 9 // 10, 1)]
            fit = {k: sum(r[0] <= p <= r[1] for p in core) / len(core) for k, r in self.p["voice_types"].items()}
            vt = max(fit, key=fit.get)
            for n in notes:
                n.effector = {"voice_type": vt}
        br = self.p.get("breath", {})
        if br and notes:
            start = notes[0].onset
            for a, b in zip(notes, notes[1:] + [None]):
                gap = (b.onset - a.offset) if b is not None else 99
                if gap >= br["min_gap"]:
                    if a.offset - start > br["max_phrase"]:
                        out.append(Violation("breath", start, f"{a.offset - start:.1f}s without a breath", [a.id]))
                    if b is not None:
                        a.tags.append("breath")
                        start = b.onset
        return out
