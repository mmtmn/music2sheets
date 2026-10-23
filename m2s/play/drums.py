"""Drum kit: limb assignment (two hands, two feet) by dynamic programming.

Feet take the kick and hi-hat pedal (fast kicks alternate feet when the
preset allows a double pedal). Hands take everything else: the cymbal hand
leads, the other hand takes the snare when both play together, and fast
single-surface runs alternate hands. Hits faster than one hand can repeat,
or more than four at once, are violations.
"""

from __future__ import annotations

import itertools

from m2s.ir import Note, Track
from m2s.play.piano import Violation, make_slices
from m2s.play.preset import load_preset

CYMBALS = {42, 46, 49, 51, 52, 55, 57, 59}


class DrumDecoder:
    def __init__(self, preset: str = "drums"):
        self.p = load_preset(preset)
        self.lim = self.p["limbs"]
        self.limbs_for = {v["midi"]: v["limbs"] for v in self.p["kit"].values()}

    def decode(self, track: Track) -> list[Violation]:
        for n in track.notes:
            n.keep = n.confidence >= 0.5
        hits = [n for n in track.notes if n.keep]
        slices = make_slices(hits, 0.025)
        min_gap = 1.0 / self.lim["single_stroke_rate"]
        last = {"RH": -9.0, "LH": -9.0, "RF": -9.0, "LF": -9.0}
        last_kick_foot = "LF"
        out: list[Violation] = []
        for sl in slices:
            t = sl[0].onset
            feet = [n for n in sl if set(self.limbs_for.get(n.pitch, ["RH"])) <= {"RF", "LF"}]
            hands = [n for n in sl if n not in feet]
            used: set[str] = set()
            for n in feet:
                if n.pitch == 44:
                    limb = "LF"
                elif self.lim.get("double_kick") and t - last["RF"] < 1.5 * min_gap and last_kick_foot == "RF":
                    limb = "LF"
                else:
                    limb = "RF"
                if limb in used:
                    limb = "LF" if limb == "RF" else "RF"
                if n.pitch != 44:
                    last_kick_foot = limb
                n.effector = {"limb": limb}
                used.add(limb)
                last[limb] = t
            # hands: try both orders, keep the one with fewer too-fast repeats and no crossing
            best, best_cost = None, 1e9
            avail = [h for h in ("RH", "LH") if h not in used]
            for perm in itertools.permutations(avail, min(len(hands), len(avail))):
                cost = 0.0
                for n, h in zip(sorted(hands, key=lambda n: (n.pitch not in CYMBALS, n.pitch)), perm):
                    if t - last[h] < min_gap:
                        cost += 5.0
                    if n.pitch in CYMBALS and h == "LH":
                        cost += 0.5
                    if n.pitch == 38 and h == "RH" and len(hands) > 1:
                        cost += 0.5
                    cost -= 0.1 * min(t - last[h], 1.0)
                if cost < best_cost:
                    best, best_cost = perm, cost
            ordered = sorted(hands, key=lambda n: (n.pitch not in CYMBALS, n.pitch))
            for n, h in zip(ordered, best or ()):
                n.effector = {"limb": h}
                if t - last[h] < min_gap:
                    out.append(Violation("speed", t, f"{h} repeats in {(t - last[h]) * 1000:.0f} ms", [n.id], h))
                last[h] = t
            for n in ordered[len(best or ()):]:
                n.effector = {"limb": "?"}
            if len(sl) > self.lim["max_simultaneous"] or len(ordered) > len(best or ()):
                out.append(Violation("count", t, f"{len(sl)} hits at once", [n.id for n in sl]))
        return out
