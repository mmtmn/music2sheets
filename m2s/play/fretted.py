"""Fretted and bowed strings (guitar, bass, violin): string/fret assignment by Viterbi.

State per time slice = one fingering (a string and fret for every note).
Within a slice the fretting hand must cover every fretted note (stretch);
between slices it pays for position shifts, scaled by how fast they must
happen, plus string skips. Open strings are cheap. Players > 1 (twin guitars
recorded as one bus) are split into lines first, like a piano's two hands.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

from m2s.ir import Note, Track
from m2s.play.piano import Violation, make_slices
from m2s.play.preset import load_preset

Fingering = tuple[tuple[int, int], ...]  # per note: (string index low->high, fret)


@dataclass
class FretWeights:
    stretch: float = 0.4
    stretch_violation: float = 30.0
    position: float = 0.03         # mild preference for lower positions
    must_keep: float = 0.6
    must_drop: float = 0.3
    max_fingerings: int = 24


class FrettedDecoder:
    def __init__(self, preset: str, weights: FretWeights | None = None):
        self.p = load_preset(preset)
        self.f = self.p["fretting"]
        self.w = weights or FretWeights()
        self.tuning = self.p["tuning"]

    # --------------------------------------------------------------- fingerings
    def options(self, pitches: list[int]) -> list[Fingering]:
        per_note = []
        for p in pitches:
            opts = [(s, p - o) for s, o in enumerate(self.tuning) if 0 <= p - o <= self.p["frets"]]
            per_note.append(opts or [(0, max(0, p - self.tuning[0]))])
        out = []
        for combo in itertools.product(*per_note):
            strings = [s for s, _ in combo]
            if len(set(strings)) < len(strings):
                continue
            if self.p.get("adjacent_strings_only") and len(strings) > 1 and max(strings) - min(strings) > len(strings) - 1:
                continue
            out.append(tuple(combo))
        if not out:  # more notes than strings: keep the best partial, it is flagged later
            out = [tuple(per_note[i][0] for i in range(len(pitches)))]
        out.sort(key=self.chord_cost)
        return out[: self.w.max_fingerings]

    def stretch_limit(self, low_fret: int) -> int:
        return self.f["max_stretch_high"] if low_fret >= self.f["high_position"] else self.f["max_stretch"]

    def chord_cost(self, fg: Fingering) -> float:
        fretted = [f for _, f in fg if f > 0]
        cost = self.w.position * (min(fretted) if fretted else 0)
        cost -= self.f["open_string_bonus"] * sum(1 for _, f in fg if f == 0)
        if fretted:
            st = max(fretted) - min(fretted)
            lim = self.stretch_limit(min(fretted))
            cost += self.w.stretch * st
            if st > lim:
                cost += self.w.stretch_violation + 3 * (st - lim)
        if len({s for s, _ in fg}) < len(fg):
            cost += 50.0
        return cost

    def position(self, fg: Fingering, prev: float | None) -> float:
        fretted = [f for _, f in fg if f > 0]
        if not fretted:
            return prev if prev is not None else 0.0
        return float(min(fretted))

    def trans_cost(self, a: Fingering, pa: float, b: Fingering, pb: float, dt: float) -> float:
        shift = abs(pb - pa)
        cost = self.f["shift_cost"] * shift
        dt = max(dt, 1e-3)
        if shift > 2 and shift / dt > self.f["shift_speed"]:
            cost += 15.0
        if len(a) == 1 and len(b) == 1:
            skip = abs(a[0][0] - b[0][0])
            if skip > 1:
                cost += self.f["string_skip_cost"] * (skip - 1)
        return cost

    # ------------------------------------------------------------------ decode
    def decode(self, track: Track) -> list[Violation]:
        w = self.w
        for n in track.notes:
            n.keep = n.confidence >= 0.5
            if n.confidence < w.must_drop:
                n.keep = False
        players = int(self.f.get("players", 1)) if track.players <= 1 else track.players
        notes = [n for n in track.notes if n.keep]
        if self.p.get("max_simultaneous"):
            self._limit_polyphony(notes, int(self.p["max_simultaneous"]))
            notes = [n for n in notes if n.keep]
        lines = split_players(notes, players) if players > 1 else [notes]
        out: list[Violation] = []
        for pi, line in enumerate(lines):
            for n in line:
                n.player = pi
            out += self._decode_line(line, pi)
        return out

    def _limit_polyphony(self, notes: list[Note], k: int) -> None:
        for sl in make_slices(notes, 0.03):
            if len(sl) > k:
                for n in sorted(sl, key=lambda n: -n.confidence)[k:]:
                    n.keep = False
                    n.tags.append("over-polyphony")

    def _decode_line(self, notes: list[Note], player: int) -> list[Violation]:
        slices = make_slices(notes, 0.03)
        if not slices:
            return []
        opts = [self.options([n.pitch for n in s]) for s in slices]
        cost = [self.chord_cost(fg) for fg in opts[0]]
        pos = [self.position(fg, None) for fg in opts[0]]
        back: list[list[int]] = []
        for i in range(1, len(slices)):
            dt = slices[i][0].onset - slices[i - 1][0].onset
            nc, npos, bp = [], [], []
            for fb in opts[i]:
                within = self.chord_cost(fb)
                best, arg, bpos = math.inf, 0, 0.0
                for j, fa in enumerate(opts[i - 1]):
                    pb = self.position(fb, pos[j])
                    c = cost[j] + self.trans_cost(fa, pos[j], fb, pb, dt)
                    if c < best:
                        best, arg, bpos = c, j, pb
                nc.append(best + within)
                npos.append(bpos)
                bp.append(arg)
            cost, pos = nc, npos
            back.append(bp)
        k = min(range(len(cost)), key=cost.__getitem__)
        path = [k]
        for bp in reversed(back):
            k = bp[k]
            path.append(k)
        path.reverse()
        out: list[Violation] = []
        nstr = len(self.tuning)
        prev_pos, prev_t = None, None
        for s, o, k in zip(slices, opts, path):
            fg = o[k]
            for n, (si, fret) in zip(s, fg):
                # tab convention: string 1 is the highest-pitched string
                n.effector = {"string": nstr - si, "fret": fret, "player": player}
            fretted = [f for _, f in fg if f > 0]
            if fretted:
                st = max(fretted) - min(fretted)
                if st > self.stretch_limit(min(fretted)):
                    out.append(Violation("stretch", s[0].onset, f"{st}-fret stretch", [n.id for n in s],
                                         f"player {player}"))
                p = min(fretted)
                if prev_pos is not None and abs(p - prev_pos) > 2:
                    dt = max(s[0].onset - prev_t, 1e-3)
                    if abs(p - prev_pos) / dt > self.f["shift_speed"]:
                        out.append(Violation("shift", s[0].onset, f"shift {abs(p - prev_pos):.0f} frets in "
                                             f"{dt * 1000:.0f} ms", [n.id for n in s], f"player {player}"))
                prev_pos, prev_t = p, s[0].onset
            if len(s) > nstr:
                out.append(Violation("count", s[0].onset, f"{len(s)} notes on {nstr} strings", [n.id for n in s]))
        return out


def split_players(notes: list[Note], players: int) -> list[list[Note]]:
    """Split a bus into lines: in chords the top note goes to player 1, the next to player 2;
    single notes follow whichever line they continue most smoothly."""
    lines: list[list[Note]] = [[] for _ in range(players)]
    last: list[float | None] = [None] * players
    for sl in make_slices(notes, 0.03):
        sl = sorted(sl, key=lambda n: -n.pitch)
        if len(sl) >= players:
            for i, n in enumerate(sl):
                li = min(i, players - 1)
                lines[li].append(n)
                last[li] = n.pitch
        else:
            for n in sl:
                li = min(range(players), key=lambda i: abs(n.pitch - last[i]) if last[i] is not None else 0)
                lines[li].append(n)
                last[li] = n.pitch
    return lines


def tab_summary(track: Track) -> dict[str, Any]:
    frets = [n.effector.get("fret", 0) for n in track.kept()]
    return {"notes": len(frets), "max_fret": max(frets, default=0),
            "open": sum(1 for f in frets if f == 0)}
