"""Piano playability decoder.

Two passes, as in the plan:

1. **Hands.** A beam search over time slices (chords) decides, for every
   candidate note, whether it is kept and which hand plays it. The cost adds
   acoustic evidence (-log p of the keep/drop choice) to kinematic effort
   (span, required shift and its speed, crossings, line continuity).
2. **Fingers.** Within each hand, a Viterbi pass over finger tuples, with
   finger-pair spans adapted from Parncutt et al. (1997).

Both passes are deterministic. Every hard-limit breach that survives is
returned as a violation with its time, so review can see where and why.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from m2s.ir import Note, Track
from m2s.play.preset import load_preset

BLACK = {1, 3, 6, 8, 10}


@dataclass
class PianoWeights:
    """Decoder weights. Defaults were fit on the ASAP dev set with tools/tune_hands.py
    (hand accuracy: 96.2% on tuning pieces, 92.1% held out; a middle-C split gets 80.8%)."""

    acoustic: float = 1.0
    span_over_comfort: float = 1.2
    span_violation: float = 40.0
    count_violation: float = 40.0
    leap: float = 0.06
    speed: float = 0.0225
    speed_violation: float = 25.0
    stream: float = 0.9
    crossing: float = 3.0
    crossing_per_semitone: float = 0.6
    pitch_prior: float = 0.01
    middle: float = 56.0
    hold_conflict: float = 0.5
    # a hand whose fingers are still holding keys is less free to take new notes
    busy: float = 0.6
    # same-hand note rate: penalty when one hand strikes again within `rate_window` s
    rate: float = 0.375
    rate_window: float = 0.06
    # prefer the idle hand over a wide chord in the busy one
    idle: float = 0.4
    # melodic continuity window (semitones)
    stream_window: float = 9.0
    # candidates above `must_keep` are always kept, below `must_drop` always dropped;
    # in between the decoder chooses
    must_keep: float = 0.75
    must_drop: float = 0.25
    beam: int = 48


@dataclass(frozen=True)
class HandState:
    lo: int
    hi: int
    t: float
    # notes the hand is still holding: (pitch, offset)
    held: tuple[tuple[int, float], ...] = ()


@dataclass
class Node:
    cost: float
    lh: HandState | None
    rh: HandState | None
    parent: "Node | None"
    decision: tuple[str, ...]  # per note in the slice: "L", "R" or "x" (dropped)
    prev: tuple[tuple[int, str], ...]  # (pitch, hand) of the previous slice's kept notes
    prev_t: float


@dataclass
class Violation:
    kind: str
    time: float
    detail: str
    note_ids: list[str] = field(default_factory=list)
    hand: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "time": round(self.time, 3), "detail": self.detail, "notes": self.note_ids,
                "hand": self.hand}


def make_slices(notes: list[Note], window: float) -> list[list[Note]]:
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    slices: list[list[Note]] = []
    for n in notes:
        if slices and n.onset - slices[-1][0].onset <= window:
            slices[-1].append(n)
        else:
            slices.append([n])
    for s in slices:
        s.sort(key=lambda n: n.pitch)
    return slices


def pedal_down(pedal: list[tuple[float, float]], t: float) -> bool:
    return any(a <= t < b for a, b in pedal)


class PianoDecoder:
    def __init__(self, preset: str = "piano", weights: PianoWeights | None = None):
        self.p = load_preset(preset)
        self.h = self.p["hands"]
        self.w = weights or PianoWeights()

    # ------------------------------------------------------------------ hands
    def _hand_cost(self, hs: HandState | None, chord: list[int], t: float, hand: str,
                   pedal: bool) -> tuple[float, list[str], HandState]:
        """Cost for one hand to strike ``chord`` (sorted pitches) at time ``t``."""
        h, w = self.h, self.w
        cost = 0.0
        flags: list[str] = []
        lo, hi = chord[0], chord[-1]
        span = hi - lo
        if len(chord) > h["max_notes"]:
            cost += w.count_violation * (len(chord) - h["max_notes"])
            flags.append(f"{len(chord)} notes in one hand")
        if span > h["max_span"]:
            cost += w.span_violation + 4.0 * (span - h["max_span"])
            flags.append(f"span {span} semitones")
        elif span > h["comfort_span"]:
            cost += w.span_over_comfort * (span - h["comfort_span"])

        # weak prior: right hand above the middle, left below
        mid = w.middle
        centre = (lo + hi) / 2
        if hand == "R" and centre < mid - 7:
            cost += w.pitch_prior * (mid - 7 - centre)
        if hand == "L" and centre > mid + 7:
            cost += w.pitch_prior * (centre - mid - 7)

        held: tuple[tuple[int, float], ...] = ()
        if hs is not None:
            dt = max(t - hs.t, 1e-3)
            union = max(hi, hs.hi) - min(lo, hs.lo)
            shift = max(0, union - h["comfort_span"])
            if shift:
                cost += w.leap * shift
                speed = shift / dt
                if speed > h["comfort_speed"]:
                    cost += w.speed * (speed - h["comfort_speed"])
                if speed > h["max_speed"] and shift > h["max_span"] - h["comfort_span"]:
                    cost += w.speed_violation
                    flags.append(f"jump {shift} st in {dt * 1000:.0f} ms")
            if dt < w.rate_window:
                cost += w.rate * (w.rate_window - dt) / w.rate_window
            # keys this hand's fingers still hold must fit with the new chord, and they tie the hand up
            still = [(p, off) for p, off in hs.held if off > t + 0.03 and p not in chord]
            if still:
                allp = [p for p, _ in still] + chord
                over = (max(allp) - min(allp)) - h["max_span"]
                if over > 0:
                    cost += w.hold_conflict * over
                cost += w.busy * min(len(still), 3)
                held = tuple(still)
        return cost, flags, HandState(lo, hi, t, held)

    def decode(self, track: Track) -> list[Violation]:
        """Assign ``keep`` and ``effector['hand']`` for every note of ``track``."""
        w = self.w
        cands = [n for n in track.notes if n.confidence >= w.must_drop]
        for n in track.notes:
            if n.confidence < w.must_drop:
                n.keep = False
                n.effector = {}
        slices = make_slices(cands, self.h["chord_window"])
        beam: list[Node] = [Node(0.0, None, None, None, (), (), -1.0)]

        for sl in slices:
            t = sl[0].onset
            ped = pedal_down(track.pedal, t)
            optional = [i for i, n in enumerate(sl) if n.confidence < w.must_keep]
            if len(optional) > 3:  # keep the search small: decide the rest by threshold
                optional = sorted(optional, key=lambda i: abs(sl[i].confidence - 0.5))[:3]
            fixed_drop = {i for i, n in enumerate(sl) if n.confidence < w.must_keep and i not in optional
                          and n.confidence < 0.5}
            options: list[tuple[list[int], float]] = []
            for mask in itertools.product((True, False), repeat=len(optional)):
                keep_idx = [i for i in range(len(sl)) if i not in fixed_drop]
                ac = 0.0
                for i, k in zip(optional, mask):
                    p = min(max(sl[i].confidence, 1e-3), 1 - 1e-3)
                    if k:
                        ac += -math.log(p)
                    else:
                        ac += -math.log(1 - p)
                        keep_idx.remove(i)
                options.append((keep_idx, w.acoustic * ac))

            expanded: dict[tuple, Node] = {}
            for node in beam:
                for keep_idx, ac in options:
                    pitches = [sl[i].pitch for i in keep_idx]
                    m = len(pitches)
                    for k in range(m + 1):
                        lchord, rchord = pitches[:k], pitches[k:]
                        cost = node.cost + ac
                        lh, rh = node.lh, node.rh
                        if lchord:
                            c, _, lh = self._hand_cost(node.lh, lchord, t, "L", ped)
                            lh = HandState(lh.lo, lh.hi, lh.t, lh.held + tuple(
                                (sl[i].pitch, sl[i].offset) for i in keep_idx[:k]))
                            cost += c
                        if rchord:
                            c, _, rh = self._hand_cost(node.rh, rchord, t, "R", ped)
                            rh = HandState(rh.lo, rh.hi, rh.t, rh.held + tuple(
                                (sl[i].pitch, sl[i].offset) for i in keep_idx[k:]))
                            cost += c
                        cost += self._crossing_cost(lh, rh, lchord, rchord, t)
                        cost += self._idle_cost(node, lchord, rchord, t)
                        dec = ["x"] * len(sl)
                        for j, i in enumerate(keep_idx):
                            dec[i] = "L" if j < k else "R"
                        cost += self._stream_cost(node, sl, dec, t)
                        key = (_bucket(lh), _bucket(rh), tuple(dec))
                        old = expanded.get(key)
                        if old is None or cost < old.cost:
                            prev = tuple((sl[i].pitch, dec[i]) for i in range(len(sl)) if dec[i] != "x")
                            expanded[key] = Node(cost, lh, rh, node, tuple(dec), prev, t)
            beam = sorted(expanded.values(), key=lambda n: n.cost)[: w.beam]

        # backtrack
        best = beam[0]
        decisions: list[tuple[str, ...]] = []
        node: Node | None = best
        while node is not None and node.parent is not None:
            decisions.append(node.decision)
            node = node.parent
        decisions.reverse()
        for sl, dec in zip(slices, decisions):
            for n, d in zip(sl, dec):
                if d == "x":
                    n.keep = False
                    n.effector = {}
                else:
                    n.keep = True
                    n.effector = {"hand": d}

        violations = self._hand_violations(track)
        if self.p.get("fingering", {}).get("enabled", True):
            violations += self.finger(track)
        return violations

    def _crossing_cost(self, lh: HandState | None, rh: HandState | None, lchord: list[int],
                       rchord: list[int], t: float) -> float:
        if lh is None or rh is None:
            return 0.0
        margin = self.h["crossing_margin"]
        # compare each hand's newest position with the other's most recent one, if recent
        if abs(lh.t - rh.t) > 0.6:
            return 0.0
        over = lh.hi - rh.lo - margin if (lchord or rchord) else 0
        if over > 0:
            return self.w.crossing + self.w.crossing_per_semitone * over
        return 0.0

    def _stream_cost(self, node: Node, sl: list[Note], dec: list[str], t: float) -> float:
        if not node.prev or t - node.prev_t > 0.3:
            return 0.0
        kept = [(n.pitch, d) for n, d in zip(sl, dec) if d != "x"]
        if not kept:
            return 0.0
        cost = 0.0
        for p, d in kept:
            q, hq = min(node.prev, key=lambda x: abs(x[0] - p))
            gap = abs(q - p)
            if gap <= self.w.stream_window and hq != d:
                cost += self.w.stream * (1 - gap / (self.w.stream_window + 1))
        return cost / len(kept)

    def _idle_cost(self, node: Node, lchord: list[int], rchord: list[int], t: float) -> float:
        # a wide chord in one hand while the other hand rests: the resting hand would usually help
        cost = 0.0
        for chord, other in ((lchord, node.rh), (rchord, node.lh)):
            if chord and len(chord) >= 2 and not (lchord and rchord):
                span = chord[-1] - chord[0]
                idle = other is None or t - other.t > 0.4
                if idle and span >= 7:
                    cost += self.w.idle * (span - 6) / 6
        return cost

    def _hand_violations(self, track: Track) -> list[Violation]:
        out: list[Violation] = []
        notes = [n for n in track.notes if n.keep]
        for hand in "LR":
            hn = [n for n in notes if n.effector.get("hand") == hand]
            for sl in make_slices(hn, self.h["chord_window"]):
                span = sl[-1].pitch - sl[0].pitch
                if span > self.h["max_span"]:
                    out.append(Violation("span", sl[0].onset, f"{span} semitones in one hand",
                                         [n.id for n in sl], hand))
                if len(sl) > self.h["max_notes"]:
                    out.append(Violation("count", sl[0].onset, f"{len(sl)} notes in one hand",
                                         [n.id for n in sl], hand))
            slices = make_slices(hn, self.h["chord_window"])
            for a, b in zip(slices, slices[1:]):
                dt = max(b[0].onset - a[0].onset, 1e-3)
                union = max(a[-1].pitch, b[-1].pitch) - min(a[0].pitch, b[0].pitch)
                shift = max(0, union - self.h["comfort_span"])
                if shift > self.h["max_span"] - self.h["comfort_span"] and shift / dt > self.h["max_speed"]:
                    out.append(Violation("speed", b[0].onset, f"hand jumps {shift} st in {dt * 1000:.0f} ms",
                                         [n.id for n in a + b], hand))
        return out

    # ---------------------------------------------------------------- fingers
    def finger(self, track: Track) -> list[Violation]:
        cfg = self.p["fingering"]
        spans = {tuple(int(x) for x in k.split("-")): v for k, v in cfg["spans"].items()}
        shift_time = cfg["shift_time"]
        out: list[Violation] = []
        for hand in "LR":
            hn = [n for n in track.notes if n.keep and n.effector.get("hand") == hand]
            slices = make_slices(hn, self.h["chord_window"])
            if not slices:
                continue
            states = [_finger_options(len(s), hand) for s in slices]
            # Viterbi
            costs = [[_chord_cost(s, f, hand, spans) for f in states[0]] for s in slices[:1]][0]
            back: list[list[int]] = []
            for i in range(1, len(slices)):
                a, b = slices[i - 1], slices[i]
                dt = b[0].onset - a[0].onset
                new_costs, bp = [], []
                for fb in states[i]:
                    within = _chord_cost(b, fb, hand, spans)
                    best, arg = math.inf, 0
                    for j, fa in enumerate(states[i - 1]):
                        c = costs[j] + _trans_cost(a, fa, b, fb, hand, spans, dt, shift_time)
                        if c < best:
                            best, arg = c, j
                    new_costs.append(best + within)
                    bp.append(arg)
                costs = new_costs
                back.append(bp)
            idx = min(range(len(costs)), key=costs.__getitem__)
            path = [idx]
            for bp in reversed(back):
                idx = bp[idx]
                path.append(idx)
            path.reverse()
            for s, opts, k in zip(slices, states, path):
                for n, f in zip(s, opts[k]):
                    n.effector["finger"] = f
            # sequential violations: same finger on a new key too fast
            for a, b in zip(slices, slices[1:]):
                dt = b[0].onset - a[0].onset
                fa = {n.effector["finger"]: n.pitch for n in a}
                for n in b:
                    f = n.effector["finger"]
                    if f in fa and fa[f] != n.pitch and dt < 0.07 and abs(fa[f] - n.pitch) > 2:
                        out.append(Violation("finger", n.onset, f"finger {f} jumps {abs(fa[f] - n.pitch)} st "
                                             f"in {dt * 1000:.0f} ms", [n.id], hand))
        return out


def _bucket(h: HandState | None) -> tuple[int, int] | None:
    return None if h is None else (h.lo // 2, h.hi // 2)


def _finger_options(n: int, hand: str) -> list[tuple[int, ...]]:
    """Finger tuples for a chord of n notes sorted low->high."""
    if n > 5:
        # more than five notes: give the extras to the outer fingers; this is a violation anyway
        base = (1, 2, 3, 4, 5) if hand == "R" else (5, 4, 3, 2, 1)
        return [tuple(base[min(i, 4)] for i in range(n))]
    combos = list(itertools.combinations(range(1, 6), n))
    if hand == "L":
        combos = [tuple(reversed(c)) for c in combos]
    return combos


def _pair_cost(p_lo: int, f_lo: int, p_hi: int, f_hi: int, hand: str, spans: dict) -> tuple[float, bool]:
    """Cost of fingers f_lo, f_hi on pitches p_lo <= p_hi (or any order for sequences)."""
    if hand == "L":  # mirror: left hand fingers count from the top
        p_lo, p_hi = -p_lo, -p_hi
    if f_lo == f_hi:
        return (0.0 if p_lo == p_hi else 3.0), False
    if f_lo < f_hi:
        a, b, d = f_lo, f_hi, p_hi - p_lo
    else:
        a, b, d = f_hi, f_lo, p_lo - p_hi
    mn_p, mn_c, mn_r, mx_r, mx_c, mx_p = spans[(a, b)]
    cost = 0.0
    impossible = d > mx_p or d < mn_p
    if d > mx_c:
        cost += 2 * (d - mx_c)
    elif d < mn_c:
        cost += 2 * (mn_c - d)
    if d > mx_r:
        cost += (1 if a == 1 else 2) * (d - mx_r)
    elif d < mn_r:
        cost += (2 if a == 1 else 1) * (mn_r - d)
    return cost, impossible


def _chord_cost(s: list[Note], fingers: tuple[int, ...], hand: str, spans: dict) -> float:
    cost = 0.0
    for (n1, f1), (n2, f2) in zip(zip(s, fingers), list(zip(s, fingers))[1:]):
        c, imp = _pair_cost(n1.pitch, f1, n2.pitch, f2, hand, spans)
        cost += c + (20 if imp else 0)
    for n, f in zip(s, fingers):
        black = n.pitch % 12 in BLACK
        if f in (4, 5):
            cost += 0.5
        if f == 1 and black:
            cost += 1.0
        if f == 5 and black:
            cost += 0.5
    return cost


def _trans_cost(a: list[Note], fa: tuple[int, ...], b: list[Note], fb: tuple[int, ...], hand: str,
                spans: dict, dt: float, shift_time: float) -> float:
    # judge the transition by the outer notes of each chord (the hand's frame)
    cost = 0.0
    pairs = [(a[0], fa[0], b[0], fb[0]), (a[-1], fa[-1], b[-1], fb[-1])]
    for na, f1, nb, f2 in pairs:
        if f1 == f2:
            if na.pitch == nb.pitch:
                continue
            c = 4.0 + (6.0 if dt < shift_time else 0.0)
        else:
            c, imp = _pair_cost(na.pitch, f1, nb.pitch, f2, hand, spans)
            if imp:
                c += 15.0
            # thumb passing under / finger crossing over
            up = (nb.pitch > na.pitch) == (hand == "R")
            if f2 == 1 and f1 != 1 and up:
                c += 1.0 + (2.0 if nb.pitch % 12 in BLACK else 0.0)
            if f1 == 1 and f2 != 1 and not up:
                c += 1.0
            if f1 == 3 and f2 == 4:
                c += 0.5
        if dt >= shift_time:
            c = min(c, 1.5)  # time to reposition the whole hand
        cost += c
    return cost / 2
