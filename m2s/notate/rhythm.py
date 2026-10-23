"""Stage 6, levels 2 and 4: tempo map, beat unit, meter, and quantization.

The beat tracker gives beat times in seconds. From those and the decoded
notes we choose:

* the beat unit (is the tracked beat a quarter, an eighth or a half note?),
* the meter and the downbeat phase (beats per bar, where bar 1 starts),
* a subdivision per beat (binary 1/2/4/8 or triplet 3/6), and
* every note's onset and duration in quarter notes.

All choices are scored, not guessed, and recorded in the returned grid so the
report can show them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m2s.ir import Note

# subdivisions per quarter and their complexity cost (in quarters of timing error)
DIVISIONS = {1: 0.0, 2: 0.008, 4: 0.02, 8: 0.09, 3: 0.04, 6: 0.09}


@dataclass
class Grid:
    beat_times: np.ndarray              # tracked beats (s)
    unit: float                          # quarters per tracked beat
    bar_quarters: float                  # quarters per bar
    beats_per_bar: int                   # tracked beats per bar
    pickup_quarters: float               # length of the incomplete first bar (0 = none)
    first_downbeat_beat: int             # index of the tracked beat where bar 1 starts
    divisions: list[int] = field(default_factory=list)  # chosen subdivision per quarter
    info: dict[str, Any] = field(default_factory=dict)

    def to_quarters(self, t: float) -> float:
        """Seconds -> quarters from the first downbeat (negative inside the pickup)."""
        bt = self.beat_times
        idx = np.arange(len(bt), dtype=float)
        if t <= bt[0]:
            b = (t - bt[0]) / max(bt[1] - bt[0], 1e-3)
        elif t >= bt[-1]:
            b = len(bt) - 1 + (t - bt[-1]) / max(bt[-1] - bt[-2], 1e-3)
        else:
            b = float(np.interp(t, bt, idx))
        return (b - self.first_downbeat_beat) * self.unit

    def time_signature(self) -> str:
        q = self.bar_quarters
        if self.info.get("compound"):
            return f"{int(round(q * 2))}/8"
        if abs(q - round(q)) < 1e-6:
            return f"{int(round(q))}/4"
        return f"{int(round(q * 2))}/8"

    def qpm(self) -> float:
        ibi = np.median(np.diff(self.beat_times))
        return float(60 / ibi / self.unit)


def clean_beats(beats: list[float]) -> np.ndarray:
    """Fix the commonest tracker slips: doubled beats and single missing beats."""
    b = np.asarray(beats, dtype=float)
    if len(b) < 4:
        return b
    med = np.median(np.diff(b))
    out = [b[0]]
    for t in b[1:]:
        gap = t - out[-1]
        if gap < 0.45 * med:      # doubled beat: skip
            continue
        if 1.6 * med < gap < 2.4 * med:  # one missing beat: insert midpoint
            out.append(out[-1] + gap / 2)
        out.append(t)
    return np.asarray(out)


def choose_unit(beat_times: np.ndarray, notes: list[Note]) -> tuple[float, dict[str, float]]:
    """Quarters per tracked beat, from a tempo prior and a note-value prior."""
    ibi = float(np.median(np.diff(beat_times)))
    onsets = np.unique(np.round([n.onset for n in notes], 2))
    iois = np.diff(onsets)
    iois = iois[iois > 0.03]
    # typical fast note in beats: the 20th percentile inter-onset interval
    fast_beats = float(np.percentile(iois, 20) / ibi) if len(iois) else 0.5
    scores = {}
    for unit in (0.5, 1.0, 2.0):
        qpm = 60 / ibi / unit
        # beat trackers learn from human beat annotations, which usually follow the notated beat,
        # so the tracked beat is taken as a quarter unless the tempo would be implausible
        bias = 0.0 if unit == 1.0 else 0.8
        tempo_pen = max(0.0, abs(math.log(qpm / 110)) - math.log(1.7)) ** 2 / 0.1   # free within ~65-187 qpm
        fast_q = fast_beats * unit                              # fast note value in quarters
        # 16ths and 8ths are the common fast values; 32nds are rarer, quarters rarer still
        val = min((abs(math.log2(fast_q / v)) + c) for v, c in ((0.25, 0.0), (0.5, 0.1), (0.125, 0.6), (1.0, 0.8)))
        scores[unit] = bias + tempo_pen + 1.5 * val
    unit = min(scores, key=scores.get)
    return unit, {f"unit={k}": round(v, 3) for k, v in scores.items()}


def beat_features(beat_times: np.ndarray, notes: list[Note]) -> np.ndarray:
    """Per tracked beat: [bass onset, harmony change, loudness] used to find downbeats."""
    nb = len(beat_times)
    feats = np.zeros((nb, 3))
    if not notes:
        return feats
    edges = np.concatenate([[-np.inf], (beat_times[1:] + beat_times[:-1]) / 2, [np.inf]])
    idx = np.clip(np.searchsorted(edges, [n.onset for n in notes]) - 1, 0, nb - 1)
    pcs = np.zeros((nb, 12))
    vel = np.zeros(nb)
    low = np.full(nb, 200)
    for n, i in zip(notes, idx):
        pcs[i, n.pitch % 12] += n.duration
        vel[i] = max(vel[i], n.velocity)
        # bass: count only notes struck close to the beat
        if abs(n.onset - beat_times[i]) < 0.08:
            low[i] = min(low[i], n.pitch)
    allmin = np.percentile([n.pitch for n in notes], 10)
    feats[:, 0] = (low < allmin + 7).astype(float)
    norm = pcs / (np.linalg.norm(pcs, axis=1, keepdims=True) + 1e-9)
    change = np.ones(nb)
    change[1:] = 1 - (norm[1:] * norm[:-1]).sum(axis=1)
    feats[:, 1] = change
    feats[:, 2] = vel / (vel.max() + 1e-9)
    return feats


def harmonic_boundaries(beat_times: np.ndarray, notes: list[Note], w: int = 2) -> np.ndarray:
    """Per beat: how much the harmony of the `w` beats after it differs from the `w` beats before.

    Counts every note sounding in a beat (not only onsets), weighted by overlap, so a held
    chord under moving figuration still reads as one harmony.
    """
    nb = len(beat_times)
    edges = np.concatenate([beat_times, [beat_times[-1] + (beat_times[-1] - beat_times[-2])]])
    chroma = np.zeros((nb, 12))
    for n in notes:
        a = np.searchsorted(edges, n.onset, side="right") - 1
        b = np.searchsorted(edges, n.offset, side="left")
        for i in range(max(a, 0), min(b, nb)):
            ov = min(n.offset, edges[i + 1]) - max(n.onset, edges[i])
            if ov > 0:
                chroma[i, n.pitch % 12] += ov
    out = np.zeros(nb)
    for i in range(w, nb - w + 1):
        x, y = chroma[i - w:i].sum(axis=0), chroma[i:i + w].sum(axis=0)
        den = np.linalg.norm(x) * np.linalg.norm(y)
        out[i] = 1 - (x @ y) / den if den > 0 else 0.0
    return out


def choose_meter(beat_times: np.ndarray, downbeats: list[float], notes: list[Note],
                 unit: float, db_strength: np.ndarray | None = None) -> tuple[int, int, dict[str, float]]:
    """(tracked beats per bar, index of first downbeat) by scoring every candidate."""
    feats = beat_features(beat_times, notes)
    nb = len(beat_times)
    if db_strength is not None and len(db_strength) == nb:
        db = np.asarray(db_strength, dtype=float)
    else:
        db = np.zeros(nb)
        for t in downbeats:
            i = int(np.argmin(np.abs(beat_times - t)))
            if abs(beat_times[i] - t) < 0.1:
                db[i] = 1
    hb = harmonic_boundaries(beat_times, notes)
    cands: dict[tuple[int, int], float] = {}
    for per_bar_q in (2.0, 3.0, 4.0, 6.0):      # bar length in quarters
        bpb = per_bar_q / unit
        if abs(bpb - round(bpb)) > 1e-6 or round(bpb) < 2:
            continue
        bpb = int(round(bpb))
        for phase in range(bpb):
            on = np.arange(phase, nb, bpb)
            off = np.setdiff1d(np.arange(nb), on)
            if len(on) < 2 or len(off) < 2:
                continue
            contrast = feats[on].mean(axis=0) - feats[off].mean(axis=0)
            harm = hb[on].mean() - hb[off].mean()
            s = (0.5 * contrast[0] + 0.3 * contrast[1] + 0.3 * contrast[2] + 2.5 * harm
                 + 1.0 * (db[on].mean() - db[off].mean()))
            # Priors. 4/4 and 3/4 are the commonest bars. 6/4 is rare, and a contrast of means lets it
            # win by picking the louder half of real 3/4 bar lines, so it must win clearly. A piece
            # most often starts on a downbeat.
            s += {2.0: -0.02, 3.0: 0.0, 4.0: 0.02, 6.0: -0.15}[per_bar_q]
            s += 0.02 if phase == 0 else 0.0
            cands[(bpb, phase)] = float(s)
    (bpb, phase), _ = max(cands.items(), key=lambda kv: kv[1])
    return bpb, phase, {f"{k[0]}x{unit}q@{k[1]}": round(v, 3) for k, v in sorted(cands.items(), key=lambda kv: -kv[1])[:6]}


def backbeat_unit(beat_times: np.ndarray, drum_hits: list[Note]) -> float | None:
    """With a drum kit, the snare backbeat (beats 2 and 4) fixes the quarter note: snare hits every
    tracked beat mean the tracker follows half notes; every other beat means quarters."""
    snare = sorted(n.onset for n in drum_hits if n.pitch in (38, 40))
    if len(snare) < 8 or len(beat_times) < 8:
        return None
    ibi = float(np.median(np.diff(beat_times)))
    gaps = np.diff(snare)
    gaps = gaps[(gaps > 0.6 * ibi) & (gaps < 4.5 * ibi)]
    if len(gaps) < 6:
        return None
    ratio = float(np.median(gaps)) / ibi
    if abs(ratio - 1) < 0.2:
        return 2.0
    if abs(ratio - 2) < 0.3:
        return 1.0
    if abs(ratio - 4) < 0.5:
        return 0.5
    return None


def build_grid(beats: list[float], downbeats: list[float], notes: list[Note],
               db_strength: np.ndarray | None = None, drum_hits: list[Note] | None = None) -> Grid:
    bt = clean_beats(beats) if db_strength is None else np.asarray(beats, dtype=float)
    unit, unit_scores = choose_unit(bt, notes)
    bb = backbeat_unit(bt, drum_hits or [])
    if bb is not None:
        unit = bb
        unit_scores["backbeat"] = bb
    bpb, phase, meter_scores = choose_meter(bt, downbeats, notes, unit, db_strength)
    pickup = phase * unit
    first_db = phase
    # if the first note comes before the first tracked beat, extend the grid backwards
    if notes:
        t0 = min(n.onset for n in notes)
        ibi = float(np.median(np.diff(bt[:8])))
        extra = 0
        while bt[0] - extra * ibi > t0 + 0.05 and extra < 16:
            extra += 1
        if extra:
            bt = np.concatenate([bt[0] - ibi * np.arange(extra, 0, -1), bt])
            first_db += extra
            pickup = (first_db % bpb) * unit
            first_db = first_db
    bar_q = bpb * unit
    grid = Grid(beat_times=bt, unit=unit, bar_quarters=bar_q, beats_per_bar=bpb,
                pickup_quarters=pickup % bar_q, first_downbeat_beat=first_db,
                info={"unit_scores": unit_scores, "meter_scores": meter_scores})
    # compound meter: if most beats divide in three, the beat is a dotted quarter (6/8, 9/8, 12/8)
    trip = triplet_share(notes, grid)
    grid.info["triplet_share"] = round(trip, 3)
    if trip >= 0.55 and bpb in (2, 3, 4):
        grid.unit = 1.5
        grid.bar_quarters = 1.5 * bpb
        grid.pickup_quarters = (first_db % bpb) * 1.5 % grid.bar_quarters
        grid.info["compound"] = True
    return grid


def triplet_share(notes: list[Note], grid: Grid) -> float:
    """Share of subdivided beats whose onsets fit thirds better than halves/quarters."""
    if not notes:
        return 0.0
    pos = np.array([grid.to_quarters(n.onset) / grid.unit for n in notes])
    frac = pos - np.floor(pos)
    beat = np.floor(pos).astype(int)
    trip = binary = 0
    for b in np.unique(beat):
        f = frac[beat == b]
        f = f[(f > 0.08) & (f < 0.92)]
        if len(f) == 0:
            continue
        e3 = np.mean(np.abs(f * 3 - np.round(f * 3)) / 3)
        e4 = np.mean(np.abs(f * 4 - np.round(f * 4)) / 4)
        if e3 + 0.01 < e4:
            trip += 1
        else:
            binary += 1
    return trip / max(trip + binary, 1)


def quantize(notes: list[Note], grid: Grid) -> None:
    """Fill ``beat`` (absolute quarters from bar 1) and ``qdur`` on every note."""
    if not notes:
        return
    qon = np.array([grid.to_quarters(n.onset) for n in notes])
    qoff = np.array([grid.to_quarters(n.offset) for n in notes])
    start = math.floor(qon.min())
    end = math.ceil(max(qon.max(), qoff.max())) + 1
    nq = end - start
    divs = []
    prev = 4
    for q in range(start, end):
        sel = (qon >= q - 1e-9) & (qon < q + 1)
        frac = qon[sel] - q
        best, arg = math.inf, prev
        for d, cost in DIVISIONS.items():
            if not len(frac):
                err = 0.0
            else:
                pos = frac * d
                # onsets that round up to the next beat count as on-beat
                err = float(np.mean(np.abs(pos - np.round(pos)) / d))
            c = err + cost + (0.0 if d == prev else 0.01)
            if c < best:
                best, arg = c, d
        divs.append(arg)
        if len(frac):
            prev = arg
    # a triplet or 32nd grid must be shared by a neighbouring beat; a lone one is usually a beat-grid
    # slip, and writing it out produces tuplet clutter
    def best_binary(q: int) -> int:
        sel = (qon >= q - 1e-9) & (qon < q + 1)
        frac = qon[sel] - q
        if not len(frac):
            return 4
        return min((1, 2, 4), key=lambda d: float(np.mean(np.abs(frac * d - np.round(frac * d)) / d))
                   + DIVISIONS[d])

    fam = {1: "b", 2: "b", 4: "b", 8: "32", 3: "t", 6: "t"}
    smoothed = list(divs)
    for i, d in enumerate(divs):
        if fam[d] == "b":
            continue
        nb = [fam[divs[j]] for j in (i - 1, i + 1) if 0 <= j < len(divs)]
        if fam[d] not in nb:
            smoothed[i] = best_binary(start + i)
    divs = smoothed
    grid.divisions = divs

    def snap(x: float) -> float:
        q = math.floor(x)
        i = min(max(q - start, 0), nq - 1)
        d = divs[i]
        return q + round((x - q) * d) / d

    for n, a, b in zip(notes, qon, qoff):
        on = snap(a)
        off = snap(b)
        q = math.floor(on)
        d = divs[min(max(q - start, 0), nq - 1)]
        if off <= on:
            off = on + 1 / d
        n.beat = round(on, 6)
        n.qdur = round(off - on, 6)
