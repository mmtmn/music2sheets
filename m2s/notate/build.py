"""Stage 6, levels 3-7: voices, measures, clefs, dynamics, pedal, parts -> music21 score.

Input notes already carry ``beat`` (quarters from bar 1), ``qdur``,
``spelling`` and an effector assignment. This module decides how they look:
which voice, which staff, where bars and ties fall, and the markings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

from music21 import (articulations, bar, chord, clef, dynamics, expressions, instrument, key, layout, meter,
                     metadata, note, pitch, stream, tempo)

from m2s.ir import Note, Piece, Track
from m2s.notate.keys import Key
from m2s.notate.rhythm import Grid
from m2s.play.preset import load_preset

Q = Fraction
STANDARD = [Q(4), Q(3), Q(2), Q(3, 2), Q(1), Q(3, 4), Q(1, 2), Q(3, 8), Q(1, 4), Q(1, 8), Q(1, 16)]
TRIPLETS = [Q(2, 3), Q(1, 3), Q(1, 6), Q(1, 12)]


def frac(x: float) -> Fraction:
    return Q(round(x * 48), 48)


@dataclass
class Event:
    on: Fraction
    dur: Fraction
    notes: list[Note]
    staccato: bool = False
    marks: list[str] = field(default_factory=list)
    graces: list[Note] = field(default_factory=list)


@dataclass
class Options:
    fingering: bool = False
    dynamics: bool = True
    pedal: bool = False    # pedal marks are optional: most editions leave pedalling to the player
    title: str = "Transcription"


# --------------------------------------------------------------------------- voices
NICE = [Q(4), Q(3), Q(2), Q(3, 2), Q(1), Q(3, 4), Q(1, 2), Q(3, 8), Q(1, 4), Q(1, 8)]
NICE_TRIPLET = [Q(2, 3), Q(1, 3), Q(1, 6)]


def nice_duration(on: Fraction, dur: Fraction, limit: Fraction | None = None) -> Fraction:
    """Round a duration to a plain value (triplet values only when the onset is on a triplet grid),
    never past `limit`. Keeps beat-grid slips from turning into 48-tuplets."""
    triplet = (on * 3).denominator == 1 and on.denominator % 3 == 0
    pool = NICE + (NICE_TRIPLET if triplet else [])
    if limit is not None:
        pool = [d for d in pool if d <= limit] or [limit]
    return min(pool, key=lambda d: (abs(d - dur), -d))


def staff_voices(notes: list[Note]) -> list[list[Event]]:
    """Split one staff's notes into at most two voices.

    Voice 1 carries every chord, legato-truncated to the next onset (pedal and
    overlapping key releases don't change notated values). A note the hand
    clearly keeps holding while others move (>= 1 quarter past the next onset
    and >= twice the gap) goes to voice 2 with its full value.
    """
    graces = sorted((n for n in notes if "grace" in n.tags), key=lambda n: n.onset)
    notes = [n for n in notes if "grace" not in n.tags]
    groups: dict[Fraction, list[Note]] = {}
    for n in notes:
        groups.setdefault(frac(n.beat), []).append(n)
    onsets = sorted(groups)
    v1: list[Event] = []
    v2: list[Event] = []
    for i, on in enumerate(onsets):
        chord_notes = sorted(groups[on], key=lambda n: n.pitch)
        nxt = onsets[i + 1] if i + 1 < len(onsets) else None
        ioi = (nxt - on) if nxt is not None else None
        if ioi is not None:
            held = [n for n in chord_notes if frac(n.qdur) >= ioi + 1 and frac(n.qdur) >= 2 * ioi]
            rest = [n for n in chord_notes if n not in held]
            if held:
                v2.append(Event(on, nice_duration(on, min(frac(n.qdur) for n in held)), held))
                chord_notes = rest
        if not chord_notes:
            continue
        dur = min(frac(n.qdur) for n in chord_notes)
        stacc = False
        if ioi is not None:
            if dur >= ioi:
                dur = ioi
            else:
                gap = ioi - dur
                if dur <= Q(3, 5) * ioi and ioi <= 1:
                    stacc, dur = True, ioi
                elif gap <= Q(1, 4) and gap < ioi / 2:
                    dur = ioi
                else:
                    dur = nice_duration(on, dur, limit=ioi)
        else:
            dur = nice_duration(on, dur)
        v1.append(Event(on, max(dur, Q(1, 48)), chord_notes, staccato=stacc))
    # each grace note hangs on the first voice-1 event at or after it
    for g in graces:
        target = next((e for e in v1 if e.notes[0].onset >= g.onset - 0.01), None)
        if target is not None:
            target.graces.append(g)
    # voice 2 must not overlap itself
    v2.sort(key=lambda e: e.on)
    for a, b in zip(v2, v2[1:]):
        if a.on + a.dur > b.on:
            a.dur = b.on - a.on
    v2 = [e for e in v2 if e.dur > 0]
    return [v1, v2] if v2 else [v1]


# --------------------------------------------------------------------------- durations
def split_duration(start: Fraction, dur: Fraction) -> list[Fraction]:
    """Break a duration into notatable pieces (tied), respecting beats where it helps."""
    out: list[Fraction] = []
    pos, left = start, dur
    while left > 0:
        piece = None
        for cand in STANDARD + TRIPLETS:
            if cand <= left and _aligned(pos, cand):
                piece = cand
                break
        if piece is None:
            piece = min(left, Q(1, 48))
        out.append(piece)
        pos += piece
        left -= piece
    return out


def _aligned(pos: Fraction, d: Fraction) -> bool:
    # a value may start anywhere its own size divides the position, or on a beat for long values
    if d.denominator % 3 == 0:
        unit = Q(1, 3) if d >= Q(1, 3) else d
        return (pos / unit).denominator == 1 or (pos * 6).denominator == 1
    if d >= 1:
        return (pos / min(d, Q(2))).denominator == 1 or (d in (Q(1), Q(3, 2), Q(2), Q(3)) and pos.denominator == 1)
    return (pos / d).denominator == 1 or (d == Q(3, 4) and (pos * 4).denominator == 1)


# --------------------------------------------------------------------------- measures
class Bars:
    def __init__(self, grid: Grid, end_q: Fraction):
        self.bar = frac(grid.bar_quarters)
        self.pickup = frac(grid.pickup_quarters)
        start = -self.pickup
        self.starts: list[Fraction] = []
        q = start
        while q < end_q or not self.starts:
            self.starts.append(q)
            q = q + (self.pickup if (q == start and self.pickup) else self.bar)
        self.ends = self.starts[1:] + [self.starts[-1] + self.bar]

    def index(self, q: Fraction) -> int:
        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            if a <= q < b:
                return i
        return len(self.starts) - 1 if q >= self.starts[-1] else 0


def make_element(ev: Event, dur: Fraction, tie: str | None, spelled: Callable[[Note], pitch.Pitch],
                 perc: dict[str, Any] | None = None) -> note.GeneralNote:
    if perc is not None:
        els = []
        for n in ev.notes:
            u = note.Unpitched(displayName=perc["display"].get(n.pitch, "C5"))
            head = perc["heads"].get(n.pitch, "normal")
            if head != "normal":
                u.notehead = head
            els.append(u)
        if len(els) == 1:
            el = els[0]
        else:
            from music21.percussion import PercussionChord

            el = PercussionChord(els)
        el.duration.quarterLength = dur
        el.stemDirection = perc["stems"].get(ev.notes[0].pitch, "up")
    elif len(ev.notes) == 1:
        el = note.Note(spelled(ev.notes[0]))
        el.duration.quarterLength = dur
    else:
        el = chord.Chord([spelled(n) for n in ev.notes])
        el.duration.quarterLength = dur
    if tie:
        from music21 import tie as m21tie

        el.tie = m21tie.Tie(tie)
    return el


def fill_voice(events: list[Event], bars: Bars, spelled: Callable[[Note], pitch.Pitch],
               perc: dict[str, Any] | None, on_element: Callable[[Event, note.GeneralNote, bool], None]
               ) -> list[list[tuple[Fraction, note.GeneralNote]]]:
    """Place a voice's events into bars, splitting and tying at barlines and odd values."""
    per_bar: list[list[tuple[Fraction, note.GeneralNote]]] = [[] for _ in bars.starts]
    for ev in events:
        pos, left = ev.on, ev.dur
        first = True
        while left > 0:
            bi = bars.index(pos)
            room = bars.ends[bi] - pos
            seg = min(left, room)
            pieces = split_duration(pos - bars.starts[bi], seg)
            for k, d in enumerate(pieces):
                last_piece = (left - d) <= 0
                tie = None
                if not (first and k == 0) and not last_piece:
                    tie = "continue"
                elif not (first and k == 0):
                    tie = "stop"
                elif not last_piece:
                    tie = "start"
                el = make_element(ev, d, tie, spelled, perc)
                on_element(ev, el, first and k == 0)
                if first and k == 0 and perc is None:
                    for g in ev.graces:
                        gn = note.Note(spelled(g))
                        gn.duration.quarterLength = Q(1, 2)
                        per_bar[bi].append((pos - bars.starts[bi], gn.getGrace()))
                per_bar[bi].append((pos - bars.starts[bi], el))
                pos += d
                left -= d
            first = False
    return per_bar


def with_rests(items: list[tuple[Fraction, note.GeneralNote]], length: Fraction, start: Fraction
               ) -> list[tuple[Fraction, note.GeneralNote]]:
    out = []
    pos = Q(0)
    for off, el in sorted(items, key=lambda x: x[0]):
        if off > pos:
            for d in split_duration(start + pos, off - pos):
                r = note.Rest()
                r.duration.quarterLength = d
                out.append((pos, r))
                pos += d
        out.append((off, el))
        if not el.duration.isGrace:
            pos = max(pos, off + Q(el.duration.quarterLength))
    if pos < length:
        if pos == 0:
            r = note.Rest()
            r.duration.quarterLength = length
            r.fullMeasure = True
            return [(Q(0), r)]
        for d in split_duration(start + pos, length - pos):
            r = note.Rest()
            r.duration.quarterLength = d
            out.append((pos, r))
            pos += d
    return out


# --------------------------------------------------------------------------- clefs
def clef_plan(events: list[Event], bars: Bars, default: str, switch_cost: float = 8.0) -> list[str]:
    """Treble or bass per bar by dynamic programming: fewest ledger lines, few clef changes."""
    per_bar: dict[int, list[int]] = {}
    for ev in events:
        per_bar.setdefault(bars.index(ev.on), []).extend(n.pitch for n in ev.notes)
    names = ("treble", "bass")
    # middle line: B4 (71) for treble, D3 (50) for bass; one ledger line per ~3.5 semitones past the staff
    mid = {"treble": 71, "bass": 50}

    def bar_cost(c: str, ps: list[int]) -> float:
        cost = 0.0
        for p in ps:
            over = abs(p - mid[c]) - 9
            if over > 0:
                lines = over / 3.5
                cost += lines + (2.0 * (lines - 3) if lines > 3 else 0.0)
        return cost + (0.3 if c != default and ps else 0.0)

    n = len(bars.starts)
    costs = {c: (0.0 if c == default else switch_cost) + bar_cost(c, per_bar.get(0, [])) for c in names}
    back: list[dict[str, str]] = []
    for i in range(1, n):
        new_costs, bp = {}, {}
        for c in names:
            best = min(names, key=lambda pc: costs[pc] + (0 if pc == c else switch_cost))
            new_costs[c] = costs[best] + (0 if best == c else switch_cost) + bar_cost(c, per_bar.get(i, []))
            bp[c] = best
        costs = new_costs
        back.append(bp)
    cur = min(names, key=costs.get)
    plan = [cur]
    for bp in reversed(back):
        cur = bp[cur]
        plan.append(cur)
    return list(reversed(plan))


def make_clef(name: str) -> clef.Clef:
    return {"treble": clef.TrebleClef, "bass": clef.BassClef, "treble8vb": clef.Treble8vbClef,
            "perc": clef.PercussionClef, "tab": clef.TabClef}[name]()


# --------------------------------------------------------------------------- dynamics
LEVELS = [(40, "pp"), (54, "p"), (66, "mp"), (78, "mf"), (92, "f"), (128, "ff")]


def dynamic_plan(notes: list[Note], bars: Bars) -> dict[int, str]:
    vel: dict[int, list[int]] = {}
    for n in notes:
        vel.setdefault(bars.index(frac(n.beat)), []).append(n.velocity)
    means = [sum(vel[i]) / len(vel[i]) if i in vel else None for i in range(len(bars.starts))]
    smooth = []
    for i in range(len(means)):
        w = [m for m in means[max(0, i - 1): i + 2] if m is not None]
        smooth.append(sorted(w)[len(w) // 2] if w else None)
    plan: dict[int, str] = {}
    cur = None
    for i, m in enumerate(smooth):
        if m is None:
            continue
        lvl = next(name for top, name in LEVELS if m < top)
        nxt = smooth[i + 1] if i + 1 < len(smooth) else m
        hold = nxt is None or next(name for top, name in LEVELS if nxt < top) == lvl
        if lvl != cur and (hold or cur is None):
            plan[i] = lvl
            cur = lvl
    return plan


# --------------------------------------------------------------------------- score
@dataclass
class StaffSpec:
    name: str
    notes: list[Note]
    clef: str
    clef_auto: bool = True
    perc: dict[str, Any] | None = None
    tab: dict[str, Any] | None = None
    dynamics: bool = False
    pedal: list[tuple[float, float]] = field(default_factory=list)


def staves_for_track(track: Track, preset: dict[str, Any]) -> list[StaffSpec]:
    kept = [n for n in track.kept() if n.beat is not None]
    fam = preset["family"]
    if fam == "keyboard":
        rh = [n for n in kept if n.effector.get("hand", "R") == "R"]
        lh = [n for n in kept if n.effector.get("hand") == "L"]
        return [StaffSpec("RH", rh, "treble", dynamics=True),
                StaffSpec("LH", lh, "bass", pedal=track.pedal)]
    if fam == "drums":
        kit = preset["kit"]
        disp = {v["midi"]: v["display"] for v in kit.values()}
        heads = {v["midi"]: v.get("notehead", "normal") for v in kit.values()}
        stems = {v["midi"]: ("down" if set(v["limbs"]) <= {"RF", "LF"} else "up") for v in kit.values()}
        return [StaffSpec("Drums", kept, "perc", clef_auto=False,
                          perc={"display": disp, "heads": heads, "stems": stems}, dynamics=True)]
    c = preset.get("clef", "treble")
    if c == "auto":
        pitches = sorted(n.pitch for n in kept) or [60]
        c = "treble" if pitches[len(pitches) // 2] >= 57 else "treble8vb"
    specs = [StaffSpec(preset["display"], kept, c, clef_auto=False, dynamics=True)]
    if preset.get("tab"):
        specs.append(StaffSpec(preset["display"] + " tab", kept, "tab", clef_auto=False,
                               tab={"strings": len(preset["tuning"])}))
    return specs


def build_score(piece: Piece, grid: Grid, keys: dict[str, Key], opts: Options | None = None) -> stream.Score:
    opts = opts or Options()
    score = stream.Score()
    score.insert(0, metadata.Metadata(title=opts.title, composer=""))
    all_kept = [n for t in piece.tracks for n in t.kept() if n.beat is not None]
    end_q = max((frac(n.beat + n.qdur) for n in all_kept), default=frac(grid.bar_quarters))
    bars = Bars(grid, end_q)
    qpm = round(grid.qpm())
    first_part = True

    for track in piece.tracks:
        preset = load_preset(track.preset)
        trans = int(preset.get("transposition", 0))
        k = keys.get(track.name)
        specs = staves_for_track(track, preset)
        parts = []
        for si, spec in enumerate(specs):
            part = stream.PartStaff() if len(specs) > 1 else stream.Part()
            part.id = f"{track.name}-{si}"
            part.partName = preset["display"] if si == 0 else ""
            inst = instrument.Piano() if preset["family"] == "keyboard" else instrument.Instrument()
            if preset["family"] != "keyboard":
                inst.instrumentName = preset["display"]
            if preset["family"] == "drums":
                inst.midiChannel = 9
            part.insert(0, inst)
            parts.append(_fill_part(part, spec, bars, grid, k, trans, preset, track, opts,
                                    tempo_mark=qpm if (first_part and si == 0) else None))
        first_part = False
        for p in parts:
            score.insert(0, p)
        if len(parts) > 1:
            score.insert(0, layout.StaffGroup(parts, name=preset["display"],
                                              symbol="brace" if preset["family"] == "keyboard" else "bracket",
                                              barTogether=True))
    return score


def _spelled(trans: int) -> Callable[[Note], pitch.Pitch]:
    def f(n: Note) -> pitch.Pitch:
        if n.spelling:
            return pitch.Pitch(n.spelling)
        p = pitch.Pitch()
        p.midi = n.pitch + trans
        return p

    return f


def _fill_part(part: stream.Part, spec: StaffSpec, bars: Bars, grid: Grid, k: Key | None, trans: int,
               preset: dict[str, Any], track: Track, opts: Options, tempo_mark: int | None) -> stream.Part:
    voices = staff_voices(spec.notes) if spec.notes else [[]]
    # attach markings per first element of an event
    sf_events: set[int] = set()
    if spec.dynamics and opts.dynamics and spec.notes:
        vel_bar: dict[int, list[int]] = {}
        for n in spec.notes:
            vel_bar.setdefault(bars.index(frac(n.beat)), []).append(n.velocity)
        for ev in voices[0]:
            bi = bars.index(ev.on)
            mean = sum(vel_bar[bi]) / len(vel_bar[bi])
            peak = max(n.velocity for n in ev.notes)
            if peak >= mean + 22 and peak >= 88 and len(ev.notes) >= 2:
                sf_events.add(id(ev))

    first_elements: dict[int, note.GeneralNote] = {}

    def on_el(ev: Event, el: note.GeneralNote, first: bool) -> None:
        if not first:
            return
        first_elements[id(ev)] = el
        # tab staves carry string and fret only; articulations live on the notation staff
        if ev.staccato and spec.tab is None:
            el.articulations.append(articulations.Staccato())
        if any("trill" in n.tags for n in ev.notes) and spec.tab is None:
            el.expressions.append(expressions.Trill())
        if spec.tab is not None:
            ns = ev.notes
            if isinstance(el, chord.Chord):
                for cn, n in zip(el.notes, ns):
                    _tab_marks(cn, n)
            else:
                _tab_marks(el, ns[0])
        elif opts.fingering and spec.perc is None:
            fingers = [n.effector.get("finger") for n in ev.notes]
            if all(fingers):
                for f in fingers:
                    el.articulations.append(articulations.Fingering(f))

    spelled = _spelled(trans)
    per_voice = [fill_voice(v, bars, spelled, spec.perc, on_el) for v in voices]
    clefs = clef_plan(voices[0] + (voices[1] if len(voices) > 1 else []), bars, spec.clef) if spec.clef_auto \
        else [spec.clef] * len(bars.starts)
    dyn = dynamic_plan(spec.notes, bars) if (spec.dynamics and opts.dynamics and spec.perc is None) else {}

    prev_clef = None
    for bi, (a, b) in enumerate(zip(bars.starts, bars.ends)):
        m = stream.Measure(number=bi + (0 if bars.pickup else 1))
        length = b - a
        if bi == 0:
            if bars.pickup:
                m.paddingLeft = float(bars.bar - bars.pickup)
            m.insert(0, meter.TimeSignature(grid.time_signature()))
            if k is not None and spec.perc is None and spec.tab is None:
                ks = key.KeySignature(_written_sharps(k.sharps, trans))
                m.insert(0, ks)
            if tempo_mark:
                m.insert(0, tempo.MetronomeMark(number=tempo_mark))
        if clefs[bi] != prev_clef:
            m.insert(0, make_clef(clefs[bi]))
            prev_clef = clefs[bi]
        if spec.tab is not None and bi == 0:
            m.insert(0, layout.StaffLayout(staffLines=spec.tab["strings"]))
        if bi in dyn:
            m.insert(0, dynamics.Dynamic(dyn[bi]))
        contents = [with_rests(pv[bi], length, a) for pv in per_voice]
        # drop a second voice that only rests in this bar
        if len(contents) > 1 and all(isinstance(el, note.Rest) for _, el in contents[1]):
            contents = contents[:1]
        if len(contents) == 1:
            for off, el in contents[0]:
                m.insert(float(off), el)
        else:
            for vi, items in enumerate(contents):
                v = stream.Voice(id=str(vi + 1))
                for off, el in items:
                    v.insert(float(off), el)
                m.insert(0, v)
        if bi == len(bars.starts) - 1:
            m.rightBarline = bar.Barline("final")
        part.append(m)

    # sforzandi
    for ev_id in sf_events:
        el = first_elements.get(ev_id)
        if el is not None and el.activeSite is not None:
            el.activeSite.insert(el.offset, dynamics.Dynamic("sf"))

    if spec.pedal and opts.pedal:
        _pedal_marks(part, spec.pedal, grid)
    return part


def _written_sharps(sharps: int, trans: int) -> int:
    # transposing a key signature by `trans` semitones moves it 7*trans steps on the line of fifths (mod 12)
    s = sharps + (7 * trans) % 12
    while s > 6:
        s -= 12
    while s < -6:
        s += 12
    return s


def _tab_marks(el: note.Note, n: Note) -> None:
    s, f = n.effector.get("string"), n.effector.get("fret")
    if s is not None and f is not None:
        el.articulations.append(articulations.StringIndication(int(s)))
        el.articulations.append(articulations.FretIndication(int(f)))


def _pedal_marks(part: stream.Part, pedal: list[tuple[float, float]], grid: Grid) -> None:
    els = [e for e in part.recurse().notes if not e.tie or e.tie.type == "start"]
    if not els:
        return
    offs = [(float(e.getOffsetInHierarchy(part)), e) for e in els]
    for a, b in pedal:
        qa, qb = grid.to_quarters(a), grid.to_quarters(b)
        inside = [e for o, e in offs if qa - 0.05 <= o < qb - 0.05]
        if len(inside) >= 1:
            pm = expressions.PedalMark(inside[0], inside[-1]) if len(inside) > 1 else expressions.PedalMark(inside[0])
            pm.pedalForm = expressions.PedalForm.Symbol
            part.insert(0, pm)


def lint(score: stream.Score) -> dict[str, Any]:
    """Mechanical engraving checks: bar fullness, ledger lines, clef churn."""
    issues: list[dict[str, Any]] = []
    for part in score.parts:
        clef_changes = 0
        for m in part.getElementsByClass(stream.Measure):
            expected = m.barDuration.quarterLength - (m.paddingLeft or 0)
            voices = m.voices or [m]
            for v in voices:
                got = sum(e.duration.quarterLength for e in v.notesAndRests)
                if abs(got - expected) > 1e-3 and not any(getattr(e, "fullMeasure", False) for e in v.notesAndRests):
                    issues.append({"part": part.id, "bar": m.number, "kind": "bar-length",
                                   "detail": f"{got} of {expected}"})
            if m.getElementsByClass(clef.Clef) and m.number > 1:
                clef_changes += 1
            cl = m.getContextByClass(clef.Clef)
            for e in m.recurse().notes:
                for p in getattr(e, "pitches", []):
                    if isinstance(cl, (clef.TrebleClef, clef.BassClef)):
                        ledger = _ledger_lines(p, cl)
                        if ledger > 4:
                            issues.append({"part": part.id, "bar": m.number, "kind": "ledger",
                                           "detail": f"{p.nameWithOctave}: {ledger} ledger lines"})
        n_bars = len(part.getElementsByClass(stream.Measure))
        if n_bars and clef_changes / n_bars > 0.25:
            issues.append({"part": part.id, "kind": "clef-churn", "detail": f"{clef_changes} changes in {n_bars} bars"})
    kinds: dict[str, int] = {}
    for i in issues:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    return {"count": len(issues), "by_kind": kinds, "issues": issues[:200]}


def _ledger_lines(p: pitch.Pitch, cl: clef.Clef) -> int:
    # diatonic steps from the middle line; staff spans +-4 steps
    steps = p.diatonicNoteNum - (35 if isinstance(cl, clef.TrebleClef) else 23)
    over = abs(steps) - 4
    return max(0, math.ceil(over / 2)) if over > 0 else 0
