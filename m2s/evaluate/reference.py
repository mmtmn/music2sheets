"""Load a reference score (MusicXML, MIDI, kern, MEI via music21; Guitar Pro 3-5 via PyGuitarPro)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RefNote:
    qon: float          # quarters from the start
    qdur: float
    pitch: int
    staff: int          # 1-based staff/part index
    voice: int = 1      # 1-based voice within the staff
    sec_on: float = 0.0     # nominal seconds (score tempo)
    sec_off: float = 0.0
    on: float = 0.0     # aligned to the recording (seconds)
    off: float = 0.0


@dataclass
class Reference:
    notes: list[RefNote]
    time_signature: str | None
    key: str | None
    bars: int
    expanded: bool
    score_path: str                      # MusicXML of the (possibly expanded) reference, for MV2H / musicdiff
    meta: dict[str, Any] = field(default_factory=dict)
    bar_qons: list[float] = field(default_factory=list)   # bar starts, quarters from the start
    beat_q: float = 1.0                                    # notated beat in quarters


def load_reference(path: str | Path, repeats: str = "auto", expected_notes: int | None = None) -> Reference:
    path = Path(path)
    if path.suffix.lower() in (".gp3", ".gp4", ".gp5", ".gtp"):
        return _load_guitarpro(path)
    import music21

    sc = music21.converter.parse(str(path))
    if isinstance(sc, music21.stream.Opus):
        sc = sc.scores[0]
    versions = [(False, sc)]
    if repeats in ("auto", "expand") and path.suffix.lower() == ".krn":
        # every order the encoder declared (with repeats, without); the closest to the performance wins
        for name in humdrum_expansions(path):
            expanded_krn = expand_humdrum(path, name)
            if expanded_krn is not None:
                versions.append((True, music21.converter.parse(str(expanded_krn))))
    if repeats in ("auto", "expand") and len(versions) == 1:
        try:
            ex = sc.expandRepeats()
            if ex is not None and ex.highestTime > sc.highestTime + 1e-6:
                versions.append((True, ex))
        except Exception:  # noqa: BLE001  (music21 raises many types on odd repeat structures)
            pass
    if repeats == "expand" and len(versions) > 1:
        versions = versions[1:]
    if repeats == "none":
        versions = versions[:1]
    best = None
    for expanded, s in versions:
        notes = _notes(s)
        score = abs(len(notes) - expected_notes) if expected_notes else (0 if expanded else 1)
        if best is None or score < best[0]:
            best = (score, expanded, s, notes)
    _, expanded, s, notes = best
    out = Path(tempfile.mkdtemp(prefix="m2s-ref-")) / "reference.musicxml"
    s.write("musicxml", fp=str(out))
    ts = s.recurse().getElementsByClass(music21.meter.TimeSignature).first()
    ks = s.recurse().getElementsByClass(music21.key.KeySignature).first()
    # the key the encoder wrote (Humdrum `*c:`, MusicXML <mode>) beats an estimate from the notes
    k = next((x for x in s.recurse().getElementsByClass(music21.key.KeySignature)
              if isinstance(x, music21.key.Key) and x.mode in ("major", "minor")), None)
    if k is None:
        try:
            k = s.analyze("key")
        except Exception:  # noqa: BLE001
            k = None
    key = f"{k.tonic.name} {k.mode}" if k is not None else None
    measures = list(s.parts[0].getElementsByClass(music21.stream.Measure)) if s.parts else []
    bar_qons = [float(m.getOffsetInHierarchy(s.parts[0])) for m in measures]
    return Reference(notes, ts.ratioString if ts else None, key, len(measures), expanded, str(out),
                     {"key_signature_sharps": ks.sharps if ks else None}, bar_qons, _beat_q(ts))


def _beat_q(ts: Any) -> float:
    if ts is None:
        return 1.0
    num, den = ts.numerator, ts.denominator
    return 1.5 if den == 8 and num in (6, 9, 12) else 4 / den


def humdrum_expansions(path: Path) -> dict[str, list[str]]:
    """Every expansion list a Humdrum file declares: `*>[A,A,B]` (the default, name "") and named
    variants such as `*>norep[A,B]` (performances without repeats)."""
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        first = line.split("\t")[0]
        if first.startswith("*>") and "[" in first and first.endswith("]"):
            name = first[2:first.index("[")]
            out.setdefault(name, [x.strip() for x in first[first.index("[") + 1:-1].split(",") if x.strip()])
    return out


def expand_humdrum(path: Path, name: str = "") -> Path | None:
    """Apply one of a Humdrum file's expansion lists (default: `*>[...]`), as the `thrux` tool does.

    music21 ignores Humdrum section labels, so a kern score with repeats would otherwise be
    compared with a performance that takes them. Returns the expanded file, or None if the
    file declares no such expansion.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    order = humdrum_expansions(path).get(name)
    if not order:
        return None
    head: list[str] = []
    sections: dict[str, list[str]] = {}
    tail: list[str] = []
    current: str | None = None
    for line in lines:
        first = line.split("\t")[0]
        if first.startswith("*>") and not first.startswith("*>[") and "[" not in first:
            current = first[2:]
            sections.setdefault(current, [])
            continue
        if first.startswith("*>") and "[" in first:
            continue  # expansion lists themselves
        if first.startswith("*-") or (current is not None and first.startswith("!!!") and not tail and
                                      line is lines[-1]):
            current = None
            tail.append(line)
            continue
        if current is None:
            (tail if tail else head).append(line)
        else:
            sections[current].append(line)
    if any(lab not in sections for lab in order):
        return None
    body = [ln for lab in order for ln in sections[lab]]
    out = Path(tempfile.mkdtemp(prefix="m2s-krn-")) / (path.stem + f"-expanded{('-' + name) if name else ''}.krn")
    out.write_text("\n".join(head + body + tail) + "\n", encoding="utf-8")
    return out


def _notes(s: Any) -> list[RefNote]:
    import music21


    out: list[RefNote] = []
    for si, part in enumerate(s.parts):
        stripped = part.stripTies()
        for el in stripped.recurse().notes:
            if el.duration.isGrace:
                continue
            q = float(el.getOffsetInHierarchy(stripped))
            d = float(el.quarterLength)
            voice = 1
            site = el.activeSite
            if isinstance(site, music21.stream.Voice) and site.activeSite is not None:
                voice = list(site.activeSite.voices).index(site) + 1
            for p in el.pitches:
                out.append(RefNote(q, d, int(p.midi), si + 1, voice))
    out.sort(key=lambda n: (n.qon, n.pitch))
    _seconds(s, out)
    return out


def _seconds(s: Any, notes: list[RefNote]) -> None:
    """Nominal seconds from the score's tempo marks (default q = 100)."""
    import music21

    marks = []
    for mm in s.flatten().getElementsByClass(music21.tempo.MetronomeMark):
        try:
            qpm = mm.getQuarterBPM()
        except Exception:  # noqa: BLE001
            qpm = None
        if qpm:
            marks.append((float(mm.getOffsetInHierarchy(s)), float(qpm)))
    marks = sorted(marks) or [(0.0, 100.0)]
    if marks[0][0] > 0:
        marks.insert(0, (0.0, marks[0][1]))

    def sec(q: float) -> float:
        t, prev_q, prev_bpm = 0.0, 0.0, marks[0][1]
        for mq, bpm in marks[1:]:
            if mq >= q:
                break
            t += (mq - prev_q) * 60 / prev_bpm
            prev_q, prev_bpm = mq, bpm
        return t + (q - prev_q) * 60 / prev_bpm

    for n in notes:
        n.sec_on, n.sec_off = sec(n.qon), sec(n.qon + max(n.qdur, 0.05))


def _load_guitarpro(path: Path) -> Reference:
    import guitarpro

    song = guitarpro.parse(str(path))
    notes: list[RefNote] = []
    for ti, track in enumerate(song.tracks):
        if track.isPercussionTrack:
            continue
        tuning = {s.number: s.value for s in track.strings}
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    q = beat.start / 960 - 1     # GP ticks: quarter = 960, song starts at 960
                    d = beat.duration.time / 960
                    for n in beat.notes:
                        if n.type == guitarpro.NoteType.rest or n.type == guitarpro.NoteType.dead:
                            continue
                        if n.type == guitarpro.NoteType.tie:
                            continue
                        notes.append(RefNote(q, d, tuning[n.string] + n.value, ti + 1))
    notes.sort(key=lambda n: (n.qon, n.pitch))
    tempo = song.tempo or 120
    for n in notes:
        n.sec_on, n.sec_off = n.qon * 60 / tempo, (n.qon + n.qdur) * 60 / tempo
    ts = song.measureHeaders[0].timeSignature if song.measureHeaders else None
    tsig = f"{ts.numerator}/{ts.denominator.value}" if ts else None
    bar_qons = [h.start / 960 - 1 for h in song.measureHeaders]
    beat_q = 1.5 if ts and ts.denominator.value == 8 and ts.numerator in (6, 9, 12) else \
        (4 / ts.denominator.value if ts else 1.0)
    return Reference(notes, tsig, None, len(song.measureHeaders), False, str(path), {"format": "guitar-pro"},
                     bar_qons, beat_q)
