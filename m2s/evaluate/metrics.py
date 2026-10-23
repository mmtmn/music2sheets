"""Evaluation metrics: note F1, hand agreement, playability, MV2H, musicdiff."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from m2s.evaluate.reference import Reference, RefNote
from m2s.ir import Note, Track
from m2s.runs import Run

MV2H_DIR = Path(__file__).resolve().parents[2] / "third_party" / "MV2H"
HARD = {"span", "count", "speed", "finger", "stretch", "shift"}


def _hz(p: np.ndarray) -> np.ndarray:
    return 440.0 * 2 ** ((p - 69) / 12)


def note_scores(ref: Reference, ours: list[Note], tol: float) -> dict[str, float]:
    import mir_eval

    ri = np.array([[n.on, n.off] for n in ref.notes])
    rp = _hz(np.array([n.pitch for n in ref.notes], dtype=float))
    ei = np.array([[n.onset, max(n.offset, n.onset + 0.01)] for n in ours])
    ep = _hz(np.array([n.pitch for n in ours], dtype=float))
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=tol,
                                                                    offset_ratio=None)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}


def _matches(ref: Reference, ours: list[Note], tol: float = 0.1) -> list[tuple[RefNote, Note]]:
    import mir_eval

    ri = np.array([[n.on, n.off] for n in ref.notes])
    rp = _hz(np.array([n.pitch for n in ref.notes], dtype=float))
    ei = np.array([[n.onset, max(n.offset, n.onset + 0.01)] for n in ours])
    ep = _hz(np.array([n.pitch for n in ours], dtype=float))
    pairs = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=tol, offset_ratio=None)
    return [(ref.notes[i], ours[j]) for i, j in pairs]


def hand_agreement(ref: Reference, ours: list[Note]) -> dict[str, Any]:
    pairs = [(r, o) for r, o in _matches(ref, ours) if "hand" in o.effector]
    staffs = {r.staff for r, _ in pairs}
    if not pairs or len(staffs) < 2:
        return {"agreement": None, "matched": len(pairs)}
    ok = sum((o.effector["hand"] == "R") == (r.staff == 1) for r, o in pairs)
    return {"agreement": round(ok / len(pairs), 4), "matched": len(pairs)}


def playability(run: Run, ref: Reference, ours: list[Note]) -> dict[str, Any]:
    """Hard-limit violations per 100 notes: ours, and the reference's own (staff taken as hand)."""
    from m2s.play.piano import PianoDecoder

    viols = json.loads((run.root / "violations.json").read_text())
    ours_hard = sum(1 for v in viols if v["kind"] in HARD)
    out: dict[str, Any] = {"ours": round(100 * ours_hard / max(len(ours), 1), 2)}
    if {n.staff for n in ref.notes} <= {1, 2}:
        rnotes = [Note(f"r{i}", n.on, n.off, n.pitch, 80, effector={"hand": "R" if n.staff == 1 else "L"})
                  for i, n in enumerate(ref.notes)]
        dec = PianoDecoder()
        tr = Track("ref", "piano", rnotes)
        v = dec._hand_violations(tr) + dec.finger(tr)
        out["reference"] = round(100 * sum(1 for x in v if x.kind in HARD) / max(len(rnotes), 1), 2)
    return out


def score_facts(run: Run, ref: Reference) -> dict[str, Any]:
    info = json.loads((run.root / "notation.json").read_text())
    return {"ours": {"time_signature": info["time_signature"], "key": info["key"], "bars": info["bars"],
                     "qpm": info["qpm"]},
            "reference": {"time_signature": ref.time_signature, "key": ref.key, "bars": ref.bars}}


# --------------------------------------------------------------------------- MV2H
def _to_mv2h(musicxml: str | Path, out_txt: Path) -> None:
    """MusicXML -> MIDI (one track per staff, same-pitch overlaps trimmed) -> MV2H text format."""
    import music21
    import pretty_midi

    s = music21.converter.parse(str(musicxml))
    spq = 0.5  # fixed 120 qpm: MV2H only uses the metrical grid
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    first = s.parts[0].getElementsByClass(music21.stream.Measure).first() if s.parts else None
    pickup = 0.0
    if first is not None and first.paddingLeft:
        pickup = first.barDuration.quarterLength - first.paddingLeft
    for ts in s.parts[0].recurse().getElementsByClass(music21.meter.TimeSignature) if s.parts else []:
        pm.time_signature_changes.append(pretty_midi.TimeSignature(ts.numerator, ts.denominator,
                                                                   float(ts.getOffsetInHierarchy(s.parts[0])) * spq))
    for part in s.parts:
        inst = pretty_midi.Instrument(program=0)
        flat = part.stripTies().flatten()
        notes = []
        for el in flat.notes:
            if el.duration.isGrace:
                continue
            q, d = float(el.getOffsetInHierarchy(part)), float(el.quarterLength)
            for p in el.pitches:
                notes.append([q * spq, (q + max(d, 0.0625)) * spq, int(p.midi)])
        notes.sort()
        last: dict[int, list[float]] = {}
        for n in notes:
            prev = last.get(n[2])
            if prev is not None and prev[1] > n[0]:
                prev[1] = n[0]
            last[n[2]] = n
        for a, b, p in notes:
            if b - a > 1e-4:
                inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=a, end=b))
        pm.instruments.append(inst)
    mid = out_txt.with_suffix(".mid")
    pm.write(str(mid))
    cmd = ["java", "-cp", str(MV2H_DIR / "bin"), "mv2h.tools.Converter", "-i", str(mid), "-o", str(out_txt),
           "--track"]
    if pickup:
        cmd += ["-a", str(int(round(pickup * 4)))]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def mv2h(ours_xml: str | Path, ref_xml: str | Path) -> dict[str, Any]:
    if not (MV2H_DIR / "bin" / "mv2h").exists():
        return {"skipped": "MV2H not built; run tools/build_mv2h.sh"}
    if str(ref_xml).lower().endswith((".gp3", ".gp4", ".gp5", ".gtp")):
        return {"skipped": "MV2H needs a MusicXML or MIDI reference"}
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        try:
            _to_mv2h(ours_xml, t / "ours.txt")
            _to_mv2h(ref_xml, t / "ref.txt")
            r = subprocess.run(["java", "-cp", str(MV2H_DIR / "bin"), "mv2h.Main", "-g", str(t / "ref.txt"),
                                "-t", str(t / "ours.txt"), "-a"], capture_output=True, text=True, timeout=3600)
        except (subprocess.SubprocessError, OSError) as e:
            return {"error": str(e)[:500]}
    out: dict[str, Any] = {}
    for line in r.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = round(float(v), 4)
            except ValueError:
                pass
    return out or {"error": (r.stderr or r.stdout)[-500:]}


def musicdiff_count(ours_xml: str | Path, ref_xml: str | Path, bars: int = 16, timeout: int = 1800) -> dict[str, Any]:
    """Notation edits (musicdiff) from our score to the reference, over the first ``bars`` bars of each.
    Runs in a subprocess with a timeout: tree edit distance grows fast with score length."""
    if str(ref_xml).lower().endswith((".gp3", ".gp4", ".gp5", ".gtp")):
        return {"skipped": "musicdiff needs a MusicXML reference"}
    code = ("import sys, music21, musicdiff\n"
            "def head(p, n):\n"
            "    return music21.converter.parse(p).measures(0, n)\n"
            "n = int(sys.argv[3])\n"
            "k = musicdiff.diff(head(sys.argv[1], n), head(sys.argv[2], n), visualize_diffs=False,"
            " print_text_output=False)\n"
            "print('EDITS', k)\n")
    try:
        r = subprocess.run([sys.executable, "-c", code, str(ours_xml), str(ref_xml), str(bars)], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"skipped": f"timed out after {timeout}s"}
    for line in r.stdout.splitlines():
        if line.startswith("EDITS"):
            v = line.split()[1]
            return {"edits": None if v == "None" else int(v), "bars": bars}
    return {"error": r.stderr[-500:]}
