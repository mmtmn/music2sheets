"""Score-level metrics that need Python or Java: MV2H and musicdiff. (Note F1, hands and
playability are computed in the Rust core.)"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MV2H_DIR = Path(__file__).resolve().parents[2] / "third_party" / "MV2H"


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
