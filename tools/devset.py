"""Dev-set helpers: ASAP score notes warped onto the real performance's beat times.

Gives ground truth for the decoder (staff = hand) at realistic speeds, without
audio. Used only for tuning; the blind test never touches it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from m2s.ir import Note, Piece, Track

ROOT = Path("data/dev/asap")


def pieces() -> list[Path]:
    return sorted(p.parent for p in ROOT.rglob("midi_score.mid"))


def _beats(path: Path) -> np.ndarray:
    return np.array([float(line.split("\t")[0]) for line in path.read_text().splitlines() if line.strip()])


def load(folder: Path) -> tuple[Track, list[int]]:
    """Score notes with performance timing; returns (track, staff per note: 1 = RH, 2 = LH)."""
    perf = (folder / "performance.txt").read_text().strip()
    sb = _beats(folder / "midi_score_annotations.txt")
    pb = _beats(folder / (Path(perf).stem + "_annotations.txt"))
    n = min(len(sb), len(pb))
    sb, pb = sb[:n], pb[:n]

    def warp(t: float) -> float:
        if t <= sb[0]:
            return pb[0] + (t - sb[0]) * ((pb[1] - pb[0]) / max(sb[1] - sb[0], 1e-6))
        if t >= sb[-1]:
            return pb[-1] + (t - sb[-1]) * ((pb[-1] - pb[-2]) / max(sb[-1] - sb[-2], 1e-6))
        return float(np.interp(t, sb, pb))

    pm = pretty_midi.PrettyMIDI(str(folder / "midi_score.mid"))
    notes, staff = [], []
    for si, inst in enumerate(pm.instruments[:2]):
        for nn in inst.notes:
            notes.append(Note(id="", onset=warp(nn.start), offset=max(warp(nn.end), warp(nn.start) + 0.02),
                              pitch=nn.pitch, velocity=nn.velocity))
            staff.append(si + 1)
    order = sorted(range(len(notes)), key=lambda i: (notes[i].onset, notes[i].pitch))
    notes = [notes[i] for i in order]
    staff = [staff[i] for i in order]
    for i, nn in enumerate(notes):
        nn.id = f"dev-{i:06d}"
    return Track(name="piano", preset="piano", notes=notes), staff


def m2s_binary() -> str:
    """The Rust `m2s` binary (build it with `cargo build --release`)."""
    import os
    import shutil

    for cand in (os.environ.get("M2S_BIN"), "target/release/m2s", "target/debug/m2s", shutil.which("m2s")):
        if cand and Path(cand).exists():
            return str(cand)
    raise FileNotFoundError("build the Rust binary first: cargo build --release")


def rust_decode(track: Track, weights: dict | None = None) -> Track:
    """Decode a track with the Rust solvers (`m2s decode-file`)."""
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.json", Path(td) / "out.json"
        Piece([track]).save(inp)
        cmd = [m2s_binary(), "decode-file", str(inp), str(out)]
        if weights:
            wf = Path(td) / "w.json"
            wf.write_text(json.dumps(weights))
            cmd += ["--weights", str(wf)]
        subprocess.run(cmd, check=True)
        return Piece.load(out).tracks[0]
