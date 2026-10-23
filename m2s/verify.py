"""Stage 5: verification by resynthesis.

The decoded notes are rendered to audio, and the same transcription front end
runs on the render and on the stem. Comparing the two pitch-activation maps
(not the waveforms) cancels most of the timbre gap between a SoundFont and a
real instrument. The comparison is what accepts or rejects every edit.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from m2s import audio as A
from m2s.ir import Note, Track
from m2s.play.preset import load_preset

SF2_CANDIDATES = ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/soundfonts/FluidR3_GM.sf2",
                  "/usr/share/sounds/sf2/default-GM.sf2", "/usr/share/soundfonts/default.sf2"]
CELL = 0.05      # seconds per comparison cell
ON = 0.5         # activation counted as "sounding"


def soundfont() -> str:
    import os

    env = os.environ.get("M2S_SOUNDFONT")
    if env:
        return env
    for p in SF2_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError("no General MIDI SoundFont found; install fluid-soundfont-gm or set M2S_SOUNDFONT")


# ----------------------------------------------------------------------------- render
def render(notes: list[Note], preset: str, out: str | Path, pedal: list[tuple[float, float]] | None = None,
           t0: float = 0.0, t1: float | None = None) -> None:
    import pretty_midi

    p = load_preset(preset)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(p.get("gm_program", 0)), is_drum=p["family"] == "drums")
    for n in notes:
        if t1 is not None and (n.offset < t0 or n.onset > t1):
            continue
        inst.notes.append(pretty_midi.Note(velocity=int(np.clip(n.velocity, 1, 127)), pitch=int(n.pitch),
                                           start=max(0.0, n.onset - t0), end=max(0.01, n.offset - t0)))
    for a, b in pedal or []:
        if t1 is not None and (b < t0 or a > t1):
            continue
        inst.control_changes.append(pretty_midi.ControlChange(64, 127, max(0.0, a - t0)))
        inst.control_changes.append(pretty_midi.ControlChange(64, 0, max(0.0, b - t0)))
    pm.instruments.append(inst)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        pm.write(f.name)
        subprocess.run(["fluidsynth", "-ni", "-g", "0.8", "-r", "44100", "-F", str(out), soundfont(), f.name],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------------- front ends
def activations(wav: str | Path, preset: str) -> tuple[np.ndarray, float]:
    """(frames x 88 pitch activations for MIDI 21..108, frames per second)."""
    family = load_preset(preset)["family"]
    if family == "keyboard":
        from m2s.transcribe.piano import KONG_SR, run_kong

        y, sr = A.load(wav, sr=KONG_SR, mono=True)
        _, _, out = run_kong(y)
        return out["frame_output"].astype(np.float32), 100.0
    return basic_pitch_notes(wav)


def basic_pitch_notes(wav: str | Path) -> tuple[np.ndarray, float]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.constants import ANNOTATIONS_FPS
    from basic_pitch.inference import Model, run_inference

    model = _bp_model(Model, ICASSP_2022_MODEL_PATH)
    out = run_inference(str(wav), model)
    return out["note"].astype(np.float32), float(ANNOTATIONS_FPS)


_BP = {}


def _bp_model(Model: Any, default_path: Any) -> Any:
    if "m" not in _BP:
        path = Path(str(default_path))
        onnx = path if path.suffix == ".onnx" else path.parent / "nmp.onnx"
        _BP["m"] = Model(onnx if onnx.exists() else path)
    return _BP["m"]


def to_cells(act: np.ndarray, fps: float, n_cells: int | None = None) -> np.ndarray:
    step = fps * CELL
    n = n_cells or int(np.ceil(len(act) / step))
    out = np.zeros((n, act.shape[1]), dtype=np.float32)
    for i in range(n):
        a, b = int(i * step), int((i + 1) * step)
        if a < len(act):
            out[i] = act[a:max(b, a + 1)].max(axis=0)
    return out


# ------------------------------------------------------------------------- compare
@dataclass
class Match:
    precision: float
    recall: float
    f1: float
    missed: np.ndarray   # cells x pitches: sounding in the stem, absent from the render
    extra: np.ndarray    # cells x pitches: in the render, absent from the stem


def _dilate_t(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = x.copy()
    for s in range(1, k + 1):
        out[s:] |= x[:-s]
        out[:-s] |= x[s:]
    return out


def compare(stem_cells: np.ndarray, render_cells: np.ndarray) -> Match:
    n = min(len(stem_cells), len(render_cells))
    s = stem_cells[:n] >= ON
    r = render_cells[:n] >= ON
    sd, rd = _dilate_t(s), _dilate_t(r)
    tp_r = (s & rd).sum()
    tp_p = (r & sd).sum()
    recall = float(tp_r / max(s.sum(), 1))
    precision = float(tp_p / max(r.sum(), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return Match(precision, recall, f1, s & ~rd, r & ~sd)


def regions(m: Match, top: int = 15, window: float = 2.0) -> list[dict[str, Any]]:
    """Windows with the most disagreement, worst first."""
    per = int(window / CELL)
    out = []
    n = len(m.missed)
    for a in range(0, n, per // 2):
        miss = m.missed[a:a + per]
        ext = m.extra[a:a + per]
        bad = int(miss.sum() + ext.sum())
        if bad < 3:
            continue
        mp = sorted(set((np.nonzero(miss.any(axis=0))[0] + 21).tolist()))
        ep = sorted(set((np.nonzero(ext.any(axis=0))[0] + 21).tolist()))
        out.append({"start": round(a * CELL, 2), "end": round(min(a + per, n) * CELL, 2), "bad_cells": bad,
                    "missed_pitches": mp[:12], "extra_pitches": ep[:12]})
    out.sort(key=lambda r: -r["bad_cells"])
    # drop overlapping windows
    kept: list[dict[str, Any]] = []
    for r in out:
        if all(r["end"] <= k["start"] or r["start"] >= k["end"] for k in kept):
            kept.append(r)
        if len(kept) >= top:
            break
    return kept


def heatmap(m: Match, stem_cells: np.ndarray, out: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(m.missed)
    rows = max(1, int(np.ceil(n * CELL / 60)))
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.2 * rows), squeeze=False)
    img = np.zeros((n, 88, 3), dtype=np.float32) + 1.0
    sound = stem_cells[:n] >= ON
    img[sound] = (0.75, 0.75, 0.75)
    img[m.missed] = (0.85, 0.15, 0.15)
    img[m.extra] = (0.15, 0.35, 0.85)
    per = int(60 / CELL)
    for r in range(rows):
        ax = axes[r, 0]
        seg = img[r * per:(r + 1) * per]
        ax.imshow(np.transpose(seg, (1, 0, 2)), origin="lower", aspect="auto",
                  extent=(r * 60, r * 60 + len(seg) * CELL, 21, 109))
        ax.set_ylabel("MIDI")
    axes[0, 0].set_title(title + "  grey: agrees  red: in audio, missing from score  blue: in score, not in audio")
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(str(out), dpi=80)
    plt.close(fig)


def verify_track(track: Track, stem_wav: str | Path, workdir: str | Path,
                 stem_act: tuple[np.ndarray, float] | None = None) -> tuple[dict[str, Any], Match, np.ndarray]:
    workdir = Path(workdir)
    rwav = workdir / f"render_{track.name}.wav"
    render(track.kept(), track.preset, rwav, track.pedal)
    sa, sfps = stem_act if stem_act is not None else activations(stem_wav, track.preset)
    ra, rfps = activations(rwav, track.preset)
    sc = to_cells(sa, sfps)
    rc = to_cells(ra, rfps, n_cells=len(sc))
    m = compare(sc, rc)
    per_sec = []
    k = int(1 / CELL)
    for a in range(0, len(sc), k):
        mm = compare(sc[a:a + k], rc[a:a + k])
        per_sec.append(round(mm.f1, 3) if (sc[a:a + k] >= ON).any() or (rc[a:a + k] >= ON).any() else None)
    report = {"track": track.name, "precision": round(m.precision, 4), "recall": round(m.recall, 4),
              "audio_match": round(m.f1, 4), "per_second": per_sec, "regions": regions(m)}
    return report, m, sc


def local_match(track: Track, stem_cells: np.ndarray, t0: float, t1: float, workdir: str | Path) -> float:
    """Audio match of the render against the stem inside [t0, t1] (for edit acceptance)."""
    pad = 1.0
    a, b = max(0.0, t0 - pad), t1 + pad
    rwav = Path(workdir) / f"render_local_{track.name}.wav"
    render(track.kept(), track.preset, rwav, track.pedal, t0=a, t1=b)
    ra, rfps = activations(rwav, track.preset)
    rc = to_cells(ra, rfps)
    ia, ib = int(a / CELL), int(b / CELL)
    sc = stem_cells[ia:ib]
    return compare(sc, rc[: len(sc)]).f1
