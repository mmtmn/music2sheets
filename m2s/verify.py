"""Model front ends for verification: frame activations (frames x 88 pitches) of a WAV.

The Rust core renders the decoded notes, asks the worker for the activations of
both the render and the stem (`python -m m2s.worker activations`), and compares
them. Using the same front end on both sides cancels most of the timbre gap
between a SoundFont render and the recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from m2s import audio as A
from m2s.presets import load_preset


def activations(wav: str | Path, preset: str) -> tuple[np.ndarray, float]:
    """(frames x 88 pitch activations for MIDI 21..108, frames per second)."""
    family = load_preset(preset)["family"]
    if family == "keyboard":
        # onset activations: unlike frame activations they don't depend on how long a note rings,
        # so pedal sustain in a SoundFont render can't masquerade as extra notes
        from m2s.transcribe.piano import KONG_SR, run_kong

        y, sr = A.load(wav, sr=KONG_SR, mono=True)
        _, _, out = run_kong(y)
        return out["reg_onset_output"].astype(np.float32), 100.0
    if family == "drums":
        return drum_activations(wav)
    return basic_pitch_notes(wav)


# ADTOF's five classes, placed in the 88-wide grid at their GM drum notes (kick, snare, tom, hi-hat, cymbal)
DRUM_COLS = [36 - 21, 38 - 21, 47 - 21, 42 - 21, 49 - 21]


def drum_activations(wav: str | Path) -> tuple[np.ndarray, float]:
    """Drums are compared in ADTOF's own activation space, the same front end that transcribed them."""
    import tempfile

    import torch
    from adtof_pytorch import transcribe_to_midi

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as td:
        res = transcribe_to_midi(str(wav), Path(td) / "d.mid", device=dev, return_activations=True)
    act = res[1] if isinstance(res, tuple) else res
    act = np.asarray(act, dtype=np.float32)
    if act.ndim == 3:
        act = act[0]
    out = np.zeros((len(act), 88), dtype=np.float32)
    for k, col in enumerate(DRUM_COLS[: act.shape[1]]):
        out[:, col] = act[:, k]
    return out, 100.0


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
