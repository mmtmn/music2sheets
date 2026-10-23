"""Stage 1 analysis: tuning offset, instrument presence, beats and downbeats.

Key and form are estimated later from the decoded notes (see m2s.notate),
where they are far more reliable than from raw audio.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np

from m2s import audio as A

# AudioSet labels -> our presets. Several labels can vote for one preset.
TAGS: dict[str, list[str]] = {
    "piano": ["Piano", "Keyboard (musical)"],
    "electric_guitar": ["Electric guitar", "Tapping (guitar technique)"],
    "guitar": ["Acoustic guitar", "Guitar", "Steel guitar, slide guitar"],
    "bass": ["Bass guitar"],
    "drums": ["Drum kit", "Drum", "Bass drum", "Drum roll"],
    "voice": ["Singing", "Vocal music"],
    "violin": ["Violin, fiddle"],
    "sax_alto": ["Saxophone"],
    "synth": ["Synthesizer", "Organ"],
}
PRESENT = 0.12  # mean window probability above which an instrument counts as present


def estimate_tuning(y_mono: np.ndarray, sr: int) -> float:
    """Deviation from A4 = 440 Hz, in cents."""
    import librosa

    seg = y_mono[: sr * 120]
    return float(round(100 * librosa.estimate_tuning(y=seg, sr=sr, resolution=0.01, bins_per_octave=12), 1))


def instrument_presence(y_mono: np.ndarray, sr: int, window: float = 2.0) -> dict[str, Any]:
    """Run an AudioSet tagger over 2 s windows; returns per-preset mean and timeline."""
    import torch
    from panns_inference import AudioTagging, labels

    import soxr

    y = soxr.resample(y_mono, sr, 32000).astype(np.float32)
    hop = int(32000 * window)
    frames = [y[i: i + hop] for i in range(0, max(len(y) - hop // 2, 1), hop)]
    frames = [np.pad(f, (0, hop - len(f))) for f in frames]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        at = AudioTagging(checkpoint_path=None, device=dev)
    probs = []
    for i in range(0, len(frames), 32):
        clip, _ = at.inference(np.stack(frames[i: i + 32]))
        probs.append(clip)
    probs = np.concatenate(probs)
    idx = {lab: i for i, lab in enumerate(labels)}
    timeline: dict[str, list[float]] = {}
    for preset, labs in TAGS.items():
        cols = [idx[lab] for lab in labs if lab in idx]
        timeline[preset] = [round(float(x), 3) for x in probs[:, cols].max(axis=1)]
    mean = {k: round(float(np.mean(v)), 3) for k, v in timeline.items()}
    del at
    torch.cuda.empty_cache()
    return {"window": window, "mean": mean, "timeline": timeline,
            "present": sorted(k for k, v in mean.items() if v >= PRESENT)}


def track_beats(wav: str | Path) -> tuple[list[float], list[float], np.ndarray, float]:
    """Neural beat tracking. Returns beats, downbeats, frame activations (frames x 2), fps."""
    import torch
    from beat_this.inference import Audio2Frames, Postprocessor, load_audio

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a2f = Audio2Frames(checkpoint_path="final0", device=dev)
    signal, sr = load_audio(str(wav))
    with torch.no_grad():
        beat_logits, db_logits = a2f(signal, sr)
    beats, downbeats = Postprocessor(type="minimal", fps=50)(beat_logits, db_logits)
    acts = torch.sigmoid(torch.stack([beat_logits, db_logits], dim=-1)).float().cpu().numpy()
    del a2f
    torch.cuda.empty_cache()
    return ([round(float(b), 4) for b in beats], [round(float(b), 4) for b in downbeats], acts, 50.0)


def analyze(wav: str | Path, instruments: list[str] | None = None, act_path: str | Path | None = None
            ) -> dict[str, Any]:
    y, sr = A.load(wav)
    mono = y.mean(axis=1)
    prof: dict[str, Any] = {"duration": round(len(mono) / sr, 3), "sr": sr}
    prof["tuning_cents"] = estimate_tuning(mono, sr)
    if instruments:
        prof["instruments"] = {"present": instruments, "source": "user"}
    else:
        prof["instruments"] = instrument_presence(mono, sr) | {"source": "audioset-tagger"}
    beats, downbeats, acts, fps = track_beats(wav)
    prof["beats"], prof["downbeats"] = beats, downbeats
    prof["beat_act_fps"] = fps
    if act_path is not None:
        np.save(act_path, acts.astype(np.float16))
    if len(beats) > 2:
        ibi = np.diff(beats)
        prof["tempo_bpm"] = round(float(60 / np.median(ibi)), 1)
    return prof


def solo(profile: dict[str, Any]) -> str | None:
    """The single instrument of a solo recording, or None for an ensemble."""
    present = [p for p in profile["instruments"]["present"] if p != "synth"]
    if len(present) == 1:
        return present[0]
    mean = profile["instruments"].get("mean")
    if mean and present:
        top = max(present, key=lambda k: mean[k])
        others = [k for k in present if k != top and mean[k] > 0.3 * mean[top]]
        if not others:
            return top
    return None
