"""Audio IO helpers. Everything internal is float32, shape (samples, channels)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def decode(src: str | Path, dst: str | Path, sr: int = SR) -> None:
    """Decode any ffmpeg-readable file to float WAV, stereo, ``sr`` Hz, tags stripped."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map_metadata", "-1", "-ac", "2", "-ar", str(sr),
         "-c:a", "pcm_f32le", str(dst)],
        check=True,
    )


def load(path: str | Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    y, fs = sf.read(str(path), dtype="float32", always_2d=True)
    if sr is not None and fs != sr:
        import soxr

        y = soxr.resample(y, fs, sr).astype(np.float32)
        fs = sr
    if mono:
        y = y.mean(axis=1)
    return y, fs


def save(path: str | Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y, sr, subtype="FLOAT")
