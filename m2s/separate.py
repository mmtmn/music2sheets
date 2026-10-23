"""Stage 2: source separation for band recordings.

Two Demucs models vote on the stems they share (drums, bass, vocals): the
6-source model (which also gives guitar and piano) and the fine-tuned
4-source model. Their average is the stem; their agreement per window is its
confidence. A guitar stem is then split by stereo position, because rock mixes
usually pan doubled or harmonized guitars apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from m2s import audio as A

# detected instrument -> stem name
STEM_OF = {"piano": "piano", "synth": "piano", "electric_guitar": "guitar", "guitar": "guitar", "bass": "bass",
           "drums": "drums", "voice": "vocals", "violin": "other", "sax_alto": "other", "sax_tenor": "other"}


def _separate(model: str, wav: Path) -> tuple[dict[str, np.ndarray], int]:
    import torch
    from demucs.api import Separator

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # shifts=0: Demucs' default applies a random time shift per run, which makes stems (and every
    # later stage) differ between identical runs
    torch.manual_seed(0)
    sep = Separator(model=model, device=dev, shifts=0, progress=False)
    _, stems = sep.separate_audio_file(str(wav))
    out = {k: v.cpu().numpy().T.astype(np.float32) for k, v in stems.items()}
    sr = sep.samplerate
    del sep
    torch.cuda.empty_cache()
    return out, sr


def agreement(a: np.ndarray, b: np.ndarray, sr: int, window: float = 2.0) -> list[float]:
    """Per-window correlation of two estimates of the same stem (1 = identical)."""
    n = int(window * sr)
    out = []
    for i in range(0, min(len(a), len(b)), n):
        x, y = a[i:i + n].mean(axis=1), b[i:i + n].mean(axis=1)
        den = np.sqrt((x * x).sum() * (y * y).sum())
        out.append(round(float((x * y).sum() / den), 3) if den > 1e-6 else 1.0)
    return out


def pan_split(y: np.ndarray, sr: int, edges: tuple[float, float] = (0.4, 0.6)) -> dict[str, np.ndarray]:
    """Split a stereo stem into left / centre / right by per-bin pan position (ADRess-style)."""
    import librosa

    L = librosa.stft(y[:, 0], n_fft=4096, hop_length=1024)
    R = librosa.stft(y[:, 1], n_fft=4096, hop_length=1024)
    pan = np.abs(R) / (np.abs(L) + np.abs(R) + 1e-9)          # 0 = hard left, 1 = hard right
    out = {}
    for name, mask in (("left", pan < edges[0]), ("centre", (pan >= edges[0]) & (pan <= edges[1])),
                       ("right", pan > edges[1])):
        l = librosa.istft(L * mask, hop_length=1024, length=len(y))
        r = librosa.istft(R * mask, hop_length=1024, length=len(y))
        out[name] = np.stack([l, r], axis=1).astype(np.float32)
    return out


def separate_band(wav: str | Path, outdir: str | Path, present: list[str],
                  log: Callable[[str], None] = print) -> dict[str, str]:
    """Returns {track name: stem wav}. Track names are presets, or preset@L / preset@R for split guitars."""
    wav, outdir = Path(wav), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    six, sr = _separate("htdemucs_6s", wav)
    ft, sr2 = _separate("htdemucs_ft", wav)
    assert sr == sr2
    stems: dict[str, np.ndarray] = {}
    conf: dict[str, list[float]] = {}
    for name in ("drums", "bass", "vocals"):
        stems[name] = 0.5 * (six[name] + ft[name])
        conf[name] = agreement(six[name], ft[name], sr)
    stems["guitar"], stems["piano"], stems["other"] = six["guitar"], six["piano"], 0.5 * (six["other"] + ft["other"])
    A.save(outdir / "residual.wav", A.load(wav, sr=sr)[0][: len(stems["drums"])] - sum(
        stems[k] for k in ("drums", "bass", "vocals", "guitar", "piano", "other")), sr)

    out: dict[str, str] = {}
    for preset in present:
        stem = STEM_OF.get(preset)
        if stem is None or preset in out:
            continue
        y = stems[stem]
        if preset in ("electric_guitar", "guitar"):
            parts = pan_split(y, sr)
            e = {k: float((v ** 2).sum()) for k, v in parts.items()}
            tot = sum(e.values()) + 1e-9
            if e["left"] / tot > 0.2 and e["right"] / tot > 0.2:
                for side in ("left", "right"):
                    p = outdir / f"{preset}_{side}.wav"
                    A.save(p, parts[side] + 0.5 * parts["centre"], sr)
                    out[f"{preset}@{side[0].upper()}"] = str(p)
                log(f"separate: {preset} split by panning (L {e['left'] / tot:.0%}, R {e['right'] / tot:.0%})")
                continue
        p = outdir / f"{preset}.wav"
        A.save(p, y, sr)
        out[preset] = str(p)
    (outdir / "confidence.json").write_text(__import__("json").dumps(conf))
    return out
