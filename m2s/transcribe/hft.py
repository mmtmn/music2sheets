"""hFT-Transformer (Toyama et al., ISMIR 2023; github.com/sony/hFT-Transformer), the third piano model.

Fetched by tools/fetch_hft.sh into third_party/hFT-Transformer (code + the `ismir2023`
checkpoint). If it isn't there, `available()` is False and piano fusion uses its two models.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from m2s.ir import Note

HFT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "hFT-Transformer"
CKPT = HFT_DIR / "checkpoint" / "MAESTRO-V3" / "model_016_003.pkl"
SR = 16000
HOP = 256
FPS = SR / HOP  # 62.5 frames per second


def available() -> bool:
    return CKPT.exists() and (HFT_DIR / "model" / "amt.py").exists()


def _config() -> dict:
    cfg = json.loads((HFT_DIR / "corpus" / "config.json").read_text())
    # as corpus/make_dataset.py writes it for the released model: raw log-mel input (max_value 0),
    # padding with log(log_offset)
    cfg["input"]["max_value"] = 0.0
    cfg["input"]["min_value"] = float(np.log(cfg["feature"]["log_offset"]))
    return cfg


def run_hft(y_mono_16k: np.ndarray) -> tuple[list[Note], np.ndarray]:
    """Notes and the second-stage onset map (frames x 88 at 62.5 fps) for a 16 kHz mono signal."""
    import torch
    import torchaudio

    if str(HFT_DIR) not in sys.path:
        sys.path.insert(0, str(HFT_DIR))
    from model import amt  # noqa: E402  (hFT's own package; the pickle refers to it)

    cfg = _config()
    with contextlib.redirect_stdout(io.StringIO()):
        model = amt.AMT(cfg, str(CKPT), verbose_flag=False)
    f = cfg["feature"]
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=f["sr"], n_fft=f["fft_bins"], win_length=f["window_length"],
                                               hop_length=f["hop_sample"], pad_mode=f["pad_mode"], n_mels=f["mel_bins"],
                                               norm="slaney")
    wave = torch.from_numpy(np.ascontiguousarray(y_mono_16k, dtype=np.float32))
    feature = torch.log(mel(wave) + f["log_offset"]).T
    with torch.no_grad():
        _, _, _, _, onset, offset, mpe, velocity = model.transcript(feature.numpy(), mode="combination")
    raw = model.mpe2note(a_onset=onset, a_offset=offset, a_mpe=mpe, a_velocity=velocity, thred_onset=0.5,
                         thred_offset=0.5, thred_mpe=0.5, mode_velocity="ignore_zero", mode_offset="shorter")
    n_frames = feature.shape[0]
    notes = [Note(id="", onset=float(r["onset"]), offset=float(max(r["offset"], r["onset"] + 0.01)),
                  pitch=int(r["pitch"]), velocity=int(np.clip(r["velocity"], 1, 127)), provenance="hft")
             for r in raw]
    del model
    torch.cuda.empty_cache()
    return notes, np.asarray(onset[:n_frames], dtype=np.float32)
