"""Audio-to-score alignment with synctoolbox (multiresolution DTW, chroma + DLNCO onset features).

The reference side is built from the symbolic notes; the recording side from
audio. Our transcription is not used, so alignment can't flatter it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from m2s import audio as A
from m2s.evaluate.reference import Reference

RATE = 50


def _audio_features(wav: str | Path) -> tuple[np.ndarray, np.ndarray]:
    from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
    from synctoolbox.feature.pitch import audio_to_pitch_features
    from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
    from synctoolbox.feature.utils import estimate_tuning

    y, sr = A.load(wav, sr=22050, mono=True)
    y = y.astype(np.float64)
    tuning = estimate_tuning(y, sr)
    f_pitch = audio_to_pitch_features(f_audio=y, Fs=sr, tuning_offset=tuning, feature_rate=RATE, verbose=False)
    chroma = quantize_chroma(f_chroma=pitch_to_chroma(f_pitch=f_pitch))
    onsets = audio_to_pitch_onset_features(f_audio=y, Fs=sr, tuning_offset=tuning, verbose=False)
    dlnco = pitch_onset_features_to_DLNCO(f_peaks=onsets, feature_rate=RATE,
                                          feature_sequence_length=chroma.shape[1], visualize=False)
    return chroma, dlnco


def _score_features(ref: Reference) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
    from synctoolbox.feature.csv_tools import df_to_pitch_features, df_to_pitch_onset_features
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO

    df = pd.DataFrame({"start": [n.sec_on for n in ref.notes],
                       "duration": [max(n.sec_off - n.sec_on, 0.05) for n in ref.notes],
                       "pitch": [n.pitch for n in ref.notes], "velocity": [80] * len(ref.notes),
                       "instrument": ["piano"] * len(ref.notes)})
    f_pitch = df_to_pitch_features(df, feature_rate=RATE)
    chroma = quantize_chroma(f_chroma=pitch_to_chroma(f_pitch=f_pitch))
    onsets = df_to_pitch_onset_features(df)
    dlnco = pitch_onset_features_to_DLNCO(f_peaks=onsets, feature_rate=RATE,
                                          feature_sequence_length=chroma.shape[1], visualize=False)
    return chroma, dlnco


def align_reference(ref: Reference, wav: str | Path) -> tuple[Reference, dict[str, Any]]:
    from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw

    sc, so = _score_features(ref)
    ac, ao = _audio_features(wav)
    # A recording tuned about half a semitone off A440 can have its chroma land one bin away from
    # the score's; then DTW aligns against the wrong notes. Undo that with the semitone roll (-1, 0,
    # +1) whose overall pitch-class profile best matches the score's.
    prof_s = sc.sum(axis=1)
    roll = max((0, -1, 1), key=lambda r: float(np.corrcoef(np.roll(ac.sum(axis=1), r), prof_s)[0, 1]))
    if roll:
        ac, ao = np.roll(ac, roll, axis=0), np.roll(ao, roll, axis=0)
    wp = sync_via_mrmsdtw(f_chroma1=sc, f_onset1=so, f_chroma2=ac, f_onset2=ao, input_feature_rate=RATE,
                          step_weights=np.array([1.5, 1.5, 2.0]), threshold_rec=10 ** 6, verbose=False)
    src = wp[0] / RATE
    dst = wp[1] / RATE
    # make the map strictly increasing for interpolation
    keep = np.concatenate([[True], np.diff(src) > 0])
    src, dst = src[keep], dst[keep]
    for n in ref.notes:
        n.on = float(np.interp(n.sec_on, src, dst))
        n.off = max(float(np.interp(n.sec_off, src, dst)), n.on + 0.03)
    info = {"method": "synctoolbox MrMsDTW (chroma + DLNCO), 50 Hz", "path_len": int(len(src)), "chroma_roll": roll,
            "ref_seconds": round(float(src[-1]), 1), "audio_seconds": round(float(dst[-1]), 1)}
    return ref, info
