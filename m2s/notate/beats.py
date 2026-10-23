"""Beat helpers on top of the neural tracker's frame activations."""

from __future__ import annotations

import numpy as np


def downbeat_strength(beats: np.ndarray, db_act: np.ndarray | None, act_fps: float) -> np.ndarray:
    """Peak downbeat activation within +-60 ms of each beat."""
    if db_act is None or not len(db_act):
        return np.zeros(len(beats))
    out = np.zeros(len(beats))
    for i, b in enumerate(beats):
        f = int(round(b * act_fps))
        lo, hi = max(0, f - 3), min(len(db_act), f + 4)
        out[i] = float(db_act[lo:hi].max()) if hi > lo else 0.0
    return out
