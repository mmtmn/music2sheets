"""Fit the piano decoder's weights on the dev set by coordinate descent.

Tunes on all dev pieces except the held-out ones, then reports both. Decoding
runs in the Rust binary (`m2s decode-file --weights`). The result is printed as
JSON; copy it into PianoWeights::default() in crates/m2s-core/src/play/piano.rs.

    python tools/tune_hands.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from devset import load, pieces, rust_decode  # noqa: E402

# current defaults, mirrored from crates/m2s-core/src/play/piano.rs (PianoWeights::default)
DEFAULTS = {"span_over_comfort": 1.2, "leap": 0.06, "speed": 0.0225, "stream": 0.9, "crossing": 3.0,
            "crossing_per_semitone": 0.6, "pitch_prior": 0.01, "middle": 56.0, "hold_conflict": 0.5, "busy": 0.6,
            "rate": 0.375, "rate_window": 0.06, "idle": 0.4, "stream_window": 9.0}

HELD_OUT = ("Mozart", "Schubert")
PARAMS = ["span_over_comfort", "leap", "speed", "stream", "crossing", "crossing_per_semitone", "pitch_prior",
          "middle", "hold_conflict", "busy", "rate", "rate_window", "idle", "stream_window"]
STEPS = [0.5, 0.75, 1.33, 2.0]


def score(weights: dict, data) -> float:
    ok = tot = 0
    for track, staff in data:
        out = rust_decode(track, weights)
        hand = {n.id: n.effector.get("hand") for n in out.notes}
        ok += sum((hand[n.id] == "R") == (s == 1) for n, s in zip(track.notes, staff))
        tot += len(staff)
    return ok / tot


def main() -> None:
    all_pieces = pieces()
    train = [load(p) for p in all_pieces if not any(h in str(p) for h in HELD_OUT)]
    test = [load(p) for p in all_pieces if any(h in str(p) for h in HELD_OUT)]
    w = dict(DEFAULTS)
    best = score(w, train)
    print(f"start train {best:.4f}  held-out {score(w, test):.4f}", flush=True)
    for rnd in range(2):
        for name in PARAMS:
            base = w[name]
            for f in STEPS:
                cand = dict(w, **{name: base * f if name != "middle" else base + (f - 1) * 8})
                s = score(cand, train)
                if s > best + 1e-4:
                    best, w = s, cand
                    print(f"round {rnd} {name} -> {w[name]:.4g}  train {best:.4f}", flush=True)
    print(f"final train {best:.4f}  held-out {score(w, test):.4f}")
    print(json.dumps(w, indent=1))


if __name__ == "__main__":
    main()
