"""Fit the piano decoder's weights on the dev set by coordinate descent.

Tunes on all dev pieces except the held-out ones, then reports both. The
result is printed as a PianoWeights(...) literal to paste into the defaults.

    python tools/tune_hands.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from devset import load, pieces  # noqa: E402

from m2s.play.piano import PianoDecoder, PianoWeights  # noqa: E402

HELD_OUT = ("Mozart", "Schubert")
PARAMS = ["span_over_comfort", "leap", "speed", "stream", "crossing", "crossing_per_semitone", "pitch_prior",
          "middle", "hold_conflict", "busy", "rate", "rate_window", "idle", "stream_window"]
STEPS = [0.5, 0.75, 1.33, 2.0]


def score(weights: PianoWeights, data) -> float:
    ok = tot = 0
    for track, staff in data:
        for n in track.notes:
            n.effector = {}
            n.keep = True
        PianoDecoder(weights=weights).decode(track)
        ok += sum((n.effector.get("hand") == "R") == (s == 1) for n, s in zip(track.notes, staff))
        tot += len(staff)
    return ok / tot


def main() -> None:
    all_pieces = pieces()
    train = [load(p) for p in all_pieces if not any(h in str(p) for h in HELD_OUT)]
    test = [load(p) for p in all_pieces if any(h in str(p) for h in HELD_OUT)]
    w = PianoWeights()
    best = score(w, train)
    print(f"start train {best:.4f}  held-out {score(w, test):.4f}", flush=True)
    for rnd in range(2):
        for name in PARAMS:
            base = getattr(w, name)
            for f in STEPS:
                cand = dataclasses.replace(w, **{name: type(base)(base * f) if name != "middle" else base + (f - 1) * 8})
                s = score(cand, train)
                if s > best + 1e-4:
                    best, w = s, cand
                    print(f"round {rnd} {name} -> {getattr(w, name):.4g}  train {best:.4f}", flush=True)
    print(f"final train {best:.4f}  held-out {score(w, test):.4f}")
    print(json.dumps(dataclasses.asdict(w), indent=1))


if __name__ == "__main__":
    main()
