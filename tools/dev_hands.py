"""Hand-assignment accuracy on the dev set (Rust decoder), against a middle-C split baseline.

    python tools/dev_hands.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from devset import load, pieces, rust_decode  # noqa: E402


def evaluate(weights: dict | None = None, verbose: bool = True) -> float:
    tot = ok = base_ok = 0
    for folder in pieces():
        track, staff = load(folder)
        t0 = time.time()
        track = rust_decode(track, weights)
        dt = time.time() - t0
        pred = [1 if n.effector.get("hand") == "R" else 2 for n in track.notes]
        acc = sum(p == s for p, s in zip(pred, staff)) / len(staff)
        base = sum((1 if n.pitch >= 60 else 2) == s for n, s in zip(track.notes, staff)) / len(staff)
        tot += len(staff)
        ok += acc * len(staff)
        base_ok += base * len(staff)
        if verbose:
            print(f"{str(folder).split('asap/')[1]:42s} notes {len(staff):5d}  hands {acc:6.1%}  "
                  f"split@C4 {base:6.1%}  {dt:5.2f}s")
    if verbose:
        print(f"{'ALL':42s} notes {tot:5d}  hands {ok / tot:6.1%}  split@C4 {base_ok / tot:6.1%}")
    return ok / tot


if __name__ == "__main__":
    evaluate()
