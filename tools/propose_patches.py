"""Turn the verification report into patch proposals for review.

For each of the worst disagreement regions:
  * a *dropped* candidate whose pitch the stem shows but the score lacks -> propose restoring it;
  * a *kept* low-confidence note whose pitch the score has but the stem lacks -> propose deleting it.

Proposals are written as patch files; nothing is applied. A reviewer applies
them one at a time with `m2s edit`, and the verification gate decides.

    python tools/propose_patches.py runs/<hash> [--regions 10] [--out runs/<hash>/proposals]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m2s.ir import Piece


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--regions", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out or run / "proposals")
    out.mkdir(parents=True, exist_ok=True)
    piece = Piece.load(run / "notes.json")
    rep = json.loads((run / "verify.json").read_text())
    k = 0
    for tr_rep in rep["tracks"]:
        tr = piece.track(tr_rep["track"])
        for reg in tr_rep["regions"][: a.regions]:
            t0, t1 = reg["start"], reg["end"]
            for n in tr.notes:
                if not (t0 <= n.onset <= t1):
                    continue
                if not n.keep and n.pitch in reg["missed_pitches"] and "trill-note" not in n.tags:
                    patch = {"reason": f"verify region {t0}-{t1}s: stem has pitch {n.pitch}, score lacks it; "
                                       f"restoring dropped candidate {n.id} (conf {n.confidence})",
                             "ops": [{"op": "add", "track": tr.name, "onset": n.onset, "offset": n.offset,
                                      "pitch": n.pitch, "velocity": n.velocity}]}
                elif n.keep and n.pitch in reg["extra_pitches"] and n.confidence < 0.9:
                    patch = {"reason": f"verify region {t0}-{t1}s: score has pitch {n.pitch} without stem "
                                       f"evidence; deleting low-confidence note {n.id} (conf {n.confidence})",
                             "ops": [{"op": "delete", "id": n.id}]}
                else:
                    continue
                k += 1
                (out / f"p{k:03d}.json").write_text(json.dumps(patch, indent=1))
    print(f"{k} proposals in {out}")


if __name__ == "__main__":
    main()
