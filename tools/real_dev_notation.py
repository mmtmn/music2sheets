"""Compare notation options on the real-recording dev set without re-running the models.

Re-notates every finished run in the dev runs folder with the given `m2s notate` flags, scores
it with `m2s eval` (the reference alignment is cached per run), and prints the mean of each
measure. Use it after `python tools/real_dev.py` has produced the runs.

    python tools/real_dev_notation.py [--runs data/realdevruns] -- --beats dbn --pauses
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from devset import m2s_binary  # noqa: E402

from m2s.runs import file_hash  # noqa: E402

ROOT = Path("data/realdev")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/realdevruns")
    ap.add_argument("flags", nargs="*", help="extra flags for `m2s notate` (after --)")
    a = ap.parse_args()
    m2s = m2s_binary()
    rows = {}
    for audio in sorted((ROOT / "audio").glob("*.ogg")):
        key = audio.stem
        run = Path(a.runs) / file_hash(audio, 12)
        if not (run / "notes.json").exists():
            continue
        subprocess.run([m2s, "notate", str(run), "--force"] + a.flags, check=True, capture_output=True)
        subprocess.run([m2s, "eval", str(run), "--ref", str(ROOT / "refs" / f"{key}.krn"), "--no-mv2h"], check=True,
                       capture_output=True)
        ev = json.loads((run / "eval.json").read_text())
        s, sc, sl = ev["summary"], ev["score"], ev.get("score_level", {})
        rows[key] = {"f1_50": s["note_f1_50ms"], "f1_100": s["note_f1_100ms"], "hands": s["hand_agreement"],
                     "key_ok": sc["ours"]["key"] == sc["reference"]["key"],
                     "bar_err": abs(sc["ours"]["bars"] - sc["reference"]["bars"]) / max(sc["reference"]["bars"], 1),
                     "ts_ok": sc["ours"]["time_signature"] == sc["reference"]["time_signature"],
                     "score": sl.get("score"), "meter": sl.get("meter"), "value": sl.get("value"),
                     "voice": sl.get("voice")}
        r = rows[key]
        print(f"{key}  bars {sc['ours']['bars']}/{sc['reference']['bars']}  ts {sc['ours']['time_signature']}/"
              f"{sc['reference']['time_signature']}  score-level {r['score']} (meter {r['meter']}, value {r['value']}, "
              f"voice {r['voice']})", flush=True)
    if rows:
        n = len(rows)
        mean = {k: round(sum((r[k] or 0) for r in rows.values()) / n, 4)
                for k in ("f1_50", "f1_100", "hands", "bar_err", "score", "meter", "value", "voice")}
        mean["ts_ok"] = f"{sum(r['ts_ok'] for r in rows.values())}/{n}"
        mean["key_ok"] = f"{sum(r['key_ok'] for r in rows.values())}/{n}"
        print("MEAN", " ".join(a.flags) or "(defaults)", json.dumps(mean))


if __name__ == "__main__":
    main()
