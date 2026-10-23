"""Run and score the real-recording dev set (tools/fetch_real_dev.py) with the Rust pipeline.

    python tools/real_dev.py [--runs data/realdevruns] [--force] [--only 08-1,15-3] [--tag NAME]

Prints one row per movement and the mean; writes data/logs/real_dev_<tag>.json for comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from devset import m2s_binary  # noqa: E402

ROOT = Path("data/realdev")


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{r.stderr[-3000:]}")
    return r.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/realdevruns")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="current")
    a = ap.parse_args()
    m2s = m2s_binary()
    only = {x for x in a.only.split(",") if x}
    rows = {}
    for audio in sorted((ROOT / "audio").glob("*.ogg")):
        key = audio.stem
        if only and key not in only:
            continue
        log = run([m2s, "run", str(audio), "--instruments", "piano", "--runs-root", a.runs]
                  + (["--force"] if a.force else []))
        run_dir = next(line.split("run ")[1] for line in log.splitlines() if "] run " in line)
        run([m2s, "eval", run_dir, "--ref", str(ROOT / "refs" / f"{key}.krn"), "--no-mv2h"])
        ev = json.loads((Path(run_dir) / "eval.json").read_text())
        s, sc = ev["summary"], ev["score"]
        rows[key] = {
            "f1_50": s["note_f1_50ms"], "f1_100": s["note_f1_100ms"], "f1_250": s["note_f1_250ms"],
            "hands": s["hand_agreement"], "viol": s["violations_per_100_notes"]["ours"],
            "ref_viol": s["violations_per_100_notes"].get("reference"),
            "bars": [sc["ours"]["bars"], sc["reference"]["bars"]],
            "ts": [sc["ours"]["time_signature"], sc["reference"]["time_signature"]],
            "key": [sc["ours"]["key"], sc["reference"]["key"]],
            "score": ev.get("score_level", {}).get("score"),
        }
        r = rows[key]
        print(f"{key}  F1 {r['f1_50']:.3f}/{r['f1_100']:.3f}/{r['f1_250']:.3f}  hands {r['hands']}  "
              f"viol {r['viol']} (ref {r['ref_viol']})  bars {r['bars'][0]}/{r['bars'][1]}  ts {r['ts'][0]}/{r['ts'][1]}  "
              f"key {r['key'][0]}/{r['key'][1]}  score-level {r['score']}", flush=True)
    if rows:
        mean = lambda k: sum(r[k] for r in rows.values() if r[k] is not None) / max(1, sum(r[k] is not None for r in rows.values()))  # noqa: E731
        ts_ok = sum(r["ts"][0] == r["ts"][1] for r in rows.values())
        key_ok = sum(r["key"][0] == r["key"][1] for r in rows.values())
        bar_err = sum(abs(r["bars"][0] - r["bars"][1]) / r["bars"][1] for r in rows.values()) / len(rows)
        summary = {"f1_50": mean("f1_50"), "f1_100": mean("f1_100"), "f1_250": mean("f1_250"), "hands": mean("hands"),
                   "viol": mean("viol"), "ts_ok": f"{ts_ok}/{len(rows)}", "key_ok": f"{key_ok}/{len(rows)}",
                   "bar_count_error": bar_err}
        if any(r["score"] is not None for r in rows.values()):
            summary["score_level"] = mean("score")
        print("MEAN", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}))
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path(f"data/logs/real_dev_{a.tag}.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
