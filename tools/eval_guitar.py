"""Electric-guitar transcription on held-out real data: stock Basic Pitch vs the fine-tuned model.

Both go through the worker's own path (m2s.transcribe.band.polyphonic: overtone suppression,
onset-checked confidence, retrigger merging) and keep notes with confidence >= 0.5, as the
fretted decoder does. Scores are mir_eval note onset P/R/F1 at 50 and 100 ms.

    python tools/eval_guitar.py [--model models/guitar_bp.onnx] [--sets techs-test techs-test-di egdb-test egdb-real]
                                [--align]   # per-file label offset from spectral-flux onsets (Guitar-TECHS labels lag ~25 ms)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train_guitar import (aligned_notes, egdb_items, guitarset_items, label_shift, midi_notes,  # noqa: E402
                          split_items, techs_items)

SETS = {"val": lambda: [x for x in split_items()[1] if "directinput" not in x[2] and "egdb-DI" not in x[2]],"techs-test": lambda: techs_items("test"), "techs-test-di": lambda: techs_items("test", ("directinput",)),
        "egdb-test": lambda: egdb_items("test"), "egdb-real": lambda: egdb_items("real"),
        "guitarset": lambda: guitarset_items()}


def score(ref, est, tol: float) -> tuple[int, int, int]:
    import mir_eval

    if not ref or not est:
        return 0, len(est), len(ref)
    ri = np.array([[s, max(e, s + 0.01)] for s, e, _ in ref])
    rp = 440.0 * 2 ** ((np.array([p for *_, p in ref], float) - 69) / 12)
    ei = np.array([[s, max(e, s + 0.01)] for s, e, _ in est])
    ep = 440.0 * 2 ** ((np.array([p for *_, p in est], float) - 69) / 12)
    m = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=tol, offset_ratio=None)
    return len(m), len(est), len(ref)


def transcribe(wav: Path, model: str | None) -> list[tuple[float, float, int]]:
    from m2s.transcribe.band import polyphonic

    notes = polyphonic(wav, "electric_guitar", None, model_path=model)
    return [(n.onset, n.offset, n.pitch) for n in notes if n.confidence >= 0.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/guitar_bp.onnx")
    ap.add_argument("--sets", nargs="*", default=[k for k in SETS if k != "val"])
    ap.add_argument("--out", default="data/logs/eval_guitar.json")
    ap.add_argument("--align", action="store_true",
                    help="shift each file's labels by its audio-derived onset offset (same for every system)")
    a = ap.parse_args()
    cache: dict = {}
    systems = {"stock": "stock"}
    if Path(a.model).exists():
        systems["finetuned"] = a.model
    results = {}
    for name in a.sets:
        items = SETS[name]()
        if not items:
            print(f"{name}: no data (run tools/train_guitar.py download)")
            continue
        shifts = [label_shift(wav, midi_notes(mid)) for wav, mid, _ in items]
        print(f"{name}: label vs audio-onset offset, median over files {np.median(shifts) * 1000:+.1f} ms"
              + ("" if a.align else " (not applied; --align applies it)"), flush=True)
        for sys_name, model in systems.items():
            tot = {0.05: [0, 0, 0], 0.1: [0, 0, 0]}
            for wav, mid, _ in items:
                ref = aligned_notes(wav, mid) if a.align else midi_notes(mid)
                est = cache.get((str(wav), sys_name))
                if est is None:
                    est = cache[(str(wav), sys_name)] = transcribe(wav, None if model == "stock" else model)
                for tol in tot:
                    m, ne, nr = score(ref, est, tol)
                    tot[tol][0] += m
                    tot[tol][1] += ne
                    tot[tol][2] += nr
            row = {}
            for tol, (m, ne, nr) in tot.items():
                p, r = m / max(ne, 1), m / max(nr, 1)
                f = 2 * p * r / max(p + r, 1e-9)
                row[f"{int(tol * 1000)}ms"] = {"P": round(p, 3), "R": round(r, 3), "F1": round(f, 3)}
            results.setdefault(name, {})[sys_name] = row | {"files": len(items)}
            print(f"{name:11s} {sys_name:9s} files {len(items):3d}  "
                  f"50ms P/R/F {row['50ms']['P']:.3f}/{row['50ms']['R']:.3f}/{row['50ms']['F1']:.3f}  "
                  f"100ms F {row['100ms']['F1']:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
