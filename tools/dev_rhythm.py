"""Rhythm-stage check on the dev set (rendered ASAP performances).

For each dev piece already run through `m2s run --until verify`, re-run the
notation stage and compare with ASAP ground truth: time signature, bar count,
beat F-measure of the refined beats, and note F1 of the transcription against
the performance MIDI.

    python tools/dev_rhythm.py [--runs data/devruns]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from devset import pieces  # noqa: E402

from m2s.ir import Piece  # noqa: E402
from m2s.runs import Run, file_hash  # noqa: E402


def gt(folder: Path) -> dict:
    perf = (folder / "performance.txt").read_text().strip()
    rows = [ln.split("\t") for ln in (folder / (Path(perf).stem + "_annotations.txt")).read_text().splitlines()
            if ln.strip()]
    beats = [float(r[0]) for r in rows]
    dbs = [r for r in rows if r[2].startswith("db")]
    ts = next((r[2].split(",")[1] for r in dbs if len(r[2].split(",")) > 1), None)
    return {"beats": np.array(beats), "bars": len(dbs), "ts": ts, "midi": folder / perf}


def note_f1(notes, midi: Path) -> float:
    import mir_eval
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi))
    ref = [n for i in pm.instruments for n in i.notes]
    ri = np.array([[n.start, n.end] for n in ref])
    rp = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in ref])
    ei = np.array([[n.onset, n.offset] for n in notes])
    ep = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    return mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)[2]


def main(runs_root: str = "data/devruns") -> None:
    import mir_eval

    from m2s.notate import notate

    rows = []
    for folder in pieces():
        wav = folder / "performance.wav"
        if not wav.exists():
            continue
        run = Run(Path(runs_root) / file_hash(wav, 12))
        if not (run.root / "notes.json").exists():
            continue
        prof = run.read_json("profile.json")
        piece = Piece.load(run.root / "notes.json")
        acts = np.load(run.root / "acts" / "beats.npy").astype(np.float32)
        g = gt(folder)
        kept = [n for t in piece.tracks for n in t.kept()]
        f1 = note_f1(kept, g["midi"])
        score, grid, info = notate(piece, prof["beats"], prof["downbeats"], beat_acts=acts,
                                   act_fps=prof.get("beat_act_fps", 50.0), duration=prof["duration"])
        bars = len(score.parts[0].getElementsByClass("Measure"))
        # beat F-measure: our quarter-note beats vs annotated beats
        q_beats = grid.beat_times[:: max(1, int(round(1 / grid.unit)))] if grid.unit < 1 else grid.beat_times
        if grid.unit > 1:  # tracked half notes: interpolate quarters
            q_beats = np.sort(np.concatenate([q_beats, (q_beats[1:] + q_beats[:-1]) / 2]))
        fm = mir_eval.beat.f_measure(g["beats"], q_beats)
        raw = mir_eval.beat.f_measure(g["beats"], np.array(prof["beats"]))
        name = str(folder).split("asap/")[1]
        rows.append((name, f1, g["ts"], info["time_signature"], g["bars"], bars, raw, fm, info["qpm"]))
        print(f"{name:36s} noteF1 {f1:.3f}  ts {g['ts']:>4s}->{info['time_signature']:>4s}  bars {g['bars']:4d}->"
              f"{bars:4d}  beatF raw {raw:.2f} refined {fm:.2f}  q={info['qpm']:.0f}", flush=True)
    ok_ts = sum(r[2] == r[3] for r in rows)
    print(f"time signature correct {ok_ts}/{len(rows)}; mean noteF1 {np.mean([r[1] for r in rows]):.3f}; "
          f"mean beatF raw {np.mean([r[6] for r in rows]):.2f} refined {np.mean([r[7] for r in rows]):.2f}")
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/logs/dev_rhythm.json").write_text(json.dumps(rows, default=float, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
