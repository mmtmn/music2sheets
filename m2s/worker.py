"""Model worker for the Rust `m2s` CLI.

The Rust side owns the pipeline, caching and every deterministic stage. It calls
this worker for the parts that need the neural models or Python-only libraries:

    python -m m2s.worker analyze      --run DIR [--instruments a,b]
    python -m m2s.worker separate     --run DIR
    python -m m2s.worker transcribe   --run DIR
    python -m m2s.worker activations  --wav FILE --preset NAME --out FILE.npy
    python -m m2s.worker engrave      --musicxml FILE --pdf FILE [--engine auto|musescore|verovio]
    python -m m2s.worker inspect      --run DIR --bars A-B [--out FILE]
    python -m m2s.worker align        --run DIR --ref FILE [--repeats auto] --out FILE.json
    python -m m2s.worker mv2h         --ours FILE --ref FILE
    python -m m2s.worker musicdiff    --ours FILE --ref FILE --bars N

Each command writes its files and prints one JSON object as its last stdout line.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _emit(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, default=_default) + "\n")
    sys.stdout.flush()


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _quiet():
    """Model libraries print progress; keep stdout clean for the JSON result line."""
    return contextlib.redirect_stdout(sys.stderr)


def cmd_analyze(a: argparse.Namespace) -> Any:
    from m2s.analyze import analyze

    run = Path(a.run)
    (run / "acts").mkdir(parents=True, exist_ok=True)
    instruments = [x.strip() for x in a.instruments.split(",") if x.strip()] if a.instruments else None
    with _quiet():
        prof = analyze(run / "audio.wav", instruments, act_path=run / "acts" / "beats.npy")
    (run / "profile.json").write_text(json.dumps(prof, indent=1, default=_default))
    return {k: prof.get(k) for k in ("duration", "tuning_cents", "tempo_bpm")} | {
        "instruments": prof["instruments"]["present"]}


def cmd_separate(a: argparse.Namespace) -> Any:
    from m2s.analyze import solo
    from m2s.separate import separate_band

    run = Path(a.run)
    prof = json.loads((run / "profile.json").read_text())
    user = prof["instruments"].get("source") == "user"
    present = [p for p in prof["instruments"]["present"] if p != "synth"]
    one = (present[0] if len(present) == 1 else None) if user else solo(prof)
    if one:
        stems = {one: str(run / "audio.wav")}
        note = f"solo {one}, no separation needed"
    else:
        with _quiet():
            stems = separate_band(run / "audio.wav", run / "stems", present if user else None,
                                  log=lambda m: print(m, file=sys.stderr), tags=prof["instruments"])
        note = f"stems {sorted(stems)}"
    (run / "stems.json").write_text(json.dumps(stems, indent=1))
    return {"stems": stems, "note": note}


def cmd_transcribe(a: argparse.Namespace) -> Any:
    from m2s.ir import Piece

    run = Path(a.run)
    prof = json.loads((run / "profile.json").read_text())
    stems = json.loads((run / "stems.json").read_text())
    (run / "acts").mkdir(parents=True, exist_ok=True)
    tracks, summary = [], {}
    for preset, wav in stems.items():
        with _quiet():
            if preset == "piano":
                from m2s.transcribe.piano import transcribe_piano

                tr, kong = transcribe_piano(wav, prof.get("tuning_cents", 0.0))
                # onset activations at 100 fps: the verification front end for piano
                np.save(run / "acts" / "stem_piano.npy", kong["reg_onset_output"].astype(np.float32))
            else:
                from m2s.transcribe.band import transcribe_stem

                tr = transcribe_stem(wav, preset, prof, None)
        tracks.append(tr)
        summary[tr.name] = len(tr.notes)
    Piece(tracks=tracks, meta={"tuning_cents": prof.get("tuning_cents", 0.0)}).save(run / "candidates.json")
    return {"candidates": summary}


def cmd_activations(a: argparse.Namespace) -> Any:
    from m2s.verify import activations

    with _quiet():
        act, fps = activations(a.wav, a.preset)
    np.save(a.out, act.astype(np.float32))
    return {"fps": fps, "frames": int(act.shape[0])}


def cmd_engrave(a: argparse.Namespace) -> Any:
    from m2s.engrave import engrave

    with _quiet():
        used = engrave(a.musicxml, a.pdf, a.engine)
    return {"engine": used}


def cmd_inspect(a: argparse.Namespace) -> Any:
    from m2s.inspect import inspect_bars
    from m2s.runs import Run

    lo, hi = (int(x) for x in a.bars.split("-"))
    with _quiet():
        out = inspect_bars(Run(a.run), lo, hi, a.out)
    return {"image": out}


def cmd_align(a: argparse.Namespace) -> Any:
    from m2s.evaluate.align import align_reference
    from m2s.evaluate.reference import load_reference
    from m2s.ir import Piece

    run = Path(a.run)
    scored = run / "scored.json"
    ours = [n for t in Piece.load(scored if scored.exists() else run / "notes.json").tracks for n in t.kept()]
    with _quiet():
        ref = load_reference(a.ref, repeats=a.repeats, expected_notes=len(ours))
        ref, info = align_reference(ref, run / "audio.wav")
    out = {
        "notes": [{"on": n.on, "off": n.off, "pitch": n.pitch, "staff": n.staff, "qon": n.qon, "qdur": n.qdur}
                  for n in ref.notes],
        "reference": {"notes": len(ref.notes), "repeats_expanded": ref.expanded, "time_signature": ref.time_signature,
                      "key": ref.key, "bars": ref.bars},
        "alignment": info,
        "score_path": ref.score_path,
    }
    Path(a.out).write_text(json.dumps(out, default=_default))
    return {"reference_notes": len(ref.notes), "expanded": ref.expanded}


def cmd_mv2h(a: argparse.Namespace) -> Any:
    from m2s.evaluate.metrics import mv2h

    return mv2h(a.ours, a.ref)


def cmd_musicdiff(a: argparse.Namespace) -> Any:
    from m2s.evaluate.metrics import musicdiff_count

    return musicdiff_count(a.ours, a.ref, bars=a.bars)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m m2s.worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--run", required=True)
    p.add_argument("--instruments", default=None)
    for name in ("separate", "transcribe"):
        sub.add_parser(name).add_argument("--run", required=True)
    p = sub.add_parser("activations")
    p.add_argument("--wav", required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("engrave")
    p.add_argument("--musicxml", required=True)
    p.add_argument("--pdf", required=True)
    p.add_argument("--engine", default="auto")
    p = sub.add_parser("inspect")
    p.add_argument("--run", required=True)
    p.add_argument("--bars", required=True)
    p.add_argument("--out", default=None)
    p = sub.add_parser("align")
    p.add_argument("--run", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--repeats", default="auto")
    p.add_argument("--out", required=True)
    p = sub.add_parser("mv2h")
    p.add_argument("--ours", required=True)
    p.add_argument("--ref", required=True)
    p = sub.add_parser("musicdiff")
    p.add_argument("--ours", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--bars", type=int, default=16)
    a = ap.parse_args(argv)
    _emit(globals()[f"cmd_{a.cmd}"](a))


if __name__ == "__main__":
    main()
