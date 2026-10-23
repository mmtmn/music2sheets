"""Score a band-test run against the synthetic ground truth (tools/make_band_test.py).

    python tools/eval_band_test.py data/bandruns/<hash> [data/bandtest]
"""

from __future__ import annotations

import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

from m2s.ir import Note, Piece

# ground-truth part -> our track names that should contain it
PARTS = {"guitar_L": ["electric_guitar_L"], "guitar_R": ["electric_guitar_R"], "bass": ["bass"],
         "drums": ["drums"], "sax_alto": ["sax_alto"]}


def f1(ref: list[pretty_midi.Note], est: list[Note], tol: float = 0.05) -> tuple[float, float, float]:
    if not ref or not est:
        return 0.0, 0.0, 0.0
    ri = np.array([[n.start, n.end] for n in ref])
    rp = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in ref])
    ei = np.array([[n.onset, max(n.offset, n.onset + 0.01)] for n in est])
    ep = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in est])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=tol,
                                                                    offset_ratio=None)
    return round(p, 3), round(r, 3), round(f, 3)


def main(run: str, gt_dir: str = "data/bandtest") -> None:
    piece = Piece.load(Path(run) / "notes.json")
    names = {t.name for t in piece.tracks}
    for part, ours in PARTS.items():
        ref = pretty_midi.PrettyMIDI(str(Path(gt_dir) / f"gt_{part}.mid")).instruments[0].notes
        est = [n for t in piece.tracks if t.name in ours for n in t.kept()]
        if part == "drums":
            for cls, gm in (("kick", 36), ("snare", 38), ("hihat", 42)):
                r = [n for n in ref if n.pitch == gm]
                e = [n for n in est if n.pitch == gm]
                print(f"drums/{cls:6s} P/R/F {f1(r, e)}  ({len(e)} est, {len(r)} ref)")
            continue
        missing = [o for o in ours if o not in names]
        print(f"{part:9s} P/R/F {f1(ref, est)}  ({len(est)} est, {len(ref)} ref){'  missing ' + str(missing) if missing else ''}")
    # the two guitars together, ignoring which side each note was given
    ref = [n for p in ("guitar_L", "guitar_R")
           for n in pretty_midi.PrettyMIDI(str(Path(gt_dir) / f"gt_{p}.mid")).instruments[0].notes]
    est = [n for t in piece.tracks if t.name.startswith("electric_guitar") for n in t.kept()]
    print(f"{'guitars':9s} P/R/F {f1(ref, est)}  (both sides pooled)")


if __name__ == "__main__":
    main(*sys.argv[1:])
