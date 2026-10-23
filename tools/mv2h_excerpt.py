"""MV2H on matching excerpts of a run and its reference (MV2H's alignment is slow on whole movements).

Takes the bars of each score that cover the first N seconds of the recording: ours from the
notes' own times, the reference's from the score-to-audio alignment.

    python tools/mv2h_excerpt.py runs/<hash> reference.krn --seconds 120
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import music21

from m2s.evaluate.align import align_reference
from m2s.evaluate.metrics import mv2h
from m2s.evaluate.reference import load_reference
from m2s.ir import Piece
from m2s.runs import Run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ref")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--our-bars", type=int, default=None, help="override: last bar of our score to keep")
    a = ap.parse_args()
    run = Run(a.run)
    piece = Piece.load(run.root / "scored.json")
    ours = [n for _, n in piece.all_notes() if n.keep]
    if a.our_bars is None:
        our_last = max(n.measure for n in ours if n.measure is not None and n.onset <= a.seconds)
    else:
        our_last = a.our_bars

    ref = load_reference(a.ref, expected_notes=len(ours))
    ref, _ = align_reference(ref, run.root / "audio.wav")
    ref_score = music21.converter.parse(ref.score_path)
    # reference bar of the last note that the alignment puts before the cut
    bar_q = ref_score.parts[0].getElementsByClass("Measure").first().barDuration.quarterLength
    last_q = max(n.qon for n in ref.notes if n.on <= a.seconds)
    ref_last = int(last_q // bar_q) + 1

    tmp = Path(tempfile.mkdtemp(prefix="m2s-mv2h-"))
    ours_x = tmp / "ours.musicxml"
    ref_x = tmp / "ref.musicxml"
    music21.converter.parse(run.root / "score.musicxml").measures(0, our_last).write("musicxml", fp=str(ours_x))
    ref_score.measures(0, ref_last).write("musicxml", fp=str(ref_x))
    res = mv2h(ours_x, ref_x)
    out = {"seconds": a.seconds, "our_bars": our_last, "ref_bars": ref_last, "mv2h": res}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
