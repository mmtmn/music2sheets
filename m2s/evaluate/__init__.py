"""Blind evaluation against a reference score.

Run this only after the transcription is committed: it reads the reference,
aligns it to the recording (audio vs. score, independent of our notes), and
scores the transcription at note level (mir_eval), score level (MV2H,
musicdiff) and for playability (violations per 100 notes, ours vs. the
reference's own).
"""

from __future__ import annotations

import json
from typing import Any

from m2s.evaluate.reference import load_reference
from m2s.ir import Piece
from m2s.runs import Run


def evaluate_run(run: Run, ref_path: str, repeats: str = "auto", diff_bars: int = 0,
                 use_mv2h: bool = True) -> dict[str, Any]:
    """Score a run against a reference. ``diff_bars`` > 0 also runs musicdiff on the first that many
    bars (musicdiff's tree edit distance is too slow for whole movements)."""
    from m2s.evaluate.align import align_reference
    from m2s.evaluate.metrics import (hand_agreement, mv2h, musicdiff_count, note_scores, playability,
                                      score_facts)

    piece = Piece.load(run.root / "notes.json")
    ours = [n for t in piece.tracks for n in t.kept()]
    ref = load_reference(ref_path, repeats=repeats, expected_notes=len(ours))
    aligned, align_info = align_reference(ref, run.root / "audio.wav")
    rep: dict[str, Any] = {"reference": {"notes": len(ref.notes), "repeats_expanded": ref.expanded,
                                         "time_signature": ref.time_signature, "key": ref.key,
                                         "bars": ref.bars},
                           "alignment": align_info}
    rep["notes"] = {f"onset_{int(tol * 1000)}ms": note_scores(aligned, ours, tol) for tol in (0.05, 0.1)}
    rep["hands"] = hand_agreement(aligned, ours)
    rep["playability"] = playability(run, aligned, ours)
    rep["score"] = score_facts(run, ref)
    rep["mv2h"] = mv2h(run.root / "score.musicxml", ref.score_path) if use_mv2h else {"skipped": "disabled"}
    rep["musicdiff"] = (musicdiff_count(run.root / "score.musicxml", ref.score_path, bars=diff_bars)
                        if diff_bars else {"skipped": "pass --diff-bars N to run musicdiff on the first N bars"})
    rep["summary"] = {
        "note_f1_50ms": rep["notes"]["onset_50ms"]["f1"],
        "note_f1_100ms": rep["notes"]["onset_100ms"]["f1"],
        "hand_agreement": rep["hands"].get("agreement"),
        "violations_per_100_notes": rep["playability"],
        "mv2h": rep["mv2h"].get("MV2H") if isinstance(rep["mv2h"], dict) else None,
        "musicdiff_edits": rep["musicdiff"].get("edits") if isinstance(rep["musicdiff"], dict) else None,
        "time_signature": [rep["score"]["ours"]["time_signature"], ref.time_signature],
        "key": [rep["score"]["ours"]["key"], ref.key],
    }
    (run.root / "eval.json").write_text(json.dumps(rep, indent=1))
    return rep
