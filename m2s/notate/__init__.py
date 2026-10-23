"""Stage 6: notation. Grid and quantization, key and spelling, then the score."""

from __future__ import annotations

from typing import Any

from m2s.ir import Piece
from m2s.notate.build import Options, build_score, lint
from m2s.notate.keys import Key, estimate_key
from m2s.notate.ornaments import fold_trills, mark_graces
from m2s.notate.rhythm import Grid, build_grid, quantize
from m2s.notate.spell import spell
from m2s.play.preset import load_preset


def notate(piece: Piece, beats: list[float], downbeats: list[float], opts: Options | None = None,
           beat_acts: Any = None, act_fps: float = 50.0, duration: float | None = None
           ) -> tuple[Any, Grid, dict[str, Any]]:
    ornaments = {"trills": 0, "graces": 0}
    for t in piece.tracks:
        if load_preset(t.preset)["family"] in ("keyboard", "fretted", "wind"):
            ornaments["trills"] += fold_trills(t.notes)
            ornaments["graces"] += mark_graces(t.notes)
    kept = [n for t in piece.tracks for n in t.kept()]
    # rhythm is decided from pitched parts when there are any; drums only if alone
    pitched = [n for t in piece.tracks if load_preset(t.preset)["family"] != "drums" for n in t.kept()] or kept
    db_strength = None
    if beat_acts is not None and pitched:
        # On the dev set, the tracker's own beats (cleaned of doubles and gaps) beat every
        # dynamic-programming refinement we tried; only its downbeat activation is added.
        from m2s.notate.beats import downbeat_strength
        from m2s.notate.rhythm import clean_beats

        cleaned = clean_beats(beats)
        if len(cleaned) >= 4:
            beats = [float(b) for b in cleaned]
            db_strength = downbeat_strength(cleaned, beat_acts[:, 1], act_fps)
    grid = build_grid(beats, downbeats, pitched, db_strength)
    quantize(kept, grid)
    keys: dict[str, Key] = {}
    glob = estimate_key(pitched)
    for t in piece.tracks:
        preset = load_preset(t.preset)
        if preset["family"] == "drums":
            continue
        trans = int(preset.get("transposition", 0))
        notes = t.kept()
        k = glob
        if trans:
            # spell the written (transposed) pitches against the written key
            written = Key((glob.tonic_pc + trans) % 12, glob.mode, glob.score)
            for n in notes:
                n.pitch += trans
            spell(notes, written)
            for n in notes:
                n.pitch -= trans
            k = glob
        else:
            spell(notes, k)
        keys[t.name] = k
    score = build_score(piece, grid, keys, opts)
    info = {
        "time_signature": grid.time_signature(),
        "pickup_quarters": grid.pickup_quarters,
        "qpm": round(grid.qpm(), 1),
        "beat_unit_quarters": grid.unit,
        "key": glob.name(),
        "key_score": round(glob.score, 3),
        "ornaments": ornaments,
        "grid": grid.info,
    }
    return score, grid, info


__all__ = ["notate", "lint", "Options"]
