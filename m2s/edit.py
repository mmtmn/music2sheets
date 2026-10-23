"""The edit protocol: Claude (or anyone) changes notes only through JSON patches.

A patch is applied to a copy of the notes, the affected window is re-rendered
and compared with the stem, and the patch is accepted only if the audio match
does not drop by more than EPS and playability does not get worse, or if the
audio match clearly improves. Every patch, accepted or not, is logged.

Patch format::

    {"reason": "bar 23: missed inner note, visible in heatmap",
     "ops": [
       {"op": "add", "track": "piano", "onset": 12.34, "offset": 12.52, "pitch": 61, "velocity": 70, "hand": "R"},
       {"op": "delete", "id": "pno-000123"},
       {"op": "retime", "id": "pno-000124", "onset": 12.40, "offset": 12.6},
       {"op": "repitch", "id": "pno-000125", "pitch": 63},
       {"op": "reassign", "id": "pno-000126", "hand": "L"},
       {"op": "retag", "id": "pno-000127", "tags": ["trill"]},
       {"op": "spell", "id": "pno-000128", "spelling": "B#3"}
     ]}
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any

import numpy as np

from m2s.ir import Note, Piece
from m2s.runs import Run

EPS = 0.005     # tolerated audio-match loss
DELTA = 0.01    # audio-match gain that accepts an edit on its own
NOTATION_ONLY = {"spell", "retag", "reassign"}


def _apply(piece: Piece, ops: list[dict[str, Any]]) -> tuple[list[str], tuple[float, float]]:
    idx = piece.note_index()
    touched: list[str] = []
    t0, t1 = float("inf"), float("-inf")
    for k, op in enumerate(ops):
        kind = op["op"]
        if kind == "add":
            tr = piece.track(op.get("track", piece.tracks[0].name))
            nid = op.get("id") or f"{tr.name[:3]}-add-{int(time.time() * 1000) % 10**9}-{k}"
            eff = {kk: op[kk] for kk in ("hand", "finger", "string", "fret", "limb") if kk in op}
            n = Note(nid, float(op["onset"]), float(op["offset"]), int(op["pitch"]), int(op.get("velocity", 70)),
                     confidence=1.0, effector=eff, provenance="edit")
            tr.notes.append(n)
            touched.append(nid)
            t0, t1 = min(t0, n.onset), max(t1, n.offset)
            continue
        if op["id"] not in idx:
            raise KeyError(f"no note {op['id']}")
        tr, n = idx[op["id"]]
        t0, t1 = min(t0, n.onset), max(t1, n.offset)
        if kind == "delete":
            n.keep = False
            n.tags.append("deleted-by-edit")
        elif kind == "retime":
            n.onset, n.offset = float(op.get("onset", n.onset)), float(op.get("offset", n.offset))
        elif kind == "repitch":
            n.pitch = int(op["pitch"])
            n.spelling = None
        elif kind == "reassign":
            n.effector.update({kk: op[kk] for kk in ("hand", "finger", "string", "fret", "limb") if kk in op})
            if "player" in op:
                n.player = int(op["player"])
        elif kind == "retag":
            n.tags = list(op["tags"])
        elif kind == "spell":
            n.spelling = op["spelling"]
        else:
            raise ValueError(f"unknown op {kind}")
        t0, t1 = min(t0, n.onset), max(t1, n.offset)
        touched.append(n.id)
    return touched, (t0, t1)


def _violations(piece: Piece, track_names: set[str]) -> int:
    from m2s.play.piano import PianoDecoder
    from m2s.play.preset import load_preset

    total = 0
    for tr in piece.tracks:
        if tr.name in track_names and load_preset(tr.preset)["family"] == "keyboard":
            total += len(PianoDecoder(tr.preset)._hand_violations(tr))
    return total


def apply_patch(run: Run, patch: dict[str, Any]) -> dict[str, Any]:
    from m2s.verify import local_match

    before = Piece.load(run.root / "notes.json")
    after = copy.deepcopy(before)
    touched, (t0, t1) = _apply(after, patch["ops"])
    tracks = {after.note_index()[i][0].name for i in touched if i in after.note_index()}
    notation_only = all(op["op"] in NOTATION_ONLY for op in patch["ops"])
    res: dict[str, Any] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": patch.get("reason", ""),
                           "ops": patch["ops"], "window": [round(t0, 3), round(t1, 3)]}
    v_before, v_after = _violations(before, tracks), _violations(after, tracks)
    res["violations"] = [v_before, v_after]
    if notation_only:
        accepted = v_after <= v_before
        res["audio_match"] = None
    else:
        m_before, m_after = [], []
        for name in tracks:
            cells_p = run.root / "acts" / f"{name}_cells.npy"
            if not cells_p.exists():
                raise FileNotFoundError("run `m2s verify` first; it caches the stem's activation cells")
            cells = np.load(cells_p).astype(np.float32)
            m_before.append(local_match(before.track(name), cells, t0, t1, run.root))
            m_after.append(local_match(after.track(name), cells, t0, t1, run.root))
        mb, ma = float(np.mean(m_before)), float(np.mean(m_after))
        res["audio_match"] = [round(mb, 4), round(ma, 4)]
        accepted = (ma - mb >= -EPS and v_after <= v_before) or ma - mb > DELTA
    res["accepted"] = bool(accepted)
    if accepted:
        after.save(run.root / "notes.json")
        run.manifest["edits"] = run.manifest.get("edits", 0) + 1
        run.save_manifest()
    with open(run.root / "edits.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    return res
