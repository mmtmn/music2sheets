"""Piano transcription: Transkun V2 (primary) + Kong et al. 2021 (second opinion, pedal).

Output is a list of *candidates*: every note either model reported, with a
confidence from how the two agree. Nothing is thresholded here; the
playability decoder makes the keep/drop call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from m2s import audio as A
from m2s.ir import Note, Track, assign_ids

KONG_FPS = 100
KONG_SR = 16000


@dataclass
class FusionParams:
    match_tol: float = 0.05     # onset tolerance (s) for two models to agree on a note
    both: float = 0.97          # confidence when both models report the note
    t_only_base: float = 0.42   # Transkun only: base + gain * Kong's local onset evidence
    t_only_gain: float = 0.55
    k_only_gain: float = 0.55   # Kong only: gain * its own onset peak


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_transkun(y: np.ndarray, sr: int) -> tuple[list[Note], list[tuple[float, float]]]:
    import importlib.resources as ir

    import moduleconf
    import torch

    dev = _device()
    base = ir.files("transkun") / "pretrained"
    conf_manager = moduleconf.parseFromFile(str(base / "2.0.conf"))
    model_cls = conf_manager["Model"].module.TransKun
    model = model_cls(conf=conf_manager["Model"].config).to(dev)
    ckpt = torch.load(str(base / "2.0.pt"), map_location=dev)
    model.load_state_dict(ckpt.get("best_state_dict", ckpt.get("state_dict")), strict=False)
    model.eval()
    if sr != model.fs:
        import soxr

        y = soxr.resample(y, sr, model.fs).astype(np.float32)
    with torch.no_grad():
        events = model.transcribe(torch.from_numpy(y).to(dev), discardSecondHalf=False)
    notes, pedal = [], []
    for e in events:
        if e.pitch < 0:
            if e.pitch == -64 and e.end is not None:
                pedal.append((float(e.start), float(e.end)))
            continue
        notes.append(Note(id="", onset=float(e.start), offset=float(e.end), pitch=int(e.pitch),
                          velocity=int(np.clip(round(e.velocity), 1, 127)), provenance="transkun"))
    del model
    torch.cuda.empty_cache()
    return notes, pedal


def run_kong(y_mono_16k: np.ndarray) -> tuple[list[Note], list[tuple[float, float]], dict[str, np.ndarray]]:
    import contextlib
    import io

    import torch
    from piano_transcription_inference import PianoTranscription

    with contextlib.redirect_stdout(io.StringIO()):
        pt = PianoTranscription(device=_device())
        out = pt.transcribe(y_mono_16k.astype(np.float32), None)
    notes = [Note(id="", onset=float(e["onset_time"]), offset=float(e["offset_time"]), pitch=int(e["midi_note"]),
                  velocity=int(e["velocity"]), provenance="kong") for e in out["est_note_events"]]
    pedal = [(float(e["onset_time"]), float(e["offset_time"])) for e in out["est_pedal_events"]]
    od = {k: np.asarray(v) for k, v in out["output_dict"].items()}
    del pt
    torch.cuda.empty_cache()
    return notes, pedal, od


def _local_peak(act: np.ndarray, t: float, pitch: int, half: int = 5) -> float:
    if act is None or not (21 <= pitch <= 108):
        return 0.0
    f = int(round(t * KONG_FPS))
    lo, hi = max(0, f - half), min(len(act), f + half + 1)
    if lo >= hi:
        return 0.0
    return float(act[lo:hi, pitch - 21].max())


def fuse(t_notes: list[Note], k_notes: list[Note], kong_out: dict[str, np.ndarray],
         fp: FusionParams | None = None) -> list[Note]:
    fp = fp or FusionParams()
    onset_act = kong_out.get("reg_onset_output")
    frame_act = kong_out.get("frame_output")
    by_pitch: dict[int, list[int]] = {}
    for j, k in enumerate(k_notes):
        by_pitch.setdefault(k.pitch, []).append(j)
    used: set[int] = set()
    out: list[Note] = []
    for t in t_notes:
        best, arg = fp.match_tol, -1
        for j in by_pitch.get(t.pitch, []):
            d = abs(k_notes[j].onset - t.onset)
            if j not in used and d <= best:
                best, arg = d, j
        on_ev = _local_peak(onset_act, t.onset, t.pitch)
        if arg >= 0:
            used.add(arg)
            t.confidence = fp.both
            t.support = {"transkun": 1.0, "kong": 1.0, "kong_onset": round(on_ev, 3)}
        else:
            fr_ev = _local_peak(frame_act, (t.onset + t.offset) / 2, t.pitch, 2)
            ev = max(on_ev, 0.6 * fr_ev)
            t.confidence = float(min(0.95, fp.t_only_base + fp.t_only_gain * ev))
            t.support = {"transkun": 1.0, "kong_onset": round(on_ev, 3), "kong_frame": round(fr_ev, 3)}
        out.append(t)
    for j, k in enumerate(k_notes):
        if j in used:
            continue
        on_ev = _local_peak(onset_act, k.onset, k.pitch)
        k.confidence = float(fp.k_only_gain * on_ev)
        k.support = {"kong": 1.0, "kong_onset": round(on_ev, 3)}
        out.append(k)
    return out


def merge_pedal(*sets: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ivs = sorted((a, b) for s in sets for a, b in s if b > a)
    merged: list[tuple[float, float]] = []
    for a, b in ivs:
        if merged and a <= merged[-1][1] + 0.02:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def transcribe_piano(wav: str | Path, tuning_cents: float = 0.0, name: str = "piano",
                     fp: FusionParams | None = None) -> tuple[Track, dict[str, np.ndarray]]:
    """Candidates for one piano stem. Retunes the audio first when it is off A440 by > 20 cents."""
    y, sr = A.load(wav)
    ratio = 1.0
    if abs(tuning_cents) > 20:
        # play the recording back slightly faster/slower so it sits at A440, then map times back
        ratio = 2 ** (-tuning_cents / 1200)
        import soxr

        y = soxr.resample(y, int(round(sr * ratio)), sr).astype(np.float32)
    t_notes, t_pedal = run_transkun(y, sr)
    import soxr

    mono16 = soxr.resample(y.mean(axis=1), sr, KONG_SR).astype(np.float32)
    k_notes, k_pedal, kong_out = run_kong(mono16)
    notes = fuse(t_notes, k_notes, kong_out, fp)
    pedal = merge_pedal(t_pedal, k_pedal)
    if ratio != 1.0:
        scale = 1 / ratio
        for n in notes:
            n.onset *= scale
            n.offset *= scale
        pedal = [(a * scale, b * scale) for a, b in pedal]
    notes = [n for n in notes if n.offset > n.onset]
    for n in notes:
        n.confidence = round(n.confidence, 4)
        n.onset = round(n.onset, 4)
        n.offset = round(n.offset, 4)
    assign_ids(notes, "pno")
    track = Track(name=name, preset="piano", notes=notes, pedal=[(round(a, 4), round(b, 4)) for a, b in pedal],
                  stem=str(wav))
    return track, kong_out


def confidence_summary(track: Track) -> dict[str, int]:
    bins = {"both": 0, "transkun_only": 0, "kong_only": 0}
    for n in track.notes:
        s = n.support
        if "transkun" in s and "kong" in s:
            bins["both"] += 1
        elif "transkun" in s:
            bins["transkun_only"] += 1
        else:
            bins["kong_only"] += 1
    return bins
