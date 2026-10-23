"""Piano transcription: Transkun V2 (primary) + Kong et al. 2021 (second opinion, pedal), plus
hFT-Transformer (third opinion) when it is fetched (tools/fetch_hft.sh).

Output is a list of *candidates*: every note any model reported, with a
confidence from how the models agree. Nothing is thresholded here; the
playability decoder makes the keep/drop call.
"""

from __future__ import annotations

import bisect
import os
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


MODELS = ("transkun", "kong", "hft")
PATTERNS = ("TKH", "TK", "TH", "KH", "T", "K", "H")
FEATURES = PATTERNS + ("kong_onset", "hft_onset", "kong_onset_alone", "hft_onset_alone", "octave_contra",
                       "neighbour_contra", "velocity")


@dataclass
class VoteParams:
    """Three-model fusion: a candidate's confidence is sigmoid(FEATURES . weights): which models
    report it, their local onset evidence, and whether a better-supported note an octave or a
    semitone/tone away contradicts it. Weights: logistic regression on the real-recording dev set
    (data/realdev movements 08-1, 08-2, 08-3, 15-3, 31-2; label = matches the aligned reference
    within 100 ms). Leave-one-movement-out it scored the same as a refit two-model vote, so hFT is
    off by default (M2S_HFT=1 turns it on)."""

    match_tol: float = 0.05     # onset tolerance (s) for two models to agree on a note
    contra_tol: float = 0.05    # onset window for octave / neighbour contradictions
    weights: tuple[float, ...] = (1.0556, 1.1185, 0.5908, -1.5157, 0.7987, -1.8147, -3.169, 0.4282, 1.0626, 0.0206,
                                  0.665, -0.7564, -0.4915, 0.3895)


# The same vote with Transkun and Kong only, fit the same way with the hFT features held at zero
# (M2S_FUSION=vote). Leave-one-movement-out it beat the rule-based fusion (`fuse`) on all eight
# real-recording movements, including the three it was never fit on (mean F1 at 100 ms 0.851 vs
# 0.846): notes only Transkun reports are right far more often than that fusion's confidence says.
# Not the default yet: on the first rendered dev piece checked (Bach BWV 846) it kept 10 wrong
# notes and note F1 fell from 0.998 to 0.991; the full rendered set has not been run.
VOTE2 = VoteParams(weights=(0.0, 1.4958, 0.0, 0.0, 1.4607, -1.3771, 0.0, 0.8771, 0.0, -0.2588, 0.0, -1.0657,
                            -0.7158, 0.6189))


def _peak(act: np.ndarray | None, fps: float, t: float, pitch: int, half: int) -> float:
    if act is None or not (21 <= pitch <= 108):
        return 0.0
    f = int(round(t * fps))
    lo, hi = max(0, f - half), min(len(act), f + half + 1)
    return float(act[lo:hi, pitch - 21].max()) if lo < hi else 0.0


def vote_groups(t_notes: list[Note], k_notes: list[Note], h_notes: list[Note], kong_out: dict[str, np.ndarray],
                hft_onset: np.ndarray | None, hft_fps: float = 62.5,
                vp: VoteParams | None = None) -> tuple[list[dict[int, Note]], np.ndarray]:
    """Group the models' notes (same pitch, onsets within match_tol of the group's first note) and
    describe each group by the FEATURES. Groups come out sorted by onset, then pitch."""
    vp = vp or VoteParams()
    groups: list[dict[int, Note]] = []
    by_pitch: dict[int, list[int]] = {}
    for mi, notes in enumerate((t_notes, k_notes, h_notes)):
        for n in sorted(notes, key=lambda x: (x.onset, x.pitch)):
            best, arg = vp.match_tol, -1
            for gi in by_pitch.get(n.pitch, []):
                g = groups[gi]
                if mi in g:
                    continue
                d = abs(g[min(g)].onset - n.onset)
                if d <= best:
                    best, arg = d, gi
            if arg >= 0:
                groups[arg][mi] = n
            else:
                groups.append({mi: n})
                by_pitch.setdefault(n.pitch, []).append(len(groups) - 1)
    groups.sort(key=lambda g: (g[min(g)].onset, g[min(g)].pitch))

    kong_on = kong_out.get("reg_onset_output")
    reps = [g[min(g)] for g in groups]
    times = [r.onset for r in reps]
    votes = [len(g) for g in groups]
    x = np.zeros((len(groups), len(FEATURES)), dtype=np.float64)
    col = {f: i for i, f in enumerate(FEATURES)}
    for i, (g, r) in enumerate(zip(groups, reps)):
        x[i, col["".join("TKH"[m] for m in sorted(g))]] = 1.0
        ek = _peak(kong_on, KONG_FPS, r.onset, r.pitch, 5)
        eh = _peak(hft_onset, hft_fps, r.onset, r.pitch, 3)
        x[i, col["kong_onset"]], x[i, col["hft_onset"]] = ek, eh
        if votes[i] == 1:
            x[i, col["kong_onset_alone"]], x[i, col["hft_onset_alone"]] = ek, eh
        x[i, col["velocity"]] = r.velocity / 127
        # a better-supported note an octave, a semitone or a tone away at the same onset: this one
        # is probably that note's octave or neighbour error
        lo = bisect.bisect_left(times, r.onset - vp.contra_tol)
        hi = bisect.bisect_right(times, r.onset + vp.contra_tol)
        for j in range(lo, hi):
            if votes[j] > votes[i]:
                d = abs(reps[j].pitch - r.pitch)
                if d == 12:
                    x[i, col["octave_contra"]] = 1.0
                elif d in (1, 2):
                    x[i, col["neighbour_contra"]] = 1.0
    return groups, x


def fuse3(t_notes: list[Note], k_notes: list[Note], h_notes: list[Note], kong_out: dict[str, np.ndarray],
          hft_onset: np.ndarray, vp: VoteParams | None = None, hft_fps: float = 62.5) -> list[Note]:
    """One candidate per group of agreeing notes, with Transkun's timing when it has the note, else
    Kong's, else hFT's; confidence = sigmoid(features . weights)."""
    vp = vp or VoteParams()
    groups, x = vote_groups(t_notes, k_notes, h_notes, kong_out, hft_onset, hft_fps, vp)
    conf = 1 / (1 + np.exp(-(x @ np.asarray(vp.weights))))
    col = {f: i for i, f in enumerate(FEATURES)}
    out = []
    for g, c, row in zip(groups, conf, x):
        r = g[min(g)]
        tags = [t for t in ("octave_contra", "neighbour_contra") if row[col[t]]]
        out.append(Note(id="", onset=r.onset, offset=r.offset, pitch=r.pitch, velocity=r.velocity,
                        confidence=float(c), provenance=r.provenance, tags=tags,
                        support={**{MODELS[m]: 1.0 for m in g}, "kong_onset": round(row[col["kong_onset"]], 3),
                                 "hft_onset": round(row[col["hft_onset"]], 3)}))
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


def use_hft_default() -> bool:
    """hFT joins the fusion only with M2S_HFT=1 (and when it is fetched)."""
    from m2s.transcribe import hft

    return os.environ.get("M2S_HFT", "0") == "1" and hft.available()


def run_models(wav: str | Path, tuning_cents: float = 0.0, use_hft: bool | None = None) -> dict:
    """Every model's raw output for one piano stem, on audio retuned to A440 when it is off by > 20
    cents (times in the retuned recording; `ratio` maps them back)."""
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
    raw = {"ratio": ratio, "t_notes": t_notes, "t_pedal": t_pedal, "k_notes": k_notes, "k_pedal": k_pedal,
           "kong_out": kong_out}
    if use_hft is None:
        use_hft = use_hft_default()
    if use_hft:
        from m2s.transcribe.hft import FPS as HFT_FPS
        from m2s.transcribe.hft import run_hft

        raw["h_notes"], raw["h_onset"] = run_hft(mono16)
        raw["h_fps"] = HFT_FPS
    return raw


def candidates(raw: dict, wav: str | Path, name: str = "piano", fp: FusionParams | None = None,
               vp: VoteParams | None = None) -> Track:
    """Fuse the models' notes into scored candidates: the rule-based fusion (default), the
    two-model vote (M2S_FUSION=vote), or the three-model vote when hFT ran."""
    if "h_notes" in raw:
        notes = fuse3(raw["t_notes"], raw["k_notes"], raw["h_notes"], raw["kong_out"], raw["h_onset"], vp,
                      raw["h_fps"])
    elif os.environ.get("M2S_FUSION") == "vote":
        notes = fuse3(raw["t_notes"], raw["k_notes"], [], raw["kong_out"], None, vp or VOTE2)
    else:
        notes = fuse(raw["t_notes"], raw["k_notes"], raw["kong_out"], fp)
    pedal = merge_pedal(raw["t_pedal"], raw["k_pedal"])
    ratio = raw["ratio"]
    if ratio != 1.0:
        # the retuned recording plays `ratio` times faster: a time t in it is t * ratio in the original
        scale = ratio
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
    return Track(name=name, preset="piano", notes=notes, pedal=[(round(a, 4), round(b, 4)) for a, b in pedal],
                 stem=str(wav))


def transcribe_piano(wav: str | Path, tuning_cents: float = 0.0, name: str = "piano",
                     fp: FusionParams | None = None, use_hft: bool | None = None,
                     vp: VoteParams | None = None) -> tuple[Track, dict[str, np.ndarray]]:
    """Candidates for one piano stem. Retunes the audio first when it is off A440 by > 20 cents."""
    raw = run_models(wav, tuning_cents, use_hft)
    kong_out = raw["kong_out"]
    if raw["ratio"] != 1.0:
        # frame-wise outputs are on the retuned recording's clock; put them back on the original one
        kong_out = {k: _restretch(v, raw["ratio"]) if getattr(v, "ndim", 0) == 2 else v for k, v in kong_out.items()}
    return candidates(raw, wav, name, fp, vp), kong_out


def _restretch(a: np.ndarray, ratio: float) -> np.ndarray:
    """Resample a (frames, bins) array computed on audio played `ratio` times faster back to the
    original clock: original frame i is retuned frame i / ratio."""
    n = int(round(a.shape[0] * ratio))
    src = np.arange(n) / ratio
    idx = np.arange(a.shape[0])
    return np.stack([np.interp(src, idx, a[:, j]) for j in range(a.shape[1])], axis=1).astype(a.dtype)


def confidence_summary(track: Track) -> dict[str, int]:
    bins = {"both": 0, "transkun_only": 0, "kong_only": 0}
    if any("hft" in n.support or "hft_onset" in n.support for n in track.notes):
        bins = {"3 models": 0, "2 models": 0, "1 model": 0}
        for n in track.notes:
            v = sum(m in n.support for m in MODELS)
            bins[f"{v} model{'s' if v > 1 else ''}"] += 1
        return bins
    for n in track.notes:
        s = n.support
        if "transkun" in s and "kong" in s:
            bins["both"] += 1
        elif "transkun" in s:
            bins["transkun_only"] += 1
        else:
            bins["kong_only"] += 1
    return bins
