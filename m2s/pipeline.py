"""The full pipeline: ingest -> analyze -> separate -> transcribe -> decode -> verify -> notate -> engrave.

Each stage reads and writes files in the run folder and records a key in the
manifest, so reruns skip finished stages unless their inputs changed.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from m2s import __version__
from m2s import audio as A
from m2s.ir import Piece, Track
from m2s.runs import Run, file_hash

STAGES = ["ingest", "analyze", "separate", "transcribe", "decode", "verify", "notate", "engrave"]
Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[m2s {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------ stages
def ingest(src: str | Path, runs_root: str | Path = "runs", log: Log = _log) -> Run:
    """Decode to a hash-named WAV. The original file name is not stored anywhere in the run."""
    run = Run.for_audio(src, runs_root)
    key = {"src": file_hash(src), "v": __version__}
    if run.is_fresh("ingest", key, ["audio.wav"]):
        return run
    A.decode(src, run.path("audio.wav"))
    run.mark("ingest", key, input_hash=key["src"])
    log(f"ingest -> {run.root}")
    return run


def analyze(run: Run, instruments: list[str] | None = None, force: bool = False, log: Log = _log) -> dict[str, Any]:
    from m2s.analyze import analyze as do_analyze

    key = {"in": run.stage_key("ingest"), "instruments": instruments, "v": __version__}
    if not force and run.is_fresh("analyze", key, ["profile.json"]):
        return run.read_json("profile.json")
    log("analyze: tuning, instruments, beats")
    prof = do_analyze(run.root / "audio.wav", instruments, act_path=run.path("acts", "beats.npy"))
    run.write_json("profile.json", prof)
    run.mark("analyze", key)
    log(f"analyze: {prof['duration']:.0f}s, tuning {prof['tuning_cents']:+.0f} cents, "
        f"instruments {prof['instruments']['present']}, ~{prof.get('tempo_bpm', 0):.0f} bpm")
    return prof


def separate(run: Run, prof: dict[str, Any], force: bool = False, log: Log = _log) -> dict[str, str]:
    """Returns {preset: stem wav path}. A solo recording is its own stem."""
    from m2s.analyze import solo

    user = prof["instruments"].get("source") == "user"
    present = [p for p in prof["instruments"]["present"] if p != "synth"]
    one = (present[0] if len(present) == 1 else None) if user else solo(prof)
    key = {"in": run.stage_key("analyze"), "present": present, "solo": one, "v": __version__}
    if not force and run.is_fresh("separate", key, ["stems.json"]):
        return run.read_json("stems.json")
    if one:
        stems = {one: str(run.root / "audio.wav")}
        log(f"separate: solo {one}, no separation needed")
    else:
        from m2s.separate import separate_band

        log(f"separate: {'band with ' + str(present) if user else 'not a clear solo; detecting from stems'}")
        stems = separate_band(run.root / "audio.wav", run.root / "stems", present if user else None, log=log,
                              tags=prof["instruments"])
    run.write_json("stems.json", stems)
    run.mark("separate", key)
    return stems


def transcribe(run: Run, prof: dict[str, Any], stems: dict[str, str], force: bool = False,
               log: Log = _log) -> Piece:
    key = {"in": run.stage_key("separate"), "stems": stems, "v": __version__}
    if not force and run.is_fresh("transcribe", key, ["candidates.json"]):
        return Piece.load(run.root / "candidates.json")
    tracks: list[Track] = []
    for preset, wav in stems.items():
        log(f"transcribe: {preset}")
        if preset == "piano":
            from m2s.transcribe.piano import confidence_summary, transcribe_piano

            tr, kong = transcribe_piano(wav, prof.get("tuning_cents", 0.0))
            np.save(run.path("acts", "piano.npy"), kong["reg_onset_output"].astype(np.float16))
            log(f"transcribe: piano {len(tr.notes)} candidates {confidence_summary(tr)}, {len(tr.pedal)} pedal")
        else:
            from m2s.transcribe.band import transcribe_stem

            tr = transcribe_stem(wav, preset, prof, run.path("acts", f"{preset}.npy"))
            log(f"transcribe: {preset} {len(tr.notes)} candidates")
        tracks.append(tr)
    piece = Piece(tracks=tracks, meta={"tuning_cents": prof.get("tuning_cents", 0.0)})
    piece.save(run.root / "candidates.json")
    run.mark("transcribe", key)
    return piece


def decode(run: Run, piece: Piece, force: bool = False, log: Log = _log) -> tuple[Piece, list[dict[str, Any]]]:
    from m2s.play import decode_track

    key = {"in": run.stage_key("transcribe"), "v": __version__}
    if not force and run.is_fresh("decode", key, ["notes.json", "violations.json"]):
        return Piece.load(run.root / "notes.json"), run.read_json("violations.json")
    viols: list[dict[str, Any]] = []
    for tr in piece.tracks:
        v = decode_track(tr)
        viols += [dict(x, track=tr.name) for x in v]
        log(f"decode: {tr.name} kept {len(tr.kept())}/{len(tr.notes)}, {len(v)} violations")
    piece.save(run.root / "notes.json")
    run.write_json("violations.json", viols)
    run.mark("decode", key)
    return piece, viols


def verify(run: Run, piece: Piece, stems: dict[str, str], force: bool = False, log: Log = _log) -> dict[str, Any]:
    from m2s.verify import heatmap, to_cells, verify_track

    key = {"in": run.stage_key("decode"), "edits": run.manifest.get("edits", 0), "v": __version__}
    if not force and run.is_fresh("verify", key, ["verify.json"]):
        return run.read_json("verify.json")
    reports = []
    for tr in piece.tracks:
        acts = run.root / "acts" / f"{tr.preset}.npy"
        stem_act = (np.load(acts).astype(np.float32), 100.0) if (acts.exists() and tr.preset == "piano") else None
        rep, m, sc = verify_track(tr, stems.get(tr.preset, run.root / "audio.wav"), run.root, stem_act)
        np.save(run.path("acts", f"{tr.name}_cells.npy"), sc.astype(np.float16))
        heatmap(m, sc, run.path(f"heatmap_{tr.name}.png"), title=tr.name)
        reports.append(rep)
        log(f"verify: {tr.name} audio match {rep['audio_match']:.3f} "
            f"(precision {rep['precision']:.3f}, recall {rep['recall']:.3f})")
    out = {"tracks": reports}
    run.write_json("verify.json", out)
    run.mark("verify", key)
    return out


def notate(run: Run, piece: Piece, prof: dict[str, Any], fingering: bool = False, force: bool = False,
           log: Log = _log) -> dict[str, Any]:
    from m2s.notate import Options, lint, write_musicxml
    from m2s.notate import notate as do_notate

    key = {"in": run.stage_key("decode"), "edits": run.manifest.get("edits", 0), "fingering": fingering,
           "v": __version__}
    if not force and run.is_fresh("notate", key, ["score.musicxml", "notation.json"]):
        return run.read_json("notation.json")
    acts_path = run.root / "acts" / "beats.npy"
    acts = np.load(acts_path).astype(np.float32) if acts_path.exists() else None
    score, grid, info = do_notate(piece, prof["beats"], prof["downbeats"],
                                  Options(fingering=fingering, title=f"Transcription {run.root.name}"),
                                  beat_acts=acts, act_fps=prof.get("beat_act_fps", 50.0),
                                  duration=prof["duration"])
    write_musicxml(score, run.root / "score.musicxml")
    info["lint"] = lint(score)
    info["bars"] = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    # notes now carry bar/beat; keep them for inspect and eval
    piece.save(run.root / "notes.json")
    run.write_json("notation.json", info)
    run.mark("notate", key)
    log(f"notate: {info['bars']} bars, {info['time_signature']}, {info['key']}, q={info['qpm']:.0f}, "
        f"lint {info['lint']['by_kind']}")
    return info


def engrave(run: Run, force: bool = False, engine: str = "auto", log: Log = _log) -> str:
    from m2s.engrave import engrave as do_engrave

    key = {"in": run.stage_key("notate"), "engine": engine, "v": __version__}
    if not force and run.is_fresh("engrave", key, ["score.pdf"]):
        return run.manifest["stages"]["engrave"].get("engine", "?")
    used = do_engrave(run.root / "score.musicxml", run.root / "score.pdf", engine)
    run.mark("engrave", key, engine=used)
    log(f"engrave: score.pdf ({used})")
    return used


def summary(run: Run) -> dict[str, Any]:
    out: dict[str, Any] = {"run": run.root.name, "version": __version__}
    for name in ("profile.json", "verify.json", "notation.json", "violations.json"):
        p = run.root / name
        if p.exists():
            d = json.loads(p.read_text())
            if name == "profile.json":
                out["profile"] = {k: d[k] for k in ("duration", "tuning_cents", "tempo_bpm") if k in d}
                out["profile"]["instruments"] = d["instruments"]["present"]
            elif name == "verify.json":
                out["audio_match"] = {t["track"]: t["audio_match"] for t in d["tracks"]}
            elif name == "notation.json":
                out["notation"] = {k: d[k] for k in ("time_signature", "key", "qpm", "bars", "ornaments") if k in d}
                out["notation"]["lint"] = d["lint"]["by_kind"]
            else:
                kinds: dict[str, int] = {}
                for v in d:
                    kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
                out["violations"] = kinds
    run.write_json("summary.json", out)
    return out


def run_all(src: str | Path, runs_root: str | Path = "runs", instruments: list[str] | None = None,
            fingering: bool = False, engine: str = "auto", until: str | None = None, force: bool = False,
            log: Log = _log) -> Run:
    t0 = time.time()
    run = ingest(src, runs_root, log)
    prof = analyze(run, instruments, force, log)
    stems = separate(run, prof, force, log)
    if until == "separate":
        return run
    piece = transcribe(run, prof, stems, force, log)
    if until == "transcribe":
        return run
    piece, _ = decode(run, piece, force, log)
    verify(run, piece, stems, force, log)
    if until == "verify":
        return run
    notate(run, piece, prof, fingering, force, log)
    engrave(run, force, engine, log)
    s = summary(run)
    log(f"done in {time.time() - t0:.0f}s -> {run.root}/score.pdf  audio match {s.get('audio_match')}")
    return run


def copy_outputs(run: Run, dest: str | Path) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("score.pdf", "score.musicxml", "summary.json", "verify.json", "violations.json", "notation.json"):
        if (run.root / name).exists():
            shutil.copy(run.root / name, dest / name)
