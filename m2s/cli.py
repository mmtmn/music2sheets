"""`m2s` command line. Every command works on a run folder (runs/<hash>) or an audio file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from m2s.runs import Run

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="music2sheets: audio to playable sheet music, verified against the recording.")


def _run(target: str, runs_root: str = "runs") -> Run:
    p = Path(target)
    if p.is_dir():
        return Run(p)
    from m2s.pipeline import ingest

    return ingest(p, runs_root)


def _instruments(value: Optional[str]) -> Optional[list[str]]:
    return [x.strip() for x in value.split(",") if x.strip()] if value else None


@app.command()
def run(audio: str, instruments: Optional[str] = typer.Option(None, help="Comma list of presets, e.g. piano or "
                                                                     "electric_guitar,bass,drums,voice. "
                                                                     "Default: detect."),
        runs_root: str = "runs", fingering: bool = False, engine: str = "auto",
        until: Optional[str] = typer.Option(None, help="Stop after this stage: separate, transcribe or verify."),
        force: bool = False, out: Optional[str] = typer.Option(None, help="Also copy score.pdf etc. here.")) -> None:
    """Run the whole pipeline on an audio file."""
    from m2s.pipeline import copy_outputs, run_all

    r = run_all(audio, runs_root, _instruments(instruments), fingering, engine, until, force)
    if out:
        copy_outputs(r, out)


@app.command()
def ingest(audio: str, runs_root: str = "runs") -> None:
    """Decode and anonymize an audio file; prints the run folder."""
    typer.echo(_run(audio, runs_root).root)


@app.command()
def analyze(target: str, instruments: Optional[str] = None, force: bool = False) -> None:
    """Tuning, instruments, beats -> profile.json."""
    from m2s.pipeline import analyze as stage

    prof = stage(_run(target), _instruments(instruments), force)
    typer.echo(json.dumps({k: v for k, v in prof.items() if k not in ("beats", "downbeats")}, indent=1)[:3000])


@app.command()
def separate(target: str, force: bool = False) -> None:
    """Split into stems (skipped for solo recordings) -> stems.json."""
    from m2s.pipeline import analyze as a, separate as stage

    r = _run(target)
    typer.echo(json.dumps(stage(r, a(r), force), indent=1))


@app.command()
def transcribe(target: str, force: bool = False) -> None:
    """Note candidates per stem -> candidates.json."""
    from m2s.pipeline import analyze as a, separate as s, transcribe as stage

    r = _run(target)
    prof = a(r)
    stage(r, prof, s(r, prof), force)


@app.command()
def decode(target: str, force: bool = False) -> None:
    """Playability decode -> notes.json, violations.json."""
    from m2s.ir import Piece
    from m2s.pipeline import decode as stage

    r = _run(target)
    stage(r, Piece.load(r.root / "candidates.json"), force)


@app.command()
def verify(target: str, force: bool = False) -> None:
    """Resynthesize and compare with the stems -> verify.json, heatmap_*.png."""
    from m2s.ir import Piece
    from m2s.pipeline import verify as stage

    r = _run(target)
    rep = stage(r, Piece.load(r.root / "notes.json"), r.read_json("stems.json"), force)
    typer.echo(json.dumps([{k: t[k] for k in ("track", "audio_match", "precision", "recall")} for t in rep["tracks"]]))


@app.command()
def notate(target: str, fingering: bool = False, force: bool = False) -> None:
    """Quantize, spell, voice, build the score -> score.musicxml."""
    from m2s.ir import Piece
    from m2s.pipeline import notate as stage

    r = _run(target)
    stage(r, Piece.load(r.root / "notes.json"), r.read_json("profile.json"), fingering, force)


@app.command()
def engrave(target: str, engine: str = "auto", force: bool = False) -> None:
    """Render score.musicxml -> score.pdf (MuseScore if installed, else Verovio)."""
    from m2s.pipeline import engrave as stage

    stage(_run(target), force, engine)


@app.command()
def inspect(target: str, bars: str = typer.Option(..., help="Bar range, e.g. 12-16"),
            out: Optional[str] = None) -> None:
    """One image of bars a-b: stem spectrogram, decoded notes by hand, and the engraved bars."""
    from m2s.inspect import inspect_bars

    a, b = (int(x) for x in bars.split("-"))
    r = _run(target)
    typer.echo(inspect_bars(r, a, b, out))


@app.command()
def edit(target: str, patch: str) -> None:
    """Apply a JSON patch through the verification gate; appends to edits.jsonl."""
    from m2s.edit import apply_patch

    res = apply_patch(_run(target), json.loads(Path(patch).read_text()))
    typer.echo(json.dumps(res, indent=1))


@app.command()
def eval(target: str, ref: str = typer.Option(..., help="Reference score: MusicXML, MIDI, kern or Guitar Pro"),
         out: Optional[str] = None, repeats: str = typer.Option("auto", help="auto, expand or none"),
         diff_bars: int = typer.Option(0, help="Also run musicdiff on the first N bars (slow)"),
         mv2h: bool = True) -> None:
    """Blind scoring against a reference score -> eval.json."""
    from m2s.evaluate import evaluate_run

    rep = evaluate_run(_run(target), ref, repeats=repeats, diff_bars=diff_bars, use_mv2h=mv2h)
    if out:
        Path(out).write_text(json.dumps(rep, indent=1))
    typer.echo(json.dumps(rep.get("summary", rep), indent=1))


@app.command()
def presets() -> None:
    """List instrument presets."""
    from m2s.play.preset import available, load_preset

    for name in available():
        p = load_preset(name)
        typer.echo(f"{name:16s} {p['family']:9s} {p['display']}")


if __name__ == "__main__":
    app()
