"""`m2s inspect`: one picture of a few bars, for review.

Top: the stem's constant-Q spectrogram. Middle: decoded notes by hand or
string (dropped candidates in grey, verify disagreements outlined). Bottom:
the engraved bars as they will print.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from m2s import audio as A
from m2s.ir import Piece
from m2s.runs import Run

COLORS = {"R": "#1f6feb", "L": "#d1495b", None: "#2a9d8f"}


def bar_times(piece: Piece, a: int, b: int) -> tuple[float, float]:
    on = [(n.measure, n.onset) for _, n in piece.all_notes() if n.keep and n.measure is not None]
    ins = [t for m, t in on if a <= m <= b]
    if not ins:
        raise ValueError(f"no notes in bars {a}-{b}")
    nxt = [t for m, t in on if m == b + 1]
    return min(ins), (min(nxt) if nxt else max(ins) + 1.0)


def inspect_bars(run: Run, a: int, b: int, out: str | None = None) -> str:
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from m2s.engrave import render_bars_png

    piece = Piece.load(run.root / "notes.json")
    _fill_measures(piece, run)
    t0, t1 = bar_times(piece, a, b)
    t0, t1 = max(0, t0 - 0.2), t1 + 0.2
    stems = run.read_json("stems.json")
    track = piece.tracks[0]
    wav = stems.get(track.preset, str(run.root / "audio.wav"))
    y, sr = A.load(wav, sr=22050, mono=True)
    seg = y[int(t0 * sr): int(t1 * sr)]
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(seg, sr=sr, hop_length=256, fmin=librosa.midi_to_hz(21),
                                                   n_bins=88)), ref=np.max)
    bars_png = run.path("inspect", f"bars_{a}-{b}.png")
    render_bars_png(run.root / "score.musicxml", (a, b), bars_png)
    img = plt.imread(str(bars_png))

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [1, 1, 1.4]})
    axes[0].imshow(C, origin="lower", aspect="auto", extent=(t0, t1, 21, 109), cmap="magma")
    axes[0].set_title(f"stem spectrogram, bars {a}-{b}")
    axes[0].set_ylabel("MIDI")
    ax = axes[1]
    for tr in piece.tracks:
        for n in tr.notes:
            if n.offset < t0 or n.onset > t1:
                continue
            hand = n.effector.get("hand")
            col = COLORS.get(hand, COLORS[None]) if n.keep else "#bbbbbb"
            ax.add_patch(plt.Rectangle((n.onset, n.pitch - 0.4), n.duration, 0.8, color=col, alpha=0.9))
            if n.keep and "string" in n.effector:
                ax.text(n.onset, n.pitch + 0.5, f"{n.effector['string']}/{n.effector['fret']}", fontsize=6)
    ax.set_xlim(t0, t1)
    ps = [n.pitch for _, n in piece.all_notes() if t0 <= n.onset <= t1] or [60]
    ax.set_ylim(min(ps) - 3, max(ps) + 3)
    ax.set_title("decoded notes (blue right hand, red left hand, grey dropped)")
    ax.set_xlabel("seconds")
    axes[2].imshow(img)
    axes[2].axis("off")
    fig.tight_layout()
    dest = Path(out) if out else run.path("inspect", f"inspect_{a}-{b}.png")
    fig.savefig(dest, dpi=80)
    plt.close(fig)
    return str(dest)


def _fill_measures(piece: Piece, run: Run) -> None:
    """notes.json carries quarter positions; derive bar numbers from the notation info."""
    info = run.read_json("notation.json")
    ts = info["time_signature"]
    num, den = (int(x) for x in ts.split("/"))
    bar_q = num * 4 / den
    pickup = info.get("pickup_quarters", 0.0)
    for _, n in piece.all_notes():
        if n.beat is not None:
            q = n.beat + (bar_q - pickup if pickup else 0)
            n.measure = int(q // bar_q) + (0 if pickup else 1)
