"""Synthetic band test with exact ground truth.

Writes an original 16-bar rock pattern (generated here, fixed seed): drums,
bass, two distorted guitars panned hard left/right playing a harmonized line,
and an alto-sax melody. Renders the stereo mix with FluidSynth and saves the
per-instrument ground-truth MIDI.

    python tools/make_band_test.py [out_dir]       # default data/bandtest
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pretty_midi

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
QPM = 132
SPQ = 60 / QPM
E_MINOR = [40, 42, 43, 45, 47, 48, 50]  # E F# G A B C D (E2 octave)
PROG = [0, 5, 2, 6]                     # i - VI - III - VII (degrees in E minor)


def scale_note(degree: int, octave: int) -> int:
    return E_MINOR[degree % 7] + 12 * (octave + degree // 7)


def build(seed: int = 7) -> dict[str, pretty_midi.Instrument]:
    rnd = random.Random(seed)
    drums = pretty_midi.Instrument(0, is_drum=True, name="drums")
    bass = pretty_midi.Instrument(33, name="bass")
    gl = pretty_midi.Instrument(30, name="guitar_L")
    gr = pretty_midi.Instrument(30, name="guitar_R")
    sax = pretty_midi.Instrument(65, name="sax_alto")
    bars = 16
    for b in range(bars):
        t0 = b * 4 * SPQ
        root = PROG[b % 4]
        # drums: kick on 1 and 3 (+ the "and" of 3), snare on 2 and 4, closed hi-hat eighths
        for q in range(8):
            t = t0 + q * SPQ / 2
            drums.notes.append(pretty_midi.Note(80, 42, t, t + 0.05))
        for q in (0, 2, 2.5):
            t = t0 + q * SPQ
            drums.notes.append(pretty_midi.Note(110, 36, t, t + 0.1))
        for q in (1, 3):
            t = t0 + q * SPQ
            drums.notes.append(pretty_midi.Note(105, 38, t, t + 0.1))
        if b % 4 == 3:  # crash into the next phrase
            drums.notes.append(pretty_midi.Note(100, 49, t0 + 4 * SPQ - 0.001, t0 + 4 * SPQ + 0.3))
        # bass: root eighths, octave jump on beat 4
        for q in range(8):
            t = t0 + q * SPQ / 2
            p = scale_note(root, -1) + (12 if q == 6 else 0)
            bass.notes.append(pretty_midi.Note(95, p, t, t + SPQ / 2 * 0.9))
        # guitars: a harmonized eighth-note line, left a third above right
        for q in range(8):
            t = t0 + q * SPQ / 2
            deg = root + [0, 2, 4, 2, 0, 2, 4, 6][q]
            gr.notes.append(pretty_midi.Note(90, scale_note(deg, 1), t, t + SPQ / 2 * 0.95))
            gl.notes.append(pretty_midi.Note(90, scale_note(deg + 2, 1), t, t + SPQ / 2 * 0.95))
        # sax melody: a seeded random walk on quarters and halves, phrases with breaths
        if b % 2 == 0:
            q, deg = 0.0, root + 7
            while q < 7.0:
                d = rnd.choice([1.0, 1.0, 2.0, 0.5, 0.5])
                deg += rnd.choice([-2, -1, 1, 2])
                t = t0 + q * SPQ
                sax.notes.append(pretty_midi.Note(85, scale_note(deg, 2) - 12, t, t + d * SPQ * 0.92))
                q += d
    return {"drums": drums, "bass": bass, "guitar_L": gl, "guitar_R": gr, "sax_alto": sax}


def main(out: str = "data/bandtest") -> None:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    parts = build()
    pm = pretty_midi.PrettyMIDI(initial_tempo=QPM)
    pan = {"guitar_L": 5, "guitar_R": 122, "bass": 64, "sax_alto": 64, "drums": 64}
    for name, inst in parts.items():
        inst.control_changes.append(pretty_midi.ControlChange(10, pan[name], 0.0))
        pm.instruments.append(inst)
        one = pretty_midi.PrettyMIDI(initial_tempo=QPM)
        one.instruments.append(inst)
        one.write(str(outdir / f"gt_{name}.mid"))
    pm.write(str(outdir / "band.mid"))
    subprocess.run(["fluidsynth", "-ni", "-g", "0.6", "-r", "44100", "-F", str(outdir / "band.wav"), SF2,
                    str(outdir / "band.mid")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", outdir / "band.wav")


if __name__ == "__main__":
    main(*sys.argv[1:])
