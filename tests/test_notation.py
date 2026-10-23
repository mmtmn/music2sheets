"""Notation pieces: spelling, keys, quantization and meter on synthetic input."""

from __future__ import annotations

import numpy as np

from m2s.ir import Note
from m2s.notate.keys import Key, estimate_key
from m2s.notate.rhythm import build_grid, quantize
from m2s.notate.spell import fifths_name, spell


def _n(i: int, t: float, p: int, d: float = 0.4, hand: str = "R") -> Note:
    return Note(f"n{i}", t, t + d, p, 80, effector={"hand": hand})


def test_fifths_names():
    assert [fifths_name(k) for k in (0, 7, 12, -2, -5)] == ["C", "C#", "B#", "B-", "D-"]


def test_leading_tone_in_c_sharp_minor_is_b_sharp():
    # G#-B#-D# (the dominant of C# minor) after a C# minor triad
    notes = [_n(0, 0.0, 61), _n(1, 0.5, 64), _n(2, 1.0, 68), _n(3, 1.5, 68), _n(4, 2.0, 72), _n(5, 2.5, 75)]
    spell(notes, Key(1, "minor", 1.0))
    assert notes[4].spelling == "B#4"
    assert notes[0].spelling == "C#4"


def test_f_major_uses_flat():
    notes = [_n(i, i * 0.5, p) for i, p in enumerate((65, 67, 69, 70, 72))]
    spell(notes, Key(5, "major", 1.0))
    assert notes[3].spelling == "B-4"


def test_estimate_key_minor_from_bass():
    # an A minor progression with A in the bass first and last; the relative major (C) shares its notes
    notes, t = [], 0.0
    for chord in ((45, 60, 64, 69), (41, 60, 65, 69), (43, 59, 62, 67), (45, 60, 64, 69)):
        for p in chord:
            notes.append(_n(len(notes), t, p, 1.0))
        t += 1.0
    k = estimate_key(notes)
    assert (k.tonic_pc, k.mode) == (9, "minor")


def test_quantize_triplets_and_sixteenths():
    spq = 0.5
    beats = [i * spq for i in range(17)]
    notes = []
    # two beats of sixteenths, then two beats of triplet eighths, played a little unevenly
    rng = np.random.default_rng(0)
    for k in range(8):
        notes.append(_n(k, k * spq / 4 + rng.normal(0, 0.008), 72, 0.1))
    for k in range(6):
        notes.append(_n(8 + k, 2 * spq + k * spq / 3 + rng.normal(0, 0.008), 74, 0.1))
    grid = build_grid(beats, beats[::4], notes)
    quantize(notes, grid)
    assert [round(n.beat, 3) for n in notes[:8]] == [round(k * 0.25, 3) for k in range(8)]
    assert [round(n.beat, 3) for n in notes[8:]] == [round(2 + k / 3, 3) for k in range(6)]


def test_meter_three_four():
    # a waltz: bass on beat 1, chords on 2 and 3, harmony changes every bar
    spq = 0.45
    roots = [45, 43, 41, 40, 45, 43, 41, 40] * 2
    notes, beats = [], []
    for b, r in enumerate(roots):
        t = b * 3 * spq
        notes.append(_n(len(notes), t, r, 3 * spq, "L"))
        for k in (1, 2):
            for p in (r + 12 + 4, r + 12 + 7):
                notes.append(_n(len(notes), t + k * spq, p, spq * 0.8))
    beats = [i * spq for i in range(len(roots) * 3 + 1)]
    grid = build_grid(beats, [], notes)
    assert grid.time_signature() == "3/4"
    assert grid.first_downbeat_beat % 3 == 0
