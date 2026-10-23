"""Playability decoders on small hand-made inputs (no models, no GPU)."""

from __future__ import annotations

from m2s.ir import Note, Track
from m2s.play.drums import DrumDecoder
from m2s.play.fretted import FrettedDecoder
from m2s.play.mono import MonoDecoder
from m2s.play.piano import PianoDecoder


def _n(i: int, t: float, p: int, d: float = 0.2, conf: float = 1.0) -> Note:
    return Note(f"n{i}", t, t + d, p, 80, confidence=conf)


def test_piano_hands_far_apart():
    # a bass line two octaves below a melody: the decoder must not mix them up
    notes = []
    for k in range(16):
        notes.append(_n(2 * k, k * 0.25, 36 + (k % 4)))
        notes.append(_n(2 * k + 1, k * 0.25, 72 + (k % 5)))
    tr = Track("piano", "piano", notes)
    v = PianoDecoder().decode(tr)
    assert all(n.effector["hand"] == ("L" if n.pitch < 60 else "R") for n in tr.notes)
    assert not [x for x in v if x.kind in ("span", "count")]


def test_piano_wide_chord_is_split():
    # C2 C3 G4 G5 at once: no single hand can hold it, two hands (an octave each) can
    tr = Track("piano", "piano", [_n(i, 0.0, p, 1.0) for i, p in enumerate((36, 48, 67, 79))])
    v = PianoDecoder().decode(tr)
    hands = {n.pitch: n.effector["hand"] for n in tr.notes}
    assert hands[36] == hands[48] == "L" and hands[67] == hands[79] == "R"
    assert not [x for x in v if x.kind == "span"]


def test_piano_impossible_chord_is_flagged():
    # C2 E3 G4 C6: each hand would have to span more than a ninth
    tr = Track("piano", "piano", [_n(i, 0.0, p, 1.0) for i, p in enumerate((36, 52, 67, 84))])
    v = PianoDecoder().decode(tr)
    assert [x for x in v if x.kind == "span"]


def test_piano_uncertain_candidate_dropped_when_unplayable():
    # a confident 10-note cluster plus a doubtful 11th note: the 11th has nowhere to go
    notes = [_n(i, 0.0, p, 1.0) for i, p in enumerate((40, 43, 47, 50, 52, 64, 67, 71, 74, 76))]
    notes.append(_n(99, 0.0, 58, 1.0, conf=0.45))
    tr = Track("piano", "piano", notes)
    PianoDecoder().decode(tr)
    assert not tr.notes[-1].keep
    assert all(n.keep for n in tr.notes[:-1])


def test_piano_fingering_scale():
    # a C major scale up from middle C in the right hand: thumb on C, then a thumb crossing
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    tr = Track("piano", "piano", [_n(i, i * 0.2, p) for i, p in enumerate(pitches)])
    PianoDecoder().decode(tr)
    fingers = [n.effector["finger"] for n in tr.notes]
    assert fingers[0] == 1
    assert all(f in range(1, 6) for f in fingers)
    # never the same finger on two different keys in a row
    assert all(a != b for a, b in zip(fingers, fingers[1:]))


def test_guitar_strings_distinct_and_frets_valid():
    # an open E-minor chord: E2 B2 E3 G3 B3 E4
    tr = Track("gtr", "guitar", [_n(i, 0.0, p, 1.0) for i, p in enumerate((40, 47, 52, 55, 59, 64))])
    v = FrettedDecoder("guitar").decode(tr)
    strings = [n.effector["string"] for n in tr.notes]
    assert len(set(strings)) == 6
    tuning = [40, 45, 50, 55, 59, 64]
    for n in tr.notes:
        assert tuning[6 - n.effector["string"]] + n.effector["fret"] == n.pitch
    assert not [x for x in v if x.kind == "stretch"]


def test_violin_double_stops_adjacent():
    tr = Track("vln", "violin", [_n(0, 0.0, 62, 0.5), _n(1, 0.0, 69, 0.5)])
    FrettedDecoder("violin").decode(tr)
    s = sorted(n.effector["string"] for n in tr.notes)
    assert s[1] - s[0] == 1


def test_drum_limbs():
    hits = [_n(0, 0.0, 36), _n(1, 0.0, 42), _n(2, 0.5, 38), _n(3, 0.5, 42), _n(4, 1.0, 36)]
    tr = Track("drums", "drums", hits)
    DrumDecoder().decode(tr)
    limbs = {(n.onset, n.pitch): n.effector["limb"] for n in tr.notes}
    assert limbs[(0.0, 36)] in ("RF", "LF")
    assert limbs[(0.0, 42)] == "RH"
    assert limbs[(0.5, 38)] == "LH"


def test_voice_breath_violation():
    # 12 seconds of uninterrupted singing breaks the 10 s phrase limit
    notes = [_n(i, i * 0.5, 64, 0.5) for i in range(24)]
    v = MonoDecoder("voice").decode(Track("vox", "voice", notes))
    assert any(x.kind == "breath" for x in v)
