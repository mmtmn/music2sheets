"""Phase 0 gate: a known MIDI file round-trips through notation and MusicXML with no note lost."""

from __future__ import annotations

import random

import music21

from m2s.ir import Note, Piece, Track
from m2s.notate import notate


def _synthetic(seed: int = 0, bars: int = 12, qpm: float = 96.0) -> tuple[Piece, list[float]]:
    rnd = random.Random(seed)
    spq = 60 / qpm
    notes = []
    i = 0
    # right hand: eighths and sixteenths, a held note now and then; left hand: quarter-note chords
    q = 0.0
    while q < bars * 4:
        d = rnd.choice([0.5, 0.5, 0.25, 0.25, 1.0, 0.75])
        p = rnd.randint(64, 84)
        notes.append(Note(f"n{i}", q * spq, (q + d) * spq, p, 80, effector={"hand": "R"}))
        i += 1
        q += d
    for b in range(bars * 4):
        root = rnd.choice([36, 41, 43, 45])
        for p in (root, root + 7, root + 12):
            notes.append(Note(f"n{i}", b * spq, (b + 1) * spq, p, 70, effector={"hand": "L"}))
            i += 1
    beats = [k * spq for k in range(bars * 4 + 1)]
    return Piece([Track("piano", "piano", notes)]), beats


def _collect(score: music21.stream.Score) -> list[tuple[float, int, float]]:
    out = []
    for part in score.parts:
        for n in part.stripTies().recurse().notes:
            off = float(n.getOffsetInHierarchy(part))
            for p in n.pitches:
                out.append((round(off, 4), p.midi, round(float(n.quarterLength), 4)))
    return sorted(out)


def test_roundtrip_lossless(tmp_path):
    piece, beats = _synthetic()
    downbeats = beats[::4]
    score, grid, info = notate(piece, beats, downbeats)
    assert info["time_signature"] == "4/4"
    xml = tmp_path / "rt.musicxml"
    score.write("musicxml", fp=str(xml))
    back = music21.converter.parse(str(xml))
    got = _collect(back)
    want = sorted((round(n.beat, 4), n.pitch, round(n.qdur, 4)) for n in piece.tracks[0].notes)
    assert len(got) == len(want)
    assert got == want


def test_last_note_rounded_past_final_bar(tmp_path):
    """A held last note whose rounded value overshoots the final bar must not hang notation."""
    spq = 0.5
    notes = [Note("a", 0.0, 0.5, 60, 80, effector={"hand": "R"}),
             Note("b", 0.5, 1.0, 62, 80, effector={"hand": "R"}),
             # held from beat 1.5 to ~beat 5.4 while another note moves: voice 2, rounded up to 4 beats
             Note("c", 0.75, 2.7, 55, 80, effector={"hand": "R"}),
             Note("d", 1.0, 1.25, 64, 80, effector={"hand": "R"})]
    beats = [k * spq for k in range(6)]
    score, _, _ = notate(Piece([Track("piano", "piano", notes)]), beats, beats[::4])
    score.write("musicxml", fp=str(tmp_path / "x.musicxml"))
