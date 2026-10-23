"""Worker helpers that do not need the models."""

from __future__ import annotations

from m2s.ir import Note, Track
from m2s.worker import semitone_tiebreak

# C major scale and I-IV-V-I bass
C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72, 48, 53, 55, 48]


def _track(pitches):
    return Track(name="piano", preset="piano",
                 notes=[Note(id=f"n{i}", onset=i * 0.5, offset=i * 0.5 + 0.5, pitch=p) for i, p in enumerate(pitches)])


def test_tiebreak_ignores_well_tuned_recordings():
    assert semitone_tiebreak([_track([p + 1 for p in C_MAJOR])], tuning_cents=10.0) == 0


def test_tiebreak_moves_a_half_sharp_recording_to_the_simpler_key():
    # C major heard a semitone high (C# major, 7 sharps) from a recording tuned +50 cents
    assert semitone_tiebreak([_track([p + 1 for p in C_MAJOR])], tuning_cents=48.0) == -1
    # B major (5 sharps) from one tuned -50 cents: C major is a semitone up
    assert semitone_tiebreak([_track([p - 1 for p in C_MAJOR])], tuning_cents=-48.0) == 1


def test_tiebreak_keeps_a_key_that_is_already_simple():
    assert semitone_tiebreak([_track(C_MAJOR)], tuning_cents=45.0) == 0


def test_retuned_activations_return_to_the_original_clock():
    import numpy as np

    from m2s.transcribe.piano import _restretch

    # a recording 50 cents flat is played 2^(50/1200) times faster to retune it, so an event at
    # original frame 1000 sits at retuned frame 1000 / ratio; mapping back must restore 1000
    ratio = 2 ** (50 / 1200)
    a = np.zeros((2000, 1), np.float32)
    a[int(round(1000 / ratio)), 0] = 1.0
    b = _restretch(a, ratio)
    assert abs(b.shape[0] - 2000 * ratio) <= 1
    assert abs(int(np.argmax(b[:, 0])) - 1000) <= 1
