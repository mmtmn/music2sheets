"""Reference loading: Guitar Pro (string/fret -> pitch, ticks -> quarters) and Humdrum repeat expansion."""

from __future__ import annotations

import pytest

guitarpro = pytest.importorskip("guitarpro")

from m2s.evaluate.reference import expand_humdrum, humdrum_expansions, load_reference  # noqa: E402


def test_guitar_pro_pitches_and_times(tmp_path):
    song = guitarpro.Song()
    voice = song.tracks[0].measures[0].voices[0]
    # one chord: high E string fret 3 (G4), B string open (B3), low E string fret 5 (A2)
    beat = guitarpro.Beat(voice, duration=guitarpro.Duration(value=4))
    for string, fret in ((1, 3), (2, 0), (6, 5)):
        beat.notes.append(guitarpro.Note(beat, value=fret, string=string, type=guitarpro.NoteType.normal))
    voice.beats.append(beat)
    path = tmp_path / "t.gp5"
    guitarpro.write(song, str(path))
    ref = load_reference(path)
    # standard tuning: string 1 = E4 (64), string 2 = B3 (59), string 6 = E2 (40)
    assert [n.pitch for n in ref.notes] == [45, 59, 67]
    # Guitar Pro starts the song at tick 960; that is quarter 0
    assert {n.qon for n in ref.notes} == {0.0}
    assert all(n.qdur == 1.0 for n in ref.notes)


def test_humdrum_expansion(tmp_path):
    krn = tmp_path / "x.krn"
    krn.write_text("\n".join([
        "**kern", "*>[A,A,B]", "*>A", "*M2/4", "=1", "4c", "4d", "=2", "4e", "4f",
        "*>B", "=3", "2g", "==", "*-"]) + "\n")
    out = expand_humdrum(krn)
    body = [ln for ln in out.read_text().splitlines() if ln and ln[0].isdigit()]
    assert body == ["4c", "4d", "4e", "4f", "4c", "4d", "4e", "4f", "2g"]


def test_humdrum_named_expansions_pick_the_closest(tmp_path):
    krn = tmp_path / "y.krn"
    krn.write_text("\n".join([
        "**kern", "*>[A,A,B]", "*>norep[A,B]", "*>A", "*M2/4", "=1", "4c", "4d", "=2", "4e", "4f",
        "*>B", "=3", "2g", "==", "*-"]) + "\n")
    assert humdrum_expansions(krn) == {"": ["A", "A", "B"], "norep": ["A", "B"]}
    body = [ln for ln in expand_humdrum(krn, "norep").read_text().splitlines() if ln and ln[0].isdigit()]
    assert body == ["4c", "4d", "4e", "4f", "2g"]
    # a performance without the repeat (5 notes) is compared with the no-repeat order
    assert len(load_reference(krn, expected_notes=5).notes) == 5
    assert len(load_reference(krn, expected_notes=9).notes) == 9
