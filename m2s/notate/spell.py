"""Pitch spelling on the line of fifths.

Each note takes the enharmonic spelling closest to a running centre of
effect: half the key's centre, half the recent notes' positions. Chromatic
neighbours lean toward the direction they resolve (sharps going up, flats
going down). This yields B-sharp as a leading tone in C-sharp minor and
D-flat in a passage that moves flatward, without any table of special cases.
"""

from __future__ import annotations

from m2s.ir import Note
from m2s.notate.keys import Key

LETTERS = "FCGDAEB"  # line of fifths, position -1 .. 5


def fifths_name(pos: int) -> str:
    """Line-of-fifths position -> pitch-class name, e.g. 0 -> C, 7 -> C#, -2 -> B-."""
    letter = LETTERS[(pos + 1) % 7]
    acc = (pos + 1) // 7
    return letter + ("#" * acc if acc > 0 else "-" * (-acc))


def candidates(pc: int) -> list[int]:
    k0 = (7 * pc) % 12
    return [k for k in (k0 - 12, k0, k0 + 12) if -15 <= k <= 19]


def spell(notes: list[Note], key: Key, memory: float = 0.82) -> None:
    """Set ``note.spelling`` (music21 pitch name such as 'B#4') for every note."""
    order = sorted(notes, key=lambda n: (n.onset, n.pitch))
    recent = key.centre
    nxt: dict[str, Note] = {}
    # next note in the same hand, for chromatic-neighbour direction
    by_hand: dict[str, list[Note]] = {}
    for n in order:
        by_hand.setdefault(str(n.effector.get("hand", n.staff)), []).append(n)
    for seq in by_hand.values():
        for a, b in zip(seq, seq[1:]):
            nxt[a.id] = b
    for n in order:
        ce = 0.5 * key.centre + 0.5 * recent
        lean = 0.0
        b = nxt.get(n.id)
        if b is not None and abs(b.pitch - n.pitch) == 1:
            lean = 1.2 if b.pitch > n.pitch else -1.2
        pos = min(candidates(n.pitch % 12), key=lambda k: abs(k - (ce + lean)))
        recent = memory * recent + (1 - memory) * pos
        name = fifths_name(pos)
        # octave follows the letter: B#3 sounds as C4, C-4 sounds as B3
        octave = n.pitch // 12 - 1
        base = name[0]
        acc = name.count("#") - name.count("-")
        natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[base]
        octave = (n.pitch - acc - natural) // 12 - 1
        n.spelling = f"{name}{octave}"
