"""Ornaments before quantization: trills and grace notes.

A trill shows up in a transcription as a burst of fast alternations between
two neighbouring pitches; written out it becomes a tuplet mess. We fold it
into its first note with a trill mark. A single very short note right before
a main note in the same hand becomes a grace note.
"""

from __future__ import annotations

from m2s.ir import Note


def fold_trills(notes: list[Note], max_ioi: float = 0.11, min_notes: int = 5) -> int:
    """Mark trills in place (extra notes get keep=False and tag 'trill-note'). Returns count."""
    found = 0
    by_hand: dict[str, list[Note]] = {}
    for n in sorted((n for n in notes if n.keep), key=lambda n: n.onset):
        by_hand.setdefault(str(n.effector.get("hand", "")), []).append(n)
    for seq in by_hand.values():
        i = 0
        while i < len(seq) - min_notes + 1:
            a, b = seq[i], seq[i + 1]
            step = abs(a.pitch - b.pitch)
            if step not in (1, 2) or b.onset - a.onset > max_ioi:
                i += 1
                continue
            j = i + 1
            while (j + 1 < len(seq) and seq[j + 1].onset - seq[j].onset <= max_ioi
                   and seq[j + 1].pitch == seq[j - 1].pitch):
                j += 1
            if j - i + 1 >= min_notes:
                main = seq[i]
                main.offset = max(n.offset for n in seq[i:j + 1])
                main.tags.append("trill")
                for n in seq[i + 1:j + 1]:
                    n.keep = False
                    n.tags.append("trill-note")
                found += 1
                i = j + 1
            else:
                i += 1
    return found


def mark_graces(notes: list[Note], max_dur: float = 0.07, max_gap: float = 0.09) -> int:
    found = 0
    by_hand: dict[str, list[Note]] = {}
    for n in sorted((n for n in notes if n.keep), key=lambda n: n.onset):
        by_hand.setdefault(str(n.effector.get("hand", "")), []).append(n)
    for seq in by_hand.values():
        for prev, a, b in zip([None] + seq[:-2], seq, seq[1:]):
            short = a.duration <= max_dur and 0.02 < b.onset - a.onset <= max_gap
            isolated = prev is None or a.onset - prev.onset > 0.25
            if short and isolated and a.pitch != b.pitch and abs(a.pitch - b.pitch) <= 4:
                a.tags.append("grace")
                found += 1
    return found
