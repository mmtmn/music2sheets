"""Stage 4: playability decoding. One entry point; the preset's family picks the effector model."""

from __future__ import annotations

from typing import Any

from m2s.ir import Track
from m2s.play.preset import load_preset


def decode_track(track: Track) -> list[dict[str, Any]]:
    family = load_preset(track.preset)["family"]
    if family == "keyboard":
        from m2s.play.piano import PianoDecoder

        return [v.as_dict() for v in PianoDecoder(track.preset).decode(track)]
    if family == "fretted":
        from m2s.play.fretted import FrettedDecoder

        return [v.as_dict() for v in FrettedDecoder(track.preset).decode(track)]
    if family == "drums":
        from m2s.play.drums import DrumDecoder

        return [v.as_dict() for v in DrumDecoder(track.preset).decode(track)]
    if family == "wind":
        from m2s.play.mono import MonoDecoder

        return [v.as_dict() for v in MonoDecoder(track.preset).decode(track)]
    raise ValueError(f"no decoder for family {family!r}")
