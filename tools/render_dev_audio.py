"""Render dev-set performance MIDI to audio with FluidSynth (General MIDI piano).

    python tools/render_dev_audio.py [folder-substring ...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from devset import pieces  # noqa: E402

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"


def render(midi: Path, wav: Path, sf2: str = SF2) -> None:
    subprocess.run(["fluidsynth", "-ni", "-g", "0.8", "-r", "44100", "-F", str(wav), sf2, str(midi)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    want = sys.argv[1:]
    for folder in pieces():
        if want and not any(w in str(folder) for w in want):
            continue
        perf = (folder / "performance.txt").read_text().strip()
        out = folder / "performance.wav"
        if not out.exists():
            render(folder / perf, out)
        print("ok", out)
