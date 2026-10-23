"""Download a small piano dev set from the ASAP dataset (scores + one performance MIDI each).

The blind-test piece (Beethoven Sonata No. 14) is excluded on purpose: tuning
must never see it.

    python tools/fetch_dev_set.py            # -> data/dev/asap/<folder>/...
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
DEV = [
    "Bach/Prelude/bwv_846",
    "Chopin/Etudes_op_10/1",
    "Chopin/Etudes_op_10/12",
    "Beethoven/Piano_Sonatas/23-1",
    "Mozart/Piano_Sonatas/12-1",
    "Schubert/Impromptu_op.90_D.899/2",
    "Beethoven/Piano_Sonatas/8-3",
    "Haydn/Keyboard_Sonatas/31-1",
]
EXCLUDE = ("Beethoven/Piano_Sonatas/14-",)


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main(out: str = "data/dev/asap") -> None:
    rows = list(csv.DictReader(io.StringIO(get(RAW + "metadata.csv").decode())))
    root = Path(out)
    for folder in DEV:
        if folder.startswith(EXCLUDE):
            sys.exit(f"refusing to fetch excluded piece {folder}")
        row = next(r for r in rows if r["folder"] == folder)
        dst = root / folder
        dst.mkdir(parents=True, exist_ok=True)
        for key in ("xml_score", "midi_score", "midi_performance", "performance_annotations",
                    "midi_score_annotations"):
            rel = row[key]
            target = dst / Path(rel).name
            if not target.exists():
                target.write_bytes(get(RAW + rel))
        (dst / "performance.txt").write_text(Path(row["midi_performance"]).name + "\n")
        print("ok", folder)


if __name__ == "__main__":
    main(*sys.argv[1:])
