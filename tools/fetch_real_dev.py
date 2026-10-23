"""Real-recording piano dev set: public-domain / CC0 recordings from Wikimedia Commons, with
Craig Sapp's Humdrum editions as references. Nothing from Sonata No. 14 (the blind test).

    python tools/fetch_real_dev.py          # -> data/realdev/audio/*.ogg, data/realdev/refs/*.krn
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "music2sheets-dev/0.2 (research)"}
KERN = "https://raw.githubusercontent.com/craigsapp/beethoven-piano-sonatas/master/kern/sonata{}.krn"
# id -> (Commons file, kern movement)
SET = {
    "08-1": ("Beethoven, Sonata No. 8 in C Minor Pathetique, Op. 13 - I. Grave - Allegro di molto e con brio.ogg", "08-1"),
    "08-2": ("Beethoven, Sonata No. 8 in C Minor Pathetique, Op. 13 - II. Adagio cantabile.ogg", "08-2"),
    "08-3": ("Beethoven, Sonata No. 8 in C Minor Pathetique, Op. 13 - III. Rondo - Allegro.ogg", "08-3"),
    "21-1": ("Ludwig van Beethoven - sonata no. 21 in c major 'waldstein', op. 53 - i. allegro con br.ogg", "21-1"),
    "21-3": ("Ludwig van Beethoven - sonata 'waldstein', op. 53 - iii. rondo- allegretto moderato.ogg", "21-3"),
    "15-1": ("Karine Gilanyan - Beethoven - Piano Sonata nr15 in D major op28 Pastoral - I Allegro.ogg", "15-1"),
    "15-3": ("Karine Gilanyan - Beethoven - Piano Sonata nr15 in D major op28 Pastoral - III Scherzo Allegro Vivace.ogg", "15-3"),
    "31-2": ("Ludwig van Beethoven - sonata no. 31 in a flat major, op. 110 - ii. allegro molto.ogg", "31-2"),
}


def get(url: str, tries: int = 6) -> bytes:
    """GET with backoff: Wikimedia answers 429 when asked too quickly."""
    import time
    import urllib.error

    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 0) or 10 * (k + 1))
    raise RuntimeError("unreachable")


def main(out: str = "data/realdev") -> None:
    root = Path(out)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(parents=True, exist_ok=True)
    meta = {}
    for key, (fname, kern) in SET.items():
        if "14-" in kern:
            sys.exit("refusing Sonata No. 14: it is the blind test")
        api = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo"
               "&iiprop=url|extmetadata&titles=" + urllib.parse.quote("File:" + fname))
        info = next(iter(json.loads(get(api))["query"]["pages"].values()))["imageinfo"][0]
        lic = info["extmetadata"].get("LicenseShortName", {}).get("value", "?")
        audio = root / "audio" / f"{key}.ogg"
        if not audio.exists():
            audio.write_bytes(get(info["url"]))
        ref = root / "refs" / f"{key}.krn"
        if not ref.exists():
            ref.write_bytes(get(KERN.format(kern)))
        __import__("time").sleep(3)
        meta[key] = {"file": fname, "license": lic, "source": info["descriptionurl"] if "descriptionurl" in info else info["url"]}
        print(f"ok {key} ({lic})")
    (root / "SOURCES.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
