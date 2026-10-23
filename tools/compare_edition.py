"""Side-by-side excerpts of a reference edition and our transcription, for the README.

Picks four 4-bar stretches of the reference (the opening, then the best, a typical and the worst by
note F1), renders the edition's bars and the bars of ours that cover the same seconds of the
recording with Verovio, and draws a piano roll of matched, missed and extra notes. Each excerpt
becomes one PNG (needs Chrome or Chromium; otherwise only the SVG and HTML parts are written).

Needs a run that has been scored with `m2s eval` (for eval_reference.json, the aligned edition):

    python tools/compare_edition.py data/rustruns/<hash> eval/references/test1/sonata14-3.krn \\
        --out docs/images --name moonlight [--scores DIR]   # DIR: scored.json etc. from elsewhere
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2s.evaluate.reference import expand_humdrum  # noqa: E402

TOL = 0.1  # onset tolerance (s) for a note to count as found
INK, MISSED, EXTRA, GRID, ED, US = "#2a2e36", "#b97700", "#c3342b", "#e3e6eb", "#2b4aa8", "#6b7280"


def written_bars(ref: Path, n: int) -> list[int]:
    """Written bar number of each expanded bar (Humdrum keeps them through repeats)."""
    if ref.suffix.lower() == ".krn":
        p = expand_humdrum(ref, "") or ref
        nums = [int(m.group(1)) for line in p.read_text().splitlines() if (m := re.match(r"^=+(\d+)", line.split("\t")[0]))]
        if len(nums) == n:
            return nums
    return list(range(1, n + 1))


def match(ref: list[dict], ours: list[dict], tol: float) -> tuple[dict, dict]:
    """Same pitch, closest onsets first, within tol."""
    by_pitch: dict[int, list[int]] = {}
    for j, n in enumerate(ours):
        by_pitch.setdefault(n["pitch"], []).append(j)
    cand = sorted((abs(ours[j]["onset"] - r["on"]), i, j) for i, r in enumerate(ref) for j in by_pitch.get(r["pitch"], [])
                  if abs(ours[j]["onset"] - r["on"]) <= tol)
    ri: dict[int, int] = {}
    oj: dict[int, int] = {}
    for _, i, j in cand:
        if i not in ri and j not in oj:
            ri[i], oj[j] = j, i
    return ri, oj


def f1(tp: int, ne: int, nr: int) -> float:
    p, r = tp / max(ne, 1), tp / max(nr, 1)
    return 2 * p * r / max(p + r, 1e-9)


def render(path: Path, bars: str) -> str:
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2400, "pageHeight": 60000, "adjustPageHeight": True, "scale": 40, "breaks": "none",
                   "header": "none", "footer": "none", "pageMarginTop": 30, "pageMarginBottom": 10,
                   "pageMarginLeft": 10, "pageMarginRight": 30})
    tk.loadFile(str(path))
    tk.select({"measureRange": bars})
    tk.redoLayout()
    return tk.renderToSVG(1)


def roll(ref, ours, rn, on, ri, oj, t0, t1, ref_marks, our_marks) -> str:
    pitches = [ref[i]["pitch"] for i in rn] + [ours[j]["pitch"] for j in on]
    lo, hi = min(pitches) - 1, max(pitches) + 1
    W, rowh, top = 1000, 6, 18
    H = (hi - lo + 1) * rowh

    def x(t):
        return (t - t0) / (t1 - t0) * W

    def y(p):
        return top + (hi - p) * rowh

    out = [f'<svg viewBox="-34 0 {W + 40} {H + top + 22}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="ui-monospace, Menlo, monospace" font-size="10">']
    for p in range(lo, hi + 1):
        if p % 12 == 0:
            yy = y(p) + rowh / 2
            out.append(f'<line x1="0" x2="{W}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>'
                       f'<text x="-6" y="{yy + 3:.1f}" text-anchor="end" fill="#5a616d">C{p // 12 - 1}</text>')
    for k, t in ref_marks:
        if t0 - 1e-6 <= t <= t1 + 1e-6:
            out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 4}" y2="{H + top}" stroke="{ED}" '
                       f'stroke-dasharray="3 3"/><text x="{x(t) + 3:.1f}" y="{top - 7}" fill="{ED}">{k}</text>')
    for k, t in our_marks:
        if t0 - 1e-6 <= t <= t1 + 1e-6:
            out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{H + top + 4}" stroke="{US}"/>'
                       f'<text x="{x(t) + 3:.1f}" y="{H + top + 16}" fill="{US}">{k}</text>')
    for i in rn:
        if i not in ri:
            n = ref[i]
            a, b = x(n["on"]), x(min(n["off"], t1))
            out.append(f'<rect x="{a:.1f}" y="{y(n["pitch"]):.1f}" width="{max(b - a, 4):.1f}" height="{rowh - 1}" '
                       f'fill="none" stroke="{MISSED}" stroke-width="1.4"/>')
    for j in on:
        n = ours[j]
        a, b = x(n["onset"]), x(min(n["offset"], t1))
        out.append(f'<rect x="{a:.1f}" y="{y(n["pitch"]):.1f}" width="{max(b - a, 4):.1f}" height="{rowh - 1}" '
                   f'fill="{INK if j in oj else EXTRA}"/>')
    out.append("</svg>")
    return "".join(out)


def page(ed_svg: str, our_svg: str, roll_svg: str, ed_label: str, our_label: str) -> str:
    b64 = lambda s: base64.b64encode(s.encode()).decode()  # noqa: E731
    sw = 'style="display:inline-block;width:20px;height:8px;margin:0 6px 0 16px;vertical-align:middle;{}"'
    return f"""<!doctype html><meta charset="utf-8"><style>
body{{margin:0;background:#fff;color:#1a1d23;font:15px/1.4 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}}
.w{{width:1400px;padding:18px 24px 22px;box-sizing:border-box}}
.lab{{font-size:14px;color:#5a616d;margin:12px 0 6px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.tag{{font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.06em;text-transform:uppercase;padding:4px 7px;border-radius:3px}}
.ed{{color:{ED};border:1px solid {ED}}} .us{{color:#fff;background:#1a1d23}}
img,svg{{display:block;width:100%;height:auto}}
</style><div class="w">
<div class="lab"><span class="tag ed">Edition</span>{ed_label}</div>
<img src="data:image/svg+xml;base64,{b64(ed_svg)}">
<div class="lab"><span class="tag us">music2sheets</span>{our_label}</div>
<img src="data:image/svg+xml;base64,{b64(our_svg)}">
<div class="lab">Note by note, in recording time:<span {sw.format('background:' + INK)}></span>in both
<span {sw.format('border:1.5px solid ' + MISSED)}></span>in the edition, missed
<span {sw.format('background:' + EXTRA)}></span>only in ours
<span style="margin-left:16px">bar lines: edition above (blue), ours below (grey)</span></div>
{roll_svg}
</div>"""


def screenshot(html_path: Path, png: Path) -> bool:
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if chrome is None:
        return False
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    "--window-size=1400,4000", f"--screenshot={png}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=120)
    from PIL import Image

    im = Image.open(png).convert("RGB")
    a = np.asarray(im)
    rows = np.where((a < 250).any(axis=(1, 2)))[0]
    im.crop((0, 0, im.width, int(rows.max()) + 20)).save(png, optimize=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ref")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--name", default="compare")
    ap.add_argument("--scores", help="folder with scored.json, notation.json, score.musicxml (default: the run)")
    ap.add_argument("--bars", type=int, default=4)
    a = ap.parse_args()
    run, refp, out = Path(a.run), Path(a.ref), Path(a.out)
    scores = Path(a.scores) if a.scores else run
    out.mkdir(parents=True, exist_ok=True)

    ev = json.loads((run / "eval_reference.json").read_text())
    ref = ev["notes"]
    bar_qons = np.array(ev["bar_qons"])
    nums = written_bars(refp, len(bar_qons))
    scored = json.loads((scores / "scored.json").read_text())
    ours = [n for t in scored["tracks"] for n in t["notes"] if n.get("keep", True)]
    info = json.loads((scores / "notation.json").read_text())
    our_bars = np.array(info["bar_times"])

    # reference quarters -> recording seconds, from the aligned notes
    qs = np.array([n["qon"] for n in ref])
    order = np.argsort(qs)
    uq, idx = np.unique(qs[order], return_index=True)
    ut = np.maximum.accumulate(np.array([n["on"] for n in ref])[order][idx])
    ref_bar_t = np.interp(bar_qons, uq, ut)
    ref_bar = np.searchsorted(bar_qons, [n["qon"] for n in ref], side="right") - 1
    end = max(n["off"] for n in ref)

    ri, oj = match(ref, ours, TOL)
    ri50, _ = match(ref, ours, 0.05)
    agree = sum(1 for i, j in ri.items() if (ours[j].get("effector", {}).get("hand") == "R") == (ref[i]["staff"] == 1))
    overall = {"f1_50": round(f1(len(ri50), len(ours), len(ref)), 3), "f1_100": round(f1(len(ri), len(ours), len(ref)), 3),
               "hands": round(agree / max(len(ri), 1), 3), "our_bars": len(our_bars), "ref_bars": len(nums),
               "key": info["key"], "time_signature": info["time_signature"]}

    def stats(b0: int, b1: int) -> dict:
        t0 = ref_bar_t[b0]
        t1 = ref_bar_t[b1 + 1] if b1 + 1 < len(ref_bar_t) else end
        rn = [i for i in range(len(ref)) if b0 <= ref_bar[i] <= b1]
        on = [j for j, n in enumerate(ours) if t0 - 0.03 <= n["onset"] < t1 - 0.03]
        tp = sum(1 for i in rn if i in ri)
        return {"t0": t0, "t1": t1, "R": rn, "O": on, "tp": tp, "f1": f1(tp, len(on), len(rn))}

    n = a.bars
    wins = [(b, stats(b, b + n - 1)["f1"]) for b in range(len(nums) - n)
            if all(nums[b + k] == nums[b] + k for k in range(n)) and len(stats(b, b + n - 1)["R"]) >= 6 * n]
    taken = [0]

    def pick(cands):
        b = next(b for b, _ in cands if all(abs(b - t) >= 2 * n for t in taken))
        taken.append(b)
        return b

    by_f1 = sorted(wins, key=lambda w: w[1])
    med = float(np.median([w[1] for w in wins]))
    chosen = [("opening", 0), ("worst", pick(by_f1)), ("best", pick(by_f1[::-1])),
              ("typical", pick(sorted(wins, key=lambda w: abs(w[1] - med))))]
    chosen = [chosen[0], chosen[2], chosen[3], chosen[1]]

    report = {"overall": overall, "excerpts": []}
    for label, b in chosen:
        s = stats(b, b + n - 1)
        w0, w1 = nums[b], nums[b + n - 1]
        k0 = max(1, int(np.searchsorted(our_bars, s["t0"] + 0.05, side="right")))
        k1 = int(np.searchsorted(our_bars, s["t1"] - 0.05, side="right"))
        ed_svg, our_svg = render(refp, f"{w0}-{w1}"), render(scores / "score.musicxml", f"{k0}-{k1}")
        ref_marks = [(nums[k], ref_bar_t[k]) for k in range(b, min(b + n + 1, len(nums)))]
        our_marks = [(k + 1, our_bars[k]) for k in range(max(k0 - 1, 0), min(k1 + 1, len(our_bars)))]
        rsvg = roll(ref, ours, s["R"], s["O"], ri, oj, s["t0"], s["t1"], ref_marks, our_marks)
        repeat = " (second time)" if nums[:b].count(w0) else ""
        html = page(ed_svg, our_svg, rsvg, f"bars {w0}–{w1}{repeat}",
                    f"bars {k0}–{k1}, transcribed from the recording ({s['t0']:.1f}–{s['t1']:.1f} s)")
        stem = f"{a.name}-{label}"
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / f"{stem}.html"
            hp.write_text(html)
            made = screenshot(hp, out / f"{stem}.png")
            if not made:
                shutil.copy(hp, out / f"{stem}.html")
        report["excerpts"].append({"label": label, "edition_bars": [w0, w1], "repeat": bool(repeat), "our_bars": [k0, k1],
                                   "seconds": [round(s["t0"], 1), round(s["t1"], 1)], "edition_notes": len(s["R"]),
                                   "found": s["tp"], "missed": len(s["R"]) - s["tp"],
                                   "extra": sum(1 for j in s["O"] if j not in oj), "f1": round(s["f1"], 3),
                                   "image": f"{stem}.png" if made else f"{stem}.html"})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
