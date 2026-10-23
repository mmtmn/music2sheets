"""Falling-notes video of a transcription played as written: the sheet music, heard.

Notes come from a run's scored.json (each note's notated beat and length) and are played at a
steady tempo with the run's sustain pedal, rendered by FluidSynth; the picture is an 88-key
keyboard with the notes falling onto it, right hand orange, left hand blue. Needs fluidsynth,
ffmpeg and a General MIDI SoundFont (M2S_SOUNDFONT, as for `m2s verify`).

    python tools/make_video.py runs/<hash> out.mp4 [--tempo 143] [--start 0] [--duration 60] \\
        [--title "..."] [--subtitle "..."] [--gif out.gif --gif-seconds 12]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG, LINE, TEXT, BRIGHT = (15, 17, 21), (38, 42, 51), (139, 147, 161), (233, 235, 239)
RH, LH = (242, 166, 90), (90, 180, 224)
KEY, KEY_EDGE, KEY_BLACK = (243, 242, 238), (201, 199, 192), (28, 30, 35)
BLACK_PC = {1, 3, 6, 8, 10}
LEAD = 1.0  # seconds of silence before the first note


def soundfont() -> str:
    for p in (os.environ.get("M2S_SOUNDFONT"), "/usr/share/sounds/sf2/FluidR3_GM.sf2",
              "/usr/share/soundfonts/FluidR3_GM.sf2", "/usr/share/sounds/sf2/default-GM.sf2",
              "/usr/share/soundfonts/default.sf2"):
        if p and Path(p).exists():
            return p
    raise SystemExit("no General MIDI SoundFont found; install fluid-soundfont-gm or set M2S_SOUNDFONT")


def font(name: str, size: int) -> ImageFont.ImageFont:
    for d in ("/usr/share/fonts/truetype/noto", "/usr/share/fonts/noto", "/usr/share/fonts/truetype/dejavu"):
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def timeline(scores: Path, tempo: float) -> tuple[list[dict], list[tuple[float, float]], list[float]]:
    """Notes, pedal intervals and bar starts, in seconds of the as-written performance."""
    scored = json.loads((scores / "scored.json").read_text())
    info = json.loads((scores / "notation.json").read_text())
    spq = 60.0 / tempo
    notes = []
    for t in scored["tracks"]:
        if t.get("preset") == "drums":
            continue
        for n in t["notes"]:
            if not n.get("keep", True) or n.get("beat") is None:
                continue
            grace = n.get("value") == 0.0
            on = LEAD + n["beat"] * spq - (0.07 if grace else 0.0)
            d = 0.07 if grace else max(n["qdur"] * spq, 0.05)
            notes.append({"on": on, "off": on + d, "pitch": n["pitch"], "vel": n["velocity"],
                          "right": n.get("effector", {}).get("hand") == "R" or (n.get("staff") == 1)})
    notes.sort(key=lambda n: n["on"])
    # pedal: recording seconds -> notated quarters (through the bar lines) -> seconds as written
    bt = np.array(info["bar_times"])
    bq = 4.0 * np.arange(len(bt)) if not info.get("time_signature") else (
        np.arange(len(bt)) * _bar_quarters(info["time_signature"]))
    pedal = []
    for t in scored["tracks"]:
        for a, b in t.get("pedal", []):
            qa, qb = np.interp([a, b], bt, bq)
            if qb > qa:
                pedal.append((LEAD + qa * spq, LEAD + qb * spq))
    bars = [LEAD + q * spq for q in bq]
    return notes, sorted(pedal), bars


def _bar_quarters(ts: str) -> float:
    n, d = ts.split("/")
    return int(n) * 4.0 / int(d)


def render_audio(notes, pedal, wav: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, name="piano")
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(min(127, max(1, n["vel"]))), pitch=n["pitch"],
                                           start=n["on"], end=n["off"]))
    for a, b in pedal:
        inst.control_changes += [pretty_midi.ControlChange(64, 127, a), pretty_midi.ControlChange(64, 0, b)]
    pm.instruments.append(inst)
    mid = wav.with_suffix(".mid")
    pm.write(str(mid))
    subprocess.run(["fluidsynth", "-ni", "-g", "0.6", "-r", "44100", "-F", str(wav), soundfont(), str(mid)],
                   check=True, capture_output=True)


class Keyboard:
    def __init__(self, width: int):
        whites = [p for p in range(21, 109) if p % 12 not in BLACK_PC]
        ww = width / len(whites)
        self.keys = {}
        wi = 0
        for p in range(21, 109):
            if p % 12 in BLACK_PC:
                self.keys[p] = (wi * ww - ww * 0.31, ww * 0.62, True)
            else:
                self.keys[p] = (wi * ww, ww, False)
                wi += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="run folder (or any folder) with scored.json and notation.json")
    ap.add_argument("out")
    ap.add_argument("--tempo", type=float, help="quarters per minute (default: the score's)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, help="seconds (default: to the end)")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--title", default="music2sheets")
    ap.add_argument("--subtitle", default="the transcription, played as written")
    ap.add_argument("--gif", help="also write a silent looping GIF of the first --gif-seconds")
    ap.add_argument("--gif-seconds", type=float, default=12.0)
    a = ap.parse_args()
    scores = Path(a.scores)
    info = json.loads((scores / "notation.json").read_text())
    tempo = a.tempo or float(info["qpm"])
    notes, pedal, bars = timeline(scores, tempo)
    W, H = (int(x) for x in a.size.split("x"))
    total = max(n["off"] for n in notes) + 2.5
    t_start = a.start
    t_end = min(total, t_start + a.duration) if a.duration else total

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        render_audio(notes, pedal, wav)

        kb = Keyboard(W)
        top, KH = int(H * 0.11), int(H * 0.155)
        fall = H - KH
        window = 3.0
        f_title, f_sub, f_mono = font("NotoSerifDisplay-MediumItalic.ttf", int(H * 0.042)), \
            font("NotoSans-Regular.ttf", int(H * 0.024)), font("NotoSansMono-Medium.ttf", int(H * 0.024))
        f_bar = font("NotoSansMono-Regular.ttf", int(H * 0.017))
        starts = [n["on"] for n in notes]
        max_d = max(n["off"] - n["on"] for n in notes)
        frames = int((t_end - t_start) * a.fps)
        fade_out = max(0.0, t_end - t_start - 1.5)
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps),
             "-i", "-", "-ss", f"{t_start:.3f}", "-t", f"{t_end - t_start:.3f}", "-i", str(wav),
             "-af", f"afade=t=in:d=0.3,afade=t=out:st={fade_out:.3f}:d=1.5,loudnorm=I=-16:TP=-1.5",
             "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
        for f in range(frames):
            t = t_start + f / a.fps
            im = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(im)
            # bar lines and numbers
            k = max(0, bisect.bisect_left(bars, t - 0.2) - 1)
            while k < len(bars) and bars[k] < t + window:
                y = fall - (bars[k] - t) / window * (fall - top)
                if top <= y <= fall:
                    d.line([(0, y), (W, y)], fill=LINE, width=1)
                    d.text((10, y - int(H * 0.024)), str(k + 1), fill=TEXT, font=f_bar)
                k += 1
            # falling notes
            active = {}
            i = bisect.bisect_left(starts, t - max_d - 0.1)
            while i < len(notes) and notes[i]["on"] < t + window:
                n = notes[i]
                i += 1
                if n["off"] < t or n["pitch"] not in kb.keys:
                    continue
                x, w, _ = kb.keys[n["pitch"]]
                yb = min(fall, fall - (n["on"] - t) / window * (fall - top))
                yt = max(top, fall - (n["off"] - t) / window * (fall - top))
                if yb - yt >= 1:
                    d.rounded_rectangle([x + 1, yt, x + max(w - 1, 2), max(yb, yt + 3)], radius=3,
                                        fill=RH if n["right"] else LH)
                if n["on"] <= t:
                    active[n["pitch"]] = n["right"]
            # header over the top band
            d.rectangle([0, 0, W, top], fill=BG)
            d.text((int(W * 0.02), int(H * 0.018)), a.title, fill=BRIGHT, font=f_title)
            tw = d.textlength(a.title, font=f_title)
            d.text((int(W * 0.02) + tw + int(W * 0.015), int(H * 0.036)), a.subtitle, fill=TEXT, font=f_sub)
            bar_no = max(1, bisect.bisect_right(bars, t))
            label = f"bar {bar_no}"
            d.text((W - int(W * 0.02) - d.textlength(label, font=f_mono), int(H * 0.036)), label, fill=BRIGHT, font=f_mono)
            d.line([(0, top), (W, top)], fill=LINE, width=1)
            # keyboard
            d.rectangle([0, fall, W, H], fill=KEY_BLACK)
            for p, (x, w, black) in kb.keys.items():
                if not black:
                    c = KEY if p not in active else (RH if active[p] else LH)
                    d.rectangle([x + 0.5, fall + 3, x + w - 1, H - 2], fill=c)
            for p, (x, w, black) in kb.keys.items():
                if black:
                    c = KEY_BLACK if p not in active else (RH if active[p] else LH)
                    d.rectangle([x, fall + 3, x + w, fall + 3 + KH * 0.62], fill=c)
            d.rectangle([0, fall, W, fall + 2], fill=KEY_EDGE)
            # progress
            d.rectangle([0, H - 3, W * (t - t_start) / max(t_end - t_start, 1e-6), H], fill=RH)
            enc.stdin.write(im.tobytes())
        enc.stdin.close()
        if enc.wait() != 0:
            raise SystemExit("ffmpeg failed")
    if a.gif:
        gw = min(W, 720)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", f"{a.gif_seconds:.2f}", "-i", a.out, "-vf",
                        f"fps=15,scale={gw}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];"
                        "[s1][p]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", a.gif], check=True)
    print(f"{a.out}: {t_end - t_start:.1f} s at {tempo:.0f} qpm" + (f"; {a.gif}" if a.gif else ""))


if __name__ == "__main__":
    main()
