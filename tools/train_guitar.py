"""Fine-tune Basic Pitch for electric (and distorted) guitar.

Steps, each reproducible with fixed seeds:

    python tools/train_guitar.py download        # Guitar-TECHS, GuitarSet (Zenodo), EGDB (Google Drive) into data/guitar/
    python tools/train_guitar.py synth           # FluidSynth guitar parts (extra training data)
    python tools/train_guitar.py check           # the split model reproduces stock Basic Pitch
    python tools/train_guitar.py train           # fine-tune the head; writes models/guitar_bp.onnx
    python tools/eval_guitar.py                  # stock vs fine-tuned on held-out real recordings

Data and splits (never train on a test split):

* EGDB (Chen et al., ICASSP 2022; https://ss12f32v.github.io/Guitar-Transcription/): 240 clips of
  real guitar playing, DI re-amped through five amp models. Following the authors' split, clips
  1-215 (DI and the Marshall, Mesa and Fender Twin tones) are training data; clips 216-240 with the
  held-out JC-Jazz (clean) and Plexi (crunch) tones are test data, as are the five real-world
  recordings in RealData. The page states the data is publicly available but gives no license:
  weights trained on it are for research use only.
* Guitar-TECHS (Pedroza et al., ICASSP 2025; https://zenodo.org/records/14963133, CC BY 4.0): real
  amplifier and DI recordings by three players. Players 1-2 (scales, techniques, player 1's chords) are
  training data; player 3's musical excerpts are the held-out test set (new player, guitar, amp, room).

Training windows get amp-sim distortion, gain, noise and synthetic drum/bass bleed (what a Demucs
guitar stem still contains). Validation is split by performance, never by file.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("data/guitar")
EGDB = ROOT / "egdb"
TECHS = ROOT / "guitar_techs"
MODEL_OUT = Path("models/guitar_bp.onnx")
EGDB_FOLDER = "https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_"
TECHS_RECORD = "https://zenodo.org/api/records/14963133/files/{}/content"
EGDB_TRAIN_TONES = ("DI", "Marshall", "Mesa", "Ftwin")
EGDB_TEST_TONES = ("JCjazz", "Plexi")
EGDB_TEST_CLIPS = range(216, 241)
SEED = 1234


# ----------------------------------------------------------------------------- download
def _gdown_files(listing: list[dict], dest: Path) -> None:
    """Download Drive files one by one (gdown is run through uvx; it is not a project dependency)."""
    todo = [x for x in listing if not (dest / x["path"]).exists()]
    if not todo:
        return
    script = (
        "import json, sys, time, gdown\n"
        "from pathlib import Path\n"
        "items = json.load(open(sys.argv[1])); dest = Path(sys.argv[2])\n"
        "for x in items:\n"
        "    p = dest / x['path']; p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    for attempt in range(4):\n"
        "        try:\n"
        "            gdown.download(x['url'], str(p), quiet=True); break\n"
        "        except Exception as e:\n"
        "            print('retry', x['path'], e, file=sys.stderr); time.sleep(5 * (attempt + 1))\n"
    )
    tmp = dest / "_todo.json"
    tmp.write_text(json.dumps(todo))
    subprocess.run(["uvx", "--with", "gdown", "python", "-c", script, str(tmp), str(dest)], check=True)
    tmp.unlink()


def download_egdb(train: bool = True) -> None:
    EGDB.mkdir(parents=True, exist_ok=True)
    listing_p = EGDB / "listing.json"
    if not listing_p.exists():
        out = subprocess.run(["uvx", "gdown", "--folder", EGDB_FOLDER, "--json"], check=True, capture_output=True,
                             text=True).stdout
        listing_p.write_text(out)
    listing = json.loads(listing_p.read_text())

    def want(path: str) -> bool:
        top = path.split("/")[0]
        if path.endswith(".DS_Store"):
            return False
        if top in ("RealData", "audio_label") or path == "README.txt":
            return True
        if not top.startswith("audio_"):
            return False
        tone = top[len("audio_"):]
        clip = int(Path(path).stem)
        if tone in EGDB_TEST_TONES:
            return clip in EGDB_TEST_CLIPS
        return train and tone in EGDB_TRAIN_TONES and clip not in EGDB_TEST_CLIPS

    def priority(x: dict) -> int:  # test labels, real recordings, test audio, then training data
        top = x["path"].split("/")[0]
        tone = top[len("audio_"):] if top.startswith("audio_") else ""
        stem = Path(x["path"]).stem
        test_clip = stem.isdigit() and int(stem) in EGDB_TEST_CLIPS
        if top == "audio_label":
            return 0 if test_clip else 3
        if top == "RealData":
            return 1
        if tone in EGDB_TEST_TONES:
            return 2
        return 4 + (EGDB_TRAIN_TONES.index(tone) if tone in EGDB_TRAIN_TONES else 0)

    _gdown_files(sorted((x for x in listing if want(x["path"])), key=priority), EGDB)


def download_techs(train: bool = True) -> None:
    TECHS.mkdir(parents=True, exist_ok=True)
    names = ["P3_music.zip"] + (["P1_scales.zip", "P2_scales.zip", "P1_techniques.zip", "P2_techniques.zip",
                                 "P1_chords.zip", "P2_chords.zip"] if train else [])
    for name in names:
        z = TECHS / name
        if not z.exists():
            subprocess.run(["curl", "-sfL", "-o", str(z), TECHS_RECORD.format(name)], check=True)
        target = TECHS / z.stem
        if not target.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)


GUITARSET = ROOT / "guitarset"
GUITARSET_RECORD = "https://zenodo.org/api/records/3371780/files/{}/content"


def download_guitarset() -> None:
    """GuitarSet (Xi et al., ISMIR 2018; CC BY 4.0): acoustic guitar, a regression check only. Note that
    stock Basic Pitch was trained on it."""
    GUITARSET.mkdir(parents=True, exist_ok=True)
    for name in ("annotation.zip", "audio_mono-mic.zip"):
        z = GUITARSET / name
        if not z.exists():
            subprocess.run(["curl", "-sfL", "-o", str(z), GUITARSET_RECORD.format(name)], check=True)
        target = GUITARSET / z.stem
        if not target.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)


def guitarset_items(every: int = 6) -> list[tuple[Path, Path, str]]:
    """Every ``every``-th excerpt (mono mic), with a MIDI file written from its JAMS note_midi annotations."""
    import pretty_midi

    out = []
    jams = sorted((GUITARSET / "annotation").rglob("*.jams"))
    for j in jams[::every]:
        wavs = list((GUITARSET / "audio_mono-mic").rglob(f"{j.stem}_mic.wav"))
        if not wavs:
            continue
        mid = WORK / "guitarset_midi" / f"{j.stem}.mid"
        if not mid.exists():
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(25)
            for ann in json.loads(j.read_text())["annotations"]:
                if ann["namespace"] != "note_midi":
                    continue
                for d in ann["data"]:
                    inst.notes.append(pretty_midi.Note(80, int(round(d["value"])), d["time"], d["time"] + d["duration"]))
            pm.instruments.append(inst)
            mid.parent.mkdir(parents=True, exist_ok=True)
            pm.write(str(mid))
        out.append((wavs[0], mid, f"guitarset/{j.stem}"))
    return out


# ----------------------------------------------------------------------------- datasets
def midi_notes(path: Path) -> list[tuple[float, float, int]]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    return sorted((n.start, n.end, n.pitch) for i in pm.instruments for n in i.notes)


def label_shift(wav: Path, notes: list[tuple[float, float, int]]) -> float:
    """Seconds to add to a file's labels so they line up with the audio: the median distance from each
    labelled onset to the nearest spectral-flux onset (within 80 ms). Uses the audio alone, no model.
    Guitar-TECHS labels come out ~25 ms late against the DI pluck transients (DI and mic'd alike)."""
    import hashlib

    import librosa

    key = hashlib.sha1(f"{wav.resolve()}".encode()).hexdigest()[:16]
    cache = WORK / "shifts.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    if key in known:
        return known[key]
    y, sr = librosa.load(str(wav), sr=44100, mono=True)
    flux = librosa.onset.onset_detect(y=y, sr=sr, hop_length=128, units="time", backtrack=False)
    d = [flux[np.argmin(np.abs(flux - s))] - s for s, _, _ in notes
         if len(flux) and np.min(np.abs(flux - s)) < 0.08]
    shift = float(np.median(d)) if len(d) >= 10 else 0.0
    known[key] = shift
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known))
    return shift


def aligned_notes(wav: Path, mid: Path) -> list[tuple[float, float, int]]:
    notes = midi_notes(mid)
    sh = label_shift(wav, notes)
    return [(max(0.0, a + sh), max(0.0, b + sh), p) for a, b, p in notes]


def egdb_items(split: str) -> list[tuple[Path, Path, str]]:
    """(audio, midi, tag) for 'train' or 'test' (held-out clips x held-out tones) or 'real'."""
    if split == "real":
        return [(p, EGDB / "RealData" / "Label" / (p.stem + ".mid"), f"egdb-real/{p.stem}")
                for p in sorted((EGDB / "RealData" / "Audio").glob("clip*.wav"))]
    tones = EGDB_TRAIN_TONES if split == "train" else EGDB_TEST_TONES
    out = []
    for tone in tones:
        for wav in sorted((EGDB / f"audio_{tone}").glob("*.wav"), key=lambda p: int(p.stem)):
            clip = int(wav.stem)
            if (clip in EGDB_TEST_CLIPS) == (split == "test"):
                lab = EGDB / "audio_label" / f"{clip}.midi"
                if lab.exists():
                    out.append((wav, lab, f"egdb-{tone}/{clip}"))
    return out


def techs_items(split: str, sources: tuple[str, ...] | None = None) -> list[tuple[Path, Path, str]]:
    """Guitar-TECHS pairs <set>/<set>/audio/<source>/<source>_<id>.wav <-> midi/midi_<id>.mid.

    'test' is player 3's music excerpts through the mic'd amplifier (unseen player, guitar, amp and
    room); 'train' is players 1-2 (scales, techniques, chords), DI and mic'd amp."""
    if split == "test":
        sets, sources = ["P3_music"], sources or ("micamp",)
    else:
        sets = ["P1_scales", "P2_scales", "P1_techniques", "P2_techniques", "P1_chords", "P2_chords"]
        sources = sources or ("micamp", "directinput")
    out = []
    for name in sets:
        base = TECHS / name / name
        for src in sources:
            for wav in sorted((base / "audio" / src).glob(f"{src}_*.wav")):
                mid = base / "midi" / f"midi_{wav.stem[len(src) + 1:]}.mid"
                if mid.exists():
                    out.append((wav, mid, f"techs-{name}-{src}/{wav.stem}"))
    return out


# ----------------------------------------------------------------------------- synthetic parts
SYNTH = ROOT / "synth"
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
# GM steel, jazz, clean, muted, overdriven (0-based). Distortion guitar (30) is left out: the synthetic
# band test (tools/make_band_test.py) uses it, and the band test must stay unseen.
SYNTH_PROGRAMS = (25, 26, 27, 28, 29)
OPEN_STRINGS = (40, 45, 50, 55, 59, 64)
SHAPES = {  # fret offsets per string (None = not played), root-position movable shapes
    "E_maj": (0, 2, 2, 1, 0, 0), "E_min": (0, 2, 2, 0, 0, 0), "E_7": (0, 2, 0, 1, 0, 0),
    "E_m7": (0, 2, 0, 0, 0, 0), "A_maj": (None, 0, 2, 2, 2, 0), "A_min": (None, 0, 2, 2, 1, 0),
    "A_7": (None, 0, 2, 0, 2, 0), "A_maj7": (None, 0, 2, 1, 2, 0), "D_maj": (None, None, 0, 2, 3, 2),
    "D_min": (None, None, 0, 2, 3, 1), "power5": (0, 2, 2, None, None, None),
    "power5A": (None, 0, 2, 2, None, None),
}


def _voicing(rng: np.random.Generator) -> list[int]:
    name = list(SHAPES)[rng.integers(len(SHAPES))]
    shape = SHAPES[name]
    fret = int(rng.integers(0, 10))
    return [OPEN_STRINGS[i] + f + fret for i, f in enumerate(shape) if f is not None]


def synth_part(rng: np.random.Generator, seconds: float):
    """An idiomatic random guitar part: strummed chords, palm-muted power chords, arpeggios, riffs."""
    import pretty_midi

    qpm = float(rng.uniform(70, 170))
    beat = 60 / qpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=qpm)
    inst = pretty_midi.Instrument(int(rng.choice(SYNTH_PROGRAMS)))
    t = 0.2
    while t < seconds:
        chord = _voicing(rng)
        style = rng.choice(["strum", "chug", "arp", "riff", "dyads"])
        bar = 4 * beat
        if style == "strum":
            for k in range(int(rng.choice([2, 4, 8]))):
                on = t + k * (bar / 8)
                if on >= t + bar or rng.random() < 0.25:
                    continue
                down = k % 2 == 0
                order = chord if down else chord[::-1]
                for j, p in enumerate(order):
                    s = on + j * float(rng.uniform(0.004, 0.015))
                    inst.notes.append(pretty_midi.Note(int(rng.integers(60, 115)), p, s, on + bar / 8 * rng.uniform(0.6, 1.0)))
        elif style == "chug":
            pc = chord[:3]
            for k in range(8):
                on = t + k * bar / 8
                dur = bar / 8 * (0.5 if rng.random() < 0.6 else 0.95)
                for p in pc[:2 if rng.random() < 0.5 else 3]:
                    inst.notes.append(pretty_midi.Note(int(rng.integers(70, 120)), p, on, on + dur))
        elif style == "arp":
            seq = chord + chord[-2:0:-1]
            step = bar / rng.choice([8, 12, 16])
            k = 0
            while k * step < bar:
                p = seq[k % len(seq)]
                inst.notes.append(pretty_midi.Note(int(rng.integers(55, 110)), p, t + k * step,
                                                   t + k * step + step * rng.uniform(1.0, 3.0)))
                k += 1
        elif style == "riff":
            root = chord[0] + 12 * int(rng.integers(0, 2))
            scale = [0, 3, 5, 7, 10, 12, 15]
            step = bar / rng.choice([8, 16])
            k = 0
            while k * step < bar:
                if rng.random() > 0.15:
                    p = root + scale[rng.integers(len(scale))]
                    inst.notes.append(pretty_midi.Note(int(rng.integers(60, 120)), int(p), t + k * step,
                                                       t + k * step + step * rng.uniform(0.5, 1.0)))
                k += 1
        else:  # dyads: thirds/sixths/fourths on adjacent strings
            step = bar / rng.choice([4, 8])
            k = 0
            while k * step < bar:
                lo = int(rng.integers(50, 72))
                for p in (lo, lo + int(rng.choice([3, 4, 5, 8, 9]))):
                    inst.notes.append(pretty_midi.Note(int(rng.integers(60, 110)), p, t + k * step,
                                                       t + (k + 0.9) * step))
                k += 1
        t += bar
    inst.notes = [n for n in inst.notes if n.start < seconds]
    pm.instruments.append(inst)
    return pm


def make_synth(n: int = 160, seconds: float = 30.0) -> None:
    """Render ``n`` synthetic parts with FluidSynth into data/guitar/synth (fixed seed)."""
    SYNTH.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 7)
    for i in range(n):
        mid, wav = SYNTH / f"synth_{i:03d}.mid", SYNTH / f"synth_{i:03d}.wav"
        pm = synth_part(rng, seconds)
        if wav.exists():
            continue
        pm.write(str(mid))
        subprocess.run(["fluidsynth", "-ni", "-g", "0.6", "-r", "44100", "-F", str(wav), SF2, str(mid)], check=True,
                       capture_output=True)


def synth_items() -> list[tuple[Path, Path, str]]:
    return [(w, w.with_suffix(".mid"), f"synth-DI/{w.stem}") for w in sorted(SYNTH.glob("synth_*.wav"))
            if w.with_suffix(".mid").exists()]


# ----------------------------------------------------------------------------- augmentation
def amp_sim(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A cheap, randomized amp: pre-EQ, tanh/asymmetric clipping, cabinet low-pass."""
    from scipy.signal import butter, lfilter

    b, a = butter(1, rng.uniform(300, 900) / (sr / 2), "highpass")
    x = lfilter(b, a, y)
    drive = 10 ** rng.uniform(0.5, 2.0)
    x = np.tanh(drive * x + rng.uniform(0, 0.2) * drive * x ** 2)
    b, a = butter(2, rng.uniform(3500, 6500) / (sr / 2), "lowpass")
    x = lfilter(b, a, x)
    return (x / (np.abs(x).max() + 1e-9) * 0.7).astype(np.float32)


def synth_bleed(sr: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """What leaks into a separated guitar stem: drum hits and a low bass line, both unlabelled."""
    t = np.arange(n) / sr
    y = np.zeros(n, np.float32)
    step = 60 / rng.uniform(80, 170) / 2  # eighth notes
    for k in range(int(n / sr / step) + 1):
        i = int(k * step * sr)
        if i >= n:
            break
        m = n - i
        tt = t[:m]
        if k % 4 == 0 or rng.random() < 0.15:  # kick: falling sine
            f = 50 + 90 * np.exp(-tt * 40)
            y[i:] += 0.8 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 12)
        if k % 4 == 2:  # snare: noise and a 180 Hz body
            y[i:] += (0.5 * rng.normal(0, 1, m) + 0.4 * np.sin(2 * np.pi * 180 * tt)) * np.exp(-tt * 25)
        if rng.random() < 0.7:  # hat: high-passed noise
            h = rng.normal(0, 1, m)
            y[i:] += 0.25 * (h - np.concatenate([[0.0], h[:-1]])) * np.exp(-tt * 60)
    bass = np.zeros(n, np.float32)
    pos = 0
    while pos < n:  # bass: low-passed saw on MIDI 28-40
        dur = int(step * sr * rng.integers(1, 5))
        f0 = 440 * 2 ** ((rng.integers(28, 41) - 69) / 12)
        seg = np.arange(min(dur, n - pos)) / sr
        bass[pos:pos + len(seg)] = (2 * ((f0 * seg) % 1) - 1) * np.exp(-seg * 3)
        pos += dur
    from scipy.signal import butter, lfilter

    b, a = butter(2, 400 / (sr / 2), "lowpass")
    y = y + 0.6 * lfilter(b, a, bass)
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def augment(y: np.ndarray, sr: int, rng: np.random.Generator, di: bool) -> np.ndarray:
    if rng.random() < (0.6 if di else 0.25):
        y = amp_sim(y, sr, rng)
    y = y / (np.abs(y).max() + 1e-9) * 10 ** rng.uniform(-1.0, -0.1)
    if rng.random() < 0.4:  # Demucs stems keep some drums and bass
        y = y + synth_bleed(sr, len(y), rng) * np.abs(y).max() * 10 ** rng.uniform(-1.6, -0.6)
    if rng.random() < 0.5:
        y = y + rng.normal(0, 10 ** rng.uniform(-4, -2.5), size=len(y)).astype(np.float32)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- model
# Basic Pitch = a fixed front end (audio -> harmonic-stacked CQT) + a small convolutional head
# (contour, note and onset branches). Only the head is fine-tuned: the front end runs frozen in
# onnxruntime, the head is converted to PyTorch, and the tuned weights are written back into the
# original ONNX graph, so the worker keeps using basic_pitch's predict() unchanged.
OUTPUTS = ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:0"]  # note, onset, contour
WORK = ROOT / "work"


def bp_onnx_path() -> Path:
    from basic_pitch import ICASSP_2022_MODEL_PATH

    p = Path(str(ICASSP_2022_MODEL_PATH))
    return p if p.suffix == ".onnx" else p.parent / "nmp.onnx"


def feature_tensor(model) -> str:
    """Name of the harmonic-stacked CQT: the input of the first head convolution."""
    inits = {i.name for i in model.graph.initializer}
    first_conv = next(n for n in model.graph.node if n.op_type == "Conv" and "cq_t2010v2" not in n.name)
    src = next(x for x in first_conv.input if x not in inits)
    producer = next(n for n in model.graph.node if src in n.output)
    # the head starts at the transpose into NCHW; cut before it
    return producer.input[0] if producer.op_type == "Transpose" else src


def split_model() -> tuple[Path, Path, str]:
    import onnx
    import onnx.utils

    WORK.mkdir(parents=True, exist_ok=True)
    full = bp_onnx_path()
    feat = feature_tensor(onnx.load(str(full)))
    front, head = WORK / "bp_front.onnx", WORK / "bp_head.onnx"
    if not front.exists():
        onnx.utils.extract_model(str(full), str(front), ["serving_default_input_2:0"], [feat])
    if not head.exists():
        onnx.utils.extract_model(str(full), str(head), [feat], OUTPUTS)
    return front, head, feat


def _stub_torchvision() -> None:
    """onnx2torch imports torchvision for converters Basic Pitch never uses (NMS, RoI). The torchvision
    wheel installed here doesn't match the CUDA torch build, so give onnx2torch a stub (this process only)."""
    import types

    try:
        import torchvision  # noqa: F401
    except Exception:  # noqa: BLE001
        for k in [k for k in sys.modules if k == "torchvision" or k.startswith("torchvision.")]:
            del sys.modules[k]
        pkg = types.ModuleType("torchvision")
        pkg.__path__ = []
        ops = types.ModuleType("torchvision.ops")
        for name in ("nms", "batched_nms", "roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool"):
            setattr(ops, name, None)
        pkg.ops = ops
        sys.modules["torchvision"] = pkg
        sys.modules["torchvision.ops"] = ops


def load_head():
    """(front-end onnxruntime session, head as a torch module, {param name: initializer name})."""
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    _stub_torchvision()
    from onnx2torch import convert

    front, head, _ = split_model()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(front), providers=providers)
    module = convert(onnx.load(str(head)))
    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(bp_onnx_path())).graph.initializer}
    mapping = {}
    for name, param in module.named_parameters():
        v = param.detach().cpu().numpy()
        hits = [k for k, a in inits.items() if a.shape == v.shape and np.allclose(a, v)]
        if len(hits) != 1:
            raise RuntimeError(f"cannot map {name} back to one initializer: {hits}")
        mapping[name] = hits[0]
    return sess, module, mapping


def front_features(sess, x: np.ndarray) -> np.ndarray:
    """Audio windows (B, 43844, 1) -> head input features."""
    return sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]


def check_split() -> float:
    """Front + head must reproduce the full model; returns the max absolute difference."""
    import onnxruntime as ort
    import torch

    sess, module, _ = load_head()
    full = ort.InferenceSession(str(bp_onnx_path()), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(0, 0.1, (2, 43844, 1)).astype(np.float32)
    ref = full.run(OUTPUTS, {full.get_inputs()[0].name: x})
    with torch.no_grad():
        out = module(torch.from_numpy(front_features(sess, x)))
    out = out if isinstance(out, (list, tuple)) else [out]
    return max(float(np.abs(o.numpy() - r).max()) for o, r in zip(out, ref))


def targets(notes: list[tuple[float, float, int]], n_frames: int, fps: float) -> tuple[np.ndarray, ...]:
    """Basic Pitch targets: note (frames x 88), onset (frames x 88), contour (frames x 264)."""
    note = np.zeros((n_frames, 88), np.float32)
    onset = np.zeros((n_frames, 88), np.float32)
    contour = np.zeros((n_frames, 264), np.float32)
    for s, e, p in notes:
        k = p - 21
        if not 0 <= k < 88:
            continue
        a = int(round(s * fps))
        b = max(int(round(e * fps)), a + 1)
        if a >= n_frames or b <= 0:
            continue
        lo, hi = max(0, a), min(n_frames, b)
        note[lo:hi, k] = 1
        contour[lo:hi, 3 * k + 1] = 1
        if 0 <= a < n_frames:
            onset[a, k] = 1
    return note, onset, contour


def load_audio(wav: Path, sr: int) -> np.ndarray:
    """Mono audio at Basic Pitch's rate, cached as float16 under data/guitar/work/cache."""
    import hashlib

    import librosa

    key = hashlib.sha1(f"{wav.resolve()}:{sr}".encode()).hexdigest()[:16]
    cache = WORK / "cache" / f"{key}.npy"
    if cache.exists():
        return np.load(cache).astype(np.float32)
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, y.astype(np.float16))
    return y.astype(np.float32)


def performance(tag: str) -> str:
    """Group key: the same EGDB clip under different amp tones (or a Guitar-TECHS take recorded DI
    and mic'd) is one performance and must not straddle the train/validation split."""
    if tag.startswith("synth-"):
        return tag
    if tag.startswith("egdb-"):
        return "egdb/" + tag.split("/")[1]
    name, stem = tag.split("/")
    return name.rsplit("-", 1)[0] + "/" + stem.split("_", 1)[1]


def split_items(val_player: str | None = None) -> tuple[list, list]:
    """(train, validation) files. Default validation = every 8th performance in a fixed hash order (so
    the split doesn't move when more data finishes downloading). ``val_player="P2"`` instead holds out
    all of that Guitar-TECHS player: a cross-player validation for choosing the recipe."""
    import hashlib

    items = egdb_items("train") + techs_items("train")

    def is_val(tag: str) -> bool:
        if val_player:
            return tag.startswith(f"techs-{val_player}_")
        return int(hashlib.sha1(performance(tag).encode()).hexdigest(), 16) % 8 == 0

    return [x for x in items if not is_val(x[2])] + synth_items(), [x for x in items if is_val(x[2])]


class Windows:
    """Random 2 s training windows (Basic Pitch's input length) with augmentation and targets."""

    def __init__(self, items, sr: int, n_samples: int, n_frames: int, fps: float, rng: np.random.Generator):
        self.sr, self.n, self.nf, self.fps, self.rng = sr, n_samples, n_frames, fps, rng
        self.data = []
        for wav, mid, tag in items:
            y = load_audio(wav, sr)
            notes = aligned_notes(wav, mid)
            if notes and len(y) > n_samples:
                self.data.append((y, notes, "DI/" in tag or "directinput" in tag))  # DI and synth get more amp-sim
        # sample files in proportion to their length
        w = np.array([len(y) for y, _, _ in self.data], float)
        self.p = w / w.sum()

    def sample(self, k: int, augment_audio: bool = True):
        xs, ns, os_, cs = [], [], [], []
        while len(xs) < k:
            y, notes, di = self.data[self.rng.choice(len(self.data), p=self.p)]
            s = int(self.rng.integers(0, len(y) - self.n))
            seg = y[s:s + self.n]
            if np.abs(seg).max() < 1e-4:
                continue
            seg = augment(seg, self.sr, self.rng, di) if augment_audio else seg / np.abs(seg).max() * 0.5
            t0 = s / self.sr
            local = [(a - t0, b - t0, p) for a, b, p in notes if b > t0 and a < t0 + self.n / self.sr]
            nt, on, co = targets(local, self.nf, self.fps)
            xs.append(seg[:, None])
            ns.append(nt)
            os_.append(on)
            cs.append(co)
        return np.stack(xs), np.stack(ns), np.stack(os_), np.stack(cs)


class ValF1:
    """Note onset F1 (50 ms) on the validation files through the worker's own note path: Basic Pitch
    note creation with the worker's thresholds, then m2s.transcribe.band.notes_from_output, keeping
    candidates with confidence >= 0.5. Front-end features are computed once."""

    def __init__(self, sess, items, max_windows: int = 60):
        import librosa
        from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
        from basic_pitch.inference import get_audio_input

        from m2s.presets import load_preset

        lo, hi = load_preset("electric_guitar").get("range", [21, 108])
        self.fmin, self.fmax = float(librosa.midi_to_hz(lo - 1)), float(librosa.midi_to_hz(hi + 1))
        overlap = 30 * FFT_HOP
        self.files = []
        for wav, mid, tag in items:
            if "directinput" in tag or "DI/" in tag:  # score the amplified signal
                continue
            wins, length = [], 0
            for w, _, n in get_audio_input(str(wav), overlap, AUDIO_N_SAMPLES - overlap):
                wins.append(w[0])
                length = n
                if len(wins) >= max_windows:
                    length = min(n, max_windows * (AUDIO_N_SAMPLES - overlap))
                    break
            feats = np.concatenate([front_features(sess, np.stack(wins[i:i + 16])) for i in range(0, len(wins), 16)])
            end = length / 22050
            ref = [x for x in aligned_notes(wav, mid) if x[0] < end - 0.1]
            self.files.append((feats, length, ref))

    def __call__(self, head, dev) -> float:
        import torch
        from basic_pitch import note_creation as infer
        from basic_pitch.inference import unwrap_output

        from m2s.transcribe.band import notes_from_output

        sys.path.insert(0, str(Path(__file__).parent))
        from eval_guitar import score

        m = ne = nr = 0
        head.eval()
        for feats, length, ref in self.files:
            outs = {"note": [], "onset": [], "contour": []}
            with torch.no_grad():
                for i in range(0, len(feats), 16):
                    o = head(torch.from_numpy(feats[i:i + 16]).to(dev))
                    for k, v in zip(("note", "onset", "contour"), o):
                        outs[k].append(v.cpu().numpy())
            out = {k: unwrap_output(np.concatenate(v), length, 30) for k, v in outs.items()}
            _, events = infer.model_output_to_notes(out, onset_thresh=0.35, frame_thresh=0.25,
                                                    min_note_len=int(np.round(50 / 1000 * (22050 / 256))),
                                                    min_freq=self.fmin, max_freq=self.fmax,
                                                    multiple_pitch_bends=False, melodia_trick=True)
            est = [(n.onset, n.offset, n.pitch) for n in notes_from_output(out, events) if n.confidence >= 0.5]
            a, b, c = score(ref, est, 0.05)
            m, ne, nr = m + a, ne + b, nr + c
        head.train()
        p, r = m / max(ne, 1), m / max(nr, 1)
        return 2 * p * r / max(p + r, 1e-9)


def train(args: argparse.Namespace) -> None:
    import time

    import torch
    from basic_pitch.constants import ANNOTATIONS_FPS, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, val = split_items(args.val_player)
    if not tr:
        sys.exit("no training data: run `python tools/train_guitar.py download`")
    print(f"train files {len(tr)}, validation files {len(val)}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sess, head, mapping = load_head()
    head = head.to(dev).train()
    x0 = np.zeros((1, AUDIO_N_SAMPLES, 1), np.float32)
    with torch.no_grad():
        n_frames = head(torch.from_numpy(front_features(sess, x0)).to(dev))[0].shape[1]
    dtr = Windows(tr, AUDIO_SAMPLE_RATE, AUDIO_N_SAMPLES, n_frames, ANNOTATIONS_FPS, rng)
    dva = Windows(val, AUDIO_SAMPLE_RATE, AUDIO_N_SAMPLES, n_frames, ANNOTATIONS_FPS, np.random.default_rng(SEED + 1))
    val_batches = [dva.sample(args.batch, augment_audio=False) for _ in range(8)]
    val_feats = [(front_features(sess, b[0]), b[1:]) for b in val_batches]
    import copy

    stock = copy.deepcopy(head).eval()
    for q in stock.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    bce = torch.nn.functional.binary_cross_entropy

    def smooth(y, ls: float = 0.2):  # Keras label smoothing, as Basic Pitch was trained
        return y * (1 - ls) + 0.5 * ls

    def loss_of(feats, tg):
        f = torch.from_numpy(feats).to(dev)
        note_t, onset_t, contour_t = (torch.from_numpy(a).to(dev) for a in tg)
        outs = [o.clamp(1e-6, 1 - 1e-6) for o in head(f)]
        note, onset, contour = outs
        # plain BCE on all three outputs: the released model's activations (negatives ~0.10, onset
        # peaks ~0.3) match it, and the worker's fixed thresholds assume that calibration; a
        # class-balanced onset loss lifts onset negatives toward the 0.35 threshold
        loss = bce(note, smooth(note_t)) + bce(onset, smooth(onset_t)) + bce(contour, smooth(contour_t))
        if args.distill > 0:  # stay near the stock model's outputs (what it knows beyond these players)
            with torch.no_grad():
                ref = stock(f)
            loss = loss + args.distill * sum(bce(o, r.clamp(1e-6, 1 - 1e-6)) - bce(r.clamp(1e-6, 1 - 1e-6), r.clamp(1e-6, 1 - 1e-6))
                                             for o, r in zip(outs, ref))
        return loss

    def val_loss() -> float:
        head.eval()
        with torch.no_grad():
            v = float(np.mean([float(loss_of(f, tg)) for f, tg in val_feats]))
        head.train()
        return v

    vf1 = ValF1(sess, val)
    best, bad = vf1(head, dev), 0
    print(f"step 0 val loss {val_loss():.4f} val F1 {best:.4f} (stock weights)", flush=True)
    torch.save(head.state_dict(), WORK / "head_best.pt")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = dtr.sample(args.batch)
        loss = loss_of(front_features(sess, batch[0]), batch[1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            v, f1 = val_loss(), vf1(head, dev)
            print(f"step {step} train {loss.item():.4f} val loss {v:.4f} val F1 {f1:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            if args.fixed_steps:
                continue
            if f1 > best + 1e-3:
                best, bad = f1, 0
                torch.save(head.state_dict(), WORK / "head_best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stop", flush=True)
                    break
    if args.fixed_steps:
        torch.save(head.state_dict(), WORK / "head_best.pt")
        best = vf1(head, dev)
    print(f"best val F1 {best:.4f}; {time.time() - t0:.0f}s")
    export(args)


def export(args: argparse.Namespace) -> None:
    """Write the tuned head weights into a copy of the full Basic Pitch graph."""
    import onnx
    import torch
    from onnx import numpy_helper

    _, head, mapping = load_head()
    head.load_state_dict(torch.load(WORK / "head_best.pt", map_location="cpu"))
    full = onnx.load(str(bp_onnx_path()))
    params = dict(head.named_parameters())
    by_init = {v: k for k, v in mapping.items()}
    for init in full.graph.initializer:
        if init.name in by_init:
            arr = params[by_init[init.name]].detach().cpu().numpy().astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(full, str(MODEL_OUT))
    print("wrote", MODEL_OUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--test-only", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--val-player", default=None, help="hold out a Guitar-TECHS player (P1/P2) for validation")
    t.add_argument("--distill", type=float, default=0.0, help="weight of the pull toward stock outputs")
    t.add_argument("--fixed-steps", action="store_true", help="train exactly --steps, keep the last weights")
    sub.add_parser("export")
    sub.add_parser("check")
    sy = sub.add_parser("synth")
    sy.add_argument("-n", type=int, default=160)
    a = ap.parse_args()
    if a.cmd == "download":
        download_techs(train=not a.test_only)
        download_guitarset()
        try:
            download_egdb(train=not a.test_only)
        except subprocess.CalledProcessError as e:  # Google Drive rate-limits shared folders
            print(f"EGDB download incomplete ({e}); rerun later to resume", file=sys.stderr)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "synth":
        make_synth(a.n)
    elif a.cmd == "check":
        print(f"front + head vs full model: max abs diff {check_split():.2e}")
    else:
        export(a)


if __name__ == "__main__":
    sys.exit(main())
