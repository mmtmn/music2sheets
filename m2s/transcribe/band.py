"""Stage 3 for band stems: polyphonic pitched (Basic Pitch), monophonic (torchcrepe), drums (ADTOF).

Every function returns *candidates* with confidences; the decoder keeps or drops them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from m2s import audio as A
from m2s.ir import Note, Track, assign_ids
from m2s.presets import load_preset

MONO = {"voice", "sax_alto", "sax_tenor", "bass"}


def track_name(name: str) -> tuple[str, str]:
    """'electric_guitar@L' -> ('electric_guitar_L', 'electric_guitar')."""
    if "@" in name:
        preset, side = name.split("@")
        return f"{preset}_{side}", preset
    return name, name


def transcribe_stem(wav: str | Path, name: str, profile: dict[str, Any], act_out: str | Path | None = None) -> Track:
    tname, preset = track_name(name)
    fam = load_preset(preset)["family"]
    if preset == "piano":
        from m2s.transcribe.piano import transcribe_piano

        tr, kong = transcribe_piano(wav, profile.get("tuning_cents", 0.0), name=tname)
        if act_out is not None:
            np.save(act_out, kong["reg_onset_output"].astype(np.float16))
        return tr
    if fam == "drums":
        notes = drums(wav)
    elif preset in MONO:
        notes = monophonic(wav, preset)
    else:
        notes = polyphonic(wav, preset, act_out)
    prefix = {"drums": "drm", "voice": "vox"}.get(preset, preset[:3])
    assign_ids(notes, f"{prefix}{tname[-1] if '@' in name else ''}")
    return Track(name=tname, preset=preset, notes=notes, stem=str(wav))


# --------------------------------------------------------------------- polyphonic
def polyphonic(wav: str | Path, preset: str, act_out: str | Path | None = None) -> list[Note]:
    from basic_pitch.inference import predict

    from m2s.verify import _bp_model

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    p = load_preset(preset)
    lo, hi = p.get("range", [21, 108])
    import librosa

    model_out, _, events = predict(str(wav), _bp_model(Model, ICASSP_2022_MODEL_PATH), onset_threshold=0.35,
                                   frame_threshold=0.25, minimum_note_length=50,
                                   minimum_frequency=float(librosa.midi_to_hz(lo - 1)),
                                   maximum_frequency=float(librosa.midi_to_hz(hi + 1)),
                                   multiple_pitch_bends=False, melodia_trick=True)
    if act_out is not None:
        np.save(act_out, model_out["note"].astype(np.float16))
    onset_act = model_out["onset"]
    fps = 22050 / 256
    notes = []
    for start, end, pitch, amp, bends in events:
        bend = []
        if bends is not None and len(bends):
            # Basic Pitch bends are in 1/3-semitone steps at its frame rate (~86 fps)
            step = (end - start) / max(len(bends), 1)
            bend = [(round(start + i * step, 3), round(b * 100 / 3, 1)) for i, b in enumerate(bends)
                    if i % 4 == 0]
        f = int(round(start * fps))
        col = int(pitch) - 21
        on_peak = float(onset_act[max(0, f - 3): f + 4, col].max()) if 0 <= col < onset_act.shape[1] else 0.0
        conf = float(np.clip(0.15 + 0.45 * amp + 0.45 * on_peak, 0.0, 0.97))
        notes.append(Note(id="", onset=round(float(start), 4), offset=round(float(end), 4), pitch=int(pitch),
                          velocity=int(np.clip(30 + 100 * amp, 1, 127)), confidence=round(conf, 3),
                          support={"basic_pitch": round(float(amp), 3), "onset": round(on_peak, 3)}, bend=bend,
                          provenance="basic_pitch"))
    suppress_harmonics(notes)
    return merge_retriggers(notes)


HARMONICS = (12, 19, 24)


def suppress_harmonics(notes: list[Note], window: float = 0.04, ratio: float = 0.85) -> None:
    """Lower the confidence of a candidate that sits an octave, twelfth or double octave above a
    louder note starting at the same time: on distorted or bright stems the model reports
    overtones as notes."""
    order = sorted(notes, key=lambda n: n.onset)
    for i, n in enumerate(order):
        amp = n.support.get("basic_pitch", 0.0)
        for m in order[max(0, i - 12): i + 12]:
            if m is n or abs(m.onset - n.onset) > window:
                continue
            if n.pitch - m.pitch in HARMONICS and amp < ratio * m.support.get("basic_pitch", 0.0) + 1e-9:
                n.confidence = round(n.confidence * 0.4, 3)
                n.tags.append("harmonic?")
                break


def merge_retriggers(notes: list[Note], gap: float = 0.04, min_onset: float = 0.35) -> list[Note]:
    """Join same-pitch fragments when the later one has no onset of its own."""
    by_pitch: dict[int, list[Note]] = {}
    for n in sorted(notes, key=lambda n: n.onset):
        by_pitch.setdefault(n.pitch, []).append(n)
    out = []
    for seq in by_pitch.values():
        cur = seq[0]
        for n in seq[1:]:
            if n.onset - cur.offset <= gap and n.support.get("onset", 1.0) < min_onset:
                cur.offset = max(cur.offset, n.offset)
            else:
                out.append(cur)
                cur = n
        out.append(cur)
    return sorted(out, key=lambda n: (n.onset, n.pitch))


# --------------------------------------------------------------------- monophonic
def monophonic(wav: str | Path, preset: str, hop: float = 0.01) -> list[Note]:
    import torch
    import torchcrepe

    p = load_preset(preset)
    lo, hi = p.get("range", [40, 84])
    y, sr = A.load(wav, sr=16000, mono=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    audio = torch.from_numpy(y)[None]
    f0, per = torchcrepe.predict(audio, 16000, hop_length=int(16000 * hop), fmin=float(_hz(lo - 2)),
                                 fmax=float(_hz(hi + 2)), model="full", decoder=torchcrepe.decode.viterbi,
                                 return_periodicity=True, batch_size=1024, device=dev)
    f0 = f0[0].cpu().numpy()
    per = torchcrepe.filter.median(per, 5)[0].cpu().numpy()
    rms = np.sqrt(np.convolve(y ** 2, np.ones(160) / 160, mode="same"))[:: int(16000 * hop)][: len(f0)]
    midi = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440)
    voiced = (per > 0.45) & (rms[: len(per)] > 0.01 * (rms.max() + 1e-9))
    import librosa

    # repeated notes at one pitch never change f0; the amplitude onsets separate them
    onsets = librosa.onset.onset_detect(y=y, sr=16000, hop_length=int(16000 * hop), units="frames",
                                        backtrack=True)
    # plucked instruments only: on legato wind and voice, vibrato and breath trip the onset detector
    plucked = load_preset(preset)["family"] == "fretted"
    return segment_notes(midi, voiced, per, hop, onsets=set(int(o) for o in onsets) if plucked else None)


def _hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def segment_notes(midi: np.ndarray, voiced: np.ndarray, per: np.ndarray, hop: float,
                  min_len: float = 0.06, jump: float = 0.7, onsets: set[int] | None = None) -> list[Note]:
    """Cut a pitch track into notes at unvoiced gaps, at stable pitch changes, and at amplitude
    onsets (repeated notes on one pitch)."""
    onsets = onsets or set()
    min_frames = int(min_len / hop)
    notes: list[Note] = []
    n = len(midi)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        ref = midi[i]
        while j + 1 < n and voiced[j + 1]:
            # a new note starts when pitch moves > `jump` semitones and stays there for 3 frames
            if abs(midi[j + 1] - ref) > jump and all(abs(midi[k] - midi[j + 1]) < 0.5 for k in range(j + 1, min(j + 4, n))):
                break
            if (j + 1) in onsets and (j + 1 - i) >= min_frames:
                break
            j += 1
            ref = 0.9 * ref + 0.1 * midi[j]
        seg = midi[i:j + 1]
        if (j - i + 1) * hop >= min_len:
            pitch = int(round(float(np.median(seg))))
            cents = (seg - pitch) * 100
            bend = [(round((i + k) * hop, 3), round(float(c), 1)) for k, c in enumerate(cents) if k % 5 == 0]
            conf = float(np.clip(np.mean(per[i:j + 1]), 0, 1))
            notes.append(Note(id="", onset=round(i * hop, 4), offset=round((j + 1) * hop, 4), pitch=pitch,
                              velocity=80, confidence=round(conf, 3), support={"crepe": round(conf, 3)},
                              bend=bend if np.ptp(cents) > 40 else [], provenance="torchcrepe"))
        i = j + 1
    return notes


# --------------------------------------------------------------------- drums
# ADTOF classes -> GM drum notes (kick, snare, tom, hi-hat, cymbal)
ADTOF_GM = {35: 36, 36: 36, 38: 38, 47: 47, 42: 42, 49: 49}


def drums(wav: str | Path) -> list[Note]:
    try:
        return _drums_adtof(wav)
    except ImportError:
        return _drums_bands(wav)


def _drums_adtof(wav: str | Path) -> list[Note]:
    import tempfile

    import pretty_midi
    import torch
    from adtof_pytorch import transcribe_to_midi

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as td:
        out = transcribe_to_midi(str(wav), Path(td) / "d.mid", device=dev)
        pm = pretty_midi.PrettyMIDI(str(out))
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(Note(id="", onset=round(n.start, 4), offset=round(n.start + 0.1, 4),
                              pitch=ADTOF_GM.get(n.pitch, n.pitch), velocity=int(n.velocity), confidence=0.9,
                              support={"adtof": 1.0}, provenance="adtof"))
    return notes


def _drums_bands(wav: str | Path) -> list[Note]:
    """Fallback without ADTOF: onsets on the drum stem, classified by band energy."""
    import librosa

    y, sr = A.load(wav, sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    bands = {"kick": (f < 150), "snare": (f >= 150) & (f < 2500), "hihat": f >= 6000}
    env = {k: np.log1p(S[m].sum(axis=0)) for k, m in bands.items()}
    notes = []
    for name, gm in (("kick", 36), ("snare", 38), ("hihat", 42)):
        e = env[name]
        flux = np.maximum(0, np.diff(e, prepend=e[0]))
        on = librosa.onset.onset_detect(onset_envelope=flux, sr=sr, hop_length=256, units="time",
                                        backtrack=False, delta=0.25 * flux.std())
        for t in on:
            notes.append(Note(id="", onset=round(float(t), 4), offset=round(float(t) + 0.1, 4), pitch=gm,
                              velocity=90, confidence=0.6, support={"bands": 1.0}, provenance="band-onsets"))
    return notes
