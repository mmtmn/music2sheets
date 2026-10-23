# music2sheets

Audio in, playable sheet music out. Separate the instruments, transcribe each
one into scored note candidates, decide which notes are kept and who plays
them with a deterministic playability solver, check the result against the
recording by resynthesis, then engrave MusicXML and PDF.

The pipeline is a Rust binary (`m2s`); a Python worker runs the neural
models. The design goal is the one most transcription tools miss: the output has to be
something a two-handed pianist, a guitarist with one fretting hand, or a
drummer with four limbs can actually play, without inventing notes the
recording doesn't contain.

## Why it works the way it does

A common approach is to transcribe to MIDI and then ask a language model to
"fix" playability note by note. That fails for three reasons: the MIDI has
already thrown away the model's uncertainty, edits are never checked against
the audio, and checking hundreds of notes for span and speed is a constraint
problem, not a judgment call. This project replaces each of those:

| Instead of | music2sheets does |
| --- | --- |
| a thresholded MIDI file | keeps every candidate note with a confidence from two models' agreement |
| an LLM judging playability | a beam-search / Viterbi decoder over an explicit model of hands, fingers, strings, frets or limbs |
| unverified edits | renders the score back to audio and compares it with the stem; an edit survives only if the match improves |

A human played the recording, so the true notes are playable. Playability is a
prior that breaks ties between uncertain candidates. It never deletes a note
the audio clearly contains; that note is flagged for review instead.

## Results

All numbers below were produced by the code in this repository. Commands to
reproduce them are in [Reproducing the numbers](#reproducing-the-numbers).

### Blind test 1: Beethoven, Sonata No. 14, 3rd movement

A live acoustic recording (Paul Pitman, Musopen, public domain), 8 min 13 s.
The transcription was committed to git before any reference score was on the
machine (`eval/blind/test1/v1` and `v2`, commits `7b8843b` and `85f4155`), then
scored against a reference edition. Full report:
[`eval/blind/test1/REPORT.md`](eval/blind/test1/REPORT.md).

| Measure | Result |
| --- | --- |
| Note F1 against the edition, onset within 50 / 100 / 250 ms | 0.768 / 0.877 / 0.910 |
| Hand agreement with the edition's staves | 95.9% of ~5,450 matched notes |
| Hard playability violations per 100 notes | 0.20 (the edition itself, same checker: 0.37) |
| Time signature, key | 4/4, C♯ minor (both correct) |
| Bars | 276 vs. 263 (beat tracking adds bars through pauses and rubato) |
| MV2H | not finished: MV2H's note alignment ran past a 1-hour limit on this 6,000-note movement |
| Runtime on an RTX 3060 | about 90 s for 8 min of audio |

About half of the 100 ms errors are notes at the right pitch whose onsets fall
100-250 ms from where the score-to-audio alignment puts them. The rest are
missed notes, octave errors and wrong neighbour pitches. Most of them are in
the coda.

This test is no longer blind: its reference has been on the machine since
it was scored. Later changes were measured on the real-recording dev set
below, never on this piece.

### Dev set (real recordings)

Eight Beethoven sonata movements: public-domain and CC0 recordings from
Wikimedia Commons, scored against Craig Sapp's Humdrum editions
(`tools/fetch_real_dev.py`; nothing from Sonata No. 14). These are real
pianos in real rooms, two of them tuned about half a semitone below A440.

| Movement | Note F1 50 / 100 ms | Hands | Score-level | Time sig. (ours / edition) | Bars (ours / edition) |
| --- | --- | --- | --- | --- | --- |
| No. 8 i | 0.756 / 0.890 | 0.899 | 0.658 | 4/4 / 4/4 | 185 / 310 |
| No. 8 ii | 0.764 / 0.926 | 0.757 | 0.653 | 3/4 / 2/4 | 93 / 73 |
| No. 8 iii | 0.832 / 0.945 | 0.937 | 0.673 | 2/2 / 2/2 | 206 / 211 |
| No. 15 i | 0.468 / 0.522 | 0.956 | 0.650 | 3/4 / 3/4 | 384 / 461 |
| No. 15 iii | 0.827 / 0.893 | 0.897 | 0.633 | 4/4 / 3/4 | 59 / 165 |
| No. 21 i | 0.804 / 0.900 | 0.964 | 0.708 | 4/4 / 4/4 | 396 / 387 |
| No. 21 iii | 0.635 / 0.739 | 0.875 | 0.696 | 4/4 / 2/4 | 271 / 684 |
| No. 31 ii | 0.926 / 0.952 | 0.903 | 0.685 | 3/4 / 2/4 | 153 / 231 |
| **Mean** | **0.752 / 0.846** | **0.898** | **0.670** | 4 of 8 right | 30.9% error |

Keys: 5 of 8 right. Two minor-key movements came out as their relative
major, and one as the dominant. No. 15 i scores low for a reason not yet
found.

The notes are mostly right; the score-level gap is in meter and written
values. The beat tracker often counts at a different level from the editor:
one beat per bar in the No. 15 scherzo, eighth notes in the No. 8 Adagio.
That multiplies every written value by the same factor. Three bugs found on
this set are fixed:

- **Tuning correction stretched time.** On recordings more than 20 cents off
  A440, every note time was compressed by 5.9%. Fixing it raised No. 21 from
  note F1 0.03 to 0.80 (i) and 0.64 (iii).
- **Tempo prior inverted.** The beat-unit choice used an inverted tempo
  prior, and the written tempo was wrong whenever the beat wasn't a quarter.
- **Evaluator chose the wrong reference.** It ignored the no-repeat orders
  that Humdrum files declare, and used an estimated reference key instead of
  the one the editor wrote.

The mean before these fixes was note F1 0.559 / 0.633 and score-level 0.524.

### Dev set (rendered piano performances, ASAP dataset)

Eight pieces by Bach, Beethoven, Chopin, Haydn, Mozart and Schubert: ASAP
performance MIDI rendered with FluidSynth, so the exact notes are known. The
blind-test piece is excluded from the dev set.

| Measure | Result |
| --- | --- |
| Note F1 (onset within 50 ms) against the performance | 0.993 mean |
| Hand assignment, all 8 pieces (staff = hand) | 95.1% (middle-C split baseline: 80.8%) |
| Hand assignment, 2 held-out pieces | 92.1% |
| Time signature correct | 6 of 8 (misses: 12/8 read as 4/4, Haydn's 4/4 read as 2/2) |
| Bar count error | 18.7% mean |
| Notated beats against ASAP's beat annotations (F-measure) | 0.873 |
| Mozart K. 332 i, scored against the ASAP score | MV2H 0.906, note F1 0.866 at 50 ms / 0.930 at 100 ms |

### Band path (synthetic test with exact ground truth)

An original 16-bar rock pattern generated by `tools/make_band_test.py`:
drums, bass, two distorted guitars panned left and right in harmony, and an
alto-sax melody, mixed by FluidSynth. Note F1 at 50 ms:

| Part | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Drums: kick / snare / hi-hat | 1.00 / 1.00 / 1.00 | 0.98 / 1.00 / 0.99 | 0.99 / 1.00 / 1.00 |
| Guitar left / right (split by stereo position) | 0.59 / 0.45 | 0.86 / 0.87 | 0.70 / 0.59 |
| Bass | 0.55 | 0.48 | 0.52 |
| Alto sax | 0.53 | 0.74 | 0.62 |

Tempo (132 vs 130 detected), meter (4/4) and key (E minor) were recovered.
Pitched band parts are the weak area: separation bleed, overtones read as
notes, and repeated bass notes merged into one. See
[Known limitations](#known-limitations).

## Install

Linux with an NVIDIA GPU is the tested setup. An RTX 3060 runs an 8-minute
piano piece end to end in about 90 seconds; nearly all of that is the neural
models. CPU works but is much slower.

You need a Rust toolchain (1.80 or newer, via [rustup](https://rustup.rs)),
[uv](https://docs.astral.sh/uv/), `ffmpeg`, `fluidsynth` and a General MIDI
SoundFont, plus a JDK if you want the MV2H score metric.

```bash
# Debian / Ubuntu
sudo apt install ffmpeg fluidsynth fluid-soundfont-gm default-jdk

git clone <this repo> music2sheets && cd music2sheets
cargo build --release      # the m2s binary: target/release/m2s
uv sync --all-extras       # the Python model worker, in .venv (CUDA 12.8 PyTorch)
tools/build_mv2h.sh        # optional: the MV2H score metric (needs a JDK)
scripts/check.sh           # fmt, clippy, Rust tests, pytest, ruff
```

`cargo install --path crates/m2s-cli` puts `m2s` on your PATH. To find the
worker, `m2s` looks for the nearest `.venv/bin/python` above the current
directory; set `M2S_PYTHON` to use another interpreter.

The Python extras:

| Extra | What it adds |
| --- | --- |
| `models` | the neural models: Transkun, Kong et al., Beat This!, Demucs, Basic Pitch, torchcrepe, the PANNs tagger |
| `drums` | ADTOF drum transcription from git (non-commercial weights). Without it, drums fall back to a band-energy onset detector |
| `eval` | synctoolbox, musicdiff, PyGuitarPro for `m2s eval` |
| `dev` | pytest |

Model weights download on first use, about 1 GB in total. If MuseScore 4 is
on your PATH it is used for engraving; otherwise Verovio renders the PDF.

Two dependency workarounds live in `pyproject.toml`:

- `basic-pitch` declares an old TensorFlow with no Python 3.12 wheels. Only
  its ONNX model is used, so uv is told to skip TensorFlow.
- `synctoolbox` pins music21 below 10, which musicdiff can't use. synctoolbox
  only touches music21 in helpers this project never calls.

The Python lock file covers Linux.

## Use

```bash
m2s run song.mp3                                   # detect instruments, do everything
m2s run piano.wav --instruments piano --out out/   # say what's playing; copy score.pdf etc. to out/
m2s run band.flac --instruments electric_guitar,bass,drums,voice
```

Without `--instruments`, a confident solo piano is transcribed directly. For
anything else the audio is separated, and instruments count as present when
their stem carries real energy through enough of the piece. The tagger then
names the guitar (electric or acoustic) and the "other" stem (sax or violin).
That last label is unreliable, so name the parts with `--instruments` when you
know them.

Each input gets a folder `runs/<hash>/` named after the audio's content hash
(the file name is never stored). The useful outputs:

| File | What it is |
| --- | --- |
| `score.pdf`, `score.musicxml` | the engraved score (MusicXML opens in MuseScore, Dorico, Sibelius, Finale) |
| `notes.json` | the decoded notes: timing, confidence, hand/finger, string/fret or limb (edits apply here) |
| `scored.json` | the same notes after notation: bar, beat, duration, voice, staff, spelling |
| `violations.json` | remaining playability problems with time, kind and the notes involved |
| `verify.json`, `heatmap_*.png` | audio match per part, worst regions, and a map of disagreements |
| `traces/*.jsonl` | every decoding and notation decision (see [Traces](#traces)) |
| `summary.json` | the headline numbers |

### Commands

| Command | Does |
| --- | --- |
| `m2s run AUDIO` | the whole pipeline (stages are cached; `--force` reruns) |
| `m2s analyze / separate / transcribe / decode / verify / notate / engrave RUN` | one stage at a time |
| `m2s trace RUN --at 308 --stage decode` | the decisions made near a time, for a note id (`--note`) or of a kind (`--kind`) |
| `m2s inspect RUN --bars 12-16` | one image: stem spectrogram, decoded notes by hand, the engraved bars |
| `m2s edit RUN patch.json` | apply a patch through the verification gate; logged to `edits.jsonl` |
| `m2s propose RUN` / `m2s review RUN` | patch proposals from the worst verify regions / propose and apply them all through the gate |
| `m2s eval RUN --ref score.musicxml` | score a run against a reference (MusicXML, MIDI, kern, Guitar Pro 3-5) |
| `m2s decode-file` / `m2s notate-file` | one stage on files outside a run (used by the dev tools) |
| `m2s presets` | list instrument presets |

Presets: `piano`, `guitar`, `electric_guitar`, `bass`, `violin`, `sax_alto`,
`sax_tenor`, `voice`, `drums`.

Two notation options for `m2s run` and `m2s notate` are experimental and off
by default. Neither improved either dev set (see
[Results](#results)), but they help on some pieces:

| Option | Does |
| --- | --- |
| `--beats dbn` | re-decode the beat tracker's activation with a dynamic Bayesian network (Krebs, Böck and Widmer 2015) and a tempo range centred on the tracker's own tempo, instead of repairing the tracker's peaks |
| `--pauses` | drop tracked beats inside pauses where the tracker is unsure (weak activation or unsteady spacing) and put a fermata on the chord before. The meter is chosen first and only whole bars are removed, so later bar lines stay in phase |

## How it works

```
                         Python worker (neural models)                 Rust core (deterministic)
audio ─► 1 ingest & analyze ─► 2 separate ─► 3 transcribe ─► 4 playability decode ─► 5 verify ─► 6 notate ─► PDF / MusicXML
             tuning, beats,      Demucs x2,     candidates      preset = effector        render &      quantize, meter,
             instruments         pan split      + confidence    model, beam/Viterbi      compare       key, spelling, voices
                                                                        ▲                   │
                                                                        └── review patches ─┘   (m2s edit: accepted only if the audio match improves)
```

The `m2s` binary runs the pipeline. Everything deterministic is Rust: the
note format, the playability solvers, verification, the edit gate, notation
and MusicXML, and evaluation scoring. A Python worker (`python -m m2s.worker`)
does what needs the neural models or Python-only libraries: analysis,
separation, transcription, the model front end for verification, engraving,
inspection images, and reference loading and alignment. The two sides share
one JSON note format and the YAML presets in `presets/`.

1. **Ingest and analyze** (worker: `m2s/analyze.py`). Decode to 44.1 kHz
   stereo, measure tuning in cents, tag instruments per 2-second window (PANNs
   AudioSet tagger), track beats and downbeats (Beat This!).
2. **Separate** (worker: `m2s/separate.py`). Skipped for a confident solo
   piano. Otherwise two Demucs models (6-source and fine-tuned 4-source) are
   averaged where they overlap. Instruments are detected from stem energy; if
   only one stem carries the music, the original mix is used as a solo. A
   guitar stem is split by stereo position when the parts are panned apart.
3. **Transcribe** (worker: `m2s/transcribe/`). Piano: Transkun V2 and Kong et
   al. (2021), fused; notes both models report get high confidence, others get
   confidence from the second model's local evidence. The tuning offset is
   corrected first. Two experimental alternatives are off by default:
   - `M2S_FUSION=vote` scores each note with a logistic function of which
     models report it, Kong's onset evidence, velocity, and whether a
     better-supported note an octave or a semitone away contradicts it. The
     weights were fit on the real-recording dev set. Leave-one-movement-out,
     it raised note F1 at 100 ms from 0.846 to 0.851, but on the one rendered
     piece checked so far (Bach) it fell from 0.998 to 0.991.
   - `M2S_HFT=1` adds hFT-Transformer (Toyama et al. 2023) as a third voter
     (fetch it with `tools/fetch_hft.sh`). It scored the same as the two-model
     vote, and voting did not reduce octave or neighbour errors: 87-90% of
     them are notes all three models agree on. It also adds about 40% model
     time. Keyboards in a band go through the same piano models on the
   keys stem. Guitar and violin: Basic Pitch with overtone suppression and
   onset-checked confidences. Bass, voice and sax: torchcrepe pitch tracking
   cut into notes (at amplitude onsets too, for bass). Drums: ADTOF.
   A recording tuned near a quarter-tone off A440 (35 cents or more) is
   ambiguous: retuning up or down fits equally well, and the wrong choice
   moves every note by a semitone. There the transcription moves a semitone
   up or down if that gives a best-fitting key with at least two fewer sharps
   or flats.
4. **Playability decode** (Rust: `crates/m2s-core/src/play/`). Every
   instrument is a YAML preset describing its effectors, states, hard limits
   and transition costs.
   - *Piano:* beam search over keep/drop and left/right hand per chord (span,
     shift speed, crossings, busy fingers, line continuity, acoustic
     confidence), then Viterbi fingering with finger-pair spans adapted from
     Parncutt et al. (1997). The weights were fit on the dev set
     (`tools/tune_hands.py`).
   - *Guitar, bass, violin:* Viterbi over string/fret fingerings (stretch by
     position, shift speed, string skips, open strings). Several players on one
     bus are split into lines first.
   - *Drums:* limb assignment with single-stroke rate and double-kick limits.
   - *Voice, sax:* one note at a time, range, voice type, breath limits.
5. **Verify** (Rust: `crates/m2s-core/src/verify.rs`). Render the decoded
   notes with FluidSynth, run the same front end on render and stem (Kong onset
   activations for piano, ADTOF for drums, Basic Pitch otherwise), and compare
   in 50 ms × pitch cells. This yields the audio match, the worst regions, and
   a heatmap.
6. **Notate** (Rust: `crates/m2s-notate/`). Beat unit and meter from the
   tracker plus bass, harmony-change and downbeat evidence (a snare backbeat
   fixes the beat when there are drums; triplet-heavy beats give compound
   meter; a half-note beat gives cut time). Per-beat subdivision choice; trills and grace notes; key from the
   notes, with relative major/minor decided by the bass; spelling on the line
   of fifths; up to two voices per staff; clefs chosen by minimising ledger
   lines; dynamics from velocity; sforzandi; staccato. MusicXML is written
   directly: grand-staff piano, notation plus TAB with string/fret, percussion,
   transposing instruments, beams and tuplets. The worker engraves it with
   Verovio or MuseScore.

### Traces

Every run writes `traces/decode.jsonl` and `traces/notate.jsonl`: one JSON
line per decision, with its time, the notes involved and the alternatives
considered. `m2s trace` filters them:

```console
$ m2s trace runs/a4c16b5e2c5e --stage decode --at 308.0 --window 0.15
{"kind":"slice","t":307.9846,"notes":["pno-004503","pno-004502"],"pitches":[63,72],"best":"LR","best_cost":595.726,"runner_up":["LL",595.861],...}
```

Here a two-note chord went to left + right hand, and keeping both in the left
hand was 0.135 more expensive. A wrong hand, a strange subdivision, a missing
bar: each can be traced to the decision and the numbers that made it.

### The review loop (for Claude or a person)

Nobody edits MusicXML or MIDI directly. Reviewers write JSON patches:

```json
{"reason": "bar 23: stem has C#5, score lacks it",
 "ops": [{"op": "add", "track": "piano", "onset": 41.52, "offset": 41.70, "pitch": 73, "hand": "R"}]}
```

Ops are `add`, `delete`, `retime`, `repitch`, `reassign` (hand, finger,
string, fret, limb, player), `retag` and `spell`. `m2s edit` applies the
patch to a copy, re-renders the affected window and compares it with the stem:

- A patch that adds or removes notes is accepted only if the audio match
  improves (> 0.002) with no new violations, or clearly improves (> 0.01).
  It is also accepted if it removes a violation at almost no audio cost.
- A notation-only patch must not add violations.
- Every patch, accepted or rejected, goes to `edits.jsonl` with its reason.

`m2s propose` turns the worst verify regions into candidate patches (restore a
dropped candidate the stem supports, delete an unsupported low-confidence
note). `m2s review` proposes and runs every proposal through the gate. A person
or Claude can also look at a region with `m2s inspect` and write patches by hand.

## Evaluation and blind tests

`m2s eval RUN --ref REFERENCE` has the worker load the reference and align it
to the recording with synctoolbox's multiresolution DTW. The alignment uses
the audio and the score only, never our notes. Rust then scores:

- note precision / recall / F1 at 50, 100 and 250 ms. 50 ms is the standard;
  the looser ones show how much of the gap is alignment timing.
- hand agreement (our hand vs. the reference's staff, on matched notes)
- playability violations per 100 notes, for us and for the reference itself
- time signature, key, bar count
- a score-level metric (below), computed in Rust in well under a second
- MV2H (multi-pitch, voice, meter, value, harmony), if built. Its note
  alignment did not finish within an hour on the 8-minute blind test, so use
  `--no-mv2h` for long pieces.
- optionally, `--diff-bars N` runs musicdiff on the first N bars; it is too
  slow for whole movements

**Score-level metric.** MV2H's five parts, computed against the aligned
reference instead of by MV2H's own search. Each part is an F1 or a weighted
score in 0..1, and `score_level` is their mean:

| Part | What is compared |
| --- | --- |
| pitch | note F1 (onset within 100 ms) |
| voice | links between consecutive chords within one voice, ours vs. the reference's, on matched notes (F1) |
| meter | our bar lines and beats against the reference's, as times in the recording (mean of the two F1s, within 100 ms) |
| value | share of matched notes whose written duration equals the reference's |
| harmony | the key, with MIREX weights: 1 exact, 0.5 a fifth away, 0.3 relative major/minor, 0.2 parallel |

**References.** Repeats are expanded automatically when the expanded
version better matches the number of notes played. Humdrum files declare
their own section orders (`*>[...]`, plus named ones such as
`*>norep[...]` for a performance without repeats). Each declared order is
built, as the `thrux` tool does, and the one closest to the performance is
used. The reference key is the one the encoder wrote (Humdrum `*c:`,
MusicXML `<mode>`); music21's estimate from the notes is used only when
none is written. A recording tuned about half a semitone off A440 can put
the alignment's chroma one bin away from the score's, so the recording's
chroma is shifted by the semitone (−1, 0 or +1) that best matches the score.

**Protocol.** Transcribe, commit the output under `eval/blind/<test>/vN/`, and
only then put the reference in `eval/references/` (git-ignored) and run
`m2s eval`. The commit hash proves the order.

### Running blind test 2 on your machine

For a commercial recording, both the audio and the reference transcription
stay on your machine. Nothing from them belongs in this repository.

```bash
m2s run "path/to/your/recording.flac" --instruments electric_guitar,bass,drums,voice,piano
mkdir -p eval/blind/test2/v1 && cp runs/<hash>/{score.pdf,score.musicxml,summary.json,verify.json} eval/blind/test2/v1/
git add eval/blind/test2 && git commit -m "Blind test 2 output v1 (reference not yet seen)"
m2s eval runs/<hash> --ref "path/to/your/reference.gp5" --no-mv2h     # or .musicxml / .mid
```

Guitar Pro 6/7 files (`.gpx`, `.gp`) aren't read directly. Export to
MusicXML or GP5 from Guitar Pro, TuxGuitar or MuseScore first.

## Known limitations

- **Metrical level** is the main source of score-level error on real
  recordings. The beat tracker often follows a different level from the one
  the editor wrote: one beat per bar in a fast 3/4 scherzo, eighth notes in
  a 2/4 Adagio, half notes in an alla breve. The notes are right, but bars
  are too long or too short and every written value is off by the same
  factor. Some of this is a real ambiguity: a 4/4 Moderato with 32nd notes
  and a 2/2 Allegro with 16th notes look alike in audio. The rendered dev set
  has one such miss (Haydn's 4/4 written as 2/2 with doubled values).
- **Beat tracking on expressive solo piano**: where the tracker drifts, notes
  are right but land a sixteenth off the beat, and pauses add bars.
- **Compound meter**: 12/8 can come out as 4/4 with triplets.
- **Minor keys** are sometimes reported as their relative major (2 of the 8
  real-recording movements).
- **Distorted guitar** has no dedicated public transcription model.
  Basic Pitch with overtone suppression reaches F1 0.6-0.7 on the synthetic
  test, and less on real mixes. Leftover overtones also pull the tab up the
  neck. `tools/train_guitar.py` fine-tunes Basic Pitch on EGDB, Guitar-TECHS
  and GuitarSet with amp-sim augmentation. `tools/eval_guitar.py` compares
  the result with stock Basic Pitch on held-out players and amps. That
  comparison isn't finished, so the fine-tuned model is not used unless you
  point `M2S_GUITAR_MODEL` at it. EGDB has no stated license, so weights
  trained on it are for research use only.
- **Bass** repeated notes on one pitch are sometimes merged.
- **Playing techniques** (bends, slides, hammer-ons, tapping, palm mutes) are
  carried as pitch curves in `notes.json` but not yet written as notation.
- **Verification** uses a General MIDI SoundFont. Timbre mismatch is reduced
  by comparing model activations rather than audio, but not removed.
- **MV2H and musicdiff** are too slow for whole movements.
- **Automatic instrument naming** for the "other" stem (sax vs. violin) is
  unreliable; pass `--instruments` for band recordings.

## Reproducing the numbers

```bash
python tools/fetch_dev_set.py && python tools/render_dev_audio.py
for f in data/dev/asap/*/*/performance.wav data/dev/asap/*/*/*/performance.wav; do
  [ -f "$f" ] && m2s run "$f" --runs-root data/devruns --until verify; done
python tools/dev_hands.py          # hand assignment
python tools/dev_rhythm.py         # meter, bars, beats, note F1
python tools/fetch_real_dev.py     # real recordings (Wikimedia Commons) + Humdrum editions
python tools/real_dev.py           # run and score all 8 movements
python tools/real_dev_notation.py -- --pauses   # re-notate and re-score with other notation options
python tools/make_band_test.py && m2s run data/bandtest/band.wav \
  --instruments electric_guitar,bass,drums,sax_alto --runs-root data/bandruns
python tools/eval_band_test.py data/bandruns/<hash>
tools/fetch_blind_audio.sh && m2s run data/blind/test1.mp3
scripts/check.sh
```

## Project layout

```
crates/
  m2s-core/      note format, presets, playability solvers (play/), verification, edit gate,
                 evaluation scoring, MIDI, traces
  m2s-notate/    notation: rhythm and meter, key, spelling, ornaments, voices and bars, MusicXML
  m2s-cli/       the m2s binary: run folders, stage caching, worker calls, resynthesis
presets/         one YAML file per instrument (shared by Rust and Python)
m2s/             the Python worker: analyze, separate, transcribe/, verify (model front ends),
                 engrave, inspect, evaluate/ (reference loading and alignment, MV2H)
tools/           dev-set fetch and tuning, band test, blind-test helpers, MV2H build
tests/           golden decoder decisions (tests/golden) and Python worker tests
eval/blind/      committed blind-test outputs and reports
scripts/         check.sh
```

### Parity with the first implementation

The pipeline was first written in Python, then the deterministic stages were
rewritten in Rust. Before the Python versions were removed, both ran on the
same inputs: the blind recording, the band test and eight dev pieces. The
playability decisions matched on every note of all ten runs, and so did the
notation (every note's position, duration and spelling). The Rust stages are
about 20 to 55 times faster. `crates/m2s-core/tests/golden_decode.rs` keeps the
decoder's decisions on the first minute of the blind test and on the band test
frozen.

Two Python bugs were fixed in the port. The engraving lint checked ledger
lines against the clef in force before each bar (93 warnings on the blind
test, 28 real). Verification compared split guitar renders with the full mix
instead of their own stem.

## Models, data and licenses

This repository's code is MIT-licensed. It downloads and runs third-party
models whose licenses are their own. Check each before any commercial use.

| Component | Used for | Notes |
| --- | --- | --- |
| [Transkun](https://github.com/Yujia-Yan/Transkun) | piano transcription | |
| [Kong et al. piano transcription](https://github.com/qiuqiangkong/piano_transcription_inference) | piano transcription, pedal, verification front end | |
| [hFT-Transformer](https://github.com/sony/hFT-Transformer) | optional third piano model (`M2S_HFT=1`) | fetched by `tools/fetch_hft.sh`, pinned commit and checkpoint |
| [Demucs](https://github.com/adefossez/demucs) | source separation | |
| [Basic Pitch](https://github.com/spotify/basic-pitch) | polyphonic pitched transcription | ONNX model only |
| [torchcrepe](https://github.com/maxrmorrison/torchcrepe) | monophonic pitch | |
| [ADTOF](https://github.com/MZehren/ADTOF) via [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) | drum transcription | weights CC BY-NC-SA 4.0 (non-commercial) |
| [Beat This!](https://github.com/CPJKU/beat_this) | beats and downbeats | |
| [PANNs](https://github.com/qiuqiangkong/panns_inference) | instrument tagging | |
| [music21](https://github.com/cuthbertLab/music21), [Verovio](https://www.verovio.org) | notation, engraving | |
| [synctoolbox](https://github.com/meinardmueller/synctoolbox), [mir_eval](https://github.com/mir-evaluation/mir_eval), [MV2H](https://github.com/apmcleod/MV2H), [musicdiff](https://github.com/gregchapman-dev/musicdiff) | evaluation | |
| [ASAP dataset](https://github.com/fosfrancesco/asap-dataset) | dev set (downloaded, not committed) | |

Finger-pair spans in `presets/piano.yaml` are adapted from Parncutt,
Sloboda, Clarke, Raekallio & Desain (1997), "An ergonomic model of keyboard
fingering for melodic fragments", *Music Perception* 14(4).
