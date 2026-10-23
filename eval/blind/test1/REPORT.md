# Blind test 1 report: Beethoven, Piano Sonata No. 14, 3rd movement

Live acoustic recording by Paul Pitman (Musopen, Public Domain Mark), 8 min
13 s. Reference: Craig Sapp's Humdrum edition (`sonata14-3.krn`), with its own
declared repeat expansion applied (263 bars, 6,413 notes).

Against the published score, the transcription gets 88% of notes right within
100 ms (F1 0.877) and 91% within 250 ms. It puts 95.9% of matched notes in the
same hand as the edition. It has fewer playability violations than the
edition's own notation, under the same checker.

## Protocol and timeline

| Step | Commit | What happened |
| --- | --- | --- |
| Transcription v1 | `7b8843b` | `m2s run` on the anonymized audio, no reference on the machine |
| Transcription v2 | `85f4155` | generic fixes found by inspecting v1 (no reference), plus the review pass |
| Reference fetched | after `85f4155` | Humdrum edition and ASAP MusicXML downloaded into `eval/references/` (git-ignored) |
| Scoring | this report | `m2s eval`, alignment by synctoolbox (audio against score, never against our notes) |

Changes between v1 and v2 were generic and made without any reference:

- Piano verification now compares onset activations. Pedal sustain in the
  SoundFont render had been reading as extra notes.
- A note sounding for at most 60% of its slot is notated staccato (was 45%).
- The edit gate is stricter: a content edit must improve the audio match.

The review pass then ran 68 patch proposals from `tools/propose_patches.py`
through `m2s edit`. The gate accepted 46. All 68 are logged in
`v2/edits.jsonl`.

One evaluation bug was found and fixed after the reference arrived.
music21 ignores Humdrum repeat labels, so the first scoring attempt compared
a performance that takes the exposition repeat with a score that doesn't.
`load_reference` now applies the file's own expansion list (`*>[I,A,A1,A,B]`),
as Humdrum's `thrux` does. That changes how the reference is read, not the
transcription, and the numbers below use the corrected reading.

## Results

| Measure | v1 | v2 | Reference / note |
| --- | --- | --- | --- |
| Note F1, onset within 50 ms | 0.768 | 0.768 | P 0.79, R 0.75 |
| Note F1, within 100 ms | 0.875 | 0.877 | P 0.90, R 0.85 |
| Note F1, within 250 ms | 0.909 | 0.910 | P 0.94, R 0.88 |
| Hand agreement (our hand vs. the edition's staff) | 95.9% | 95.9% | over ~5,450 matched notes |
| Hard playability violations per 100 notes | 0.20 | 0.20 | edition itself: 0.37 |
| Time signature | 4/4 | 4/4 | 4/4 |
| Key | C♯ minor | C♯ minor | C♯ minor |
| Bars | 276 | 276 | 263 (with the exposition repeat) |
| Audio match (our verification) | 0.962 | 0.965 | onset-based, against the recording |
| MV2H | not run | not finished | the alignment step ran past a 1-hour limit on 6,000 notes |

Full numbers: `v1/eval.json`, `v2/eval.json`.

## Where the errors are (v2, 100 ms tolerance)

Errors cluster rather than spread evenly. Ten-second windows have a median
F1 of 0.90 (lower quartile 0.83). The worst windows are all in the last two
minutes (380-490 s): the coda, where the performer's tempo moves most and
the score-to-audio alignment is least sure.

| Missed reference notes (950) | Count |
| --- | --- |
| right pitch, onset 100-250 ms away (alignment timing) | 483 |
| absent from the transcription | 452 |
| repeated note not re-struck | 15 |

| Transcribed notes with no match (589) | Count |
| --- | --- |
| right pitch, onset 100-250 ms away (alignment timing) | 189 |
| no counterpart in the score | 186 |
| octave error | 109 |
| a semitone or tone off | 105 |

About half the remaining 100 ms errors are timing, not notes. That fits the
dev-set result: on Mozart K. 332, where the true performance is known,
aligning the score to the audio alone dropped note F1 from 0.998 to 0.866 at
50 ms.

## What this says about the approach

- **Playability.** The output is playable: 12 hard violations in 6,060 notes,
  all of them single fast finger moves in dense passages. The published
  edition, read by the same checker, has more. No notes were deleted for
  playability. Uncertain candidates were dropped only when the audio didn't
  support them.
- **Hands.** The solver agrees with the editor's staff assignment 95.9% of the
  time, on a piece whose arpeggios sweep the whole keyboard. A fixed split
  point can't do that.
- **The review loop.** It worked as designed, but it hardly moved the score:
  46 accepted edits among ~6,000 notes changed F1 by +0.001. Its value is in
  catching local problems. Systematic gains have to come from the stages.
- **Score level.** The bar count (276 vs. 263) shows where to work next. The
  beat tracker keeps counting through pauses and rubato, which adds bars;
  13 extra bars over 8 minutes is about 5%.

## Next steps, in order of expected gain

1. Beat and downbeat tracking robust to rubato and pauses: fermata and pause
   detection; tempo-change points instead of a continuous beat stream.
2. Octave and neighbour-pitch errors (214 notes): a third transcription model
   in the fusion (hFT-Transformer), with majority voting on pitch.
3. A score-informed second pass for a known piece is out of scope for a blind
   system, but a self-similarity pass could make repeated sections consistent
   (the exposition and its repeat should transcribe identically).
