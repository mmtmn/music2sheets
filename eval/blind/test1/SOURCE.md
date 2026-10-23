# Blind test 1: input

- Recording: Musopen, performed by Paul Pitman, released under the Public Domain Mark 1.0.
  <https://archive.org/details/MoonlightSonata_845>, file `Sonata_no_14_in_c_sharp_minor_moonlight_op_27_no_2_Iii.Presto.mp3`
- Composition: Beethoven, Piano Sonata No. 14, Op. 27 No. 2, 3rd movement (public domain).
- Audio content hash (first 12 hex of SHA-256): `a4c16b5e2c5e`. The pipeline sees only this hash.
- The audio file itself is not committed (data/ is ignored). Fetch it with `tools/fetch_blind_audio.sh`.

## Protocol

1. Transcribe with `m2s run`. No reference score is on the machine.
2. Commit the output under `eval/blind/test1/vN/` before the reference is fetched.
3. Fetch the reference, then run `m2s eval`. The report goes in `eval/blind/test1/REPORT.md`.
