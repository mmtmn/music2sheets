//! Stage 6: notation. Quantization, meter, key, spelling, voices, measures and the
//! MusicXML writer (a port of the Python `m2s/notate/`, decision for decision).
//!
//! Entry point: [`notate`]. It fills `beat`, `qdur`, `spelling`, `measure`, `voice` and
//! `staff` on the kept notes, folds trills and marks grace notes, and returns the
//! MusicXML plus an `info` object with the same keys as the Python `notation.json`.

pub mod dbn;
pub mod keys;
pub mod lint;
pub mod musicxml;
pub mod np;
pub mod ornaments;
pub mod rhythm;
pub mod score;
pub mod spell;

use std::collections::HashMap;

use m2s_core::ir::{Hand, Note, Piece};
use m2s_core::npy::Array2;
use m2s_core::preset::{self, Family};
use m2s_core::trace::Trace;
use serde_json::{json, Value};

use crate::keys::Key;
use crate::spell::SpellNote;

/// Where the beat grid comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeatSource {
    /// the tracker's peak-picked beats with doubled and missing beats repaired
    Clean,
    /// Viterbi decoding of the tracker's activation with a tempo-smoothness prior (see `dbn`)
    Dbn,
}

impl std::str::FromStr for BeatSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "clean" => Ok(BeatSource::Clean),
            "dbn" => Ok(BeatSource::Dbn),
            o => Err(format!("unknown beat source {o:?} (clean | dbn)")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub fingering: bool,
    pub dynamics: bool,
    /// pedal marks are optional: most editions leave pedalling to the player
    pub pedal: bool,
    pub title: String,
    pub beats: BeatSource,
    /// drop beats inside pauses and mark a fermata (see rhythm::compress_pauses)
    pub pauses: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            fingering: false,
            dynamics: true,
            pedal: false,
            title: "Transcription".into(),
            beats: BeatSource::Clean,
            pauses: false,
        }
    }
}

pub struct Notation {
    pub musicxml: String,
    pub info: Value,
}

fn refs(piece: &Piece, pred: impl Fn(usize) -> bool) -> Vec<(usize, usize)> {
    let mut out = vec![];
    for (t, tr) in piece.tracks.iter().enumerate() {
        if !pred(t) {
            continue;
        }
        for (n, note) in tr.notes.iter().enumerate() {
            if note.keep {
                out.push((t, n));
            }
        }
    }
    out
}

fn notes_of<'a>(piece: &'a Piece, r: &[(usize, usize)]) -> Vec<&'a Note> {
    r.iter().map(|&(t, n)| &piece.tracks[t].notes[n]).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn notate(
    piece: &mut Piece,
    beats: &[f64],
    downbeats: &[f64],
    beat_acts: Option<&Array2>,
    act_fps: f64,
    duration: Option<f64>,
    opts: &Options,
    trace: &mut Trace,
) -> anyhow::Result<Notation> {
    let _ = duration;
    let mut families = vec![];
    for t in &piece.tracks {
        families.push(preset::load(&t.preset)?.family);
    }
    let (mut trills, mut graces) = (0, 0);
    for (t, tr) in piece.tracks.iter_mut().enumerate() {
        if matches!(families[t], Family::Keyboard | Family::Fretted | Family::Wind) {
            trills += ornaments::fold_trills(tr, 0.11, 5);
            graces += ornaments::mark_graces(tr, 0.07, 0.09);
        }
    }
    trace.with("ornaments", || json!({"trills": trills, "graces": graces}));

    let kept = refs(piece, |_| true);
    let mut pitched = refs(piece, |t| families[t] != Family::Drums);
    if pitched.is_empty() {
        pitched = kept.clone();
    }
    let drums = refs(piece, |t| families[t] == Family::Drums);

    let mut beats_v = beats.to_vec();
    let mut db: Option<Vec<f64>> = None;
    if let Some(acts) = beat_acts {
        if !pitched.is_empty() {
            let chosen = match opts.beats {
                BeatSource::Clean => rhythm::clean_beats(beats),
                BeatSource::Dbn => {
                    // repair the tracker's local slips without changing its metrical level: the
                    // tempo range is centred on the tracker's own median tempo
                    let ibi = np::median(&np::diff(&rhythm::clean_beats(beats)));
                    let bpm = if ibi > 0.0 { 60.0 / ibi } else { 120.0 };
                    let params = dbn::DbnParams {
                        min_bpm: bpm * 0.6,
                        max_bpm: bpm * 1.6,
                        ..Default::default()
                    };
                    dbn::track(&acts.column(0), act_fps, &params)
                }
            };
            let mut chosen = chosen;
            if opts.pauses && chosen.len() >= 8 {
                let mut onsets: Vec<f64> = notes_of(piece, &pitched).iter().map(|n| n.onset).collect();
                onsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
                onsets.dedup_by(|a, b| (*a - *b).abs() < 0.03);
                // choose the meter on the full beat list first, then remove whole bars only, so the
                // bar lines after each pause stay in phase
                let bpb = {
                    let db0 = rhythm::downbeat_strength(&chosen, &acts.column(1), act_fps);
                    rhythm::build_grid(
                        &chosen,
                        downbeats,
                        &notes_of(piece, &pitched),
                        Some(&db0),
                        &notes_of(piece, &drums),
                        &mut Trace::off(),
                    )
                    .beats_per_bar
                };
                let (kept, pauses) = rhythm::compress_pauses(
                    &chosen,
                    &onsets,
                    &acts.column(0),
                    act_fps,
                    1.5,
                    0.35,
                    0.25,
                    bpb.max(1) as usize,
                );
                trace.with("pauses", || {
                    json!({"pauses": pauses.iter().map(|p| json!({"t": p.after, "until": p.until,
                    "removed_beats": p.removed_beats, "mean_activation": np::round_n(p.mean_activation, 3),
                    "ibi_variation": np::round_n(p.ibi_variation, 3)})).collect::<Vec<_>>()})
                });
                // the chord before each pause carries the fermata
                for p in &pauses {
                    for (t, tr) in piece.tracks.iter_mut().enumerate() {
                        if families[t] == Family::Drums {
                            continue;
                        }
                        for n in tr
                            .notes
                            .iter_mut()
                            .filter(|n| n.keep && (n.onset - p.after).abs() < 0.03)
                        {
                            n.tags.push("fermata".into());
                        }
                    }
                }
                chosen = kept;
            }
            if chosen.len() >= 4 {
                db = Some(rhythm::downbeat_strength(&chosen, &acts.column(1), act_fps));
                beats_v = chosen;
            }
            trace.with(
                "beats",
                || json!({"source": format!("{:?}", opts.beats), "count": beats_v.len()}),
            );
        }
    }
    let mut grid = rhythm::build_grid(
        &beats_v,
        downbeats,
        &notes_of(piece, &pitched),
        db.as_deref(),
        &notes_of(piece, &drums),
        trace,
    );
    let quant = rhythm::quantize(&notes_of(piece, &kept), &mut grid, trace);
    for (&(t, n), (b, d)) in kept.iter().zip(quant) {
        piece.tracks[t].notes[n].beat = Some(b);
        piece.tracks[t].notes[n].qdur = Some(d);
    }

    let bass = refs(piece, |t| piece.tracks[t].preset == "bass");
    let glob = keys::estimate_key(&notes_of(piece, &pitched), &notes_of(piece, &bass));
    let mut key_sharps: HashMap<usize, i32> = HashMap::new();
    for (t, family) in families.iter().enumerate() {
        if *family == Family::Drums {
            continue;
        }
        let trans = preset::load(&piece.tracks[t].preset)?.transposition;
        let k = if trans != 0 {
            Key {
                tonic_pc: (glob.tonic_pc + trans).rem_euclid(12),
                mode: glob.mode,
                score: glob.score,
            }
        } else {
            glob
        };
        let idx: Vec<usize> = (0..piece.tracks[t].notes.len())
            .filter(|&n| piece.tracks[t].notes[n].keep)
            .collect();
        let spelled = {
            let tr = &piece.tracks[t];
            let sn: Vec<SpellNote> = idx
                .iter()
                .map(|&n| {
                    let note = &tr.notes[n];
                    let hand_key = match note.effector.hand {
                        Some(Hand::L) => "L".to_string(),
                        Some(Hand::R) => "R".to_string(),
                        None => note.staff.map(|s| s.to_string()).unwrap_or_else(|| "None".into()),
                    };
                    SpellNote {
                        id: &note.id,
                        onset: note.onset,
                        pitch: note.pitch + trans,
                        hand_key,
                    }
                })
                .collect();
            spell::spell(&sn, &k, 0.82)
        };
        for (&n, s) in idx.iter().zip(spelled) {
            piece.tracks[t].notes[n].spelling = Some(s);
        }
        key_sharps.insert(t, glob.sharps());
    }

    let score = score::build_score(piece, &grid, &key_sharps, opts, trace)?;
    let xml = musicxml::write(&score, piece);
    let lint = lint::lint(&score, piece);
    // bar and beat start times in seconds (for score-level evaluation against an aligned reference)
    let beat_q = grid.beat_quarters();
    let mut bar_times = vec![];
    let mut beat_times = vec![];
    for (a, b) in score.bars.starts.iter().zip(score.bars.ends.iter()) {
        let (a, b) = (score::qf(*a), score::qf(*b));
        bar_times.push(np::round_n(grid.time_of(a), 4));
        let mut x = a;
        while x < b - 1e-9 {
            beat_times.push(np::round_n(grid.time_of(x), 4));
            x += beat_q;
        }
    }
    let info = json!({
        "bar_times": bar_times,
        "beat_times": beat_times,
        "time_signature": grid.time_signature(),
        "pickup_quarters": grid.pickup_quarters,
        "qpm": np::round_n(grid.qpm(), 1),
        "beat_unit_quarters": grid.unit,
        "key": glob.name(),
        "key_score": np::round_n(glob.score, 3),
        "ornaments": {"trills": trills, "graces": graces},
        "grid": Value::Object(grid.info.clone()),
        "lint": lint,
        "bars": if score.parts.is_empty() { 0 } else { score.bars.starts.len() },
    });
    Ok(Notation { musicxml: xml, info })
}
