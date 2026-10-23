//! Stage 6: notation. Quantization, meter, key, spelling, voices, measures and the
//! MusicXML writer (a port of the Python `m2s/notate/`, decision for decision).
//!
//! Entry point: [`notate`]. It fills `beat`, `qdur`, `spelling`, `measure`, `voice` and
//! `staff` on the kept notes, folds trills and marks grace notes, and returns the
//! MusicXML plus an `info` object with the same keys as the Python `notation.json`.

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

#[derive(Debug, Clone)]
pub struct Options {
    pub fingering: bool,
    pub dynamics: bool,
    /// pedal marks are optional: most editions leave pedalling to the player
    pub pedal: bool,
    pub title: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            fingering: false,
            dynamics: true,
            pedal: false,
            title: "Transcription".into(),
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
            let cleaned = rhythm::clean_beats(beats);
            if cleaned.len() >= 4 {
                db = Some(rhythm::downbeat_strength(&cleaned, &acts.column(1), act_fps));
                beats_v = cleaned;
            }
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
    let info = json!({
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
