//! Stage 4: playability decoding. One entry point; the preset's family picks the effector model.

pub mod drums;
pub mod fretted;
pub mod mono;
pub mod piano;

use anyhow::Result;
use serde::Serialize;

use crate::ir::{Note, Track};
use crate::preset::{self, Family};
use crate::trace::Trace;

/// A remaining playability problem: what, when, which notes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Violation {
    pub kind: String,
    pub time: f64,
    pub detail: String,
    pub notes: Vec<String>,
    pub hand: String,
}

impl Violation {
    pub fn new(kind: &str, time: f64, detail: String, notes: Vec<String>, hand: &str) -> Self {
        Violation {
            kind: kind.into(),
            time: round_to(time, 3),
            detail,
            notes,
            hand: hand.into(),
        }
    }
}

/// Python's round(x, n): round half to even on the decimal expansion.
pub fn round_to(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap_or(x)
}

/// Decode one track in place: sets `keep` and the effector of every note.
pub fn decode_track(track: &mut Track, trace: &mut Trace) -> Result<Vec<Violation>> {
    let p = preset::load(&track.preset)?;
    match p.family {
        Family::Keyboard => piano::PianoDecoder::new(&track.preset)?.decode(track, trace),
        Family::Fretted => fretted::FrettedDecoder::new(&track.preset)?.decode(track, trace),
        Family::Drums => drums::DrumDecoder::new(&track.preset)?.decode(track, trace),
        Family::Wind => mono::MonoDecoder::new(&track.preset)?.decode(track, trace),
    }
}

/// Group note indices into time slices (chords): sorted by (onset, pitch); a note joins the
/// current slice if it starts within `window` of the slice's first note. Slices are sorted by pitch.
pub fn make_slices(notes: &[Note], idx: &[usize], window: f64) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = idx.to_vec();
    order.sort_by(|&a, &b| {
        notes[a]
            .onset
            .partial_cmp(&notes[b].onset)
            .unwrap()
            .then(notes[a].pitch.cmp(&notes[b].pitch))
    });
    let mut slices: Vec<Vec<usize>> = Vec::new();
    for i in order {
        match slices.last_mut() {
            Some(s) if notes[i].onset - notes[s[0]].onset <= window => s.push(i),
            _ => slices.push(vec![i]),
        }
    }
    for s in &mut slices {
        s.sort_by_key(|&i| notes[i].pitch);
    }
    slices
}
