//! The edit protocol: notes change only through JSON patches, and every patch passes the
//! verification gate.
//!
//! A patch is applied to a copy, the affected window is re-rendered and compared with the stem.
//! A patch that adds or removes notes is accepted only if the audio match improves (> MIN_GAIN)
//! without new playability violations, or improves clearly (> DELTA), or removes a violation at
//! almost no audio cost (> -EPS). Notation-only patches must not add violations. Every patch,
//! accepted or not, is logged by the caller.

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::ir::{Effector, Hand, Note, Piece, Track};
use crate::npy::Array2;
use crate::play::piano::PianoDecoder;
use crate::preset::{self, Family};
use crate::verify::{self, CELL};

pub const EPS: f64 = 0.005;
pub const MIN_GAIN: f64 = 0.002;
pub const DELTA: f64 = 0.01;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Who {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand: Option<Hand>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finger: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub string: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fret: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limb: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub player: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Op {
    Add {
        #[serde(default)]
        track: Option<String>,
        #[serde(default)]
        id: Option<String>,
        onset: f64,
        offset: f64,
        pitch: i32,
        #[serde(default)]
        velocity: Option<i32>,
        #[serde(flatten)]
        who: Who,
    },
    Delete {
        id: String,
    },
    Retime {
        id: String,
        #[serde(default)]
        onset: Option<f64>,
        #[serde(default)]
        offset: Option<f64>,
    },
    Repitch {
        id: String,
        pitch: i32,
    },
    Reassign {
        id: String,
        #[serde(flatten)]
        who: Who,
    },
    Retag {
        id: String,
        tags: Vec<String>,
    },
    Spell {
        id: String,
        spelling: String,
    },
}

impl Op {
    pub fn notation_only(&self) -> bool {
        matches!(self, Op::Spell { .. } | Op::Retag { .. } | Op::Reassign { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    #[serde(default)]
    pub reason: String,
    pub ops: Vec<Op>,
}

/// Renders a track and runs the model front end on the render (implemented by the CLI: MIDI ->
/// FluidSynth -> Python worker). Returns activation cells for [t0, t1], row 0 = t0.
pub trait Resynth {
    fn render_cells(&mut self, track: &Track, t0: f64, t1: Option<f64>) -> Result<Array2>;
}

fn apply_who(e: &mut Effector, who: &Who) {
    if let Some(h) = who.hand {
        e.hand = Some(h);
    }
    if who.finger.is_some() {
        e.finger = who.finger;
    }
    if who.string.is_some() {
        e.string = who.string;
    }
    if who.fret.is_some() {
        e.fret = who.fret;
    }
    if who.limb.is_some() {
        e.limb = who.limb.clone();
    }
}

/// What a patch touched.
pub struct Applied {
    pub notes: Vec<String>,
    pub tracks: Vec<String>,
    /// seconds
    pub window: (f64, f64),
}

/// Apply ops in place.
pub fn apply(piece: &mut Piece, ops: &[Op]) -> Result<Applied> {
    let mut touched = vec![];
    let mut tracks: Vec<String> = vec![];
    let (mut t0, mut t1) = (f64::INFINITY, f64::NEG_INFINITY);
    for (k, op) in ops.iter().enumerate() {
        if let Op::Add {
            track,
            id,
            onset,
            offset,
            pitch,
            velocity,
            who,
        } = op
        {
            let name = track
                .clone()
                .or_else(|| piece.tracks.first().map(|t| t.name.clone()))
                .ok_or_else(|| anyhow!("no tracks"))?;
            let tr = piece.track_mut(&name).ok_or_else(|| anyhow!("no track {name}"))?;
            // deterministic id from the note itself
            let nid = id.clone().unwrap_or_else(|| {
                format!(
                    "{}-add-{}-{}-{k}",
                    &name[..name.len().min(3)],
                    (onset * 1000.0).round() as i64,
                    pitch
                )
            });
            let mut n = Note::new(nid.clone(), *onset, *offset, *pitch, velocity.unwrap_or(70));
            n.provenance = "edit".into();
            apply_who(&mut n.effector, who);
            tr.notes.push(n);
            touched.push(nid);
            if !tracks.contains(&name) {
                tracks.push(name);
            }
            t0 = t0.min(*onset);
            t1 = t1.max(*offset);
            continue;
        }
        let id = match op {
            Op::Delete { id }
            | Op::Retime { id, .. }
            | Op::Repitch { id, .. }
            | Op::Reassign { id, .. }
            | Op::Retag { id, .. }
            | Op::Spell { id, .. } => id,
            Op::Add { .. } => unreachable!(),
        };
        let (ti, ni) = piece.find(id).ok_or_else(|| anyhow!("no note {id}"))?;
        let name = piece.tracks[ti].name.clone();
        let n = &mut piece.tracks[ti].notes[ni];
        t0 = t0.min(n.onset);
        t1 = t1.max(n.offset);
        match op {
            Op::Delete { .. } => {
                n.keep = false;
                n.tags.push("deleted-by-edit".into());
            }
            Op::Retime { onset, offset, .. } => {
                n.onset = onset.unwrap_or(n.onset);
                n.offset = offset.unwrap_or(n.offset);
            }
            Op::Repitch { pitch, .. } => {
                n.pitch = *pitch;
                n.spelling = None;
            }
            Op::Reassign { who, .. } => {
                apply_who(&mut n.effector, who);
                if let Some(p) = who.player {
                    n.player = p as i32;
                }
            }
            Op::Retag { tags, .. } => n.tags = tags.clone(),
            Op::Spell { spelling, .. } => n.spelling = Some(spelling.clone()),
            Op::Add { .. } => unreachable!(),
        }
        t0 = t0.min(n.onset);
        t1 = t1.max(n.offset);
        touched.push(id.clone());
        if !tracks.contains(&name) {
            tracks.push(name);
        }
    }
    if touched.is_empty() {
        bail!("patch has no ops");
    }
    Ok(Applied {
        notes: touched,
        tracks,
        window: (t0, t1),
    })
}

/// Hard playability violations on the touched keyboard tracks (the gate's playability measure).
pub fn violation_count(piece: &Piece, tracks: &[String]) -> Result<usize> {
    let mut total = 0;
    for tr in &piece.tracks {
        if tracks.contains(&tr.name) && preset::load(&tr.preset)?.family == Family::Keyboard {
            total += PianoDecoder::new(&tr.preset)?.hand_violations(tr).len();
        }
    }
    Ok(total)
}

/// Audio match of a track's render against its stem inside [t0, t1] (1 s padding).
pub fn local_match(track: &Track, stem_cells: &Array2, t0: f64, t1: f64, rs: &mut dyn Resynth) -> Result<f64> {
    let (a, b) = ((t0 - 1.0).max(0.0), t1 + 1.0);
    let rc = rs.render_cells(track, a, Some(b))?;
    let (ia, ib) = ((a / CELL) as usize, (b / CELL) as usize);
    let sc = verify::rows(stem_cells, ia, ib);
    let rc = verify::rows(&rc, 0, sc.rows);
    Ok(verify::compare(&sc, &rc).f1)
}

/// Apply a patch through the gate. `stem_cells` maps track name -> cached stem cells.
/// Returns the log record and, if accepted, the new piece.
pub fn gate(
    before: &Piece,
    patch: &Patch,
    stem_cells: &BTreeMap<String, Array2>,
    rs: &mut dyn Resynth,
) -> Result<(Value, Option<Piece>)> {
    let mut after = before.clone();
    let Applied {
        tracks,
        window: (t0, t1),
        ..
    } = apply(&mut after, &patch.ops)?;
    let notation_only = patch.ops.iter().all(|o| o.notation_only());
    let (vb, va) = (violation_count(before, &tracks)?, violation_count(&after, &tracks)?);
    let mut rec = json!({"reason": patch.reason, "ops": patch.ops, "window": [crate::play::round_to(t0, 3), crate::play::round_to(t1, 3)],
                         "violations": [vb, va]});
    let accepted = if notation_only {
        rec["audio_match"] = Value::Null;
        va <= vb
    } else {
        let (mut mb, mut ma) = (vec![], vec![]);
        for name in &tracks {
            let cells = stem_cells
                .get(name)
                .ok_or_else(|| anyhow!("no stem cells for {name}: run `m2s verify` first"))?;
            mb.push(local_match(before.track(name).unwrap(), cells, t0, t1, rs)?);
            ma.push(local_match(after.track(name).unwrap(), cells, t0, t1, rs)?);
        }
        let (mb, ma) = (
            mb.iter().sum::<f64>() / mb.len() as f64,
            ma.iter().sum::<f64>() / ma.len() as f64,
        );
        rec["audio_match"] = json!([crate::play::round_to(mb, 4), crate::play::round_to(ma, 4)]);
        let gain = ma - mb;
        (gain > MIN_GAIN && va <= vb) || gain > DELTA || (gain >= -EPS && va < vb)
    };
    rec["accepted"] = json!(accepted);
    Ok((rec, if accepted { Some(after) } else { None }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_json_matches_python_format() {
        let p: Patch = serde_json::from_str(
            r#"{"reason":"r","ops":[
            {"op":"add","track":"piano","onset":1.0,"offset":1.5,"pitch":61,"velocity":70,"hand":"R"},
            {"op":"delete","id":"pno-000001"},
            {"op":"retime","id":"pno-000002","onset":2.0},
            {"op":"reassign","id":"pno-000003","hand":"L"},
            {"op":"spell","id":"pno-000004","spelling":"B#3"}]}"#,
        )
        .unwrap();
        assert_eq!(p.ops.len(), 5);
        assert!(!p.ops[0].notation_only());
        assert!(p.ops[3].notation_only());
    }
}
