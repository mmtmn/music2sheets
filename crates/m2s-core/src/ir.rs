//! The shared note format. JSON-compatible with the Python worker in both directions:
//! field names and defaults match `m2s/ir.py`, and nothing extra is written.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Hand {
    L,
    R,
}

/// Who plays a note. Only the fields that apply to the instrument are set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Effector {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand: Option<Hand>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finger: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub string: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fret: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub player: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limb: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_type: Option<String>,
    /// anything else a stage stored, kept as is
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn d_velocity() -> i32 {
    80
}
fn d_one() -> f64 {
    1.0
}
fn d_true() -> bool {
    true
}
fn d_players() -> u32 {
    1
}
fn d_mix() -> String {
    "mix".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    pub onset: f64,
    pub offset: f64,
    pub pitch: i32,
    #[serde(default = "d_velocity")]
    pub velocity: i32,
    #[serde(default = "d_one")]
    pub confidence: f64,
    #[serde(default)]
    pub cents: f64,
    #[serde(default = "d_true")]
    pub keep: bool,
    /// which models saw this note, and how strongly
    #[serde(default)]
    pub support: BTreeMap<String, f64>,
    #[serde(default)]
    pub effector: Effector,
    #[serde(default)]
    pub player: i32,
    #[serde(default)]
    pub tags: Vec<String>,
    /// pitch curve: (seconds, cents relative to `pitch`)
    #[serde(default)]
    pub bend: Vec<[f64; 2]>,
    // score time, filled by notation
    #[serde(default)]
    pub measure: Option<i64>,
    #[serde(default)]
    pub beat: Option<f64>,
    #[serde(default)]
    pub qdur: Option<f64>,
    #[serde(default)]
    pub voice: Option<i64>,
    #[serde(default)]
    pub staff: Option<i64>,
    #[serde(default)]
    pub spelling: Option<String>,
    #[serde(default)]
    pub provenance: String,
}

impl Note {
    pub fn new(id: impl Into<String>, onset: f64, offset: f64, pitch: i32, velocity: i32) -> Self {
        Note {
            id: id.into(),
            onset,
            offset,
            pitch,
            velocity,
            confidence: 1.0,
            cents: 0.0,
            keep: true,
            support: BTreeMap::new(),
            effector: Effector::default(),
            player: 0,
            tags: vec![],
            bend: vec![],
            measure: None,
            beat: None,
            qdur: None,
            voice: None,
            staff: None,
            spelling: None,
            provenance: String::new(),
        }
    }

    pub fn duration(&self) -> f64 {
        self.offset - self.onset
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub name: String,
    pub preset: String,
    #[serde(default)]
    pub notes: Vec<Note>,
    /// sustain pedal intervals (seconds), piano only
    #[serde(default)]
    pub pedal: Vec<[f64; 2]>,
    #[serde(default = "d_mix")]
    pub stem: String,
    #[serde(default = "d_players")]
    pub players: u32,
}

impl Track {
    pub fn new(name: impl Into<String>, preset: impl Into<String>, notes: Vec<Note>) -> Self {
        Track {
            name: name.into(),
            preset: preset.into(),
            notes,
            pedal: vec![],
            stem: "mix".into(),
            players: 1,
        }
    }

    pub fn kept(&self) -> impl Iterator<Item = &Note> {
        self.notes.iter().filter(|n| n.keep)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    #[serde(default)]
    pub meta: serde_json::Map<String, Value>,
    #[serde(default)]
    pub tracks: Vec<Track>,
}

impl Piece {
    pub fn load(path: impl AsRef<Path>) -> Result<Piece> {
        let p = path.as_ref();
        let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", p.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let p = path.as_ref();
        std::fs::write(p, serde_json::to_string_pretty(self)?).with_context(|| format!("writing {}", p.display()))
    }

    pub fn track(&self, name: &str) -> Option<&Track> {
        self.tracks.iter().find(|t| t.name == name)
    }

    pub fn track_mut(&mut self, name: &str) -> Option<&mut Track> {
        self.tracks.iter_mut().find(|t| t.name == name)
    }

    /// (track index, note index) for a note id
    pub fn find(&self, id: &str) -> Option<(usize, usize)> {
        for (ti, t) in self.tracks.iter().enumerate() {
            if let Some(ni) = t.notes.iter().position(|n| n.id == id) {
                return Some((ti, ni));
            }
        }
        None
    }
}
