//! Instrument presets: one YAML file per instrument in `presets/` at the repo root,
//! shared with the Python worker and embedded into the binary at build time.
//! `M2S_PRESETS=/some/dir` loads from a directory instead (for experiments).

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Result};
use serde::Deserialize;

const EMBEDDED: &[(&str, &str)] = &[
    ("piano", include_str!("../../../presets/piano.yaml")),
    ("guitar", include_str!("../../../presets/guitar.yaml")),
    ("electric_guitar", include_str!("../../../presets/electric_guitar.yaml")),
    ("bass", include_str!("../../../presets/bass.yaml")),
    ("violin", include_str!("../../../presets/violin.yaml")),
    ("sax_alto", include_str!("../../../presets/sax_alto.yaml")),
    ("sax_tenor", include_str!("../../../presets/sax_tenor.yaml")),
    ("voice", include_str!("../../../presets/voice.yaml")),
    ("drums", include_str!("../../../presets/drums.yaml")),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Keyboard,
    Fretted,
    Wind,
    Drums,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Hands {
    pub max_notes: usize,
    pub max_span: i32,
    pub comfort_span: i32,
    pub max_speed: f64,
    pub comfort_speed: f64,
    pub chord_window: f64,
    pub middle: i32,
    pub crossing_margin: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Fingering {
    #[serde(default = "yes")]
    pub enabled: bool,
    /// "1-2" -> [min_practical, min_comfortable, min_relaxed, max_relaxed, max_comfortable, max_practical]
    pub spans: BTreeMap<String, [i32; 6]>,
    pub shift_time: f64,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct Fretting {
    pub max_stretch: i32,
    pub max_stretch_high: i32,
    pub high_position: i32,
    pub shift_cost: f64,
    pub shift_speed: f64,
    pub open_string_bonus: f64,
    pub string_skip_cost: f64,
    #[serde(default = "one")]
    pub players: u32,
}

fn one() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct Breath {
    pub max_phrase: f64,
    pub min_gap: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Limbs {
    pub max_simultaneous: usize,
    pub single_stroke_rate: f64,
    #[serde(default)]
    pub double_kick: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KitPiece {
    pub midi: i32,
    pub limbs: Vec<String>,
    pub display: String,
    #[serde(default = "normal")]
    pub notehead: String,
}

fn normal() -> String {
    "normal".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct Preset {
    pub name: String,
    pub family: Family,
    pub display: String,
    #[serde(default)]
    pub range: Option<[i32; 2]>,
    #[serde(default)]
    pub transposition: i32,
    #[serde(default)]
    pub clef: Option<String>,
    #[serde(default)]
    pub gm_program: u8,
    #[serde(default)]
    pub staves: Option<String>,
    #[serde(default)]
    pub hands: Option<Hands>,
    #[serde(default)]
    pub fingering: Option<Fingering>,
    #[serde(default)]
    pub tuning: Option<Vec<i32>>,
    #[serde(default)]
    pub frets: Option<i32>,
    #[serde(default)]
    pub tab: bool,
    #[serde(default)]
    pub fretting: Option<Fretting>,
    #[serde(default)]
    pub adjacent_strings_only: bool,
    #[serde(default)]
    pub max_simultaneous: Option<usize>,
    #[serde(default)]
    pub techniques: Vec<String>,
    #[serde(default)]
    pub breath: Option<Breath>,
    #[serde(default)]
    pub voice_types: Option<BTreeMap<String, [i32; 2]>>,
    #[serde(default)]
    pub limbs: Option<Limbs>,
    #[serde(default)]
    pub kit: Option<BTreeMap<String, KitPiece>>,
    #[serde(default)]
    pub channel: Option<u8>,
}

impl Preset {
    pub fn hands(&self) -> Result<&Hands> {
        self.hands
            .as_ref()
            .ok_or_else(|| anyhow!("preset {} has no hands", self.name))
    }
    pub fn fretting(&self) -> Result<&Fretting> {
        self.fretting
            .as_ref()
            .ok_or_else(|| anyhow!("preset {} has no fretting", self.name))
    }
}

fn cache() -> &'static Mutex<HashMap<String, &'static Preset>> {
    static C: OnceLock<Mutex<HashMap<String, &'static Preset>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load a preset by name (cached for the life of the process).
pub fn load(name: &str) -> Result<&'static Preset> {
    let mut c = cache().lock().unwrap();
    if let Some(p) = c.get(name) {
        return Ok(p);
    }
    let text = match std::env::var("M2S_PRESETS") {
        Ok(dir) => std::fs::read_to_string(std::path::Path::new(&dir).join(format!("{name}.yaml")))?,
        Err(_) => EMBEDDED
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| t.to_string())
            .ok_or_else(|| anyhow!("unknown preset {name:?}; available: {}", available().join(", ")))?,
    };
    let p: Preset = serde_yaml::from_str(&text).map_err(|e| anyhow!("preset {name}: {e}"))?;
    let leaked: &'static Preset = Box::leak(Box::new(p));
    c.insert(name.to_string(), leaked);
    Ok(leaked)
}

pub fn available() -> Vec<&'static str> {
    EMBEDDED.iter().map(|(n, _)| *n).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn all_presets_parse() {
        for name in super::available() {
            let p = super::load(name).unwrap();
            assert_eq!(p.name, name);
        }
    }
}
