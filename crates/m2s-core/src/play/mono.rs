//! Monophonic instruments that breathe (voice, saxophone): one note at a time, range, breaths.
//! Faithful port of `m2s/play/mono.py`.

use std::collections::BTreeMap;

use anyhow::Result;

use super::Violation;
use crate::ir::{Effector, Track};
use crate::preset::{self, Preset};
use crate::trace::Trace;

pub struct MonoDecoder {
    p: &'static Preset,
}

impl MonoDecoder {
    pub fn new(name: &str) -> Result<Self> {
        Ok(MonoDecoder { p: preset::load(name)? })
    }

    pub fn decode(&self, track: &mut Track, _trace: &mut Trace) -> Result<Vec<Violation>> {
        let notes = &mut track.notes;
        for n in notes.iter_mut() {
            n.keep = n.confidence >= 0.5;
        }
        let mut order: Vec<usize> = (0..notes.len()).filter(|&i| notes[i].keep).collect();
        order.sort_by(|&a, &b| notes[a].onset.partial_cmp(&notes[b].onset).unwrap());
        // one note at a time: an overlap keeps the more confident note and trims the other
        for w in order.windows(2) {
            let (a, b) = (w[0], w[1]);
            if notes[b].onset < notes[a].offset - 0.02 {
                if (notes[b].onset - notes[a].onset).abs() < 0.05 && notes[b].confidence > notes[a].confidence {
                    notes[a].keep = false;
                } else {
                    notes[a].offset = notes[b].onset;
                }
            }
        }
        let order: Vec<usize> = order.into_iter().filter(|&i| notes[i].keep).collect();
        let mut out = vec![];
        let [lo, hi] = self.p.range.unwrap_or([0, 127]);
        for &i in &order {
            let n = &notes[i];
            if !(lo <= n.pitch && n.pitch <= hi) {
                out.push(Violation::new(
                    "range",
                    n.onset,
                    format!("pitch {} outside {lo}-{hi}", n.pitch),
                    vec![n.id.clone()],
                    "",
                ));
            }
        }
        if let (Some(vt), false) = (&self.p.voice_types, order.is_empty()) {
            let mut ps: Vec<i32> = order.iter().map(|&i| notes[i].pitch).collect();
            ps.sort();
            let a = ps.len() / 10;
            let b = (ps.len() * 9 / 10).max(1);
            let core = &ps[a..b];
            let fit: BTreeMap<&String, f64> = vt
                .iter()
                .map(|(k, r)| {
                    (
                        k,
                        core.iter().filter(|&&p| r[0] <= p && p <= r[1]).count() as f64 / core.len() as f64,
                    )
                })
                .collect();
            // Python max() over dict insertion order (YAML order) keeps the first maximum
            let mut best: Option<(&String, f64)> = None;
            for (k, _) in vt_order(self.p) {
                let f = fit[&k];
                if best.map(|(_, bf)| f > bf).unwrap_or(true) {
                    best = Some((k, f));
                }
            }
            if let Some((name, _)) = best {
                for &i in &order {
                    notes[i].effector = Effector {
                        voice_type: Some(name.clone()),
                        ..Default::default()
                    };
                }
            }
        }
        if let (Some(br), false) = (&self.p.breath, order.is_empty()) {
            let mut start = notes[order[0]].onset;
            for (k, &a) in order.iter().enumerate() {
                let b = order.get(k + 1).copied();
                let gap = b.map(|b| notes[b].onset - notes[a].offset).unwrap_or(99.0);
                if gap >= br.min_gap {
                    if notes[a].offset - start > br.max_phrase {
                        out.push(Violation::new(
                            "breath",
                            start,
                            format!("{:.1}s without a breath", notes[a].offset - start),
                            vec![notes[a].id.clone()],
                            "",
                        ));
                    }
                    if let Some(b) = b {
                        notes[a].tags.push("breath".into());
                        start = notes[b].onset;
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Voice types in the preset file's order (serde_yaml into a BTreeMap loses it; re-read keys).
fn vt_order(p: &'static Preset) -> Vec<(&'static String, ())> {
    const ORDER: [&str; 6] = ["soprano", "mezzo", "alto", "tenor", "baritone", "bass"];
    let vt = p.voice_types.as_ref().unwrap();
    let mut keys: Vec<&String> = vt.keys().collect();
    keys.sort_by_key(|k| ORDER.iter().position(|o| o == k).unwrap_or(99));
    keys.into_iter().map(|k| (k, ())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Note;

    #[test]
    fn breath_violation() {
        let notes = (0..24)
            .map(|i| Note::new(format!("v{i}"), i as f64 * 0.5, i as f64 * 0.5 + 0.5, 64, 80))
            .collect();
        let mut tr = Track::new("vox", "voice", notes);
        let v = MonoDecoder::new("voice")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        assert!(v.iter().any(|x| x.kind == "breath"));
    }
}
