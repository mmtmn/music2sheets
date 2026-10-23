//! Drum kit: limb assignment (two hands, two feet).
//!
//! Feet take the kick and hi-hat pedal (fast kicks alternate feet with a double pedal). Hands
//! take everything else: the cymbal hand leads, the other hand takes the snare when both play,
//! and fast runs alternate hands. Hits faster than one hand can repeat, or more than four at
//! once, are violations. Faithful port of `m2s/play/drums.py`.

use std::collections::BTreeMap;

use anyhow::Result;
use serde_json::json;

use super::{make_slices, Violation};
use crate::ir::{Effector, Track};
use crate::preset::{self, Limbs};
use crate::trace::Trace;

const CYMBALS: [i32; 8] = [42, 46, 49, 51, 52, 55, 57, 59];

pub struct DrumDecoder {
    lim: Limbs,
    limbs_for: BTreeMap<i32, Vec<String>>,
}

impl DrumDecoder {
    pub fn new(name: &str) -> Result<Self> {
        let p = preset::load(name)?;
        let lim = p
            .limbs
            .clone()
            .ok_or_else(|| anyhow::anyhow!("preset {name} has no limbs"))?;
        let limbs_for = p
            .kit
            .as_ref()
            .map(|k| k.values().map(|v| (v.midi, v.limbs.clone())).collect())
            .unwrap_or_default();
        Ok(DrumDecoder { lim, limbs_for })
    }

    fn is_foot(&self, pitch: i32) -> bool {
        match self.limbs_for.get(&pitch) {
            Some(l) => l.iter().all(|x| x == "RF" || x == "LF"),
            None => false,
        }
    }

    pub fn decode(&self, track: &mut Track, trace: &mut Trace) -> Result<Vec<Violation>> {
        for n in track.notes.iter_mut() {
            n.keep = n.confidence >= 0.5;
        }
        let hits: Vec<usize> = (0..track.notes.len()).filter(|&i| track.notes[i].keep).collect();
        let slices = make_slices(&track.notes, &hits, 0.025);
        let min_gap = 1.0 / self.lim.single_stroke_rate;
        let mut last: BTreeMap<&str, f64> = [("RH", -9.0), ("LH", -9.0), ("RF", -9.0), ("LF", -9.0)]
            .into_iter()
            .collect();
        let mut last_kick_foot = "LF";
        let mut out = vec![];
        let notes = &mut track.notes;
        for sl in slices {
            let t = notes[sl[0]].onset;
            let feet: Vec<usize> = sl.iter().copied().filter(|&i| self.is_foot(notes[i].pitch)).collect();
            let hands: Vec<usize> = sl.iter().copied().filter(|i| !feet.contains(i)).collect();
            let mut used: Vec<&str> = vec![];
            for &i in &feet {
                let pitch = notes[i].pitch;
                let mut limb = if pitch == 44
                    || (self.lim.double_kick && t - last["RF"] < 1.5 * min_gap && last_kick_foot == "RF")
                {
                    "LF"
                } else {
                    "RF"
                };
                if used.contains(&limb) {
                    limb = if limb == "RF" { "LF" } else { "RF" };
                }
                if pitch != 44 {
                    last_kick_foot = limb;
                }
                notes[i].effector = Effector {
                    limb: Some(limb.into()),
                    ..Default::default()
                };
                used.push(limb);
                last.insert(limb, t);
            }
            // hands: try both orders, keep the one with fewer too-fast repeats and no crossing
            let avail: Vec<&str> = ["RH", "LH"].into_iter().filter(|h| !used.contains(h)).collect();
            let mut ordered = hands.clone();
            ordered.sort_by_key(|&i| (!CYMBALS.contains(&notes[i].pitch), notes[i].pitch));
            let k = hands.len().min(avail.len());
            let mut best: Option<Vec<&str>> = None;
            let mut best_cost = 1e9;
            for perm in permutations(&avail, k) {
                let mut cost = 0.0;
                for (&i, &h) in ordered.iter().zip(perm.iter()) {
                    let p = notes[i].pitch;
                    if t - last[h] < min_gap {
                        cost += 5.0;
                    }
                    if CYMBALS.contains(&p) && h == "LH" {
                        cost += 0.5;
                    }
                    if p == 38 && h == "RH" && hands.len() > 1 {
                        cost += 0.5;
                    }
                    cost -= 0.1 * (t - last[h]).min(1.0);
                }
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(perm);
                }
            }
            let best = best.unwrap_or_default();
            for (&i, &h) in ordered.iter().zip(best.iter()) {
                notes[i].effector = Effector {
                    limb: Some(h.into()),
                    ..Default::default()
                };
                if t - last[h] < min_gap {
                    out.push(Violation::new(
                        "speed",
                        t,
                        format!("{h} repeats in {:.0} ms", (t - last[h]) * 1000.0),
                        vec![notes[i].id.clone()],
                        h,
                    ));
                }
                last.insert(h, t);
            }
            for &i in &ordered[best.len()..] {
                notes[i].effector = Effector {
                    limb: Some("?".into()),
                    ..Default::default()
                };
            }
            if sl.len() > self.lim.max_simultaneous || ordered.len() > best.len() {
                out.push(Violation::new(
                    "count",
                    t,
                    format!("{} hits at once", sl.len()),
                    sl.iter().map(|&i| notes[i].id.clone()).collect(),
                    "",
                ));
            }
            trace.with("limbs", || json!({"t": t, "hits": sl.iter().map(|&i| (notes[i].pitch, notes[i].effector.limb.clone())).collect::<Vec<_>>()}));
        }
        Ok(out)
    }
}

/// itertools.permutations order
fn permutations<'a>(items: &[&'a str], k: usize) -> Vec<Vec<&'a str>> {
    fn rec<'a>(items: &[&'a str], k: usize, used: &mut Vec<bool>, cur: &mut Vec<&'a str>, out: &mut Vec<Vec<&'a str>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in 0..items.len() {
            if !used[i] {
                used[i] = true;
                cur.push(items[i]);
                rec(items, k, used, cur, out);
                cur.pop();
                used[i] = false;
            }
        }
    }
    let mut out = vec![];
    rec(items, k, &mut vec![false; items.len()], &mut vec![], &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Note;

    #[test]
    fn limbs() {
        let hits = [(0.0, 36), (0.0, 42), (0.5, 38), (0.5, 42), (1.0, 36)];
        let notes = hits
            .iter()
            .enumerate()
            .map(|(i, &(t, p))| Note::new(format!("h{i}"), t, t + 0.1, p, 90))
            .collect();
        let mut tr = Track::new("drums", "drums", notes);
        DrumDecoder::new("drums")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        let limb = |t: f64, p: i32| {
            tr.notes
                .iter()
                .find(|n| n.onset == t && n.pitch == p)
                .unwrap()
                .effector
                .limb
                .clone()
                .unwrap()
        };
        assert!(["RF", "LF"].contains(&limb(0.0, 36).as_str()));
        assert_eq!(limb(0.0, 42), "RH");
        assert_eq!(limb(0.5, 38), "LH");
    }
}
