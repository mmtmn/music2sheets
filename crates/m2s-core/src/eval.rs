//! Evaluation metrics against an aligned reference (the alignment itself runs in the Python
//! worker): note precision/recall/F1 like mir_eval, hand agreement, playability.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::ir::{Effector, Hand, Note, Track};
use crate::play::piano::PianoDecoder;

#[derive(Debug, Clone, Deserialize)]
pub struct RefNote {
    pub on: f64,
    pub off: f64,
    pub pitch: i32,
    pub staff: i32,
    #[serde(default)]
    pub qon: f64,
    #[serde(default)]
    pub qdur: f64,
}

/// Maximum matching between reference and estimated notes: same pitch, onsets within `tol`
/// (mir_eval's rule with offsets ignored). Onset windows on a line make the graph convex, so a
/// greedy left-to-right match per pitch is maximum. Returns (ref index, est index) pairs.
pub fn match_notes(r: &[RefNote], e: &[&Note], tol: f64) -> Vec<(usize, usize)> {
    use std::collections::BTreeMap;
    let mut by_pitch: BTreeMap<i32, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for (i, x) in r.iter().enumerate() {
        by_pitch.entry(x.pitch).or_default().0.push(i);
    }
    for (j, x) in e.iter().enumerate() {
        by_pitch.entry(x.pitch).or_default().1.push(j);
    }
    let round7 = |x: f64| (x * 1e7).round() / 1e7; // mir_eval rounds distances to 7 decimals
    let mut out = vec![];
    for (_, (mut ri, mut ei)) in by_pitch {
        ri.sort_by(|&a, &b| r[a].on.partial_cmp(&r[b].on).unwrap());
        ei.sort_by(|&a, &b| e[a].onset.partial_cmp(&e[b].onset).unwrap());
        let mut j = 0;
        for &i in &ri {
            while j < ei.len() && e[ei[j]].onset < r[i].on && round7(r[i].on - e[ei[j]].onset) > tol {
                j += 1;
            }
            if j < ei.len() && round7((e[ei[j]].onset - r[i].on).abs()) <= tol {
                out.push((i, ei[j]));
                j += 1;
            }
        }
    }
    out
}

pub fn prf(r: &[RefNote], e: &[&Note], tol: f64) -> Value {
    let m = match_notes(r, e, tol).len() as f64;
    let p = if e.is_empty() { 0.0 } else { m / e.len() as f64 };
    let rc = if r.is_empty() { 0.0 } else { m / r.len() as f64 };
    let f = if p + rc > 0.0 { 2.0 * p * rc / (p + rc) } else { 0.0 };
    let rd = |x: f64| crate::play::round_to(x, 4);
    json!({"precision": rd(p), "recall": rd(rc), "f1": rd(f)})
}

/// Share of matched notes (100 ms) whose hand equals the reference staff (1 = right hand).
pub fn hand_agreement(r: &[RefNote], e: &[&Note]) -> Value {
    let pairs: Vec<(usize, usize)> = match_notes(r, e, 0.1)
        .into_iter()
        .filter(|&(_, j)| e[j].effector.hand.is_some())
        .collect();
    let staffs: std::collections::BTreeSet<i32> = pairs.iter().map(|&(i, _)| r[i].staff).collect();
    if pairs.is_empty() || staffs.len() < 2 {
        return json!({"agreement": null, "matched": pairs.len()});
    }
    let ok = pairs
        .iter()
        .filter(|&&(i, j)| (e[j].effector.hand == Some(Hand::R)) == (r[i].staff == 1))
        .count();
    json!({"agreement": crate::play::round_to(ok as f64 / pairs.len() as f64, 4), "matched": pairs.len()})
}

/// Hard violations per 100 notes in the reference itself, reading staff 1 as the right hand.
pub fn reference_playability(r: &[RefNote]) -> anyhow::Result<Option<f64>> {
    if !r.iter().all(|x| x.staff == 1 || x.staff == 2) {
        return Ok(None);
    }
    let notes: Vec<Note> = r
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let mut n = Note::new(format!("r{i}"), x.on, x.off, x.pitch, 80);
            n.effector = Effector {
                hand: Some(if x.staff == 1 { Hand::R } else { Hand::L }),
                ..Default::default()
            };
            n
        })
        .collect();
    let mut tr = Track::new("ref", "piano", notes);
    let dec = PianoDecoder::new("piano")?;
    let mut v = dec.hand_violations(&tr);
    v.extend(dec.finger(&mut tr)?);
    let hard = v.iter().filter(|x| HARD.contains(&x.kind.as_str())).count();
    Ok(Some(crate::play::round_to(
        100.0 * hard as f64 / r.len().max(1) as f64,
        2,
    )))
}

pub const HARD: [&str; 6] = ["span", "count", "speed", "finger", "stretch", "shift"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_maximum_on_a_line() {
        // ref at 0.00 and 0.09, est at 0.05 and 0.14 (tol 0.05): both can match
        let r = vec![
            RefNote {
                on: 0.0,
                off: 0.1,
                pitch: 60,
                staff: 1,
                qon: 0.0,
                qdur: 0.0,
            },
            RefNote {
                on: 0.09,
                off: 0.2,
                pitch: 60,
                staff: 1,
                qon: 0.0,
                qdur: 0.0,
            },
        ];
        let e1 = Note::new("a", 0.05, 0.1, 60, 80);
        let e2 = Note::new("b", 0.14, 0.2, 60, 80);
        let m = match_notes(&r, &[&e1, &e2], 0.05);
        assert_eq!(m.len(), 2);
    }
}
