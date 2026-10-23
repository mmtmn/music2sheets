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
    #[serde(default = "one")]
    pub voice: i32,
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

fn one() -> i32 {
    1
}

pub const HARD: [&str; 6] = ["span", "count", "speed", "finger", "stretch", "shift"];

// ------------------------------------------------------------------------------------------
// Score-level metric
//
// Same five aspects as MV2H (McLeod & Steedman 2018) -- pitch, voice, meter, value, harmony --
// but computed on the score-to-audio alignment that `m2s eval` already has, instead of MV2H's
// own note-list alignment (which takes over an hour on a long movement). Each part is in [0, 1]:
//   pitch    note F1, onsets within 100 ms
//   voice    F1 of next-note links within each voice, over links whose two notes are matched
//   meter    mean of bar-line F1 and beat F1 in time (100 ms)
//   value    share of matched notes whose written value equals the reference's
//   harmony  key agreement with MIREX weights (1 / fifth 0.5 / relative 0.3 / parallel 0.2)
// ------------------------------------------------------------------------------------------

/// Our side of the comparison: notated notes (from scored.json) plus bar and beat times.
pub struct Ours<'a> {
    pub notes: Vec<&'a Note>,
    /// voice key per note: (track, staff, voice)
    pub voice_of: Vec<(usize, i64, i64)>,
    pub bar_times: Vec<f64>,
    pub beat_times: Vec<f64>,
    pub key: String,
}

pub struct RefMeta {
    pub bar_qons: Vec<f64>,
    pub beat_q: f64,
    pub key: String,
}

fn f1(tp: usize, n_est: usize, n_ref: usize) -> f64 {
    let p = if n_est == 0 { 0.0 } else { tp as f64 / n_est as f64 };
    let r = if n_ref == 0 { 0.0 } else { tp as f64 / n_ref as f64 };
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

/// One-to-one matches between two sorted time lists within `tol` (greedy is maximum on a line).
fn match_times(a: &[f64], b: &[f64], tol: f64) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if (a[i] - b[j]).abs() <= tol {
            n += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    n
}

/// Reference score position (quarters) -> recording time, through the aligned notes.
fn qon_to_time(r: &[RefNote]) -> impl Fn(f64) -> f64 {
    use std::collections::BTreeMap;
    let mut by_q: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for x in r {
        by_q.entry((x.qon * 48.0).round() as i64).or_default().push(x.on);
    }
    let mut pts: Vec<(f64, f64)> = by_q
        .into_iter()
        .map(|(q, mut v)| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (q as f64 / 48.0, v[v.len() / 2])
        })
        .collect();
    // keep it monotone in time
    let mut last = f64::NEG_INFINITY;
    pts.retain(|&(_, t)| {
        let ok = t >= last;
        if ok {
            last = t;
        }
        ok
    });
    move |q: f64| {
        let n = pts.len();
        if n == 0 {
            return q;
        }
        if n == 1 {
            return pts[0].1;
        }
        let seg = |i: usize, j: usize| {
            let (q0, t0) = pts[i];
            let (q1, t1) = pts[j];
            t0 + (q - q0) * (t1 - t0) / (q1 - q0).max(1e-9)
        };
        if q <= pts[0].0 {
            return seg(0, 1);
        }
        if q >= pts[n - 1].0 {
            return seg(n - 2, n - 1);
        }
        let k = pts.partition_point(|p| p.0 <= q).max(1);
        seg(k - 1, k)
    }
}

fn key_pc(name: &str) -> Option<(i32, bool)> {
    let mut it = name.split_whitespace();
    let tonic = it.next()?;
    let minor = it.next().map(|m| m.eq_ignore_ascii_case("minor")).unwrap_or(false);
    let mut chars = tonic.chars();
    let base = match chars.next()?.to_ascii_uppercase() {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let acc: i32 = chars
        .map(|c| match c {
            '#' => 1,
            '-' | 'b' => -1,
            _ => 0,
        })
        .sum();
    Some(((base + acc).rem_euclid(12), minor))
}

/// MIREX key score of `est` against `refk`.
pub fn key_score(est: &str, refk: &str) -> f64 {
    let (Some((e, em)), Some((r, rm))) = (key_pc(est), key_pc(refk)) else {
        return 0.0;
    };
    if e == r && em == rm {
        1.0
    } else if em == rm && ((e - r).rem_euclid(12) == 7 || (r - e).rem_euclid(12) == 7) {
        0.5
    } else if em != rm && ((!rm && e == (r + 9) % 12) || (rm && e == (r + 3) % 12)) {
        0.3
    } else if e == r {
        0.2
    } else {
        0.0
    }
}

pub fn score_level(r: &[RefNote], meta: &RefMeta, ours: &Ours) -> Value {
    use std::collections::{BTreeMap, HashMap, HashSet};
    let e = &ours.notes;
    let pairs = match_notes(r, e, 0.1);
    let pitch = f1(pairs.len(), e.len(), r.len());
    let r2e: HashMap<usize, usize> = pairs.iter().copied().collect();

    // voice links: consecutive onsets within each voice, chord to chord
    fn links(groups: BTreeMap<(i64, i64, i64), Vec<(f64, usize)>>) -> HashSet<(usize, usize)> {
        let mut out = HashSet::new();
        for (_, mut v) in groups {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut chords: Vec<Vec<usize>> = vec![];
            let mut last = f64::NEG_INFINITY;
            for (t, i) in v {
                if chords.is_empty() || (t - last).abs() > 1e-6 {
                    chords.push(vec![]);
                    last = t;
                }
                chords.last_mut().unwrap().push(i);
            }
            for w in chords.windows(2) {
                for &a in &w[0] {
                    for &b in &w[1] {
                        out.insert((a, b));
                    }
                }
            }
        }
        out
    }
    let mut rg: BTreeMap<(i64, i64, i64), Vec<(f64, usize)>> = BTreeMap::new();
    for (i, x) in r.iter().enumerate() {
        rg.entry((0, x.staff as i64, x.voice as i64))
            .or_default()
            .push((x.qon, i));
    }
    let mut eg: BTreeMap<(i64, i64, i64), Vec<(f64, usize)>> = BTreeMap::new();
    for (j, n) in e.iter().enumerate() {
        let (t, s, v) = ours.voice_of[j];
        if n.value == Some(0.0) {
            continue; // grace notes sit outside the voice's rhythm
        }
        eg.entry((t as i64, s, v))
            .or_default()
            .push((n.beat.unwrap_or(n.onset), j));
    }
    let (rl, el) = (links(rg), links(eg));
    let matched_e: HashSet<usize> = pairs.iter().map(|p| p.1).collect();
    let ref_links: Vec<(usize, usize)> = rl
        .iter()
        .copied()
        .filter(|(a, b)| r2e.contains_key(a) && r2e.contains_key(b))
        .collect();
    let est_links = el
        .iter()
        .filter(|(a, b)| matched_e.contains(a) && matched_e.contains(b))
        .count();
    let tp = ref_links.iter().filter(|(a, b)| el.contains(&(r2e[a], r2e[b]))).count();
    let voice = f1(tp, est_links, ref_links.len());

    // meter: bar lines and beats in time
    let to_t = qon_to_time(r);
    let mut rbars: Vec<f64> = meta.bar_qons.iter().map(|&q| to_t(q)).collect();
    rbars.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut rbeats = vec![];
    for (k, &qa) in meta.bar_qons.iter().enumerate() {
        let qb = meta.bar_qons.get(k + 1).copied().unwrap_or(qa + meta.beat_q * 4.0);
        let mut q = qa;
        while q < qb - 1e-9 {
            rbeats.push(to_t(q));
            q += meta.beat_q.max(0.25);
        }
    }
    rbeats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bar_f1 = f1(
        match_times(&ours.bar_times, &rbars, 0.1),
        ours.bar_times.len(),
        rbars.len(),
    );
    let beat_f1 = f1(
        match_times(&ours.beat_times, &rbeats, 0.1),
        ours.beat_times.len(),
        rbeats.len(),
    );
    let meter = (bar_f1 + beat_f1) / 2.0;

    // value: written durations of matched notes
    let vals: Vec<bool> = pairs
        .iter()
        .filter_map(|&(i, j)| e[j].value.filter(|&v| v > 0.0).map(|v| (v - r[i].qdur).abs() < 1e-3))
        .collect();
    let value = if vals.is_empty() {
        0.0
    } else {
        vals.iter().filter(|&&b| b).count() as f64 / vals.len() as f64
    };

    let harmony = key_score(&ours.key, &meta.key);
    let score = (pitch + voice + meter + value + harmony) / 5.0;
    let rd = |x: f64| crate::play::round_to(x, 4);
    json!({"score": rd(score), "pitch": rd(pitch), "voice": rd(voice), "meter": rd(meter), "value": rd(value),
           "harmony": rd(harmony), "bar_f1": rd(bar_f1), "beat_f1": rd(beat_f1),
           "counts": {"matched": pairs.len(), "ref_links": ref_links.len(), "est_links": est_links,
                      "our_bars": ours.bar_times.len(), "ref_bars": rbars.len()}})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirex_key_weights() {
        assert_eq!(key_score("C# minor", "C# minor"), 1.0);
        assert_eq!(key_score("G# minor", "C# minor"), 0.5);
        assert_eq!(key_score("E major", "C# minor"), 0.3);
        assert_eq!(key_score("C# major", "C# minor"), 0.2);
        assert_eq!(key_score("B- major", "A# major"), 1.0);
        assert_eq!(key_score("F major", "C# minor"), 0.0);
    }

    #[test]
    fn perfect_transcription_scores_one() {
        // two bars of 4/4 at 1 s per quarter: a scale in one voice
        let r: Vec<RefNote> = (0..8)
            .map(|i| RefNote {
                on: i as f64,
                off: i as f64 + 1.0,
                pitch: 60 + i,
                staff: 1,
                voice: 1,
                qon: i as f64,
                qdur: 1.0,
            })
            .collect();
        let notes: Vec<Note> = (0..8)
            .map(|i| {
                let mut n = Note::new(format!("n{i}"), i as f64, i as f64 + 1.0, 60 + i, 80);
                n.beat = Some(i as f64);
                n.value = Some(1.0);
                n
            })
            .collect();
        let ours = Ours {
            notes: notes.iter().collect(),
            voice_of: vec![(0, 1, 1); 8],
            bar_times: vec![0.0, 4.0],
            beat_times: (0..8).map(|i| i as f64).collect(),
            key: "C major".into(),
        };
        let meta = RefMeta {
            bar_qons: vec![0.0, 4.0],
            beat_q: 1.0,
            key: "C major".into(),
        };
        let s = score_level(&r, &meta, &ours);
        assert_eq!(s["score"].as_f64().unwrap(), 1.0, "{s}");
    }

    #[test]
    fn greedy_is_maximum_on_a_line() {
        // ref at 0.00 and 0.09, est at 0.05 and 0.14 (tol 0.05): both can match
        let r = vec![
            RefNote {
                on: 0.0,
                off: 0.1,
                pitch: 60,
                staff: 1,
                voice: 1,
                qon: 0.0,
                qdur: 0.0,
            },
            RefNote {
                on: 0.09,
                off: 0.2,
                pitch: 60,
                staff: 1,
                voice: 1,
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
