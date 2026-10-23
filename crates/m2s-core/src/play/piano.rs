//! Piano playability decoder.
//!
//! 1. **Hands.** Beam search over time slices: for every candidate note, keep or drop it and
//!    pick the hand. Cost = acoustic evidence (-log p of the keep/drop choice) + kinematic
//!    effort (span, required shift and its speed, crossings, busy fingers, line continuity).
//! 2. **Fingers.** Within each hand, Viterbi over finger tuples with finger-pair spans adapted
//!    from Parncutt et al. (1997).
//!
//! Faithful port of `m2s/play/piano.py`: same costs, same tie-breaking (insertion-ordered
//! buckets, stable sorts, first minimum), so both produce identical assignments.

use std::collections::BTreeMap;
use std::rc::Rc;

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use serde_json::json;

use super::{make_slices, Violation};
use crate::ir::{Effector, Hand, Track};
use crate::preset::{self, Hands};
use crate::trace::Trace;

const BLACK: [i32; 5] = [1, 3, 6, 8, 10];

fn is_black(p: i32) -> bool {
    BLACK.contains(&p.rem_euclid(12))
}

/// Decoder weights, fit on the ASAP dev set (`tools/tune_hands.py`): hand accuracy 96.2% on
/// the tuning pieces, 92.1% held out; a middle-C split gets 80.8%.
#[derive(Debug, Clone)]
pub struct PianoWeights {
    pub acoustic: f64,
    pub span_over_comfort: f64,
    pub span_violation: f64,
    pub count_violation: f64,
    pub leap: f64,
    pub speed: f64,
    pub speed_violation: f64,
    pub stream: f64,
    pub crossing: f64,
    pub crossing_per_semitone: f64,
    pub pitch_prior: f64,
    pub middle: f64,
    pub hold_conflict: f64,
    /// a hand whose fingers still hold keys is less free to take new notes
    pub busy: f64,
    /// same-hand note rate: penalty when one hand strikes again within `rate_window` s
    pub rate: f64,
    pub rate_window: f64,
    /// prefer the idle hand over a wide chord in the busy one
    pub idle: f64,
    /// melodic continuity window (semitones)
    pub stream_window: f64,
    /// candidates above `must_keep` are always kept, below `must_drop` always dropped
    pub must_keep: f64,
    pub must_drop: f64,
    pub beam: usize,
}

impl Default for PianoWeights {
    fn default() -> Self {
        PianoWeights {
            acoustic: 1.0,
            span_over_comfort: 1.2,
            span_violation: 40.0,
            count_violation: 40.0,
            leap: 0.06,
            speed: 0.0225,
            speed_violation: 25.0,
            stream: 0.9,
            crossing: 3.0,
            crossing_per_semitone: 0.6,
            pitch_prior: 0.01,
            middle: 56.0,
            hold_conflict: 0.5,
            busy: 0.6,
            rate: 0.375,
            rate_window: 0.06,
            idle: 0.4,
            stream_window: 9.0,
            must_keep: 0.75,
            must_drop: 0.25,
            beam: 48,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct HandState {
    lo: i32,
    hi: i32,
    t: f64,
    /// keys still held: (pitch, offset)
    held: Vec<(i32, f64)>,
}

struct Node {
    cost: f64,
    lh: Option<Rc<HandState>>,
    rh: Option<Rc<HandState>>,
    parent: Option<Rc<Node>>,
    /// per note in the slice: b'L', b'R' or b'x' (dropped)
    decision: Vec<u8>,
    /// (pitch, hand) of this slice's kept notes
    prev: Vec<(i32, u8)>,
    prev_t: f64,
}

type Key = (Option<(i32, i32)>, Option<(i32, i32)>, Vec<u8>);

fn bucket(h: &Option<Rc<HandState>>) -> Option<(i32, i32)> {
    h.as_ref().map(|h| (h.lo.div_euclid(2), h.hi.div_euclid(2)))
}

impl PianoWeights {
    /// Override fields by name from a JSON object (used by `m2s decode-file --weights` for tuning).
    pub fn with_overrides(mut self, v: &serde_json::Value) -> Result<Self> {
        let Some(obj) = v.as_object() else { return Ok(self) };
        for (k, x) in obj {
            let f = x.as_f64().ok_or_else(|| anyhow!("weight {k} is not a number"))?;
            match k.as_str() {
                "acoustic" => self.acoustic = f,
                "span_over_comfort" => self.span_over_comfort = f,
                "span_violation" => self.span_violation = f,
                "count_violation" => self.count_violation = f,
                "leap" => self.leap = f,
                "speed" => self.speed = f,
                "speed_violation" => self.speed_violation = f,
                "stream" => self.stream = f,
                "crossing" => self.crossing = f,
                "crossing_per_semitone" => self.crossing_per_semitone = f,
                "pitch_prior" => self.pitch_prior = f,
                "middle" => self.middle = f,
                "hold_conflict" => self.hold_conflict = f,
                "busy" => self.busy = f,
                "rate" => self.rate = f,
                "rate_window" => self.rate_window = f,
                "idle" => self.idle = f,
                "stream_window" => self.stream_window = f,
                "must_keep" => self.must_keep = f,
                "must_drop" => self.must_drop = f,
                "beam" => self.beam = f as usize,
                other => return Err(anyhow!("unknown weight {other}")),
            }
        }
        Ok(self)
    }
}

pub struct PianoDecoder {
    h: Hands,
    pub w: PianoWeights,
    fingering: Option<crate::preset::Fingering>,
}

impl PianoDecoder {
    pub fn new(preset_name: &str) -> Result<Self> {
        Self::with_weights(preset_name, PianoWeights::default())
    }

    pub fn with_weights(preset_name: &str, w: PianoWeights) -> Result<Self> {
        let p = preset::load(preset_name)?;
        Ok(PianoDecoder {
            h: p.hands()?.clone(),
            w,
            fingering: p.fingering.clone(),
        })
    }

    /// Cost for one hand to strike `chord` (sorted pitches) at time `t`.
    fn hand_cost(&self, hs: Option<&HandState>, chord: &[i32], t: f64, hand: Hand) -> (f64, HandState) {
        let (h, w) = (&self.h, &self.w);
        let mut cost = 0.0;
        let (lo, hi) = (chord[0], chord[chord.len() - 1]);
        let span = hi - lo;
        if chord.len() > h.max_notes {
            cost += w.count_violation * (chord.len() - h.max_notes) as f64;
        }
        if span > h.max_span {
            cost += w.span_violation + 4.0 * (span - h.max_span) as f64;
        } else if span > h.comfort_span {
            cost += w.span_over_comfort * (span - h.comfort_span) as f64;
        }
        // weak prior: right hand above the middle, left below
        let mid = w.middle;
        let centre = (lo + hi) as f64 / 2.0;
        if hand == Hand::R && centre < mid - 7.0 {
            cost += w.pitch_prior * (mid - 7.0 - centre);
        }
        if hand == Hand::L && centre > mid + 7.0 {
            cost += w.pitch_prior * (centre - mid - 7.0);
        }
        let mut held: Vec<(i32, f64)> = vec![];
        if let Some(hs) = hs {
            let dt = (t - hs.t).max(1e-3);
            let union = hi.max(hs.hi) - lo.min(hs.lo);
            let shift = (union - h.comfort_span).max(0);
            if shift != 0 {
                cost += w.leap * shift as f64;
                let speed = shift as f64 / dt;
                if speed > h.comfort_speed {
                    cost += w.speed * (speed - h.comfort_speed);
                }
                if speed > h.max_speed && shift > h.max_span - h.comfort_span {
                    cost += w.speed_violation;
                }
            }
            if dt < w.rate_window {
                cost += w.rate * (w.rate_window - dt) / w.rate_window;
            }
            // keys still held must fit with the new chord, and they tie the hand up
            let still: Vec<(i32, f64)> = hs
                .held
                .iter()
                .copied()
                .filter(|&(p, off)| off > t + 0.03 && !chord.contains(&p))
                .collect();
            if !still.is_empty() {
                let mut mx = i32::MIN;
                let mut mn = i32::MAX;
                for &p in still.iter().map(|(p, _)| p).chain(chord.iter()) {
                    mx = mx.max(p);
                    mn = mn.min(p);
                }
                let over = (mx - mn) - h.max_span;
                if over > 0 {
                    cost += w.hold_conflict * over as f64;
                }
                cost += w.busy * still.len().min(3) as f64;
                held = still;
            }
        }
        (cost, HandState { lo, hi, t, held })
    }

    fn crossing_cost(&self, lh: &Option<Rc<HandState>>, rh: &Option<Rc<HandState>>, any: bool) -> f64 {
        let (Some(lh), Some(rh)) = (lh, rh) else { return 0.0 };
        if (lh.t - rh.t).abs() > 0.6 {
            return 0.0;
        }
        let over = if any { lh.hi - rh.lo - self.h.crossing_margin } else { 0 };
        if over > 0 {
            self.w.crossing + self.w.crossing_per_semitone * over as f64
        } else {
            0.0
        }
    }

    fn idle_cost(&self, node: &Node, lchord: &[i32], rchord: &[i32], t: f64) -> f64 {
        let mut cost = 0.0;
        for (chord, other) in [(lchord, &node.rh), (rchord, &node.lh)] {
            if !chord.is_empty() && chord.len() >= 2 && !(!lchord.is_empty() && !rchord.is_empty()) {
                let span = chord[chord.len() - 1] - chord[0];
                let idle = match other {
                    None => true,
                    Some(o) => t - o.t > 0.4,
                };
                if idle && span >= 7 {
                    cost += self.w.idle * (span - 6) as f64 / 6.0;
                }
            }
        }
        cost
    }

    fn stream_cost(&self, node: &Node, pitches: &[i32], dec: &[u8], t: f64) -> f64 {
        if node.prev.is_empty() || t - node.prev_t > 0.3 {
            return 0.0;
        }
        let kept: Vec<(i32, u8)> = pitches
            .iter()
            .zip(dec.iter())
            .filter(|(_, &d)| d != b'x')
            .map(|(&p, &d)| (p, d))
            .collect();
        if kept.is_empty() {
            return 0.0;
        }
        let mut cost = 0.0;
        for &(p, d) in &kept {
            // first minimum, like Python's min()
            let mut best = node.prev[0];
            for &x in &node.prev[1..] {
                if (x.0 - p).abs() < (best.0 - p).abs() {
                    best = x;
                }
            }
            let gap = (best.0 - p).abs() as f64;
            if gap <= self.w.stream_window && best.1 != d {
                cost += self.w.stream * (1.0 - gap / (self.w.stream_window + 1.0));
            }
        }
        cost / kept.len() as f64
    }

    /// Assign `keep` and `effector.hand` (and fingers) for every note of `track`.
    pub fn decode(&self, track: &mut Track, trace: &mut Trace) -> Result<Vec<Violation>> {
        let w = &self.w;
        let notes = &mut track.notes;
        let mut cands = vec![];
        for (i, n) in notes.iter_mut().enumerate() {
            if n.confidence >= w.must_drop {
                cands.push(i);
            } else {
                n.keep = false;
                n.effector = Effector::default();
            }
        }
        let slices = make_slices(notes, &cands, self.h.chord_window);
        let root = Rc::new(Node {
            cost: 0.0,
            lh: None,
            rh: None,
            parent: None,
            decision: vec![],
            prev: vec![],
            prev_t: -1.0,
        });
        let mut beam: Vec<Rc<Node>> = vec![root];

        for sl in &slices {
            let t = notes[sl[0]].onset;
            let conf: Vec<f64> = sl.iter().map(|&i| notes[i].confidence).collect();
            let pitches_all: Vec<i32> = sl.iter().map(|&i| notes[i].pitch).collect();
            let mut optional: Vec<usize> = (0..sl.len()).filter(|&j| conf[j] < w.must_keep).collect();
            if optional.len() > 3 {
                // keep the search small: decide the rest by threshold (stable sort, like Python)
                optional.sort_by(|&a, &b| (conf[a] - 0.5).abs().partial_cmp(&(conf[b] - 0.5).abs()).unwrap());
                optional.truncate(3);
            }
            let fixed_drop: Vec<usize> = (0..sl.len())
                .filter(|&j| conf[j] < w.must_keep && !optional.contains(&j) && conf[j] < 0.5)
                .collect();
            // itertools.product((True, False), repeat=n): True first, last position fastest
            let mut options: Vec<(Vec<usize>, f64)> = vec![];
            let nopt = optional.len();
            for m in 0..(1usize << nopt) {
                let mut keep_idx: Vec<usize> = (0..sl.len()).filter(|j| !fixed_drop.contains(j)).collect();
                let mut ac = 0.0;
                for (pos, &j) in optional.iter().enumerate() {
                    let keep = (m >> (nopt - 1 - pos)) & 1 == 0;
                    let p = conf[j].clamp(1e-3, 1.0 - 1e-3);
                    if keep {
                        ac += -p.ln();
                    } else {
                        ac += -(1.0 - p).ln();
                        keep_idx.retain(|&x| x != j);
                    }
                }
                options.push((keep_idx, w.acoustic * ac));
            }

            let mut expanded: IndexMap<Key, Rc<Node>> = IndexMap::new();
            for node in &beam {
                for (keep_idx, ac) in &options {
                    let pitches: Vec<i32> = keep_idx.iter().map(|&j| pitches_all[j]).collect();
                    let m = pitches.len();
                    for k in 0..=m {
                        let (lchord, rchord) = pitches.split_at(k);
                        let mut cost = node.cost + ac;
                        let mut lh = node.lh.clone();
                        let mut rh = node.rh.clone();
                        if !lchord.is_empty() {
                            let (c, mut st) = self.hand_cost(node.lh.as_deref(), lchord, t, Hand::L);
                            st.held
                                .extend(keep_idx[..k].iter().map(|&j| (notes[sl[j]].pitch, notes[sl[j]].offset)));
                            lh = Some(Rc::new(st));
                            cost += c;
                        }
                        if !rchord.is_empty() {
                            let (c, mut st) = self.hand_cost(node.rh.as_deref(), rchord, t, Hand::R);
                            st.held
                                .extend(keep_idx[k..].iter().map(|&j| (notes[sl[j]].pitch, notes[sl[j]].offset)));
                            rh = Some(Rc::new(st));
                            cost += c;
                        }
                        cost += self.crossing_cost(&lh, &rh, !lchord.is_empty() || !rchord.is_empty());
                        cost += self.idle_cost(node, lchord, rchord, t);
                        let mut dec = vec![b'x'; sl.len()];
                        for (jj, &j) in keep_idx.iter().enumerate() {
                            dec[j] = if jj < k { b'L' } else { b'R' };
                        }
                        cost += self.stream_cost(node, &pitches_all, &dec, t);
                        let key: Key = (bucket(&lh), bucket(&rh), dec.clone());
                        let better = match expanded.get(&key) {
                            None => true,
                            Some(old) => cost < old.cost,
                        };
                        if better {
                            let prev: Vec<(i32, u8)> = (0..sl.len())
                                .filter(|&j| dec[j] != b'x')
                                .map(|j| (pitches_all[j], dec[j]))
                                .collect();
                            let nn = Rc::new(Node {
                                cost,
                                lh,
                                rh,
                                parent: Some(node.clone()),
                                decision: dec,
                                prev,
                                prev_t: t,
                            });
                            expanded.insert(key, nn); // keeps the key's original position, like a dict
                        }
                    }
                }
            }
            let mut next: Vec<Rc<Node>> = expanded.into_values().collect();
            next.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap()); // stable
            next.truncate(w.beam);
            trace.with("slice", || {
                json!({"t": t, "notes": sl.iter().map(|&i| &notes[i].id).collect::<Vec<_>>(),
                       "pitches": pitches_all, "conf": conf,
                       "best": String::from_utf8_lossy(&next[0].decision), "best_cost": next[0].cost,
                       "runner_up": next.get(1).map(|n| (String::from_utf8_lossy(&n.decision).to_string(), n.cost))})
            });
            beam = next;
        }

        // backtrack
        let mut decisions: Vec<Vec<u8>> = vec![];
        let mut node = beam.first().cloned().ok_or_else(|| anyhow!("empty beam"))?;
        while let Some(parent) = node.parent.clone() {
            decisions.push(node.decision.clone());
            node = parent;
        }
        decisions.reverse();
        for (sl, dec) in slices.iter().zip(decisions.iter()) {
            for (&i, &d) in sl.iter().zip(dec.iter()) {
                let n = &mut notes[i];
                if d == b'x' {
                    n.keep = false;
                    n.effector = Effector::default();
                } else {
                    n.keep = true;
                    n.effector = Effector {
                        hand: Some(if d == b'L' { Hand::L } else { Hand::R }),
                        ..Default::default()
                    };
                }
            }
        }

        let mut violations = self.hand_violations(track);
        if self.fingering.as_ref().map(|f| f.enabled).unwrap_or(false) {
            violations.extend(self.finger(track)?);
        }
        trace.with("violations", || json!({"count": violations.len(), "items": violations}));
        Ok(violations)
    }

    fn hand_notes(&self, track: &Track, hand: Hand) -> Vec<usize> {
        track
            .notes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.keep && n.effector.hand == Some(hand))
            .map(|(i, _)| i)
            .collect()
    }

    /// Hard-limit checks on the current assignment (also used by the edit gate and eval).
    pub fn hand_violations(&self, track: &Track) -> Vec<Violation> {
        let notes = &track.notes;
        let mut out = vec![];
        for hand in [Hand::L, Hand::R] {
            let hs = if hand == Hand::L { "L" } else { "R" };
            let hn = self.hand_notes(track, hand);
            let slices = make_slices(notes, &hn, self.h.chord_window);
            for sl in &slices {
                let span = notes[sl[sl.len() - 1]].pitch - notes[sl[0]].pitch;
                let ids = || sl.iter().map(|&i| notes[i].id.clone()).collect::<Vec<_>>();
                if span > self.h.max_span {
                    out.push(Violation::new(
                        "span",
                        notes[sl[0]].onset,
                        format!("{span} semitones in one hand"),
                        ids(),
                        hs,
                    ));
                }
                if sl.len() > self.h.max_notes {
                    out.push(Violation::new(
                        "count",
                        notes[sl[0]].onset,
                        format!("{} notes in one hand", sl.len()),
                        ids(),
                        hs,
                    ));
                }
            }
            for ab in slices.windows(2) {
                let (a, b) = (&ab[0], &ab[1]);
                let dt = (notes[b[0]].onset - notes[a[0]].onset).max(1e-3);
                let union = notes[a[a.len() - 1]].pitch.max(notes[b[b.len() - 1]].pitch)
                    - notes[a[0]].pitch.min(notes[b[0]].pitch);
                let shift = (union - self.h.comfort_span).max(0);
                if shift > self.h.max_span - self.h.comfort_span && shift as f64 / dt > self.h.max_speed {
                    let ids = a.iter().chain(b.iter()).map(|&i| notes[i].id.clone()).collect();
                    out.push(Violation::new(
                        "speed",
                        notes[b[0]].onset,
                        format!("hand jumps {shift} st in {:.0} ms", dt * 1000.0),
                        ids,
                        hs,
                    ));
                }
            }
        }
        out
    }

    /// Viterbi fingering per hand; sets `effector.finger`.
    pub fn finger(&self, track: &mut Track) -> Result<Vec<Violation>> {
        let cfg = self.fingering.as_ref().ok_or_else(|| anyhow!("no fingering config"))?;
        let mut spans: BTreeMap<(u8, u8), [i32; 6]> = BTreeMap::new();
        for (k, v) in &cfg.spans {
            let mut it = k.split('-').map(|x| x.parse::<u8>().unwrap());
            spans.insert((it.next().unwrap(), it.next().unwrap()), *v);
        }
        let shift_time = cfg.shift_time;
        let mut out = vec![];
        for hand in [Hand::L, Hand::R] {
            let hn = self.hand_notes(track, hand);
            let notes = &track.notes;
            let slices = make_slices(notes, &hn, self.h.chord_window);
            if slices.is_empty() {
                continue;
            }
            let states: Vec<Vec<Vec<u8>>> = slices.iter().map(|s| finger_options(s.len(), hand)).collect();
            let pitches: Vec<Vec<i32>> = slices
                .iter()
                .map(|s| s.iter().map(|&i| notes[i].pitch).collect())
                .collect();
            let mut costs: Vec<f64> = states[0]
                .iter()
                .map(|f| chord_cost(&pitches[0], f, hand, &spans))
                .collect();
            let mut back: Vec<Vec<usize>> = vec![];
            for i in 1..slices.len() {
                let dt = notes[slices[i][0]].onset - notes[slices[i - 1][0]].onset;
                let mut new_costs = Vec::with_capacity(states[i].len());
                let mut bp = Vec::with_capacity(states[i].len());
                for fb in &states[i] {
                    let within = chord_cost(&pitches[i], fb, hand, &spans);
                    let (mut best, mut arg) = (f64::INFINITY, 0);
                    for (j, fa) in states[i - 1].iter().enumerate() {
                        let c =
                            costs[j] + trans_cost(&pitches[i - 1], fa, &pitches[i], fb, hand, &spans, dt, shift_time);
                        if c < best {
                            best = c;
                            arg = j;
                        }
                    }
                    new_costs.push(best + within);
                    bp.push(arg);
                }
                costs = new_costs;
                back.push(bp);
            }
            let mut idx = 0;
            for (j, &c) in costs.iter().enumerate() {
                if c < costs[idx] {
                    idx = j;
                }
            }
            let mut path = vec![idx];
            for bp in back.iter().rev() {
                idx = bp[idx];
                path.push(idx);
            }
            path.reverse();
            let hs = if hand == Hand::L { "L" } else { "R" };
            for ((s, opts), &k) in slices.iter().zip(states.iter()).zip(path.iter()) {
                for (&i, &f) in s.iter().zip(opts[k].iter()) {
                    track.notes[i].effector.finger = Some(f);
                }
            }
            let notes = &track.notes;
            for ab in slices.windows(2) {
                let (a, b) = (&ab[0], &ab[1]);
                let dt = notes[b[0]].onset - notes[a[0]].onset;
                let mut fa: BTreeMap<u8, i32> = BTreeMap::new();
                for &i in a {
                    fa.insert(notes[i].effector.finger.unwrap(), notes[i].pitch);
                }
                for &i in b {
                    let n = &notes[i];
                    let f = n.effector.finger.unwrap();
                    if let Some(&pa) = fa.get(&f) {
                        if pa != n.pitch && dt < 0.07 && (pa - n.pitch).abs() > 2 {
                            out.push(Violation::new(
                                "finger",
                                n.onset,
                                format!("finger {f} jumps {} st in {:.0} ms", (pa - n.pitch).abs(), dt * 1000.0),
                                vec![n.id.clone()],
                                hs,
                            ));
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Finger tuples for a chord of n notes sorted low to high.
fn finger_options(n: usize, hand: Hand) -> Vec<Vec<u8>> {
    if n > 5 {
        let base: [u8; 5] = if hand == Hand::R {
            [1, 2, 3, 4, 5]
        } else {
            [5, 4, 3, 2, 1]
        };
        return vec![(0..n).map(|i| base[i.min(4)]).collect()];
    }
    let mut out = vec![];
    combinations(&[1, 2, 3, 4, 5], n, 0, &mut vec![], &mut out);
    if hand == Hand::L {
        for c in &mut out {
            c.reverse();
        }
    }
    out
}

fn combinations(items: &[u8], k: usize, start: usize, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    if cur.len() == k {
        out.push(cur.clone());
        return;
    }
    for i in start..items.len() {
        cur.push(items[i]);
        combinations(items, k, i + 1, cur, out);
        cur.pop();
    }
}

fn pair_cost(
    p_lo: i32,
    f_lo: u8,
    p_hi: i32,
    f_hi: u8,
    hand: Hand,
    spans: &BTreeMap<(u8, u8), [i32; 6]>,
) -> (f64, bool) {
    let (p_lo, p_hi) = if hand == Hand::L { (-p_lo, -p_hi) } else { (p_lo, p_hi) };
    if f_lo == f_hi {
        return (if p_lo == p_hi { 0.0 } else { 3.0 }, false);
    }
    let (a, b, d) = if f_lo < f_hi {
        (f_lo, f_hi, p_hi - p_lo)
    } else {
        (f_hi, f_lo, p_lo - p_hi)
    };
    let [mn_p, mn_c, mn_r, mx_r, mx_c, mx_p] = spans[&(a, b)];
    let mut cost = 0.0;
    let impossible = d > mx_p || d < mn_p;
    if d > mx_c {
        cost += 2.0 * (d - mx_c) as f64;
    } else if d < mn_c {
        cost += 2.0 * (mn_c - d) as f64;
    }
    if d > mx_r {
        cost += (if a == 1 { 1.0 } else { 2.0 }) * (d - mx_r) as f64;
    } else if d < mn_r {
        cost += (if a == 1 { 2.0 } else { 1.0 }) * (mn_r - d) as f64;
    }
    (cost, impossible)
}

fn chord_cost(p: &[i32], fingers: &[u8], hand: Hand, spans: &BTreeMap<(u8, u8), [i32; 6]>) -> f64 {
    let mut cost = 0.0;
    for i in 0..p.len().saturating_sub(1) {
        let (c, imp) = pair_cost(p[i], fingers[i], p[i + 1], fingers[i + 1], hand, spans);
        cost += c + if imp { 20.0 } else { 0.0 };
    }
    for (&pp, &f) in p.iter().zip(fingers.iter()) {
        let black = is_black(pp);
        if f == 4 || f == 5 {
            cost += 0.5;
        }
        if f == 1 && black {
            cost += 1.0;
        }
        if f == 5 && black {
            cost += 0.5;
        }
    }
    cost
}

#[allow(clippy::too_many_arguments)]
fn trans_cost(
    a: &[i32],
    fa: &[u8],
    b: &[i32],
    fb: &[u8],
    hand: Hand,
    spans: &BTreeMap<(u8, u8), [i32; 6]>,
    dt: f64,
    shift_time: f64,
) -> f64 {
    // judge the transition by the outer notes of each chord (the hand's frame)
    let mut cost = 0.0;
    let pairs = [
        (a[0], fa[0], b[0], fb[0]),
        (a[a.len() - 1], fa[fa.len() - 1], b[b.len() - 1], fb[fb.len() - 1]),
    ];
    for (na, f1, nb, f2) in pairs {
        let mut c;
        if f1 == f2 {
            if na == nb {
                continue;
            }
            c = 4.0 + if dt < shift_time { 6.0 } else { 0.0 };
        } else {
            let (pc, imp) = pair_cost(na, f1, nb, f2, hand, spans);
            c = pc;
            if imp {
                c += 15.0;
            }
            // thumb passing under / finger crossing over
            let up = (nb > na) == (hand == Hand::R);
            if f2 == 1 && f1 != 1 && up {
                c += 1.0 + if is_black(nb) { 2.0 } else { 0.0 };
            }
            if f1 == 1 && f2 != 1 && !up {
                c += 1.0;
            }
            if f1 == 3 && f2 == 4 {
                c += 0.5;
            }
        }
        if dt >= shift_time {
            c = c.min(1.5); // time to reposition the whole hand
        }
        cost += c;
    }
    cost / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Note;

    fn n(i: usize, t: f64, p: i32, d: f64, conf: f64) -> Note {
        let mut x = Note::new(format!("n{i}"), t, t + d, p, 80);
        x.confidence = conf;
        x
    }

    #[test]
    fn hands_far_apart() {
        let mut notes = vec![];
        for k in 0..16 {
            notes.push(n(2 * k, k as f64 * 0.25, 36 + (k as i32 % 4), 0.2, 1.0));
            notes.push(n(2 * k + 1, k as f64 * 0.25, 72 + (k as i32 % 5), 0.2, 1.0));
        }
        let mut tr = Track::new("piano", "piano", notes);
        let v = PianoDecoder::new("piano")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        for x in &tr.notes {
            assert_eq!(x.effector.hand, Some(if x.pitch < 60 { Hand::L } else { Hand::R }));
        }
        assert!(!v.iter().any(|x| x.kind == "span" || x.kind == "count"));
    }

    #[test]
    fn wide_chord_split_and_impossible_flagged() {
        let mut tr = Track::new(
            "p",
            "piano",
            [36, 48, 67, 79]
                .iter()
                .enumerate()
                .map(|(i, &p)| n(i, 0.0, p, 1.0, 1.0))
                .collect(),
        );
        let v = PianoDecoder::new("piano")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        let hands: Vec<_> = tr.notes.iter().map(|x| x.effector.hand.unwrap()).collect();
        assert_eq!(hands, vec![Hand::L, Hand::L, Hand::R, Hand::R]);
        assert!(!v.iter().any(|x| x.kind == "span"));
        let mut tr = Track::new(
            "p",
            "piano",
            [36, 52, 67, 84]
                .iter()
                .enumerate()
                .map(|(i, &p)| n(i, 0.0, p, 1.0, 1.0))
                .collect(),
        );
        let v = PianoDecoder::new("piano")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        assert!(v.iter().any(|x| x.kind == "span"));
    }

    #[test]
    fn uncertain_eleventh_note_dropped() {
        let mut notes: Vec<Note> = [40, 43, 47, 50, 52, 64, 67, 71, 74, 76]
            .iter()
            .enumerate()
            .map(|(i, &p)| n(i, 0.0, p, 1.0, 1.0))
            .collect();
        notes.push(n(99, 0.0, 58, 1.0, 0.45));
        let mut tr = Track::new("p", "piano", notes);
        PianoDecoder::new("piano")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        assert!(!tr.notes[10].keep);
        assert!(tr.notes[..10].iter().all(|x| x.keep));
    }

    #[test]
    fn product_mask_order_matches_python() {
        // with optional = [a, b]: (T,T), (T,F), (F,T), (F,F)
        let nopt = 2;
        let masks: Vec<Vec<bool>> = (0..(1 << nopt))
            .map(|m| (0..nopt).map(|pos| (m >> (nopt - 1 - pos)) & 1 == 0).collect())
            .collect();
        assert_eq!(
            masks,
            vec![
                vec![true, true],
                vec![true, false],
                vec![false, true],
                vec![false, false]
            ]
        );
    }
}
