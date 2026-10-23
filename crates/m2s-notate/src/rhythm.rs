//! Tempo map, beat unit, meter and quantization (port of `m2s/notate/rhythm.py` and `beats.py`).

use m2s_core::ir::Note;
use m2s_core::trace::Trace;
use serde_json::{json, Map, Value};

use crate::np;

/// Subdivisions per quarter and their complexity cost, in insertion order (it breaks ties).
pub const DIVISIONS: [(i64, f64); 6] = [(1, 0.0), (2, 0.008), (4, 0.02), (8, 0.09), (3, 0.04), (6, 0.09)];

fn div_cost(d: i64) -> f64 {
    DIVISIONS.iter().find(|(k, _)| *k == d).map(|(_, c)| *c).unwrap()
}

#[derive(Debug, Clone)]
pub struct Grid {
    pub beat_times: Vec<f64>,
    pub unit: f64,
    pub bar_quarters: f64,
    pub beats_per_bar: i64,
    pub pickup_quarters: f64,
    pub first_downbeat_beat: i64,
    pub divisions: Vec<i64>,
    pub info: Map<String, Value>,
}

impl Grid {
    /// Seconds -> quarters from the first downbeat (negative inside the pickup).
    pub fn to_quarters(&self, t: f64) -> f64 {
        let bt = &self.beat_times;
        let n = bt.len();
        let b = if t <= bt[0] {
            (t - bt[0]) / (bt[1] - bt[0]).max(1e-3)
        } else if t >= bt[n - 1] {
            (n - 1) as f64 + (t - bt[n - 1]) / (bt[n - 1] - bt[n - 2]).max(1e-3)
        } else {
            let idx: Vec<f64> = (0..n).map(|i| i as f64).collect();
            np::interp(t, bt, &idx)
        };
        (b - self.first_downbeat_beat as f64) * self.unit
    }

    /// Quarters from the first downbeat -> seconds (inverse of `to_quarters`, linear outside the beats).
    pub fn time_of(&self, q: f64) -> f64 {
        let bt = &self.beat_times;
        let n = bt.len();
        let b = q / self.unit + self.first_downbeat_beat as f64;
        if b <= 0.0 {
            bt[0] + b * (bt[1] - bt[0])
        } else if b >= (n - 1) as f64 {
            bt[n - 1] + (b - (n - 1) as f64) * (bt[n - 1] - bt[n - 2])
        } else {
            let i = b.floor() as usize;
            bt[i] + (b - i as f64) * (bt[i + 1] - bt[i])
        }
    }

    /// Length of the notated beat in quarters: a dotted quarter in compound meter, else the
    /// time signature's denominator.
    pub fn beat_quarters(&self) -> f64 {
        if self.info.get("compound").and_then(Value::as_bool).unwrap_or(false) {
            return 1.5;
        }
        4.0 / self.time_signature_parts().1 as f64
    }

    pub fn time_signature(&self) -> String {
        let q = self.bar_quarters;
        if self.info.get("compound").and_then(Value::as_bool).unwrap_or(false) {
            return format!("{}/8", np::round_i(q * 2.0));
        }
        // a half-note beat is cut time (2/2, 3/2), not 4/4 or 6/4
        if self.unit == 2.0 {
            return format!("{}/2", self.beats_per_bar);
        }
        if (q - q.round_ties_even()).abs() < 1e-6 {
            return format!("{}/4", np::round_i(q));
        }
        format!("{}/8", np::round_i(q * 2.0))
    }

    /// (beats, beat-type) of the time signature.
    pub fn time_signature_parts(&self) -> (i64, i64) {
        let ts = self.time_signature();
        let mut it = ts.split('/');
        (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
    }

    /// Quarters per minute: tracked beats per minute times quarters per tracked beat.
    pub fn qpm(&self) -> f64 {
        let ibi = np::median(&np::diff(&self.beat_times));
        60.0 / ibi * self.unit
    }
}

/// Fix the commonest tracker slips: doubled beats and single missing beats.
pub fn clean_beats(beats: &[f64]) -> Vec<f64> {
    if beats.len() < 4 {
        return beats.to_vec();
    }
    let med = np::median(&np::diff(beats));
    let mut out = vec![beats[0]];
    for &t in &beats[1..] {
        let gap = t - out[out.len() - 1];
        if gap < 0.45 * med {
            continue;
        }
        if 1.6 * med < gap && gap < 2.4 * med {
            let last = out[out.len() - 1];
            out.push(last + gap / 2.0);
        }
        out.push(t);
    }
    out
}

/// Peak downbeat activation within +-60 ms of each beat.
pub fn downbeat_strength(beats: &[f64], db_act: &[f32], act_fps: f64) -> Vec<f64> {
    if db_act.is_empty() {
        return vec![0.0; beats.len()];
    }
    beats
        .iter()
        .map(|&b| {
            let f = np::round_i(b * act_fps);
            let lo = (f - 3).max(0) as usize;
            let hi = ((f + 4).max(0) as usize).min(db_act.len());
            if hi > lo {
                db_act[lo..hi].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64
            } else {
                0.0
            }
        })
        .collect()
}

pub fn choose_unit(beat_times: &[f64], notes: &[&Note]) -> (f64, Map<String, Value>) {
    let ibi = np::median(&np::diff(beat_times));
    let onsets: Vec<f64> = notes.iter().map(|n| np::np_round_n(n.onset, 2)).collect();
    let onsets = np::unique(&onsets);
    let iois: Vec<f64> = np::diff(&onsets).into_iter().filter(|&d| d > 0.03).collect();
    let fast_beats = if iois.is_empty() {
        0.5
    } else {
        np::percentile(&iois, 20.0) / ibi
    };
    let mut scores: Vec<(f64, f64)> = vec![];
    for unit in [0.5, 1.0, 2.0] {
        let qpm = 60.0 / ibi * unit;
        let bias = if unit == 1.0 { 0.0 } else { 0.8 };
        let tempo_pen = (((qpm / 110.0).ln()).abs() - 1.7f64.ln()).max(0.0).powi(2) / 0.1;
        let fast_q = fast_beats * unit;
        let val = [(0.25, 0.0), (0.5, 0.1), (0.125, 0.6), (1.0, 0.8)]
            .iter()
            .map(|(v, c)| (fast_q / v).log2().abs() + c)
            .fold(f64::INFINITY, f64::min);
        scores.push((unit, bias + tempo_pen + 1.5 * val));
    }
    let mut best = scores[0];
    for s in &scores[1..] {
        if s.1 < best.1 {
            best = *s;
        }
    }
    let mut m = Map::new();
    for (u, v) in &scores {
        m.insert(format!("unit={}", np::py_repr(*u)), json!(np::round_n(*v, 3)));
    }
    (best.0, m)
}

/// Per tracked beat: [bass onset, harmony change, loudness].
pub fn beat_features(beat_times: &[f64], notes: &[&Note]) -> Vec<[f64; 3]> {
    let nb = beat_times.len();
    let mut feats = vec![[0.0; 3]; nb];
    if notes.is_empty() {
        return feats;
    }
    let mut edges = vec![f64::NEG_INFINITY];
    edges.extend(beat_times.windows(2).map(|w| (w[1] + w[0]) / 2.0));
    edges.push(f64::INFINITY);
    let mut pcs = vec![[0.0f64; 12]; nb];
    let mut vel = vec![0.0f64; nb];
    let mut low = vec![200i32; nb];
    for n in notes {
        let i = (np::searchsorted(&edges, n.onset, false) as i64 - 1).clamp(0, nb as i64 - 1) as usize;
        pcs[i][n.pitch.rem_euclid(12) as usize] += n.duration();
        vel[i] = vel[i].max(n.velocity as f64);
        if (n.onset - beat_times[i]).abs() < 0.08 {
            low[i] = low[i].min(n.pitch);
        }
    }
    let pitches: Vec<i32> = notes.iter().map(|n| n.pitch).collect();
    let allmin = np::percentile_i(&pitches, 10.0);
    let norms: Vec<f64> = pcs.iter().map(|r| np::norm_rows(r) + 1e-9).collect();
    let normed: Vec<[f64; 12]> = pcs
        .iter()
        .zip(&norms)
        .map(|(r, &d)| {
            let mut o = [0.0; 12];
            for k in 0..12 {
                o[k] = r[k] / d;
            }
            o
        })
        .collect();
    let vmax = vel.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1e-9;
    for i in 0..nb {
        feats[i][0] = if (low[i] as f64) < allmin + 7.0 { 1.0 } else { 0.0 };
        feats[i][1] = if i == 0 {
            1.0
        } else {
            let prod: Vec<f64> = (0..12).map(|k| normed[i][k] * normed[i - 1][k]).collect();
            1.0 - np::pairwise_sum(&prod)
        };
        feats[i][2] = vel[i] / vmax;
    }
    feats
}

/// Per beat: how much the harmony of the `w` beats after it differs from the `w` beats before.
pub fn harmonic_boundaries(beat_times: &[f64], notes: &[&Note], w: usize) -> Vec<f64> {
    let nb = beat_times.len();
    let mut edges = beat_times.to_vec();
    edges.push(beat_times[nb - 1] + (beat_times[nb - 1] - beat_times[nb - 2]));
    let mut chroma = vec![[0.0f64; 12]; nb];
    for n in notes {
        let a = np::searchsorted(&edges, n.onset, true) as i64 - 1;
        let b = np::searchsorted(&edges, n.offset, false) as i64;
        for i in a.max(0)..b.min(nb as i64) {
            let i = i as usize;
            let ov = n.offset.min(edges[i + 1]) - n.onset.max(edges[i]);
            if ov > 0.0 {
                chroma[i][n.pitch.rem_euclid(12) as usize] += ov;
            }
        }
    }
    let mut out = vec![0.0; nb];
    if nb + 1 > 2 * w {
        for i in w..(nb - w + 1) {
            let mut x = [0.0f64; 12];
            let mut y = [0.0f64; 12];
            for r in &chroma[i - w..i] {
                for k in 0..12 {
                    x[k] += r[k];
                }
            }
            for r in &chroma[i..i + w] {
                for k in 0..12 {
                    y[k] += r[k];
                }
            }
            let den = dot(&x, &x).sqrt() * dot(&y, &y).sqrt();
            out[i] = if den > 0.0 { 1.0 - dot(&x, &y) / den } else { 0.0 };
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

fn mean_rows(feats: &[[f64; 3]], idx: &[usize]) -> [f64; 3] {
    let mut s = [0.0; 3];
    for &i in idx {
        for k in 0..3 {
            s[k] += feats[i][k];
        }
    }
    let n = idx.len() as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

fn mean_idx(v: &[f64], idx: &[usize]) -> f64 {
    let x: Vec<f64> = idx.iter().map(|&i| v[i]).collect();
    np::mean(&x)
}

/// (tracked beats per bar, index of first downbeat, top candidate scores).
pub fn choose_meter(
    beat_times: &[f64],
    downbeats: &[f64],
    notes: &[&Note],
    unit: f64,
    db_strength: Option<&[f64]>,
) -> (i64, i64, Vec<(String, f64)>) {
    let feats = beat_features(beat_times, notes);
    let nb = beat_times.len();
    let db: Vec<f64> = match db_strength {
        Some(d) if d.len() == nb => d.to_vec(),
        _ => {
            let mut db = vec![0.0; nb];
            for &t in downbeats {
                let mut best = 0;
                for i in 1..nb {
                    if (beat_times[i] - t).abs() < (beat_times[best] - t).abs() {
                        best = i;
                    }
                }
                if (beat_times[best] - t).abs() < 0.1 {
                    db[best] = 1.0;
                }
            }
            db
        }
    };
    let hb = harmonic_boundaries(beat_times, notes, 2);
    let mut cands: Vec<((i64, i64), f64)> = vec![];
    for per_bar_q in [2.0, 3.0, 4.0, 6.0] {
        let bpbf = per_bar_q / unit;
        if (bpbf - bpbf.round_ties_even()).abs() > 1e-6 || bpbf.round_ties_even() < 2.0 {
            continue;
        }
        let bpb = bpbf.round_ties_even() as i64;
        for phase in 0..bpb {
            let on: Vec<usize> = (phase as usize..nb).step_by(bpb as usize).collect();
            let off: Vec<usize> = (0..nb).filter(|i| !on.contains(i)).collect();
            if on.len() < 2 || off.len() < 2 {
                continue;
            }
            let fo = mean_rows(&feats, &on);
            let ff = mean_rows(&feats, &off);
            let contrast = [fo[0] - ff[0], fo[1] - ff[1], fo[2] - ff[2]];
            let harm = mean_idx(&hb, &on) - mean_idx(&hb, &off);
            let mut s = 0.5 * contrast[0]
                + 0.3 * contrast[1]
                + 0.3 * contrast[2]
                + 2.5 * harm
                + 1.0 * (mean_idx(&db, &on) - mean_idx(&db, &off));
            s += match per_bar_q as i64 {
                2 => -0.02,
                3 => 0.0,
                4 => 0.02,
                _ => -0.15,
            };
            s += if phase == 0 { 0.02 } else { 0.0 };
            cands.push(((bpb, phase), s));
        }
    }
    let mut best = cands[0];
    for c in &cands[1..] {
        if c.1 > best.1 {
            best = *c;
        }
    }
    let mut sorted = cands.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top = sorted
        .iter()
        .take(6)
        .map(|((bpb, ph), v)| (format!("{bpb}x{}q@{ph}", np::py_repr(unit)), np::round_n(*v, 3)))
        .collect();
    (best.0 .0, best.0 .1, top)
}

/// With a drum kit, the snare backbeat fixes the quarter note.
pub fn backbeat_unit(beat_times: &[f64], drum_hits: &[&Note]) -> Option<f64> {
    let mut snare: Vec<f64> = drum_hits
        .iter()
        .filter(|n| n.pitch == 38 || n.pitch == 40)
        .map(|n| n.onset)
        .collect();
    snare.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if snare.len() < 8 || beat_times.len() < 8 {
        return None;
    }
    let ibi = np::median(&np::diff(beat_times));
    let gaps: Vec<f64> = np::diff(&snare)
        .into_iter()
        .filter(|&g| g > 0.6 * ibi && g < 4.5 * ibi)
        .collect();
    if gaps.len() < 6 {
        return None;
    }
    let ratio = np::median(&gaps) / ibi;
    if (ratio - 1.0).abs() < 0.2 {
        return Some(2.0);
    }
    if (ratio - 2.0).abs() < 0.3 {
        return Some(1.0);
    }
    if (ratio - 4.0).abs() < 0.5 {
        return Some(0.5);
    }
    None
}

pub fn build_grid(
    beats: &[f64],
    downbeats: &[f64],
    notes: &[&Note],
    db_strength: Option<&[f64]>,
    drum_hits: &[&Note],
    trace: &mut Trace,
) -> Grid {
    let mut bt = if db_strength.is_none() {
        clean_beats(beats)
    } else {
        beats.to_vec()
    };
    let (mut unit, mut unit_scores) = choose_unit(&bt, notes);
    if let Some(bb) = backbeat_unit(&bt, drum_hits) {
        unit = bb;
        unit_scores.insert("backbeat".into(), json!(bb));
    }
    let (bpb, phase, meter_scores) = choose_meter(&bt, downbeats, notes, unit, db_strength);
    let mut pickup = phase as f64 * unit;
    let mut first_db = phase;
    if !notes.is_empty() {
        let t0 = notes.iter().map(|n| n.onset).fold(f64::INFINITY, f64::min);
        let ibi = np::median(&np::diff(&bt[..bt.len().min(8)]));
        let mut extra = 0i64;
        while bt[0] - extra as f64 * ibi > t0 + 0.05 && extra < 16 {
            extra += 1;
        }
        if extra > 0 {
            let b0 = bt[0];
            let mut pre: Vec<f64> = (1..=extra).rev().map(|k| b0 - ibi * k as f64).collect();
            pre.extend_from_slice(&bt);
            bt = pre;
            first_db += extra;
            pickup = (first_db % bpb) as f64 * unit;
        }
    }
    let bar_q = bpb as f64 * unit;
    let mut info = Map::new();
    info.insert("unit_scores".into(), Value::Object(unit_scores));
    let mut ms = Map::new();
    for (k, v) in &meter_scores {
        ms.insert(k.clone(), json!(v));
    }
    info.insert("meter_scores".into(), Value::Object(ms));
    let mut grid = Grid {
        beat_times: bt,
        unit,
        bar_quarters: bar_q,
        beats_per_bar: bpb,
        pickup_quarters: py_mod(pickup, bar_q),
        first_downbeat_beat: first_db,
        divisions: vec![],
        info,
    };
    let trip = triplet_share(notes, &grid);
    grid.info.insert("triplet_share".into(), json!(np::round_n(trip, 3)));
    if trip >= 0.55 && (2..=4).contains(&bpb) {
        grid.unit = 1.5;
        grid.bar_quarters = 1.5 * bpb as f64;
        grid.pickup_quarters = py_mod((first_db % bpb) as f64 * 1.5, grid.bar_quarters);
        grid.info.insert("compound".into(), json!(true));
    }
    trace.with("grid", || {
        json!({"unit": grid.unit, "beats_per_bar": grid.beats_per_bar, "time_signature": grid.time_signature(),
               "pickup_quarters": grid.pickup_quarters, "first_downbeat_beat": grid.first_downbeat_beat,
               "beats": grid.beat_times.len(), "info": grid.info})
    });
    grid
}

/// Python's float modulo (result has the sign of the divisor).
fn py_mod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
}

/// Share of subdivided beats whose onsets fit thirds better than halves/quarters.
pub fn triplet_share(notes: &[&Note], grid: &Grid) -> f64 {
    if notes.is_empty() {
        return 0.0;
    }
    let pos: Vec<f64> = notes.iter().map(|n| grid.to_quarters(n.onset) / grid.unit).collect();
    let frac: Vec<f64> = pos.iter().map(|p| p - p.floor()).collect();
    let beat: Vec<i64> = pos.iter().map(|p| p.floor() as i64).collect();
    let mut beats_u = beat.clone();
    beats_u.sort();
    beats_u.dedup();
    let (mut trip, mut binary) = (0usize, 0usize);
    for b in beats_u {
        let f: Vec<f64> = frac
            .iter()
            .zip(&beat)
            .filter(|(_, &bb)| bb == b)
            .map(|(&f, _)| f)
            .filter(|&f| f > 0.08 && f < 0.92)
            .collect();
        if f.is_empty() {
            continue;
        }
        let e3: Vec<f64> = f
            .iter()
            .map(|x| (x * 3.0 - (x * 3.0).round_ties_even()).abs() / 3.0)
            .collect();
        let e4: Vec<f64> = f
            .iter()
            .map(|x| (x * 4.0 - (x * 4.0).round_ties_even()).abs() / 4.0)
            .collect();
        if np::mean(&e3) + 0.01 < np::mean(&e4) {
            trip += 1;
        } else {
            binary += 1;
        }
    }
    trip as f64 / (trip + binary).max(1) as f64
}

fn frac_err(frac: &[f64], d: i64) -> f64 {
    let df = d as f64;
    let e: Vec<f64> = frac
        .iter()
        .map(|x| {
            let p = x * df;
            (p - p.round_ties_even()).abs() / df
        })
        .collect();
    np::mean(&e)
}

/// Quantized onset (absolute quarters from bar 1) and duration for each note, in order.
pub fn quantize(notes: &[&Note], grid: &mut Grid, trace: &mut Trace) -> Vec<(f64, f64)> {
    if notes.is_empty() {
        return vec![];
    }
    let qon: Vec<f64> = notes.iter().map(|n| grid.to_quarters(n.onset)).collect();
    let qoff: Vec<f64> = notes.iter().map(|n| grid.to_quarters(n.offset)).collect();
    let qmin = qon.iter().cloned().fold(f64::INFINITY, f64::min);
    let qmax = qon
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(qoff.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let start = qmin.floor() as i64;
    let end = qmax.ceil() as i64 + 1;
    let nq = end - start;
    let in_beat = |q: i64| -> Vec<f64> {
        qon.iter()
            .filter(|&&x| x >= q as f64 - 1e-9 && x < (q + 1) as f64)
            .map(|&x| x - q as f64)
            .collect()
    };
    let mut divs = Vec::with_capacity(nq as usize);
    let mut prev = 4i64;
    for q in start..end {
        let frac = in_beat(q);
        let mut best = f64::INFINITY;
        let mut arg = prev;
        let mut errs = vec![];
        for (d, cost) in DIVISIONS {
            let err = if frac.is_empty() { 0.0 } else { frac_err(&frac, d) };
            let c = err + cost + if d == prev { 0.0 } else { 0.01 };
            if trace.enabled() {
                errs.push(json!([d, np::round_n(c, 4)]));
            }
            if c < best {
                best = c;
                arg = d;
            }
        }
        if trace.enabled() && !frac.is_empty() {
            trace.emit(
                "subdivision",
                json!({"quarter": q, "onsets": frac.len(), "chosen": arg, "costs": errs}),
            );
        }
        divs.push(arg);
        if !frac.is_empty() {
            prev = arg;
        }
    }
    let best_binary = |q: i64| -> i64 {
        let frac = in_beat(q);
        if frac.is_empty() {
            return 4;
        }
        let mut best = (1, f64::INFINITY);
        for d in [1, 2, 4] {
            let v = frac_err(&frac, d) + div_cost(d);
            if v < best.1 {
                best = (d, v);
            }
        }
        best.0
    };
    let fam = |d: i64| match d {
        1 | 2 | 4 => 'b',
        8 => '3',
        _ => 't',
    };
    let mut smoothed = divs.clone();
    for (i, &d) in divs.iter().enumerate() {
        if fam(d) == 'b' {
            continue;
        }
        let nb: Vec<char> = [i as i64 - 1, i as i64 + 1]
            .iter()
            .filter(|&&j| j >= 0 && (j as usize) < divs.len())
            .map(|&j| fam(divs[j as usize]))
            .collect();
        if !nb.contains(&fam(d)) {
            smoothed[i] = best_binary(start + i as i64);
            trace.with(
                "subdivision-smoothed",
                || json!({"quarter": start + i as i64, "from": d, "to": smoothed[i]}),
            );
        }
    }
    let divs = smoothed;
    let snap = |x: f64| -> f64 {
        let q = x.floor();
        let i = ((q as i64) - start).clamp(0, nq - 1) as usize;
        let d = divs[i] as f64;
        q + ((x - q) * d).round_ties_even() / d
    };
    let mut out = Vec::with_capacity(notes.len());
    for (k, n) in notes.iter().enumerate() {
        let on = snap(qon[k]);
        let mut off = snap(qoff[k]);
        let q = on.floor() as i64;
        let d = divs[(q - start).clamp(0, nq - 1) as usize];
        if off <= on {
            off = on + 1.0 / d as f64;
        }
        let beat = np::round_n(on, 6);
        let qdur = np::round_n(off - on, 6);
        trace.with(
            "quantize",
            || json!({"id": n.id, "onset": n.onset, "q": np::round_n(qon[k], 4), "beat": beat, "qdur": qdur}),
        );
        out.push((beat, qdur));
    }
    grid.divisions = divs;
    out
}

/// A silence that is a pause (fermata, breath, a held chord), not a written rest.
#[derive(Debug, Clone)]
pub struct Pause {
    /// onset of the last chord before the silence
    pub after: f64,
    pub until: f64,
    pub removed_beats: usize,
    pub mean_activation: f64,
    pub ibi_variation: f64,
}

/// Remove tracked beats inside pauses so a fermata or breath doesn't become extra bars.
///
/// A gap between consecutive onsets longer than `min_beats` beats is a pause when the tracker
/// itself is unsure there: its beat activation inside the gap is weak (mean < `max_act`), or the
/// beats it placed there don't keep the local tempo (spacing varies by more than `max_var`).
/// A steady, confident pulse through a silence is left alone: that is a written rest.
///
/// Only whole bars (multiples of `bar` beats) are removed, so the bar lines after a pause keep
/// their phase; the leftover beats lengthen the chord before it.
#[allow(clippy::too_many_arguments)]
pub fn compress_pauses(
    beats: &[f64],
    onsets: &[f64],
    beat_act: &[f32],
    fps: f64,
    min_beats: f64,
    max_act: f64,
    max_var: f64,
    bar: usize,
) -> (Vec<f64>, Vec<Pause>) {
    let bar = bar.max(1);
    if beats.len() < 8 || onsets.len() < 2 {
        return (beats.to_vec(), vec![]);
    }
    let ibi = np::median(&np::diff(beats));
    let act_at = |t: f64| -> f64 {
        let f = (t * fps).round() as i64;
        let lo = (f - 3).max(0) as usize;
        let hi = ((f + 4).max(0) as usize).min(beat_act.len());
        (lo..hi).map(|i| beat_act[i] as f64).fold(0.0, f64::max)
    };
    let mut drop = vec![false; beats.len()];
    let mut pauses = vec![];
    for w in onsets.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b - a <= min_beats * ibi {
            continue;
        }
        let inside: Vec<usize> = (0..beats.len())
            .filter(|&i| beats[i] > a + 0.5 * ibi && beats[i] < b - 0.25 * ibi)
            .collect();
        if inside.is_empty() {
            continue;
        }
        let mean_act = inside.iter().map(|&i| act_at(beats[i])).sum::<f64>() / inside.len() as f64;
        // spacing of the beats through the gap, including its two edges
        let mut pts = vec![a];
        pts.extend(inside.iter().map(|&i| beats[i]));
        pts.push(b);
        let d = np::diff(&pts);
        let m = d.iter().sum::<f64>() / d.len() as f64;
        let sd = (d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / d.len() as f64).sqrt();
        let var = sd / ibi;
        let k = inside.len() / bar * bar;
        if k > 0 && (mean_act < max_act || var > max_var) {
            for &i in &inside[inside.len() - k..] {
                drop[i] = true;
            }
            pauses.push(Pause {
                after: a,
                until: b,
                removed_beats: k,
                mean_activation: mean_act,
                ibi_variation: var,
            });
        }
    }
    let kept: Vec<f64> = beats
        .iter()
        .zip(drop.iter())
        .filter(|(_, &d)| !d)
        .map(|(&b, _)| b)
        .collect();
    (kept, pauses)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 120 bpm, notes on every beat except a 3-beat silence after t = 5 s
    fn pulse(act_in_gap: f32) -> (Vec<f64>, Vec<f64>, Vec<f32>) {
        let fps = 50.0;
        let beats: Vec<f64> = (0..24).map(|k| k as f64 * 0.5).collect();
        let onsets: Vec<f64> = beats.iter().copied().filter(|&t| !(t > 5.0 && t < 7.0)).collect();
        let mut act = vec![0.02f32; 700];
        for &b in &beats {
            let f = (b * fps) as usize;
            act[f] = if b > 5.0 && b < 7.0 { act_in_gap } else { 0.9 };
        }
        (beats, onsets, act)
    }

    #[test]
    fn unsure_tracker_in_a_silence_is_a_pause() {
        let (beats, onsets, act) = pulse(0.1);
        let (kept, pauses) = compress_pauses(&beats, &onsets, &act, 50.0, 1.5, 0.35, 0.25, 1);
        assert_eq!(pauses.len(), 1);
        assert_eq!(pauses[0].after, 5.0);
        assert_eq!(pauses[0].removed_beats, 3);
        assert_eq!(kept.len(), beats.len() - 3);
    }

    #[test]
    fn confident_steady_pulse_through_a_silence_is_a_rest() {
        let (beats, onsets, act) = pulse(0.9);
        let (kept, pauses) = compress_pauses(&beats, &onsets, &act, 50.0, 1.5, 0.35, 0.25, 1);
        assert!(pauses.is_empty());
        assert_eq!(kept, beats);
    }

    #[test]
    fn pauses_remove_whole_bars_only() {
        // 3 beats inside the silence, 2 beats per bar: one bar (2 beats) goes, the first beat stays
        let (beats, onsets, act) = pulse(0.1);
        let (kept, pauses) = compress_pauses(&beats, &onsets, &act, 50.0, 1.5, 0.35, 0.25, 2);
        assert_eq!(pauses[0].removed_beats, 2);
        assert!(kept.contains(&5.5) && !kept.contains(&6.0) && !kept.contains(&6.5));
        // 4 beats per bar: less than a bar of pause, nothing is removed
        let (kept, pauses) = compress_pauses(&beats, &onsets, &act, 50.0, 1.5, 0.35, 0.25, 4);
        assert!(pauses.is_empty());
        assert_eq!(kept, beats);
    }
}
