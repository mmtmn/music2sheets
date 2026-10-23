//! Beat tracking by a dynamic Bayesian network over the neural beat activation
//! (Krebs, Böck & Widmer 2015, "An efficient state-space model for joint tempo and meter
//! tracking"; the beat-only variant is madmom's `DBNBeatTrackingProcessor`).
//!
//! State: (beat interval I in frames, position p in 0..I). Within a beat the position advances
//! one frame at a time; at the end of a beat the interval may change, with probability
//! exp(-lambda * |I'/I - 1|) (normalised). The first 1/obs_lambda of every beat is the "beat"
//! region: there the observation probability is the activation a, elsewhere (1 - a) /
//! (obs_lambda - 1). Viterbi gives the beat frames; each beat is then moved to the activation
//! peak within +-`correct` frames.

#[derive(Debug, Clone)]
pub struct DbnParams {
    pub min_bpm: f64,
    pub max_bpm: f64,
    pub transition_lambda: f64,
    pub observation_lambda: f64,
    /// search radius (frames) for snapping beats to the activation peak
    pub correct: usize,
}

impl Default for DbnParams {
    fn default() -> Self {
        // madmom's defaults
        DbnParams {
            min_bpm: 55.0,
            max_bpm: 215.0,
            transition_lambda: 100.0,
            observation_lambda: 16.0,
            correct: 3,
        }
    }
}

/// Beat times (seconds) from a beat activation sampled at `fps`.
pub fn track(act: &[f32], fps: f64, p: &DbnParams) -> Vec<f64> {
    let n = act.len();
    if n < 4 {
        return vec![];
    }
    let min_i = ((60.0 * fps / p.max_bpm).round() as usize).max(2);
    let max_i = ((60.0 * fps / p.min_bpm).round() as usize).max(min_i + 1);
    let intervals: Vec<usize> = (min_i..=max_i).collect();
    // state index = offset[k] + position
    let mut offset = Vec::with_capacity(intervals.len());
    let mut total = 0usize;
    for &i in &intervals {
        offset.push(total);
        total += i;
    }
    // tempo transition log-probabilities between intervals (row = from, normalised)
    let ni = intervals.len();
    let mut trans = vec![f64::NEG_INFINITY; ni * ni];
    for (a, &ia) in intervals.iter().enumerate() {
        let row: Vec<f64> = intervals
            .iter()
            .map(|&ib| (-p.transition_lambda * (ib as f64 / ia as f64 - 1.0).abs()).exp())
            .collect();
        let s: f64 = row.iter().sum();
        for (b, v) in row.iter().enumerate() {
            let pr = v / s;
            if pr > 1e-12 {
                trans[a * ni + b] = pr.ln();
            }
        }
    }
    let obs_beat = |a: f32| (a.clamp(1e-6, 1.0) as f64).ln();
    let obs_none = |a: f32| (((1.0 - a.clamp(0.0, 1.0 - 1e-6)) as f64) / (p.observation_lambda - 1.0)).ln();
    let is_beat_region = |pos: usize, i: usize| (pos as f64) < (i as f64 / p.observation_lambda).max(1.0);

    let mut score = vec![f64::NEG_INFINITY; total];
    // uniform start
    for (k, &i) in intervals.iter().enumerate() {
        for pos in 0..i {
            let s = offset[k] + pos;
            let o = if is_beat_region(pos, i) {
                obs_beat(act[0])
            } else {
                obs_none(act[0])
            };
            score[s] = -(total as f64).ln() + o;
        }
    }
    // back pointers: for position > 0 the predecessor is fixed (pos - 1, same interval); only
    // position 0 needs the arg-max over the previous beat's last state, so store just that.
    let mut back0: Vec<Vec<u16>> = Vec::with_capacity(n);
    let mut next = vec![f64::NEG_INFINITY; total];
    for &a in &act[1..] {
        let (ob, on) = (obs_beat(a), obs_none(a));
        let mut bp = vec![0u16; ni];
        for (k, &i) in intervals.iter().enumerate() {
            // advance within the beat
            for pos in 1..i {
                let s = offset[k] + pos;
                let o = if is_beat_region(pos, i) { ob } else { on };
                next[s] = score[s - 1] + o;
            }
            // start of a new beat: from the last position of any interval
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0usize;
            for (j, &ij) in intervals.iter().enumerate() {
                let c = score[offset[j] + ij - 1] + trans[j * ni + k];
                if c > best {
                    best = c;
                    arg = j;
                }
            }
            next[offset[k]] = best + ob;
            bp[k] = arg as u16;
        }
        back0.push(bp);
        std::mem::swap(&mut score, &mut next);
    }
    // backtrack
    let mut s = (0..total)
        .max_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap())
        .unwrap();
    let state_of = |s: usize| -> (usize, usize) {
        let k = offset.partition_point(|&o| o <= s) - 1;
        (k, s - offset[k])
    };
    let mut beats_frames = vec![];
    for t in (1..n).rev() {
        let (k, pos) = state_of(s);
        if pos == 0 {
            beats_frames.push(t);
            let j = back0[t - 1][k] as usize;
            s = offset[j] + intervals[j] - 1;
        } else {
            s -= 1;
        }
    }
    if state_of(s).1 == 0 {
        beats_frames.push(0);
    }
    beats_frames.reverse();
    // snap each beat to the activation peak nearby
    let mut out: Vec<f64> = beats_frames
        .iter()
        .map(|&f| {
            let lo = f.saturating_sub(p.correct);
            let hi = (f + p.correct + 1).min(n);
            // move only toward a strictly higher activation; on a flat stretch keep the decoded frame
            let mut best = f;
            for k in lo..hi {
                if act[k] > act[best] {
                    best = k;
                }
            }
            best as f64 / fps
        })
        .collect();
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_pulse_with_a_doubled_and_a_missing_peak() {
        // 120 bpm at 50 fps = a beat every 25 frames; add a spurious peak and drop one real one
        let fps = 50.0;
        let mut act = vec![0.02f32; 1000];
        for k in 0..40 {
            if k != 17 {
                act[k * 25] = 0.9;
            }
        }
        act[13 * 25 + 12] = 0.6; // spurious half-beat
        let beats = track(&act, fps, &DbnParams::default());
        assert!((beats.len() as i64 - 40).abs() <= 1, "{} beats", beats.len());
        let ibi: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(ibi.iter().all(|d| (d - 0.5).abs() < 0.06), "{ibi:?}");
    }
}
