//! Stage 5 maths: turn model activations into 50 ms x pitch cells, compare the render with the
//! stem, find the worst regions, draw a heatmap. Rendering and the model front end run outside
//! (FluidSynth and the Python worker); this module only compares.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use crate::npy::Array2;

/// seconds per comparison cell
pub const CELL: f64 = 0.05;
/// activation counted as "sounding"
pub const ON: f32 = 0.5;
pub const PITCHES: usize = 88;

/// Max-pool frame activations (frames x 88) into cells of CELL seconds.
pub fn to_cells(act: &Array2, fps: f64, n_cells: Option<usize>) -> Array2 {
    let step = fps * CELL;
    let n = n_cells.unwrap_or_else(|| (act.rows as f64 / step).ceil() as usize);
    let mut out = Array2::zeros(n, act.cols);
    for i in 0..n {
        let a = (i as f64 * step) as usize;
        let b = ((i + 1) as f64 * step) as usize;
        if a < act.rows {
            let end = b.max(a + 1).min(act.rows);
            for c in 0..act.cols {
                let mut m = f32::MIN;
                for r in a..end {
                    m = m.max(act.get(r, c));
                }
                out.set(i, c, m);
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Match {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub n: usize,
    pub cols: usize,
    /// sounding in the stem, absent from the render
    pub missed: Vec<bool>,
    /// in the render, absent from the stem
    pub extra: Vec<bool>,
}

fn dilate(x: &[bool], n: usize, cols: usize) -> Vec<bool> {
    let mut out = x.to_vec();
    for i in 0..n {
        for c in 0..cols {
            if x[i * cols + c] {
                if i + 1 < n {
                    out[(i + 1) * cols + c] = true;
                }
                if i > 0 {
                    out[(i - 1) * cols + c] = true;
                }
            }
        }
    }
    out
}

pub fn compare(stem: &Array2, render: &Array2) -> Match {
    let n = stem.rows.min(render.rows);
    let cols = stem.cols.min(render.cols);
    let mut s = vec![false; n * cols];
    let mut r = vec![false; n * cols];
    for i in 0..n {
        for c in 0..cols {
            s[i * cols + c] = stem.get(i, c) >= ON;
            r[i * cols + c] = render.get(i, c) >= ON;
        }
    }
    let (sd, rd) = (dilate(&s, n, cols), dilate(&r, n, cols));
    let (mut tp_r, mut tp_p, mut ns, mut nr) = (0usize, 0usize, 0usize, 0usize);
    let mut missed = vec![false; n * cols];
    let mut extra = vec![false; n * cols];
    for k in 0..n * cols {
        ns += s[k] as usize;
        nr += r[k] as usize;
        tp_r += (s[k] && rd[k]) as usize;
        tp_p += (r[k] && sd[k]) as usize;
        missed[k] = s[k] && !rd[k];
        extra[k] = r[k] && !sd[k];
    }
    let recall = tp_r as f64 / ns.max(1) as f64;
    let precision = tp_p as f64 / nr.max(1) as f64;
    let f1 = 2.0 * precision * recall / (precision + recall).max(1e-9);
    Match {
        precision,
        recall,
        f1,
        n,
        cols,
        missed,
        extra,
    }
}

/// Sub-block of rows [a, b) (for local comparisons and per-second scores).
pub fn rows(x: &Array2, a: usize, b: usize) -> Array2 {
    let b = b.min(x.rows);
    let a = a.min(b);
    Array2 {
        rows: b - a,
        cols: x.cols,
        data: x.data[a * x.cols..b * x.cols].to_vec(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Region {
    pub start: f64,
    pub end: f64,
    pub bad_cells: usize,
    pub missed_pitches: Vec<i32>,
    pub extra_pitches: Vec<i32>,
}

/// Windows with the most disagreement, worst first, non-overlapping.
pub fn regions(m: &Match, top: usize, window: f64) -> Vec<Region> {
    let per = (window / CELL) as usize;
    let mut out = vec![];
    let mut a = 0;
    while a < m.n {
        let b = (a + per).min(m.n);
        let (mut bad, mut mp, mut ep) = (0usize, vec![false; m.cols], vec![false; m.cols]);
        for i in a..b {
            for c in 0..m.cols {
                let k = i * m.cols + c;
                if m.missed[k] {
                    bad += 1;
                    mp[c] = true;
                }
                if m.extra[k] {
                    bad += 1;
                    ep[c] = true;
                }
            }
        }
        if bad >= 3 {
            let pick = |v: &[bool]| {
                (0..m.cols)
                    .filter(|&c| v[c])
                    .map(|c| c as i32 + 21)
                    .take(12)
                    .collect::<Vec<_>>()
            };
            out.push(Region {
                start: crate::play::round_to(a as f64 * CELL, 2),
                end: crate::play::round_to(b as f64 * CELL, 2),
                bad_cells: bad,
                missed_pitches: pick(&mp),
                extra_pitches: pick(&ep),
            });
        }
        a += per / 2;
    }
    out.sort_by(|x, y| y.bad_cells.cmp(&x.bad_cells)); // stable
    let mut kept: Vec<Region> = vec![];
    for r in out {
        if kept.iter().all(|k| r.end <= k.start || r.start >= k.end) {
            kept.push(r);
        }
        if kept.len() >= top {
            break;
        }
    }
    kept
}

/// The full comparison report for one track (same shape as the Python verify.json entry).
pub fn report(track: &str, stem: &Array2, render: &Array2) -> (serde_json::Value, Match) {
    let m = compare(stem, render);
    let k = (1.0 / CELL) as usize;
    let mut per_sec = vec![];
    let mut a = 0;
    while a < stem.rows {
        let (s, r) = (rows(stem, a, a + k), rows(render, a, a + k));
        let any = s.data.iter().any(|&x| x >= ON) || r.data.iter().any(|&x| x >= ON);
        per_sec.push(if any {
            Some(crate::play::round_to(compare(&s, &r).f1, 3))
        } else {
            None
        });
        a += k;
    }
    let rep = json!({
        "track": track,
        "precision": crate::play::round_to(m.precision, 4),
        "recall": crate::play::round_to(m.recall, 4),
        "audio_match": crate::play::round_to(m.f1, 4),
        "per_second": per_sec,
        "regions": regions(&m, 15, 2.0),
    });
    (rep, m)
}

/// Heatmap: grey = agrees, red = in the audio but missing from the score, blue = in the score
/// but not in the audio. One 60-second strip per row, 2 px per cell, pitch up.
pub fn heatmap(m: &Match, stem: &Array2, out: &Path) -> Result<()> {
    let per = (60.0 / CELL) as usize;
    let strips = m.n.div_ceil(per).max(1);
    let (w, h) = ((per * 2) as u32, (strips * (m.cols * 2 + 6)) as u32);
    let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
    for i in 0..m.n {
        let strip = i / per;
        let x = ((i % per) * 2) as u32;
        let y0 = (strip * (m.cols * 2 + 6)) as u32;
        for c in 0..m.cols {
            let k = i * m.cols + c;
            let col = if m.missed[k] {
                [215, 40, 40]
            } else if m.extra[k] {
                [40, 90, 215]
            } else if stem.get(i, c) >= ON {
                [190, 190, 190]
            } else {
                continue;
            };
            let y = y0 + ((m.cols - 1 - c) * 2) as u32;
            for dx in 0..2 {
                for dy in 0..2 {
                    img.put_pixel(x + dx, y + dy, image::Rgb(col));
                }
            }
        }
    }
    img.save(out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_maps_match_perfectly() {
        let mut a = Array2::zeros(40, 88);
        for i in 5..20 {
            a.set(i, 40, 0.9);
        }
        let m = compare(&a, &a);
        assert!((m.f1 - 1.0).abs() < 1e-12);
        assert!(regions(&m, 5, 2.0).is_empty());
    }

    #[test]
    fn missing_note_is_a_region() {
        let mut stem = Array2::zeros(80, 88);
        let render = Array2::zeros(80, 88);
        for i in 10..30 {
            stem.set(i, 39, 0.9); // C4
        }
        let m = compare(&stem, &render);
        assert_eq!(m.recall, 0.0);
        let r = regions(&m, 5, 2.0);
        assert_eq!(r[0].missed_pitches, vec![60]);
    }

    #[test]
    fn cells_max_pool() {
        let mut a = Array2::zeros(10, 2);
        a.set(3, 1, 0.7);
        let c = to_cells(&a, 100.0, None); // 5 frames per cell
        assert_eq!(c.rows, 2);
        assert_eq!(c.get(0, 1), 0.7);
    }
}
