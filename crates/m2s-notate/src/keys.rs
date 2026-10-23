//! Key estimation from notes (Krumhansl-Kessler profiles, duration-weighted). Port of `keys.py`.

use m2s_core::ir::Note;

use crate::np;
use crate::spell::fifths_name;

pub const KK_MAJOR: [f64; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
pub const KK_MINOR: [f64; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Major,
    Minor,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Major => "major",
            Mode::Minor => "minor",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Key {
    pub tonic_pc: i32,
    pub mode: Mode,
    pub score: f64,
}

impl Key {
    pub fn tonic_fifths(&self) -> i32 {
        let pc = self.tonic_pc.rem_euclid(12);
        match self.mode {
            Mode::Major => [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5][pc as usize],
            Mode::Minor => [0, 7, 2, -3, 4, -1, 6, 1, 8, 3, -2, 5][pc as usize],
        }
    }
    /// Key signature: + sharps / - flats.
    pub fn sharps(&self) -> i32 {
        self.tonic_fifths() - if self.mode == Mode::Major { 0 } else { 3 }
    }
    /// Centre of the key on the line of fifths (minor leans sharp for its raised 6th and 7th).
    pub fn centre(&self) -> f64 {
        self.tonic_fifths() as f64 + if self.mode == Mode::Major { 2.0 } else { -0.5 }
    }
    pub fn name(&self) -> String {
        format!("{} {}", fifths_name(self.tonic_fifths()), self.mode.name())
    }
}

pub fn histogram(notes: &[&Note], bass_weight: f64) -> [f64; 12] {
    let mut h = [0.0; 12];
    if notes.is_empty() {
        return h;
    }
    let pitches: Vec<i32> = notes.iter().map(|n| n.pitch).collect();
    let lo = np::percentile_i(&pitches, 15.0);
    for n in notes {
        let mut w = n.duration().min(2.0) + 0.05;
        if n.pitch as f64 <= lo {
            w *= 1.0 + bass_weight;
        }
        h[n.pitch.rem_euclid(12) as usize] += w;
    }
    h
}

/// `np.corrcoef(x, y)[0, 1]`.
fn corrcoef(x: &[f64; 12], y: &[f64; 12]) -> f64 {
    let mx = np::mean(x);
    let my = np::mean(y);
    let xc: Vec<f64> = x.iter().map(|v| v - mx).collect();
    let yc: Vec<f64> = y.iter().map(|v| v - my).collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).fold(0.0, |s, v| s + v);
    let c01 = dot(&xc, &yc) / 11.0;
    let s0 = (dot(&xc, &xc) / 11.0).sqrt();
    let s1 = (dot(&yc, &yc) / 11.0).sqrt();
    (c01 / s0 / s1).clamp(-1.0, 1.0)
}

fn roll(p: &[f64; 12], k: i32) -> [f64; 12] {
    let mut o = [0.0; 12];
    for (i, v) in o.iter_mut().enumerate() {
        *v = p[(i as i32 - k).rem_euclid(12) as usize];
    }
    o
}

pub fn estimate_key(notes: &[&Note], bass: &[&Note]) -> Key {
    let h = histogram(notes, 0.5);
    let mut best = Key {
        tonic_pc: 0,
        mode: Mode::Major,
        score: -2.0,
    };
    let total = np::pairwise_sum(&h);
    for pc in 0..12 {
        for (mode, prof) in [(Mode::Major, &KK_MAJOR), (Mode::Minor, &KK_MINOR)] {
            let r = if total != 0.0 {
                corrcoef(&h, &roll(prof, pc))
            } else {
                0.0
            };
            if r > best.score {
                best = Key {
                    tonic_pc: pc,
                    mode,
                    score: r,
                };
            }
        }
    }
    if bass.is_empty() {
        let low = low_notes(notes);
        relative_by_bass(best, &low)
    } else {
        relative_by_bass(best, bass)
    }
}

fn low_notes<'a>(notes: &[&'a Note]) -> Vec<&'a Note> {
    if notes.is_empty() {
        return vec![];
    }
    let pitches: Vec<i32> = notes.iter().map(|n| n.pitch).collect();
    let lo = np::percentile_i(&pitches, 20.0);
    notes.iter().filter(|n| n.pitch as f64 <= lo).copied().collect()
}

/// Relative major and minor share their notes; the bass decides which tonic is home.
fn relative_by_bass(k: Key, bass: &[&Note]) -> Key {
    if bass.is_empty() {
        return k;
    }
    let rel = match k.mode {
        Mode::Major => Key {
            tonic_pc: (k.tonic_pc + 9) % 12,
            mode: Mode::Minor,
            score: k.score,
        },
        Mode::Minor => Key {
            tonic_pc: (k.tonic_pc + 3) % 12,
            mode: Mode::Major,
            score: k.score,
        },
    };
    let mut order: Vec<&&Note> = bass.iter().collect();
    order.sort_by(|a, b| a.onset.partial_cmp(&b.onset).unwrap());
    let weight = |pc: i32| -> f64 {
        let mut w = 0.0;
        for n in bass {
            if n.pitch.rem_euclid(12) == pc {
                w += n.duration().min(2.0);
            }
        }
        w += 2.0 * f64::from(u8::from(order[0].pitch.rem_euclid(12) == pc));
        w += 3.0 * f64::from(u8::from(order[order.len() - 1].pitch.rem_euclid(12) == pc));
        w
    };
    if weight(rel.tonic_pc) > 1.25 * weight(k.tonic_pc) {
        rel
    } else {
        k
    }
}
