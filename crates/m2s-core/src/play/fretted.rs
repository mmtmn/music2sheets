//! Fretted and bowed strings (guitar, bass, violin): string/fret assignment by Viterbi.
//!
//! State per time slice = one fingering (a string and fret for every note). Within a slice
//! the fretting hand must cover every fretted note (stretch); between slices it pays for
//! position shifts, scaled by how fast they must happen, plus string skips. Open strings are
//! cheap. Several players on one bus are split into lines first, like a piano's two hands.
//! Faithful port of `m2s/play/fretted.py`.

use anyhow::Result;
use serde_json::json;

use super::{make_slices, Violation};
use crate::ir::{Effector, Note, Track};
use crate::preset::{self, Fretting, Preset};
use crate::trace::Trace;

/// per note: (string index low->high, fret)
type Fingering = Vec<(usize, i32)>;

#[derive(Debug, Clone)]
pub struct FretWeights {
    pub stretch: f64,
    pub stretch_violation: f64,
    /// mild preference for lower positions
    pub position: f64,
    pub must_keep: f64,
    pub must_drop: f64,
    pub max_fingerings: usize,
}

impl Default for FretWeights {
    fn default() -> Self {
        FretWeights {
            stretch: 0.4,
            stretch_violation: 30.0,
            position: 0.03,
            must_keep: 0.6,
            must_drop: 0.3,
            max_fingerings: 24,
        }
    }
}

pub struct FrettedDecoder {
    p: &'static Preset,
    f: Fretting,
    tuning: Vec<i32>,
    pub w: FretWeights,
}

impl FrettedDecoder {
    pub fn new(name: &str) -> Result<Self> {
        let p = preset::load(name)?;
        Ok(FrettedDecoder {
            p,
            f: p.fretting()?.clone(),
            tuning: p.tuning.clone().unwrap_or_default(),
            w: FretWeights::default(),
        })
    }

    pub fn options(&self, pitches: &[i32]) -> Vec<Fingering> {
        let frets = self.p.frets.unwrap_or(24);
        let per_note: Vec<Vec<(usize, i32)>> = pitches
            .iter()
            .map(|&p| {
                let o: Vec<(usize, i32)> = self
                    .tuning
                    .iter()
                    .enumerate()
                    .filter(|(_, &o)| 0 <= p - o && p - o <= frets)
                    .map(|(s, &o)| (s, p - o))
                    .collect();
                if o.is_empty() {
                    vec![(0, (p - self.tuning[0]).max(0))]
                } else {
                    o
                }
            })
            .collect();
        let mut out: Vec<Fingering> = vec![];
        let mut cur: Fingering = vec![];
        product(&per_note, 0, &mut cur, &mut |fg| {
            let strings: Vec<usize> = fg.iter().map(|x| x.0).collect();
            let mut uniq = strings.clone();
            uniq.sort();
            uniq.dedup();
            if uniq.len() < strings.len() {
                return;
            }
            if self.p.adjacent_strings_only && strings.len() > 1 {
                let span = strings.iter().max().unwrap() - strings.iter().min().unwrap();
                if span > strings.len() - 1 {
                    return;
                }
            }
            out.push(fg.clone());
        });
        if out.is_empty() {
            out = vec![per_note.iter().map(|o| o[0]).collect()];
        }
        let costs: Vec<f64> = out.iter().map(|fg| self.chord_cost(fg)).collect();
        let mut order: Vec<usize> = (0..out.len()).collect();
        order.sort_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap()); // stable, like Python
        order
            .into_iter()
            .take(self.w.max_fingerings)
            .map(|i| out[i].clone())
            .collect()
    }

    fn stretch_limit(&self, low_fret: i32) -> i32 {
        if low_fret >= self.f.high_position {
            self.f.max_stretch_high
        } else {
            self.f.max_stretch
        }
    }

    fn chord_cost(&self, fg: &Fingering) -> f64 {
        let fretted: Vec<i32> = fg.iter().map(|x| x.1).filter(|&f| f > 0).collect();
        let mut cost = self.w.position * fretted.iter().min().copied().unwrap_or(0) as f64;
        cost -= self.f.open_string_bonus * fg.iter().filter(|x| x.1 == 0).count() as f64;
        if !fretted.is_empty() {
            let (mn, mx) = (*fretted.iter().min().unwrap(), *fretted.iter().max().unwrap());
            let st = mx - mn;
            let lim = self.stretch_limit(mn);
            cost += self.w.stretch * st as f64;
            if st > lim {
                cost += self.w.stretch_violation + 3.0 * (st - lim) as f64;
            }
        }
        let mut s: Vec<usize> = fg.iter().map(|x| x.0).collect();
        s.sort();
        s.dedup();
        if s.len() < fg.len() {
            cost += 50.0;
        }
        cost
    }

    fn position(fg: &Fingering, prev: Option<f64>) -> f64 {
        match fg.iter().map(|x| x.1).filter(|&f| f > 0).min() {
            Some(m) => m as f64,
            None => prev.unwrap_or(0.0),
        }
    }

    fn trans_cost(&self, a: &Fingering, pa: f64, b: &Fingering, pb: f64, dt: f64) -> f64 {
        let shift = (pb - pa).abs();
        let mut cost = self.f.shift_cost * shift;
        let dt = dt.max(1e-3);
        if shift > 2.0 && shift / dt > self.f.shift_speed {
            cost += 15.0;
        }
        if a.len() == 1 && b.len() == 1 {
            let skip = (a[0].0 as i64 - b[0].0 as i64).abs();
            if skip > 1 {
                cost += self.f.string_skip_cost * (skip - 1) as f64;
            }
        }
        cost
    }

    pub fn decode(&self, track: &mut Track, trace: &mut Trace) -> Result<Vec<Violation>> {
        for n in track.notes.iter_mut() {
            n.keep = n.confidence >= 0.5;
            if n.confidence < self.w.must_drop {
                n.keep = false;
            }
        }
        let players = if track.players <= 1 {
            self.f.players
        } else {
            track.players
        };
        if let Some(k) = self.p.max_simultaneous {
            let kept: Vec<usize> = (0..track.notes.len()).filter(|&i| track.notes[i].keep).collect();
            for sl in make_slices(&track.notes, &kept, 0.03) {
                if sl.len() > k {
                    let mut by_conf = sl.clone();
                    by_conf.sort_by(|&a, &b| {
                        track.notes[b]
                            .confidence
                            .partial_cmp(&track.notes[a].confidence)
                            .unwrap()
                    });
                    for &i in &by_conf[k..] {
                        track.notes[i].keep = false;
                        track.notes[i].tags.push("over-polyphony".into());
                    }
                }
            }
        }
        let kept: Vec<usize> = (0..track.notes.len()).filter(|&i| track.notes[i].keep).collect();
        let lines = if players > 1 {
            split_players(&track.notes, &kept, players as usize)
        } else {
            vec![kept]
        };
        let mut out = vec![];
        for (pi, line) in lines.iter().enumerate() {
            for &i in line {
                track.notes[i].player = pi as i32;
            }
            out.extend(self.decode_line(&mut track.notes, line, pi as u32, trace));
        }
        Ok(out)
    }

    fn decode_line(&self, notes: &mut [Note], line: &[usize], player: u32, trace: &mut Trace) -> Vec<Violation> {
        let slices = make_slices(notes, line, 0.03);
        if slices.is_empty() {
            return vec![];
        }
        let opts: Vec<Vec<Fingering>> = slices
            .iter()
            .map(|s| self.options(&s.iter().map(|&i| notes[i].pitch).collect::<Vec<_>>()))
            .collect();
        let mut cost: Vec<f64> = opts[0].iter().map(|fg| self.chord_cost(fg)).collect();
        let mut pos: Vec<f64> = opts[0].iter().map(|fg| Self::position(fg, None)).collect();
        let mut back: Vec<Vec<usize>> = vec![];
        for i in 1..slices.len() {
            let dt = notes[slices[i][0]].onset - notes[slices[i - 1][0]].onset;
            let (mut nc, mut npos, mut bp) = (vec![], vec![], vec![]);
            for fb in &opts[i] {
                let within = self.chord_cost(fb);
                let (mut best, mut arg, mut bpos) = (f64::INFINITY, 0usize, 0.0);
                for (j, fa) in opts[i - 1].iter().enumerate() {
                    let pb = Self::position(fb, Some(pos[j]));
                    let c = cost[j] + self.trans_cost(fa, pos[j], fb, pb, dt);
                    if c < best {
                        best = c;
                        arg = j;
                        bpos = pb;
                    }
                }
                nc.push(best + within);
                npos.push(bpos);
                bp.push(arg);
            }
            cost = nc;
            pos = npos;
            back.push(bp);
        }
        let mut k = 0;
        for (j, &c) in cost.iter().enumerate() {
            if c < cost[k] {
                k = j;
            }
        }
        let mut path = vec![k];
        for bp in back.iter().rev() {
            k = bp[k];
            path.push(k);
        }
        path.reverse();
        let nstr = self.tuning.len();
        let mut out = vec![];
        let (mut prev_pos, mut prev_t): (Option<i32>, f64) = (None, 0.0);
        let who = format!("player {player}");
        for ((s, o), &k) in slices.iter().zip(opts.iter()).zip(path.iter()) {
            let fg = &o[k];
            for (&i, &(si, fret)) in s.iter().zip(fg.iter()) {
                notes[i].effector = Effector {
                    string: Some((nstr - si) as u8),
                    fret: Some(fret),
                    player: Some(player),
                    ..Default::default()
                };
            }
            let fretted: Vec<i32> = fg.iter().map(|x| x.1).filter(|&f| f > 0).collect();
            let ids = || s.iter().map(|&i| notes[i].id.clone()).collect::<Vec<_>>();
            let t = notes[s[0]].onset;
            if !fretted.is_empty() {
                let (mn, mx) = (*fretted.iter().min().unwrap(), *fretted.iter().max().unwrap());
                if mx - mn > self.stretch_limit(mn) {
                    out.push(Violation::new(
                        "stretch",
                        t,
                        format!("{}-fret stretch", mx - mn),
                        ids(),
                        &who,
                    ));
                }
                if let Some(pp) = prev_pos {
                    if (mn - pp).abs() > 2 {
                        let dt = (t - prev_t).max(1e-3);
                        if (mn - pp).abs() as f64 / dt > self.f.shift_speed {
                            out.push(Violation::new(
                                "shift",
                                t,
                                format!("shift {} frets in {:.0} ms", (mn - pp).abs(), dt * 1000.0),
                                ids(),
                                &who,
                            ));
                        }
                    }
                }
                prev_pos = Some(mn);
                prev_t = t;
            }
            if s.len() > nstr {
                out.push(Violation::new(
                    "count",
                    t,
                    format!("{} notes on {nstr} strings", s.len()),
                    ids(),
                    "",
                ));
            }
            trace.with("fingering", || {
                json!({"t": t, "player": player, "notes": ids(),
                                               "frets": fg.iter().map(|&(si, f)| (nstr - si, f)).collect::<Vec<_>>()})
            });
        }
        out
    }
}

fn product(per: &[Vec<(usize, i32)>], i: usize, cur: &mut Fingering, f: &mut dyn FnMut(&Fingering)) {
    if i == per.len() {
        f(cur);
        return;
    }
    for &o in &per[i] {
        cur.push(o);
        product(per, i + 1, cur, f);
        cur.pop();
    }
}

/// Split a bus into lines: in chords the top note goes to player 1, the next to player 2;
/// single notes follow whichever line they continue most smoothly.
pub fn split_players(notes: &[Note], idx: &[usize], players: usize) -> Vec<Vec<usize>> {
    let mut lines: Vec<Vec<usize>> = vec![vec![]; players];
    let mut last: Vec<Option<i32>> = vec![None; players];
    for mut sl in make_slices(notes, idx, 0.03) {
        sl.sort_by(|&a, &b| notes[b].pitch.cmp(&notes[a].pitch));
        if sl.len() >= players {
            for (i, &n) in sl.iter().enumerate() {
                let li = i.min(players - 1);
                lines[li].push(n);
                last[li] = Some(notes[n].pitch);
            }
        } else {
            for &n in &sl {
                let p = notes[n].pitch;
                let mut li = 0;
                let key = |i: usize| last[i].map(|l| (p - l).abs()).unwrap_or(0);
                for i in 1..players {
                    if key(i) < key(li) {
                        li = i;
                    }
                }
                lines[li].push(n);
                last[li] = Some(p);
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(i: usize, t: f64, p: i32, d: f64) -> Note {
        Note::new(format!("n{i}"), t, t + d, p, 80)
    }

    #[test]
    fn open_e_minor_chord() {
        let mut tr = Track::new(
            "g",
            "guitar",
            [40, 47, 52, 55, 59, 64]
                .iter()
                .enumerate()
                .map(|(i, &p)| n(i, 0.0, p, 1.0))
                .collect(),
        );
        let v = FrettedDecoder::new("guitar")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        let tuning = [40, 45, 50, 55, 59, 64];
        let mut strings: Vec<u8> = tr.notes.iter().map(|x| x.effector.string.unwrap()).collect();
        for x in &tr.notes {
            assert_eq!(
                tuning[6 - x.effector.string.unwrap() as usize] + x.effector.fret.unwrap(),
                x.pitch
            );
        }
        strings.sort();
        strings.dedup();
        assert_eq!(strings.len(), 6);
        assert!(!v.iter().any(|x| x.kind == "stretch"));
    }

    #[test]
    fn violin_double_stop_adjacent() {
        let mut tr = Track::new("v", "violin", vec![n(0, 0.0, 62, 0.5), n(1, 0.0, 69, 0.5)]);
        FrettedDecoder::new("violin")
            .unwrap()
            .decode(&mut tr, &mut Trace::off())
            .unwrap();
        let mut s: Vec<u8> = tr.notes.iter().map(|x| x.effector.string.unwrap()).collect();
        s.sort();
        assert_eq!(s[1] - s[0], 1);
    }
}
