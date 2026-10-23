//! Voices, measures, clefs, dynamics and parts (port of `m2s/notate/build.py`), as a plain
//! score model that `musicxml` writes and `lint` checks.

use std::collections::{BTreeMap, HashMap, HashSet};

use m2s_core::ir::{Hand, Piece};
use m2s_core::preset::{Family, Preset};
use m2s_core::trace::Trace;
use num_rational::Ratio;
use num_traits::{Signed, Zero};
use serde_json::json;

use crate::np;
use crate::rhythm::Grid;
use crate::Options;

pub type Q = Ratio<i64>;

pub fn q(n: i64, d: i64) -> Q {
    Q::new(n, d)
}

/// `Fraction(round(x * 48), 48)`.
pub fn frac(x: f64) -> Q {
    Q::new(np::round_i(x * 48.0), 48)
}

pub fn qf(x: Q) -> f64 {
    *x.numer() as f64 / *x.denom() as f64
}

/// A note in the piece: (track index, note index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NRef {
    pub t: usize,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub on: Q,
    pub dur: Q,
    pub notes: Vec<NRef>,
    pub staccato: bool,
    pub graces: Vec<NRef>,
}

fn nice() -> [Q; 10] {
    [
        q(4, 1),
        q(3, 1),
        q(2, 1),
        q(3, 2),
        q(1, 1),
        q(3, 4),
        q(1, 2),
        q(3, 8),
        q(1, 4),
        q(1, 8),
    ]
}
fn nice_triplet() -> [Q; 3] {
    [q(2, 3), q(1, 3), q(1, 6)]
}
fn standard_and_triplets() -> Vec<Q> {
    vec![
        q(4, 1),
        q(3, 1),
        q(2, 1),
        q(3, 2),
        q(1, 1),
        q(3, 4),
        q(1, 2),
        q(3, 8),
        q(1, 4),
        q(1, 8),
        q(1, 16),
        q(2, 3),
        q(1, 3),
        q(1, 6),
        q(1, 12),
    ]
}

/// Round a duration to a plain value, never past `limit`.
pub fn nice_duration(on: Q, dur: Q, limit: Option<Q>) -> Q {
    let triplet = (on * 3).is_integer() && on.denom() % 3 == 0;
    let mut pool: Vec<Q> = nice().to_vec();
    if triplet {
        pool.extend(nice_triplet());
    }
    if let Some(l) = limit {
        pool.retain(|d| *d <= l);
        if pool.is_empty() {
            pool.push(l);
        }
    }
    let mut best = pool[0];
    for &d in &pool[1..] {
        let (bd, dd) = ((best - dur).abs(), (d - dur).abs());
        if dd < bd || (dd == bd && -d < -best) {
            best = d;
        }
    }
    best
}

/// Split one staff's notes into at most two voices (see `staff_voices` in build.py).
pub fn staff_voices(notes: &[NRef], piece: &Piece, trace: &mut Trace) -> Vec<Vec<Event>> {
    let note = |r: &NRef| &piece.tracks[r.t].notes[r.n];
    let mut graces: Vec<NRef> = notes.iter().copied().filter(|r| note(r).has_tag("grace")).collect();
    graces.sort_by(|a, b| note(a).onset.partial_cmp(&note(b).onset).unwrap());
    let plain: Vec<NRef> = notes.iter().copied().filter(|r| !note(r).has_tag("grace")).collect();
    let mut groups: BTreeMap<Q, Vec<NRef>> = BTreeMap::new();
    for r in plain {
        groups.entry(frac(note(&r).beat.unwrap())).or_default().push(r);
    }
    let onsets: Vec<Q> = groups.keys().copied().collect();
    let mut v1: Vec<Event> = vec![];
    let mut v2: Vec<Event> = vec![];
    for (i, &on) in onsets.iter().enumerate() {
        let mut chord_notes = groups[&on].clone();
        chord_notes.sort_by_key(|r| note(r).pitch);
        let ioi = onsets.get(i + 1).map(|&nx| nx - on);
        if let Some(ioi) = ioi {
            let qd = |r: &NRef| frac(note(r).qdur.unwrap());
            let held: Vec<NRef> = chord_notes
                .iter()
                .copied()
                .filter(|r| qd(r) >= ioi + q(1, 1) && qd(r) >= ioi * 2)
                .collect();
            if !held.is_empty() {
                let rest: Vec<NRef> = chord_notes.iter().copied().filter(|r| !held.contains(r)).collect();
                let mind = held.iter().map(qd).min().unwrap();
                trace.with(
                    "voice2",
                    || json!({"on": qf(on), "notes": held.iter().map(|r| note(r).id.clone()).collect::<Vec<_>>()}),
                );
                v2.push(Event {
                    on,
                    dur: nice_duration(on, mind, None),
                    notes: held,
                    staccato: false,
                    graces: vec![],
                });
                chord_notes = rest;
            }
        }
        if chord_notes.is_empty() {
            continue;
        }
        let mut dur = chord_notes.iter().map(|r| frac(note(r).qdur.unwrap())).min().unwrap();
        let mut stacc = false;
        match ioi {
            Some(ioi) => {
                if dur >= ioi {
                    dur = ioi;
                } else {
                    let gap = ioi - dur;
                    if dur <= q(3, 5) * ioi && ioi <= q(1, 1) {
                        stacc = true;
                        dur = ioi;
                    } else if gap <= q(1, 4) && gap < ioi / 2 {
                        dur = ioi;
                    } else {
                        dur = nice_duration(on, dur, Some(ioi));
                    }
                }
            }
            None => dur = nice_duration(on, dur, None),
        }
        v1.push(Event {
            on,
            dur: dur.max(q(1, 48)),
            notes: chord_notes,
            staccato: stacc,
            graces: vec![],
        });
    }
    for g in graces {
        let go = note(&g).onset;
        if let Some(e) = v1.iter_mut().find(|e| note(&e.notes[0]).onset >= go - 0.01) {
            e.graces.push(g);
        }
    }
    v2.sort_by(|a, b| a.on.cmp(&b.on));
    for i in 0..v2.len().saturating_sub(1) {
        if v2[i].on + v2[i].dur > v2[i + 1].on {
            v2[i].dur = v2[i + 1].on - v2[i].on;
        }
    }
    v2.retain(|e| e.dur > Q::zero());
    if v2.is_empty() {
        vec![v1]
    } else {
        vec![v1, v2]
    }
}

fn aligned(pos: Q, d: Q) -> bool {
    if d.denom() % 3 == 0 {
        let unit = if d >= q(1, 3) { q(1, 3) } else { d };
        return (pos / unit).is_integer() || (pos * 6).is_integer();
    }
    if d >= q(1, 1) {
        let m = if d < q(2, 1) { d } else { q(2, 1) };
        return (pos / m).is_integer() || ([q(1, 1), q(3, 2), q(2, 1), q(3, 1)].contains(&d) && pos.is_integer());
    }
    (pos / d).is_integer() || (d == q(3, 4) && (pos * 4).is_integer())
}

/// Break a duration into notatable pieces (tied).
pub fn split_duration(start: Q, dur: Q) -> Vec<Q> {
    let all = standard_and_triplets();
    let mut out = vec![];
    let (mut pos, mut left) = (start, dur);
    let mut guard = 0;
    while left > Q::zero() {
        let piece = all
            .iter()
            .copied()
            .find(|&c| c <= left && aligned(pos, c))
            .unwrap_or_else(|| all.iter().copied().find(|&c| c <= left).unwrap_or(left));
        out.push(piece);
        pos += piece;
        left -= piece;
        guard += 1;
        if guard > 10_000 {
            break;
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Bars {
    pub bar: Q,
    pub pickup: Q,
    pub starts: Vec<Q>,
    pub ends: Vec<Q>,
}

impl Bars {
    pub fn new(grid: &Grid, end_q: Q) -> Bars {
        let bar = frac(grid.bar_quarters);
        let pickup = frac(grid.pickup_quarters);
        let start = -pickup;
        let mut starts = vec![];
        let mut x = start;
        while x < end_q || starts.is_empty() {
            starts.push(x);
            x += if x == start && !pickup.is_zero() { pickup } else { bar };
            if starts.len() > 100_000 {
                break;
            }
        }
        let mut ends: Vec<Q> = starts[1..].to_vec();
        ends.push(starts[starts.len() - 1] + bar);
        Bars {
            bar,
            pickup,
            starts,
            ends,
        }
    }

    pub fn index(&self, x: Q) -> usize {
        let i = self.starts.partition_point(|s| *s <= x) as i64 - 1;
        i.clamp(0, self.starts.len() as i64 - 1) as usize
    }

    /// The printed measure number of bar index `bi`.
    pub fn number(&self, bi: usize) -> i64 {
        bi as i64 + if self.pickup.is_zero() { 1 } else { 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tie {
    Start,
    Stop,
    Continue,
}

/// One written element: a note, chord, rest or grace note.
#[derive(Debug, Clone)]
pub struct El {
    pub dur: Q,
    pub grace: bool,
    pub rest: bool,
    pub measure_rest: bool,
    pub notes: Vec<NRef>,
    pub tie: Option<Tie>,
    pub staccato: bool,
    pub trill: bool,
    pub sf: bool,
    pub fingering: bool,
    pub pedal_start: bool,
    pub pedal_stop: bool,
}

impl El {
    fn rest(dur: Q) -> El {
        El {
            dur,
            grace: false,
            rest: true,
            measure_rest: false,
            notes: vec![],
            tie: None,
            staccato: false,
            trill: false,
            sf: false,
            fingering: false,
            pedal_start: false,
            pedal_stop: false,
        }
    }
    fn notes(dur: Q, notes: Vec<NRef>, tie: Option<Tie>) -> El {
        El {
            rest: false,
            notes,
            tie,
            ..El::rest(dur)
        }
    }
}

/// Place a voice's events into bars, splitting and tying at barlines and odd values.
/// Returns the per-bar items and, per event index, the (bar, item index) of its first piece.
#[allow(clippy::type_complexity)]
pub fn fill_voice(
    events: &[Event],
    bars: &Bars,
    perc: bool,
    piece: &Piece,
    tab: bool,
    opts: &Options,
) -> (Vec<Vec<(Q, El)>>, Vec<Option<(usize, usize)>>) {
    let mut per_bar: Vec<Vec<(Q, El)>> = vec![vec![]; bars.starts.len()];
    let mut firsts = vec![None; events.len()];
    for (ei, ev) in events.iter().enumerate() {
        let (mut pos, mut left) = (ev.on, ev.dur);
        let mut first = true;
        while left > Q::zero() {
            let bi = bars.index(pos);
            let room = bars.ends[bi] - pos;
            if room <= Q::zero() {
                break;
            }
            let seg = left.min(room);
            let pieces = split_duration(pos - bars.starts[bi], seg);
            for (k, &d) in pieces.iter().enumerate() {
                let last_piece = left - d <= Q::zero();
                let is_first = first && k == 0;
                let tie = if !is_first && !last_piece {
                    Some(Tie::Continue)
                } else if !is_first {
                    Some(Tie::Stop)
                } else if !last_piece {
                    Some(Tie::Start)
                } else {
                    None
                };
                let mut el = El::notes(d, ev.notes.clone(), tie);
                if is_first {
                    let n0 = |r: &NRef| &piece.tracks[r.t].notes[r.n];
                    if !tab {
                        el.staccato = ev.staccato;
                        el.trill = ev.notes.iter().any(|r| n0(r).has_tag("trill"));
                    }
                    if !tab && opts.fingering && !perc {
                        el.fingering = ev.notes.iter().all(|r| n0(r).effector.finger.unwrap_or(0) != 0);
                    }
                    if !perc {
                        for &g in &ev.graces {
                            let mut gn = El::notes(q(1, 2), vec![g], None);
                            gn.grace = true;
                            per_bar[bi].push((pos - bars.starts[bi], gn));
                        }
                    }
                    firsts[ei] = Some((bi, per_bar[bi].len()));
                }
                per_bar[bi].push((pos - bars.starts[bi], el));
                pos += d;
                left -= d;
            }
            first = false;
        }
    }
    (per_bar, firsts)
}

pub fn with_rests(items: Vec<(Q, El)>, length: Q, start: Q) -> Vec<(Q, El)> {
    let mut items = items;
    items.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = vec![];
    let mut pos = Q::zero();
    for (off, el) in items {
        if off > pos {
            for d in split_duration(start + pos, off - pos) {
                out.push((pos, El::rest(d)));
                pos += d;
            }
        }
        let grace = el.grace;
        let end = off + el.dur;
        out.push((off, el));
        if !grace {
            pos = pos.max(end);
        }
    }
    if pos < length {
        if pos.is_zero() {
            let mut r = El::rest(length);
            r.measure_rest = true;
            return vec![(Q::zero(), r)];
        }
        for d in split_duration(start + pos, length - pos) {
            out.push((pos, El::rest(d)));
            pos += d;
        }
    }
    out
}

/// Treble or bass per bar by dynamic programming: fewest ledger lines, few clef changes.
pub fn clef_plan(events: &[&Event], bars: &Bars, default: &str, piece: &Piece) -> Vec<String> {
    let switch_cost = 8.0;
    let mut per_bar: HashMap<usize, Vec<i32>> = HashMap::new();
    for ev in events {
        per_bar
            .entry(bars.index(ev.on))
            .or_default()
            .extend(ev.notes.iter().map(|r| piece.tracks[r.t].notes[r.n].pitch));
    }
    let names = ["treble", "bass"];
    let mid = |c: &str| if c == "treble" { 71 } else { 50 };
    let bar_cost = |c: &str, ps: &[i32]| -> f64 {
        let mut cost = 0.0;
        for &p in ps {
            let over = (p - mid(c)).abs() - 9;
            if over > 0 {
                let lines = over as f64 / 3.5;
                cost += lines + if lines > 3.0 { 2.0 * (lines - 3.0) } else { 0.0 };
            }
        }
        cost + if c != default && !ps.is_empty() { 0.3 } else { 0.0 }
    };
    let empty = vec![];
    let n = bars.starts.len();
    let mut costs: Vec<f64> = names
        .iter()
        .map(|c| (if *c == default { 0.0 } else { switch_cost }) + bar_cost(c, per_bar.get(&0).unwrap_or(&empty)))
        .collect();
    let mut back: Vec<[usize; 2]> = vec![];
    for i in 1..n {
        let mut nc = [0.0; 2];
        let mut bp = [0usize; 2];
        for (ci, c) in names.iter().enumerate() {
            let tot = |pi: usize| costs[pi] + if pi == ci { 0.0 } else { switch_cost };
            let best = if tot(1) < tot(0) { 1 } else { 0 };
            nc[ci] = costs[best]
                + (if best == ci { 0.0 } else { switch_cost })
                + bar_cost(c, per_bar.get(&i).unwrap_or(&empty));
            bp[ci] = best;
        }
        costs = nc.to_vec();
        back.push(bp);
    }
    let mut cur = if costs[1] < costs[0] { 1 } else { 0 };
    let mut plan = vec![cur];
    for bp in back.iter().rev() {
        cur = bp[cur];
        plan.push(cur);
    }
    plan.reverse();
    plan.into_iter().map(|i| names[i].to_string()).collect()
}

const LEVELS: [(f64, &str); 6] = [
    (40.0, "pp"),
    (54.0, "p"),
    (66.0, "mp"),
    (78.0, "mf"),
    (92.0, "f"),
    (128.0, "ff"),
];

fn level(m: f64) -> &'static str {
    LEVELS.iter().find(|(top, _)| m < *top).map(|(_, n)| *n).unwrap_or("ff")
}

pub fn dynamic_plan(notes: &[NRef], bars: &Bars, piece: &Piece) -> BTreeMap<usize, String> {
    let mut vel: HashMap<usize, Vec<i32>> = HashMap::new();
    for r in notes {
        let n = &piece.tracks[r.t].notes[r.n];
        vel.entry(bars.index(frac(n.beat.unwrap())))
            .or_default()
            .push(n.velocity);
    }
    let nb = bars.starts.len();
    let means: Vec<Option<f64>> = (0..nb)
        .map(|i| {
            vel.get(&i)
                .map(|v| v.iter().map(|&x| x as i64).sum::<i64>() as f64 / v.len() as f64)
        })
        .collect();
    let smooth: Vec<Option<f64>> = (0..nb)
        .map(|i| {
            let mut w: Vec<f64> = means[i.saturating_sub(1)..(i + 2).min(nb)]
                .iter()
                .flatten()
                .copied()
                .collect();
            if w.is_empty() {
                None
            } else {
                w.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Some(w[w.len() / 2])
            }
        })
        .collect();
    let mut plan = BTreeMap::new();
    let mut cur: Option<&str> = None;
    for i in 0..nb {
        let Some(m) = smooth[i] else { continue };
        let lvl = level(m);
        let nxt = if i + 1 < nb { smooth[i + 1] } else { Some(m) };
        let hold = match nxt {
            None => true,
            Some(x) => level(x) == lvl,
        };
        if Some(lvl) != cur && (hold || cur.is_none()) {
            plan.insert(i, lvl.to_string());
            cur = Some(lvl);
        }
    }
    plan
}

#[derive(Debug, Clone, PartialEq)]
pub enum StaffKind {
    Pitched,
    Perc,
    Tab(usize),
}

#[derive(Debug, Clone)]
pub struct StaffSpec {
    pub name: String,
    pub notes: Vec<NRef>,
    pub clef: String,
    pub clef_auto: bool,
    pub kind: StaffKind,
    pub dynamics: bool,
    pub pedal: Vec<[f64; 2]>,
}

pub fn staves_for_track(t: usize, piece: &Piece, preset: &Preset) -> Vec<StaffSpec> {
    let track = &piece.tracks[t];
    let kept: Vec<NRef> = (0..track.notes.len())
        .filter(|&n| track.notes[n].keep && track.notes[n].beat.is_some())
        .map(|n| NRef { t, n })
        .collect();
    let spec = |name: &str, notes: Vec<NRef>, clef: &str, clef_auto: bool, kind: StaffKind, dynamics: bool| StaffSpec {
        name: name.to_string(),
        notes,
        clef: clef.to_string(),
        clef_auto,
        kind,
        dynamics,
        pedal: vec![],
    };
    match preset.family {
        Family::Keyboard => {
            let hand = |r: &NRef| track.notes[r.n].effector.hand;
            let rh: Vec<NRef> = kept
                .iter()
                .copied()
                .filter(|r| hand(r).unwrap_or(Hand::R) == Hand::R)
                .collect();
            let lh: Vec<NRef> = kept.iter().copied().filter(|r| hand(r) == Some(Hand::L)).collect();
            let mut l = spec("LH", lh, "bass", true, StaffKind::Pitched, false);
            l.pedal = track.pedal.clone();
            vec![spec("RH", rh, "treble", true, StaffKind::Pitched, true), l]
        }
        Family::Drums => vec![spec("Drums", kept, "perc", false, StaffKind::Perc, true)],
        _ => {
            let mut c = preset.clef.clone().unwrap_or_else(|| "treble".into());
            if c == "auto" {
                let mut ps: Vec<i32> = kept.iter().map(|r| track.notes[r.n].pitch).collect();
                ps.sort();
                if ps.is_empty() {
                    ps.push(60);
                }
                c = if ps[ps.len() / 2] >= 57 {
                    "treble".into()
                } else {
                    "treble8vb".into()
                };
            }
            let mut specs = vec![spec(&preset.display, kept.clone(), &c, false, StaffKind::Pitched, true)];
            if preset.tab {
                let strings = preset.tuning.as_ref().map(|t| t.len()).unwrap_or(6);
                specs.push(spec(
                    &format!("{} tab", preset.display),
                    kept,
                    "tab",
                    false,
                    StaffKind::Tab(strings),
                    false,
                ));
            }
            specs
        }
    }
}

#[derive(Debug, Clone)]
pub struct StaffMeasure {
    pub clef: Option<String>,
    pub dynamic: Option<String>,
    pub voices: Vec<Vec<(Q, El)>>,
}

#[derive(Debug, Clone)]
pub struct StaffOut {
    pub id: String,
    pub kind: StaffKind,
    pub measures: Vec<StaffMeasure>,
    /// key signature (sharps) for pitched staves
    pub key: Option<i32>,
    pub tempo: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct PartOut {
    pub track: usize,
    pub preset: &'static Preset,
    pub name: String,
    pub trans: i32,
    pub staves: Vec<StaffOut>,
}

#[derive(Debug, Clone)]
pub struct ScoreOut {
    pub title: String,
    pub bars: Bars,
    pub ts: (i64, i64),
    pub compound: bool,
    pub parts: Vec<PartOut>,
}

/// Written key signature for a transposing part.
pub fn written_sharps(sharps: i32, trans: i32) -> i32 {
    let mut s = sharps + (7 * trans).rem_euclid(12);
    while s > 6 {
        s -= 12;
    }
    while s < -6 {
        s += 12;
    }
    s
}

/// Build the score model. Also fills `measure`, `voice` and `staff` on the notes it places.
pub fn build_score(
    piece: &mut Piece,
    grid: &Grid,
    key_sharps: &HashMap<usize, i32>,
    opts: &Options,
    trace: &mut Trace,
) -> anyhow::Result<ScoreOut> {
    let mut end_q: Option<Q> = None;
    for t in &piece.tracks {
        for n in t.kept() {
            if let (Some(b), Some(d)) = (n.beat, n.qdur) {
                let e = frac(b + d);
                end_q = Some(end_q.map_or(e, |x: Q| x.max(e)));
            }
        }
    }
    let end_q = end_q.unwrap_or_else(|| frac(grid.bar_quarters));
    let bars = Bars::new(grid, end_q);
    let qpm = np::round_i(grid.qpm());
    let mut parts = vec![];
    let mut assignments: Vec<(NRef, i64, i64, i64)> = vec![];
    for t in 0..piece.tracks.len() {
        let preset = m2s_core::preset::load(&piece.tracks[t].preset)?;
        let trans = preset.transposition;
        let specs = staves_for_track(t, piece, preset);
        let mut staves = vec![];
        for (si, spec) in specs.iter().enumerate() {
            let tempo = if t == 0 && si == 0 { Some(qpm) } else { None };
            let (staff, assigned) = fill_staff(piece, spec, &bars, grid, opts, tempo, trace);
            let mut staff = staff;
            staff.id = format!("{}-{si}", piece.tracks[t].name);
            if spec.kind == StaffKind::Pitched {
                staff.key = key_sharps.get(&t).map(|&s| written_sharps(s, trans));
            }
            if !matches!(spec.kind, StaffKind::Tab(_)) {
                for (r, voice) in assigned {
                    let b = piece.tracks[r.t].notes[r.n].beat.unwrap();
                    assignments.push((r, bars.number(bars.index(frac(b))), voice, si as i64 + 1));
                }
            }
            staves.push(staff);
        }
        parts.push(PartOut {
            track: t,
            preset,
            name: preset.display.clone(),
            trans,
            staves,
        });
    }
    for (r, m, v, s) in assignments {
        let n = &mut piece.tracks[r.t].notes[r.n];
        n.measure = Some(m);
        n.voice = Some(v);
        n.staff = Some(s);
    }
    Ok(ScoreOut {
        title: opts.title.clone(),
        ts: grid.time_signature_parts(),
        compound: grid.info.get("compound").and_then(|v| v.as_bool()).unwrap_or(false),
        bars,
        parts,
    })
}

#[allow(clippy::type_complexity)]
fn fill_staff(
    piece: &Piece,
    spec: &StaffSpec,
    bars: &Bars,
    grid: &Grid,
    opts: &Options,
    tempo: Option<i64>,
    trace: &mut Trace,
) -> (StaffOut, Vec<(NRef, i64)>) {
    let voices = if spec.notes.is_empty() {
        vec![vec![]]
    } else {
        staff_voices(&spec.notes, piece, trace)
    };
    let note = |r: &NRef| &piece.tracks[r.t].notes[r.n];
    let perc = spec.kind == StaffKind::Perc;
    let tab = matches!(spec.kind, StaffKind::Tab(_));
    // sforzandi: events much louder than their bar
    let mut sf_events: HashSet<usize> = HashSet::new();
    if spec.dynamics && opts.dynamics && !spec.notes.is_empty() {
        let mut vel_bar: HashMap<usize, Vec<i32>> = HashMap::new();
        for r in &spec.notes {
            vel_bar
                .entry(bars.index(frac(note(r).beat.unwrap())))
                .or_default()
                .push(note(r).velocity);
        }
        for (ei, ev) in voices[0].iter().enumerate() {
            let v = &vel_bar[&bars.index(ev.on)];
            let mean = v.iter().map(|&x| x as i64).sum::<i64>() as f64 / v.len() as f64;
            let peak = ev.notes.iter().map(|r| note(r).velocity).max().unwrap() as f64;
            if peak >= mean + 22.0 && peak >= 88.0 && ev.notes.len() >= 2 {
                sf_events.insert(ei);
            }
        }
    }
    let mut per_voice = vec![];
    for (vi, v) in voices.iter().enumerate() {
        let (mut pb, firsts) = fill_voice(v, bars, perc, piece, tab, opts);
        if vi == 0 {
            for &ei in &sf_events {
                if let Some((bi, k)) = firsts[ei] {
                    pb[bi][k].1.sf = true;
                }
            }
        }
        per_voice.push(pb);
    }
    if !spec.pedal.is_empty() && opts.pedal {
        pedal_marks(&mut per_voice, &spec.pedal, grid, bars);
    }
    let clefs: Vec<String> = if spec.clef_auto {
        let mut evs: Vec<&Event> = voices[0].iter().collect();
        if voices.len() > 1 {
            evs.extend(voices[1].iter());
        }
        clef_plan(&evs, bars, &spec.clef, piece)
    } else {
        vec![spec.clef.clone(); bars.starts.len()]
    };
    let dyn_plan = if spec.dynamics && opts.dynamics && !perc {
        dynamic_plan(&spec.notes, bars, piece)
    } else {
        BTreeMap::new()
    };
    let mut measures = vec![];
    let mut prev_clef: Option<&str> = None;
    for bi in 0..bars.starts.len() {
        let (a, b) = (bars.starts[bi], bars.ends[bi]);
        let mut contents: Vec<Vec<(Q, El)>> = per_voice
            .iter_mut()
            .map(|pv| with_rests(std::mem::take(&mut pv[bi]), b - a, a))
            .collect();
        if contents.len() > 1 && contents[1].iter().all(|(_, e)| e.rest) {
            contents.truncate(1);
        }
        let clef = if prev_clef != Some(clefs[bi].as_str()) {
            prev_clef = Some(clefs[bi].as_str());
            if spec.clef_auto && bi > 0 {
                trace.with(
                    "clef",
                    || json!({"staff": spec.name, "bar": bars.number(bi), "clef": clefs[bi]}),
                );
            }
            Some(clefs[bi].clone())
        } else {
            None
        };
        measures.push(StaffMeasure {
            clef,
            dynamic: dyn_plan.get(&bi).cloned(),
            voices: contents,
        });
    }
    if !dyn_plan.is_empty() {
        trace.with("dynamics", || json!({"staff": spec.name, "plan": dyn_plan.iter().map(|(k, v)| json!([bars.number(*k), v])).collect::<Vec<_>>()}));
    }
    let mut assigned = vec![];
    for (vi, v) in voices.iter().enumerate() {
        for ev in v {
            for r in ev.notes.iter().chain(ev.graces.iter()) {
                assigned.push((*r, vi as i64 + 1));
            }
        }
    }
    (
        StaffOut {
            id: String::new(),
            kind: spec.kind.clone(),
            measures,
            key: None,
            tempo,
        },
        assigned,
    )
}

/// Pedal marks (optional): start at the first struck element inside each interval, stop at the last.
fn pedal_marks(per_voice: &mut [Vec<Vec<(Q, El)>>], pedal: &[[f64; 2]], grid: &Grid, bars: &Bars) {
    let mut els: Vec<(f64, usize, usize, usize)> = vec![];
    for (vi, pv) in per_voice.iter().enumerate() {
        for (bi, items) in pv.iter().enumerate() {
            for (k, (off, el)) in items.iter().enumerate() {
                if !el.rest && !el.grace && matches!(el.tie, None | Some(Tie::Start)) {
                    els.push((qf(bars.starts[bi] + off), vi, bi, k));
                }
            }
        }
    }
    els.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for p in pedal {
        let (qa, qb) = (grid.to_quarters(p[0]), grid.to_quarters(p[1]));
        let inside: Vec<&(f64, usize, usize, usize)> =
            els.iter().filter(|e| qa - 0.05 <= e.0 && e.0 < qb - 0.05).collect();
        if let (Some(f), Some(l)) = (inside.first(), inside.last()) {
            per_voice[f.1][f.2][f.3].1.pedal_start = true;
            per_voice[l.1][l.2][l.3].1.pedal_stop = true;
        }
    }
}

/// Note value -> (MusicXML type, dots, triplet). None for values that need no type (measure rests).
pub fn note_type(d: Q) -> Option<(&'static str, u8, bool)> {
    let names: [(Q, &str); 10] = [
        (q(8, 1), "breve"),
        (q(4, 1), "whole"),
        (q(2, 1), "half"),
        (q(1, 1), "quarter"),
        (q(1, 2), "eighth"),
        (q(1, 4), "16th"),
        (q(1, 8), "32nd"),
        (q(1, 16), "64th"),
        (q(1, 32), "128th"),
        (q(1, 64), "256th"),
    ];
    for (factor, trip) in [(q(1, 1), false), (q(2, 3), true)] {
        let base = d / factor;
        for (v, name) in names {
            if base == v {
                return Some((name, 0, trip));
            }
            if base == v * q(3, 2) {
                return Some((name, 1, trip));
            }
            if base == v * q(7, 4) {
                return Some((name, 2, trip));
            }
        }
    }
    None
}

/// Beams per note value (eighth = 1 ... ).
pub fn beam_count(t: &str) -> usize {
    match t {
        "eighth" => 1,
        "16th" => 2,
        "32nd" => 3,
        "64th" => 4,
        "128th" => 5,
        "256th" => 6,
        _ => 0,
    }
}
