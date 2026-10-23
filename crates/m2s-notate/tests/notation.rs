//! Ports of tests/test_notation.py and tests/test_roundtrip.py, plus property tests.

mod common;

use m2s_core::ir::{Effector, Hand, Note, Piece, Track};
use m2s_core::trace::Trace;
use m2s_notate::keys::{estimate_key, Key, Mode};
use m2s_notate::rhythm::{build_grid, quantize};
use m2s_notate::spell::{spell, SpellNote};
use m2s_notate::{notate, Options};
use proptest::prelude::*;

fn n(i: usize, t: f64, p: i32, d: f64, hand: Hand) -> Note {
    let mut x = Note::new(format!("n{i}"), t, t + d, p, 80);
    x.effector = Effector {
        hand: Some(hand),
        ..Default::default()
    };
    x
}

fn spell_all(notes: &[Note], key: Key) -> Vec<String> {
    let sn: Vec<SpellNote> = notes
        .iter()
        .map(|x| SpellNote {
            id: &x.id,
            onset: x.onset,
            pitch: x.pitch,
            hand_key: "R".into(),
        })
        .collect();
    spell(&sn, &key, 0.82)
}

#[test]
fn leading_tone_in_c_sharp_minor_is_b_sharp() {
    let ps = [(0.0, 61), (0.5, 64), (1.0, 68), (1.5, 68), (2.0, 72), (2.5, 75)];
    let notes: Vec<Note> = ps
        .iter()
        .enumerate()
        .map(|(i, &(t, p))| n(i, t, p, 0.4, Hand::R))
        .collect();
    let s = spell_all(
        &notes,
        Key {
            tonic_pc: 1,
            mode: Mode::Minor,
            score: 1.0,
        },
    );
    assert_eq!(s[4], "B#4");
    assert_eq!(s[0], "C#4");
}

#[test]
fn f_major_uses_flat() {
    let notes: Vec<Note> = [65, 67, 69, 70, 72]
        .iter()
        .enumerate()
        .map(|(i, &p)| n(i, i as f64 * 0.5, p, 0.4, Hand::R))
        .collect();
    let s = spell_all(
        &notes,
        Key {
            tonic_pc: 5,
            mode: Mode::Major,
            score: 1.0,
        },
    );
    assert_eq!(s[3], "B-4");
}

#[test]
fn estimate_key_minor_from_bass() {
    let mut notes = vec![];
    let mut t = 0.0;
    for chord in [[45, 60, 64, 69], [41, 60, 65, 69], [43, 59, 62, 67], [45, 60, 64, 69]] {
        for p in chord {
            notes.push(n(notes.len(), t, p, 1.0, Hand::R));
        }
        t += 1.0;
    }
    let refs: Vec<&Note> = notes.iter().collect();
    let k = estimate_key(&refs, &[]);
    assert_eq!((k.tonic_pc, k.mode), (9, Mode::Minor));
}

/// deterministic small jitter (stands in for numpy's normal(0, 0.008))
fn jitter(i: usize) -> f64 {
    let x = ((i as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        >> 33) as f64
        / (1u64 << 31) as f64;
    (x - 0.5) * 0.02
}

#[test]
fn quantize_triplets_and_sixteenths() {
    let spq = 0.5;
    let beats: Vec<f64> = (0..17).map(|i| i as f64 * spq).collect();
    let mut notes = vec![];
    for k in 0..8 {
        notes.push(n(k, k as f64 * spq / 4.0 + jitter(k), 72, 0.1, Hand::R));
    }
    for k in 0..6 {
        notes.push(n(
            8 + k,
            2.0 * spq + k as f64 * spq / 3.0 + jitter(8 + k),
            74,
            0.1,
            Hand::R,
        ));
    }
    let refs: Vec<&Note> = notes.iter().collect();
    let downbeats: Vec<f64> = beats.iter().step_by(4).copied().collect();
    let mut grid = build_grid(&beats, &downbeats, &refs, None, &[], &mut Trace::off());
    let q = quantize(&refs, &mut grid, &mut Trace::off());
    let got: Vec<f64> = q.iter().map(|(b, _)| (b * 1000.0).round() / 1000.0).collect();
    let want: Vec<f64> = (0..8)
        .map(|k| k as f64 * 0.25)
        .chain((0..6).map(|k| ((2.0 + k as f64 / 3.0) * 1000.0).round() / 1000.0))
        .collect();
    assert_eq!(got, want);
}

#[test]
fn meter_three_four() {
    let spq = 0.45;
    let roots: Vec<i32> = [45, 43, 41, 40, 45, 43, 41, 40].repeat(2);
    let mut notes = vec![];
    for (b, &r) in roots.iter().enumerate() {
        let t = b as f64 * 3.0 * spq;
        notes.push(n(notes.len(), t, r, 3.0 * spq, Hand::L));
        for k in [1.0, 2.0] {
            for p in [r + 16, r + 19] {
                notes.push(n(notes.len(), t + k * spq, p, spq * 0.8, Hand::R));
            }
        }
    }
    let beats: Vec<f64> = (0..roots.len() * 3 + 1).map(|i| i as f64 * spq).collect();
    let refs: Vec<&Note> = notes.iter().collect();
    let grid = build_grid(&beats, &[], &refs, None, &[], &mut Trace::off());
    assert_eq!(grid.time_signature(), "3/4");
    assert_eq!(grid.first_downbeat_beat % 3, 0);
}

/// Synthetic piece: RH eighths/sixteenths with held notes, LH quarter-note chords (test_roundtrip.py).
fn synthetic(seed: u64, bars: usize, qpm: f64) -> (Piece, Vec<f64>) {
    let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    let mut rnd = |m: u64| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) % m
    };
    let spq = 60.0 / qpm;
    let mut notes = vec![];
    let mut q = 0.0;
    while q < (bars * 4) as f64 {
        let d = [0.5, 0.5, 0.25, 0.25, 1.0, 0.75][rnd(6) as usize];
        let p = 64 + rnd(21) as i32;
        notes.push(n(notes.len(), q * spq, p, d * spq, Hand::R));
        notes.last_mut().unwrap().offset = (q + d) * spq;
        q += d;
    }
    for b in 0..bars * 4 {
        let root = [36, 41, 43, 45][rnd(4) as usize];
        for p in [root, root + 7, root + 12] {
            let mut x = n(notes.len(), b as f64 * spq, p, spq, Hand::L);
            x.offset = (b + 1) as f64 * spq;
            notes.push(x);
        }
    }
    let beats = (0..=bars * 4).map(|k| k as f64 * spq).collect();
    (
        Piece {
            meta: Default::default(),
            tracks: vec![Track::new("piano", "piano", notes)],
        },
        beats,
    )
}

#[test]
fn roundtrip_lossless() {
    let (mut piece, beats) = synthetic(0, 12, 96.0);
    let downbeats: Vec<f64> = beats.iter().step_by(4).copied().collect();
    let res = notate(
        &mut piece,
        &beats,
        &downbeats,
        None,
        50.0,
        None,
        &Options::default(),
        &mut Trace::off(),
    )
    .unwrap();
    assert_eq!(res.info["time_signature"], "4/4");
    let got: Vec<(f64, i32, f64)> = common::read_notes(&res.musicxml).concat();
    let mut got = got;
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut want: Vec<(f64, i32, f64)> = piece.tracks[0]
        .notes
        .iter()
        .map(|x| {
            (
                (x.beat.unwrap() * 1e4).round() / 1e4,
                x.pitch,
                (x.qdur.unwrap() * 1e4).round() / 1e4,
            )
        })
        .collect();
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got.len(), want.len());
    assert_eq!(got, want);
    assert_eq!(res.info["lint"]["by_kind"].get("bar-length"), None);
}

#[test]
fn last_note_rounded_past_final_bar() {
    let spq = 0.5;
    let mut notes = vec![
        n(0, 0.0, 60, 0.5, Hand::R),
        n(1, 0.5, 62, 0.5, Hand::R),
        n(2, 0.75, 55, 1.95, Hand::R),
        n(3, 1.0, 64, 0.25, Hand::R),
    ];
    notes[0].id = "a".into();
    let beats: Vec<f64> = (0..6).map(|k| k as f64 * spq).collect();
    let downbeats: Vec<f64> = beats.iter().step_by(4).copied().collect();
    let mut piece = Piece {
        meta: Default::default(),
        tracks: vec![Track::new("piano", "piano", notes)],
    };
    let res = notate(
        &mut piece,
        &beats,
        &downbeats,
        None,
        50.0,
        None,
        &Options::default(),
        &mut Trace::off(),
    )
    .unwrap();
    assert!(res.musicxml.contains("<score-partwise"));
}

fn random_piece(seed: u64, n_notes: usize, tempo: f64, preset: &str) -> (Piece, Vec<f64>) {
    let mut s = seed.wrapping_add(17);
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let spb = 60.0 / tempo;
    let mut notes = vec![];
    let mut t = 0.05 + rnd() * 0.5;
    for i in 0..n_notes {
        t += rnd() * spb * 0.8;
        let d = 0.08 + rnd() * spb * 3.0;
        let p = 36 + (rnd() * 50.0) as i32;
        let hand = if p < 60 { Hand::L } else { Hand::R };
        notes.push(n(i, t, p, d, hand));
    }
    let end = t + 4.0 * spb;
    let mut beats = vec![];
    let mut b = 0.0;
    while b < end {
        beats.push(b + (rnd() - 0.5) * 0.02 * spb);
        b += spb * (0.9 + rnd() * 0.2);
    }
    (
        Piece {
            meta: Default::default(),
            tracks: vec![Track::new("p", preset, notes)],
        },
        beats,
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

    #[test]
    fn notate_invariants(seed in 0u64..10_000, n_notes in 4usize..120, tempo in 50.0f64..200.0,
                         preset in prop::sample::select(vec!["piano", "guitar", "voice", "sax_alto"])) {
        let (mut piece, beats) = random_piece(seed, n_notes, tempo, preset);
        prop_assume!(beats.len() >= 8);
        let downbeats: Vec<f64> = beats.iter().step_by(4).copied().collect();
        let res = notate(&mut piece, &beats, &downbeats, None, 50.0, None, &Options::default(), &mut Trace::off()).unwrap();
        // every bar of every voice is exactly full
        prop_assert_eq!(res.info["lint"]["by_kind"].get("bar-length"), None);
        // every kept, placed, non-grace note is in the MusicXML at its written pitch
        let trans = m2s_core::preset::load(preset).unwrap().transposition;
        let mut want: Vec<i32> = piece.tracks[0].notes.iter()
            .filter(|x| x.keep && x.beat.is_some() && !x.has_tag("grace"))
            .map(|x| x.pitch + trans).collect();
        let parts = common::read_notes(&res.musicxml);
        let mut got: Vec<i32> = parts[0].iter().filter(|r| r.2 > 0.0).map(|r| r.1).collect();
        if preset == "guitar" {
            // notation staff and tab staff carry the same notes; keep one copy
            got.sort();
            let mut half = vec![];
            for (i, p) in got.iter().enumerate() { if i % 2 == 0 { half.push(*p); } }
            got = half;
        }
        want.sort();
        got.sort();
        prop_assert_eq!(got, want);
    }
}
