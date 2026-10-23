//! Trills and grace notes before quantization. Port of `ornaments.py`.

use m2s_core::ir::{Hand, Track};

fn hand_key(h: Option<Hand>) -> &'static str {
    match h {
        Some(Hand::L) => "L",
        Some(Hand::R) => "R",
        None => "",
    }
}

fn kept_by_hand(track: &Track) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..track.notes.len()).filter(|&i| track.notes[i].keep).collect();
    idx.sort_by(|&a, &b| track.notes[a].onset.partial_cmp(&track.notes[b].onset).unwrap());
    let mut groups: Vec<(&str, Vec<usize>)> = vec![];
    for i in idx {
        let k = hand_key(track.notes[i].effector.hand);
        match groups.iter_mut().find(|(h, _)| *h == k) {
            Some((_, v)) => v.push(i),
            None => groups.push((k, vec![i])),
        }
    }
    groups.into_iter().map(|(_, v)| v).collect()
}

/// Fold trills into their first note (extra notes get keep=false, tag "trill-note"). Returns count.
pub fn fold_trills(track: &mut Track, max_ioi: f64, min_notes: usize) -> usize {
    let mut found = 0;
    for seq in kept_by_hand(track) {
        let len = seq.len() as i64;
        let mut i: i64 = 0;
        while i < len - min_notes as i64 + 1 {
            let (a, b) = (&track.notes[seq[i as usize]], &track.notes[seq[i as usize + 1]]);
            let step = (a.pitch - b.pitch).abs();
            if !(step == 1 || step == 2) || b.onset - a.onset > max_ioi {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j + 1 < len {
                let nj = &track.notes[seq[j as usize]];
                let nj1 = &track.notes[seq[j as usize + 1]];
                let njm = &track.notes[seq[j as usize - 1]];
                if nj1.onset - nj.onset <= max_ioi && nj1.pitch == njm.pitch {
                    j += 1;
                } else {
                    break;
                }
            }
            if (j - i + 1) as usize >= min_notes {
                let range: Vec<usize> = seq[i as usize..=j as usize].to_vec();
                let off = range
                    .iter()
                    .map(|&k| track.notes[k].offset)
                    .fold(f64::NEG_INFINITY, f64::max);
                let main = &mut track.notes[range[0]];
                main.offset = off;
                main.tags.push("trill".into());
                for &k in &range[1..] {
                    track.notes[k].keep = false;
                    track.notes[k].tags.push("trill-note".into());
                }
                found += 1;
                i = j + 1;
            } else {
                i += 1;
            }
        }
    }
    found
}

pub fn mark_graces(track: &mut Track, max_dur: f64, max_gap: f64) -> usize {
    let mut found = 0;
    for seq in kept_by_hand(track) {
        if seq.len() < 2 {
            continue;
        }
        for i in 0..seq.len() - 1 {
            let a = &track.notes[seq[i]];
            let b = &track.notes[seq[i + 1]];
            let gap = b.onset - a.onset;
            let short = a.duration() <= max_dur && 0.02 < gap && gap <= max_gap;
            let isolated = i == 0 || a.onset - track.notes[seq[i - 1]].onset > 0.25;
            if short && isolated && a.pitch != b.pitch && (a.pitch - b.pitch).abs() <= 4 {
                track.notes[seq[i]].tags.push("grace".into());
                found += 1;
            }
        }
    }
    found
}
