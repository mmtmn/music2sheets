//! Pitch spelling on the line of fifths. Port of `spell.py`.

use std::collections::HashMap;

use crate::keys::Key;

const LETTERS: &[u8] = b"FCGDAEB";

/// Line-of-fifths position -> pitch-class name (music21 style: '#' sharps, '-' flats).
pub fn fifths_name(pos: i32) -> String {
    let letter = LETTERS[(pos + 1).rem_euclid(7) as usize] as char;
    let acc = (pos + 1).div_euclid(7);
    let mut s = letter.to_string();
    if acc > 0 {
        s.push_str(&"#".repeat(acc as usize));
    } else {
        s.push_str(&"-".repeat((-acc) as usize));
    }
    s
}

pub fn candidates(pc: i32) -> Vec<i32> {
    let k0 = (7 * pc).rem_euclid(12);
    [k0 - 12, k0, k0 + 12]
        .into_iter()
        .filter(|k| (-15..=19).contains(k))
        .collect()
}

/// What spelling needs from a note.
pub struct SpellNote<'a> {
    pub id: &'a str,
    pub onset: f64,
    pub pitch: i32,
    /// the grouping key Python uses: str(effector.get("hand", staff))
    pub hand_key: String,
}

/// Spelling (e.g. "B#4") for each note, in input order.
pub fn spell(notes: &[SpellNote], key: &Key, memory: f64) -> Vec<String> {
    let mut order: Vec<usize> = (0..notes.len()).collect();
    order.sort_by(|&a, &b| {
        notes[a]
            .onset
            .partial_cmp(&notes[b].onset)
            .unwrap()
            .then(notes[a].pitch.cmp(&notes[b].pitch))
    });
    let mut by_hand: Vec<(String, Vec<usize>)> = vec![];
    for &i in &order {
        let k = &notes[i].hand_key;
        match by_hand.iter_mut().find(|(h, _)| h == k) {
            Some((_, v)) => v.push(i),
            None => by_hand.push((k.clone(), vec![i])),
        }
    }
    let mut nxt: HashMap<&str, usize> = HashMap::new();
    for (_, seq) in &by_hand {
        for w in seq.windows(2) {
            nxt.insert(notes[w[0]].id, w[1]);
        }
    }
    let mut recent = key.centre();
    let mut out = vec![String::new(); notes.len()];
    for &i in &order {
        let n = &notes[i];
        let ce = 0.5 * key.centre() + 0.5 * recent;
        let mut lean = 0.0;
        if let Some(&b) = nxt.get(n.id) {
            let bp = notes[b].pitch;
            if (bp - n.pitch).abs() == 1 {
                lean = if bp > n.pitch { 1.2 } else { -1.2 };
            }
        }
        let cands = candidates(n.pitch.rem_euclid(12));
        let mut pos = cands[0];
        for &k in &cands[1..] {
            if (k as f64 - (ce + lean)).abs() < (pos as f64 - (ce + lean)).abs() {
                pos = k;
            }
        }
        recent = memory * recent + (1.0 - memory) * pos as f64;
        let name = fifths_name(pos);
        let base = name.chars().next().unwrap();
        let acc = name.matches('#').count() as i32 - name.matches('-').count() as i32;
        let natural = match base {
            'C' => 0,
            'D' => 2,
            'E' => 4,
            'F' => 5,
            'G' => 7,
            'A' => 9,
            _ => 11,
        };
        let octave = (n.pitch - acc - natural).div_euclid(12) - 1;
        out[i] = format!("{name}{octave}");
    }
    out
}

/// Parse a music21 pitch name such as "B#4", "E-3", "C##5" into (step, alter, octave).
pub fn parse_spelling(s: &str) -> Option<(char, i32, i32)> {
    let mut chars = s.chars();
    let step = chars.next()?;
    let rest: String = chars.collect();
    let acc_len = rest.chars().take_while(|c| *c == '#' || *c == '-').count();
    let (acc, oct) = rest.split_at(acc_len);
    let alter = acc.matches('#').count() as i32 - acc.matches('-').count() as i32;
    Some((step, alter, oct.parse().ok()?))
}

/// music21's default spelling for a MIDI number (used only when a note has no spelling).
pub fn default_spelling(midi: i32) -> (char, i32, i32) {
    let pc = midi.rem_euclid(12);
    let (step, alter) = match pc {
        0 => ('C', 0),
        1 => ('C', 1),
        2 => ('D', 0),
        3 => ('E', -1),
        4 => ('E', 0),
        5 => ('F', 0),
        6 => ('F', 1),
        7 => ('G', 0),
        8 => ('G', 1),
        9 => ('A', 0),
        10 => ('B', -1),
        _ => ('B', 0),
    };
    (step, alter, midi.div_euclid(12) - 1)
}

pub fn step_pc(step: char) -> i32 {
    match step {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        _ => 11,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let got: Vec<String> = [0, 7, 12, -2, -5].iter().map(|&k| fifths_name(k)).collect();
        assert_eq!(got, ["C", "C#", "B#", "B-", "D-"]);
        assert_eq!(parse_spelling("B#3"), Some(('B', 1, 3)));
        assert_eq!(parse_spelling("E--5"), Some(('E', -2, 5)));
    }
}
