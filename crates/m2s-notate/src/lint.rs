//! Mechanical engraving checks on the score model: bar fullness, ledger lines, clef churn.
//! Port of `lint()` in build.py (same issue kinds, order and part ids).

use serde_json::{json, Map, Value};

use crate::score::{qf, ScoreOut, StaffKind, Q};
use crate::spell::parse_spelling;
use m2s_core::ir::Piece;
use num_traits::Signed;
use num_traits::Zero;

fn q_repr(x: Q) -> String {
    let d = *x.denom();
    if d & (d - 1) == 0 {
        format!("{:?}", qf(x))
    } else {
        format!("{}/{}", x.numer(), x.denom())
    }
}

fn diatonic(step: char, octave: i32) -> i32 {
    let idx = "CDEFGAB".find(step).unwrap_or(0) as i32;
    idx + 7 * octave + 1
}

pub fn lint(score: &ScoreOut, piece: &Piece) -> Value {
    let mut issues: Vec<Value> = vec![];
    let bars = &score.bars;
    for part in &score.parts {
        for st in &part.staves {
            let mut clef_changes = 0;
            let mut clef: Option<String> = None;
            for (bi, m) in st.measures.iter().enumerate() {
                let number = bars.number(bi);
                let expected = if bi == 0 && !bars.pickup.is_zero() {
                    bars.pickup
                } else {
                    bars.bar
                };
                for v in &m.voices {
                    let got: Q = v.iter().filter(|(_, e)| !e.grace).map(|(_, e)| e.dur).sum();
                    let full = v.iter().any(|(_, e)| e.measure_rest);
                    if (got - expected).abs() > Q::new(1, 1000) && !full {
                        issues.push(json!({"part": st.id, "bar": number, "kind": "bar-length",
                                           "detail": format!("{} of {:?}", q_repr(got), qf(expected))}));
                    }
                }
                if m.clef.is_some() && number > 1 {
                    clef_changes += 1;
                }
                // The clef in force inside this bar, including a change at its start. (The Python
                // lint used music21's context lookup, which returned the clef *before* the bar: bar 1
                // was never checked and clef-change bars were judged against the old clef.)
                if let Some(c) = &m.clef {
                    clef = Some(c.clone());
                }
                // diatonic number of the middle line: B4 treble, B3 for treble-8vb, D3 bass
                let mid = match clef.as_deref() {
                    Some("treble") => 35,
                    Some("treble8vb") => 28,
                    Some("bass") => 23,
                    _ => continue,
                };
                if st.kind != StaffKind::Pitched {
                    continue;
                }
                for v in &m.voices {
                    let mut els: Vec<&(Q, crate::score::El)> = v.iter().collect();
                    els.sort_by(|a, b| a.0.cmp(&b.0).then((!a.1.grace).cmp(&!b.1.grace)));
                    for (_, el) in els {
                        for r in &el.notes {
                            let n = &piece.tracks[r.t].notes[r.n];
                            let Some(sp) = n.spelling.as_deref() else { continue };
                            let Some((step, _, oct)) = parse_spelling(sp) else {
                                continue;
                            };
                            let steps = diatonic(step, oct) - mid;
                            let over = steps.abs() - 4;
                            let ledger = if over > 0 { (over + 1) / 2 } else { 0 };
                            if ledger > 4 {
                                issues.push(json!({"part": st.id, "bar": number, "kind": "ledger",
                                                   "detail": format!("{sp}: {ledger} ledger lines")}));
                            }
                        }
                    }
                }
            }
            let n_bars = st.measures.len();
            if n_bars > 0 && clef_changes as f64 / n_bars as f64 > 0.25 {
                issues.push(json!({"part": st.id, "kind": "clef-churn", "detail": format!("{clef_changes} changes in {n_bars} bars")}));
            }
        }
    }
    let mut kinds: Map<String, Value> = Map::new();
    for i in &issues {
        let k = i["kind"].as_str().unwrap().to_string();
        let c = kinds.get(&k).and_then(|v| v.as_i64()).unwrap_or(0);
        kinds.insert(k, json!(c + 1));
    }
    let count = issues.len();
    issues.truncate(200);
    json!({"count": count, "by_kind": kinds, "issues": issues})
}
