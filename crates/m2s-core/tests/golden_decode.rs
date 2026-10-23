//! Decoder regression: the Rust solvers must reproduce the frozen decisions of the original
//! (Python) implementation on real data: the first 60 s of blind test 1 and the synthetic band.

use std::collections::BTreeMap;

use m2s_core::ir::Piece;
use m2s_core::play::decode_track;
use m2s_core::trace::Trace;
use serde_json::Value;

fn check(name: &str) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden");
    let mut piece = Piece::load(root.join(format!("{name}_candidates.json"))).unwrap();
    let expect: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join(format!("{name}_decoded.json"))).unwrap()).unwrap();
    let mut viols = vec![];
    for tr in piece.tracks.iter_mut() {
        for v in decode_track(tr, &mut Trace::off()).unwrap() {
            viols.push(serde_json::json!([v.kind, v.time, v.notes]));
        }
    }
    let mut diffs = BTreeMap::new();
    for tr in &piece.tracks {
        for n in &tr.notes {
            let e = &expect["notes"][&n.id];
            let got = serde_json::json!({"keep": n.keep, "effector": n.effector, "player": n.player, "tags": n.tags, "offset": n.offset});
            if &got != e {
                diffs.insert(n.id.clone(), (got, e.clone()));
            }
        }
    }
    assert!(
        diffs.is_empty(),
        "{name}: {} notes differ, first: {:?}",
        diffs.len(),
        diffs.iter().next()
    );
    let mut got_v: Vec<String> = viols.iter().map(|v| v.to_string()).collect();
    got_v.sort();
    let mut exp_v: Vec<String> = expect["violations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.to_string())
        .collect();
    exp_v.sort();
    assert_eq!(got_v, exp_v, "{name}: violations differ");
}

#[test]
fn blind_first_minute_matches_reference_decisions() {
    check("blind60");
}

#[test]
fn band_matches_reference_decisions() {
    check("band");
}
