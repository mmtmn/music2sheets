//! A small MusicXML reader for tests: notes per part as (offset, midi, quarter length), ties merged.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

pub fn read_notes(xml: &str) -> Vec<Vec<(f64, i32, f64)>> {
    let opts = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    let doc = roxmltree::Document::parse_with_options(xml, opts).expect("valid xml");
    let mut parts = vec![];
    for part in doc.descendants().filter(|n| n.has_tag_name("part")) {
        let mut divisions = 1.0;
        let mut measure_start = 0.0;
        let mut rows: Vec<(f64, i32, f64)> = vec![];
        let mut pending: HashMap<(i32, String), VecDeque<usize>> = HashMap::new();
        for m in part.children().filter(|n| n.has_tag_name("measure")) {
            let mut pos: f64 = 0.0;
            let mut maxpos: f64 = 0.0;
            let mut last_start = 0.0;
            for el in m.children().filter(|n| n.is_element()) {
                match el.tag_name().name() {
                    "attributes" => {
                        if let Some(d) = el.descendants().find(|n| n.has_tag_name("divisions")) {
                            divisions = d.text().unwrap().parse().unwrap();
                        }
                    }
                    "backup" | "forward" => {
                        let d: f64 = child_text(&el, "duration").parse::<f64>().unwrap() / divisions;
                        pos += if el.has_tag_name("backup") { -d } else { d };
                    }
                    "note" => {
                        let grace = el.children().any(|n| n.has_tag_name("grace"));
                        let chord = el.children().any(|n| n.has_tag_name("chord"));
                        let dur = if grace {
                            0.0
                        } else {
                            child_text(&el, "duration").parse::<f64>().unwrap() / divisions
                        };
                        let start = if chord { last_start } else { pos };
                        if !chord {
                            last_start = pos;
                            pos += dur;
                        }
                        maxpos = maxpos.max(pos);
                        let Some(p) = el.children().find(|n| n.has_tag_name("pitch")) else {
                            continue;
                        };
                        let step = child_text(&p, "step").chars().next().unwrap();
                        let alter: i32 = p
                            .children()
                            .find(|n| n.has_tag_name("alter"))
                            .map(|a| a.text().unwrap().parse().unwrap())
                            .unwrap_or(0);
                        let oct: i32 = child_text(&p, "octave").parse().unwrap();
                        let pc = match step {
                            'C' => 0,
                            'D' => 2,
                            'E' => 4,
                            'F' => 5,
                            'G' => 7,
                            'A' => 9,
                            _ => 11,
                        };
                        let midi = 12 * (oct + 1) + pc + alter;
                        let voice = el
                            .children()
                            .find(|n| n.has_tag_name("voice"))
                            .and_then(|v| v.text())
                            .unwrap_or("1")
                            .to_string();
                        let ties: Vec<&str> = el
                            .children()
                            .filter(|n| n.has_tag_name("tie"))
                            .filter_map(|t| t.attribute("type"))
                            .collect();
                        let key = (midi, voice);
                        if ties.contains(&"stop") {
                            // chains of the same pitch in one voice continue in the order they started
                            if let Some(i) = pending.get_mut(&key).and_then(|q| q.pop_front()) {
                                rows[i].2 += dur;
                                if ties.contains(&"start") {
                                    pending.entry(key).or_default().push_back(i);
                                }
                                continue;
                            }
                        }
                        rows.push((measure_start + start, midi, dur));
                        if ties.contains(&"start") {
                            pending.entry(key).or_default().push_back(rows.len() - 1);
                        }
                    }
                    _ => {}
                }
            }
            measure_start += maxpos;
        }
        for r in rows.iter_mut() {
            r.0 = (r.0 * 1e4).round() / 1e4;
            r.2 = (r.2 * 1e4).round() / 1e4;
        }
        rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
        parts.push(rows);
    }
    parts
}

fn child_text(n: &roxmltree::Node, tag: &str) -> String {
    n.children()
        .find(|c| c.has_tag_name(tag))
        .and_then(|c| c.text())
        .unwrap_or("0")
        .to_string()
}
