//! MusicXML 4.0 (partwise) writer for the score model. Written directly, so every element
//! (including string/fret on tab chord notes) is under our control.

use std::collections::HashMap;
use std::fmt::Write;

use m2s_core::ir::Piece;
use m2s_core::preset::Family;
use num_traits::Zero;

use crate::score::{beam_count, note_type, q, El, PartOut, ScoreOut, StaffKind, Tie, Q};
use crate::spell::{default_spelling, parse_spelling};

pub const DIVISIONS: i64 = 48;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn dur_div(d: Q) -> i64 {
    let x = d * DIVISIONS;
    if x.is_integer() {
        x.to_integer()
    } else {
        (*x.numer() as f64 / *x.denom() as f64).round() as i64
    }
}

fn clef_xml(name: &str, number: Option<usize>) -> String {
    let num = number.map(|n| format!(" number=\"{n}\"")).unwrap_or_default();
    match name {
        "bass" => format!("<clef{num}><sign>F</sign><line>4</line></clef>"),
        "treble8vb" => {
            format!("<clef{num}><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>")
        }
        "perc" => format!("<clef{num}><sign>percussion</sign></clef>"),
        "tab" => format!("<clef{num}><sign>TAB</sign><line>5</line></clef>"),
        _ => format!("<clef{num}><sign>G</sign><line>2</line></clef>"),
    }
}

fn key_alter(sharps: i32, step: char) -> i32 {
    const SHARPS: &str = "FCGDAEB";
    const FLATS: &str = "BEADGCF";
    if sharps > 0 && SHARPS[..sharps as usize].contains(step) {
        1
    } else if sharps < 0 && FLATS[..(-sharps) as usize].contains(step) {
        -1
    } else {
        0
    }
}

fn accidental_name(alter: i32) -> &'static str {
    match alter {
        2 => "double-sharp",
        1 => "sharp",
        -1 => "flat",
        -2 => "flat-flat",
        _ => "natural",
    }
}

struct Ctx<'a> {
    piece: &'a Piece,
    part: &'a PartOut,
    drum_display: HashMap<i32, (String, String, &'static str)>,
}

impl Ctx<'_> {
    fn spelled(&self, r: &crate::score::NRef) -> (char, i32, i32) {
        let n = &self.piece.tracks[r.t].notes[r.n];
        n.spelling
            .as_deref()
            .and_then(parse_spelling)
            .unwrap_or_else(|| default_spelling(n.pitch + self.part.trans))
    }
}

/// Accidental display per (voice, item, note) for one staff-measure.
fn accidentals(ctx: &Ctx, voices: &[Vec<(Q, El)>], key: i32) -> HashMap<(usize, usize, usize), &'static str> {
    let mut entries = vec![];
    for (vi, items) in voices.iter().enumerate() {
        for (k, (off, el)) in items.iter().enumerate() {
            if el.rest {
                continue;
            }
            entries.push((*off, !el.grace, vi, k));
        }
    }
    entries.sort();
    let mut state: HashMap<(char, i32), i32> = HashMap::new();
    let mut out = HashMap::new();
    for (_, _, vi, k) in entries {
        let el = &voices[vi][k].1;
        let continuation = matches!(el.tie, Some(Tie::Stop) | Some(Tie::Continue));
        for (ni, r) in el.notes.iter().enumerate() {
            let (step, alter, oct) = ctx.spelled(r);
            if continuation {
                continue;
            }
            let cur = *state.get(&(step, oct)).unwrap_or(&key_alter(key, step));
            if alter != cur {
                out.insert((vi, k, ni), accidental_name(alter));
                state.insert((step, oct), alter);
            }
        }
    }
    out
}

/// Beam values per (item index) for one voice: Vec of (number, value).
fn beams(items: &[(Q, El)], pad: Q, beat: Q) -> HashMap<usize, Vec<(usize, &'static str)>> {
    let mut out = HashMap::new();
    let mut groups: Vec<Vec<(usize, usize)>> = vec![];
    let mut cur: Vec<(usize, usize)> = vec![];
    let mut cur_beat: Option<i64> = None;
    for (k, (off, el)) in items.iter().enumerate() {
        if el.grace {
            continue;
        }
        let levels = if el.rest || el.measure_rest {
            0
        } else {
            note_type(el.dur).map(|t| beam_count(t.0)).unwrap_or(0)
        };
        let b = ((pad + off) / beat).floor().to_integer();
        if levels == 0 || cur_beat != Some(b) {
            if cur.len() >= 2 {
                groups.push(std::mem::take(&mut cur));
            }
            cur.clear();
            cur_beat = None;
        }
        if levels > 0 {
            cur.push((k, levels));
            cur_beat = Some(b);
        }
    }
    if cur.len() >= 2 {
        groups.push(cur);
    }
    for g in groups {
        let maxl = g.iter().map(|x| x.1).max().unwrap();
        for level in 1..=maxl {
            for (i, &(k, l)) in g.iter().enumerate() {
                if l < level {
                    continue;
                }
                let prev = i > 0 && g[i - 1].1 >= level;
                let next = i + 1 < g.len() && g[i + 1].1 >= level;
                let v = match (prev, next) {
                    (true, true) => "continue",
                    (false, true) => "begin",
                    (true, false) => "end",
                    (false, false) => {
                        if i + 1 < g.len() {
                            "forward hook"
                        } else {
                            "backward hook"
                        }
                    }
                };
                out.entry(k).or_insert_with(Vec::new).push((level, v));
            }
        }
    }
    out
}

/// Tuplet bracket start/stop per item index for one voice.
fn tuplets(items: &[(Q, El)]) -> HashMap<usize, Vec<&'static str>> {
    let mut out: HashMap<usize, Vec<&'static str>> = HashMap::new();
    let mut open: Option<(usize, usize, Q)> = None; // (start item, last item, accumulated)
    let close = |out: &mut HashMap<usize, Vec<&'static str>>, s: usize, l: usize| {
        if s != l {
            out.entry(s).or_default().push("start");
            out.entry(l).or_default().push("stop");
        }
    };
    for (k, (_, el)) in items.iter().enumerate() {
        if el.grace {
            continue;
        }
        let trip = !el.measure_rest && note_type(el.dur).map(|t| t.2).unwrap_or(false);
        if !trip {
            if let Some((s, l, _)) = open.take() {
                close(&mut out, s, l);
            }
            continue;
        }
        let (s, acc) = match open {
            Some((s, _, acc)) => (s, acc + el.dur),
            None => (k, el.dur),
        };
        if acc.denom() % 3 != 0 {
            close(&mut out, s, k);
            open = None;
        } else {
            open = Some((s, k, acc));
        }
    }
    if let Some((s, l, _)) = open {
        close(&mut out, s, l);
    }
    out
}

fn direction(placement: &str, inner: &str, staff: usize, sound: Option<String>) -> String {
    format!(
        "<direction placement=\"{placement}\"><direction-type>{inner}</direction-type><staff>{staff}</staff>{}</direction>",
        sound.unwrap_or_default()
    )
}

pub fn write(score: &ScoreOut, piece: &Piece) -> String {
    let mut x = String::new();
    x.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    x.push_str("<!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 4.0 Partwise//EN\" \"http://www.musicxml.org/dtds/partwise.dtd\">\n");
    x.push_str("<score-partwise version=\"4.0\">\n");
    let t = esc(&score.title);
    let _ = writeln!(
        x,
        "  <work><work-title>{t}</work-title></work>\n  <movement-title>{t}</movement-title>"
    );
    x.push_str("  <identification><encoding><software>music2sheets</software></encoding></identification>\n");
    x.push_str("  <part-list>\n");
    let mut channel = 0;
    for (i, part) in score.parts.iter().enumerate() {
        let pid = format!("P{}", i + 1);
        let name = esc(&part.name);
        let (ch, prog) = if part.preset.family == Family::Drums {
            (10, 1)
        } else {
            channel += 1;
            if channel == 10 {
                channel += 1;
            }
            (((channel - 1) % 16) + 1, part.preset.gm_program as i32 + 1)
        };
        let _ = writeln!(
            x,
            "    <score-part id=\"{pid}\"><part-name>{name}</part-name><score-instrument id=\"{pid}-I1\"><instrument-name>{name}</instrument-name></score-instrument><midi-instrument id=\"{pid}-I1\"><midi-channel>{ch}</midi-channel><midi-program>{prog}</midi-program></midi-instrument></score-part>"
        );
    }
    x.push_str("  </part-list>\n");
    for (i, part) in score.parts.iter().enumerate() {
        write_part(&mut x, score, part, &format!("P{}", i + 1), piece);
    }
    x.push_str("</score-partwise>\n");
    x
}

fn write_part(x: &mut String, score: &ScoreOut, part: &PartOut, pid: &str, piece: &Piece) {
    let mut drum_display = HashMap::new();
    if let Some(kit) = &part.preset.kit {
        for kp in kit.values() {
            let stem = if kp.limbs.iter().all(|l| l == "RF" || l == "LF") {
                "down"
            } else {
                "up"
            };
            drum_display.insert(kp.midi, (kp.display.clone(), kp.notehead.clone(), stem));
        }
    }
    let ctx = Ctx {
        piece,
        part,
        drum_display,
    };
    let bars = &score.bars;
    let nst = part.staves.len();
    let multi = nst > 1;
    let beat = if score.compound { q(3, 2) } else { q(1, 1) };
    let _ = writeln!(x, "  <part id=\"{pid}\">");
    for bi in 0..bars.starts.len() {
        let len = bars.ends[bi] - bars.starts[bi];
        let implicit = if bi == 0 && !bars.pickup.is_zero() {
            " implicit=\"yes\""
        } else {
            ""
        };
        let _ = write!(x, "    <measure number=\"{}\"{implicit}>", bars.number(bi));
        // attributes
        if bi == 0 {
            x.push_str("<attributes>");
            let _ = write!(x, "<divisions>{DIVISIONS}</divisions>");
            let keys: Vec<Option<i32>> = part.staves.iter().map(|s| s.key).collect();
            if keys.iter().all(|k| k.is_some()) && keys.windows(2).all(|w| w[0] == w[1]) {
                if let Some(k) = keys[0] {
                    let _ = write!(x, "<key><fifths>{k}</fifths></key>");
                }
            } else {
                for (si, k) in keys.iter().enumerate() {
                    if let Some(k) = k {
                        let _ = write!(x, "<key number=\"{}\"><fifths>{k}</fifths></key>", si + 1);
                    }
                }
            }
            let _ = write!(
                x,
                "<time><beats>{}</beats><beat-type>{}</beat-type></time>",
                score.ts.0, score.ts.1
            );
            if multi {
                let _ = write!(x, "<staves>{nst}</staves>");
                let sym = if part.preset.family == Family::Keyboard {
                    "brace"
                } else {
                    "bracket"
                };
                let _ = write!(x, "<part-symbol>{sym}</part-symbol>");
            }
            for (si, st) in part.staves.iter().enumerate() {
                if let Some(c) = &st.measures[0].clef {
                    x.push_str(&clef_xml(c, multi.then_some(si + 1)));
                }
            }
            for (si, st) in part.staves.iter().enumerate() {
                if let StaffKind::Tab(lines) = st.kind {
                    let num = if multi {
                        format!(" number=\"{}\"", si + 1)
                    } else {
                        String::new()
                    };
                    let _ = write!(x, "<staff-details{num}><staff-lines>{lines}</staff-lines>");
                    if let Some(tuning) = &part.preset.tuning {
                        for (li, &m) in tuning.iter().enumerate() {
                            let (step, alter, oct) = default_spelling(m);
                            let alter = if alter != 0 {
                                format!("<tuning-alter>{alter}</tuning-alter>")
                            } else {
                                String::new()
                            };
                            let _ = write!(
                                x,
                                "<staff-tuning line=\"{}\"><tuning-step>{step}</tuning-step>{alter}<tuning-octave>{oct}</tuning-octave></staff-tuning>",
                                li + 1
                            );
                        }
                    }
                    x.push_str("</staff-details>");
                }
            }
            if part.trans != 0 {
                let tr = part.trans;
                let table = [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6];
                let oct = tr.div_euclid(12);
                let rem = tr.rem_euclid(12);
                let _ = write!(
                    x,
                    "<transpose><diatonic>{}</diatonic><chromatic>{}</chromatic>{}</transpose>",
                    -table[rem as usize],
                    -rem,
                    if oct != 0 {
                        format!("<octave-change>{}</octave-change>", -oct)
                    } else {
                        String::new()
                    }
                );
            }
            x.push_str("</attributes>");
        } else {
            let changes: Vec<String> = part
                .staves
                .iter()
                .enumerate()
                .filter_map(|(si, st)| {
                    st.measures[bi]
                        .clef
                        .as_ref()
                        .map(|c| clef_xml(c, multi.then_some(si + 1)))
                })
                .collect();
            if !changes.is_empty() {
                let _ = write!(x, "<attributes>{}</attributes>", changes.concat());
            }
        }
        let pad = if bi == 0 && !bars.pickup.is_zero() {
            bars.bar - bars.pickup
        } else {
            Q::zero()
        };
        let total_voices: usize = part.staves.iter().map(|s| s.measures[bi].voices.len()).sum();
        let mut written = 0;
        for (si, st) in part.staves.iter().enumerate() {
            let staff_no = si + 1;
            let m = &st.measures[bi];
            if bi == 0 {
                if let Some(tq) = st.tempo {
                    x.push_str(&direction(
                        "above",
                        &format!("<metronome><beat-unit>quarter</beat-unit><per-minute>{tq}</per-minute></metronome>"),
                        staff_no,
                        Some(format!("<sound tempo=\"{tq}\"/>")),
                    ));
                }
            }
            if let Some(d) = &m.dynamic {
                x.push_str(&direction(
                    "below",
                    &format!("<dynamics><{d}/></dynamics>"),
                    staff_no,
                    None,
                ));
            }
            let accs = if st.kind == StaffKind::Pitched {
                accidentals(&ctx, &m.voices, st.key.unwrap_or(0))
            } else {
                HashMap::new()
            };
            for (vi, items) in m.voices.iter().enumerate() {
                let voice_no = si * 4 + vi + 1;
                let bm = beams(items, pad, beat);
                let tp = tuplets(items);
                let mut pos = Q::zero();
                for (k, (off, el)) in items.iter().enumerate() {
                    if *off > pos {
                        let _ = write!(x, "<forward><duration>{}</duration></forward>", dur_div(*off - pos));
                        pos = *off;
                    }
                    write_el(
                        x,
                        &ctx,
                        el,
                        &st.kind,
                        staff_no,
                        voice_no,
                        multi,
                        &accs,
                        (vi, k),
                        bm.get(&k),
                        tp.get(&k),
                    );
                    if !el.grace {
                        pos += el.dur;
                    }
                }
                written += 1;
                if written < total_voices && !pos.is_zero() {
                    let _ = write!(x, "<backup><duration>{}</duration></backup>", dur_div(pos));
                }
            }
        }
        if bi == bars.starts.len() - 1 {
            x.push_str("<barline location=\"right\"><bar-style>light-heavy</bar-style></barline>");
        }
        let _ = len;
        x.push_str("</measure>\n");
    }
    x.push_str("  </part>\n");
}

#[allow(clippy::too_many_arguments)]
fn write_el(
    x: &mut String,
    ctx: &Ctx,
    el: &El,
    kind: &StaffKind,
    staff: usize,
    voice: usize,
    multi: bool,
    accs: &HashMap<(usize, usize, usize), &'static str>,
    key: (usize, usize),
    beams: Option<&Vec<(usize, &'static str)>>,
    tuplet: Option<&Vec<&'static str>>,
) {
    let staff_xml = if multi {
        format!("<staff>{staff}</staff>")
    } else {
        String::new()
    };
    let typ = note_type(el.dur);
    let type_xml = |grace: bool| -> String {
        let (name, dots, trip) = if grace {
            ("eighth", 0, false)
        } else {
            typ.unwrap_or(("quarter", 0, false))
        };
        let mut s = format!("<type>{name}</type>");
        for _ in 0..dots {
            s.push_str("<dot/>");
        }
        let _ = trip;
        s
    };
    let timemod = if !el.grace && typ.map(|t| t.2).unwrap_or(false) {
        "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>"
    } else {
        ""
    };
    if el.rest {
        let _ = write!(
            x,
            "<note>{}<duration>{}</duration><voice>{voice}</voice>",
            if el.measure_rest {
                "<rest measure=\"yes\"/>"
            } else {
                "<rest/>"
            },
            dur_div(el.dur)
        );
        if !el.measure_rest {
            x.push_str(&type_xml(false));
            x.push_str(timemod);
        }
        x.push_str(&staff_xml);
        if let Some(tp) = tuplet {
            x.push_str("<notations>");
            for t in tp {
                let _ = write!(x, "<tuplet type=\"{t}\"/>");
            }
            x.push_str("</notations>");
        }
        x.push_str("</note>");
        return;
    }
    let pitched = *kind != StaffKind::Perc;
    if el.sf {
        x.push_str(&direction("below", "<dynamics><sf/></dynamics>", staff, None));
    }
    if el.pedal_start {
        x.push_str(&direction(
            "below",
            "<pedal type=\"start\" line=\"no\" sign=\"yes\"/>",
            staff,
            None,
        ));
    }
    for (ni, r) in el.notes.iter().enumerate() {
        let n = &ctx.piece.tracks[r.t].notes[r.n];
        x.push_str("<note>");
        if el.grace {
            x.push_str("<grace slash=\"yes\"/>");
        }
        if ni > 0 {
            x.push_str("<chord/>");
        }
        if pitched {
            let (step, alter, oct) = ctx.spelled(r);
            let alter_xml = if alter != 0 {
                format!("<alter>{alter}</alter>")
            } else {
                String::new()
            };
            let _ = write!(x, "<pitch><step>{step}</step>{alter_xml}<octave>{oct}</octave></pitch>");
        } else {
            let (disp, _, _) = ctx
                .drum_display
                .get(&n.pitch)
                .cloned()
                .unwrap_or(("C5".into(), "normal".into(), "up"));
            let (step, _, oct) = parse_spelling(&disp).unwrap_or(('C', 0, 5));
            let _ = write!(
                x,
                "<unpitched><display-step>{step}</display-step><display-octave>{oct}</display-octave></unpitched>"
            );
        }
        if !el.grace {
            let _ = write!(x, "<duration>{}</duration>", dur_div(el.dur));
        }
        match el.tie {
            Some(Tie::Start) => x.push_str("<tie type=\"start\"/>"),
            Some(Tie::Stop) => x.push_str("<tie type=\"stop\"/>"),
            Some(Tie::Continue) => x.push_str("<tie type=\"stop\"/><tie type=\"start\"/>"),
            None => {}
        }
        let _ = write!(x, "<voice>{voice}</voice>");
        x.push_str(&type_xml(el.grace));
        if let Some(a) = accs.get(&(key.0, key.1, ni)) {
            let _ = write!(x, "<accidental>{a}</accidental>");
        }
        x.push_str(timemod);
        if !pitched {
            let first = &ctx.piece.tracks[el.notes[0].t].notes[el.notes[0].n];
            let stem = ctx.drum_display.get(&first.pitch).map(|d| d.2).unwrap_or("up");
            let _ = write!(x, "<stem>{stem}</stem>");
            if let Some((_, head, _)) = ctx.drum_display.get(&n.pitch) {
                if head != "normal" {
                    let _ = write!(x, "<notehead>{head}</notehead>");
                }
            }
        }
        x.push_str(&staff_xml);
        if ni == 0 {
            if let Some(bs) = beams {
                for (num, v) in bs {
                    let _ = write!(x, "<beam number=\"{num}\">{v}</beam>");
                }
            }
        }
        let mut nots = String::new();
        match el.tie {
            Some(Tie::Start) => nots.push_str("<tied type=\"start\"/>"),
            Some(Tie::Stop) => nots.push_str("<tied type=\"stop\"/>"),
            Some(Tie::Continue) => nots.push_str("<tied type=\"stop\"/><tied type=\"start\"/>"),
            None => {}
        }
        if ni == 0 {
            if let Some(tp) = tuplet {
                for t in tp {
                    let _ = write!(nots, "<tuplet type=\"{t}\" bracket=\"yes\"/>");
                }
            }
            if el.fermata {
                nots.push_str("<fermata type=\"upright\"/>");
            }
            if el.trill {
                nots.push_str("<ornaments><trill-mark/></ornaments>");
            }
            if el.staccato {
                nots.push_str("<articulations><staccato/></articulations>");
            }
        }
        let mut tech = String::new();
        if el.fingering {
            if let Some(f) = n.effector.finger {
                let _ = write!(tech, "<fingering>{f}</fingering>");
            }
        }
        if matches!(kind, StaffKind::Tab(_)) {
            if let (Some(s), Some(f)) = (n.effector.string, n.effector.fret) {
                let _ = write!(tech, "<string>{s}</string><fret>{f}</fret>");
            }
        }
        if !tech.is_empty() {
            let _ = write!(nots, "<technical>{tech}</technical>");
        }
        if !nots.is_empty() {
            let _ = write!(x, "<notations>{nots}</notations>");
        }
        x.push_str("</note>");
    }
    if el.pedal_stop {
        x.push_str(&direction(
            "below",
            "<pedal type=\"stop\" line=\"no\" sign=\"yes\"/>",
            staff,
            None,
        ));
    }
}
