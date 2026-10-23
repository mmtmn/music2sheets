//! Pipeline stages: ingest -> analyze -> separate -> transcribe (Python worker) -> decode ->
//! verify -> notate (Rust) -> engrave (worker). Each stage records an input key in the run's
//! manifest and is skipped when its inputs haven't changed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use m2s_core::edit::{self, Patch};
use m2s_core::eval as ev;
use m2s_core::ir::Piece;
use m2s_core::npy::{self, Array2};
use m2s_core::play::decode_track;
use m2s_core::trace::Trace;
use m2s_core::verify;
use serde_json::{json, Value};

use crate::resynth::FluidResynth;
use crate::run::{file_hash, Run};
use crate::worker::Worker;

/// bump when the worker's outputs change meaning
const WORKER_V: &str = "1";
const CORE_V: &str = env!("CARGO_PKG_VERSION");

pub struct Ctx {
    pub run: Run,
    pub worker: Worker,
    pub trace: bool,
    pub t0: Instant,
}

impl Ctx {
    pub fn log(&self, msg: impl AsRef<str>) {
        eprintln!("[m2s {:6.1}s] {}", self.t0.elapsed().as_secs_f64(), msg.as_ref());
    }

    fn trace(&self, stage: &str) -> Result<Trace> {
        if self.trace {
            Ok(Trace::to_file(self.run.path(&format!("traces/{stage}.jsonl")), stage)?)
        } else {
            Ok(Trace::off())
        }
    }
}

/// Decode to a hash-named WAV. The original file name is not stored anywhere in the run.
pub fn ingest(src: &Path, runs_root: &Path) -> Result<Run> {
    let h = file_hash(src, 12)?;
    let mut run = Run::open(runs_root.join(&h))?;
    let key = json!({"src": h, "v": WORKER_V});
    if run.is_fresh("ingest", &key, &["audio.wav"]) {
        return Ok(run);
    }
    let st = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args(["-map_metadata", "-1", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le"])
        .arg(run.path("audio.wav"))
        .status()
        .context("running ffmpeg")?;
    if !st.success() {
        bail!("ffmpeg could not decode {}", src.display());
    }
    run.mark("ingest", &key, json!({"input_hash": h}))?;
    Ok(run)
}

pub fn analyze(c: &mut Ctx, instruments: Option<&str>, force: bool) -> Result<Value> {
    let key = json!({"in": c.run.stage_key("ingest"), "instruments": instruments, "v": WORKER_V});
    if !force && c.run.is_fresh("analyze", &key, &["profile.json", "acts/beats.npy"]) {
        return c.run.read_json("profile.json");
    }
    c.log("analyze: tuning, instruments, beats");
    let root = c.run.root.to_string_lossy().to_string();
    let mut args = vec!["analyze", "--run", root.as_str()];
    if let Some(i) = instruments {
        args.extend(["--instruments", i]);
    }
    let r = c.worker.call(&args)?;
    c.run.mark("analyze", &key, json!({}))?;
    c.log(format!("analyze: {r}"));
    c.run.read_json("profile.json")
}

pub fn separate(c: &mut Ctx, force: bool) -> Result<Value> {
    let key = json!({"in": c.run.stage_key("analyze"), "v": WORKER_V});
    if !force && c.run.is_fresh("separate", &key, &["stems.json"]) {
        return c.run.read_json("stems.json");
    }
    let root = c.run.root.to_string_lossy().to_string();
    let r = c.worker.call(&["separate", "--run", &root])?;
    c.log(format!("separate: {}", r["note"].as_str().unwrap_or("")));
    c.run.mark("separate", &key, json!({}))?;
    c.run.read_json("stems.json")
}

pub fn transcribe(c: &mut Ctx, force: bool) -> Result<()> {
    let key = json!({"in": c.run.stage_key("separate"), "v": WORKER_V});
    if !force && c.run.is_fresh("transcribe", &key, &["candidates.json"]) {
        return Ok(());
    }
    c.log("transcribe: running the models");
    let root = c.run.root.to_string_lossy().to_string();
    let r = c.worker.call(&["transcribe", "--run", &root])?;
    c.log(format!("transcribe: candidates {}", r["candidates"]));
    c.run.mark("transcribe", &key, json!({}))
}

pub fn decode(c: &mut Ctx, force: bool) -> Result<Piece> {
    let key = json!({"in": c.run.stage_key("transcribe"), "v": CORE_V});
    if !force && c.run.is_fresh("decode", &key, &["notes.json", "violations.json"]) {
        return Piece::load(c.run.root.join("notes.json"));
    }
    let mut piece = Piece::load(c.run.root.join("candidates.json"))?;
    let mut trace = c.trace("decode")?;
    let t = Instant::now();
    let mut viols = vec![];
    for tr in piece.tracks.iter_mut() {
        let v = decode_track(tr, &mut trace)?;
        c.log(format!(
            "decode: {} kept {}/{}, {} violations",
            tr.name,
            tr.kept().count(),
            tr.notes.len(),
            v.len()
        ));
        for x in v {
            let mut j = serde_json::to_value(&x)?;
            j["track"] = json!(tr.name);
            viols.push(j);
        }
    }
    c.log(format!("decode: {:.2}s", t.elapsed().as_secs_f64()));
    piece.save(c.run.root.join("notes.json"))?;
    c.run.write_json("violations.json", &json!(viols))?;
    c.run.mark("decode", &key, json!({}))?;
    Ok(piece)
}

fn stem_wav(c: &Ctx, track: &m2s_core::ir::Track) -> PathBuf {
    if Path::new(&track.stem).exists() {
        return track.stem.clone().into();
    }
    if let Ok(stems) = c.run.read_json("stems.json") {
        for k in [&track.name, &track.preset] {
            if let Some(p) = stems[k.as_str()].as_str() {
                return p.into();
            }
        }
    }
    c.run.root.join("audio.wav")
}

/// Cached stem activation cells for a track (piano: the transcription's own onset map).
fn stem_cells(c: &Ctx, rs: &FluidResynth, track: &m2s_core::ir::Track) -> Result<Array2> {
    let cached = c.run.root.join(format!("acts/{}_cells.npy", track.name));
    if cached.exists() {
        return npy::read(&cached);
    }
    let piano = c.run.root.join("acts/stem_piano.npy");
    let cells = if track.preset == "piano" && piano.exists() {
        verify::to_cells(&npy::read(&piano)?, 100.0, None)
    } else {
        rs.cells_of(&stem_wav(c, track), &track.preset, None)?
    };
    npy::write(c.run.path(&format!("acts/{}_cells.npy", track.name)), &cells)?;
    Ok(cells)
}

pub fn verify(c: &mut Ctx, force: bool) -> Result<Value> {
    let key = json!({"in": c.run.stage_key("decode"), "edits": c.run.edits(), "v": CORE_V});
    if !force && c.run.is_fresh("verify", &key, &["verify.json"]) {
        return c.run.read_json("verify.json");
    }
    let piece = Piece::load(c.run.root.join("notes.json"))?;
    if force {
        for tr in &piece.tracks {
            let _ = std::fs::remove_file(c.run.root.join(format!("acts/{}_cells.npy", tr.name)));
        }
    }
    let rs = FluidResynth {
        worker: &c.worker,
        workdir: c.run.root.clone(),
    };
    let mut reports = vec![];
    for tr in &piece.tracks {
        let sc = stem_cells(c, &rs, tr)?;
        let wav = c.run.root.join(format!("render_{}.wav", tr.name));
        rs.render(tr, 0.0, None, &wav)?;
        let rc = rs.cells_of(&wav, &tr.preset, Some(sc.rows))?;
        let (rep, m) = verify::report(&tr.name, &sc, &rc);
        verify::heatmap(&m, &sc, &c.run.path(&format!("heatmap_{}.png", tr.name)))?;
        c.log(format!(
            "verify: {} audio match {} (precision {}, recall {})",
            tr.name, rep["audio_match"], rep["precision"], rep["recall"]
        ));
        reports.push(rep);
    }
    let out = json!({"tracks": reports});
    c.run.write_json("verify.json", &out)?;
    c.run.mark("verify", &key, json!({}))?;
    Ok(out)
}

pub fn notate(c: &mut Ctx, force: bool, fingering: bool) -> Result<Value> {
    let key = json!({"in": c.run.stage_key("decode"), "edits": c.run.edits(), "fingering": fingering, "v": CORE_V});
    if !force
        && c.run
            .is_fresh("notate", &key, &["score.musicxml", "notation.json", "scored.json"])
    {
        return c.run.read_json("notation.json");
    }
    let prof = c.run.read_json("profile.json")?;
    let mut piece = Piece::load(c.run.root.join("notes.json"))?;
    let f64s = |v: &Value| {
        v.as_array()
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    let (beats, downbeats) = (f64s(&prof["beats"]), f64s(&prof["downbeats"]));
    let acts = c.run.root.join("acts/beats.npy");
    let acts = if acts.exists() { Some(npy::read(&acts)?) } else { None };
    let opts = m2s_notate::Options {
        fingering,
        dynamics: true,
        pedal: false,
        title: format!("Transcription {}", c.run.root.file_name().unwrap().to_string_lossy()),
    };
    let mut trace = c.trace("notate")?;
    let t = Instant::now();
    let n = m2s_notate::notate(
        &mut piece,
        &beats,
        &downbeats,
        acts.as_ref(),
        prof["beat_act_fps"].as_f64().unwrap_or(50.0),
        prof["duration"].as_f64(),
        &opts,
        &mut trace,
    )?;
    std::fs::write(c.run.root.join("score.musicxml"), &n.musicxml)?;
    // notation changes notes (trills, graces, score positions) and is not idempotent: its input stays
    // the decoder's notes.json (plus edits); the notated notes go to scored.json
    piece.save(c.run.root.join("scored.json"))?;
    c.run.write_json("notation.json", &n.info)?;
    c.run.mark("notate", &key, json!({}))?;
    let i = &n.info;
    c.log(format!(
        "notate: {} bars, {}, {}, q={}, lint {} ({:.2}s)",
        i["bars"],
        i["time_signature"],
        i["key"],
        i["qpm"],
        i["lint"]["by_kind"],
        t.elapsed().as_secs_f64()
    ));
    Ok(n.info)
}

pub fn engrave(c: &mut Ctx, force: bool, engine: &str) -> Result<String> {
    let key = json!({"in": c.run.stage_key("notate"), "engine": engine, "v": WORKER_V});
    if !force && c.run.is_fresh("engrave", &key, &["score.pdf"]) {
        return Ok(c.run.manifest["stages"]["engrave"]["engine"]
            .as_str()
            .unwrap_or("?")
            .to_string());
    }
    let (xml, pdf) = (c.run.root.join("score.musicxml"), c.run.root.join("score.pdf"));
    let r = c.worker.call(&[
        "engrave",
        "--musicxml",
        &xml.to_string_lossy(),
        "--pdf",
        &pdf.to_string_lossy(),
        "--engine",
        engine,
    ])?;
    let used = r["engine"].as_str().unwrap_or("?").to_string();
    c.run.mark("engrave", &key, json!({"engine": used}))?;
    c.log(format!("engrave: score.pdf ({used})"));
    Ok(used)
}

pub fn summary(c: &Ctx) -> Result<Value> {
    let mut out = json!({"run": c.run.root.file_name().unwrap().to_string_lossy(), "version": CORE_V});
    if let Ok(p) = c.run.read_json("profile.json") {
        out["profile"] = json!({"duration": p["duration"], "tuning_cents": p["tuning_cents"], "tempo_bpm": p["tempo_bpm"],
                                "instruments": p["instruments"]["present"]});
    }
    if let Ok(v) = c.run.read_json("verify.json") {
        let m: serde_json::Map<String, Value> = v["tracks"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| (t["track"].as_str().unwrap_or("").to_string(), t["audio_match"].clone()))
            .collect();
        out["audio_match"] = Value::Object(m);
    }
    if let Ok(n) = c.run.read_json("notation.json") {
        out["notation"] = json!({"time_signature": n["time_signature"], "key": n["key"], "qpm": n["qpm"], "bars": n["bars"],
                                 "ornaments": n["ornaments"], "lint": n["lint"]["by_kind"]});
    }
    if let Ok(v) = c.run.read_json("violations.json") {
        let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
        for x in v.as_array().into_iter().flatten() {
            *kinds.entry(x["kind"].as_str().unwrap_or("?").to_string()).or_default() += 1;
        }
        out["violations"] = json!(kinds);
    }
    c.run.write_json("summary.json", &out)?;
    Ok(out)
}

pub struct RunOpts<'a> {
    pub instruments: Option<&'a str>,
    pub until: Option<&'a str>,
    pub force: bool,
    pub engine: &'a str,
    pub fingering: bool,
}

pub fn run_all(c: &mut Ctx, o: &RunOpts) -> Result<()> {
    analyze(c, o.instruments, o.force)?;
    separate(c, o.force)?;
    if o.until == Some("separate") {
        return Ok(());
    }
    transcribe(c, o.force)?;
    if o.until == Some("transcribe") {
        return Ok(());
    }
    decode(c, o.force)?;
    verify(c, o.force)?;
    if o.until == Some("verify") {
        return Ok(());
    }
    notate(c, o.force, o.fingering)?;
    engrave(c, o.force, o.engine)?;
    let s = summary(c)?;
    c.log(format!(
        "done -> {}/score.pdf  audio match {}",
        c.run.root.display(),
        s["audio_match"]
    ));
    Ok(())
}

fn all_stem_cells(c: &Ctx, piece: &Piece) -> Result<BTreeMap<String, Array2>> {
    let mut out = BTreeMap::new();
    for tr in &piece.tracks {
        let p = c.run.root.join(format!("acts/{}_cells.npy", tr.name));
        if p.exists() {
            out.insert(tr.name.clone(), npy::read(&p)?);
        }
    }
    Ok(out)
}

/// Apply one patch through the gate; log it; save if accepted.
pub fn edit(c: &mut Ctx, patch: &Patch) -> Result<Value> {
    let before = Piece::load(c.run.root.join("notes.json"))?;
    let cells = all_stem_cells(c, &before)?;
    let mut rs = FluidResynth {
        worker: &c.worker,
        workdir: c.run.root.clone(),
    };
    let (mut rec, after) = edit::gate(&before, patch, &cells, &mut rs)?;
    rec["time"] = json!(crate::run::now());
    if let Some(after) = after {
        after.save(c.run.root.join("notes.json"))?;
        c.run.bump_edits()?;
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(c.run.root.join("edits.jsonl"))?;
    writeln!(f, "{rec}")?;
    Ok(rec)
}

/// Patch proposals from the worst verify regions (see README, "The review loop").
pub fn propose(c: &Ctx, n_regions: usize) -> Result<Vec<PathBuf>> {
    let piece = Piece::load(c.run.root.join("notes.json"))?;
    let rep = c.run.read_json("verify.json")?;
    let dir = c.run.path("proposals/.keep");
    let dir = dir.parent().unwrap().to_path_buf();
    for e in std::fs::read_dir(&dir)?.flatten() {
        if e.path().extension().map(|x| x == "json").unwrap_or(false) {
            std::fs::remove_file(e.path())?;
        }
    }
    let mut out = vec![];
    for tr_rep in rep["tracks"].as_array().into_iter().flatten() {
        let Some(tr) = piece.track(tr_rep["track"].as_str().unwrap_or("")) else {
            continue;
        };
        for reg in tr_rep["regions"].as_array().into_iter().flatten().take(n_regions) {
            let (t0, t1) = (reg["start"].as_f64().unwrap_or(0.0), reg["end"].as_f64().unwrap_or(0.0));
            let has = |key: &str, p: i32| {
                reg[key]
                    .as_array()
                    .map(|a| a.iter().any(|x| x.as_i64() == Some(p as i64)))
                    .unwrap_or(false)
            };
            for n in tr.notes.iter().filter(|n| t0 <= n.onset && n.onset <= t1) {
                let patch = if !n.keep && has("missed_pitches", n.pitch) && !n.has_tag("trill-note") {
                    json!({"reason": format!("verify region {t0}-{t1}s: stem has pitch {}, score lacks it; restoring dropped candidate {} (conf {})", n.pitch, n.id, n.confidence),
                           "ops": [{"op": "add", "track": tr.name, "onset": n.onset, "offset": n.offset, "pitch": n.pitch, "velocity": n.velocity}]})
                } else if n.keep && has("extra_pitches", n.pitch) && n.confidence < 0.9 {
                    json!({"reason": format!("verify region {t0}-{t1}s: score has pitch {} without stem evidence; deleting low-confidence note {} (conf {})", n.pitch, n.id, n.confidence),
                           "ops": [{"op": "delete", "id": n.id}]})
                } else {
                    continue;
                };
                let p = dir.join(format!("p{:03}.json", out.len() + 1));
                std::fs::write(&p, serde_json::to_string_pretty(&patch)?)?;
                out.push(p);
            }
        }
    }
    Ok(out)
}

/// The review loop, automated: propose from the worst regions, apply each through the gate.
pub fn review(c: &mut Ctx, n_regions: usize) -> Result<Value> {
    let props = propose(c, n_regions)?;
    let (mut acc, mut rej) = (0, 0);
    for p in &props {
        let patch: Patch = serde_json::from_str(&std::fs::read_to_string(p)?)?;
        let rec = edit(c, &patch)?;
        if rec["accepted"].as_bool() == Some(true) {
            acc += 1;
        } else {
            rej += 1;
        }
        c.log(format!(
            "review: {} -> {} {}",
            p.file_name().unwrap().to_string_lossy(),
            if rec["accepted"].as_bool() == Some(true) {
                "accepted"
            } else {
                "rejected"
            },
            rec["audio_match"]
        ));
    }
    Ok(json!({"proposals": props.len(), "accepted": acc, "rejected": rej}))
}

pub struct EvalOpts<'a> {
    pub reference: &'a Path,
    pub repeats: &'a str,
    pub mv2h: bool,
    pub diff_bars: usize,
}

pub fn eval(c: &mut Ctx, o: &EvalOpts) -> Result<Value> {
    let root = c.run.root.to_string_lossy().to_string();
    let aligned = c.run.path("eval_reference.json");
    c.log("eval: loading and aligning the reference (worker)");
    c.worker.call(&[
        "align",
        "--run",
        &root,
        "--ref",
        &o.reference.to_string_lossy(),
        "--repeats",
        o.repeats,
        "--out",
        &aligned.to_string_lossy(),
    ])?;
    let a: Value = serde_json::from_str(&std::fs::read_to_string(&aligned)?)?;
    let refs: Vec<ev::RefNote> = serde_json::from_value(a["notes"].clone())?;
    // score what was engraved: the notated notes when there are any
    let scored = c.run.root.join("scored.json");
    let piece = Piece::load(if scored.exists() {
        scored
    } else {
        c.run.root.join("notes.json")
    })?;
    let ours: Vec<&m2s_core::ir::Note> = piece.tracks.iter().flat_map(|t| t.kept()).collect();
    let viols = c.run.read_json("violations.json")?;
    let hard = viols
        .as_array()
        .into_iter()
        .flatten()
        .filter(|v| ev::HARD.contains(&v["kind"].as_str().unwrap_or("")))
        .count();
    let notation = c.run.read_json("notation.json")?;
    let mut rep = json!({
        "reference": a["reference"], "alignment": a["alignment"],
        "notes": {"onset_50ms": ev::prf(&refs, &ours, 0.05), "onset_100ms": ev::prf(&refs, &ours, 0.1),
                  "onset_250ms": ev::prf(&refs, &ours, 0.25)},
        "hands": ev::hand_agreement(&refs, &ours),
        "playability": {"ours": m2s_core::play::round_to(100.0 * hard as f64 / ours.len().max(1) as f64, 2),
                        "reference": ev::reference_playability(&refs)?},
        "score": {"ours": {"time_signature": notation["time_signature"], "key": notation["key"], "bars": notation["bars"],
                           "qpm": notation["qpm"]},
                  "reference": {"time_signature": a["reference"]["time_signature"], "key": a["reference"]["key"],
                                "bars": a["reference"]["bars"]}},
    });
    let score_path = a["score_path"].as_str().unwrap_or("").to_string();
    let xml = c.run.root.join("score.musicxml").to_string_lossy().to_string();
    rep["mv2h"] = if o.mv2h {
        c.log("eval: MV2H (slow on long pieces)");
        c.worker
            .call(&["mv2h", "--ours", &xml, "--ref", &score_path])
            .unwrap_or_else(|e| json!({"error": e.to_string()}))
    } else {
        json!({"skipped": "disabled"})
    };
    rep["musicdiff"] = if o.diff_bars > 0 {
        c.worker
            .call(&[
                "musicdiff",
                "--ours",
                &xml,
                "--ref",
                &score_path,
                "--bars",
                &o.diff_bars.to_string(),
            ])
            .unwrap_or_else(|e| json!({"error": e.to_string()}))
    } else {
        json!({"skipped": "pass --diff-bars N to run musicdiff on the first N bars"})
    };
    rep["summary"] = json!({
        "note_f1_50ms": rep["notes"]["onset_50ms"]["f1"], "note_f1_100ms": rep["notes"]["onset_100ms"]["f1"],
        "note_f1_250ms": rep["notes"]["onset_250ms"]["f1"], "hand_agreement": rep["hands"]["agreement"],
        "violations_per_100_notes": rep["playability"], "mv2h": rep["mv2h"]["MV2H"],
        "musicdiff_edits": rep["musicdiff"]["edits"],
        "time_signature": [notation["time_signature"], a["reference"]["time_signature"]],
        "key": [notation["key"], a["reference"]["key"]],
    });
    c.run.write_json("eval.json", &rep)?;
    Ok(rep)
}

pub struct TraceQuery<'a> {
    pub stage: Option<&'a str>,
    pub at: Option<f64>,
    pub window: f64,
    pub note: Option<&'a str>,
    pub kind: Option<&'a str>,
    pub limit: usize,
}

/// Print trace events matching a stage, a time (±window) or a note id.
pub fn trace_query(run: &Run, q: &TraceQuery) -> Result<usize> {
    let dir = run.root.join("traces");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| format!("no traces in {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "jsonl").unwrap_or(false))
        .collect();
    files.sort();
    let mut shown = 0;
    for f in files {
        let stage = f.file_stem().unwrap().to_string_lossy().to_string();
        if q.stage.map(|s| s != stage).unwrap_or(false) {
            continue;
        }
        for line in std::fs::read_to_string(&f)?.lines() {
            if let Some(n) = q.note {
                if !line.contains(&format!("\"{n}\"")) {
                    continue;
                }
            }
            let v: Value = serde_json::from_str(line).unwrap_or(Value::Null);
            if let Some(k) = q.kind {
                if v["kind"].as_str() != Some(k) {
                    continue;
                }
            }
            if let Some(at) = q.at {
                match v["t"].as_f64().or_else(|| v["onset"].as_f64()) {
                    Some(t) if (t - at).abs() <= q.window => {}
                    _ => continue,
                }
            }
            println!("{line}");
            shown += 1;
            if shown >= q.limit {
                return Ok(shown);
            }
        }
    }
    Ok(shown)
}
