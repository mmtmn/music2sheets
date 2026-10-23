//! `m2s`: audio to playable sheet music. Rust runs the pipeline and every deterministic stage;
//! a Python worker runs the neural models (see `m2s/worker.py`).

mod resynth;
mod run;
mod stages;
mod worker;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use m2s_core::edit::Patch;
use serde_json::json;

use crate::run::Run;
use crate::stages::Ctx;
use crate::worker::Worker;

#[derive(Parser)]
#[command(
    name = "m2s",
    version,
    about = "Audio to playable sheet music, verified against the recording."
)]
struct Cli {
    /// Don't write stage traces (runs/<hash>/traces/*.jsonl)
    #[arg(long, global = true)]
    no_trace: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the whole pipeline on an audio file
    Run {
        audio: PathBuf,
        /// Comma list of presets (e.g. piano or electric_guitar,bass,drums,voice). Default: detect.
        #[arg(long)]
        instruments: Option<String>,
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
        /// Stop after: separate, transcribe or verify
        #[arg(long)]
        until: Option<String>,
        #[arg(long)]
        force: bool,
        #[arg(long, default_value = "auto")]
        engine: String,
        #[arg(long)]
        fingering: bool,
        /// Also copy score.pdf, score.musicxml and the reports here
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Decode and anonymize an audio file; prints the run folder
    Ingest {
        audio: PathBuf,
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Tuning, instruments, beats -> profile.json
    Analyze {
        run: PathBuf,
        #[arg(long)]
        instruments: Option<String>,
        #[arg(long)]
        force: bool,
    },
    /// Split into stems (skipped for a clear solo piano) -> stems.json
    Separate {
        run: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Note candidates per stem -> candidates.json
    Transcribe {
        run: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Playability decode -> notes.json, violations.json
    Decode {
        run: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Resynthesize and compare with the stems -> verify.json, heatmap_*.png
    Verify {
        run: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Quantize, spell, voice, build the score -> score.musicxml
    Notate {
        run: PathBuf,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        fingering: bool,
    },
    /// Render score.musicxml -> score.pdf (MuseScore if installed, else Verovio)
    Engrave {
        run: PathBuf,
        #[arg(long, default_value = "auto")]
        engine: String,
        #[arg(long)]
        force: bool,
    },
    /// Apply a JSON patch through the verification gate; appends to edits.jsonl
    Edit { run: PathBuf, patch: PathBuf },
    /// Write patch proposals from the worst verify regions to proposals/
    Propose {
        run: PathBuf,
        #[arg(long, default_value_t = 12)]
        regions: usize,
    },
    /// The review loop: propose, then apply every proposal through the gate
    Review {
        run: PathBuf,
        #[arg(long, default_value_t = 12)]
        regions: usize,
    },
    /// One image of bars A-B: stem spectrogram, decoded notes, engraved bars
    Inspect {
        run: PathBuf,
        #[arg(long)]
        bars: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Blind scoring against a reference score -> eval.json
    Eval {
        run: PathBuf,
        #[arg(long = "ref")]
        reference: PathBuf,
        #[arg(long, default_value = "auto")]
        repeats: String,
        /// Skip MV2H (its alignment can take over an hour on a long movement)
        #[arg(long)]
        no_mv2h: bool,
        #[arg(long, default_value_t = 0)]
        diff_bars: usize,
    },
    /// Show trace events by stage, time or note id
    Trace {
        run: PathBuf,
        #[arg(long)]
        stage: Option<String>,
        /// time in seconds; events within --window of it
        #[arg(long)]
        at: Option<f64>,
        #[arg(long, default_value_t = 0.5)]
        window: f64,
        #[arg(long)]
        note: Option<String>,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long, default_value_t = 200)]
        limit: usize,
    },
    /// Decode a candidates JSON file outside a run (used by the Python dev tools)
    DecodeFile {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        weights: Option<PathBuf>,
    },
    /// Notate decoded notes outside a run: NOTES.json PROFILE.json BEATS.npy OUT_DIR (dev tools)
    NotateFile {
        notes: PathBuf,
        profile: PathBuf,
        beats: PathBuf,
        out: PathBuf,
    },
    /// List instrument presets
    Presets,
}

fn ctx_for(target: &Path, trace: bool) -> Result<Ctx> {
    let run = if target.is_dir() {
        Run::open(target)?
    } else {
        stages::ingest(target, Path::new("runs"))?
    };
    Ok(Ctx {
        run,
        worker: Worker::discover(),
        trace,
        t0: Instant::now(),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let trace = !cli.no_trace;
    match cli.cmd {
        Cmd::Run {
            audio,
            instruments,
            runs_root,
            until,
            force,
            engine,
            fingering,
            out,
        } => {
            let run = stages::ingest(&audio, &runs_root)?;
            let mut c = Ctx {
                run,
                worker: Worker::discover(),
                trace,
                t0: Instant::now(),
            };
            c.log(format!("run {}", c.run.root.display()));
            stages::run_all(
                &mut c,
                &stages::RunOpts {
                    instruments: instruments.as_deref(),
                    until: until.as_deref(),
                    force,
                    engine: &engine,
                    fingering,
                },
            )?;
            if let Some(out) = out {
                std::fs::create_dir_all(&out)?;
                for f in [
                    "score.pdf",
                    "score.musicxml",
                    "summary.json",
                    "verify.json",
                    "violations.json",
                    "notation.json",
                ] {
                    let p = c.run.root.join(f);
                    if p.exists() {
                        std::fs::copy(&p, out.join(f))?;
                    }
                }
            }
        }
        Cmd::Ingest { audio, runs_root } => println!("{}", stages::ingest(&audio, &runs_root)?.root.display()),
        Cmd::Analyze {
            run,
            instruments,
            force,
        } => {
            let mut c = ctx_for(&run, trace)?;
            let p = stages::analyze(&mut c, instruments.as_deref(), force)?;
            println!(
                "{}",
                json!({"duration": p["duration"], "tuning_cents": p["tuning_cents"], "tempo_bpm": p["tempo_bpm"],
                                  "instruments": p["instruments"]["present"]})
            );
        }
        Cmd::Separate { run, force } => println!("{}", stages::separate(&mut ctx_for(&run, trace)?, force)?),
        Cmd::Transcribe { run, force } => stages::transcribe(&mut ctx_for(&run, trace)?, force)?,
        Cmd::Decode { run, force } => {
            stages::decode(&mut ctx_for(&run, trace)?, force)?;
        }
        Cmd::Verify { run, force } => {
            let v = stages::verify(&mut ctx_for(&run, trace)?, force)?;
            for t in v["tracks"].as_array().into_iter().flatten() {
                println!(
                    "{}",
                    json!({"track": t["track"], "audio_match": t["audio_match"], "precision": t["precision"], "recall": t["recall"]})
                );
            }
        }
        Cmd::Notate { run, force, fingering } => {
            stages::notate(&mut ctx_for(&run, trace)?, force, fingering)?;
        }
        Cmd::Engrave { run, engine, force } => {
            stages::engrave(&mut ctx_for(&run, trace)?, force, &engine)?;
        }
        Cmd::Edit { run, patch } => {
            let p: Patch = serde_json::from_str(&std::fs::read_to_string(&patch)?)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&stages::edit(&mut ctx_for(&run, trace)?, &p)?)?
            );
        }
        Cmd::Propose { run, regions } => {
            let ps = stages::propose(&ctx_for(&run, trace)?, regions)?;
            println!("{} proposals", ps.len());
        }
        Cmd::Review { run, regions } => println!("{}", stages::review(&mut ctx_for(&run, trace)?, regions)?),
        Cmd::Inspect { run, bars, out } => {
            let c = ctx_for(&run, trace)?;
            let root = c.run.root.to_string_lossy().to_string();
            let mut args = vec!["inspect", "--run", root.as_str(), "--bars", bars.as_str()];
            let o = out.map(|o| o.to_string_lossy().to_string());
            if let Some(o) = &o {
                args.extend(["--out", o.as_str()]);
            }
            println!("{}", c.worker.call(&args)?["image"].as_str().unwrap_or(""));
        }
        Cmd::Eval {
            run,
            reference,
            repeats,
            no_mv2h,
            diff_bars,
        } => {
            let mut c = ctx_for(&run, trace)?;
            let rep = stages::eval(
                &mut c,
                &stages::EvalOpts {
                    reference: &reference,
                    repeats: &repeats,
                    mv2h: !no_mv2h,
                    diff_bars,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&rep["summary"])?);
        }
        Cmd::Trace {
            run,
            stage,
            at,
            window,
            note,
            kind,
            limit,
        } => {
            let r = Run::open(run)?;
            let n = stages::trace_query(
                &r,
                &stages::TraceQuery {
                    stage: stage.as_deref(),
                    at,
                    window,
                    note: note.as_deref(),
                    kind: kind.as_deref(),
                    limit,
                },
            )?;
            eprintln!("{n} events");
        }
        Cmd::DecodeFile { input, output, weights } => {
            let mut piece = m2s_core::ir::Piece::load(&input)?;
            let w = match weights {
                Some(p) => Some(serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p)?)?),
                None => None,
            };
            for tr in piece.tracks.iter_mut() {
                match (&w, m2s_core::preset::load(&tr.preset)?.family) {
                    (Some(w), m2s_core::preset::Family::Keyboard) => {
                        let weights = m2s_core::play::piano::PianoWeights::default().with_overrides(w)?;
                        m2s_core::play::piano::PianoDecoder::with_weights(&tr.preset, weights)?
                            .decode(tr, &mut m2s_core::trace::Trace::off())?;
                    }
                    _ => {
                        m2s_core::play::decode_track(tr, &mut m2s_core::trace::Trace::off())?;
                    }
                }
            }
            piece.save(&output)?;
        }
        Cmd::NotateFile {
            notes,
            profile,
            beats,
            out,
        } => {
            std::fs::create_dir_all(&out)?;
            let mut piece = m2s_core::ir::Piece::load(&notes)?;
            let prof: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&profile)?)?;
            let f64s = |v: &serde_json::Value| {
                v.as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect::<Vec<_>>())
                    .unwrap_or_default()
            };
            let acts = m2s_core::npy::read(&beats)?;
            let mut tr = if trace {
                m2s_core::trace::Trace::to_file(out.join("trace_notate.jsonl"), "notate")?
            } else {
                m2s_core::trace::Trace::off()
            };
            let n = m2s_notate::notate(
                &mut piece,
                &f64s(&prof["beats"]),
                &f64s(&prof["downbeats"]),
                Some(&acts),
                prof["beat_act_fps"].as_f64().unwrap_or(50.0),
                prof["duration"].as_f64(),
                &m2s_notate::Options::default(),
                &mut tr,
            )?;
            std::fs::write(out.join("score.musicxml"), &n.musicxml)?;
            std::fs::write(out.join("notation.json"), serde_json::to_string_pretty(&n.info)?)?;
            piece.save(out.join("scored.json"))?;
        }
        Cmd::Presets => {
            for name in m2s_core::preset::available() {
                let p = m2s_core::preset::load(name)?;
                println!(
                    "{:16} {:9} {}",
                    name,
                    format!("{:?}", p.family).to_lowercase(),
                    p.display
                );
            }
        }
    }
    Ok(())
}
