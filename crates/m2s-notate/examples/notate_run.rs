//! Run the notation stage on files (for parity checks against the Python implementation).
//!
//!     cargo run -p m2s-notate --example notate_run -- notes.json profile.json beats.npy OUT_DIR [--trace]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use m2s_core::ir::Piece;
use m2s_core::trace::Trace;
use m2s_notate::{notate, Options};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        anyhow::bail!("usage: notate_run notes.json profile.json beats.npy OUT_DIR [--trace]");
    }
    let mut piece = Piece::load(&args[1])?;
    let prof: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let acts = if std::path::Path::new(&args[3]).exists() {
        Some(m2s_core::npy::read(&args[3])?)
    } else {
        None
    };
    let out = PathBuf::from(&args[4]);
    std::fs::create_dir_all(&out)?;
    let floats = |k: &str| -> Vec<f64> {
        prof[k]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default()
    };
    let mut trace = if args.iter().any(|a| a == "--trace") {
        Trace::to_file(out.join("trace_notate.jsonl"), "notate")?
    } else {
        Trace::off()
    };
    let t0 = Instant::now();
    let res = notate(
        &mut piece,
        &floats("beats"),
        &floats("downbeats"),
        acts.as_ref(),
        prof["beat_act_fps"].as_f64().unwrap_or(50.0),
        prof["duration"].as_f64(),
        &Options::default(),
        &mut trace,
    )
    .context("notate")?;
    let dt = t0.elapsed().as_secs_f64();
    std::fs::write(out.join("score.musicxml"), &res.musicxml)?;
    std::fs::write(out.join("notation.json"), serde_json::to_string_pretty(&res.info)?)?;
    piece.save(out.join("notes.json"))?;
    eprintln!("notate: {:.3}s", dt);
    Ok(())
}
