//! Parity runner: decode a candidates.json with the Rust solvers.
//! usage: decode_run CANDIDATES.json OUT_DIR   -> OUT_DIR/notes.json, violations.json, trace_decode.jsonl

use m2s_core::ir::Piece;
use m2s_core::play::decode_track;
use m2s_core::trace::Trace;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (input, out) = (&args[1], std::path::Path::new(&args[2]));
    std::fs::create_dir_all(out)?;
    let mut piece = Piece::load(input)?;
    let mut trace = Trace::to_file(out.join("trace_decode.jsonl"), "decode")?;
    let t0 = std::time::Instant::now();
    let mut viols = vec![];
    for tr in piece.tracks.iter_mut() {
        let name = tr.name.clone();
        for v in decode_track(tr, &mut trace)? {
            let mut j = serde_json::to_value(&v)?;
            j["track"] = serde_json::Value::String(name.clone());
            viols.push(j);
        }
    }
    eprintln!("decoded in {:.3}s", t0.elapsed().as_secs_f64());
    piece.save(out.join("notes.json"))?;
    std::fs::write(out.join("violations.json"), serde_json::to_string_pretty(&viols)?)?;
    Ok(())
}
