//! Resynthesis for verification and the edit gate: notes -> MIDI -> FluidSynth -> model front
//! end (Python worker) -> 50 ms activation cells.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use m2s_core::edit::Resynth;
use m2s_core::ir::Track;
use m2s_core::npy::{self, Array2};
use m2s_core::preset::{self, Family};
use m2s_core::verify::to_cells;

use crate::worker::Worker;

pub fn soundfont() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("M2S_SOUNDFONT") {
        return Ok(p.into());
    }
    for p in [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default-GM.sf2",
        "/usr/share/soundfonts/default.sf2",
    ] {
        if Path::new(p).exists() {
            return Ok(p.into());
        }
    }
    bail!("no General MIDI SoundFont found; install fluid-soundfont-gm or set M2S_SOUNDFONT")
}

pub struct FluidResynth<'a> {
    pub worker: &'a Worker,
    pub workdir: PathBuf,
}

impl FluidResynth<'_> {
    /// Render kept notes in [t0, t1] (shifted to start at 0) to a WAV.
    pub fn render(&self, track: &Track, t0: f64, t1: Option<f64>, wav: &Path) -> Result<()> {
        let p = preset::load(&track.preset)?;
        let kept: Vec<_> = track.kept().collect();
        let bytes = m2s_core::midi::write(&kept, p.gm_program, p.family == Family::Drums, &track.pedal, t0, t1)?;
        let mid = wav.with_extension("mid");
        std::fs::write(&mid, bytes)?;
        let st = Command::new("fluidsynth")
            .args(["-ni", "-g", "0.8", "-r", "44100", "-F"])
            .arg(wav)
            .arg(soundfont()?)
            .arg(&mid)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .context("running fluidsynth")?;
        if !st.success() {
            bail!("fluidsynth failed");
        }
        Ok(())
    }

    /// Model activations of a WAV, as 50 ms cells.
    pub fn cells_of(&self, wav: &Path, preset_name: &str, n_cells: Option<usize>) -> Result<Array2> {
        let out = wav.with_extension("npy");
        let r = self.worker.call(&[
            "activations",
            "--wav",
            &wav.to_string_lossy(),
            "--preset",
            preset_name,
            "--out",
            &out.to_string_lossy(),
        ])?;
        let fps = r["fps"].as_f64().unwrap_or(100.0);
        Ok(to_cells(&npy::read(&out)?, fps, n_cells))
    }
}

impl Resynth for FluidResynth<'_> {
    fn render_cells(&mut self, track: &Track, t0: f64, t1: Option<f64>) -> Result<Array2> {
        let wav = self.workdir.join(format!("render_local_{}.wav", track.name));
        self.render(track, t0, t1, &wav)?;
        self.cells_of(&wav, &track.preset, None)
    }
}
