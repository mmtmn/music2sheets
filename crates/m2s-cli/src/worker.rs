//! The Python model worker (`python -m m2s.worker ...`): the neural models and Python-only
//! libraries. Each call prints one JSON object as its last stdout line.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use serde_json::Value;

pub struct Worker {
    pub python: PathBuf,
}

impl Worker {
    /// `M2S_PYTHON`, else the nearest `.venv/bin/python` above the current directory, else `python3`.
    pub fn discover() -> Worker {
        if let Ok(p) = std::env::var("M2S_PYTHON") {
            return Worker { python: p.into() };
        }
        let mut dir = std::env::current_dir().ok();
        while let Some(d) = dir {
            let cand = d.join(".venv/bin/python");
            if cand.exists() {
                return Worker { python: cand };
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }
        Worker {
            python: "python3".into(),
        }
    }

    pub fn call(&self, args: &[&str]) -> Result<Value> {
        let mut cmd = Command::new(&self.python);
        cmd.arg("-m")
            .arg("m2s.worker")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let out = cmd
            .output()
            .with_context(|| format!("starting {} -m m2s.worker", self.python.display()))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let tail: Vec<&str> = err.lines().filter(|l| !l.to_lowercase().contains("warn")).collect();
            bail!(
                "worker `{}` failed:\n{}",
                args.join(" "),
                tail[tail.len().saturating_sub(25)..].join("\n")
            );
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let last = stdout
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('{'))
            .unwrap_or("{}");
        serde_json::from_str(last).with_context(|| format!("worker `{}` returned {last:?}", args[0]))
    }
}
