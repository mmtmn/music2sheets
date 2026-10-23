//! Run folders: one per input, named after the audio's content hash (the file name is never
//! stored). `manifest.json` records each stage's input key so reruns skip finished stages.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn file_hash(path: &Path, n: usize) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(hex::encode(Sha256::digest(&bytes))[..n].to_string())
}

pub fn key_hash(v: &Value) -> String {
    hex::encode(Sha256::digest(v.to_string().as_bytes()))[..16].to_string()
}

pub struct Run {
    pub root: PathBuf,
    pub manifest: Value,
}

impl Run {
    pub fn open(root: impl Into<PathBuf>) -> Result<Run> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let mp = root.join("manifest.json");
        let manifest = if mp.exists() {
            serde_json::from_str(&std::fs::read_to_string(&mp)?).unwrap_or_else(|_| json!({"stages": {}}))
        } else {
            json!({"stages": {}})
        };
        let mut r = Run { root, manifest };
        if r.manifest.get("stages").is_none() {
            r.manifest["stages"] = json!({});
        }
        Ok(r)
    }

    pub fn path(&self, rel: &str) -> PathBuf {
        let p = self.root.join(rel);
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        p
    }

    pub fn has(&self, rel: &str) -> bool {
        self.root.join(rel).exists()
    }

    pub fn is_fresh(&self, stage: &str, key: &Value, outputs: &[&str]) -> bool {
        let rec = &self.manifest["stages"][stage];
        rec["key"].as_str() == Some(key_hash(key).as_str()) && outputs.iter().all(|o| self.has(o))
    }

    pub fn stage_key(&self, stage: &str) -> Value {
        self.manifest["stages"][stage]["key"].clone()
    }

    pub fn mark(&mut self, stage: &str, key: &Value, info: Value) -> Result<()> {
        let mut rec = json!({"key": key_hash(key), "time": now()});
        if let (Some(r), Value::Object(i)) = (rec.as_object_mut(), info) {
            r.extend(i);
        }
        self.manifest["stages"][stage] = rec;
        self.save()
    }

    pub fn edits(&self) -> u64 {
        self.manifest["edits"].as_u64().unwrap_or(0)
    }

    pub fn bump_edits(&mut self) -> Result<()> {
        self.manifest["edits"] = json!(self.edits() + 1);
        self.save()
    }

    pub fn save(&self) -> Result<()> {
        std::fs::write(
            self.root.join("manifest.json"),
            serde_json::to_string_pretty(&self.manifest)?,
        )?;
        Ok(())
    }

    pub fn read_json(&self, rel: &str) -> Result<Value> {
        let p = self.root.join(rel);
        serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?)
            .with_context(|| format!("parsing {}", p.display()))
    }

    pub fn write_json(&self, rel: &str, v: &Value) -> Result<()> {
        std::fs::write(self.path(rel), serde_json::to_string_pretty(v)?)?;
        Ok(())
    }
}

pub fn now() -> String {
    let out = std::process::Command::new("date").arg("+%Y-%m-%dT%H:%M:%S").output();
    out.ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}
