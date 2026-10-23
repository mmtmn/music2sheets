//! Stage traces: one JSON object per line, so any decision can be found later by
//! stage, time or note id (`m2s trace RUN --stage decode --at 12.3`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::{json, Value};

#[derive(Default)]
pub struct Trace {
    out: Option<BufWriter<File>>,
    stage: String,
    pub events: usize,
}

impl Trace {
    /// A trace that records nothing.
    pub fn off() -> Self {
        Trace::default()
    }

    pub fn to_file(path: impl AsRef<Path>, stage: &str) -> std::io::Result<Self> {
        Ok(Trace {
            out: Some(BufWriter::new(File::create(path)?)),
            stage: stage.to_string(),
            events: 0,
        })
    }

    pub fn enabled(&self) -> bool {
        self.out.is_some()
    }

    /// Record one event. `data` should be a JSON object; it is merged with {stage, kind}.
    pub fn emit(&mut self, kind: &str, data: Value) {
        let Some(out) = self.out.as_mut() else { return };
        let mut obj = json!({"stage": self.stage, "kind": kind});
        if let (Some(o), Value::Object(d)) = (obj.as_object_mut(), data) {
            o.extend(d);
        }
        let _ = writeln!(out, "{obj}");
        self.events += 1;
    }

    /// Emit only when tracing is on; `f` builds the payload lazily.
    pub fn with(&mut self, kind: &str, f: impl FnOnce() -> Value) {
        if self.enabled() {
            let v = f();
            self.emit(kind, v);
        }
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        if let Some(o) = self.out.as_mut() {
            let _ = o.flush();
        }
    }
}
