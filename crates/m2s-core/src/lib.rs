//! music2sheets core: everything after "note candidates".
//!
//! The neural models (separation, transcription, model activations) run in the
//! Python worker; this crate owns the deterministic part: the shared note
//! format, instrument presets, the playability solvers, notation and MusicXML,
//! verification maths and the edit gate. Every stage can write a trace
//! (see [`trace`]) so a wrong decision can be found by time and note id.

pub mod edit;
pub mod eval;
pub mod ir;
pub mod midi;
pub mod npy;
pub mod play;
pub mod preset;
pub mod trace;
pub mod verify;
