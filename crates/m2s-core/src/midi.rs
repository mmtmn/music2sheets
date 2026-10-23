//! MIDI writing for resynthesis (FluidSynth renders it; the model front end then compares
//! the render with the stem). Fixed 120 qpm at 480 ticks per quarter: 960 ticks per second.

use anyhow::Result;
use midly::num::{u15, u28, u4, u7};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

use crate::ir::Note;

const TPS: f64 = 960.0;

/// One instrument's notes (and sustain pedal) in the window [t0, t1], shifted so t0 is time 0.
pub fn write(
    notes: &[&Note],
    program: u8,
    drums: bool,
    pedal: &[[f64; 2]],
    t0: f64,
    t1: Option<f64>,
) -> Result<Vec<u8>> {
    let ch: u8 = if drums { 9 } else { 0 };
    // (tick, order, kind): order puts note-offs before note-ons at the same tick
    let mut ev: Vec<(u32, u8, MidiMessage)> = vec![];
    let tick = |s: f64| ((s - t0).max(0.0) * TPS).round() as u32;
    for n in notes {
        if let Some(t1) = t1 {
            if n.offset < t0 || n.onset > t1 {
                continue;
            }
        }
        let p = u7::new(n.pitch.clamp(0, 127) as u8);
        let v = u7::new(n.velocity.clamp(1, 127) as u8);
        let (a, b) = (tick(n.onset), tick(n.offset.max(n.onset + 0.01)).max(tick(n.onset) + 1));
        ev.push((a, 2, MidiMessage::NoteOn { key: p, vel: v }));
        ev.push((
            b,
            0,
            MidiMessage::NoteOff {
                key: p,
                vel: u7::new(0),
            },
        ));
    }
    for &[a, b] in pedal {
        if let Some(t1) = t1 {
            if b < t0 || a > t1 {
                continue;
            }
        }
        ev.push((
            tick(a),
            1,
            MidiMessage::Controller {
                controller: u7::new(64),
                value: u7::new(127),
            },
        ));
        ev.push((
            tick(b),
            1,
            MidiMessage::Controller {
                controller: u7::new(64),
                value: u7::new(0),
            },
        ));
    }
    ev.sort_by_key(|e| (e.0, e.1));
    let mut track: Vec<TrackEvent> = vec![TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(500_000.into())),
    }];
    if !drums {
        track.push(TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Midi {
                channel: u4::new(ch),
                message: MidiMessage::ProgramChange {
                    program: u7::new(program),
                },
            },
        });
    }
    let mut last = 0u32;
    for (t, _, m) in ev {
        track.push(TrackEvent {
            delta: u28::new(t - last),
            kind: TrackEventKind::Midi {
                channel: u4::new(ch),
                message: m,
            },
        });
        last = t;
    }
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    let smf = Smf {
        header: Header::new(Format::SingleTrack, Timing::Metrical(u15::new(480))),
        tracks: vec![track],
    };
    let mut out = Vec::new();
    smf.write(&mut out).map_err(|e| anyhow::anyhow!("midi: {e}"))?;
    Ok(out)
}
