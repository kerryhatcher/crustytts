//! Audio output sinks.
//!
//! [`AplaySink`] plays through ALSA's `aplay` — works on any Linux box without
//! linking an audio library. [`CaptureSink`] collects audio in memory for
//! tests or file output.

use std::io::Write;
use std::process::{Command, Stdio};

use crustytts_core::{Audio, AudioSink, Error};

/// Plays audio through ALSA's `aplay`.
#[derive(Debug, Clone)]
pub struct AplaySink {
    binary: String,
}

impl Default for AplaySink {
    fn default() -> Self {
        Self {
            binary: "aplay".into(),
        }
    }
}

impl AplaySink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Use a different binary — a wrapper script, or an absolute path.
    pub fn with_binary(binary: impl Into<String>) -> Self {
        Self {
            binary: binary.into(),
        }
    }
}

impl AudioSink for AplaySink {
    fn play(&self, audio: &Audio) -> Result<(), Error> {
        if audio.samples.is_empty() {
            return Ok(());
        }

        let mut child = Command::new(&self.binary)
            .args([
                "-r",
                &audio.sample_rate.to_string(),
                "-f",
                "FLOAT_LE",
                "-t",
                "raw",
                "-c",
                "1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::Audio(format!("cannot start {}: {e}", self.binary)))?;

        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| Error::Audio("no stdin on audio process".into()))?;
            stdin
                .write_all(&audio.to_le_bytes())
                .map_err(|e| Error::Audio(format!("writing samples: {e}")))?;
        }

        let status = child
            .wait()
            .map_err(|e| Error::Audio(format!("waiting for playback: {e}")))?;

        if !status.success() {
            return Err(Error::Audio(format!("{} exited {status}", self.binary)));
        }

        Ok(())
    }
}

/// Collects audio instead of playing it — for tests, or to write a file.
#[derive(Debug, Default)]
pub struct CaptureSink {
    captured: std::sync::Mutex<Vec<Audio>>,
}

impl CaptureSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything played so far, in order.
    pub fn captured(&self) -> Vec<Audio> {
        self.captured.lock().expect("capture lock").clone()
    }
}

impl AudioSink for CaptureSink {
    fn play(&self, audio: &Audio) -> Result<(), Error> {
        self.captured
            .lock()
            .expect("capture lock")
            .push(audio.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_sink_records_what_it_is_given() {
        let sink = CaptureSink::new();
        let audio = Audio {
            samples: vec![0.1, -0.2],
            sample_rate: 24_000,
        };
        sink.play(&audio).expect("capture never fails");
        assert_eq!(sink.captured(), vec![audio]);
    }

    #[test]
    fn silence_is_not_sent_to_the_player() {
        let audio = Audio {
            samples: vec![],
            sample_rate: 24_000,
        };
        assert!(AplaySink::with_binary("/nonexistent").play(&audio).is_ok());
    }
}
