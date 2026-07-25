//! The seams between stages.
//!
//! Speech synthesis is a pipeline — text becomes phonemes, phonemes become
//! token ids, token ids become audio — and each stage is useful on its own.
//! Someone may want this crate's phonemizer in front of a different model, or
//! its Kokoro synthesizer behind a different phonemizer. Each stage is a trait
//! so any one of them can be replaced without touching the others.
//!
//! Every trait here is object-safe: `Box<dyn Phonemizer>` works, so a pipeline
//! can be assembled at runtime from config.

use crate::Outcome;

/// Converts written text into phonemes.
///
/// Implementors decide the phoneme alphabet. The bundled implementation emits
/// Kokoro's native tokens; an implementation targeting another model would
/// emit whatever that model expects, and pair with a matching [`Tokenizer`].
pub trait Phonemizer {
    /// Phonemize `text`, reporting which words needed a fallback.
    fn phonemize(&self, text: &str) -> Outcome;
}

/// Rewrites text before phonemization.
///
/// Split out because it is model-independent: expanding "3:30" to "3 30" is
/// useful regardless of which phonemizer or model follows. Chain several with
/// [`crate::text::Chain`].
pub trait Normalizer {
    /// Return `text` with problem constructs rewritten.
    fn normalize(&self, text: &str) -> String;
}

/// Maps phonemes to the integer ids a model consumes.
///
/// Kept separate from [`Phonemizer`] because the same phoneme string can feed
/// models with different vocabularies — only this mapping changes.
pub trait Tokenizer {
    /// Convert `phonemes` to model input ids.
    ///
    /// Characters outside the vocabulary are skipped; they would be rejected
    /// downstream anyway.
    fn encode(&self, phonemes: &str) -> Vec<i64>;
}

/// Turns token ids into audio samples.
pub trait Synthesizer {
    /// Synthesize `tokens` using `voice`.
    fn synthesize(&self, tokens: &[i64], voice: &Voice) -> Result<Audio, Error>;

    /// Sample rate of the audio this synthesizer produces, in Hz.
    fn sample_rate(&self) -> u32;
}

/// Plays or otherwise consumes finished audio.
///
/// Implement this to write a WAV file, stream over a socket, or feed a
/// different audio backend instead of the bundled `aplay` sink.
pub trait AudioSink {
    /// Consume `audio`, blocking until it has been handled.
    fn play(&self, audio: &Audio) -> Result<(), Error>;
}

/// Mono audio samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    /// Interleaved samples — mono, so simply sequential.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl Audio {
    /// Playback length in seconds.
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Samples as little-endian bytes, the layout most audio tools expect.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }
}

/// A speaker embedding.
///
/// Kokoro voices are `[positions, 256]` tensors: the row is chosen by input
/// length, letting prosody adapt to utterance size.
#[derive(Debug, Clone, PartialEq)]
pub struct Voice {
    /// Style rows, each `style_dim` long.
    pub rows: Vec<Vec<f32>>,
    /// Width of one row.
    pub style_dim: usize,
}

impl Voice {
    /// The row appropriate for an utterance of `token_count` tokens.
    ///
    /// Clamped to the last row, since real utterances can exceed the table.
    pub fn row_for(&self, token_count: usize) -> &[f32] {
        let idx = token_count.min(self.rows.len().saturating_sub(1));
        &self.rows[idx]
    }
}

/// Anything that can go wrong in the pipeline.
#[derive(Debug)]
pub enum Error {
    /// An asset was missing or unreadable.
    Asset(String),
    /// A file was present but malformed.
    Format(String),
    /// Inference failed.
    Inference(String),
    /// Audio output failed.
    Audio(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Asset(m) => write!(f, "asset unavailable: {m}"),
            Error::Format(m) => write!(f, "malformed data: {m}"),
            Error::Inference(m) => write!(f, "synthesis failed: {m}"),
            Error::Audio(m) => write!(f, "audio output failed: {m}"),
        }
    }
}

impl std::error::Error for Error {}
