//! Shared types and traits for the crustytts TTS pipeline.
//!
//! Every stage — normalization, phonemization, tokenization, synthesis, audio
//! output — is a trait, so any one of them can be replaced without touching
//! the others. All traits are object-safe: `Box<dyn Phonemizer>` works, so a
//! pipeline can be assembled at runtime from config.

use std::fmt;

// ── Outcome ─────────────────────────────────────────────────────────────────────

/// Result of phonemizing a string.
///
/// `spelled_out` names the words that fell through to the letter-spelling net.
/// A non-empty list means the audio will sound clumsy but remain truthful; it
/// is also the signal to consider a real letter-to-sound engine.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    /// IPA phonemes in Kokoro's native token alphabet, ready to synthesize.
    pub phonemes: String,
    /// Words no dictionary entry covered, in order of appearance.
    pub spelled_out: Vec<String>,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.phonemes)
    }
}

// ── traits ──────────────────────────────────────────────────────────────────────

/// Converts written text into phonemes.
///
/// Implementors decide the phoneme alphabet. The bundled implementation emits
/// Kokoro's native tokens; an implementation targeting another model would
/// emit whatever that model expects, and pair with a matching [`Tokenizer`].
pub trait Phonemizer {
    /// Phonemize `text`, reporting which words needed a fallback.
    fn phonemize(&self, text: &str) -> Outcome;
}

/// Attempts to phonemize words the dictionary doesn't know.
///
/// When a [`Phonemizer`] encounters a word with no dictionary entry, it
/// falls back to spelling it out letter by letter. An `OovPhonemizer` is
/// given a chance to produce real phonemes first — via an LLM, a rule-based
/// L2S engine, or any other out-of-vocabulary strategy.
///
/// Return `Some(phonemes)` in Kokoro's IPA alphabet, or `None` to let the
/// caller fall through to letter spelling.
pub trait OovPhonemizer: Send + Sync {
    /// Try to produce Kokoro IPA phonemes for `word` (lowercase).
    fn phonemize_oov(&self, word: &str) -> Option<String>;
}

/// Rewrites text before phonemization.
///
/// Split out because it is model-independent: expanding "3:30" to "3 30" is
/// useful regardless of which phonemizer or model follows.
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

/// A single stage in a proofreading pipeline.
///
/// Each stage takes text and returns text. Stages are deliberately simple —
/// they don't know about each other — so they can be composed, reordered, or
/// replaced without touching the rest of the pipeline.
///
/// # Example
///
/// ```rust
/// use crustytts_core::ProofingStage;
///
/// struct Capitalizer;
/// impl ProofingStage for Capitalizer {
///     fn proof(&self, text: &str) -> String {
///         // ...
///         # text.to_string()
///     }
/// }
/// ```
pub trait ProofingStage: Send + Sync {
    /// Process `text` and return the corrected version.
    fn proof(&self, text: &str) -> String;
}

/// A pipeline of [`ProofingStage`]s applied in order.
///
/// Each stage's output becomes the next stage's input.
pub struct ProofingPipeline {
    stages: Vec<Box<dyn ProofingStage>>,
}

impl ProofingPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage to the pipeline.
    pub fn push(mut self, stage: impl ProofingStage + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Run all stages in order.
    pub fn run(&self, text: &str) -> String {
        self.stages
            .iter()
            .fold(text.to_string(), |acc, stage| stage.proof(&acc))
    }
}

impl Default for ProofingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── data types ──────────────────────────────────────────────────────────────────

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
