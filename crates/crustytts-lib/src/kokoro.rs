//! Kokoro-82M inference via ONNX Runtime.
//!
//! Feature-gated behind `kokoro-onnx` so that using this crate purely as a
//! phonemizer costs nothing — ONNX Runtime is a heavy dependency and most
//! callers of [`phonemize`](crate::phonemize) do not need it.

use std::path::Path;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use crate::traits::{Audio, Error, Synthesizer, Voice};

/// Kokoro-82M outputs at a fixed 24 kHz.
pub const SAMPLE_RATE: u32 = 24_000;

/// Kokoro-82M running on ONNX Runtime.
///
/// Load once and reuse — construction parses an 80 MB graph, whereas each
/// [`synthesize`](Synthesizer::synthesize) call is fast.
pub struct KokoroOnnx {
    /// `ort` requires `&mut` to run a session, but [`Synthesizer`] takes
    /// `&self` so a loaded model can be shared. The mutex reconciles the two;
    /// contention is irrelevant since inference is serial anyway.
    session: std::sync::Mutex<Session>,
    speed: f32,
}

impl KokoroOnnx {
    /// Load a model from an `.onnx` file.
    ///
    /// Use the `model_q8f16.onnx` (or full-precision) file from the
    /// `onnx-community/Kokoro-82M-v1.0-ONNX` repository.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = model_path.as_ref();

        let session = Session::builder()
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| Error::Asset(format!("cannot load model {}: {e}", path.display())))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            speed: 1.0,
        })
    }

    /// Set the speaking rate. 1.0 is natural; higher is faster.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.clamp(0.1, 5.0);
        self
    }
}

impl Synthesizer for KokoroOnnx {
    fn synthesize(&self, tokens: &[i64], voice: &Voice) -> Result<Audio, Error> {
        if tokens.is_empty() {
            return Ok(Audio {
                samples: Vec::new(),
                sample_rate: SAMPLE_RATE,
            });
        }

        // The style row is selected by input length, so prosody adapts to how
        // much is being said.
        let style = voice.row_for(tokens.len());

        let input_ids = Tensor::from_array(([1, tokens.len()], tokens.to_vec()))
            .map_err(|e| Error::Inference(format!("input_ids tensor: {e}")))?;
        let style_tensor = Tensor::from_array(([1, style.len()], style.to_vec()))
            .map_err(|e| Error::Inference(format!("style tensor: {e}")))?;
        let speed = Tensor::from_array(([1], vec![self.speed]))
            .map_err(|e| Error::Inference(format!("speed tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Inference("model lock poisoned".into()))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "style" => style_tensor,
                "speed" => speed,
            ])
            .map_err(|e| Error::Inference(e.to_string()))?;

        let (_, samples) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(format!("reading audio output: {e}")))?;

        Ok(Audio {
            samples: samples.to_vec(),
            sample_rate: SAMPLE_RATE,
        })
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}
