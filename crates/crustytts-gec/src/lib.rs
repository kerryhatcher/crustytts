//! T5-small ONNX grammar error correction.
//!
//! Provides a pluggable grammar error correction stage backed by a T5-small
//! model exported to ONNX. When the `onnx` feature is enabled, the corrector
//! loads the model and tokenizer and runs fully offline inference via ONNX
//! Runtime.
//!
//! Without the `onnx` feature, a no-op corrector is provided that passes
//! text through unchanged — useful for development or when the model isn't
//! available.
//!
//! # Model acquisition
//!
//! Download a pre-converted ONNX model from HuggingFace:
//!
//! ```bash
//! # T5-small (~240MB) — recommended
//! huggingface-cli download TeXlyre/grammar-t5-small-onnx \
//!   --local-dir ./models/grammar-t5-small-onnx
//!
//! # T5-base (~880MB) — higher quality, slower
//! huggingface-cli download onnx-community/t5-base-grammar-correction-ONNX \
//!   --local-dir ./models/t5-base-grammar-correction-ONNX
//! ```
//!
//! Set `CRUSTYTTS_GEC_MODEL` to the directory containing `model.onnx` and
//! `tokenizer.json`.
//!
//! # Example
//!
//! ```rust,ignore
//! use crustytts_gec::T5OnnxCorrector;
//! use crustytts_core::ProofingStage;
//!
//! let gec = T5OnnxCorrector::load("models/grammar-t5-small-onnx").unwrap();
//! let corrected = gec.proof("She go to the store every day.");
//! assert_eq!(corrected, "She goes to the store every day.");
//! ```

use crustytts_core::ProofingStage;

// ── GEC corrector trait ────────────────────────────────────────────────────────

/// A grammar error corrector.
///
/// Separate from [`ProofingStage`] so correctors can be used standalone or
/// wrapped in a pipeline.
pub trait GecCorrector: Send + Sync {
    /// Correct grammar, spelling, and punctuation in `text`.
    fn correct(&self, text: &str) -> Result<String, GecError>;
}

/// Errors that can occur during GEC.
#[derive(Debug)]
pub enum GecError {
    /// The model file was not found.
    ModelNotFound(String),
    /// The model failed to load.
    ModelLoad(String),
    /// Tokenization failed.
    Tokenize(String),
    /// Inference failed.
    Inference(String),
    /// Detokenization failed.
    Detokenize(String),
}

impl std::fmt::Display for GecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GecError::ModelNotFound(m) => write!(f, "GEC model not found: {m}"),
            GecError::ModelLoad(m) => write!(f, "GEC model load failed: {m}"),
            GecError::Tokenize(m) => write!(f, "GEC tokenization failed: {m}"),
            GecError::Inference(m) => write!(f, "GEC inference failed: {m}"),
            GecError::Detokenize(m) => write!(f, "GEC detokenization failed: {m}"),
        }
    }
}

impl std::error::Error for GecError {}

// ── T5 ONNX corrector ──────────────────────────────────────────────────────────

#[cfg(feature = "onnx")]
mod onnx_impl {
    use super::*;
    use ort::{inputs, Environment, GraphOptimizationLevel, Session, SessionBuilder};
    use std::sync::OnceLock;
    use tokenizers::Tokenizer;

    /// A grammar error corrector backed by a T5-small ONNX model.
    ///
    /// Loads the model once and reuses it for all corrections. Thread-safe.
    pub struct T5OnnxCorrector {
        session: Session,
        tokenizer: Tokenizer,
        max_input_length: usize,
        max_output_length: usize,
    }

    impl T5OnnxCorrector {
        /// Load the model from `model_dir`.
        ///
        /// `model_dir` must contain `model.onnx` and `tokenizer.json`.
        pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, GecError> {
            let model_dir = model_dir.as_ref();
            let model_path = model_dir.join("model.onnx");
            let tokenizer_path = model_dir.join("tokenizer.json");

            if !model_path.exists() {
                return Err(GecError::ModelNotFound(model_path.display().to_string()));
            }
            if !tokenizer_path.exists() {
                return Err(GecError::ModelNotFound(
                    tokenizer_path.display().to_string(),
                ));
            }

            // ONNX Runtime environment (one-time init)
            static ENV: OnceLock<Environment> = OnceLock::new();
            let env = ENV.get_or_init(|| {
                Environment::builder()
                    .with_name("crustytts-gec")
                    .build()
                    .expect("failed to create ONNX environment")
            });

            let session = SessionBuilder::new(env)
                .map_err(|e| GecError::ModelLoad(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| GecError::ModelLoad(e.to_string()))?
                .with_intra_threads(1)
                .map_err(|e| GecError::ModelLoad(e.to_string()))?
                .commit_from_file(&model_path)
                .map_err(|e| GecError::ModelLoad(e.to_string()))?;

            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| GecError::ModelLoad(format!("tokenizer: {e}")))?;

            Ok(Self {
                session,
                tokenizer,
                max_input_length: 128,
                max_output_length: 128,
            })
        }

        /// Set the maximum input token length (default: 128).
        pub fn with_max_input_length(mut self, len: usize) -> Self {
            self.max_input_length = len;
            self
        }

        /// Set the maximum output token length (default: 128).
        pub fn with_max_output_length(mut self, len: usize) -> Self {
            self.max_output_length = len;
            self
        }

        /// Run the T5 model to correct `text`.
        fn run_model(&self, text: &str) -> Result<String, GecError> {
            // T5 expects a task prefix for GEC
            let input_text = format!("grammar: {text}");

            // Tokenize
            let encoding = self
                .tokenizer
                .encode(input_text, true)
                .map_err(|e| GecError::Tokenize(e.to_string()))?;

            let input_ids: Vec<i64> = encoding
                .get_ids()
                .iter()
                .take(self.max_input_length)
                .map(|&id| id as i64)
                .collect();

            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .take(self.max_input_length)
                .map(|&m| m as i64)
                .collect();

            // Decoder start token (T5 uses 0 as pad/start)
            let decoder_input_ids: Vec<i64> = vec![0];

            // Reshape to [1, seq_len]
            let input_len = input_ids.len();
            let mask_len = attention_mask.len();

            let input_tensor = ort::Tensor::from_shape_vec((1, input_len), input_ids)
                .map_err(|e| GecError::Inference(e.to_string()))?;

            let mask_tensor = ort::Tensor::from_shape_vec((1, mask_len), attention_mask)
                .map_err(|e| GecError::Inference(e.to_string()))?;

            let decoder_tensor = ort::Tensor::from_shape_vec((1, 1), decoder_input_ids)
                .map_err(|e| GecError::Inference(e.to_string()))?;

            // Run inference
            let outputs = self
                .session
                .run(
                    inputs![
                        "input_ids" => input_tensor,
                        "attention_mask" => mask_tensor,
                        "decoder_input_ids" => decoder_tensor,
                    ]
                    .map_err(|e| GecError::Inference(e.to_string()))?,
                )
                .map_err(|e| GecError::Inference(e.to_string()))?;

            // Extract output token ids
            let output_name = outputs
                .iter()
                .next()
                .map(|o| o.0.to_string())
                .ok_or_else(|| GecError::Inference("no output from model".into()))?;

            let output_tensor = outputs[&*output_name]
                .try_extract_tensor::<i64>()
                .map_err(|e| GecError::Inference(e.to_string()))?;

            let output_ids: Vec<i64> = output_tensor.iter().copied().collect();

            // Decode
            let output_text = self
                .tokenizer
                .decode(&output_ids, true)
                .map_err(|e| GecError::Detokenize(e.to_string()))?;

            // Clean up: remove the task prefix if echoed, strip special tokens
            let cleaned = output_text
                .replace("<pad>", "")
                .replace("</s>", "")
                .replace("<unk>", "")
                .trim()
                .to_string();

            Ok(cleaned)
        }
    }

    impl GecCorrector for T5OnnxCorrector {
        fn correct(&self, text: &str) -> Result<String, GecError> {
            self.run_model(text)
        }
    }

    impl ProofingStage for T5OnnxCorrector {
        fn proof(&self, text: &str) -> String {
            match self.correct(text) {
                Ok(corrected) if !corrected.is_empty() => corrected,
                _ => text.to_string(),
            }
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_impl::T5OnnxCorrector;

// ── no-op corrector (when ONNX feature is disabled) ────────────────────────────

/// A no-op corrector that passes text through unchanged.
///
/// Used when the `onnx` feature is disabled. This lets code compile and run
/// without the ONNX model, falling through to other stages in the pipeline.
pub struct NoOpCorrector;

impl GecCorrector for NoOpCorrector {
    fn correct(&self, text: &str) -> Result<String, GecError> {
        Ok(text.to_string())
    }
}

impl ProofingStage for NoOpCorrector {
    fn proof(&self, text: &str) -> String {
        text.to_string()
    }
}

// ── tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_corrector_passes_through() {
        let gec = NoOpCorrector;
        assert_eq!(gec.proof("anything"), "anything");
        assert_eq!(gec.correct("unchanged").unwrap(), "unchanged");
    }
}
