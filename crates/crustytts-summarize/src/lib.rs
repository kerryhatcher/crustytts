//! Generic text summarization via Ollama.
//!
//! Sends text to a local Ollama model and returns a concise summary. No
//! application-specific logic — feed it any prompt and text.

use serde::Deserialize;

const OLLAMA_URL: &str = "http://localhost:11434/api/generate";

#[derive(Deserialize)]
struct OllamaResponse {
    response: Option<String>,
}

/// Configuration for a summarization request.
#[derive(Debug, Clone)]
pub struct SummarizeRequest<'a> {
    /// The Ollama model to use (e.g. `"qwen3:8b"`).
    pub model: &'a str,
    /// The system prompt / instructions for the model.
    pub prompt: &'a str,
    /// The text to summarize.
    pub text: &'a str,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Request timeout.
    pub timeout_secs: u64,
}

impl<'a> SummarizeRequest<'a> {
    /// Create a request with sensible defaults.
    pub fn new(model: &'a str, prompt: &'a str, text: &'a str) -> Self {
        Self {
            model,
            prompt,
            text,
            max_tokens: 60,
            timeout_secs: 8,
        }
    }

    /// Override the max output tokens.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Override the request timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

/// Send a summarization request to Ollama.
///
/// Returns `Ok(Some(summary))` on success, `Ok(None)` if the model returned
/// empty, and `Err` on network or protocol errors.
pub fn summarize(req: &SummarizeRequest<'_>) -> Result<Option<String>, String> {
    let full_prompt = format!("{}\n\n{}", req.prompt, req.text);

    let resp = reqwest::blocking::Client::new()
        .post(OLLAMA_URL)
        .timeout(std::time::Duration::from_secs(req.timeout_secs))
        .json(&serde_json::json!({
            "model": req.model,
            "prompt": full_prompt,
            "stream": false,
            "think": false,
            "options": {"num_predict": req.max_tokens},
        }))
        .send()
        .map_err(|e| format!("Ollama request failed: {e}"))?;

    let body: OllamaResponse = resp
        .json()
        .map_err(|e| format!("Ollama response parse failed: {e}"))?;

    Ok(body
        .response
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty()))
}
