//! English grapheme-to-phoneme for Kokoro TTS that never silently drops a word.
//!
//! Wraps `voice-g2p` (Misaki's 90k-entry dictionary + POS tagger) with:
//!
//! 1. **Normalization** — rewrites clock times and month abbreviations via
//!    [`crustytts_normalize`].
//! 2. **A safety net** — any word that still phonemizes to nothing is spelled
//!    out letter by letter.
//! 3. **Optional OOV phonemizer** — before falling back to letter spelling,
//!    an [`OovPhonemizer`] (e.g. an LLM) can attempt real phonemes. Enable
//!    the `oov-llm` feature for the bundled [`OllamaOovPhonemizer`].
//!
//! # Example
//!
//! ```rust
//! let out = crustytts_phonemize::phonemize("Claude deployed to kubernetes at 1:00");
//! assert!(!out.phonemes.is_empty());
//! assert_eq!(out.spelled_out, vec!["kubernetes"]);
//! ```

use crustytts_core::{OovPhonemizer, Outcome, Phonemizer};

#[cfg(feature = "oov-llm")]
use std::collections::HashMap;
#[cfg(feature = "oov-llm")]
use std::sync::Mutex;

/// Phonemize English text for Kokoro, guaranteeing no word is silently lost.
///
/// This is a convenience wrapper for [`phonemize_with_oov`] with no OOV handler.
/// Unknown words are spelled out letter by letter.
pub fn phonemize(text: &str) -> Outcome {
    phonemize_with_oov(text, None)
}

/// Phonemize English text with an optional out-of-vocabulary handler.
///
/// When `oov` is `Some`, each word the dictionary doesn't know is first sent
/// to the handler. If the handler returns phonemes, they are stitched directly
/// into the output (bypassing the G2P engine). Otherwise the word falls through
/// to letter-by-letter spelling.
pub fn phonemize_with_oov(text: &str, oov: Option<&dyn OovPhonemizer>) -> Outcome {
    let normalized = crustytts_normalize::normalize(text);

    // Pass 1: build the prepared text for the G2P engine, collecting OOV
    // phonemes separately so they aren't fed back through the G2P.
    let mut prepared = String::with_capacity(normalized.len());
    let mut spelled_out = Vec::new();
    // (token_index, oov_phonemes) for tokens the OOV handler resolved
    let mut oov_replacements: Vec<(usize, String)> = Vec::new();
    let mut token_idx = 0;

    for token in normalized.split_whitespace() {
        if !prepared.is_empty() {
            prepared.push(' ');
        }

        let (word, leading, trailing) = split_affixes(token);
        if word.is_empty() || phonemize_word(word).is_some() {
            prepared.push_str(token);
            token_idx += 1;
            continue;
        }

        // Try the OOV handler before falling back to letter spelling
        let mut handled = false;
        if let Some(handler) = oov {
            if let Some(phonemes) = handler.phonemize_oov(word) {
                let trimmed = phonemes.trim().to_string();
                if !trimmed.is_empty() {
                    // Insert one known placeholder word so the G2P output keeps
                    // the same token count. Spelling the OOV word into multiple
                    // letter tokens leaves trailing letters behind when the
                    // single original token is replaced in pass 3.
                    oov_replacements.push((token_idx, format!("{leading}{trimmed}{trailing}")));
                    prepared.push_str(leading);
                    prepared.push_str("test");
                    prepared.push_str(trailing);
                    handled = true;
                }
            }
        }

        if !handled {
            spelled_out.push(word.to_string());
            prepared.push_str(leading);
            prepared.push_str(&spaced_letters(word));
            prepared.push_str(trailing);
        }

        token_idx += 1;
    }

    // Pass 2: run the G2P engine on the prepared text (with spelled-out placeholders)
    let raw_phonemes = voice_g2p::english_to_phonemes(&prepared)
        .map(|p| p.trim().to_string())
        .unwrap_or_default();

    // Pass 3: stitch OOV phonemes into the G2P output, replacing the
    // spelled-out-letter phonemes for those tokens.
    let phonemes = if oov_replacements.is_empty() {
        raw_phonemes
    } else {
        stitch_oov_phonemes(&raw_phonemes, &oov_replacements)
    };

    Outcome {
        phonemes,
        spelled_out,
    }
}

/// Replace spelled-out-letter phoneme sequences with OOV phonemes.
///
/// The G2P engine turns "K U B E R N E T E S" into something like
/// "kˈA jˈu bˈi ...". We find each token boundary (space) and swap in
/// the OOV phonemes for the tokens at the recorded indices.
fn stitch_oov_phonemes(raw: &str, replacements: &[(usize, String)]) -> String {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    let mut result = Vec::with_capacity(tokens.len());

    for (i, token) in tokens.iter().enumerate() {
        if let Some((_, oov_phonemes)) = replacements.iter().find(|(idx, _)| *idx == i) {
            result.push(oov_phonemes.as_str());
        } else {
            result.push(token);
        }
    }

    result.join(" ")
}

/// Convenience wrapper returning just the phoneme string.
pub fn phonemize_str(text: &str) -> String {
    phonemize(text).phonemes
}

/// The bundled phonemizer: Misaki dictionary, POS tagging, letter-spelling net.
#[derive(Debug, Clone, Copy, Default)]
pub struct MisakiPhonemizer;

impl Phonemizer for MisakiPhonemizer {
    fn phonemize(&self, text: &str) -> Outcome {
        phonemize(text)
    }
}

// ── Ollama OOV phonemizer ───────────────────────────────────────────────────────

/// An [`OovPhonemizer`] that asks a local Ollama model for Kokoro IPA phonemes.
///
/// Caches results in memory so repeated words don't re-hit the LLM.
///
/// Requires the `oov-llm` feature.
///
/// ```rust
/// use crustytts_phonemize::{OllamaOovPhonemizer, phonemize_with_oov};
///
/// let oov = OllamaOovPhonemizer::new("qwen3:0.6b");
/// let out = phonemize_with_oov("deployed to kubernetes", Some(&oov));
/// ```
#[cfg(feature = "oov-llm")]
pub struct OllamaOovPhonemizer {
    model: String,
    endpoint: String,
    timeout_secs: u64,
    cache: Mutex<HashMap<String, Option<String>>>,
}

#[cfg(feature = "oov-llm")]
impl OllamaOovPhonemizer {
    /// Create a new OOV phonemizer using the given Ollama `model`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            endpoint: "http://localhost:11434/api/generate".into(),
            timeout_secs: 5,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Override the Ollama API endpoint.
    pub fn with_endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = url.into();
        self
    }

    /// Override the request timeout in seconds (default: 5).
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

#[cfg(feature = "oov-llm")]
impl OovPhonemizer for OllamaOovPhonemizer {
    fn phonemize_oov(&self, word: &str) -> Option<String> {
        // Check cache first
        {
            let cache = self.cache.lock().ok()?;
            if let Some(cached) = cache.get(word) {
                return cached.clone();
            }
        }

        let prompt = format!(
            "Convert this English word to Kokoro IPA phonemes.\n\n\
             Kokoro uses a compact IPA alphabet. Here are examples:\n\
             - deployed -> dəplˈɔɪd\n\
             - kubernetes -> kjˈubɚnɛtɪs\n\
             - application -> æpləkˈAʃən\n\
             - Claude -> klˈɔd\n\
             - nginx -> ˈɛnʤənˈɛks\n\n\
             Rules:\n\
             - Use ONLY lowercase letters and these IPA symbols: ˈ ɔ ɪ ɛ ɚ ɹ ʃ ʒ θ ð ŋ ʤ ʧ æ ɑ ʌ ə ʊ i u o e a b d f g h j k l m n p r s t v w z Y\n\
             - Begin multi-syllable words with stress marker ˈ on the primary stressed syllable.\n\
             - The symbol Y represents the 'eye' diphthong (as in 'my', 'like').\n\
             - The symbol A represents the 'ay' diphthong (as in 'day', 'make').\n\
             - The symbol O represents the 'oh' diphthong (as in 'go', 'no').\n\
             - The symbol W represents the 'ow' diphthong (as in 'now', 'how').\n\
             - The symbol T represents the 'oy' diphthong (as in 'boy', 'toy').\n\
             - The symbol I represents the 'ee' vowel (as in 'see', 'be').\n\
             - The symbol U represents the 'oo' vowel (as in 'blue', 'true').\n\n\
             Word: {word}\n\n\
             Respond with ONLY the phoneme string, nothing else."
        );

        let result = (|| -> Result<Option<String>, String> {
            let resp = reqwest::blocking::Client::new()
                .post(&self.endpoint)
                .timeout(std::time::Duration::from_secs(self.timeout_secs))
                .json(&serde_json::json!({
                    "model": &self.model,
                    "prompt": prompt,
                    "stream": false,
                    "think": false,
                    "options": {"num_predict": 40},
                }))
                .send()
                .map_err(|e| format!("Ollama request failed: {e}"))?;

            #[derive(serde::Deserialize)]
            struct OllamaResponse {
                response: Option<String>,
            }

            let body: OllamaResponse = resp
                .json()
                .map_err(|e| format!("Ollama response parse failed: {e}"))?;

            let text = body.response.unwrap_or_default().trim().to_string();

            Ok(if text.is_empty() { None } else { Some(text) })
        })();

        let phonemes = match result {
            Ok(Some(p)) => {
                // Basic validation: must contain at least some phoneme characters
                if p.chars().any(|c| c.is_alphabetic()) {
                    Some(p)
                } else {
                    None
                }
            }
            _ => None,
        };

        // Cache the result
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(word.to_string(), phonemes.clone());
        }

        phonemes
    }
}

// ── CustomMapping (deterministic OOV handler) ────────────────────────────────────

/// A deterministic [`OovPhonemizer`] backed by a custom word-to-phoneme map.
///
/// Use this to teach the TTS how to pronounce words the Misaki dictionary
/// doesn't know — as an alternative to the LLM-based [`OllamaOovPhonemizer`].
///
/// # Example
///
/// ```rust
/// use crustytts_phonemize::{CustomMapping, phonemize_with_oov};
///
/// let mut mapping = CustomMapping::new();
/// mapping.insert("kubernetes", "kjˈubɚnɛtɪs");
/// mapping.insert("nginx", "ˈɛnʤənˈɛks");
///
/// let out = phonemize_with_oov("deployed to kubernetes", Some(&mapping));
/// assert!(out.spelled_out.is_empty());
/// assert!(out.phonemes.contains("kjˈu"));
/// ```
///
/// Mappings are case-insensitive: looking up "Kubernetes" or "KUBERNETES"
/// will match an entry for "kubernetes".
#[derive(Debug, Clone, Default)]
pub struct CustomMapping {
    map: std::collections::HashMap<String, String>,
}

impl CustomMapping {
    /// Create an empty custom mapping.
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
        }
    }

    /// Create a custom mapping from an existing word-to-phoneme map.
    ///
    /// Keys should be lowercase words; lookups are case-insensitive.
    pub fn from_map(map: std::collections::HashMap<String, String>) -> Self {
        Self { map }
    }

    /// Add a word-to-phoneme mapping.
    ///
    /// `word` is lowercased internally, so lookups are case-insensitive.
    pub fn insert(&mut self, word: &str, phonemes: &str) {
        self.map.insert(word.to_lowercase(), phonemes.to_string());
    }

    /// Extend from an iterator of (word, phonemes) pairs.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>) {
        for (word, phonemes) in iter {
            self.insert(word.as_ref(), phonemes.as_ref());
        }
    }

    /// Builder-style: add a mapping and return `self`.
    pub fn with_mapping(mut self, word: &str, phonemes: &str) -> Self {
        self.insert(word, phonemes);
        self
    }

    /// Load mappings from a JSON file where keys are words and values are phoneme strings.
    ///
    /// Supports two formats:
    ///
    /// **Flat** (simple word → phoneme object):
    /// ```json
    /// {
    ///   "kubernetes": "kjˈubɚnɛtɪs",
    ///   "nginx": "ˈɛnʤənˈɛks"
    /// }
    /// ```
    ///
    /// **Wrapped** (with metadata):
    /// ```json
    /// {
    ///   "version": "1.0",
    ///   "description": "...",
    ///   "mappings": {
    ///     "kubernetes": "kjˈubɚnɛtɪs",
    ///     "nginx": "ˈɛnʤənˈɛks"
    ///   }
    /// }
    /// ```
    ///
    /// Requires the `json-import` feature.
    #[cfg(feature = "json-import")]
    pub fn from_json_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);
        let raw: serde_json::Value = serde_json::from_reader(reader)?;

        let map: std::collections::HashMap<String, String> = match raw {
            // Wrapped format: { "version": ..., "mappings": { ... } }
            serde_json::Value::Object(ref obj) if obj.contains_key("mappings") => {
                let mappings =
                    obj.get("mappings")
                        .and_then(|m| m.as_object())
                        .ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "'mappings' field must be an object",
                            )
                        })?;
                mappings
                    .iter()
                    .map(|(k, v)| {
                        let val = v.as_str().unwrap_or_default().to_string();
                        (k.to_lowercase(), val)
                    })
                    .collect()
            }
            // Flat format: { "word": "phonemes", ... }
            _ => {
                let parsed: std::collections::HashMap<String, String> = serde_json::from_value(raw)
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("invalid phoneme mapping JSON: {e}"),
                        )
                    })?;
                parsed
                    .into_iter()
                    .map(|(k, v)| (k.to_lowercase(), v))
                    .collect()
            }
        };

        Ok(Self { map })
    }

    /// The number of entries in this mapping.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl OovPhonemizer for CustomMapping {
    fn phonemize_oov(&self, word: &str) -> Option<String> {
        self.map.get(&word.to_lowercase()).cloned()
    }
}

// ── internals ───────────────────────────────────────────────────────────────────

/// Phonemize one word, returning `None` if the dictionary had nothing for it.
fn phonemize_word(word: &str) -> Option<String> {
    let phonemes = voice_g2p::english_to_phonemes(word).ok()?;
    let phonemes = phonemes.trim();
    (!phonemes.is_empty()).then(|| phonemes.to_string())
}

/// Rewrite a word as space-separated capital letters: "nginx" -> "N G I N X".
fn spaced_letters(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_uppercase().to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split leading/trailing punctuation off a token.
fn split_affixes(token: &str) -> (&str, &str, &str) {
    let start = token
        .find(|c: char| c.is_alphanumeric())
        .unwrap_or(token.len());
    let end = token
        .rfind(|c: char| c.is_alphanumeric())
        .map_or(start, |i| {
            i + token[i..].chars().next().map_or(1, char::len_utf8)
        });

    (&token[start..end], &token[..start], &token[end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oov_words_are_spelled_not_dropped() {
        let out = phonemize("Claude deployed to kubernetes");
        assert_eq!(out.spelled_out, vec!["kubernetes"]);
        assert!(out.phonemes.contains("kˈA"), "got: {}", out.phonemes);
        assert!(out.phonemes.contains("jˈu"), "got: {}", out.phonemes);
    }

    #[test]
    fn known_words_are_untouched_by_the_safety_net() {
        let out = phonemize("Claude finished the task");
        assert!(out.spelled_out.is_empty());
        assert_eq!(out.phonemes, "klˈɔd fˈɪnəʃt ðə tˈæsk");
    }

    #[test]
    fn reports_every_oov_word() {
        let out = phonemize("restarted nginx and tokio");
        assert_eq!(out.spelled_out, vec!["nginx", "tokio"]);
    }

    #[test]
    fn preserves_pos_disambiguation() {
        assert!(
            phonemize("I read the book yesterday")
                .phonemes
                .contains("ɹˈɛd"),
            "past tense should be ɹˈɛd"
        );
        assert!(
            phonemize("I will read the book tomorrow")
                .phonemes
                .contains("ɹˈid"),
            "future should be ɹˈid"
        );
    }

    #[test]
    fn preserves_pos_disambiguation_alongside_oov() {
        let out = phonemize("I read the kubernetes docs yesterday");
        assert_eq!(out.spelled_out, vec!["kubernetes"]);
        assert!(out.phonemes.contains("ɹˈɛd"), "got: {}", out.phonemes);
    }

    #[test]
    fn preserves_punctuation() {
        let out = phonemize("Claude finished, then stopped.");
        assert!(out.phonemes.contains(','), "got: {}", out.phonemes);
        assert!(out.phonemes.ends_with('.'), "got: {}", out.phonemes);
    }

    #[test]
    fn handles_empty_and_punctuation_only_input() {
        assert_eq!(phonemize("").phonemes, "");
        assert!(phonemize("...").spelled_out.is_empty());
    }

    // ── CustomMapping tests ──────────────────────────────────────────────────

    #[test]
    fn custom_mapping_covers_oov_word() {
        let mut mapping = CustomMapping::new();
        mapping.insert("kubernetes", "kjˈubɚnɛtɪs");
        let out = phonemize_with_oov("deployed to kubernetes", Some(&mapping));
        assert!(
            out.spelled_out.is_empty(),
            "kubernetes should not be spelled out: {:?}",
            out.spelled_out
        );
        assert!(
            out.phonemes.contains("kjˈu"),
            "expected custom phonemes for kubernetes, got: {}",
            out.phonemes
        );
        assert_eq!(out.phonemes, "dəplˈYd tə kjˈubɚnɛtɪs");
    }

    #[test]
    fn custom_mapping_case_insensitive() {
        let mut mapping = CustomMapping::new();
        mapping.insert("nginx", "ˈɛnʤənˈɛks");
        let out = phonemize_with_oov("restarted Nginx", Some(&mapping));
        assert!(
            out.spelled_out.is_empty(),
            "case-insensitive match failed: {:?}",
            out.spelled_out
        );
    }

    #[test]
    fn custom_mapping_unknown_word_still_spelled() {
        let mut mapping = CustomMapping::new();
        mapping.insert("kubernetes", "kjˈubɚnɛtɪs");
        let out = phonemize_with_oov("kubernetes and tokio", Some(&mapping));
        assert_eq!(
            out.spelled_out,
            vec!["tokio"],
            "tokio should still be spelled out"
        );
    }

    #[test]
    fn custom_mapping_empty_is_noop() {
        let mapping = CustomMapping::new();
        let out = phonemize_with_oov("deployed to kubernetes", Some(&mapping));
        assert_eq!(out.spelled_out, vec!["kubernetes"]);
    }

    #[test]
    fn custom_mapping_from_map() {
        let mut map = std::collections::HashMap::new();
        map.insert("tokio".into(), "tˈOkiO".into());
        let mapping = CustomMapping::from_map(map);
        let out = phonemize_with_oov("tokio runtime", Some(&mapping));
        assert!(out.spelled_out.is_empty());
    }

    #[test]
    fn custom_mapping_builder_style() {
        let mapping = CustomMapping::new()
            .with_mapping("kubernetes", "kjˈubɚnɛtɪs")
            .with_mapping("nginx", "ˈɛnʤənˈɛks");
        assert_eq!(mapping.len(), 2);
    }

    #[test]
    fn custom_mapping_extend() {
        let mut mapping = CustomMapping::new();
        mapping.extend([("kubernetes", "kjˈubɚnɛtɪs"), ("nginx", "ˈɛnʤənˈɛks")]);
        assert_eq!(mapping.len(), 2);
        let out = phonemize_with_oov("kubernetes nginx", Some(&mapping));
        assert!(out.spelled_out.is_empty());
    }

    #[test]
    fn custom_mapping_can_be_empty() {
        let mapping = CustomMapping::new();
        assert!(mapping.is_empty());
        assert_eq!(mapping.len(), 0);
    }

    #[test]
    fn phonemize_with_oov_falls_back_when_handler_is_none() {
        let out = phonemize_with_oov("deployed to kubernetes", None);
        assert_eq!(out.spelled_out, vec!["kubernetes"]);
    }

    /// A stub OOV handler that always returns None — behaves like no handler.
    #[test]
    fn phonemize_with_oov_stub_returns_none() {
        struct Stub;
        impl OovPhonemizer for Stub {
            fn phonemize_oov(&self, _word: &str) -> Option<String> {
                None
            }
        }
        let out = phonemize_with_oov("deployed to kubernetes", Some(&Stub));
        assert_eq!(out.spelled_out, vec!["kubernetes"]);
    }

    /// A stub OOV handler that returns fixed phonemes for any word.
    #[test]
    fn phonemize_with_oov_stub_replaces_word() {
        struct Stub;
        impl OovPhonemizer for Stub {
            fn phonemize_oov(&self, _word: &str) -> Option<String> {
                Some("tˈɛst".into())
            }
        }
        let out = phonemize_with_oov("deployed to kubernetes", Some(&Stub));
        assert!(
            out.spelled_out.is_empty(),
            "stub should handle all OOV words"
        );
        assert!(out.phonemes.contains("tˈɛst"), "got: {}", out.phonemes);
    }
}
