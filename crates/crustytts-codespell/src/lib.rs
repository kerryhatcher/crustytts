//! Codespell dictionary integration for proofreading.
//!
//! Embeds the [codespell](https://github.com/codespell-project/codespell)
//! dictionary (~65,000 known misspelling→correction mappings) and exposes it
//! as a [`ProofingStage`]. The dictionary is parsed once at init time into a
//! `HashMap` — lookups are O(1) with zero inference cost.
//!
//! # Dictionary format
//!
//! Each line is `typo->correction` with optional comma-separated alternatives:
//!
//! ```text
//! recieve->receive
//! occured->occurred
//! impliment->implement
//! seperate->separate, separate,
//! ```
//!
//! When multiple corrections are listed, the first one is used.
//!
//! # License
//!
//! The embedded dictionary is from the codespell project, licensed under
//! CC-BY-SA 3.0. See the NOTICE file for attribution.
//!
//! # Example
//!
//! ```rust
//! use crustytts_codespell::CodespellDict;
//! use crustytts_core::ProofingStage;
//!
//! let dict = CodespellDict::load();
//! assert_eq!(dict.proof("we impliment the feature"), "we implement the feature");
//! assert_eq!(dict.proof("an occured error"), "an occurred error");
//! ```

use crustytts_core::ProofingStage;
use std::collections::HashMap;
use std::sync::OnceLock;

/// The embedded codespell dictionary, parsed at init time.
///
/// Thread-safe: the dictionary is parsed once and shared across all threads.
#[derive(Clone)]
pub struct CodespellDict {
    corrections: HashMap<String, String>,
}

impl CodespellDict {
    /// Parse the embedded dictionary and return a ready-to-use corrector.
    ///
    /// This is cheap — call it once and share the result. For even cheaper
    /// sharing, use [`CodespellDict::global`].
    pub fn load() -> Self {
        let corrections = parse_dictionary(DICTIONARY);
        Self { corrections }
    }

    /// Return a reference to a global singleton, parsed once.
    pub fn global() -> &'static Self {
        static DICT: OnceLock<CodespellDict> = OnceLock::new();
        DICT.get_or_init(Self::load)
    }

    /// Look up a single word. Returns `Some(correction)` if the word is a
    /// known misspelling, or `None` if it's not in the dictionary.
    pub fn correct_word(&self, word: &str) -> Option<&str> {
        self.corrections
            .get(&word.to_lowercase())
            .map(|s| s.as_str())
    }

    /// Number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }
}

impl Default for CodespellDict {
    fn default() -> Self {
        Self::load()
    }
}

impl ProofingStage for CodespellDict {
    fn proof(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() + 16);
        let chars: Vec<char> = text.chars().collect();
        let mut word_start: Option<usize> = None;

        for (i, ch) in chars.iter().enumerate() {
            if ch.is_alphabetic() || (*ch == '\'' && word_start.is_some()) {
                if word_start.is_none() {
                    word_start = Some(i);
                }
            } else {
                if let Some(start) = word_start {
                    let word: String = chars[start..i].iter().collect();
                    result.push_str(&self.lookup_word(&word));
                    word_start = None;
                }
                result.push(*ch);
            }
        }

        if let Some(start) = word_start {
            let word: String = chars[start..].iter().collect();
            result.push_str(&self.lookup_word(&word));
        }

        result
    }
}

impl CodespellDict {
    fn lookup_word(&self, word: &str) -> String {
        let lower = word.to_lowercase();
        if let Some(correction) = self.corrections.get(&lower) {
            // Preserve capitalization of the original word
            if word.chars().all(|c| c.is_uppercase()) {
                return correction.to_uppercase();
            }
            if word.chars().next().is_some_and(|c| c.is_uppercase()) {
                return titlecase_first(correction);
            }
            return correction.clone();
        }
        word.to_string()
    }
}

fn titlecase_first(s: &str) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    if let Some(first) = chars.first_mut() {
        *first = first.to_uppercase().next().unwrap_or(*first);
    }
    chars.into_iter().collect()
}

// ── dictionary parsing ─────────────────────────────────────────────────────────

/// The raw dictionary text, embedded at compile time.
const DICTIONARY: &str = include_str!("../dictionary.txt");

/// Parse the codespell dictionary format into a HashMap.
///
/// Format: `typo->correction1, correction2, ...`
/// - Lines starting with `#` are comments (none in the current dictionary)
/// - Empty lines are skipped
/// - When multiple corrections are listed, the first one is used
/// - Trailing commas are stripped
fn parse_dictionary(raw: &str) -> HashMap<String, String> {
    let mut map = HashMap::with_capacity(65_000);

    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Split on "->"
        let Some(arrow_pos) = line.find("->") else {
            continue;
        };

        let typo = line[..arrow_pos].trim();
        let corrections = line[arrow_pos + 2..].trim();

        if typo.is_empty() || corrections.is_empty() {
            continue;
        }

        // Take the first correction (before any comma)
        let correction = corrections.split(',').next().unwrap_or(corrections).trim();

        if correction.is_empty() {
            continue;
        }

        // Only insert if we don't already have an entry (first wins)
        map.entry(typo.to_lowercase())
            .or_insert_with(|| correction.to_string());
    }

    map
}

// ── tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_entry() {
        let raw = "recieve->receive\n";
        let dict = parse_dictionary(raw);
        assert_eq!(dict.get("recieve").unwrap(), "receive");
    }

    #[test]
    fn parses_entry_with_alternatives() {
        let raw = "seperate->separate, separate,\n";
        let dict = parse_dictionary(raw);
        assert_eq!(dict.get("seperate").unwrap(), "separate");
    }

    #[test]
    fn skips_empty_and_comment_lines() {
        let raw = "\n# comment\nrecieve->receive\n\n";
        let dict = parse_dictionary(raw);
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn skips_malformed_lines() {
        let raw = "no arrow here\nmissing->\n->missing\n";
        let dict = parse_dictionary(raw);
        assert!(dict.is_empty());
    }

    #[test]
    fn full_dictionary_loads() {
        let dict = CodespellDict::load();
        assert!(
            dict.len() > 60_000,
            "expected >60k entries, got {}",
            dict.len()
        );
    }

    #[test]
    fn corrects_known_typos() {
        let dict = CodespellDict::load();
        assert_eq!(dict.correct_word("recieve"), Some("receive"));
        assert_eq!(dict.correct_word("occured"), Some("occurred"));
        assert_eq!(dict.correct_word("impliment"), Some("implement"));
        assert_eq!(dict.correct_word("seperate"), Some("separate"));
        assert_eq!(dict.correct_word("begining"), Some("beginning"));
        assert_eq!(dict.correct_word("enviroment"), Some("environment"));
    }

    #[test]
    fn passes_through_correct_words() {
        let dict = CodespellDict::load();
        assert_eq!(dict.correct_word("receive"), None);
        assert_eq!(dict.correct_word("implementation"), None);
        assert_eq!(dict.correct_word("the"), None);
    }

    #[test]
    fn proof_stage_corrects_in_context() {
        let dict = CodespellDict::load();
        let result = dict.proof("we impliment the feature");
        assert_eq!(result, "we implement the feature");
    }

    #[test]
    fn proof_stage_preserves_capitalization() {
        let dict = CodespellDict::load();
        assert_eq!(dict.proof("Impliment this"), "Implement this");
        assert_eq!(dict.proof("IMPLIMENT"), "IMPLEMENT");
    }

    #[test]
    fn proof_stage_preserves_punctuation() {
        let dict = CodespellDict::load();
        let result = dict.proof("the impliment, and the occured error.");
        assert_eq!(result, "the implement, and the occurred error.");
    }

    #[test]
    fn global_singleton_works() {
        let a = CodespellDict::global();
        let b = CodespellDict::global();
        assert!(std::ptr::eq(a, b));
        assert_eq!(a.correct_word("recieve"), Some("receive"));
    }
}
