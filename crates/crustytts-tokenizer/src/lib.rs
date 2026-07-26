//! Kokoro-82M phoneme-to-token-id mapping.
//!
//! The vocabulary (114 entries) is embedded at compile time. Boundary tokens
//! wrap every sequence — Kokoro is trained with them and produces noticeably
//! worse prosody without them.

use std::collections::HashMap;
use std::sync::OnceLock;

use crustytts_core::Tokenizer;

/// Kokoro's phoneme vocabulary, embedded at compile time (114 entries, <1 KB).
const VOCAB_JSON: &str = include_str!("kokoro_vocab.json");

/// Boundary token wrapping every sequence.
const BOUNDARY: i64 = 0;

fn vocab() -> &'static HashMap<String, i64> {
    static VOCAB: OnceLock<HashMap<String, i64>> = OnceLock::new();
    VOCAB.get_or_init(|| serde_json::from_str(VOCAB_JSON).expect("bundled vocab is valid JSON"))
}

/// The tokenizer for Kokoro-82M.
#[derive(Debug, Clone, Copy, Default)]
pub struct KokoroTokenizer;

impl KokoroTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Report characters that are not in the vocabulary.
    ///
    /// [`encode`](Tokenizer::encode) drops these silently, which is the right
    /// runtime behavior but hides bugs during development — a phonemizer
    /// emitting the wrong alphabet shows up here as a long list.
    pub fn unknown_chars(&self, phonemes: &str) -> Vec<char> {
        let v = vocab();
        phonemes
            .chars()
            .filter(|c| !v.contains_key(&c.to_string()))
            .collect()
    }
}

impl Tokenizer for KokoroTokenizer {
    fn encode(&self, phonemes: &str) -> Vec<i64> {
        let v = vocab();
        let mut ids = Vec::with_capacity(phonemes.len() + 2);

        ids.push(BOUNDARY);
        ids.extend(
            phonemes
                .chars()
                .filter_map(|c| v.get(&c.to_string()).copied()),
        );
        ids.push(BOUNDARY);

        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_sequence_in_boundary_tokens() {
        let ids = KokoroTokenizer.encode("klˈɔd");
        assert_eq!(ids.first(), Some(&BOUNDARY));
        assert_eq!(ids.last(), Some(&BOUNDARY));
        assert!(ids.len() > 2, "expected real tokens between boundaries");
    }

    #[test]
    fn encodes_kokoro_compact_tokens() {
        for (token, expected) in [("A", 24), ("I", 25), ("O", 31), ("T", 36), ("W", 39)] {
            let ids = KokoroTokenizer.encode(token);
            assert_eq!(ids, vec![BOUNDARY, expected, BOUNDARY], "token {token}");
        }
    }

    #[test]
    fn skips_characters_outside_the_vocabulary() {
        let ids = KokoroTokenizer.encode("k\tl");
        assert_eq!(ids.len(), 4, "boundaries plus two real tokens");
    }

    #[test]
    fn reports_unknown_characters_for_debugging() {
        assert!(KokoroTokenizer.unknown_chars("klˈɔd").is_empty());
        assert_eq!(KokoroTokenizer.unknown_chars("k\tl"), vec!['\t']);
    }
}
