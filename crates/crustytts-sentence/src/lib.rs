//! Sentence boundary detection, capitalization, and punctuation restoration.
//!
//! A deterministic normalizer that:
//! 1. Splits text into sentences using existing punctuation or heuristics
//! 2. Capitalizes the first word of each sentence
//! 3. Adds missing terminal punctuation (`.` or `?`) based on sentence structure
//!
//! Implements [`crustytts_core::ProofingStage`] so it can be plugged into a
//! proofreading pipeline.
//!
//! # Example
//!
//! ```rust
//! use crustytts_sentence::SentenceNormalizer;
//! use crustytts_core::ProofingStage;
//!
//! let sn = SentenceNormalizer::new();
//! assert_eq!(sn.proof("claude finished. can we do a game mode"),
//!            "Claude finished. Can we do a game mode?");
//! ```

use crustytts_core::ProofingStage;

/// Detects sentence boundaries, capitalizes, and restores punctuation.
pub struct SentenceNormalizer {
    /// Words that signal a question when they start a sentence.
    question_starters: Vec<&'static str>,
}

impl SentenceNormalizer {
    /// Create a new normalizer with the built-in question-word list.
    pub fn new() -> Self {
        Self {
            question_starters: default_question_starters(),
        }
    }
}

impl Default for SentenceNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofingStage for SentenceNormalizer {
    fn proof(&self, text: &str) -> String {
        if text.trim().is_empty() {
            return text.to_string();
        }

        // Step 1: split into sentence-like segments
        let segments = split_sentences(text);

        // Step 2: process each segment
        let mut result = String::with_capacity(text.len() + 32);
        for (i, segment) in segments.iter().enumerate() {
            let trimmed = segment.trim();
            if trimmed.is_empty() {
                if i > 0 {
                    result.push(' ');
                }
                continue;
            }

            let processed = self.process_segment(trimmed);

            if i > 0 {
                result.push(' ');
            }
            result.push_str(&processed);
        }

        result
    }
}

impl SentenceNormalizer {
    fn process_segment(&self, text: &str) -> String {
        let text = text.trim();
        if text.is_empty() {
            return String::new();
        }

        // Capitalize first letter
        let mut result = capitalize_first(text);

        // Check if it already has terminal punctuation
        let has_terminal = result.ends_with('.')
            || result.ends_with('?')
            || result.ends_with('!')
            || result.ends_with(':')
            || result.ends_with(';');

        if !has_terminal {
            // Determine what punctuation to add
            let punct = self.infer_punctuation(&result);
            result.push(punct);
        }

        result
    }

    /// Infer whether a sentence is a question based on its structure.
    fn infer_punctuation(&self, text: &str) -> char {
        let lower = text.to_lowercase();
        let first_word = lower.split_whitespace().next().unwrap_or("");

        // Check if it starts with a question word
        if self
            .question_starters
            .iter()
            .any(|qw| first_word == *qw || lower.starts_with(&format!("{qw} ")))
        {
            // "is", "are", "was", "were", "has", "have", "had", "do", "does", "did"
            // are only question starters when followed by a subject pronoun or "there"
            if is_ambiguous_question_starter(first_word) {
                if has_question_subject(&lower, first_word) {
                    return '?';
                }
                return '.';
            }
            return '?';
        }

        // Check for question-like patterns mid-sentence
        if lower.contains("can we")
            || lower.contains("could we")
            || lower.contains("would you")
            || lower.contains("will you")
            || lower.contains("do you")
            || lower.contains("does it")
            || lower.contains("is it")
            || lower.contains("is there")
            || lower.contains("are you")
            || lower.contains("are we")
            || lower.contains("should i")
            || lower.contains("should we")
            || lower.contains("shall we")
            || lower.contains("how do")
            || lower.contains("how can")
            || lower.contains("how does")
            || lower.contains("what is")
            || lower.contains("what are")
            || lower.contains("where is")
            || lower.contains("where are")
            || lower.contains("when is")
            || lower.contains("when will")
            || lower.contains("why is")
            || lower.contains("why are")
            || lower.contains("who is")
            || lower.contains("who are")
        {
            return '?';
        }

        '.'
    }
}

// ── sentence splitting ──────────────────────────────────────────────────────────

/// Split text into sentence-like segments.
///
/// Uses existing punctuation as primary boundaries, then falls back to
/// comma-separated clauses when no terminal punctuation exists.
fn split_sentences(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        current.push(ch);

        // Terminal punctuation: `.`, `?`, `!`
        if ch == '.' || ch == '?' || ch == '!' {
            // Check it's not part of an abbreviation (e.g. "Dr.", "vs.")
            if !is_abbreviation_period(&chars, i) {
                segments.push(current.trim().to_string());
                current = String::new();
                // Skip whitespace after punctuation
                while i + 1 < chars.len() && chars[i + 1].is_whitespace() {
                    i += 1;
                }
            }
        }

        i += 1;
    }

    // Handle remaining text
    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        // If no terminal punctuation was found at all, try splitting on commas
        if segments.is_empty() && remaining.contains(',') {
            let comma_parts: Vec<String> = remaining
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            segments.extend(comma_parts);
        } else {
            segments.push(remaining);
        }
    }

    if segments.is_empty() {
        segments.push(text.trim().to_string());
    }

    segments
}

/// Check if a period is part of a known abbreviation.
fn is_abbreviation_period(chars: &[char], period_pos: usize) -> bool {
    // Look backward for a short (1-3 char) word before the period
    if period_pos == 0 {
        return false;
    }

    let mut word_start = period_pos;
    while word_start > 0 && chars[word_start - 1].is_alphabetic() {
        word_start -= 1;
    }

    let word_len = period_pos - word_start;
    if word_len == 0 || word_len > 4 {
        return false;
    }

    let word: String = chars[word_start..period_pos].iter().collect();
    let lower = word.to_lowercase();

    matches!(
        lower.as_str(),
        "dr" | "mr"
            | "mrs"
            | "ms"
            | "prof"
            | "sr"
            | "jr"
            | "vs"
            | "etc"
            | "inc"
            | "ltd"
            | "co"
            | "corp"
            | "st"
            | "ave"
            | "blvd"
            | "rd"
            | "dept"
            | "est"
            | "approx"
            | "esp"
            | "gen"
            | "govt"
            | "a"
            | "b"
            | "c"
            | "d"
            | "e"
            | "f"
            | "g"
            | "h"
            | "i"
            | "j"
            | "k"
            | "l"
            | "m"
            | "n"
            | "o"
            | "p"
            | "q"
            | "r"
            | "s"
            | "t"
            | "u"
            | "v"
            | "w"
            | "x"
            | "y"
            | "z"
    )
}

// ── capitalization ─────────────────────────────────────────────────────────────

fn capitalize_first(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    if let Some(first) = chars.first_mut() {
        if first.is_lowercase() {
            *first = first.to_uppercase().next().unwrap_or(*first);
        }
    }
    chars.into_iter().collect()
}

// ── question detection helpers ─────────────────────────────────────────────────

/// Words that are only question starters when followed by a subject.
fn is_ambiguous_question_starter(word: &str) -> bool {
    matches!(
        word,
        "is" | "are"
            | "was"
            | "were"
            | "has"
            | "have"
            | "had"
            | "do"
            | "does"
            | "did"
            | "am"
            | "may"
            | "might"
            | "must"
            | "can"
            | "could"
            | "would"
            | "should"
            | "shall"
            | "will"
            | "need"
    )
}

/// Check if an ambiguous starter is followed by a question-like subject.
fn has_question_subject(lower: &str, first_word: &str) -> bool {
    let after_first = lower[first_word.len()..].trim();
    let second_word = after_first.split_whitespace().next().unwrap_or("");
    matches!(
        second_word,
        "i" | "you" | "he" | "she" | "it" | "we" | "they" | "there"
    )
}

// ── question starters ──────────────────────────────────────────────────────────

fn default_question_starters() -> Vec<&'static str> {
    vec![
        "who", "what", "where", "when", "why", "how", "which", "whose", "whom", "can", "could",
        "would", "should", "shall", "will", "do", "does", "did", "is", "are", "was", "were", "has",
        "have", "had", "am", "may", "might", "must", "need", "dare", "ought",
    ]
}

// ── tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_first_word() {
        let sn = SentenceNormalizer::new();
        assert_eq!(sn.proof("claude finished."), "Claude finished.");
        assert_eq!(
            sn.proof("the display is working."),
            "The display is working."
        );
    }

    #[test]
    fn adds_period_to_plain_sentences() {
        let sn = SentenceNormalizer::new();
        assert_eq!(sn.proof("claude finished"), "Claude finished.");
        assert_eq!(sn.proof("the display is broken"), "The display is broken.");
    }

    #[test]
    fn adds_question_mark_for_questions() {
        let sn = SentenceNormalizer::new();
        assert_eq!(sn.proof("can we do a game mode"), "Can we do a game mode?");
        assert_eq!(
            sn.proof("how do we jump brown foxes"),
            "How do we jump brown foxes?"
        );
        assert_eq!(sn.proof("what is this"), "What is this?");
        assert_eq!(sn.proof("where are you"), "Where are you?");
    }

    #[test]
    fn handles_multiple_sentences() {
        let sn = SentenceNormalizer::new();
        let result = sn.proof("claude finished. can we do a game mode");
        assert_eq!(result, "Claude finished. Can we do a game mode?");
    }

    #[test]
    fn handles_comma_separated_clauses() {
        let sn = SentenceNormalizer::new();
        let result = sn.proof("your text here, can we do a game mode, how do we jump");
        // Comma-separated clauses each become sentences
        assert!(result.contains("Your text here."));
        assert!(result.contains("Can we do a game mode?"));
        assert!(result.contains("How do we jump?"));
    }

    #[test]
    fn preserves_existing_punctuation() {
        let sn = SentenceNormalizer::new();
        assert_eq!(sn.proof("Hello!"), "Hello!");
        assert_eq!(sn.proof("Really?"), "Really?");
    }

    #[test]
    fn handles_empty_text() {
        let sn = SentenceNormalizer::new();
        assert_eq!(sn.proof(""), "");
        assert_eq!(sn.proof("   "), "   ");
    }

    #[test]
    fn does_not_split_on_abbreviation_periods() {
        let sn = SentenceNormalizer::new();
        let result = sn.proof("dr. smith finished the work");
        assert_eq!(result, "Dr. smith finished the work.");
    }

    #[test]
    fn question_with_mid_sentence_pattern() {
        let sn = SentenceNormalizer::new();
        // "can we" mid-sentence triggers question mark
        assert_eq!(sn.proof("so can we start now"), "So can we start now?");
    }
}
