//! English grapheme-to-phoneme for Kokoro TTS that never silently drops a word.
//!
//! # The problem this solves
//!
//! Kokoro-82M consumes IPA phonemes, not text, so quality depends entirely on
//! the G2P step. The best available Rust engine — [`voice_g2p`], a port of
//! Kokoro's own Misaki with a 90k-entry dictionary and a POS tagger — is
//! excellent on words it knows and **silently emits nothing** for words it
//! doesn't:
//!
//! ```text
//! "Claude deployed to kubernetes"  ->  "klˈɔd dəplˈYd tu "
//!                                                        ^ word gone
//! ```
//!
//! For a developer-facing notification that is the worst failure mode: the
//! sentence stays fluent while losing the one term that carried the meaning.
//! Mispronouncing "kubernetes" is a nuisance; dropping it is misinformation.
//!
//! Two narrower cases fail the same way — `"1:00"` speaks as "zero zero" and
//! `"Feb 2nd"` loses the month — because the dictionary has no entry for the
//! literal token.
//!
//! # What this crate does
//!
//! Wraps `voice-g2p` with the two pieces it lacks:
//!
//! 1. **Normalization** — rewrites clock times and month abbreviations into
//!    words before lookup.
//! 2. **A safety net** — any word that still phonemizes to nothing is spelled
//!    out letter by letter. "K-U-B-E-R-N-E-T-E-S" is clumsy, but it conveys
//!    the word; silence conveys nothing.
//!
//! This is deliberately a *safety net*, not a letter-to-sound engine. A
//! rule-based or neural L2S would pronounce novel words properly rather than
//! spelling them; that is the natural next step and is why [`Outcome`] reports
//! which words took the fallback — feed it real traffic to find out whether
//! the extra machinery is worth it.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "phonemize")] {
//! let out = crustytts_lib::phonemize("Claude deployed to kubernetes at 1:00");
//! assert!(!out.phonemes.is_empty());
//! // "kubernetes" is not in the dictionary, so it was spelled out:
//! assert_eq!(out.spelled_out, vec!["kubernetes"]);
//! # }
//! ```
//!
//! # Taking only part of it
//!
//! Each stage is a trait in [`traits`] and a feature flag, so you can use one
//! piece without the rest — the phonemizer in front of a different model, or
//! the Kokoro synthesizer behind a different phonemizer.
//!
//! ```toml
//! # phonemes only — no ONNX Runtime, fast build (this is the default)
//! crustytts-lib = "0.1"
//!
//! # synthesis only — bring your own phonemes
//! crustytts-lib = { version = "0.1", default-features = false, features = ["kokoro-onnx"] }
//!
//! # the whole pipeline
//! crustytts-lib = { version = "0.1", features = ["full"] }
//! ```

pub mod sink;
pub mod tokenizer;
pub mod traits;
pub mod voice;

#[cfg(feature = "kokoro-onnx")]
pub mod kokoro;

pub use traits::{Audio, AudioSink, Error, Normalizer, Phonemizer, Synthesizer, Tokenizer, Voice};
pub use tokenizer::KokoroTokenizer;
pub use voice::load_voice;

use std::fmt;

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

/// Phonemize English text for Kokoro, guaranteeing no word is silently lost.
///
/// Normalizes times and month abbreviations, phonemizes via the Misaki
/// dictionary, and spells out anything left over. See [`Outcome`].
#[cfg(feature = "phonemize")]
pub fn phonemize(text: &str) -> Outcome {
    let normalized = normalize(text);

    // Substitute unknown words BEFORE phonemizing, so the sentence reaches the
    // POS tagger intact.
    //
    // The obvious implementation — phonemize word by word and patch the empty
    // results — silently breaks heteronyms: the tagger needs surrounding words
    // to tell past-tense "read" (ɹˈɛd) from future "read" (ɹˈid), and in
    // isolation every word looks like a citation form. Rewriting the text
    // first keeps one whole-sentence call, so disambiguation still works.
    let mut prepared = String::with_capacity(normalized.len());
    let mut spelled_out = Vec::new();

    for token in normalized.split_whitespace() {
        if !prepared.is_empty() {
            prepared.push(' ');
        }

        let (word, leading, trailing) = split_affixes(token);
        if word.is_empty() || phonemize_word(word).is_some() {
            prepared.push_str(token);
            continue;
        }

        // Unknown: replace with spaced letters, which the dictionary reads as
        // letter names. Punctuation stays put so prosody is preserved.
        spelled_out.push(word.to_string());
        prepared.push_str(leading);
        prepared.push_str(&spaced_letters(word));
        prepared.push_str(trailing);
    }

    let phonemes = voice_g2p::english_to_phonemes(&prepared)
        .map(|p| p.trim().to_string())
        .unwrap_or_default();

    Outcome {
        phonemes,
        spelled_out,
    }
}

/// Convenience wrapper returning just the phoneme string.
#[cfg(feature = "phonemize")]
pub fn phonemize_str(text: &str) -> String {
    phonemize(text).phonemes
}

/// Phonemize one word, returning `None` if the dictionary had nothing for it.
#[cfg(feature = "phonemize")]
fn phonemize_word(word: &str) -> Option<String> {
    let phonemes = voice_g2p::english_to_phonemes(word).ok()?;
    let phonemes = phonemes.trim();
    (!phonemes.is_empty()).then(|| phonemes.to_string())
}

/// Rewrite a word as space-separated capital letters: "nginx" -> "N G I N X".
///
/// Spacing makes the dictionary read each letter as its name (`ˈɛn`, `ʤˈi`, …)
/// rather than attempting the word. Non-alphanumerics are dropped since they
/// have no letter name.
fn spaced_letters(word: &str) -> String {
    let letters: Vec<String> = word
        .chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_uppercase().to_string())
        .collect();

    letters.join(" ")
}

/// Split leading/trailing punctuation off a token.
///
/// Punctuation carries prosody in Kokoro (a comma is a pause), so it is
/// preserved around the phonemes rather than stripped.
fn split_affixes(token: &str) -> (&str, &str, &str) {
    let start = token
        .find(|c: char| c.is_alphanumeric())
        .unwrap_or(token.len());
    let end = token
        .rfind(|c: char| c.is_alphanumeric())
        .map_or(start, |i| i + token[i..].chars().next().map_or(1, char::len_utf8));

    (&token[start..end], &token[..start], &token[end..])
}

// ── normalization ───────────────────────────────────────────────────────────────

/// Expand constructs the dictionary drops: clock times and month abbreviations.
///
/// Deliberately narrow. `$50`, `100%`, `v1.2.3`, `50/50` and spelled-out month
/// names already phonemize correctly and are left untouched.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if let Some((spoken, len)) = match_clock_time(&chars, i) {
            out.push_str(&spoken);
            i += len;
            continue;
        }
        if let Some((spoken, len)) = match_month_abbrev(&chars, i) {
            out.push_str(spoken);
            i += len;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }

    out
}

/// Rewrite `H:MM` as spoken words: "3:30" -> "3 30", "1:00" -> "1 o'clock".
///
/// Requires a word boundary and exactly two minute digits so version strings
/// and ratios are left alone.
fn match_clock_time(chars: &[char], start: usize) -> Option<(String, usize)> {
    if start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == ':') {
        return None;
    }

    let hour_len = (0..2)
        .take_while(|n| chars.get(start + n).is_some_and(char::is_ascii_digit))
        .count();
    if hour_len == 0 || chars.get(start + hour_len) != Some(&':') {
        return None;
    }

    let m = start + hour_len + 1;
    let minute: String = (0..2)
        .filter_map(|n| chars.get(m + n).filter(|c| c.is_ascii_digit()))
        .collect();
    if minute.len() != 2 || chars.get(m + 2).is_some_and(char::is_ascii_digit) {
        return None;
    }

    let hour: String = chars[start..start + hour_len].iter().collect();
    let spoken = match minute.as_str() {
        "00" => format!("{hour} o'clock"),
        // "9:05" reads as "nine oh five", not "nine five".
        _ if minute.starts_with('0') => format!("{hour} oh {}", &minute[1..]),
        _ => format!("{hour} {minute}"),
    };

    Some((spoken, hour_len + 3))
}

/// Expand a month abbreviation to its full name.
///
/// Only the abbreviations the dictionary drops; "Sept" and full names already work.
fn match_month_abbrev(chars: &[char], start: usize) -> Option<(&'static str, usize)> {
    const MONTHS: &[(&str, &str)] = &[
        ("Jan", "January"),
        ("Feb", "February"),
        ("Mar", "March"),
        ("Apr", "April"),
        ("Jun", "June"),
        ("Jul", "July"),
        ("Aug", "August"),
        ("Oct", "October"),
        ("Nov", "November"),
        ("Dec", "December"),
    ];

    if start > 0 && chars[start - 1].is_alphanumeric() {
        return None;
    }

    for (abbrev, full) in MONTHS {
        let len = abbrev.len();
        if !abbrev
            .chars()
            .enumerate()
            .all(|(n, c)| chars.get(start + n) == Some(&c))
        {
            continue;
        }
        // Must end the word, so "January" isn't clipped by its own "Jan" prefix.
        let consumed = match chars.get(start + len) {
            Some('.') => len + 1,
            Some(c) if c.is_alphanumeric() => continue,
            _ => len,
        };
        return Some((full, consumed));
    }

    None
}

// ── trait implementations for the bundled pieces ────────────────────────────────

/// The bundled phonemizer: Misaki dictionary, POS tagging, letter-spelling net.
#[cfg(feature = "phonemize")]
#[derive(Debug, Clone, Copy, Default)]
pub struct MisakiPhonemizer;

#[cfg(feature = "phonemize")]
impl traits::Phonemizer for MisakiPhonemizer {
    fn phonemize(&self, text: &str) -> Outcome {
        phonemize(text)
    }
}

/// The bundled normalizer: clock times and month abbreviations.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultNormalizer;

impl traits::Normalizer for DefaultNormalizer {
    fn normalize(&self, text: &str) -> String {
        normalize(text)
    }
}


#[cfg(all(test, feature = "phonemize"))]
mod tests {
    use super::*;

    /// The reason this crate exists: an unknown word must never vanish.
    #[test]
    fn oov_words_are_spelled_not_dropped() {
        let out = phonemize("Claude deployed to kubernetes");
        assert_eq!(out.spelled_out, vec!["kubernetes"]);
        // Letter names, not silence.
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

    /// Heteronyms must still resolve by part of speech.
    ///
    /// Regression guard: an earlier implementation phonemized word by word to
    /// find the empty results, which stripped the sentence context the POS
    /// tagger needs — past-tense "read" came out as `ɹˈid`. The whole sentence
    /// must reach the tagger in one piece.
    #[test]
    fn preserves_pos_disambiguation() {
        assert!(
            phonemize("I read the book yesterday").phonemes.contains("ɹˈɛd"),
            "past tense should be ɹˈɛd"
        );
        assert!(
            phonemize("I will read the book tomorrow").phonemes.contains("ɹˈid"),
            "future should be ɹˈid"
        );
    }

    /// Disambiguation must survive even when the sentence also has an OOV word.
    #[test]
    fn preserves_pos_disambiguation_alongside_oov() {
        let out = phonemize("I read the kubernetes docs yesterday");
        assert_eq!(out.spelled_out, vec!["kubernetes"]);
        assert!(out.phonemes.contains("ɹˈɛd"), "got: {}", out.phonemes);
    }

    #[test]
    fn expands_clock_times() {
        assert_eq!(normalize("at 3:30 PM"), "at 3 30 PM");
        assert_eq!(normalize("done at 1:00"), "done at 1 o'clock");
        assert_eq!(normalize("sync at 9:05"), "sync at 9 oh 5");
    }

    #[test]
    fn expands_month_abbreviations() {
        assert_eq!(normalize("Feb 2nd"), "February 2nd");
        assert_eq!(normalize("Jan. 3rd"), "January 3rd");
        assert_eq!(normalize("January 5th"), "January 5th");
    }

    /// Constructs the dictionary already handles must pass through untouched.
    #[test]
    fn leaves_working_constructs_alone() {
        assert_eq!(normalize("v1.2.3"), "v1.2.3");
        assert_eq!(normalize("50/50 split"), "50/50 split");
        assert_eq!(normalize("$50 and 100%"), "$50 and 100%");
        assert_eq!(normalize("ratio 1:000"), "ratio 1:000");
    }

    /// Punctuation carries prosody in Kokoro, so it must survive.
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
}
