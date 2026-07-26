//! Text normalization for TTS: expands constructs the Misaki dictionary drops.
//!
//! Handles clock times (`3:30` → `3 30`, `1:00` → `1 o'clock`) and month
//! abbreviations (`Feb` → `February`). Deliberately narrow — `$50`, `100%`,
//! `v1.2.3`, and spelled-out month names already phonemize correctly.

use crustytts_core::Normalizer;

/// Expand clock times and month abbreviations in `text`.
///
/// ```rust
/// assert_eq!(crustytts_normalize::normalize("at 3:30 PM"), "at 3 30 PM");
/// assert_eq!(crustytts_normalize::normalize("Feb 2nd"), "February 2nd");
/// ```
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

/// The bundled normalizer: clock times and month abbreviations.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultNormalizer;

impl Normalizer for DefaultNormalizer {
    fn normalize(&self, text: &str) -> String {
        normalize(text)
    }
}

// ── clock times ─────────────────────────────────────────────────────────────────

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
        _ if minute.starts_with('0') => format!("{hour} oh {}", &minute[1..]),
        _ => format!("{hour} {minute}"),
    };

    Some((spoken, hour_len + 3))
}

// ── month abbreviations ────────────────────────────────────────────────────────

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
        let consumed = match chars.get(start + len) {
            Some('.') => len + 1,
            Some(c) if c.is_alphanumeric() => continue,
            _ => len,
        };
        return Some((full, consumed));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn leaves_working_constructs_alone() {
        assert_eq!(normalize("v1.2.3"), "v1.2.3");
        assert_eq!(normalize("50/50 split"), "50/50 split");
        assert_eq!(normalize("$50 and 100%"), "$50 and 100%");
        assert_eq!(normalize("ratio 1:000"), "ratio 1:000");
    }
}
