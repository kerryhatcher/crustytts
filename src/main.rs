/// Claude Code Stop hook — announces session completion via TTS.
///
/// Fires when Claude finishes a turn and is waiting for user input. Reads the
/// session transcript, asks a local Ollama model for a short summary, then
/// synthesizes and plays it with Kokoro TTS (82M-parameter neural model).
///
/// Fully self-contained Rust binary — no Python, no ONNX Runtime system
/// dependencies. Uses the `any-tts` crate with Candle (pure Rust ML) backend.
///
/// Architecture: the binary uses a self-spawning pattern. The first invocation
/// reads stdin, writes Claude Code's expected control JSON, spawns a detached
/// child with `--background`, and exits in milliseconds. The child does all
/// the heavy work without blocking the terminal.
///
/// Hook registration — in ~/.claude/settings.json "Stop" array:
///     {"type": "command", "command": "crustytts"}
///
/// Pronunciation: upstream `any-tts` ships a hand-rolled English G2P with an
/// 80-word dictionary that mangles ordinary text ("Claude" -> `klæjuːd`, heard
/// as "ca-laa-due"). `vendor/any-tts` patches US English to use `voice-g2p`,
/// which bundles Misaki's 90,201-entry dictionary and a POS tagger — so
/// heteronyms resolve by grammatical role ("I read it yesterday" -> `ɹˈɛd`,
/// "I will read it" -> `ɹˈid`), which eSpeak gets wrong. No system packages,
/// no subprocesses, and MIT-licensed.
///
/// Environment variables:
///   CLAUDE_TTS_LLM             Ollama model for summarization  (default: qwen3:8b)
///   CLAUDE_TTS_VOICE           Kokoro voice id                 (default: af_heart)
///   CRUSTYTTS_KOKORO_MODEL     Path to Kokoro model directory  (auto-detected from HF cache)

use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::Context;
use fs2::FileExt;
use serde::Deserialize;

// ── constants ──────────────────────────────────────────────────────────────────

const OLLAMA_URL: &str = "http://localhost:11434/api/generate";
const LOG_FILE: &str = "/tmp/claude-stop-tts.log";
const SPEAKER_LOCK: &str = "/tmp/crustytts-speaker.lock";

fn ollama_model() -> String {
    env::var("CLAUDE_TTS_LLM").unwrap_or_else(|_| "qwen3:8b".into())
}

fn kokoro_voice() -> String {
    env::var("CLAUDE_TTS_VOICE").unwrap_or_else(|_| "af_heart".into())
}

/// Locate the Kokoro model directory (containing config.json + model weights).
fn kokoro_model_dir() -> Option<PathBuf> {
    if let Ok(p) = env::var("CRUSTYTTS_KOKORO_MODEL") {
        let path = PathBuf::from(&p);
        if path.is_dir() {
            return Some(path);
        }
    }
    // Search HF cache for the Kokoro model
    for hf_base in hf_cache_dirs() {
        let snapshots = hf_base.join("hub/models--hexgrad--Kokoro-82M/snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let config = entry.path().join("config.json");
                if config.exists() {
                    return Some(entry.path());
                }
            }
        }
    }
    None
}

fn hf_cache_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(hf) = env::var("HF_HOME") {
        dirs.push(PathBuf::from(hf));
    }
    if let Ok(home) = env::var("HOME") {
        dirs.push(PathBuf::from(home).join(".cache/huggingface"));
    }
    dirs
}

// ── logging ─────────────────────────────────────────────────────────────────────

fn log(msg: &str) {
    if let Ok(mut fh) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
    {
        let _ = writeln!(fh, "{msg}");
    }
}

// ── JSON / transcript parsing ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct HookPayload {
    transcript_path: Option<String>,
}

/// A single entry from the JSONL transcript. Fields are optional because the
/// format varies: sometimes `type`/`message` wrapper, sometimes bare `role`/`content`.
#[derive(Deserialize, Debug)]
struct TranscriptEntry {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    role: Option<String>,
    message: Option<MessageBlock>,
    content: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct MessageBlock {
    role: Option<String>,
    content: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: Option<String>,
    text: Option<String>,
}

fn load_jsonl(path: &str) -> Vec<TranscriptEntry> {
    let mut entries = Vec::new();
    let Ok(data) = std::fs::read_to_string(path) else {
        return entries;
    };
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<TranscriptEntry>(line) {
            entries.push(entry);
        }
    }
    entries
}

/// Extract readable text from one transcript entry, tolerating format variations.
fn text_from_entry(entry: &TranscriptEntry) -> String {
    // Claude Code sometimes wraps: {"type":"assistant","message":{"role":..,"content":..}}
    let msg = entry.message.as_ref().map(|m| (m.role.as_deref(), m.content.as_ref()));
    let (role, content) = match msg {
        Some((r, c)) => (r, c),
        None => (entry.role.as_deref(), entry.content.as_ref()),
    };

    match content.and_then(|c| c.as_str()) {
        Some(s) => return s.to_string(),
        None => {}
    }

    // Content might be an array of blocks: [{"type":"text","text":"..."}]
    if let Some(arr) = content.and_then(|c| c.as_array()) {
        let parts: Vec<String> = arr
            .iter()
            .filter_map(|b| {
                let cb: Option<ContentBlock> = serde_json::from_value(b.clone()).ok();
                cb.and_then(|cb| {
                    if cb.block_type.as_deref() == Some("text") {
                        cb.text
                    } else {
                        None
                    }
                })
            })
            .collect();
        return parts.join(" ");
    }

    // Fallback: use role as a hint for empty content
    let _ = role;
    String::new()
}

/// Collect recent exchanges into a compact string for the summarizer.
fn build_context(entries: &[TranscriptEntry], max_chars: usize) -> String {
    let mut segments: Vec<String> = Vec::new();
    for entry in entries.iter().rev() {
        let role = entry
            .role
            .as_deref()
            .or(entry.entry_type.as_deref())
            .or(entry.message.as_ref().and_then(|m| m.role.as_deref()));

        let text = text_from_entry(entry);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }

        match role {
            Some("assistant" | "ai") => {
                let truncated: String = text.chars().take(700).collect();
                segments.push(format!("[Claude]: {truncated}"));
            }
            Some("user" | "human") => {
                if !text.starts_with('<') {
                    let truncated: String = text.chars().take(200).collect();
                    segments.push(format!("[User]: {truncated}"));
                }
            }
            _ => continue,
        }

        let total: usize = segments.iter().map(|s| s.len()).sum();
        if total > max_chars {
            break;
        }
    }
    segments.reverse();
    // Keep only the last 10 segments
    let start = if segments.len() > 10 {
        segments.len() - 10
    } else {
        0
    };
    segments[start..].join("\n\n")
}

// ── summarization ───────────────────────────────────────────────────────────────

fn build_summary_prompt(context: &str) -> String {
    format!(
        "You are a text-to-speech notification. Claude Code just stopped working \
         and is waiting for user input.\n\n\
         Write ONE sentence, strictly under 20 words, describing the single most \
         recent thing Claude did. Start with 'Claude'. Past tense. No filler, no \
         preamble — the sentence is the entire response.\n\n\
         Session (most recent last):\n{context}\n\n\
         Respond with the sentence only."
    )
}

/// Extract the first plain sentence of Claude's last message when Ollama is unavailable.
fn naive_fallback(context: &str) -> String {
    for line in context.lines().rev() {
        let line = line.trim();
        if !line.starts_with("[Claude]:") {
            continue;
        }
        let text = &line["[Claude]:".len()..].trim();
        // Skip XML, code fences, and markdown artifacts
        if text.starts_with('<')
            || text.starts_with("```")
            || text.starts_with('#')
            || text.starts_with('|')
            || text.starts_with('!')
        {
            continue;
        }
        for end in ['.', '!', '?'] {
            if let Some(idx) = text.find(end) {
                if idx > 0 && idx < 140 {
                    return text[..=idx].to_string();
                }
            }
        }
        if text.len() > 10 {
            return text.chars().take(120).collect();
        }
    }
    "Claude has finished and is waiting for your input.".into()
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: Option<String>,
}

fn summarize(context: &str) -> String {
    if context.trim().is_empty() {
        log("summarize: context was empty");
        return "Claude is ready and waiting for your next instruction.".into();
    }

    let prompt = build_summary_prompt(context);

    match reqwest::blocking::Client::new()
        .post(OLLAMA_URL)
        .timeout(std::time::Duration::from_secs(8))
        .json(&serde_json::json!({
            "model": ollama_model(),
            "prompt": prompt,
            "stream": false,
            "think": false,
            "options": {"num_predict": 60},
        }))
        .send()
    {
        Ok(resp) => {
            if let Ok(body) = resp.json::<OllamaResponse>() {
                if let Some(text) = body.response {
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        return text;
                    }
                }
            }
            log("summarize: Ollama returned empty response");
        }
        Err(e) => {
            log(&format!("summarize: Ollama failed ({e})"));
        }
    }

    naive_fallback(context)
}

// ── text normalization ──────────────────────────────────────────────────────────

/// Expand constructs that `voice-g2p` silently drops.
///
/// Its dictionary handles `$50`, `100%`, and `v1.2.3` correctly, but two cases
/// lose tokens outright rather than mispronouncing them — which is worse for a
/// notification, since the sentence comes out confidently wrong:
///
///   "at 3:30 PM" -> "æt θˈɜɹTi pˌiˈɛm"   the hour vanishes; 1:00 became "zero zero"
///   "Feb 2nd"    -> " sˈɛkənd "           the month vanishes ("February" is fine)
///
/// Rewriting to words before phonemization sidesteps both.
fn normalize_for_tts(text: &str) -> String {
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

/// Rewrite `H:MM` at `start` as spoken words, e.g. "3:30" -> "3 30", "1:00" -> "1 o'clock".
///
/// Only fires at a word boundary and when the minutes are two digits, so version
/// strings and ratios ("50/50", "v1.2.3") are left for the dictionary to handle.
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
    if minute.len() != 2 {
        return None;
    }
    // A third digit means this isn't a clock time.
    if chars.get(m + 2).is_some_and(char::is_ascii_digit) {
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

/// Expand a month abbreviation at `start` to its full name.
///
/// Only the abbreviations voice-g2p drops; `Sept` already works, and full month
/// names always did.
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
        let matches = abbrev
            .chars()
            .enumerate()
            .all(|(n, c)| chars.get(start + n) == Some(&c));
        if !matches {
            continue;
        }
        // Must end the word — optionally with a period ("Feb." or "Feb"), but
        // never mid-word, so "January" isn't clipped to "JanuaryuARY".
        let after = chars.get(start + len);
        let consumed = match after {
            Some('.') => len + 1,
            Some(c) if c.is_alphanumeric() => continue,
            _ => len,
        };
        return Some((full, consumed));
    }

    None
}

// ── TTS ─────────────────────────────────────────────────────────────────────────

/// Synthesize and play `text` using Kokoro TTS (82M neural model, pure Rust).
///
/// Uses the `any-tts` crate with Candle backend — no Python, no ONNX Runtime,
/// no system dependencies beyond aplay for audio output. Falls back to espeak
/// if the model isn't found or synthesis fails.
/// Acquire an exclusive file lock at `/tmp/crustytts-speaker.lock`.
///
/// Uses `flock(LOCK_EX)` via the `fs2` crate — the kernel queues waiters, so
/// contending processes block here until the current speaker finishes. The lock
/// is released automatically when the returned `File` is dropped (speak() returns
/// or the process exits/crashes). No stale-lock cleanup needed.
fn acquire_speaker_lock() -> Option<File> {
    match File::create(SPEAKER_LOCK) {
        Ok(file) => {
            if let Err(e) = file.lock_exclusive() {
                log(&format!("speaker lock blocked: {e}"));
                None
            } else {
                Some(file)
            }
        }
        Err(e) => {
            log(&format!("cannot create speaker lock: {e}"));
            None
        }
    }
}

fn speak(text: &str) {
    // Serialize audio output — multiple Claude Code sessions won't talk over each other
    let _lock = acquire_speaker_lock();

    let text = &normalize_for_tts(text);

    match speak_kokoro(text) {
        Ok(()) => return,
        Err(e) => log(&format!("kokoro failed ({e}), falling back to espeak")),
    }
    if let Err(e) = speak_espeak(text) {
        log(&format!("speak() failed: {e}"));
    }
}

/// Synthesize with any-tts Kokoro (Candle backend) → aplay.
fn speak_kokoro(text: &str) -> anyhow::Result<()> {
    use any_tts::{ModelType, SynthesisRequest, TtsConfig};

    let model_dir = kokoro_model_dir()
        .context("Kokoro model not found — set CRUSTYTTS_KOKORO_MODEL or download hexgrad/Kokoro-82M")?;
    let voice = kokoro_voice();

    log(&format!(
        "  kokoro model: {} | voice: {voice}",
        model_dir.display()
    ));

    let config = TtsConfig::new(ModelType::Kokoro)
        .with_model_path(model_dir.to_string_lossy());
    let model = any_tts::load_model(config)
        .context("failed to load Kokoro model")?;

    let sample_rate = model.sample_rate();
    let request = SynthesisRequest::new(text).with_language("en");
    let audio = model.synthesize(&request).context("kokoro synthesis failed")?;

    log(&format!(
        "  kokoro synth: {} samples at {sample_rate} Hz",
        audio.samples.len()
    ));

    // Convert f32 samples to little-endian bytes for aplay
    let raw: Vec<u8> = audio
        .samples
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();

    let rate_str = sample_rate.to_string();
    let mut aplay = Command::new("aplay")
        .args(["-r", &rate_str, "-f", "FLOAT_LE", "-t", "raw", "-c", "1"])
        .stdin(Stdio::piped())
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .context("failed to spawn aplay")?;

    if let Some(mut stdin) = aplay.stdin.take() {
        stdin
            .write_all(&raw)
            .context("failed to write audio to aplay")?;
    }

    aplay.wait().context("aplay failed")?;
    Ok(())
}

/// Speak `text` using espeak-ng (or espeak fallback) → aplay pipeline.
fn speak_espeak(text: &str) -> anyhow::Result<()> {
    // Try espeak-ng first, fall back to espeak (older Ubuntu packages)
    let bin = if Command::new("espeak-ng")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
    {
        "espeak-ng"
    } else {
        "espeak"
    };

    let mut tts = Command::new(bin)
        .args(["--stdout"])
        .arg(text)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to spawn {bin}"))?;

    let tts_stdout = tts.stdout.take().with_context(|| format!("{bin} has no stdout"))?;

    let mut aplay = Command::new("aplay")
        .stdin(tts_stdout)
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .context("failed to spawn aplay")?;

    aplay.wait().context("aplay failed")?;
    Ok(())
}

// ── background work ─────────────────────────────────────────────────────────────

fn background_work(transcript_path: &str) {
    log(&format!(
        "--- stop hook fired | transcript={transcript_path:?}"
    ));

    let entries = load_jsonl(transcript_path);
    log(&format!("  entries loaded: {}", entries.len()));

    let context = build_context(&entries, 3000);
    log(&format!(
        "  context chars: {} | preview: {:.120?}",
        context.len(),
        context
    ));

    let text = summarize(&context);
    log(&format!("  speaking: {text:?}"));

    speak(&text);
}

// ── setup ───────────────────────────────────────────────────────────────────────

/// Path to the Claude Code settings file.
fn settings_path() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| "/home/kwhatcher".into());
    PathBuf::from(home).join(".claude/settings.json")
}

/// Update ~/.claude/settings.json so the Stop hook points to `crustytts` on PATH.
///
/// Finds any Stop hook entry whose command references crustytts (or the old
/// claude-stop-tts name) and replaces it with the bare binary name. If no
/// matching hook exists, a new Stop hook entry is appended.
fn setup_hook() -> anyhow::Result<()> {
    const BIN_NAME: &str = "crustytts";

    let path = settings_path();
    let data = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;

    let mut root: serde_json::Value =
        serde_json::from_str(&data).with_context(|| "invalid JSON in settings.json")?;

    // Navigate into hooks.Stop, creating intermediate objects as needed
    let hooks = root
        .as_object_mut()
        .context("settings.json root is not an object")?
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));

    let stop_array = hooks
        .as_object_mut()
        .context("settings.json 'hooks' is not an object")?
        .entry("Stop")
        .or_insert_with(|| serde_json::json!([]));

    let entries = stop_array
        .as_array_mut()
        .context("settings.json 'hooks.Stop' is not an array")?;

    let mut updated = 0;

    for entry in entries.iter_mut() {
        let hook_list = match entry
            .as_object_mut()
            .and_then(|e| e.get_mut("hooks"))
            .and_then(|h| h.as_array_mut())
        {
            Some(h) => h,
            None => continue,
        };

        for hook in hook_list.iter_mut() {
            let obj = match hook.as_object_mut() {
                Some(o) => o,
                None => continue,
            };
            if obj.get("type").and_then(|v| v.as_str()) != Some("command") {
                continue;
            }
            let cmd = match obj.get("command").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => continue,
            };
            if !(cmd.contains("crustytts") || cmd.contains("claude-stop-tts")) {
                continue;
            }
            if cmd == BIN_NAME {
                eprintln!("crustytts: hook already configured");
                return Ok(());
            }
            obj.insert("command".into(), serde_json::Value::String(BIN_NAME.into()));
            updated += 1;
        }
    }

    if updated == 0 {
        // No existing hook found — append a new Stop entry
        entries.push(serde_json::json!({
            "matcher": "",
            "hooks": [
                {"type": "command", "command": BIN_NAME}
            ]
        }));
        updated = 1;
    }

    let out = serde_json::to_string_pretty(&root)
        .context("failed to serialize settings.json")?;
    std::fs::write(&path, out + "\n")
        .with_context(|| format!("cannot write {}", path.display()))?;

    eprintln!("crustytts: updated {updated} hook(s) in {}", path.display());
    Ok(())
}

// ── entry point ─────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    // ── say mode (test TTS directly) ─────────────────────────────────────────
    if args.len() >= 3 && args[1] == "--say" {
        let text = args[2..].join(" ");
        speak(&text);
        return Ok(());
    }

    // ── setup mode ───────────────────────────────────────────────────────────
    if args.len() >= 2 && args[1] == "--setup" {
        return setup_hook();
    }

    // ── background mode ─────────────────────────────────────────────────────
    if args.len() >= 3 && args[1] == "--background" {
        background_work(&args[2]);
        return Ok(());
    }

    // ── foreground mode ──────────────────────────────────────────────────────
    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw)?;

    let payload: HookPayload = serde_json::from_str(&raw).unwrap_or(HookPayload {
        transcript_path: None,
    });

    let transcript_path = payload.transcript_path.unwrap_or_default();

    // Skip background/observer sessions (claude-mem, subagent summarizers, etc.)
    if transcript_path.contains("observer-session") {
        println!("{{\"continue\": true, \"suppressOutput\": true}}");
        return Ok(());
    }

    // Spawn detached child for heavy work
    let exe = env::current_exe().context("cannot determine own executable path")?;
    let _child = Command::new(exe)
        .arg("--background")
        .arg(&transcript_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn background child")?;
    // Don't wait — child runs independently

    // Signal Claude Code and exit (non-blocking)
    println!("{{\"continue\": true, \"suppressOutput\": true}}");
    Ok(())
}

// ── tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::normalize_for_tts;

    /// Times: voice-g2p drops everything before the colon, so "1:00" spoke as
    /// "zero zero" — confidently wrong rather than merely odd.
    #[test]
    fn expands_clock_times() {
        assert_eq!(normalize_for_tts("at 3:30 PM"), "at 3 30 PM");
        assert_eq!(normalize_for_tts("done at 1:00"), "done at 1 o'clock");
        assert_eq!(normalize_for_tts("sync at 9:05"), "sync at 9 oh 5");
        assert_eq!(normalize_for_tts("12:45"), "12 45");
    }

    /// Month abbreviations: voice-g2p drops these entirely, though it handles
    /// the spelled-out names and "Sept" fine.
    #[test]
    fn expands_month_abbreviations() {
        assert_eq!(normalize_for_tts("Feb 2nd"), "February 2nd");
        assert_eq!(normalize_for_tts("Dec 25"), "December 25");
        assert_eq!(normalize_for_tts("Jan. 3rd"), "January 3rd");
    }

    /// A month name must not be clipped by its own prefix.
    #[test]
    fn leaves_full_month_names_alone() {
        assert_eq!(normalize_for_tts("January 5th"), "January 5th");
        assert_eq!(normalize_for_tts("December"), "December");
    }

    /// Constructs voice-g2p already gets right must reach it untouched.
    #[test]
    fn leaves_non_times_alone() {
        assert_eq!(normalize_for_tts("v1.2.3"), "v1.2.3");
        assert_eq!(normalize_for_tts("50/50 split"), "50/50 split");
        assert_eq!(normalize_for_tts("$50 and 100%"), "$50 and 100%");
        // Three-digit tail is not a clock time.
        assert_eq!(normalize_for_tts("ratio 1:000"), "ratio 1:000");
        // Already-expanded text is idempotent.
        assert_eq!(normalize_for_tts("3 30"), "3 30");
    }
}
