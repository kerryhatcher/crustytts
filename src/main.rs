/// Claude Code Stop hook — announces session completion via TTS.
///
/// Fires when Claude finishes a turn and is waiting for user input. Reads the
/// session transcript, asks a local Ollama model for a short summary, then
/// synthesizes and plays it with Kokoro TTS (82M-parameter neural model).
///
/// Fully self-contained Rust binary — no Python, no system TTS packages.
/// Synthesis runs through `crustytts-lib`: text -> phonemes -> tokens -> ONNX
/// Kokoro -> audio, with ONNX Runtime statically linked.
///
/// Architecture: the binary uses a self-spawning pattern. The first invocation
/// reads stdin, writes Claude Code's expected control JSON, spawns a detached
/// child with `--background`, and exits in milliseconds. The child does all
/// the heavy work without blocking the terminal.
///
/// Hook registration — in ~/.claude/settings.json "Stop" array:
///     {"type": "command", "command": "crustytts"}
///
/// Pronunciation is handled by `crustytts-lib`, which pairs Misaki's
/// 90k-entry dictionary with a POS tagger — so heteronyms resolve by
/// grammatical role ("I read it yesterday" -> `ɹˈɛd`, "I will read it" ->
/// `ɹˈid`) — and spells out any word the dictionary lacks rather than
/// dropping it silently.
///
/// Environment variables:
///   CLAUDE_TTS_LLM             Ollama model for summarization  (default: qwen3:8b)
///   CLAUDE_TTS_VOICE           Kokoro voice id                 (default: af_heart)
///   CRUSTYTTS_ONNX_MODEL       Path to Kokoro .onnx file       (auto-detected from HF cache)
///   CRUSTYTTS_VOICE            Path to a voice .bin file       (auto-detected from HF cache)

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

/// Locate the Kokoro ONNX model file.
fn kokoro_model_path() -> Option<PathBuf> {
    if let Ok(p) = env::var("CRUSTYTTS_ONNX_MODEL") {
        let path = PathBuf::from(&p);
        if path.is_file() {
            return Some(path);
        }
    }
    // Prefer the quantized build: a third the size, no audible difference for
    // a one-sentence notification.
    for snapshot in onnx_snapshots() {
        for name in ["model_q8f16.onnx", "model.onnx", "model_quantized.onnx"] {
            let candidate = snapshot.join("onnx").join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Locate the voice embedding for [`kokoro_voice`].
fn kokoro_voice_path() -> Option<PathBuf> {
    if let Ok(p) = env::var("CRUSTYTTS_VOICE") {
        let path = PathBuf::from(&p);
        if path.is_file() {
            return Some(path);
        }
    }
    let file = format!("{}.bin", kokoro_voice());
    for snapshot in onnx_snapshots() {
        let candidate = snapshot.join("voices").join(&file);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Snapshot directories of the Kokoro ONNX repo in the HuggingFace cache.
fn onnx_snapshots() -> Vec<PathBuf> {
    let mut found = Vec::new();
    for hf_base in hf_cache_dirs() {
        let snapshots = hf_base.join("hub/models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            found.extend(entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
        }
    }
    found
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

// ── TTS ─────────────────────────────────────────────────────────────────────────

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

/// Synthesize and play `text`.
///
/// Runs through `crustytts-lib` end to end — no Python, no system TTS
/// packages, nothing beyond `aplay` for output.
fn speak(text: &str) {
    // Serialize audio output — multiple Claude Code sessions won't talk over each other
    let _lock = acquire_speaker_lock();

    if let Err(e) = speak_kokoro(text) {
        log(&format!("speak() failed: {e}"));
    }
}

/// Synthesize with Kokoro via crustytts-lib: text → phonemes → tokens → audio.
fn speak_kokoro(text: &str) -> anyhow::Result<()> {
    use crustytts_lib::{
        kokoro::KokoroOnnx, sink::AplaySink, AudioSink, KokoroTokenizer, Synthesizer, Tokenizer,
    };

    let model_path = kokoro_model_path().context(
        "Kokoro ONNX model not found — set CRUSTYTTS_ONNX_MODEL or download \
         onnx-community/Kokoro-82M-v1.0-ONNX",
    )?;
    let voice_path = kokoro_voice_path().with_context(|| {
        format!(
            "voice '{}' not found — set CRUSTYTTS_VOICE to a .bin file",
            kokoro_voice()
        )
    })?;

    let outcome = crustytts_lib::phonemize(text);
    log(&format!("  phonemes: {}", outcome.phonemes));
    if !outcome.spelled_out.is_empty() {
        // Not an error — these were spelled out rather than dropped. Logged so
        // the words worth adding to a dictionary are visible.
        log(&format!("  spelled out: {:?}", outcome.spelled_out));
    }

    let tokens = KokoroTokenizer.encode(&outcome.phonemes);
    let voice = crustytts_lib::load_voice(&voice_path)?;
    let audio = KokoroOnnx::load(&model_path)?.synthesize(&tokens, &voice)?;

    log(&format!(
        "  synth: {} samples at {} Hz ({:.2}s)",
        audio.samples.len(),
        audio.sample_rate,
        audio.duration_secs()
    ));

    AplaySink::new().play(&audio)?;
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
