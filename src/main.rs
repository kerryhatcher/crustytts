/// Claude Code Stop hook — announces session completion via TTS.
///
/// Fires when Claude finishes a turn and is waiting for user input. Reads the
/// session transcript, asks a local Ollama model for a short summary, then
/// synthesizes and plays it with Kokoro TTS (82M-parameter neural model).
///
/// Fully self-contained Rust binary — no Python, no system TTS packages.
/// Synthesis runs through the individual crustytts crates: text -> phonemes -> tokens -> ONNX
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
/// Pronunciation is handled by `crustytts-phonemize`, which pairs Misaki's
/// 90k-entry dictionary with a POS tagger — so heteronyms resolve by
/// grammatical role ("I read it yesterday" -> `ɹˈɛd`, "I will read it" ->
/// `ɹˈid`) — and spells out any word the dictionary lacks rather than
/// dropping it silently.
///
/// Environment variables:
///   CLAUDE_TTS_LLM              Ollama model for summarization  (default: qwen3:8b)
///   CLAUDE_TTS_SPELLCHECK_LLM   Ollama model for spellcheck    (default: qwen3:0.6b)
///   CLAUDE_TTS_VOICE            Kokoro voice id                (default: af_heart)
///   CRUSTYTTS_ONNX_MODEL        Path to Kokoro .onnx file      (auto-detected from HF cache)
///   CRUSTYTTS_VOICE             Path to a voice .bin file      (auto-detected from HF cache)

use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::Context;
use fs2::FileExt;
use serde::Deserialize;

// ── constants ──────────────────────────────────────────────────────────────────

const LOG_FILE: &str = "/tmp/claude-stop-tts.log";
const SPEAKER_LOCK: &str = "/tmp/crustytts-speaker.lock";

fn ollama_model() -> String {
    env::var("CLAUDE_TTS_LLM").unwrap_or_else(|_| "qwen3:8b".into())
}

fn spellcheck_model() -> String {
    env::var("CLAUDE_TTS_SPELLCHECK_LLM").unwrap_or_else(|_| "qwen3:0.6b".into())
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
#[derive(Deserialize, Debug, Clone)]
struct TranscriptEntry {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    role: Option<String>,
    message: Option<MessageBlock>,
    content: Option<serde_json::Value>,
    /// CherryPi unified-log field — identifies the working directory for this session.
    #[serde(default)]
    cwd: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
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

/// Filter transcript entries by `cwd` (CherryPi unified-log field).
///
/// CherryPi writes all sessions to a single `chat.jsonl`, keyed by `cwd`.
/// Returns only entries whose `cwd` matches the given path. When `cwd` is
/// `None`, all entries are returned (Claude Code behavior — one transcript
/// per file).
fn filter_by_cwd(entries: Vec<TranscriptEntry>, cwd: Option<&str>) -> Vec<TranscriptEntry> {
    let cwd = match cwd {
        Some(c) if !c.is_empty() => c,
        _ => return entries,
    };
    entries
        .into_iter()
        .filter(|e| e.cwd.as_deref() == Some(cwd))
        .collect()
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
                segments.push(format!("[Assistant]: {truncated}"));
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
        "You are a text-to-speech notification. An AI coding agent just stopped working \
         and is waiting for user input.\n\n\
         Write ONE sentence, strictly under 20 words, describing the single most \
         recent thing the agent did. Past tense. No filler, no \
         preamble — the sentence is the entire response.\n\n\
         Session (most recent last):\n{context}\n\n\
         Respond with the sentence only."
    )
}

/// Extract the first plain sentence of Claude's last message when Ollama is unavailable.
fn naive_fallback(context: &str) -> String {
    for line in context.lines().rev() {
        let line = line.trim();
        if !line.starts_with("[Assistant]:") {
            continue;
        }
        let text = &line["[Assistant]:".len()..].trim();
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
    "The agent has finished and is waiting for your input.".into()
}

fn summarize(context: &str) -> String {
    if context.trim().is_empty() {
        log("summarize: context was empty");
        return "The agent is ready and waiting for your next instruction.".into();
    }

    let prompt = build_summary_prompt(context);
    let model = ollama_model();

    let req = crustytts_summarize::SummarizeRequest::new(&model, &prompt, "")
        .with_max_tokens(60)
        .with_timeout(8);

    match crustytts_summarize::summarize(&req) {
        Ok(Some(text)) => return text,
        Ok(None) => log("summarize: Ollama returned empty response"),
        Err(e) => log(&format!("summarize: Ollama failed ({e})")),
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

/// Synthesize with Kokoro: spellcheck → phonemes (with custom mapping + OOV LLM) → tokens → audio.
///
/// Custom phoneme mappings are loaded from `CRUSTYTTS_CUSTOM_PHONEMES` env var (JSON file).
/// When set, those mappings are checked before the OOV LLM handler.
fn speak_kokoro(text: &str) -> anyhow::Result<()> {
    use crustytts_chatspeak::ChatSpeakNormalizer;
    use crustytts_codespell::CodespellDict;
    use crustytts_core::{AudioSink, OovPhonemizer, ProofingPipeline, Synthesizer, Tokenizer};
    use crustytts_kokoro::KokoroOnnx;
    use crustytts_phonemize::{phonemize_with_oov, CustomMapping, OllamaOovPhonemizer};
    use crustytts_sentence::SentenceNormalizer;
    use crustytts_sink::AplaySink;
    use crustytts_spellcheck::{OllamaProvider, SpellChecker};
    use crustytts_tokenizer::KokoroTokenizer;
    use crustytts_voice::load_voice;
    use std::sync::OnceLock;

    // ── Pre-processing pipeline: chatspeak → sentence → codespell ──
    static PREPROCESS: OnceLock<ProofingPipeline> = OnceLock::new();
    let preprocess = PREPROCESS.get_or_init(|| {
        ProofingPipeline::new()
            .push(ChatSpeakNormalizer::new())
            .push(SentenceNormalizer::new())
            .push(CodespellDict::global().clone())
    });
    let preprocessed = preprocess.run(text);
    if preprocessed != text {
        log(&format!("  preprocess: \"{text}\" -> \"{preprocessed}\""));
    }

    static SPELLCHECK: OnceLock<SpellChecker> = OnceLock::new();
    let spellcheck = SPELLCHECK.get_or_init(|| {
        SpellChecker::new()
            .allow("kubernetes")
            .allow("nginx")
            .allow("tokio")
            .allow("claude")
            .allow("crustytts")
            .allow("Kokoro")
            .allow("ONNX")
            .allow("Ollama")
            .allow("Misaki")
            .with_llm(OllamaProvider::new(spellcheck_model()).with_timeout(5))
    });

    static OOV: OnceLock<OllamaOovPhonemizer> = OnceLock::new();
    let oov = OOV.get_or_init(|| {
        OllamaOovPhonemizer::new(spellcheck_model()).with_timeout(5)
    });

    // ── Custom phoneme mappings (loaded once from CRUSTYTTS_CUSTOM_PHONEMES) ──
    static CUSTOM_PHONEMES: OnceLock<Option<CustomMapping>> = OnceLock::new();
    let custom = CUSTOM_PHONEMES.get_or_init(|| {
        let env_path = std::env::var("CRUSTYTTS_CUSTOM_PHONEMES").ok()?;
        match CustomMapping::from_json_file(&env_path) {
            Ok(m) => {
                log(&format!("  loaded {} custom phoneme mappings from {env_path:?}", m.len()));
                Some(m)
            }
            Err(e) => {
                log(&format!("  failed to load custom phonemes from {env_path:?}: {e}"));
                None
            }
        }
    });

    // Build the OOV handler chain: custom mappings → LLM fallback
    struct ChainedOov<'a> {
        custom: &'a Option<CustomMapping>,
        llm: &'a OllamaOovPhonemizer,
    }
    impl OovPhonemizer for ChainedOov<'_> {
        fn phonemize_oov(&self, word: &str) -> Option<String> {
            self.custom
                .as_ref()
                .and_then(|m| m.phonemize_oov(word))
                .or_else(|| self.llm.phonemize_oov(word))
        }
    }
    let handler = ChainedOov { custom, llm: oov };

    let corrected = match spellcheck.correct_with_llm(&preprocessed) {
        Ok(c) => c,
        Err(e) => {
            log(&format!("  spellcheck LLM failed ({e}), using dictionary-only"));
            spellcheck.correct(&preprocessed)
        }
    };
    if corrected != preprocessed {
        log(&format!("  spellcheck: \"{preprocessed}\" -> \"{corrected}\""));
    }

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

    let outcome = phonemize_with_oov(&corrected, Some(&handler as &dyn OovPhonemizer));
    log(&format!("  phonemes: {}", outcome.phonemes));
    if !outcome.spelled_out.is_empty() {
        // Not an error — these were spelled out rather than dropped. Logged so
        // the words worth adding to a dictionary are visible.
        log(&format!("  spelled out: {:?}", outcome.spelled_out));
    }

    let tokens = KokoroTokenizer.encode(&outcome.phonemes);
    let voice = load_voice(&voice_path)?;
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

/// Print spellcheck results for a string — for testing/debugging.
fn spellcheck_test(text: &str, use_llm: bool) {
    use crustytts_spellcheck::{OllamaProvider, SpellChecker};
    use std::sync::OnceLock;

    static SPELLCHECK: OnceLock<SpellChecker> = OnceLock::new();
    let spellcheck = SPELLCHECK.get_or_init(|| {
        SpellChecker::new()
            .allow("kubernetes")
            .allow("nginx")
            .allow("tokio")
            .allow("claude")
            .allow("crustytts")
            .allow("Kokoro")
            .allow("ONNX")
            .allow("Ollama")
            .allow("Misaki")
            .with_llm(OllamaProvider::new(spellcheck_model()).with_timeout(5))
    });

    // First, show what the dictionary-only pass finds
    let issues = spellcheck.check(text);
    if issues.is_empty() {
        println!("input:  {text}");
        println!("result: no issues found");
        return;
    }

    println!("input:  {text}");
    for issue in &issues {
        match &issue.suggestion {
            Some(s) => println!("  {:>4}: \"{}\" -> \"{s}\"", "HIGH", issue.word),
            None => println!("  {:>4}: \"{}\" (no close match)", "FLAG", issue.word),
        }
    }

    let corrected = spellcheck.correct(text);
    if corrected != text {
        println!("dict:   {corrected}");
    }

    if use_llm {
        match spellcheck.correct_with_llm(text) {
            Ok(result) => {
                if result != text {
                    println!("llm:    {result}");
                } else {
                    println!("llm:    (no changes)");
                }
            }
            Err(e) => {
                eprintln!("llm error: {e}");
            }
        }
    }
}

/// Run the full proof pipeline: chatspeak → sentence → spellcheck → grammar → LLM.
///
/// Uses the pluggable [`ProofingStage`] pipeline so each stage can be
/// enabled, disabled, or reordered independently.
fn proof_test(text: &str) {
    use crustytts_codespell::CodespellDict;
    use crustytts_core::ProofingStage;
    use crustytts_chatspeak::ChatSpeakNormalizer;
    use crustytts_sentence::SentenceNormalizer;
    use crustytts_spellcheck::{GrammarChecker, HarperChecker, OllamaProvider, SpellChecker};
    use std::sync::OnceLock;

    println!("input:  {text}");

    // ── Stage 1: Chat-speak abbreviation expansion + acronym capitalization ──
    let chatspeak = ChatSpeakNormalizer::new();
    let after_chatspeak = chatspeak.proof(text);
    if after_chatspeak != text {
        println!("--- chatspeak ---");
        println!("  {after_chatspeak}");
    }

    // ── Stage 2: Sentence boundary detection + capitalization + punctuation ──
    let sentence = SentenceNormalizer::new();
    let after_sentence = sentence.proof(&after_chatspeak);
    if after_sentence != after_chatspeak {
        println!("--- sentence ---");
        println!("  {after_sentence}");
    }

    // ── Stage 3: Codespell known-typo dictionary ──
    let codespell = CodespellDict::global();
    let after_codespell = codespell.proof(&after_sentence);
    if after_codespell != after_sentence {
        println!("--- codespell ---");
        println!("  {after_codespell}");
    }

    // ── Stage 4: Spellcheck + grammar + LLM final decision ──
    static ENGINE: OnceLock<SpellChecker> = OnceLock::new();
    let checker = ENGINE.get_or_init(|| {
        SpellChecker::new()
            .allow("kubernetes")
            .allow("nginx")
            .allow("tokio")
            .allow("claude")
            .allow("crustytts")
            .allow("Kokoro")
            .allow("ONNX")
            .allow("Ollama")
            .allow("Misaki")
            .with_llm(OllamaProvider::new(spellcheck_model()).with_timeout(10))
    });

    // Show spellcheck issues on the codespell-corrected text
    let issues = checker.check(&after_codespell);
    let has_spell_suggestions = issues.iter().any(|i| i.suggestion.is_some());
    if !issues.is_empty() {
        println!("--- spellcheck proposals ---");
        for issue in &issues {
            match &issue.suggestion {
                Some(s) => println!("  [spell] \"{}\" → \"{s}\"", issue.word),
                None => println!("  [spell] \"{}\" (no suggestion)", issue.word),
            }
        }
    }

    // Grammar checker (diagnostic only)
    let harper = HarperChecker::new();
    match harper.correct(&after_codespell) {
        Ok(ref corrected) if corrected != &after_codespell => {
            println!("--- grammar: harper ---");
            println!("  [grammar] \"{after_codespell}\" → \"{corrected}\"");
        }
        _ => println!("--- grammar: harper (no changes) ---"),
    }

    // Apply high-confidence spellcheck fixes (dictionary-only, no LLM).
    // The LLM stage is available via correct_full() for when a reliable model
    // is present, but --proof uses the deterministic path exclusively.
    let final_output = if has_spell_suggestions {
        let fixed = checker.correct(&after_codespell);
        if fixed != after_codespell {
            println!("--- spellcheck applied ---");
            println!("  {fixed}");
        }
        fixed
    } else {
        after_codespell.clone()
    };

    // ── Application output ──────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   APPLICATION OUTPUT                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ This is what gets passed to the TTS engine.                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("{final_output}");
}

// ── CherryPi context (from stdin_context in hooks) ─────────────────────────────

/// JSON payload CherryPi pipes to stdin when a hook has `stdin_context: true`.
#[derive(Deserialize)]
struct CherryPiContext {
    workspace: Option<CherryPiWorkspace>,
    #[allow(dead_code)]
    session_id: Option<String>,
}

#[derive(Deserialize)]
struct CherryPiWorkspace {
    current_dir: Option<String>,
}

/// Locate the CherryPi chat log by platform convention.
fn cherrypi_log_path() -> Option<String> {
    // Check XDG_DATA_HOME first
    if let Ok(data_home) = env::var("XDG_DATA_HOME") {
        let path = std::path::Path::new(&data_home).join("cherrypi/logs/chat.jsonl");
        if path.is_file() {
            return Some(path.to_string_lossy().into());
        }
    }
    // Fallback: $HOME/.local/share/cherrypi/logs/chat.jsonl (Linux)
    if let Ok(home) = env::var("HOME") {
        let path = std::path::Path::new(&home)
            .join(".local/share/cherrypi/logs/chat.jsonl");
        if path.is_file() {
            return Some(path.to_string_lossy().into());
        }
        // macOS fallback
        let mac_path = std::path::Path::new(&home)
            .join("Library/Application Support/cherrypi/logs/chat.jsonl");
        if mac_path.is_file() {
            return Some(mac_path.to_string_lossy().into());
        }
    }
    None
}

/// Handle a CherryPi hook invocation: read unified log, filter by workspace, speak.
fn handle_cherrypi_hook(current_dir: &str) {
    let log_path = match cherrypi_log_path() {
        Some(p) => p,
        None => {
            log("cherrypi: cannot locate chat.jsonl — skipping");
            return;
        }
    };

    let mut entries = load_jsonl(&log_path);
    let before = entries.len();
    entries = filter_by_cwd(entries, Some(current_dir));
    log(&format!(
        "cherrypi: loaded {} raw, filtered {} -> {} by cwd={}",
        before,
        before,
        entries.len(),
        current_dir
    ));

    if entries.is_empty() {
        log("cherrypi: no entries match cwd — skipping");
        return;
    }

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

    // ── spellcheck mode (test spellchecker) ──────────────────────────────────
    if args.len() >= 3 && args[1] == "--spellcheck" {
        let text = args[2..].join(" ");
        spellcheck_test(&text, false);
        return Ok(());
    }

    // ── spellcheck + LLM mode ────────────────────────────────────────────────
    if args.len() >= 3 && args[1] == "--spellcheck-llm" {
        let text = args[2..].join(" ");
        spellcheck_test(&text, true);
        return Ok(());
    }

    // ── full proof mode (spellcheck → grammar → LLM final decision) ──────────
    if args.len() >= 3 && args[1] == "--proof" {
        let text = args[2..].join(" ");
        proof_test(&text);
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

    // ── transcript mode (read + summarize + speak any JSONL transcript) ─────
    // Usage: crustytts --transcript <path> [--cwd <path>]
    // Supports any JSONL transcript format (Claude Code, CherryPi, etc.).
    // When --cwd is provided, entries are filtered by CherryPi's cwd field.
    if args.len() >= 3 && args[1] == "--transcript" {
        let path = &args[2];
        let cwd = if args.len() >= 5 && args[3] == "--cwd" {
            Some(args[4].clone())
        } else {
            // Default to PWD when no explicit cwd is given
            env::var("PWD").ok().filter(|s| !s.is_empty())
        };

        let mut entries = load_jsonl(path);
        log(&format!(
            "transcript: loaded {} raw entries from {}",
            entries.len(),
            path
        ));

        if let Some(ref cwd_val) = cwd {
            let before = entries.len();
            entries = filter_by_cwd(entries, Some(cwd_val));
            log(&format!(
                "transcript: filtered {} -> {} entries by cwd={}",
                before,
                entries.len(),
                cwd_val
            ));
        }

        let context = build_context(&entries, 3000);
        log(&format!(
            "  context chars: {} | preview: {:.120?}",
            context.len(),
            context
        ));

        let text = summarize(&context);
        log(&format!("  speaking: {text:?}"));
        speak(&text);
        return Ok(());
    }

    // ── foreground mode ──────────────────────────────────────────────────────
    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw)?;
    let raw = raw.trim();

    // Try Claude Code protocol first: stdin has {"transcript_path": "..."}
    if let Ok(payload) = serde_json::from_str::<HookPayload>(raw) {
        if let Some(ref transcript_path) = payload.transcript_path {
            if !transcript_path.is_empty() {
                // Skip background/observer sessions (claude-mem, subagent summarizers)
                if transcript_path.contains("observer-session") {
                    println!("{{\"continue\": true, \"suppressOutput\": true}}");
                    return Ok(());
                }

                // Spawn detached child for heavy work
                let exe = env::current_exe().context("cannot determine own executable path")?;
                let _child = Command::new(exe)
                    .arg("--background")
                    .arg(transcript_path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .context("failed to spawn background child")?;

                // Signal Claude Code and exit (non-blocking)
                println!("{{\"continue\": true, \"suppressOutput\": true}}");
                return Ok(());
            }
        }
    }

    // Try CherryPi stdin_context protocol: stdin has {"workspace": {"current_dir": "..."}}
    if let Ok(ctx) = serde_json::from_str::<CherryPiContext>(raw) {
        if let Some(ref current_dir) = ctx.workspace.as_ref().and_then(|w| w.current_dir.as_ref()) {
            if !current_dir.is_empty() {
                log(&format!("cherrypi hook: workspace={current_dir}"));
                handle_cherrypi_hook(current_dir);
                return Ok(());
            }
        }
    }

    // Neither protocol matched — silently exit (flag day for old hooks without stdin_context)
    log("stdin: unrecognized payload — neither Claude Code nor CherryPi format");
    Ok(())
}
