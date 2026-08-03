# crustytts

**Local text-to-speech notifications for AI coding agents.**

crustytts is a Claude Code and Codex Stop hook that summarizes the completed response using a local Ollama model and speaks it aloud via Kokoro TTS (82M-parameter neural model on ONNX Runtime).

**No Python. No cloud services. No API keys.** Everything runs locally on your machine.

## Quick Start

```bash
# Install
cargo install --path .

# Configure the Claude Code Stop hook
just setup

# Or configure the Codex Stop hook
crustytts --setup-codex

# Test it
just say "Hello, I am ready."
```

## How It Works

crustytts is a self-contained Rust binary that:

1. **Hooks into Claude Code or Codex** via the Stop event — fires when the agent finishes a turn
2. **Reads the Claude transcript** or Codex's stable `last_assistant_message` hook field
3. **Summarizes** the last exchange using a local Ollama model
4. **Synthesizes speech** via Kokoro ONNX TTS
5. **Plays audio** through `aplay`

### Architecture

The binary uses a self-spawning pattern: the first invocation reads stdin, writes Claude Code's expected control JSON, spawns a detached child with `--background`, and exits in milliseconds. The child does all the heavy work without blocking the terminal.

### Pronunciation Pipeline

```
Text → ChatSpeak Normalizer → Sentence Normalizer → Codespell
     → Spellchecker → Phonemizer (Misaki 90k dict + POS tagging)
     → Kokoro Tokenizer → Kokoro ONNX → Audio
```

Heteronyms resolve by grammatical role ("I read it yesterday" → /ɹɛd/, "I will read it" → /ɹid/). Unknown words are spelled out rather than dropped.

## Prerequisites

- **crustytts binary** installed on PATH (`just setup` handles this)
- **Kokoro ONNX model** — `onnx-community/Kokoro-82M-v1.0-ONNX` from HuggingFace (auto-detected)
- **Ollama** running locally (defaults: `qwen3:8b` for summarization, `qwen3:0.6b` for spellcheck)
- **aplay** (ALSA) for audio output on Linux

## Commands

```bash
# Run the Stop hook (foreground test)
just run

# Speak arbitrary text (bypasses transcript + Ollama)
just say "Your text here"

# Summarize and speak a transcript file
cargo run --release -- --transcript ~/.local/share/cherrypi/logs/chat.jsonl --cwd /path/to/project

# Test spellchecker
cargo run --release -- --spellcheck "Some text with typos"

# Full proof pipeline (spellcheck → grammar → LLM)
cargo run --release -- --proof "Text to proofread"

# Configure the Claude Code hook
just setup

# Configure the Codex hook without replacing existing hooks
just setup-codex

# Install binary to PATH
just install

# Build only
just build
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_TTS_VOICE` | `af_heart` | Kokoro voice ID |
| `CLAUDE_TTS_LLM` | `qwen3:8b` | Ollama model for summarization |
| `CLAUDE_TTS_SPELLCHECK_LLM` | `qwen3:0.6b` | Ollama model for spellcheck |
| `CRUSTYTTS_VOICE` | auto-detected | Path to voice `.bin` file |
| `CRUSTYTTS_ONNX_MODEL` | auto-detected | Path to Kokoro `.onnx` file |

## Open Plugins

crustytts includes an [Open Plugins](https://open-plugins.com/) format plugin at `./plugin/` that Claude Code, CherryPi, and other conformant tools can consume directly.

### Plugin components

| Component | Description |
|---|---|
| **Stop Hook** | Reads the session transcript, summarizes the last exchange via Ollama, and speaks the summary. |
| **Session End Hook** | Same transcript-summarize-speak pipeline on session end. |
| **TTS Skill** | Teaches the agent about crustytts capabilities and when to use them. |
| **`/crustytts:speak`** | Slash command to speak arbitrary text aloud. |
| **TTS Speaker Agent** | A specialized sub-agent for TTS announcements. |

### How the transcript pipeline works

The hooks use the `--transcript` flag to read a JSONL transcript file and run the full summarize+speak pipeline:

| Tool | Transcript source | Filtering |
|---|---|---|
| **Claude Code** | Per-session JSONL file (path from stdin) | One file = one session, no filtering needed |
| **Codex** | `last_assistant_message` from Stop hook stdin | No transcript parsing; Codex rollout JSONL is intentionally not treated as a stable interface |
| **CherryPi** | Unified `~/.local/share/cherrypi/logs/chat.jsonl` | Filtered by `--cwd` — only entries matching the current working directory |

### Usage with Claude Code

```bash
claude --plugin-dir ./plugin
```

### Usage with Codex

```bash
crustytts --setup-codex
```

This safely merges a crustytts `Stop` handler into `$CODEX_HOME/hooks.json` (or
`~/.codex/hooks.json` when `CODEX_HOME` is unset) and preserves existing events
and handlers. Start a new Codex session, run `/hooks`, and review and trust the
new hook before it can run.

### Usage with CherryPi

```bash
cherrypi --plugin-dir ./plugin
```

Or symlink into CherryPi's plugin directory:

```bash
ln -s /path/to/crustytts/plugin /path/to/CherryPi/plugins/crustytts
```

### Plugin structure

```
plugin/
├── .plugin/
│   └── plugin.json              # Open Plugins v1.0.0 manifest
├── hooks/
│   └── hooks.json                 # Stop + SessionEnd hook config
│                                  # Uses --transcript + --cwd for CherryPi
├── skills/
│   └── tts/
│       └── SKILL.md               # TTS capability skill
├── commands/
│   └── speak.md                   # TTS speak command
├── agents/
│   └── tts-speaker.md              # TTS sub-agent
├── rules/
│   └── announce-completions.mdc   # TTS usage rule
├── README.md
└── LICENSE
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Claude Code                                             │
│  ┌──────────────┐    Stop event    ┌──────────────────┐ │
│  │ AI Agent      │ ──────────────► │ crustytts binary  │ │
│  │               │ ◄────────────── │                   │ │
│  │               │  {"continue":   │ 1. Read transcript│ │
│  │               │   true}         │ 2. Summarize (LLM)│ │
│  └──────────────┘                  │ 3. Phonemize      │ │
│                                    │ 4. Synthesize     │ │
│                                    │ 5. Play audio     │ │
│                                    └──────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Crates

crustytts is a Rust workspace with modular crates:

| Crate | Description |
|---|---|
| `crustytts-core` | Core traits: Synthesizer, Tokenizer, AudioSink, ProofingPipeline |
| `crustytts-kokoro` | Kokoro ONNX TTS engine |
| `crustytts-tokenizer` | Kokoro phoneme tokenizer |
| `crustytts-phonemize` | Misaki phonemizer with OOV LLM fallback |
| `crustytts-voice` | Voice embedding loader |
| `crustytts-sink` | Audio playback (aplay) |
| `crustytts-normalize` | Text normalization |
| `crustytts-sentence` | Sentence boundary detection |
| `crustytts-summarize` | LLM summarization client |
| `crustytts-spellcheck` | Spellchecker with LLM support |
| `crustytts-codespell` | Codespell dictionary integration |
| `crustytts-chatspeak` | Chat-speak abbreviation expansion |
| `crustytts-gec` | Grammar error correction |

## License

MIT
