# crustytts Plugin

Text-to-speech notifications for AI coding agents using [Kokoro ONNX TTS](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) — an 82M-parameter neural model that runs entirely locally. No Python, no cloud services, no API keys.

This is an [Open Plugins](https://open-plugins.com/) plugin compatible with conformant agent tools including **Claude Code** and **CherryPi**.

## Prerequisites

1. **crustytts binary** installed on PATH ([github.com/kerryhatcher/crustytts](https://github.com/kerryhatcher/crustytts))
2. **Kokoro ONNX model** downloaded via HuggingFace (`onnx-community/Kokoro-82M-v1.0-ONNX`)
3. **Ollama** running locally (for summarization and spellcheck)

## Components

| Component | Description |
|---|---|
| **Stop Hook** | Fires when the agent is stopped. Reads the session transcript, summarizes the last exchange via Ollama, and speaks the summary aloud. |
| **Session End Hook** | Fires when a session ends. Same transcript-summarize-speak pipeline. |
| **TTS Skill** | Teaches the agent about crustytts capabilities and when to use them. |
| **`/crustytts:speak`** | Slash command to speak arbitrary text aloud. |
| **TTS Speaker Agent** | A specialized sub-agent for TTS announcements. |
| **TTS Rule** | Persistent rule recommending TTS for long-running operations. |
| **Status Bar** | Shows current voice and TTS status in CherryPi's status bar. |

## How It Works

### Transcript → Summarize → Speak pipeline

Both hooks use the same pipeline:

```
Transcript JSONL → load entries → filter by cwd → build context
→ Ollama summarization (one sentence, <20 words) → spellcheck → phonemize
→ Kokoro ONNX TTS → audio
```

| Flag | Tool | What it does |
|---|---|---|
| (no flags, reads stdin) | **Claude Code** | Reads transcript path from stdin JSON payload, summarizes, speaks |
| `--transcript <path> --cwd <cwd>` | **CherryPi** | Reads unified `chat.jsonl`, filters by `cwd` (working directory), summarizes, speaks |

CherryPi writes all sessions to a single `chat.jsonl` with a `cwd` field identifying which project each entry belongs to. The `--cwd` flag filters to only the current session's entries before summarizing.

## Installation

### As a Claude Code plugin

```bash
claude --plugin-dir ./plugin
```

### As a CherryPi plugin

```bash
cherrypi --plugin-dir /path/to/plugin
```

Or symlink into CherryPi's plugin directory:

```bash
ln -s /path/to/plugin /path/to/CherryPi/plugins/crustytts
```

## Usage

Both hooks fire crustytts with no arguments. The binary detects the protocol automatically from stdin:

| Protocol | Tool | How it works |
|---|---|---|
| `{"transcript_path": "..."}` | **Claude Code** | Claude Code passes the per-session transcript path. Binary reads the JSONL, summarizes, speaks. |
| `{"workspace": {"current_dir": "..."}}` | **CherryPi** | CherryPi pipes workspace context via `stdin_context`. Binary finds the unified `chat.jsonl` by platform convention, filters by `current_dir`, summarizes, speaks. |

### Ad-hoc usage

```bash
# Read a JSONL transcript directly (any format)
crustytts --transcript /path/to/chat.jsonl --cwd /path/to/project
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_TTS_VOICE` | `af_heart` | Kokoro voice ID |
| `CLAUDE_TTS_LLM` | `qwen3:8b` | Ollama model for summarization |
| `CLAUDE_TTS_SPELLCHECK_LLM` | `qwen3:0.6b` | Ollama model for spellcheck |
| `CRUSTYTTS_VOICE` | auto-detected | Path to voice `.bin` file |
| `CRUSTYTTS_ONNX_MODEL` | auto-detected | Path to Kokoro `.onnx` file |

## Plugin Structure

```
plugin/
├── .plugin/
│   └── plugin.json              # Plugin manifest (Open Plugins v1.0.0)
├── hooks/
│   └── hooks.json                 # Stop + SessionEnd hook config
├── skills/
│   └── tts/
│       └── SKILL.md               # TTS capability skill
├── commands/
│   └── speak.md                   # TTS speak command
├── agents/
│   └── tts-speaker.md              # TTS speaker sub-agent
├── rules/
│   └── announce-completions.mdc   # TTS usage rule
├── README.md
└── LICENSE
```

## License

MIT