---
name: tts
description: Text-to-speech via crustytts — speak output aloud using Kokoro ONNX TTS (local, no cloud).
---

# TTS Skill

You have access to crustytts, a local text-to-speech engine that uses the Kokoro 82M-parameter neural TTS model. All processing happens on-device — no cloud services, no API keys, no network calls.

## Capabilities

- **Speak text aloud**: Synthesize any text to speech and play it through speakers
- **Spellcheck + proof**: Automatically corrects spelling and grammar before speaking
- **Context-aware pronunciation**: Uses Misaki's 90k-entry phoneme dictionary with POS tagging — heteronyms resolve by grammatical role ("I read it" → /rɛd/ vs "I will read" → /rid/)
- **OLLM OOV handling**: Unknown words are spelled out rather than dropped

## Requirements

The `crustytts` binary must be on PATH and the Kokoro ONNX model must be downloaded (auto-detected from HuggingFace cache).

## Usage

Speak text by invoking the binary:

```bash
# Speak a phrase
crustytts --say "Your text here"

# Let the Stop hook trigger automatically (reads the session transcript)
# This fires whenever the agent finishes a turn and is waiting for input.
```

## Voice Configuration

Set the `CLAUDE_TTS_VOICE` environment variable to change the voice:

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_TTS_VOICE` | `af_heart` | Kokoro voice ID (e.g., `af_bella`, `am_adam`, `af_heart`) |
| `CLAUDE_TTS_LLM` | `qwen3:8b` | Ollama model for summarization |
| `CLAUDE_TTS_SPELLCHECK_LLM` | `qwen3:0.6b` | Ollama model for spellcheck |
| `CRUSTYTTS_VOICE` | (auto) | Path to a voice `.bin` file |
| `CRUSTYTTS_ONNX_MODEL` | (auto) | Path to Kokoro `.onnx` file |

## When to use TTS

Use TTS when:
- You want to announce completed tasks audibly
- You're running long operations and want verbal notification when done
- You want accessibility — hearing results instead of reading
- You want to add voice to an agent's output for demonstration or presentation
