---
description: Speak text aloud using crustytts Kokoro TTS.
disable-model-invocation: false
---

# `/crustytts:speak`

Synthesize text to speech and play it through local speakers using the crustytts TTS engine. All processing is local — no cloud services.

## Usage

```
/crustytts:speak <text>
```

## Examples

```
/crustytts:speak Hello, I have finished reviewing your code. Everything looks good.
/crustytts:speak The build completed successfully with no errors.
```

## Notes

- Text is automatically run through spellcheck and pronunciation normalization before synthesis
- Supports OOV (out-of-vocabulary) words by spelling them out
- Uses the Kokoro 82M ONNX model for high-quality neural TTS
- Set `CLAUDE_TTS_VOICE` env var to change voice
