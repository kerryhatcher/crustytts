---
name: crustytts-phonemes
description: Manage custom phoneme mappings for crustytts TTS. Add, edit, or remove word-to-phoneme entries so tech jargon, project names, and uncommon words are pronounced correctly instead of being spelled out letter-by-letter.
---

# Custom Phoneme Mappings for crustytts TTS

crustytts uses the Misaki 90k-entry G2P dictionary, but many tech terms (kubernetes, nginx, ONNX, crate names, etc.) aren't in it. Without a mapping, these get spelled out letter by letter — e.g., "kubernetes" sounds like "K A Y U B E E A R E N E E T E E E E S".

The custom mapping file lets you provide proper IPA phonemes so these words sound natural.

## File Location

The mapping file is at `plugin/custom_phonemes.json` in the repo root. It uses this format:

```json
{
  "version": "1.0",
  "description": "Custom word-to-phoneme mappings for crustytts TTS.",
  "mappings": {
    "kubernetes": "kjˈubɚnɛtɪs",
    "nginx": "ˈɛnʤənˈɛks"
  }
}
```

## How It Works

The binary loads the file at startup from the `CRUSTYTTS_CUSTOM_PHONEMES` environment variable. The mappings are checked **before** the Ollama OOV LLM fallback, in this order:

```
G2P Dictionary → CustomMapping → OOV LLM → Letter spelling
```

So a custom mapping is the most deterministic, fastest way to fix pronunciation.

## Adding a Mapping

1. Open `plugin/custom_phonemes.json`
2. Add a new entry under `"mappings"`:
   ```json
   "yourword": "jˈɔr wˈɜrd"
   ```
3. Set `CRUSTYTTS_CUSTOM_PHONEMES=/path/to/project/plugin/custom_phonemes.json` in your environment
4. Test it:
   ```bash
   crustytts --say "yourword is now pronounced correctly"
   ```

## Editing a Mapping

1. Find the word in `"mappings"`
2. Update its IPA phoneme string
3. Test with a spoken sentence that includes the word

## Removing a Mapping

Delete the entire line for the word from the `"mappings"` object. If the word isn't known to the G2P dictionary or OOV LLM, it will fall back to letter spelling.

## Finding IPA Phonemes for a New Word

Use the Ollama OOV handler to discover a starting-point phoneme string. Since the binary logs any custom-mapped or OOV-produced phonemes, you can:

1. Add a rough phoneme guess
2. Run `crustytts --say "yourword"`
3. Listen and adjust the IPA string
4. Repeat until it sounds right

### Kokoro IPA alphabet reference

| Symbol | Sounds like | Example |
|--------|-------------|---------|
| `ˈ` | Primary stress mark | before stressed syllable |
| `ˌ` | Secondary stress mark | before unstressed syllable |
| `A` | "ay" | d**A** (day) |
| `I` | "ee" | s**I** (see) |
| `O` | "oh" | g**O** (go) |
| `W` | "ow" | n**W** (now) |
| `Y` | "eye" | m**Y** (my) |
| `T` | flap t | be**TT**er |
| `ɔ` | "aw" | **ɔ**l (all) |
| `ɛ` | "eh" | b**ɛ**t (bet) |
| `ɪ` | "ih" | b**ɪ**t (bit) |
| `ɚ` | "er" | n**ɚ**s (nurse) |
| `ə` | "uh" | **ə**bout |
| `æ` | "a" in cat | c**æ**t |
| `ɑ` | "ah" | f**ɑ**ther |
| `ʌ` | "uh" | b**ʌ**t (but) |
| `ʊ` | "oo" in foot | f**ʊ**t |
| `θ` | "th" unvoiced | **th**ink |
| `ð` | "th" voiced | **th**e |
| `ʃ` | "sh" | **sh**ip |
| `ʒ` | "zh" | mea**s**ure |
| `ŋ` | "ng" | so**ng** |
| `ʤ` | "j" | **j**ump |
| `ʧ` | "ch" | **ch**ip |

## Using in Other Projects

If your project depends on the `crustytts-phonemize` crate directly, use `CustomMapping::from_json_file()`:

```rust
use crustytts_phonemize::{CustomMapping, phonemize_with_oov};

// Load from a JSON file (supports both flat and wrapped formats)
let mapping = CustomMapping::from_json_file("my_words.json")?;

// Or build programmatically
let mut mapping = CustomMapping::new();
mapping.insert("mycrate", "mˈIkɹAt");

// Pass to the phonemizer
let outcome = phonemize_with_oov("using mycrate", Some(&mapping));
```

This requires the `json-import` feature:

```toml
[dependencies]
crustytts-phonemize = { version = "0.2", features = ["json-import"] }
```

## Environment

| Variable | Required | Description |
|---|---|---|
| `CRUSTYTTS_CUSTOM_PHONEMES` | No | Path to a JSON file with custom phoneme mappings. If unset or the file is missing, the binary falls back to the OOV LLM handler only. |

## Quick Reference

```bash
# Set the mappings file
export CRUSTYTTS_CUSTOM_PHONEMES=/path/to/crustytts/plugin/custom_phonemes.json

# Test a word
crustytts --say "kubernetes is deployed on nginx"
```
