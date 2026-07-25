# crustytts-lib

English text-to-speech for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
that never silently drops a word. Four independent stages — use all of them, or one.

```rust
let out = crustytts_lib::phonemize("Claude deployed to kubernetes at 1:00");
// out.phonemes    -> klˈɔd dəplˈYd tə kˈA jˈu bˈi ˈi ˈɑɹ ˈɛn ˈi tˈi ˈi ˈɛs æt wˈʌn əklˈɑk
// out.spelled_out -> ["kubernetes"]
```

## Why

Kokoro consumes phonemes, not text, so quality lives or dies on the grapheme-to-phoneme
step. The best Rust engine available — [`voice-g2p`](https://crates.io/crates/voice-g2p),
a port of Kokoro's own Misaki with a 90k-entry dictionary and a POS tagger — is excellent
on words it knows and **emits nothing at all** for words it doesn't:

```text
"Claude deployed to kubernetes"  ->  "klˈɔd dəplˈYd tu "
                                                       ^ gone
```

For a developer-facing notification that is the worst failure mode: the sentence stays
fluent while losing the term that carried the meaning. Mispronouncing *kubernetes* is a
nuisance; dropping it is misinformation. `1:00` and `Feb 2nd` fail the same way — spoken
as "zero zero" and "second".

This crate adds the two missing pieces: **normalization** for constructs the dictionary
has no entry for, and a **safety net** that spells out anything still unresolved.
`K-U-B-E-R-N-E-T-E-S` is clumsy, but it conveys the word.

## Pick only what you need

Every stage is a trait in `traits` behind a feature flag.

```toml
# phonemes only — no ONNX Runtime, fast build (default)
crustytts-lib = "0.1"

# synthesis only — bring your own phonemes
crustytts-lib = { version = "0.1", default-features = false, features = ["kokoro-onnx"] }

# everything
crustytts-lib = { version = "0.1", features = ["full"] }
```

| Trait | Bundled implementation | Feature |
|---|---|---|
| `Normalizer` | `DefaultNormalizer` — times, month abbreviations | `phonemize` |
| `Phonemizer` | `MisakiPhonemizer` — dictionary, POS, safety net | `phonemize` |
| `Tokenizer` | `KokoroTokenizer` — 114-entry vocab, embedded | always |
| `Synthesizer` | `KokoroOnnx` — ONNX Runtime inference | `kokoro-onnx` |
| `AudioSink` | `AplaySink`, `CaptureSink` | always |

Swap any one of them: put the phonemizer in front of a different model, or the Kokoro
synthesizer behind a different phonemizer. The traits are object-safe, so a pipeline can
be assembled from config at runtime.

## Full pipeline

```rust,no_run
use crustytts_lib::{
    kokoro::KokoroOnnx, sink::AplaySink,
    AudioSink, KokoroTokenizer, Synthesizer, Tokenizer,
};

let out    = crustytts_lib::phonemize("Claude finished the refactor");
let tokens = KokoroTokenizer.encode(&out.phonemes);
let voice  = crustytts_lib::load_voice("voices/af_heart.bin")?;
let audio  = KokoroOnnx::load("model_q8f16.onnx")?.synthesize(&tokens, &voice)?;

AplaySink::new().play(&audio)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Run it: `cargo run --features full --example speak -- "your text here"`

## Assets

From [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX):

- `onnx/model_q8f16.onnx` — 83 MB quantized (full precision also available)
- `voices/*.bin` — flat `[510, 256]` little-endian `f32`, one row per input length

The phoneme vocabulary is embedded in the crate; nothing to download for phonemization.

## Notes

**`ort` is pinned to `=2.0.0-rc.10`.** Newer release candidates ship a prebuilt ONNX
Runtime built against glibc ≥ 2.38, which fails to link on older distributions with
undefined `__isoc23_strtol`. rc.10 links cleanly and links *statically*, so the binary
needs no system ONNX install.

**Uppercase letters in the output are not a bug.** `A`, `I`, `O`, `T`, `W` are Kokoro's
own compact tokens for `eɪ`, `aɪ`, `oʊ`, `ɾ`, `aʊ` — real vocabulary entries with real ids.

**The safety net is a floor, not a ceiling.** A rule-based or neural letter-to-sound
engine would pronounce novel words properly rather than spelling them. `Outcome::spelled_out`
reports which words took the fallback, so you can measure whether that work is worth doing
before doing it.

## License

MIT
