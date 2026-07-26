//! End-to-end: text -> phonemes -> tokens -> audio -> speakers.
//!
//! cargo run --example speak -- "hello there"

use crustytts_core::{AudioSink, Synthesizer, Tokenizer};
use crustytts_kokoro::KokoroOnnx;
use crustytts_phonemize::phonemize;
use crustytts_sink::AplaySink;
use crustytts_tokenizer::KokoroTokenizer;
use crustytts_voice::load_voice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "Claude deployed to kubernetes at 1:00".to_string()
    } else {
        text
    };

    let model = std::env::var("CRUSTYTTS_ONNX_MODEL")?;
    let voice_path = std::env::var("CRUSTYTTS_VOICE")?;

    let out = phonemize(&text);
    println!("text     : {text}");
    println!("phonemes : {}", out.phonemes);
    if !out.spelled_out.is_empty() {
        println!("spelled  : {:?}", out.spelled_out);
    }

    let tokens = KokoroTokenizer.encode(&out.phonemes);
    let voice = load_voice(&voice_path)?;
    let audio = KokoroOnnx::load(&model)?.synthesize(&tokens, &voice)?;

    println!("audio    : {} samples, {:.2}s", audio.samples.len(), audio.duration_secs());
    AplaySink::new().play(&audio)?;
    Ok(())
}
