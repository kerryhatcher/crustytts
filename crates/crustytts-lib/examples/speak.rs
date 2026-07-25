//! End-to-end: text -> phonemes -> tokens -> audio -> speakers.
//!
//! cargo run -p crustytts-lib --features full --example speak -- "hello there"

use crustytts_lib::{
    kokoro::KokoroOnnx, sink::AplaySink, AudioSink, KokoroTokenizer, Synthesizer, Tokenizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "Claude deployed to kubernetes at 1:00".to_string()
    } else {
        text
    };

    let model = std::env::var("CRUSTYTTS_ONNX_MODEL")?;
    let voice_path = std::env::var("CRUSTYTTS_VOICE")?;

    let out = crustytts_lib::phonemize(&text);
    println!("text     : {text}");
    println!("phonemes : {}", out.phonemes);
    if !out.spelled_out.is_empty() {
        println!("spelled  : {:?}", out.spelled_out);
    }

    let tokens = KokoroTokenizer.encode(&out.phonemes);
    let voice = crustytts_lib::load_voice(&voice_path)?;
    let audio = KokoroOnnx::load(&model)?.synthesize(&tokens, &voice)?;

    println!("audio    : {} samples, {:.2}s", audio.samples.len(), audio.duration_secs());
    AplaySink::new().play(&audio)?;
    Ok(())
}
