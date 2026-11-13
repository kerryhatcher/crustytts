//! File output: 16-bit PCM WAV and Ogg/Opus.
//!
//! The format is picked from the file extension (`.wav` or `.ogg`). Audio is
//! encoded in memory and written in one go, so a failed encode never leaves a
//! partial file behind.
//!
//! Ogg output carries Opus, not Vorbis: libopus is the encoder that is
//! available here, and Opus in Ogg is a standard container (RFC 7845).

use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crustytts_core::{Audio, Error};

/// Input rates libopus accepts without resampling. Kokoro produces 24 kHz.
const OPUS_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

/// Ogg Opus pre-skip and granule positions are always counted at 48 kHz (RFC 7845).
const OPUS_GRANULE_RATE: u64 = 48_000;

/// Opus frame length in milliseconds. 20 ms is the usual speech frame size.
const FRAME_MS: u32 = 20;

/// Largest packet we let libopus emit; 4000 bytes is the RFC 6716 recommendation.
const MAX_PACKET_BYTES: usize = 4000;

/// Default Opus bitrate. Mono speech is transparent at this rate.
pub const DEFAULT_OGG_BITRATE_KBPS: u32 = 64;

/// Output container, inferred from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Wav,
    Ogg,
}

impl FileFormat {
    /// Infer the format from `path`'s extension, case-insensitively.
    pub fn from_path(path: &Path) -> Result<Self, Error> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("wav") => Ok(FileFormat::Wav),
            Some("ogg") => Ok(FileFormat::Ogg),
            _ => Err(Error::Audio(format!(
                "cannot pick an audio format for {}: use a .wav or .ogg extension",
                path.display()
            ))),
        }
    }
}

/// Encoder settings for file output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Opus target bitrate in kbit/s, between 6 and 510. Ignored for WAV.
    pub ogg_bitrate_kbps: u32,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            ogg_bitrate_kbps: DEFAULT_OGG_BITRATE_KBPS,
        }
    }
}

/// Encode `audio` in the format implied by `path`'s extension and write it there.
pub fn write_file(audio: &Audio, path: &Path, opts: EncodeOptions) -> Result<(), Error> {
    let bytes = match FileFormat::from_path(path)? {
        FileFormat::Wav => encode_wav(audio)?,
        FileFormat::Ogg => encode_ogg_opus(audio, opts.ogg_bitrate_kbps)?,
    };
    std::fs::write(path, bytes)
        .map_err(|e| Error::Audio(format!("cannot write {}: {e}", path.display())))
}

/// Encode as a canonical 44-byte-header WAV: mono, 16-bit signed PCM.
///
/// Samples outside [-1, 1] are clipped.
pub fn encode_wav(audio: &Audio) -> Result<Vec<u8>, Error> {
    check_rate(audio.sample_rate)?;

    let data_len = audio
        .samples
        .len()
        .checked_mul(2)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::Audio("audio is too long for a WAV file".into()))?;
    let rate = audio.sample_rate;
    let byte_rate = rate
        .checked_mul(2)
        .ok_or_else(|| Error::Audio(format!("sample rate {rate} Hz is out of range")))?;

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size for PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // format: PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels: mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align: 1 channel × 2 bytes
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in &audio.samples {
        out.extend_from_slice(&f32_to_i16(s).to_le_bytes());
    }
    Ok(out)
}

/// Encode as Opus inside an Ogg container (RFC 7845), mono, at `bitrate_kbps`.
///
/// The input rate must be one libopus accepts (8, 12, 16, 24 or 48 kHz).
/// Silent input still yields one valid frame, so the stream always ends properly.
pub fn encode_ogg_opus(audio: &Audio, bitrate_kbps: u32) -> Result<Vec<u8>, Error> {
    let rate = audio.sample_rate;
    if !OPUS_RATES.contains(&rate) {
        return Err(Error::Audio(format!(
            "Ogg Opus needs 8, 12, 16, 24 or 48 kHz audio, got {rate} Hz"
        )));
    }
    if !(6..=510).contains(&bitrate_kbps) {
        return Err(Error::Audio(format!(
            "Opus bitrate must be 6–510 kbit/s, got {bitrate_kbps}"
        )));
    }

    let mut encoder = opus::Encoder::new(rate, opus::Channels::Mono, opus::Application::Audio)
        .map_err(opus_err)?;
    encoder
        .set_bitrate(opus::Bitrate::Bits(bitrate_kbps as i32 * 1000))
        .map_err(opus_err)?;

    // Encoder look-ahead is in input samples. The Ogg header wants 48 kHz units.
    let lookahead = encoder.get_lookahead().map_err(opus_err)?.max(0) as usize;
    let pre_skip = (lookahead as u64 * OPUS_GRANULE_RATE / rate as u64) as u16;

    let frame_len = (rate * FRAME_MS / 1000) as usize;
    let total = audio.samples.len();
    // The encoder holds back `lookahead` samples, so pad with silence until
    // the last real sample has come out. Always at least one frame, so silent
    // or empty input still has audio packets.
    let frames = (total + lookahead).div_ceil(frame_len).max(1);

    let serial = stream_serial();
    let mut out = Vec::new();
    {
        let mut w = ogg::PacketWriter::new(&mut out);
        w.write_packet(
            opus_head(pre_skip, rate),
            serial,
            ogg::PacketWriteEndInfo::EndPage,
            0,
        )
        .map_err(io_err)?;
        w.write_packet(opus_tags(), serial, ogg::PacketWriteEndInfo::EndPage, 0)
            .map_err(io_err)?;

        let mut frame = vec![0f32; frame_len];
        let mut packet = vec![0u8; MAX_PACKET_BYTES];
        for i in 0..frames {
            let start = (i * frame_len).min(total);
            let end = ((i + 1) * frame_len).min(total);
            frame.fill(0.0);
            frame[..end - start].copy_from_slice(&audio.samples[start..end]);

            let n = encoder
                .encode_float(&frame, &mut packet)
                .map_err(opus_err)?;

            // Granule position: samples decoded so far, plus pre-skip. The final
            // packet stops at the real length, so padding is trimmed on decode.
            let granule = pre_skip as u64 + end as u64 * OPUS_GRANULE_RATE / rate as u64;
            let info = if i + 1 == frames {
                ogg::PacketWriteEndInfo::EndStream
            } else {
                ogg::PacketWriteEndInfo::NormalPacket
            };
            w.write_packet(packet[..n].to_vec(), serial, info, granule)
                .map_err(io_err)?;
        }
    }
    Ok(out)
}

fn check_rate(rate: u32) -> Result<(), Error> {
    if rate == 0 {
        return Err(Error::Audio("audio has a sample rate of 0 Hz".into()));
    }
    Ok(())
}

/// Clip to [-1, 1] and scale to signed 16-bit.
fn f32_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// The `OpusHead` identification header (RFC 7845 §5.1).
fn opus_head(pre_skip: u16, input_rate: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(1); // channel count: mono
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_rate.to_le_bytes());
    h.extend_from_slice(&0i16.to_le_bytes()); // output gain
    h.push(0); // channel mapping family 0: mono/stereo
    h
}

/// The `OpusTags` comment header (RFC 7845 §5.2), with no user comments.
fn opus_tags() -> Vec<u8> {
    let vendor = concat!("crustytts ", env!("CARGO_PKG_VERSION"));
    let mut t = Vec::new();
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor.as_bytes());
    t.extend_from_slice(&0u32.to_le_bytes());
    t
}

/// A serial number for the Ogg logical stream. Only needs to differ between files.
fn stream_serial() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(1)
}

fn opus_err(e: opus::Error) -> Error {
    Error::Audio(format!("Opus encoder: {e}"))
}

fn io_err(e: io::Error) -> Error {
    Error::Audio(format!("Ogg writer: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tone(rate: u32, seconds: f32, amplitude: f32) -> Audio {
        let n = (rate as f32 * seconds) as usize;
        let samples = (0..n)
            .map(|i| {
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin()
            })
            .collect();
        Audio {
            samples,
            sample_rate: rate,
        }
    }

    fn rms(xs: &[f32]) -> f32 {
        (xs.iter().map(|x| x * x).sum::<f32>() / xs.len() as f32).sqrt()
    }

    #[test]
    fn format_comes_from_extension() {
        assert_eq!(
            FileFormat::from_path(Path::new("a.wav")).unwrap(),
            FileFormat::Wav
        );
        assert_eq!(
            FileFormat::from_path(Path::new("A.OGG")).unwrap(),
            FileFormat::Ogg
        );
        assert!(FileFormat::from_path(Path::new("a.mp3")).is_err());
        assert!(FileFormat::from_path(Path::new("no_extension")).is_err());
    }

    #[test]
    fn wav_has_canonical_header_and_clipped_samples() {
        let audio = Audio {
            samples: vec![0.0, 1.0, -1.0, 2.0],
            sample_rate: 24_000,
        };
        let b = encode_wav(&audio).unwrap();

        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(&b[8..12], b"WAVE");
        assert_eq!(&b[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([b[20], b[21]]), 1, "PCM");
        assert_eq!(u16::from_le_bytes([b[22], b[23]]), 1, "mono");
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u16::from_le_bytes([b[34], b[35]]), 16, "bits per sample");
        assert_eq!(&b[36..40], b"data");
        assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 8);
        assert_eq!(b.len(), 44 + 8);

        let sample = |i: usize| i16::from_le_bytes([b[44 + 2 * i], b[45 + 2 * i]]);
        assert_eq!(sample(0), 0);
        assert_eq!(sample(1), 32767);
        assert_eq!(sample(2), -32767);
        assert_eq!(sample(3), 32767, "out-of-range samples are clipped");
    }

    #[test]
    fn ogg_round_trips_through_an_opus_decoder() {
        let rate = 24_000;
        let audio = tone(rate, 1.0, 0.5);
        let bytes = encode_ogg_opus(&audio, 64).unwrap();

        let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
        let head = reader.read_packet_expected().unwrap();
        assert_eq!(&head.data[0..8], b"OpusHead");
        let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]) as usize;
        assert_eq!(
            u32::from_le_bytes(head.data[12..16].try_into().unwrap()),
            rate
        );
        let tags = reader.read_packet_expected().unwrap();
        assert_eq!(&tags.data[0..8], b"OpusTags");

        let mut decoder = opus::Decoder::new(rate, opus::Channels::Mono).unwrap();
        let mut decoded = Vec::new();
        let mut buf = vec![0f32; 5760];
        let mut last = None;
        while let Some(packet) = reader.read_packet().unwrap() {
            let n = decoder.decode_float(&packet.data, &mut buf, false).unwrap();
            decoded.extend_from_slice(&buf[..n]);
            last = Some((packet.last_in_stream(), packet.absgp_page()));
        }

        // The final granule is pre-skip plus the real length, at 48 kHz.
        let (is_last, granule) = last.expect("at least one audio packet");
        assert!(is_last, "stream must end with an end-of-stream page");
        assert_eq!(granule, pre_skip as u64 + 2 * audio.samples.len() as u64);

        // Decoding the audio after pre-skip should give back the tone's level.
        let start = pre_skip * rate as usize / 48_000;
        let body = &decoded[start..start + audio.samples.len()];
        let expected = 0.5 / 2f32.sqrt();
        let got = rms(body);
        assert!(
            (got - expected).abs() < expected * 0.1,
            "rms {got} should be close to {expected}"
        );
    }

    #[test]
    fn empty_audio_still_produces_a_valid_stream() {
        let audio = Audio {
            samples: vec![],
            sample_rate: 24_000,
        };
        let bytes = encode_ogg_opus(&audio, 64).unwrap();
        let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
        let mut count = 0;
        let mut last_granule = None;
        while let Some(p) = reader.read_packet().unwrap() {
            count += 1;
            last_granule = Some(p.absgp_page());
        }
        assert_eq!(count, 3, "OpusHead, OpusTags and one silent frame");
        // Nothing to play: the granule is pre-skip alone.
        let pre_skip = last_granule.unwrap();
        assert!(pre_skip < 48_000 / 100);
    }

    #[test]
    fn unsupported_rates_and_bitrates_are_rejected() {
        let audio = tone(22_050, 0.1, 0.5);
        assert!(encode_ogg_opus(&audio, 64).is_err());
        assert!(encode_ogg_opus(&tone(24_000, 0.1, 0.5), 1).is_err());
    }
}
