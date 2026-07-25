//! Loading Kokoro voice embeddings.

use std::path::Path;

use crate::traits::{Error, Voice};

/// Style vector width for Kokoro-82M: 128 decoder + 128 predictor dimensions.
pub const KOKORO_STYLE_DIM: usize = 256;

/// Load a voice from a `.bin` file of raw little-endian `f32`.
///
/// This is the layout of the `voices/*.bin` files in the
/// `onnx-community/Kokoro-82M-v1.0-ONNX` repository: a flat `[510, 256]`
/// tensor with no header, one row per input length.
pub fn load_voice(path: impl AsRef<Path>) -> Result<Voice, Error> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| Error::Asset(format!("cannot read voice {}: {e}", path.display())))?;

    from_bytes(&bytes, KOKORO_STYLE_DIM)
        .map_err(|e| Error::Format(format!("{} — {e}", path.display())))
}

/// Parse a voice from raw bytes, for embedding a voice in your own binary.
pub fn from_bytes(bytes: &[u8], style_dim: usize) -> Result<Voice, Error> {
    if style_dim == 0 {
        return Err(Error::Format("style_dim must be non-zero".into()));
    }
    if bytes.len() % 4 != 0 {
        return Err(Error::Format(format!(
            "{} bytes is not a whole number of f32 values",
            bytes.len()
        )));
    }

    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if floats.is_empty() || floats.len() % style_dim != 0 {
        return Err(Error::Format(format!(
            "{} floats is not a multiple of style_dim {style_dim}",
            floats.len()
        )));
    }

    let rows = floats.chunks_exact(style_dim).map(<[f32]>::to_vec).collect();

    Ok(Voice { rows, style_dim })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_for(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn parses_rows_of_style_dim() {
        let voice = from_bytes(&bytes_for(&[1.0, 2.0, 3.0, 4.0]), 2).expect("valid");
        assert_eq!(voice.rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(voice.style_dim, 2);
    }

    #[test]
    fn rejects_sizes_that_do_not_divide_evenly() {
        assert!(from_bytes(&bytes_for(&[1.0, 2.0, 3.0]), 2).is_err());
        assert!(from_bytes(&[0u8; 6], 2).is_err(), "not whole f32 values");
        assert!(from_bytes(&[], 2).is_err(), "empty");
        assert!(from_bytes(&bytes_for(&[1.0]), 0).is_err(), "zero style_dim");
    }

    /// Longer utterances must not index past the table.
    #[test]
    fn clamps_row_lookup_to_the_last_row() {
        let voice = from_bytes(&bytes_for(&[1.0, 2.0, 3.0, 4.0]), 2).expect("valid");
        assert_eq!(voice.row_for(0), &[1.0, 2.0]);
        assert_eq!(voice.row_for(1), &[3.0, 4.0]);
        assert_eq!(voice.row_for(9_999), &[3.0, 4.0]);
    }
}
