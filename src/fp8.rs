//! Reference conversion for block-scaled FP8 E4M3FN weights.
//!
//! This module does not yet enable FP8 checkpoint ingestion. In particular,
//! the ingestor rejects FP8 until scale tensors, their shard identities and
//! the checkpoint's quantization metadata are validated.

use std::{error::Error, fmt};

/// Row-major weight and scale geometry. Each scale multiplies the weight
/// values in one `block_rows` by `block_columns` tile, including edge tiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockLayout {
    pub rows: usize,
    pub columns: usize,
    pub block_rows: usize,
    pub block_columns: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8ConversionError {
    EmptyDimension,
    SizeOverflow,
    WeightLength,
    ScaleLength,
    NonFiniteScale,
    AllocationFailed,
}

impl fmt::Display for Fp8ConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyDimension => "weight and block dimensions must be nonzero",
            Self::SizeOverflow => "weight or scale dimensions overflow",
            Self::WeightLength => "FP8 weight length does not match its shape",
            Self::ScaleLength => "scale count does not match the block geometry",
            Self::NonFiniteScale => "FP8 scales must be finite",
            Self::AllocationFailed => "cannot allocate the dequantized weight",
        })
    }
}

impl Error for Fp8ConversionError {}

/// Decode an E4M3FN byte. This is the finite-only IEEE-style variant used by
/// PyTorch's `float8_e4m3fn`: exponent 15 remains finite except for mantissa
/// 7, which represents NaN. The signed zeros are retained.
pub fn decode_e4m3fn(byte: u8) -> f32 {
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    let value = match (exponent, mantissa) {
        (0, _) => f32::from(mantissa) * (1.0 / 512.0),
        (15, 7) => f32::NAN,
        _ => (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7),
    };
    if byte & 0x80 != 0 { -value } else { value }
}

/// Convert a row-major E4M3FN matrix into `f32`, multiplying each value by
/// its tile's `f32` dequantization scale. `weight_scale_inv` in the pinned
/// DeepSeek/Kimi convention is multiplied, despite its name.
pub fn dequantize_blocks(
    weight: &[u8],
    scales: &[f32],
    layout: BlockLayout,
) -> Result<Vec<f32>, Fp8ConversionError> {
    let BlockLayout {
        rows,
        columns,
        block_rows,
        block_columns,
    } = layout;
    if rows == 0 || columns == 0 || block_rows == 0 || block_columns == 0 {
        return Err(Fp8ConversionError::EmptyDimension);
    }
    let elements = rows
        .checked_mul(columns)
        .ok_or(Fp8ConversionError::SizeOverflow)?;
    if weight.len() != elements {
        return Err(Fp8ConversionError::WeightLength);
    }
    let scale_rows = rows.div_ceil(block_rows);
    let scale_columns = columns.div_ceil(block_columns);
    let scale_count = scale_rows
        .checked_mul(scale_columns)
        .ok_or(Fp8ConversionError::SizeOverflow)?;
    if scales.len() != scale_count {
        return Err(Fp8ConversionError::ScaleLength);
    }
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err(Fp8ConversionError::NonFiniteScale);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| Fp8ConversionError::AllocationFailed)?;
    for (index, byte) in weight.iter().copied().enumerate() {
        let row = index / columns;
        let column = index % columns;
        let scale_index = (row / block_rows) * scale_columns + column / block_columns;
        output.push(decode_e4m3fn(byte) * scales[scale_index]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
