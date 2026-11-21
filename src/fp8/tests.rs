use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    decoded_bits: Vec<Option<u32>>,
    weight: Vec<u8>,
    scales: Vec<f32>,
    scaled_bits: Vec<Option<u32>>,
}

fn reference() -> Reference {
    serde_json::from_slice(include_bytes!("../../testdata/fp8/e4m3fn-reference.json")).unwrap()
}

fn check_bits(actual: f32, expected: Option<u32>) {
    match expected {
        Some(bits) => assert_eq!(actual.to_bits(), bits),
        None => assert!(actual.is_nan()),
    }
}

#[test]
fn every_e4m3fn_bit_pattern_matches_pytorch() {
    let reference = reference();
    assert_eq!(reference.decoded_bits.len(), 256);
    for (byte, expected) in reference.decoded_bits.into_iter().enumerate() {
        check_bits(decode_e4m3fn(byte as u8), expected);
    }
    assert_eq!(decode_e4m3fn(0x7e), 448.0);
    assert_eq!(decode_e4m3fn(0xfe), -448.0);
    assert_eq!(decode_e4m3fn(0x01), 1.0 / 512.0);
    assert!(decode_e4m3fn(0x80).is_sign_negative());
}

#[test]
fn block_scales_and_partial_tiles_match_pytorch() {
    let reference = reference();
    let output = dequantize_blocks(
        &reference.weight,
        &reference.scales,
        BlockLayout {
            rows: 5,
            columns: 7,
            block_rows: 2,
            block_columns: 3,
        },
    )
    .unwrap();
    assert_eq!(output.len(), reference.scaled_bits.len());
    for (actual, expected) in output.into_iter().zip(reference.scaled_bits) {
        check_bits(actual, expected);
    }
}

#[test]
fn invalid_geometry_and_scales_fail_closed() {
    let layout = BlockLayout {
        rows: 2,
        columns: 3,
        block_rows: 2,
        block_columns: 2,
    };
    assert_eq!(
        dequantize_blocks(&[0; 6], &[1.0], layout),
        Err(Fp8ConversionError::ScaleLength)
    );
    assert_eq!(
        dequantize_blocks(&[0; 5], &[1.0, 1.0], layout),
        Err(Fp8ConversionError::WeightLength)
    );
    assert_eq!(
        dequantize_blocks(&[0; 6], &[1.0, f32::NAN], layout),
        Err(Fp8ConversionError::NonFiniteScale)
    );
    assert_eq!(
        dequantize_blocks(&[], &[], BlockLayout { rows: 0, ..layout }),
        Err(Fp8ConversionError::EmptyDimension)
    );
    assert_eq!(
        dequantize_blocks(
            &[],
            &[],
            BlockLayout {
                rows: usize::MAX,
                columns: 2,
                ..layout
            }
        ),
        Err(Fp8ConversionError::SizeOverflow)
    );
}
