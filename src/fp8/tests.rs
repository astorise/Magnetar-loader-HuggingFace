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

#[test]
fn checkpoint_block_metadata_is_explicit_and_validated() {
    let mut config = serde_json::json!({
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128]
    });
    assert_eq!(
        checkpoint_spec(Some(&config)).unwrap(),
        Some(CheckpointBlockSpecV1 {
            block_rows: 128,
            block_columns: 128,
        })
    );
    for invalid in [
        serde_json::json!([128]),
        serde_json::json!([0, 128]),
        serde_json::json!([128, -1]),
    ] {
        config["weight_block_size"] = invalid;
        assert!(checkpoint_spec(Some(&config)).is_err());
    }
    config["weight_block_size"] = serde_json::json!([128, 128]);
    config["fmt"] = "e5m2".into();
    assert!(checkpoint_spec(Some(&config)).is_err());
    config["fmt"] = "e4m3".into();
    config["activation_scheme"] = "static".into();
    assert!(checkpoint_spec(Some(&config)).is_err());
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

#[test]
fn projection_association_rejects_missing_or_incompatible_scales() {
    let tensor = |name: &str, dtype: ModelDType, shape: Vec<u64>, length| ModelTensorMetadata {
        name: name.into(),
        shape,
        storage_dtype: dtype,
        layout: None,
        shard: None,
        offset_bytes: Some(0),
        size_bytes: Some(length),
        quantization: None,
        expected_compute_dtype: None,
        digest: None,
    };
    let weight = tensor(
        "model.layers.0.mlp.up_proj.weight",
        ModelDType::F8E4M3Fn,
        vec![3, 5],
        15,
    );
    let scale = tensor(
        "model.layers.0.mlp.up_proj.weight_scale_inv",
        ModelDType::F32,
        vec![2, 3],
        24,
    );
    let spec = Some(CheckpointBlockSpecV1 {
        block_rows: 2,
        block_columns: 2,
    });
    assert!(extract_projections(&mut vec![weight.clone()], spec).is_err());
    assert!(extract_projections(&mut vec![weight.clone(), scale.clone()], None).is_err());
    for invalid in [
        ModelTensorMetadata {
            storage_dtype: ModelDType::Bf16,
            ..scale.clone()
        },
        ModelTensorMetadata {
            shape: vec![2, 2],
            ..scale.clone()
        },
        ModelTensorMetadata {
            size_bytes: Some(20),
            ..scale.clone()
        },
    ] {
        assert!(extract_projections(&mut vec![weight.clone(), invalid], spec).is_err());
    }
    let mut valid = vec![weight, scale];
    assert_eq!(extract_projections(&mut valid, spec).unwrap().len(), 1);
    assert_eq!(valid.len(), 1);
    assert_eq!(valid[0].storage_dtype, ModelDType::F32);
    assert_eq!(valid[0].size_bytes, Some(60));
}
