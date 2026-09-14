//! Real AWQ 4-bit weight dequantization (`quant_method: "awq"`,
//! `version: "gemm"` -- the AutoAWQ/`llm-awq` GEMM kernel packing, the
//! overwhelmingly common real-world AWQ export shape).
//!
//! A real AWQ-quantized Hugging Face checkpoint replaces each quantized
//! `nn.Linear` projection's plain `<prefix>.weight` tensor with three
//! sibling tensors sharing the same module prefix: `<prefix>.qweight`
//! (packed 4-bit quantized codes), `<prefix>.qzeros` (packed 4-bit
//! zero-points), and `<prefix>.scales` (per-group `f16` scales) -- the
//! same three names GPTQ (`gptq.rs`) uses, but AWQ never declares a
//! `<prefix>.g_idx` sibling; that absence is this module's own
//! disambiguating signal (`extract_gptq_projections` deliberately skips,
//! rather than errors on, a `.qweight` tensor with no `.g_idx`, leaving
//! it for this module's [`extract_awq_projections`] to claim).
//!
//! This module dequantizes those three raw tensors into one real `F32`
//! logical tensor at ingestion time, mirroring `gptq.rs`'s own
//! dequantize-before-transpose precedent exactly.
//!
//! # The AWQ nibble reorder
//!
//! Unlike GPTQ's simple sequential nibble packing, AWQ's real GEMM kernel
//! packs each `i32`'s 8 nibbles in a specific interleaved order (`llm-awq`/
//! AutoAWQ's own `AWQ_REVERSE_ORDER`), not `[0,1,2,...,7]`. Ported here
//! directly from AutoAWQ's real `awq/utils/packing_utils.py`, composing
//! `unpack_awq` and `reverse_awq_order` exactly as its own
//! `dequantize_gemm` does, and independently verified against real bytes
//! downloaded from the public `Qwen/Qwen2.5-0.5B-Instruct-AWQ` checkpoint
//! (`self_attn.v_proj`), compared element-by-element against the same
//! real weight's own unquantized value from `Qwen/Qwen2.5-0.5B-Instruct`
//! -- the naive sequential order reproduces the real unquantized weight
//! with roughly 2.4x the mean absolute error of the real reorder, a
//! systematic misordering, not quantization noise. Unlike GPTQ, AWQ's
//! real `dequantize_gemm` applies no zero-point offset (`iweight -
//! izeros` directly); confirmed against the same real checkpoint bytes.

use magnetar_runtime::model::{ModelDType, ModelTensorMetadata};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::collections::BTreeMap;

/// How many 4-bit codes one `i32` packs (`32 / 4`).
const PACK_FACTOR: u64 = 8;

/// AutoAWQ's real `AWQ_REVERSE_ORDER` (`awq/utils/packing_utils.py`):
/// nibble position `k` (0..7) within a packed `i32` holds the real,
/// semantic column offset `AWQ_REVERSE_ORDER[k]` within that column's
/// 8-wide block -- not `k` itself, unlike GPTQ's plain sequential packing.
const AWQ_REVERSE_ORDER: [u64; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Converts one IEEE 754 binary16 ("half float") value to `f32`, exactly.
/// Verbatim-ported algorithm (not shared -- see `gptq.rs`'s own identical
/// copy and its doc comment explaining why each externalized crate/module
/// carries its own copy rather than depending on `magnetar-runtime`'s
/// `pub(crate)` version).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = u32::from(bits & 0x3FF);
    let magnitude_bits = if exponent == 0 {
        if mantissa == 0 {
            0
        } else {
            let mut mantissa = mantissa;
            let mut shift = 0u32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x3FF;
            let f32_exponent = 127 - 15 - shift + 1;
            (f32_exponent << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1F {
        (0xFFu32 << 23) | (mantissa << 13)
    } else {
        let f32_exponent = (i32::from(exponent) - 15 + 127) as u32;
        (f32_exponent << 23) | (mantissa << 13)
    };
    f32::from_bits(sign | magnitude_bits)
}

fn read_i32_le(bytes: &[u8], reason: &str) -> Result<Vec<i32>, ProductionIngestionError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "{reason}: byte length {} is not a multiple of 4",
                bytes.len()
            ),
        });
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
        .collect())
}

fn read_f16_le(bytes: &[u8], reason: &str) -> Result<Vec<f32>, ProductionIngestionError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "{reason}: byte length {} is not a multiple of 2",
                bytes.len()
            ),
        });
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
        .collect())
}

/// Dequantizes one AWQ (GEMM) 4-bit-quantized projection into real `F32`
/// bytes, logical shape `[in_features, out_features]` row-major --
/// already the Runtime's expected orientation (AWQ's own `qweight` packs
/// one row per input channel directly). See this module's own doc
/// comment for the verified nibble-reorder convention and the (absent)
/// zero-point offset.
pub fn dequantize_awq_projection(
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
    in_features: u64,
    out_features: u64,
) -> Result<Vec<u8>, ProductionIngestionError> {
    if !out_features.is_multiple_of(PACK_FACTOR) {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "AWQ dequantization only supports 4-bit packing (8 codes per i32); \
                 out_features={out_features} must be divisible by {PACK_FACTOR}"
            ),
        });
    }
    let out_blocks = out_features / PACK_FACTOR;

    let qweight = read_i32_le(qweight, "qweight")?;
    if qweight.len() as u64 != in_features * out_blocks {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "qweight has {} i32 elements, expected {in_features}x{out_blocks}",
                qweight.len()
            ),
        });
    }
    let qzeros = read_i32_le(qzeros, "qzeros")?;
    if !(qzeros.len() as u64).is_multiple_of(out_blocks) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "qzeros has {} i32 elements, not a multiple of {out_blocks} columns",
                qzeros.len()
            ),
        });
    }
    let num_groups = qzeros.len() as u64 / out_blocks;
    if num_groups == 0 || !in_features.is_multiple_of(num_groups) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "in_features={in_features} is not evenly divisible by the {num_groups} groups \
                 qzeros declares"
            ),
        });
    }
    let group_size = in_features / num_groups;
    let scales = read_f16_le(scales, "scales")?;
    if scales.len() as u64 != num_groups * out_features {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "scales has {} f16 elements, expected {num_groups}x{out_features} groups \
                 (derived from qzeros)",
                scales.len()
            ),
        });
    }

    let in_features = in_features as usize;
    let out_features = out_features as usize;
    let out_blocks = out_blocks as usize;
    let group_size = group_size as usize;
    let mut dequantized = vec![0f32; in_features * out_features];
    for i in 0..in_features {
        let group = i / group_size;
        for o in 0..out_features {
            let block = o / 8;
            let sub = o % 8;
            let shift = 4 * AWQ_REVERSE_ORDER[sub];

            let packed_weight = qweight[i * out_blocks + block];
            let quant_code = (packed_weight >> shift) & 0xF;

            let packed_zero = qzeros[group * out_blocks + block];
            let zero_code = (packed_zero >> shift) & 0xF;

            let scale = scales[group * out_features + o];
            dequantized[i * out_features + o] = scale * ((quant_code - zero_code) as f32);
        }
    }

    Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// Where one quantized projection's three sibling raw tensors physically
/// live -- captured before `extract_awq_projections` removes their raw
/// entries from the discovered tensor inventory, so
/// [`AwqDequantizingPayloadSource`] can still read their real bytes from
/// the underlying file afterward.
#[derive(Debug)]
struct AwqSiblingRanges {
    qweight: ProductionPayloadRange,
    qzeros: ProductionPayloadRange,
    scales: ProductionPayloadRange,
}

/// One AWQ-quantized projection this ingestor will serve as a single
/// dequantized `F32` logical tensor.
#[derive(Debug)]
pub struct AwqProjection {
    canonical_name: String,
    siblings: AwqSiblingRanges,
    in_features: u64,
    out_features: u64,
}

impl AwqProjection {
    /// The canonical Model Artifact name (e.g. `layers.0.self_attn.
    /// v_proj`) this projection's dequantized bytes are served under.
    pub fn canonical_name(&self) -> &str {
        &self.canonical_name
    }
}

fn payload_range(
    tensor: &ModelTensorMetadata,
) -> Result<ProductionPayloadRange, ProductionIngestionError> {
    let (offset, length) = match (tensor.offset_bytes, tensor.size_bytes) {
        (Some(offset), Some(length)) => (offset, length),
        _ => {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("AWQ tensor '{}' has no declared byte range", tensor.name),
            });
        }
    };
    Ok(ProductionPayloadRange {
        identity: tensor.name.clone(),
        offset,
        length,
        digest: tensor.digest.clone(),
    })
}

/// Removes every raw AWQ-quantized tensor triple (`<prefix>.qweight`/
/// `.qzeros`/`.scales`, with no `.g_idx` sibling -- `gptq.rs`'s own
/// `extract_gptq_projections` already claimed anything that has one) from
/// `tensors`, replacing each with one synthesized placeholder entry named
/// `<prefix>.weight` -- so the existing per-tensor renaming loop in
/// `weights.rs` canonicalizes it exactly like a real plain `.weight`
/// tensor would be -- declaring the projection's real, already
/// Runtime-oriented `[in_features, out_features]` logical shape and `F32`
/// storage dtype. Must run *after* [`super::gptq::extract_gptq_projections`]
/// (so a real GPTQ bundle's tensors are already claimed and cannot be
/// misinterpreted here) and, like it, *before* the raw-to-canonical
/// tensor-name renaming loop.
pub fn extract_awq_projections(
    tensors: &mut Vec<ModelTensorMetadata>,
) -> Result<Vec<AwqProjection>, ProductionIngestionError> {
    let qweight_names: Vec<String> = tensors
        .iter()
        .filter(|tensor| tensor.name.ends_with(".qweight"))
        .map(|tensor| tensor.name.clone())
        .collect();

    let mut projections = Vec::with_capacity(qweight_names.len());
    for qweight_name in qweight_names {
        let prefix = qweight_name
            .strip_suffix(".qweight")
            .expect("filtered by suffix above");
        let qzeros_name = format!("{prefix}.qzeros");
        let scales_name = format!("{prefix}.scales");

        let find = |name: &str| tensors.iter().find(|tensor| tensor.name == name).cloned();
        let qweight_meta = find(&qweight_name).expect("this crate's own filter found it above");
        let qzeros_meta =
            find(&qzeros_name).ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("AWQ tensor '{qweight_name}' has no matching '{qzeros_name}'"),
            })?;
        let scales_meta =
            find(&scales_name).ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("AWQ tensor '{qweight_name}' has no matching '{scales_name}'"),
            })?;

        let [in_features, out_blocks] = qweight_meta.shape[..] else {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("'{qweight_name}' does not have a 2D shape"),
            });
        };
        if out_blocks.checked_mul(PACK_FACTOR).is_none() {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("'{qweight_name}' out_features overflows"),
            });
        }
        let out_features = out_blocks * PACK_FACTOR;

        let canonical_name = crate::naming::normalize_tensor_name(&format!("{prefix}.weight"))?;
        projections.push(AwqProjection {
            canonical_name,
            siblings: AwqSiblingRanges {
                qweight: payload_range(&qweight_meta)?,
                qzeros: payload_range(&qzeros_meta)?,
                scales: payload_range(&scales_meta)?,
            },
            in_features,
            out_features,
        });

        tensors.retain(|tensor| {
            tensor.name != qweight_name && tensor.name != qzeros_name && tensor.name != scales_name
        });
        tensors.push(ModelTensorMetadata {
            name: format!("{prefix}.weight"),
            shape: vec![in_features, out_features],
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: Some(0),
            size_bytes: Some(in_features * out_features * 4),
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        });
    }
    Ok(projections)
}

/// Wraps an inner payload source: a request for an AWQ-quantized
/// projection's canonical identity is answered by reading its three raw
/// sibling tensors' real bytes from `inner` and dequantizing them; every
/// other identity passes through unchanged.
#[derive(Debug)]
pub struct AwqDequantizingPayloadSource<S> {
    inner: S,
    projections: BTreeMap<String, AwqProjection>,
}

impl<S> AwqDequantizingPayloadSource<S> {
    pub fn new(inner: S, projections: Vec<AwqProjection>) -> Self {
        Self {
            inner,
            projections: projections
                .into_iter()
                .map(|projection| (projection.canonical_name.clone(), projection))
                .collect(),
        }
    }
}

impl<S: ProductionArtifactPayloadSource> ProductionArtifactPayloadSource
    for AwqDequantizingPayloadSource<S>
{
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let Some(projection) = self.projections.get(&range.identity) else {
            return self.inner.read_payload(range);
        };
        let qweight = self.inner.read_payload(&projection.siblings.qweight)?;
        let qzeros = self.inner.read_payload(&projection.siblings.qzeros)?;
        let scales = self.inner.read_payload(&projection.siblings.scales)?;
        dequantize_awq_projection(
            &qweight,
            &qzeros,
            &scales,
            projection.in_features,
            projection.out_features,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_i32(nibbles_in_awq_order: &[i32; 8]) -> i32 {
        // Packs 8 semantic column values (already in real column order
        // 0..7) into one i32 the way AutoAWQ's own packer would: nibble
        // position `AWQ_REVERSE_ORDER[k]` holds semantic column `k`.
        let mut value: i32 = 0;
        for (semantic_col, nibble) in nibbles_in_awq_order.iter().enumerate() {
            let position = AWQ_REVERSE_ORDER[semantic_col];
            value |= (nibble & 0xF) << (4 * position);
        }
        value
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f16_bits_from_f32(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mantissa = (bits >> 13) & 0x3FF;
        (sign | ((exponent as u32) << 10) | mantissa) as u16
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| f16_bits_from_f32(*v).to_le_bytes())
            .collect()
    }

    #[test]
    fn dequantizes_a_hand_built_two_group_projection() {
        // in_features=16 (2 groups of 8), out_features=8 (1 packed
        // column-block). Semantic codes [0,1,2,...,7] for every input
        // channel's row, zero_code=3 for group 0, zero_code=1 for group 1
        // (AWQ applies no "+1" offset, unlike GPTQ).
        let scale_g0 = 0.1f32;
        let scale_g1 = 0.2f32;
        let codes = [0, 1, 2, 3, 4, 5, 6, 7];
        let packed_row = pack_i32(&codes);
        let qweight = i32_bytes(&[packed_row; 16]);
        let qzeros = i32_bytes(&[pack_i32(&[3; 8]), pack_i32(&[1; 8])]);
        let scales = f16_bytes(
            &[scale_g0; 8]
                .into_iter()
                .chain([scale_g1; 8])
                .collect::<Vec<_>>(),
        );

        let dequantized = dequantize_awq_projection(&qweight, &qzeros, &scales, 16, 8)
            .expect("well-formed AWQ tensors dequantize");
        let values: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values.len(), 16 * 8);

        // Every input channel in group 0 (i=0..7): code - 3 = -3..4,
        // scaled by 0.1.
        for i in 0..8 {
            for o in 0..8 {
                let expected = scale_g0 * ((o as i32 - 3) as f32);
                let actual = values[i * 8 + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "input {i} col {o}: got {actual}, expected {expected}"
                );
            }
        }
        // Group 1 (i=8..15): code - 1, scaled by 0.2.
        for i in 8..16 {
            for o in 0..8 {
                let expected = scale_g1 * ((o as i32 - 1) as f32);
                let actual = values[i * 8 + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "input {i} col {o}: got {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rejects_an_out_features_not_divisible_by_eight() {
        let error = dequantize_awq_projection(&[], &[], &[], 16, 12).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }

    #[test]
    fn extract_awq_projections_replaces_the_three_siblings_with_one_placeholder() {
        let tensor = |name: &str, shape: Vec<u64>| ModelTensorMetadata {
            name: name.to_string(),
            shape,
            storage_dtype: ModelDType::I32,
            layout: None,
            shard: None,
            offset_bytes: Some(0),
            size_bytes: Some(0),
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        };
        let mut tensors = vec![
            tensor("model.layers.0.self_attn.v_proj.qweight", vec![896, 16]),
            tensor("model.layers.0.self_attn.v_proj.qzeros", vec![7, 16]),
            tensor("model.layers.0.self_attn.v_proj.scales", vec![7, 128]),
            tensor("model.norm.weight", vec![896]),
        ];
        let projections = extract_awq_projections(&mut tensors).expect("extraction succeeds");
        assert_eq!(projections.len(), 1);
        assert_eq!(projections[0].canonical_name, "layers.0.self_attn.v_proj");
        assert_eq!(projections[0].in_features, 896);
        assert_eq!(projections[0].out_features, 128);

        let names: std::collections::BTreeSet<String> =
            tensors.iter().map(|tensor| tensor.name.clone()).collect();
        assert_eq!(
            names,
            std::collections::BTreeSet::from([
                "model.layers.0.self_attn.v_proj.weight".to_string(),
                "model.norm.weight".to_string(),
            ]),
            "the three raw siblings are gone, replaced by one placeholder .weight entry"
        );
        let placeholder = tensors
            .iter()
            .find(|tensor| tensor.name == "model.layers.0.self_attn.v_proj.weight")
            .unwrap();
        assert_eq!(placeholder.shape, vec![896, 128]);
        assert_eq!(placeholder.storage_dtype, ModelDType::F32);
    }

    /// Converts one `bfloat16` value to `f32` exactly (see `gptq.rs`'s
    /// own identical helper and doc comment).
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// Real-checkpoint verification: dequantizes the actual
    /// `self_attn.v_proj` AWQ tensors from the public `Qwen/Qwen2.5-0.5B-
    /// Instruct-AWQ` checkpoint (downloaded via targeted HTTP Range
    /// requests against the real `model.safetensors` file, not
    /// hand-built) and compares every one of its 896x128 elements against
    /// the same real weight's own unquantized `bfloat16` value from the
    /// public `Qwen/Qwen2.5-0.5B-Instruct` checkpoint. The naive
    /// sequential nibble order (no reorder) was tested during development
    /// and reproduces the real weight with ~2.4x this test's mean
    /// error -- this test's tolerance is tight enough to have caught that
    /// misordering, not just "does not crash".
    #[test]
    fn dequantize_awq_projection_matches_a_real_public_checkpoint() {
        const QWEIGHT: &[u8] =
            include_bytes!("../fixtures/awq/qwen2.5-0.5b-instruct-awq.v_proj.qweight.bin");
        const QZEROS: &[u8] =
            include_bytes!("../fixtures/awq/qwen2.5-0.5b-instruct-awq.v_proj.qzeros.bin");
        const SCALES: &[u8] =
            include_bytes!("../fixtures/awq/qwen2.5-0.5b-instruct-awq.v_proj.scales.bin");
        const BASE_WEIGHT_BF16: &[u8] =
            include_bytes!("../fixtures/awq/qwen2.5-0.5b-instruct.v_proj.weight.bf16.bin");

        const IN_FEATURES: u64 = 896;
        const OUT_FEATURES: u64 = 128;

        let dequantized =
            dequantize_awq_projection(QWEIGHT, QZEROS, SCALES, IN_FEATURES, OUT_FEATURES)
                .expect("the real checkpoint's own AWQ tensors dequantize");
        let dequantized: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(dequantized.len(), (IN_FEATURES * OUT_FEATURES) as usize);

        let base: Vec<f32> = BASE_WEIGHT_BF16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk)))
            .collect();
        assert_eq!(base.len(), (IN_FEATURES * OUT_FEATURES) as usize);

        let mut sum_abs_error = 0f64;
        let mut max_abs_error = 0f32;
        for i in 0..IN_FEATURES as usize {
            for o in 0..OUT_FEATURES as usize {
                let dequant_value = dequantized[i * OUT_FEATURES as usize + o];
                let base_value = base[o * IN_FEATURES as usize + i];
                let error = (dequant_value - base_value).abs();
                sum_abs_error += error as f64;
                max_abs_error = max_abs_error.max(error);
            }
        }
        let mean_abs_error = sum_abs_error / (IN_FEATURES * OUT_FEATURES) as f64;
        // Real, measured values (not guessed): the real AWQ_REVERSE_ORDER
        // gives mean=0.00376/max=0.0536 against this real checkpoint;
        // the naive sequential order gives mean=0.00899 -- a clear,
        // systematic ~2.4x separation, not overlapping noise.
        assert!(
            mean_abs_error < 0.006,
            "mean absolute error {mean_abs_error} is too high for correct AWQ dequantization"
        );
        assert!(
            max_abs_error < 0.15,
            "max absolute error {max_abs_error} is too high for correct AWQ dequantization"
        );
    }
}
