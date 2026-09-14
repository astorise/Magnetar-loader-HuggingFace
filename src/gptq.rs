//! Real GPTQ 4-bit weight dequantization.
//!
//! A real GPTQ-quantized Hugging Face checkpoint replaces each quantized
//! `nn.Linear` projection's plain `<prefix>.weight` tensor with four
//! sibling tensors sharing the same module prefix: `<prefix>.qweight`
//! (packed 4-bit quantized codes), `<prefix>.qzeros` (packed 4-bit
//! zero-points), `<prefix>.scales` (per-group `f16` scales), and
//! `<prefix>.g_idx` (each input channel's real quantization group index).
//! This module dequantizes those four raw tensors into one real `F32`
//! logical tensor at ingestion time, mirroring `loaders/gguf`'s own
//! `dequantize.rs` precedent exactly: dequantize *before* this crate's
//! projection-weight transpose (`weight_layout.rs`) ever runs, so every
//! later step treats a GPTQ projection exactly like a tensor that was
//! never quantized.
//!
//! One real difference from `weight_layout.rs`'s transpose, though:
//! GPTQ's own `qweight` packs the *input* dimension first
//! (`[in_features / 8, out_features]`), so the dequantized logical shape
//! is already `[in_features, out_features]` -- the Runtime's expected
//! orientation for every projection weight. A GPTQ-derived tensor must
//! therefore be *excluded* from `weight_layout.rs`'s
//! `nn.Linear`-storage-convention transpose (`swap_declared_projection_
//! shapes`/`TransposingPayloadSource`), not routed through it a second
//! time.
//!
//! Verified against real bytes downloaded from the public
//! `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` checkpoint (`self_attn.v_proj`,
//! `bits: 4`, `group_size: 128`, `sym: true`, `desc_act: false`),
//! compared element-by-element against the same real weight's own
//! unquantized `f32` reconstruction from the public `Qwen/Qwen2.5-0.5B-
//! Instruct` checkpoint -- not recalled from memory or guessed from
//! documentation alone. `dequantize_gptq_projection_matches_a_real_public_
//! checkpoint` (this module's own test) ports that verification into a
//! real, checked-in fixture-backed test.

use magnetar_runtime::model::{ModelDType, ModelTensorMetadata};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::collections::BTreeMap;

/// How many 4-bit codes one `i32` packs (`32 / 4`). GPTQ variants using a
/// different bit width (2/3/8-bit) are real but genuinely rarer in
/// practice and pack differently (3-bit famously straddles `i32`
/// boundaries) -- explicitly rejected below rather than silently
/// mis-decoded, matching this repository's established "support one
/// well-verified format first" precedent (GGUF's own initial `Q8_0`/
/// `Q4_K`/`Q5_K`-only scope).
const PACK_FACTOR: u64 = 8;

/// Converts one IEEE 754 binary16 ("half float") value to `f32`, exactly.
/// Verbatim-ported algorithm (not shared -- see `loaders/gguf/src/
/// dequantize.rs`'s own identical copy and its doc comment explaining why
/// each externalized crate carries its own copy rather than depending on
/// `magnetar-runtime`'s `pub(crate)` version).
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

/// Dequantizes one GPTQ 4-bit-quantized projection into real `F32` bytes,
/// logical shape `[in_features, out_features]` row-major. See this
/// module's own doc comment for the verified packing convention and the
/// `+ 1` zero-point offset (GPTQ's own well-documented packing quirk,
/// confirmed empirically against real checkpoint bytes: omitting it
/// reproduces the real unquantized weight with a systematic one-scale-step
/// bias, not quantization noise).
pub fn dequantize_gptq_projection(
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
    g_idx: &[u8],
    in_features: u64,
    out_features: u64,
) -> Result<Vec<u8>, ProductionIngestionError> {
    if !in_features.is_multiple_of(PACK_FACTOR) || !out_features.is_multiple_of(PACK_FACTOR) {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "GPTQ dequantization only supports 4-bit packing (8 codes per i32); \
                 in_features={in_features}, out_features={out_features} must both be \
                 divisible by {PACK_FACTOR}"
            ),
        });
    }
    let qweight_rows = in_features / PACK_FACTOR;
    let qzeros_cols = out_features / PACK_FACTOR;

    let qweight = read_i32_le(qweight, "qweight")?;
    if qweight.len() as u64 != qweight_rows * out_features {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "qweight has {} i32 elements, expected {qweight_rows}x{out_features}",
                qweight.len()
            ),
        });
    }
    let qzeros = read_i32_le(qzeros, "qzeros")?;
    if !(qzeros.len() as u64).is_multiple_of(qzeros_cols) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "qzeros has {} i32 elements, not a multiple of {qzeros_cols} columns",
                qzeros.len()
            ),
        });
    }
    let num_groups = qzeros.len() as u64 / qzeros_cols;
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
    let g_idx = read_i32_le(g_idx, "g_idx")?;
    if g_idx.len() as u64 != in_features {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "g_idx has {} elements, expected in_features={in_features}",
                g_idx.len()
            ),
        });
    }

    let in_features = in_features as usize;
    let out_features = out_features as usize;
    let mut dequantized = vec![0f32; in_features * out_features];
    for (i, &group) in g_idx.iter().enumerate() {
        let group =
            u64::try_from(group).map_err(|_| ProductionIngestionError::MalformedMetadata {
                reason: format!("g_idx[{i}] is negative"),
            })?;
        if group >= num_groups {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "g_idx[{i}]={group} names a group beyond the {num_groups} declared by qzeros/scales"
                ),
            });
        }
        let row = i as u64 / PACK_FACTOR;
        let shift = 4 * (i as u64 % PACK_FACTOR);
        for o in 0..out_features {
            let packed_weight = qweight[(row * out_features as u64 + o as u64) as usize];
            let quant_code = (packed_weight >> shift) & 0xF;

            let col_block = o as u64 / PACK_FACTOR;
            let zshift = 4 * (o as u64 % PACK_FACTOR);
            let packed_zero = qzeros[(group * qzeros_cols + col_block) as usize];
            let zero_code = (packed_zero >> zshift) & 0xF;

            let scale = scales[(group * out_features as u64 + o as u64) as usize];
            dequantized[i * out_features + o] = scale * ((quant_code - (zero_code + 1)) as f32);
        }
    }

    Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// Where one quantized projection's four sibling raw tensors physically
/// live -- captured before `extract_gptq_projections` removes their raw
/// entries from the discovered tensor inventory, so
/// [`GptqDequantizingPayloadSource`] can still read their real bytes from
/// the underlying file afterward.
#[derive(Debug)]
struct GptqSiblingRanges {
    qweight: ProductionPayloadRange,
    qzeros: ProductionPayloadRange,
    scales: ProductionPayloadRange,
    g_idx: ProductionPayloadRange,
}

/// One GPTQ-quantized projection this ingestor will serve as a single
/// dequantized `F32` logical tensor.
#[derive(Debug)]
pub struct GptqProjection {
    canonical_name: String,
    siblings: GptqSiblingRanges,
    in_features: u64,
    out_features: u64,
}

impl GptqProjection {
    /// The canonical Model Artifact name (e.g. `layers.0.self_attn.
    /// q_proj`) this projection's dequantized bytes are served under --
    /// the same identity `weight_layout.rs`'s transpose exclusion set is
    /// built from, so it must never diverge from what `extract_gptq_
    /// projections` actually placed in the tensor inventory.
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
                reason: format!("GPTQ tensor '{}' has no declared byte range", tensor.name),
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

/// Removes every raw GPTQ-quantized tensor quadruple (`<prefix>.qweight`/
/// `.qzeros`/`.scales`/`.g_idx`) from `tensors`, replacing each with one
/// synthesized placeholder entry named `<prefix>.weight` -- so the
/// existing per-tensor renaming loop in `weights.rs` canonicalizes it
/// exactly like a real plain `.weight` tensor would be -- declaring the
/// projection's real, already Runtime-oriented `[in_features,
/// out_features]` logical shape and `F32` storage dtype. No declared
/// digest or meaningful byte offset: the placeholder describes bytes this
/// ingestor derives, not bytes a real file declares a digest or fixed
/// location for (the same "`None` means derived, not unknown" convention
/// `derived_lm_head.rs`'s synthetic `lm_head` entry already uses).
///
/// Must run *before* the raw-to-canonical tensor-name renaming loop, since
/// `naming::normalize_tensor_name` does not recognize the `.qweight`/
/// `.qzeros`/`.g_idx` suffixes and would reject them.
pub fn extract_gptq_projections(
    tensors: &mut Vec<ModelTensorMetadata>,
) -> Result<Vec<GptqProjection>, ProductionIngestionError> {
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
        let gidx_name = format!("{prefix}.g_idx");

        let find = |name: &str| tensors.iter().find(|tensor| tensor.name == name).cloned();
        let qweight_meta = find(&qweight_name).expect("this crate's own filter found it above");
        let qzeros_meta =
            find(&qzeros_name).ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("GPTQ tensor '{qweight_name}' has no matching '{qzeros_name}'"),
            })?;
        let scales_meta =
            find(&scales_name).ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("GPTQ tensor '{qweight_name}' has no matching '{scales_name}'"),
            })?;
        let gidx_meta =
            find(&gidx_name).ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("GPTQ tensor '{qweight_name}' has no matching '{gidx_name}'"),
            })?;

        let [qweight_rows, out_features] = qweight_meta.shape[..] else {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("'{qweight_name}' does not have a 2D shape"),
            });
        };
        let in_features = gidx_meta.shape.first().copied().ok_or_else(|| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("'{gidx_name}' does not have a 1D shape"),
            }
        })?;
        if qweight_rows.checked_mul(PACK_FACTOR) != Some(in_features) {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!(
                    "'{qweight_name}' packs {qweight_rows} rows for {in_features} input \
                     channels ('{gidx_name}''s own length) -- only 4-bit (8x int32 packing) \
                     GPTQ is supported"
                ),
            });
        }

        let canonical_name = crate::naming::normalize_tensor_name(&format!("{prefix}.weight"))?;
        projections.push(GptqProjection {
            canonical_name,
            siblings: GptqSiblingRanges {
                qweight: payload_range(&qweight_meta)?,
                qzeros: payload_range(&qzeros_meta)?,
                scales: payload_range(&scales_meta)?,
                g_idx: payload_range(&gidx_meta)?,
            },
            in_features,
            out_features,
        });

        tensors.retain(|tensor| {
            tensor.name != qweight_name
                && tensor.name != qzeros_name
                && tensor.name != scales_name
                && tensor.name != gidx_name
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

/// Wraps an inner payload source: a request for a GPTQ-quantized
/// projection's canonical identity (e.g. `layers.0.self_attn.q_proj`) is
/// answered by reading its four raw sibling tensors' real bytes from
/// `inner` and dequantizing them; every other identity passes through
/// unchanged.
#[derive(Debug)]
pub struct GptqDequantizingPayloadSource<S> {
    inner: S,
    projections: BTreeMap<String, GptqProjection>,
}

impl<S> GptqDequantizingPayloadSource<S> {
    pub fn new(inner: S, projections: Vec<GptqProjection>) -> Self {
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
    for GptqDequantizingPayloadSource<S>
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
        let g_idx = self.inner.read_payload(&projection.siblings.g_idx)?;
        dequantize_gptq_projection(
            &qweight,
            &qzeros,
            &scales,
            &g_idx,
            projection.in_features,
            projection.out_features,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_i32(nibbles: &[i32]) -> i32 {
        let mut value: i32 = 0;
        for (index, nibble) in nibbles.iter().enumerate() {
            value |= (nibble & 0xF) << (4 * index);
        }
        value
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f16_bits_from_f32(value: f32) -> u16 {
        // Minimal, test-only f32 -> f16 (round-to-nearest, normal range
        // only -- every scale this test uses is a small positive normal
        // value, so subnormal/inf/NaN handling is not needed here).
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
        // column-block). Group 0 uses zero_code=7 (the common symmetric
        // 4-bit midpoint, zero_code+1=8), group 1 uses zero_code=5.
        let scale_g0 = 0.1f32;
        let scale_g1 = 0.2f32;

        // qweight: [2 rows (16/8), 8 cols] -- one i32 per (row, output
        // column), each packing that column's 8 consecutive input-channel
        // codes. Row 0 -> input channels 0..7 (group 0), row 1 -> input
        // channels 8..15 (group 1). Every column uses the same code
        // pattern [8,9,10,11,12,13,14,15] for simplicity (quant_code -
        // (zero+1) = code - 8 for group 0), so each row is 8 identical
        // packed i32 values, one per output column.
        let packed_row = pack_i32(&[8, 9, 10, 11, 12, 13, 14, 15]);
        let qweight = i32_bytes(&[vec![packed_row; 8], vec![packed_row; 8]].concat());

        // qzeros: [2 groups, 1 col-block (8/8)]. All 8 output channels
        // share zero_code=7 for group 0, zero_code=5 for group 1.
        let qzeros = i32_bytes(&[pack_i32(&[7; 8]), pack_i32(&[5; 8])]);

        // scales: [2 groups, 8 out_features].
        let scales = f16_bytes(
            &[scale_g0; 8]
                .into_iter()
                .chain([scale_g1; 8])
                .collect::<Vec<_>>(),
        );

        // g_idx: [16] -- sequential (desc_act: false).
        let g_idx = i32_bytes(&(0..16).map(|i| i / 8).collect::<Vec<i32>>());

        let dequantized = dequantize_gptq_projection(&qweight, &qzeros, &scales, &g_idx, 16, 8)
            .expect("well-formed GPTQ tensors dequantize");
        let values: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values.len(), 16 * 8);

        // Column 0, input channels 0..7 (group 0, zero+1=8): every column
        // shares the same packed value, so the code varies with the input
        // channel's position within its group of 8, not with the output
        // column: code - 8 = [0,1,2,3,4,5,6,7] * scale_g0.
        for i in 0..8 {
            let expected = scale_g0 * (i as f32);
            let actual = values[i * 8];
            assert!(
                (actual - expected).abs() < 1e-3,
                "input channel {i}, col 0: got {actual}, expected {expected}"
            );
        }
        // Column 0, input channels 8..15 (group 1, zero+1=6): same code
        // pattern (packed value repeats per row), so code - 6 =
        // [2,3,4,5,6,7,8,9] * scale_g1.
        for i in 0..8 {
            let expected = scale_g1 * (2.0 + i as f32);
            let actual = values[(8 + i) * 8];
            assert!(
                (actual - expected).abs() < 1e-3,
                "input channel {}, col 0: got {actual}, expected {expected}",
                8 + i
            );
        }
    }

    #[test]
    fn rejects_a_bit_width_other_than_four() {
        // in_features not divisible by 8 (PACK_FACTOR) is the structural
        // signal this crate uses to reject anything other than real 4-bit
        // packing.
        let error = dequantize_gptq_projection(&[], &[], &[], &[], 12, 8).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }

    #[test]
    fn extract_gptq_projections_replaces_the_four_siblings_with_one_placeholder() {
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
            tensor("model.layers.0.self_attn.q_proj.qweight", vec![112, 896]),
            tensor("model.layers.0.self_attn.q_proj.qzeros", vec![7, 112]),
            tensor("model.layers.0.self_attn.q_proj.scales", vec![7, 896]),
            tensor("model.layers.0.self_attn.q_proj.g_idx", vec![896]),
            tensor("model.norm.weight", vec![896]),
        ];
        let projections = extract_gptq_projections(&mut tensors).expect("extraction succeeds");
        assert_eq!(projections.len(), 1);
        assert_eq!(projections[0].canonical_name, "layers.0.self_attn.q_proj");
        assert_eq!(projections[0].in_features, 896);
        assert_eq!(projections[0].out_features, 896);

        let names: std::collections::BTreeSet<String> =
            tensors.iter().map(|tensor| tensor.name.clone()).collect();
        assert_eq!(
            names,
            std::collections::BTreeSet::from([
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                "model.norm.weight".to_string(),
            ]),
            "the four raw siblings are gone, replaced by one placeholder .weight entry"
        );
        let placeholder = tensors
            .iter()
            .find(|tensor| tensor.name == "model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(placeholder.shape, vec![896, 896]);
        assert_eq!(placeholder.storage_dtype, ModelDType::F32);
    }

    /// Converts one `bfloat16` value to `f32` exactly: `bfloat16` is
    /// simply `f32`'s own top 16 bits (sign + 8-bit exponent + 7-bit
    /// mantissa), so this is a plain bit-shift, not the `f16` conversion
    /// above -- a real distinction real Hugging Face checkpoints care
    /// about (`Qwen/Qwen2.5-0.5B-Instruct`'s own `torch_dtype` is
    /// `bfloat16`).
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// Real-checkpoint verification: dequantizes the actual
    /// `self_attn.v_proj` GPTQ tensors from the public `Qwen/Qwen2.5-0.5B-
    /// Instruct-GPTQ-Int4` checkpoint (downloaded via targeted HTTP Range
    /// requests against the real `model.safetensors` file, not
    /// hand-built) and compares every one of its 896x128 elements against
    /// the same real weight's own unquantized `bfloat16` value from the
    /// public `Qwen/Qwen2.5-0.5B-Instruct` checkpoint -- the strongest
    /// available proof this module's packing/zero-point convention is
    /// correct without re-deriving it from documentation or memory alone.
    /// A wrong zero-point offset (omitting GPTQ's `+ 1`) was caught this
    /// way during development: it reproduces the real weight with a
    /// systematic one-scale-step bias (mean absolute error ~3x higher),
    /// not quantization noise -- this test's tolerance is tight enough to
    /// have caught that, not just "does not crash".
    #[test]
    fn dequantize_gptq_projection_matches_a_real_public_checkpoint() {
        const QWEIGHT: &[u8] =
            include_bytes!("../fixtures/gptq/qwen2.5-0.5b-instruct-gptq-int4.v_proj.qweight.bin");
        const QZEROS: &[u8] =
            include_bytes!("../fixtures/gptq/qwen2.5-0.5b-instruct-gptq-int4.v_proj.qzeros.bin");
        const SCALES: &[u8] =
            include_bytes!("../fixtures/gptq/qwen2.5-0.5b-instruct-gptq-int4.v_proj.scales.bin");
        const G_IDX: &[u8] =
            include_bytes!("../fixtures/gptq/qwen2.5-0.5b-instruct-gptq-int4.v_proj.g_idx.bin");
        const BASE_WEIGHT_BF16: &[u8] =
            include_bytes!("../fixtures/gptq/qwen2.5-0.5b-instruct.v_proj.weight.bf16.bin");

        const IN_FEATURES: u64 = 896;
        const OUT_FEATURES: u64 = 128;

        let dequantized =
            dequantize_gptq_projection(QWEIGHT, QZEROS, SCALES, G_IDX, IN_FEATURES, OUT_FEATURES)
                .expect("the real checkpoint's own GPTQ tensors dequantize");
        let dequantized: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(dequantized.len(), (IN_FEATURES * OUT_FEATURES) as usize);

        // The real unquantized checkpoint stores this weight as `nn.
        // Linear`'s own [out_features, in_features] convention (HF
        // storage, not yet transposed) -- base[o * IN_FEATURES + i].
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
        // Real, measured values (not guessed): the correct `+ 1`
        // zero-point offset gives mean=0.00144/max=0.01395 against this
        // real checkpoint; deliberately removing it during development
        // (the exact bug this test exists to catch) gave
        // mean=0.00479/max=0.02841 -- a clear, systematic ~3.3x/~2x
        // separation, not overlapping noise. These thresholds sit between
        // the two, close enough to the correct values to actually reject
        // that bug class, not just "does not crash".
        assert!(
            mean_abs_error < 0.0025,
            "mean absolute error {mean_abs_error} is too high for correct 4-bit GPTQ dequantization"
        );
        assert!(
            max_abs_error < 0.02,
            "max absolute error {max_abs_error} is too high for correct 4-bit GPTQ dequantization"
        );
    }
}
