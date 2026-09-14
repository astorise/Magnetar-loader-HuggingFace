//! Real BitsAndBytes 4-bit weight dequantization (`quant_method:
//! "bitsandbytes"`, `NF4`/`FP4`, with or without double quantization) --
//! `transformers`' modern `Params4bit` safetensors export shape.
//!
//! Unlike GPTQ (`gptq.rs`) and AWQ (`awq.rs`), a BitsAndBytes-quantized
//! projection's packed codes are stored under the exact same raw name a
//! plain, unquantized weight would use: `<prefix>.weight`. What marks it
//! as quantized is a set of *sibling* tensors sharing that same
//! `<prefix>.weight` name as their own prefix: `<prefix>.weight.absmax`
//! (per-block scale, itself quantized to `U8` when double quantization is
//! enabled), `<prefix>.weight.quant_map` (the real, non-uniform 16-value
//! NF4/FP4 codebook -- unlike GPTQ/AWQ's simple `scale * code` linear
//! dequantization, BitsAndBytes' codebook values are themselves real,
//! quantile-derived floats), and (only when double-quantized)
//! `<prefix>.weight.nested_absmax`/`<prefix>.weight.nested_quant_map` (a
//! second quantization level dequantizing `absmax` itself), plus a real
//! JSON blob at `<prefix>.weight.quant_state.bitsandbytes__nf4` carrying
//! the block sizes, the real logical shape, and (for double quantization)
//! a scalar offset -- values this ingestor cannot derive from any
//! tensor's own declared shape, unlike GPTQ/AWQ's ingestion-time
//! extraction, which never needs to *read* file bytes ahead of the
//! Runtime's own later payload access.
//!
//! Verified against real bytes downloaded from the public `unsloth/
//! Qwen2.5-0.5B-Instruct-bnb-4bit` checkpoint (`self_attn.v_proj`,
//! double-quantized `nf4`, `blocksize: 64`, `nested_blocksize: 256`),
//! compared element-by-element against the same real weight's own
//! unquantized value from `Qwen/Qwen2.5-0.5B-Instruct` -- confirming both
//! the packed-nibble order (high nibble is the even-indexed element) and
//! the full double-dequantization formula empirically: the correct order
//! reproduces the real unquantized weight with roughly 1/14th the mean
//! absolute error of the other nibble order, a systematic misordering,
//! not quantization noise.
//!
//! Unlike GPTQ/AWQ's already-`[in_features, out_features]`-oriented
//! dequantized output, BitsAndBytes' packed bytes reconstruct directly to
//! the real logical shape the quant-state JSON declares --
//! `[out_features, in_features]`, `nn.Linear`'s own storage convention --
//! so a BitsAndBytes-dequantized projection SHALL still go through this
//! crate's existing projection-weight transpose, unlike a GPTQ/AWQ one.

use magnetar_runtime::model::{ModelDType, ModelTensorMetadata};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use serde::Deserialize;
use std::collections::BTreeMap;

const CODEBOOK_SIZE: usize = 16;

/// The real JSON state BitsAndBytes/`transformers` serializes at
/// `<prefix>.weight.quant_state.bitsandbytes__nf4` -- confirmed against a
/// real downloaded blob:
/// `{"quant_type": "nf4", "blocksize": 64, "dtype": "bfloat16", "shape":
/// [128, 896], "nested_blocksize": 256, "nested_dtype": "float32",
/// "nested_offset": 0.031807661056518555}`. Only the fields this module
/// actually needs are modeled; `dtype`/`nested_dtype` are real but unused
/// (dequantization always produces `F32` regardless of the original
/// compute dtype).
#[derive(Debug, Deserialize)]
struct BnbQuantState {
    blocksize: u64,
    shape: Vec<u64>,
    #[serde(default)]
    nested_blocksize: Option<u64>,
    #[serde(default)]
    nested_offset: Option<f64>,
}

fn read_f32_le(bytes: &[u8], reason: &str) -> Result<Vec<f32>, ProductionIngestionError> {
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
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// Dequantizes one BitsAndBytes 4-bit-quantized projection into real
/// `F32` bytes, logical shape `[out_features, in_features]` row-major --
/// `nn.Linear`'s own storage convention, matching what the real
/// quant-state JSON's own `shape` field declares (see this module's own
/// doc comment for why this, unlike GPTQ/AWQ, still needs this crate's
/// existing transpose).
///
/// `nested` is `Some((nested_absmax_bytes, nested_quant_map_bytes,
/// nested_offset, nested_blocksize))` for a double-quantized `absmax`
/// (`absmax_bytes` holds `U8` quantized codes in that case), or `None` for
/// a single-level `absmax` (`absmax_bytes` holds real `F32` values
/// directly).
#[allow(clippy::too_many_arguments)]
pub fn dequantize_bnb_projection(
    weight: &[u8],
    absmax_bytes: &[u8],
    quant_map: &[u8],
    nested: Option<(&[u8], &[u8], f32, u64)>,
    blocksize: u64,
    out_features: u64,
    in_features: u64,
) -> Result<Vec<u8>, ProductionIngestionError> {
    let quant_map = read_f32_le(quant_map, "quant_map")?;
    if quant_map.len() != CODEBOOK_SIZE {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "BitsAndBytes dequantization only supports a {CODEBOOK_SIZE}-value codebook \
                 (4-bit NF4/FP4); quant_map declared {} values",
                quant_map.len()
            ),
        });
    }

    let total_elements = out_features.checked_mul(in_features).ok_or_else(|| {
        ProductionIngestionError::MalformedMetadata {
            reason: "BitsAndBytes projection element count overflows".into(),
        }
    })?;
    if blocksize == 0 || !total_elements.is_multiple_of(blocksize) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "total element count {total_elements} is not evenly divisible by \
                 blocksize={blocksize}"
            ),
        });
    }
    let num_blocks = total_elements / blocksize;

    let absmax_real: Vec<f32> = match nested {
        Some((nested_absmax, nested_quant_map, nested_offset, nested_blocksize)) => {
            let nested_quant_map = read_f32_le(nested_quant_map, "nested_quant_map")?;
            if nested_quant_map.len() != 256 {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "nested_quant_map declared {} values, expected 256 (8-bit double \
                         quantization)",
                        nested_quant_map.len()
                    ),
                });
            }
            let nested_absmax = read_f32_le(nested_absmax, "nested_absmax")?;
            if absmax_bytes.len() as u64 != num_blocks {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "absmax has {} u8 elements, expected {num_blocks} blocks",
                        absmax_bytes.len()
                    ),
                });
            }
            if nested_blocksize == 0 {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: "nested_blocksize must not be zero".into(),
                });
            }
            absmax_bytes
                .iter()
                .enumerate()
                .map(|(block, &code)| {
                    let nested_block = block as u64 / nested_blocksize;
                    nested_quant_map[code as usize]
                        * nested_absmax
                            .get(nested_block as usize)
                            .copied()
                            .unwrap_or(0.0)
                        + nested_offset
                })
                .collect()
        }
        None => {
            let absmax_real = read_f32_le(absmax_bytes, "absmax")?;
            if absmax_real.len() as u64 != num_blocks {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "absmax has {} f32 elements, expected {num_blocks} blocks",
                        absmax_real.len()
                    ),
                });
            }
            absmax_real
        }
    };

    let expected_packed_bytes = total_elements.div_ceil(2);
    if weight.len() as u64 != expected_packed_bytes {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "weight has {} packed bytes, expected {expected_packed_bytes} for \
                 {total_elements} 4-bit-packed elements",
                weight.len()
            ),
        });
    }

    let blocksize = blocksize as usize;
    let mut dequantized = vec![0f32; total_elements as usize];
    for (element_index, value) in dequantized.iter_mut().enumerate() {
        let byte = weight[element_index / 2];
        let code = if element_index.is_multiple_of(2) {
            (byte >> 4) & 0xF
        } else {
            byte & 0xF
        };
        let block = element_index / blocksize;
        *value = quant_map[code as usize] * absmax_real[block];
    }

    Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// Where one quantized projection's sibling raw tensors physically live
/// -- captured before `extract_bnb_projections` removes their raw entries
/// from the discovered tensor inventory, so
/// [`BnbDequantizingPayloadSource`] can still read their real bytes from
/// the underlying file afterward. `weight` is the packed-codes tensor's
/// own *original* range (before this extractor overwrites its declared
/// shape/dtype in place) -- its declared byte length no longer matches
/// the corrected, dequantized `size_bytes` a caller requests, so it must
/// be remembered separately, the same reason every sibling range is.
#[derive(Debug)]
struct BnbSiblingRanges {
    weight: ProductionPayloadRange,
    absmax: ProductionPayloadRange,
    quant_map: ProductionPayloadRange,
    nested: Option<BnbNestedRanges>,
}

#[derive(Debug)]
struct BnbNestedRanges {
    nested_absmax: ProductionPayloadRange,
    nested_quant_map: ProductionPayloadRange,
    nested_offset: f32,
    nested_blocksize: u64,
}

/// One BitsAndBytes-quantized projection this ingestor will serve as a
/// single dequantized `F32` logical tensor.
#[derive(Debug)]
pub struct BnbProjection {
    canonical_name: String,
    siblings: BnbSiblingRanges,
    blocksize: u64,
    out_features: u64,
    in_features: u64,
}

fn payload_range(
    tensor: &ModelTensorMetadata,
) -> Result<ProductionPayloadRange, ProductionIngestionError> {
    let (offset, length) = match (tensor.offset_bytes, tensor.size_bytes) {
        (Some(offset), Some(length)) => (offset, length),
        _ => {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "BitsAndBytes tensor '{}' has no declared byte range",
                    tensor.name
                ),
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

/// Finds every raw BitsAndBytes-quantized projection (a `<prefix>.weight`
/// tensor with a `<prefix>.weight.absmax` sibling -- GPTQ's/AWQ's own
/// `.qweight`-suffixed tensors never collide with this, since neither
/// scheme keeps the literal `.weight` name), reads its tiny (~100-200
/// byte) real `quant_state` JSON blob via `payload_source` to learn its
/// real logical shape and block sizes (the one piece of real information
/// this format never exposes through any tensor's own declared `shape`),
/// and *mutates* that `<prefix>.weight` entry's declared shape/storage
/// dtype in place to the real, dequantized `[out_features, in_features]`
/// `F32` reality -- correcting, not replacing, since a BitsAndBytes
/// projection's dequantized bytes are served under the exact same raw
/// name its packed codes already occupy. Every other sibling tensor
/// (`.absmax`, `.quant_map`, `.nested_absmax`, `.nested_quant_map`,
/// `.quant_state.*`) is removed from `tensors` entirely.
///
/// Must run *before* the raw-to-canonical tensor-name renaming loop
/// (matching `gptq.rs`/`awq.rs`'s own requirement), even though
/// `<prefix>.weight` itself needs no special renaming treatment -- its
/// sibling suffixes (`.weight.absmax`, ...) are not `.weight`/`.bias`
/// tensor names `naming::normalize_tensor_name` recognizes.
pub fn extract_bnb_projections(
    tensors: &mut Vec<ModelTensorMetadata>,
    payload_source: &dyn ProductionArtifactPayloadSource,
) -> Result<Vec<BnbProjection>, ProductionIngestionError> {
    let weight_names: Vec<String> = tensors
        .iter()
        .filter(|tensor| {
            tensor.name.ends_with(".weight")
                && tensors
                    .iter()
                    .any(|other| other.name == format!("{}.absmax", tensor.name))
        })
        .map(|tensor| tensor.name.clone())
        .collect();

    let mut projections = Vec::with_capacity(weight_names.len());
    for weight_name in weight_names {
        let absmax_name = format!("{weight_name}.absmax");
        let quant_map_name = format!("{weight_name}.quant_map");
        let quant_state_name = format!("{weight_name}.quant_state.bitsandbytes__nf4");
        let nested_absmax_name = format!("{weight_name}.nested_absmax");
        let nested_quant_map_name = format!("{weight_name}.nested_quant_map");

        let find = |name: &str| tensors.iter().find(|tensor| tensor.name == name).cloned();
        let weight_meta = find(&weight_name).expect("this crate's own filter found it above");
        let absmax_meta = find(&absmax_name).expect("this crate's own filter found it above");
        let quant_map_meta =
            find(&quant_map_name).ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "BitsAndBytes tensor '{weight_name}' has no matching '{quant_map_name}'"
                ),
            })?;
        let quant_state_meta =
            find(&quant_state_name).ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "BitsAndBytes tensor '{weight_name}' has no matching '{quant_state_name}'"
                ),
            })?;

        let quant_state_range = payload_range(&quant_state_meta)?;
        let quant_state_bytes = payload_source.read_payload(&quant_state_range)?;
        let quant_state: BnbQuantState =
            serde_json::from_slice(&quant_state_bytes).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("'{quant_state_name}' is not the expected JSON shape: {error}"),
                }
            })?;
        let [out_features, in_features] = quant_state.shape[..] else {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("'{quant_state_name}' declares a non-2D shape"),
            });
        };

        let nested_absmax_meta = find(&nested_absmax_name);
        let nested_quant_map_meta = find(&nested_quant_map_name);
        let nested = match (
            nested_absmax_meta,
            nested_quant_map_meta,
            quant_state.nested_blocksize,
            quant_state.nested_offset,
        ) {
            (
                Some(nested_absmax_meta),
                Some(nested_quant_map_meta),
                Some(nested_blocksize),
                Some(nested_offset),
            ) => Some(BnbNestedRanges {
                nested_absmax: payload_range(&nested_absmax_meta)?,
                nested_quant_map: payload_range(&nested_quant_map_meta)?,
                nested_offset: nested_offset as f32,
                nested_blocksize,
            }),
            (None, None, None, None) => None,
            _ => {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "'{weight_name}' declares an inconsistent mix of double-quantization \
                         fields (nested_absmax/nested_quant_map tensors, nested_blocksize/\
                         nested_offset in the quant_state JSON) -- all four or none"
                    ),
                });
            }
        };

        let canonical_name = crate::naming::normalize_tensor_name(&weight_name)?;
        // The packed weight tensor's own raw name flows through the same
        // renaming loop `weights.rs` runs for every other tensor (it is
        // never removed from `tensors`, only corrected in place below),
        // so by the time this range is actually read, its location in
        // the underlying payload source has already been relocated from
        // its raw key to this canonical one -- unlike every sibling
        // range below, which keeps its raw identity because its own
        // entry is removed from `tensors` before that loop ever runs.
        let mut weight_range = payload_range(&weight_meta)?;
        weight_range.identity = canonical_name.clone();
        projections.push(BnbProjection {
            canonical_name,
            siblings: BnbSiblingRanges {
                weight: weight_range,
                absmax: payload_range(&absmax_meta)?,
                quant_map: payload_range(&quant_map_meta)?,
                nested,
            },
            blocksize: quant_state.blocksize,
            out_features,
            in_features,
        });

        tensors.retain(|tensor| {
            tensor.name != absmax_name
                && tensor.name != quant_map_name
                && tensor.name != quant_state_name
                && tensor.name != nested_absmax_name
                && tensor.name != nested_quant_map_name
        });
        let weight_tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == weight_name)
            .expect("the packed weight tensor is still present -- only its siblings were removed");
        weight_tensor.shape = vec![out_features, in_features];
        weight_tensor.storage_dtype = ModelDType::F32;
        weight_tensor.size_bytes = Some(out_features * in_features * 4);
        weight_tensor.quantization = None;
        weight_tensor.digest = None;
    }
    Ok(projections)
}

/// Wraps an inner payload source: a request for a BitsAndBytes-quantized
/// projection's canonical identity is answered by reading its packed
/// weight bytes and sibling scale/codebook tensors' real bytes from
/// `inner` and dequantizing them; every other identity passes through
/// unchanged.
#[derive(Debug)]
pub struct BnbDequantizingPayloadSource<S> {
    inner: S,
    projections: BTreeMap<String, BnbProjection>,
}

impl<S> BnbDequantizingPayloadSource<S> {
    pub fn new(inner: S, projections: Vec<BnbProjection>) -> Self {
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
    for BnbDequantizingPayloadSource<S>
{
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let Some(projection) = self.projections.get(&range.identity) else {
            return self.inner.read_payload(range);
        };
        let weight = self.inner.read_payload(&projection.siblings.weight)?;
        let absmax = self.inner.read_payload(&projection.siblings.absmax)?;
        let quant_map = self.inner.read_payload(&projection.siblings.quant_map)?;
        let (nested_absmax, nested_quant_map);
        let nested = match &projection.siblings.nested {
            Some(nested_ranges) => {
                nested_absmax = self.inner.read_payload(&nested_ranges.nested_absmax)?;
                nested_quant_map = self.inner.read_payload(&nested_ranges.nested_quant_map)?;
                Some((
                    nested_absmax.as_slice(),
                    nested_quant_map.as_slice(),
                    nested_ranges.nested_offset,
                    nested_ranges.nested_blocksize,
                ))
            }
            None => None,
        };
        dequantize_bnb_projection(
            &weight,
            &absmax,
            &quant_map,
            nested,
            projection.blocksize,
            projection.out_features,
            projection.in_features,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn dequantizes_a_hand_built_single_level_projection() {
        // No double quantization: absmax is real F32 directly. 4 elements
        // (out_features=2, in_features=2), blocksize=2 (2 blocks), a
        // trivial 16-value identity-ish codebook (quant_map[c] = c as
        // f32, so dequantized values are simply block_scale * code).
        let quant_map: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let absmax = vec![0.5f32, 2.0f32];
        // element 0 -> code 3 (high nibble of byte 0), element 1 -> code
        // 5 (low nibble of byte 0), element 2 -> code 1 (high nibble of
        // byte 1), element 3 -> code 2 (low nibble of byte 1).
        let weight = vec![(3u8 << 4) | 5u8, (1u8 << 4) | 2u8];

        let dequantized = dequantize_bnb_projection(
            &weight,
            &f32_bytes(&absmax),
            &f32_bytes(&quant_map),
            None,
            2,
            2,
            2,
        )
        .expect("well-formed single-level BnB tensors dequantize");
        let values: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values, vec![3.0 * 0.5, 5.0 * 0.5, 1.0 * 2.0, 2.0 * 2.0]);
    }

    #[test]
    fn dequantizes_a_hand_built_double_quantized_projection() {
        // 4 elements, blocksize=2 -> 2 first-level blocks; nested_blocksize=2
        // -> both first-level blocks share one nested block. quant_map is
        // the identity-ish codebook again; nested_quant_map maps u8 code 10
        // -> 100.0 and 20 -> 200.0. nested_absmax=[10.0] (one nested block),
        // nested_offset=1.0.
        let quant_map: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let mut nested_quant_map = vec![0.0f32; 256];
        nested_quant_map[10] = 100.0;
        nested_quant_map[20] = 200.0;
        let nested_absmax = vec![10.0f32];
        let absmax_u8 = vec![10u8, 20u8]; // block 0 -> code 10, block 1 -> code 20
        // absmax_real[0] = 100.0*10.0 + 1.0 = 1001.0
        // absmax_real[1] = 200.0*10.0 + 1.0 = 2001.0
        let weight = vec![(1u8 << 4) | 1u8, (1u8 << 4) | 1u8]; // all code 1

        let dequantized = dequantize_bnb_projection(
            &weight,
            &absmax_u8,
            &f32_bytes(&quant_map),
            Some((
                &f32_bytes(&nested_absmax),
                &f32_bytes(&nested_quant_map),
                1.0,
                2,
            )),
            2,
            2,
            2,
        )
        .expect("well-formed double-quantized BnB tensors dequantize");
        let values: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values, vec![1001.0, 1001.0, 2001.0, 2001.0]);
    }

    #[test]
    fn rejects_a_codebook_that_is_not_sixteen_values() {
        let error =
            dequantize_bnb_projection(&[], &[], &f32_bytes(&[0.0; 8]), None, 2, 2, 2).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }

    /// Real-checkpoint verification: dequantizes the actual
    /// `self_attn.v_proj` double-quantized NF4 tensors from the public
    /// `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` checkpoint (downloaded via
    /// targeted HTTP Range requests, not hand-built) and compares every
    /// one of its 128x896 elements against the same real weight's own
    /// unquantized `bfloat16` value from the public `Qwen/Qwen2.5-0.5B-
    /// Instruct` checkpoint. The wrong nibble order (low nibble first)
    /// was tested during development and reproduces the real weight with
    /// ~14.7x this test's mean error -- this test's tolerance is tight
    /// enough to have caught that, not just "does not crash".
    #[test]
    fn dequantize_bnb_projection_matches_a_real_public_checkpoint() {
        const WEIGHT: &[u8] = include_bytes!(
            "../fixtures/bnb/qwen2.5-0.5b-instruct-unsloth-bnb-4bit.v_proj.weight.bin"
        );
        const ABSMAX: &[u8] = include_bytes!(
            "../fixtures/bnb/qwen2.5-0.5b-instruct-unsloth-bnb-4bit.v_proj.weight.absmax.bin"
        );
        const NESTED_ABSMAX: &[u8] = include_bytes!(
            "../fixtures/bnb/qwen2.5-0.5b-instruct-unsloth-bnb-4bit.v_proj.weight.nested_absmax.bin"
        );
        const NESTED_QUANT_MAP: &[u8] = include_bytes!(
            "../fixtures/bnb/qwen2.5-0.5b-instruct-unsloth-bnb-4bit.v_proj.weight.nested_quant_map.bin"
        );
        const QUANT_MAP: &[u8] = include_bytes!(
            "../fixtures/bnb/qwen2.5-0.5b-instruct-unsloth-bnb-4bit.v_proj.weight.quant_map.bin"
        );
        const BASE_WEIGHT_BF16: &[u8] =
            include_bytes!("../fixtures/bnb/qwen2.5-0.5b-instruct.v_proj.weight.bf16.bin");

        const OUT_FEATURES: u64 = 128;
        const IN_FEATURES: u64 = 896;
        const BLOCKSIZE: u64 = 64;
        const NESTED_BLOCKSIZE: u64 = 256;
        const NESTED_OFFSET: f32 = 0.031_807_66;

        let dequantized = dequantize_bnb_projection(
            WEIGHT,
            ABSMAX,
            QUANT_MAP,
            Some((
                NESTED_ABSMAX,
                NESTED_QUANT_MAP,
                NESTED_OFFSET,
                NESTED_BLOCKSIZE,
            )),
            BLOCKSIZE,
            OUT_FEATURES,
            IN_FEATURES,
        )
        .expect("the real checkpoint's own BnB tensors dequantize");
        let dequantized: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(dequantized.len(), (OUT_FEATURES * IN_FEATURES) as usize);

        let base: Vec<f32> = BASE_WEIGHT_BF16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
            .collect();
        assert_eq!(base.len(), (OUT_FEATURES * IN_FEATURES) as usize);

        // Row-major [out_features, in_features]: both arrays already
        // share this exact orientation (BnB's real shape, and HF's own
        // storage convention for the base checkpoint) -- direct
        // element-by-element comparison, no transpose needed.
        let mut sum_abs_error = 0f64;
        let mut max_abs_error = 0f32;
        for (dequant_value, base_value) in dequantized.iter().zip(base.iter()) {
            let error = (dequant_value - base_value).abs();
            sum_abs_error += error as f64;
            max_abs_error = max_abs_error.max(error);
        }
        let mean_abs_error = sum_abs_error / (OUT_FEATURES * IN_FEATURES) as f64;
        // Real, measured values (not guessed): the correct nibble order
        // gives mean=0.00085/max=0.01095 against this real checkpoint;
        // the wrong order gives mean=0.01246 -- a clear, systematic
        // ~14.7x separation, not overlapping noise.
        assert!(
            mean_abs_error < 0.003,
            "mean absolute error {mean_abs_error} is too high for correct BnB dequantization"
        );
        assert!(
            max_abs_error < 0.03,
            "max absolute error {max_abs_error} is too high for correct BnB dequantization"
        );
    }
}
