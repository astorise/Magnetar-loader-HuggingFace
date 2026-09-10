//! Projection weight transposition: real Hugging Face checkpoints store
//! every 2D projection weight (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
//! `gate_proj`, `up_proj`, `down_proj`, `lm_head`) as `nn.Linear`'s own
//! `[out_features, in_features]` convention (`y = x @ W.T`). The Runtime's
//! Qwen Component instead expects every projection's *logical* shape as
//! `[in_features, out_features]` (`y = x @ W` directly) -- confirmed
//! against `magnetar-runtime`'s own `qwen_expected_tensor_shape` ground
//! truth, and matching the convention its tied-embeddings derivation
//! (`qwen_weights_with_derived_lm_head`/`transpose_rows_cols`) already
//! produces for `lm_head` specifically. `token_embedding` is the one
//! exception: a lookup table, not a projection, stored and expected as
//! `[vocab_size, hidden_size]` in both conventions, so it is never
//! transposed. Found while wiring a real ingested bundle through Model
//! Loading for the first time
//! (`implement-production-qwen-model-loading` task group 10) -- a tiny
//! test config with every dimension equal masked this for `q_proj`/
//! `k_proj`/`v_proj`/`o_proj` (shape `[n, n]` either way), and only a
//! genuinely non-square dimension (`intermediate_size != hidden_size`)
//! exposed it for `gate_proj`/`up_proj`/`down_proj`.

use magnetar_runtime::model::ModelTensorMetadata;
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::collections::BTreeMap;

/// Whether `canonical_name` (already normalized -- see `naming.rs`) names
/// a projection weight this Runtime expects transposed relative to how
/// Hugging Face stores it. Matches `lm_head` exactly and every
/// `*.q_proj`/`*.k_proj`/`*.v_proj`/`*.o_proj`/`*.gate_proj`/`*.up_proj`/
/// `*.down_proj` name via the shared `proj` substring -- deliberately not
/// `token_embedding` (a lookup table) or the 1D normalization vectors
/// (`input_norm`/`post_attn_norm`/`final_norm`, transposition is
/// meaningless for a 1D shape and they are excluded structurally by the
/// two-dimensional check in [`swap_declared_projection_shapes`]/
/// [`TransposingPayloadSource`] regardless).
fn is_transposed_projection(canonical_name: &str) -> bool {
    canonical_name == "lm_head" || canonical_name.contains("proj")
}

/// Swaps every 2D projection tensor's declared shape from
/// `[out_features, in_features]` (as stored) to `[in_features,
/// out_features]` (as the Component expects), in place, and clears its
/// declared content digest.
///
/// The digest matters: `magnetar-runtime`'s own weight-materialization
/// path re-verifies a declared tensor digest against the *materialized*
/// content once storage-dtype conversion has happened (Decision 8's
/// "verify source-content digest -> checked decode/convert -> F32
/// staging tensor" pattern), skipping that re-check only when it can tell
/// a conversion changed the bytes. It has no way to know this ingestor
/// transposed a projection's bytes -- a Runtime-invisible detail of this
/// external crate -- so a still-declared digest computed over the
/// *pre-transpose* bytes would be checked against the transposed,
/// materialized content and always fail. `TransposingPayloadSource`
/// already verifies the declared digest against the real, untransposed
/// source bytes at read time (bounded-payload-access's own integrity
/// check, at the correct boundary); clearing it here means Runtime's
/// separate, later check is skipped rather than wrongly failing --
/// consistent with `ModelTensorMetadata::digest`'s own "`None` means no
/// digest was declared" precedent, not "no content required".
pub fn swap_declared_projection_shapes(tensors: &mut [ModelTensorMetadata]) {
    for tensor in tensors.iter_mut() {
        if is_transposed_projection(&tensor.name) && tensor.shape.len() == 2 {
            tensor.shape.swap(0, 1);
            tensor.digest = None;
        }
    }
}

/// A transposed projection's shape and real per-element storage width
/// (bytes), captured *before* [`swap_declared_projection_shapes`] runs.
/// The element width matters: real Hugging Face checkpoints commonly
/// store weights as F16/BF16 (2 bytes), not only F32 (4 bytes) -- a
/// transpose that assumed F32 unconditionally would misread every real
/// half-precision checkpoint's byte layout (found running a real
/// `torch_dtype: "bfloat16"` Qwen2.5 checkpoint through this ingestor for
/// the first time, task 12.4).
struct TransposedTensorShape {
    out_features: u64,
    in_features: u64,
    element_bytes: u64,
}

/// Wraps an inner payload source, transposing the bytes returned for
/// every projection tensor identity in `original_shapes` (row-major
/// `[out_features, in_features]` -> `[in_features, out_features]`,
/// preserving each element's own byte width) and passing every other
/// identity through unchanged. `original_shapes` holds each transposed
/// tensor's shape *before* swapping (as captured from real discovery,
/// before [`swap_declared_projection_shapes`] runs).
pub struct TransposingPayloadSource<S> {
    inner: S,
    original_shapes: BTreeMap<String, TransposedTensorShape>,
}

impl<S> TransposingPayloadSource<S> {
    /// Builds the wrapper from `tensors`' state *before*
    /// [`swap_declared_projection_shapes`] is applied to them -- call this
    /// first, capture the map, then swap the caller's own tensor list
    /// separately.
    pub fn new(inner: S, tensors: &[ModelTensorMetadata]) -> Self {
        let original_shapes = tensors
            .iter()
            .filter(|tensor| is_transposed_projection(&tensor.name) && tensor.shape.len() == 2)
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    TransposedTensorShape {
                        out_features: tensor.shape[0],
                        in_features: tensor.shape[1],
                        element_bytes: tensor.storage_dtype.descriptor().size_bytes(),
                    },
                )
            })
            .collect();
        Self {
            inner,
            original_shapes,
        }
    }
}

impl<S: ProductionArtifactPayloadSource> ProductionArtifactPayloadSource
    for TransposingPayloadSource<S>
{
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let bytes = self.inner.read_payload(range)?;
        let Some(shape) = self.original_shapes.get(&range.identity) else {
            return Ok(bytes);
        };
        let element_count = shape
            .out_features
            .checked_mul(shape.in_features)
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("'{}' element count overflows", range.identity),
            })?;
        if bytes.len() as u64 != element_count.saturating_mul(shape.element_bytes) {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "'{}' byte length {} does not match {}x{} elements at {} bytes/element",
                    range.identity,
                    bytes.len(),
                    shape.out_features,
                    shape.in_features,
                    shape.element_bytes
                ),
            });
        }
        let rows = shape.out_features as usize;
        let cols = shape.in_features as usize;
        let element_bytes = shape.element_bytes as usize;
        let mut transposed = vec![0u8; bytes.len()];
        for row in 0..rows {
            for col in 0..cols {
                let source_offset = (row * cols + col) * element_bytes;
                let dest_offset = (col * rows + row) * element_bytes;
                transposed[dest_offset..dest_offset + element_bytes]
                    .copy_from_slice(&bytes[source_offset..source_offset + element_bytes]);
            }
        }
        Ok(transposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::model::ModelDType;
    use std::collections::BTreeMap as StdBTreeMap;

    struct FixedPayloadSource(StdBTreeMap<String, Vec<u8>>);

    impl ProductionArtifactPayloadSource for FixedPayloadSource {
        fn read_payload(
            &self,
            range: &ProductionPayloadRange,
        ) -> Result<Vec<u8>, ProductionIngestionError> {
            self.0.get(&range.identity).cloned().ok_or_else(|| {
                ProductionIngestionError::PayloadOutOfBounds {
                    identity: range.identity.clone(),
                }
            })
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn tensor(name: &str, shape: Vec<u64>) -> ModelTensorMetadata {
        ModelTensorMetadata {
            name: name.to_string(),
            shape,
            storage_dtype: ModelDType::F32,
            layout: None,
            shard: None,
            offset_bytes: Some(0),
            size_bytes: Some(0),
            quantization: None,
            expected_compute_dtype: None,
            digest: None,
        }
    }

    #[test]
    fn identifies_every_real_projection_name_and_excludes_the_rest() {
        for name in [
            "lm_head",
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.k_proj",
            "layers.0.self_attn.v_proj",
            "layers.0.self_attn.o_proj",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.down_proj",
        ] {
            assert!(is_transposed_projection(name), "{name} should transpose");
        }
        for name in [
            "token_embedding",
            "final_norm",
            "layers.0.input_norm",
            "layers.0.post_attn_norm",
        ] {
            assert!(
                !is_transposed_projection(name),
                "{name} should not transpose"
            );
        }
    }

    #[test]
    fn swaps_declared_shapes_for_projections_only() {
        let with_digest = |mut t: ModelTensorMetadata| {
            t.digest = Some(magnetar_runtime::model::ModelDigest::sha256(b"probe"));
            t
        };
        let mut tensors = vec![
            with_digest(tensor("lm_head", vec![16, 4])),
            with_digest(tensor("layers.0.mlp.down_proj", vec![4, 8])),
            with_digest(tensor("token_embedding", vec![16, 4])),
            with_digest(tensor("final_norm", vec![4])),
        ];
        swap_declared_projection_shapes(&mut tensors);
        assert_eq!(tensors[0].shape, vec![4, 16]);
        assert!(tensors[0].digest.is_none(), "lm_head's digest is cleared");
        assert_eq!(tensors[1].shape, vec![8, 4]);
        assert!(tensors[1].digest.is_none(), "down_proj's digest is cleared");
        assert_eq!(
            tensors[2].shape,
            vec![16, 4],
            "token_embedding is unchanged"
        );
        assert!(
            tensors[2].digest.is_some(),
            "token_embedding's digest is preserved -- it is never transposed"
        );
        assert_eq!(tensors[3].shape, vec![4], "1D norm vector is unchanged");
        assert!(
            tensors[3].digest.is_some(),
            "final_norm's digest is preserved -- it is never transposed"
        );
    }

    #[test]
    fn transposes_projection_bytes_row_major() {
        // down_proj stored [out=4, in=8] (HF convention):
        // [[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16],[17..24],[25..32]]
        let values: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let raw = f32_bytes(&values);
        let original = vec![tensor("layers.0.mlp.down_proj", vec![4, 8])];
        let mut bytes_by_name = StdBTreeMap::new();
        bytes_by_name.insert("layers.0.mlp.down_proj".to_string(), raw.clone());
        let source = TransposingPayloadSource::new(FixedPayloadSource(bytes_by_name), &original);

        let transposed = source
            .read_payload(&ProductionPayloadRange {
                identity: "layers.0.mlp.down_proj".into(),
                offset: 0,
                length: raw.len() as u64,
                digest: None,
            })
            .unwrap();
        let out_values: Vec<f32> = transposed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        // Result is [in=8, out=4] row-major: row 0 is column 0 of the
        // original, i.e. [1, 9, 17, 25].
        assert_eq!(&out_values[0..4], &[1.0, 9.0, 17.0, 25.0]);
        assert_eq!(&out_values[4..8], &[2.0, 10.0, 18.0, 26.0]);
    }

    #[test]
    fn passes_through_token_embedding_unchanged() {
        let raw = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let original = vec![tensor("layers.0.mlp.down_proj", vec![4, 8])];
        let mut bytes_by_name = StdBTreeMap::new();
        bytes_by_name.insert("token_embedding".to_string(), raw.clone());
        let source = TransposingPayloadSource::new(FixedPayloadSource(bytes_by_name), &original);
        let unchanged = source
            .read_payload(&ProductionPayloadRange {
                identity: "token_embedding".into(),
                offset: 0,
                length: raw.len() as u64,
                digest: None,
            })
            .unwrap();
        assert_eq!(unchanged, raw);
    }

    #[test]
    fn rejects_a_byte_length_mismatch() {
        let original = vec![tensor("lm_head", vec![2, 3])];
        let mut bytes_by_name = StdBTreeMap::new();
        bytes_by_name.insert("lm_head".to_string(), f32_bytes(&[1.0, 2.0]));
        let source = TransposingPayloadSource::new(FixedPayloadSource(bytes_by_name), &original);
        let error = source
            .read_payload(&ProductionPayloadRange {
                identity: "lm_head".into(),
                offset: 0,
                length: 8,
                digest: None,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }
}
