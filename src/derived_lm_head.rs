//! Tied-embedding `lm_head` derivation: a real Hugging Face checkpoint
//! with `tie_word_embeddings: true` genuinely omits `lm_head.weight`
//! entirely (the model reuses `token_embedding` for the output
//! projection). `magnetar-runtime`'s own fixture loading path already
//! derives a tied model's `lm_head` from `token_embedding` at load time
//! (`qwen_weights_with_derived_lm_head`/`transpose_rows_cols`); this is
//! the same derivation for the streaming production path, where no
//! whole-model weight map exists to mutate in place -- a synthetic
//! `lm_head` tensor entry is added to the manifest, and a payload-source
//! wrapper answers a request for it by reading `token_embedding`'s real
//! bytes and transposing them, the same row-major transpose
//! `weight_layout.rs` already implements for real, stored projection
//! weights (`implement-production-qwen-model-loading` task 10.5).

use magnetar_runtime::model::ModelTensorMetadata;
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};

const LM_HEAD: &str = "lm_head";
const TOKEN_EMBEDDING: &str = "token_embedding";

/// If `tensors` has a `token_embedding` entry, no `lm_head` entry, and
/// `tied`, appends a synthetic `lm_head` tensor with `token_embedding`'s
/// shape swapped (`[vocab_size, hidden_size]` -> `[hidden_size,
/// vocab_size]`, matching the Runtime's expected logical shape for every
/// projection weight -- see `weight_layout.rs`). No declared digest: it
/// describes bytes this ingestor derives, not bytes a real file declared
/// a digest for. Returns `true` if a synthetic tensor was appended.
pub fn append_synthetic_lm_head_if_tied(
    tensors: &mut Vec<ModelTensorMetadata>,
    tied: bool,
) -> bool {
    if !tied || tensors.iter().any(|tensor| tensor.name == LM_HEAD) {
        return false;
    }
    let Some(token_embedding) = tensors.iter().find(|tensor| tensor.name == TOKEN_EMBEDDING) else {
        return false;
    };
    if token_embedding.shape.len() != 2 {
        return false;
    }
    let mut lm_head = token_embedding.clone();
    lm_head.name = LM_HEAD.to_string();
    lm_head.shape.swap(0, 1);
    lm_head.digest = None;
    tensors.push(lm_head);
    true
}

/// Wraps an inner payload source: a request for the `lm_head` identity is
/// answered by reading `token_embedding`'s real bytes from the inner
/// source and transposing them (row-major `[vocab_size, hidden_size]` ->
/// `[hidden_size, vocab_size]`); every other identity passes through
/// unchanged. `vocab_size`/`hidden_size` are `token_embedding`'s own
/// (real, stored) dimensions.
pub struct DerivedLmHeadPayloadSource<S> {
    inner: S,
    token_embedding_range: ProductionPayloadRange,
    vocab_size: u64,
    hidden_size: u64,
}

impl<S> DerivedLmHeadPayloadSource<S> {
    /// `token_embedding` is the real tensor's own metadata (as discovered,
    /// *not* the synthetic `lm_head` entry) -- used to build the exact
    /// range this wrapper requests from `inner` for every `lm_head` call.
    pub fn new(inner: S, token_embedding: &ModelTensorMetadata) -> Option<Self> {
        let (offset, length) = (token_embedding.offset_bytes?, token_embedding.size_bytes?);
        let [vocab_size, hidden_size] = token_embedding.shape[..] else {
            return None;
        };
        Some(Self {
            inner,
            token_embedding_range: ProductionPayloadRange {
                identity: TOKEN_EMBEDDING.to_string(),
                offset,
                length,
                digest: token_embedding.digest.clone(),
            },
            vocab_size,
            hidden_size,
        })
    }
}

impl<S: ProductionArtifactPayloadSource> ProductionArtifactPayloadSource
    for DerivedLmHeadPayloadSource<S>
{
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        if range.identity != LM_HEAD {
            return self.inner.read_payload(range);
        }
        let bytes = self.inner.read_payload(&self.token_embedding_range)?;
        let element_count = self
            .vocab_size
            .checked_mul(self.hidden_size)
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: "derived lm_head element count overflows".into(),
            })?;
        if bytes.len() as u64 != element_count.saturating_mul(4) {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "token_embedding byte length {} does not match {}x{} F32 elements while \
                     deriving lm_head",
                    bytes.len(),
                    self.vocab_size,
                    self.hidden_size
                ),
            });
        }
        let rows = self.vocab_size as usize;
        let cols = self.hidden_size as usize;
        let mut transposed = vec![0u8; bytes.len()];
        for row in 0..rows {
            for col in 0..cols {
                let source_offset = (row * cols + col) * 4;
                let dest_offset = (col * rows + row) * 4;
                transposed[dest_offset..dest_offset + 4]
                    .copy_from_slice(&bytes[source_offset..source_offset + 4]);
            }
        }
        Ok(transposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::model::ModelDType;
    use std::collections::BTreeMap;

    struct FixedPayloadSource(BTreeMap<String, Vec<u8>>);

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

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn appends_synthetic_lm_head_only_when_tied_and_absent() {
        let mut tensors = vec![tensor("token_embedding", vec![16, 4])];
        assert!(append_synthetic_lm_head_if_tied(&mut tensors, true));
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[1].name, "lm_head");
        assert_eq!(tensors[1].shape, vec![4, 16]);
        assert!(tensors[1].digest.is_none());
    }

    #[test]
    fn does_not_append_when_untied() {
        let mut tensors = vec![tensor("token_embedding", vec![16, 4])];
        assert!(!append_synthetic_lm_head_if_tied(&mut tensors, false));
        assert_eq!(tensors.len(), 1);
    }

    #[test]
    fn does_not_append_when_lm_head_already_present() {
        let mut tensors = vec![
            tensor("token_embedding", vec![16, 4]),
            tensor("lm_head", vec![4, 16]),
        ];
        assert!(!append_synthetic_lm_head_if_tied(&mut tensors, true));
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn derives_lm_head_bytes_as_the_transpose_of_token_embedding() {
        // 2x3 (vocab=2, hidden=3): [[1,2,3],[4,5,6]]
        let raw = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut token_embedding = tensor("token_embedding", vec![2, 3]);
        token_embedding.offset_bytes = Some(0);
        token_embedding.size_bytes = Some(raw.len() as u64);
        let mut bytes_by_name = BTreeMap::new();
        bytes_by_name.insert("token_embedding".to_string(), raw);
        let source =
            DerivedLmHeadPayloadSource::new(FixedPayloadSource(bytes_by_name), &token_embedding)
                .unwrap();

        let derived = source
            .read_payload(&ProductionPayloadRange {
                identity: "lm_head".into(),
                offset: 0,
                length: 24,
                digest: None,
            })
            .unwrap();
        let values: Vec<f32> = derived
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        // 3x2 transpose: [[1,4],[2,5],[3,6]]
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
