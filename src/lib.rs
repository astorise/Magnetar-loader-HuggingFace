//! Production Hugging Face-style Qwen bundle ingestion, implementing
//! `magnetar-runtime`'s generic [`ProductionModelArtifactIngestor`]
//! contract (`implement-production-qwen-model-loading`, `production-
//! model-ingestion` capability, Decision 1). Composes the real
//! `magnetar-format-safetensors` parser and this crate's own real
//! `config.json`/`tokenizer.json`/`tokenizer_config.json`/
//! `generation_config.json` parsing -- `magnetar-runtime` never imports
//! this crate (enforced by `submodule-integration`'s dependency guard).
//!
//! Recognizes, at minimum: `config.json`, `tokenizer.json`,
//! `tokenizer_config.json`, `generation_config.json`, single-file
//! `model.safetensors`, and Hugging Face-style indexed sharded
//! `model.safetensors.index.json` + `model-*-of-*.safetensors`.
//!
//! Parsing/normalizing a bundle never grants trust (Decision 2): the
//! returned [`magnetar_runtime::model::ModelManifest`] still goes through
//! `magnetar-runtime`'s own `ModelManifest::validate` and
//! `ModelTrustStore::evaluate` like any other manifest before Model
//! Loading may materialize anything from it.

mod config;
mod derived_lm_head;
mod naming;
mod tokenizer;
mod weight_layout;
mod weights;

pub use config::{NormalizedHfConfig, parse as parse_config};
pub use derived_lm_head::{DerivedLmHeadPayloadSource, append_synthetic_lm_head_if_tied};
pub use naming::normalize_tensor_name;
pub use tokenizer::{
    HuggingFaceTokenizer, parse_generation_config, parse_tokenizer_config,
    required_special_token_kinds,
};
pub use weight_layout::{TransposingPayloadSource, swap_declared_projection_shapes};
pub use weights::SafetensorsPayloadSource;

use magnetar_runtime::model::{
    ModelArchitecture, ModelArtifactId, ModelArtifactKind, ModelArtifactPart, ModelDType,
    ModelDigest, ModelManifest, ModelName, ModelRevision,
};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionIngestionResult,
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use std::{collections::BTreeMap, collections::BTreeSet, fs, sync::Arc};

const CONFIG_FILE_NAME: &str = "config.json";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";
const GENERATION_CONFIG_FILE_NAME: &str = "generation_config.json";

/// This ingestor's stable registry identity
/// (`ProductionModelArtifactIngestor::ingestor_id`).
pub const INGESTOR_ID: &str = "huggingface-qwen";

/// The external, pinned production Model Artifact ingestor for Hugging
/// Face-style Qwen bundles.
#[derive(Default)]
pub struct HuggingFaceIngestor;

impl HuggingFaceIngestor {
    pub fn new() -> Self {
        Self
    }
}

fn sanitize_model_name(model_type: &str) -> String {
    let sanitized: String = model_type
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "huggingface-bundle".to_string()
    } else {
        format!("hf-{sanitized}")
    }
}

impl ProductionModelArtifactIngestor for HuggingFaceIngestor {
    fn ingestor_id(&self) -> &str {
        INGESTOR_ID
    }

    fn ingest(
        &self,
        source: &ProductionModelSource,
    ) -> Result<ProductionIngestionResult, ProductionIngestionError> {
        let config_path = source.resolve(CONFIG_FILE_NAME).map_err(|_| {
            ProductionIngestionError::RequiredPartMissing {
                part: CONFIG_FILE_NAME.to_string(),
            }
        })?;
        let config_bytes = fs::read(&config_path).map_err(|error| {
            ProductionIngestionError::RequiredPartMissing {
                part: format!("{CONFIG_FILE_NAME} ({error})"),
            }
        })?;
        let mut normalized_config = config::parse(&config_bytes)?;

        let (mut tensors, shards, payload_source) = weights::discover_and_parse_weights(source)?;
        if tensors.is_empty() {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: "no tensors were discovered in this bundle's weight files".into(),
            });
        }
        // `config.json` never declares attention bias for real Qwen2/2.5
        // checkpoints (a model-class default, not a config field) -- the
        // only real evidence is whether the bundle's own Safetensors
        // header actually declares a `q_bias` tensor (canonicalized by
        // `naming.rs` from `self_attn.q_proj.bias`). `k_bias`/`v_bias`
        // always accompany it on every real such checkpoint; checking one
        // is sufficient and keeps this a simple presence check rather
        // than a partial-bias inconsistency policy this change does not
        // need.
        normalized_config.architecture_config.attention_bias = tensors
            .iter()
            .any(|tensor| tensor.name.ends_with("self_attn.q_bias"));

        // Every 2D projection weight (q/k/v/o_proj, gate/up/down_proj,
        // lm_head) is stored (and therefore discovered) as `nn.Linear`'s
        // own [out_features, in_features] convention; the Component
        // expects [in_features, out_features] instead -- see
        // `weight_layout.rs`. `token_embedding` (a lookup table) and the
        // 1D normalization vectors are excluded structurally, never
        // transposed.
        let transposing_source =
            weight_layout::TransposingPayloadSource::new(payload_source, &tensors);
        weight_layout::swap_declared_projection_shapes(&mut tensors);

        // A real `tie_word_embeddings: true` checkpoint genuinely omits
        // `lm_head.weight` -- the model reuses `token_embedding` for the
        // output projection. Derive it here, at load time, from the real
        // already-discovered `token_embedding` tensor rather than
        // requiring the Component to special-case a missing weight
        // (`implement-production-qwen-model-loading` task 10.5).
        let token_embedding = tensors
            .iter()
            .find(|tensor| tensor.name == "token_embedding")
            .cloned();
        let synthetic_lm_head_added = derived_lm_head::append_synthetic_lm_head_if_tied(
            &mut tensors,
            normalized_config.architecture_config.tie_word_embeddings,
        );
        let payload_source: Arc<dyn ProductionArtifactPayloadSource> = if synthetic_lm_head_added {
            let token_embedding = token_embedding.expect(
                "append_synthetic_lm_head_if_tied only returns true when a token_embedding \
                 tensor was found",
            );
            Arc::new(
                derived_lm_head::DerivedLmHeadPayloadSource::new(
                    transposing_source,
                    &token_embedding,
                )
                .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                    reason: "token_embedding is missing the offset/size/shape metadata needed \
                             to derive a tied lm_head"
                        .into(),
                })?,
            )
        } else {
            Arc::new(transposing_source)
        };

        let storage_dtype = tensors.first().map(|tensor| tensor.storage_dtype);
        let mut supported_compute_dtypes = BTreeSet::new();
        supported_compute_dtypes.insert(ModelDType::F32);

        let mut parts = BTreeMap::new();
        parts.insert(
            "config".to_string(),
            ModelArtifactPart {
                name: "config".to_string(),
                kind: ModelArtifactKind::ModelConfig,
                digest: ModelDigest::sha256(&config_bytes),
                size_bytes: Some(config_bytes.len() as u64),
                required: true,
            },
        );
        // A production bundle's weights may span multiple shard files;
        // this "weights" part is a bundle-level identity marker (digest
        // over every tensor's own name/shape/byte-range, deterministic
        // and order-independent), not a substitute for the real per-shard
        // digests already carried on `shards` -- Model Loading validates
        // shard/tensor content against those, not against this part.
        let weights_identity = {
            let mut entries: Vec<String> = tensors
                .iter()
                .map(|tensor| {
                    format!(
                        "{}:{:?}:{:?}:{:?}",
                        tensor.name, tensor.shape, tensor.offset_bytes, tensor.size_bytes
                    )
                })
                .collect();
            entries.sort();
            ModelDigest::sha256(entries.join("\n").as_bytes())
        };
        parts.insert(
            "weights".to_string(),
            ModelArtifactPart {
                name: "weights".to_string(),
                kind: ModelArtifactKind::ModelWeights,
                digest: weights_identity,
                size_bytes: None,
                required: true,
            },
        );

        let mut tokenizer_reference = None;
        let mut tokenizer_config_metadata = None;
        if let Ok(tokenizer_config_path) = source.resolve(TOKENIZER_CONFIG_FILE_NAME) {
            let bytes = fs::read(&tokenizer_config_path).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("{TOKENIZER_CONFIG_FILE_NAME}: {error}"),
                }
            })?;
            tokenizer_config_metadata = Some(tokenizer::parse_tokenizer_config(&bytes)?);
        }
        if let Ok(tokenizer_path) = source.resolve(TOKENIZER_FILE_NAME) {
            let bytes = fs::read(&tokenizer_path).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("{TOKENIZER_FILE_NAME}: {error}"),
                }
            })?;
            parts.insert(
                "tokenizer".to_string(),
                ModelArtifactPart {
                    name: "tokenizer".to_string(),
                    kind: ModelArtifactKind::Tokenizer,
                    digest: ModelDigest::sha256(&bytes),
                    size_bytes: Some(bytes.len() as u64),
                    required: false,
                },
            );
            tokenizer_reference = Some("tokenizer".to_string());
        }

        let generation =
            if let Ok(generation_config_path) = source.resolve(GENERATION_CONFIG_FILE_NAME) {
                let bytes = fs::read(&generation_config_path).map_err(|error| {
                    ProductionIngestionError::MalformedMetadata {
                        reason: format!("{GENERATION_CONFIG_FILE_NAME}: {error}"),
                    }
                })?;
                Some(tokenizer::parse_generation_config(&bytes)?.as_defaults())
            } else {
                None
            };

        let name = sanitize_model_name(&normalized_config.metadata.model_type);
        let id = ModelArtifactId::new(
            ModelArtifactKind::ModelBundle,
            ModelName::new(name).map_err(|error| ProductionIngestionError::MalformedMetadata {
                reason: error.to_string(),
            })?,
            ModelRevision::new("local").map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: error.to_string(),
                }
            })?,
            ModelDigest::sha256(&config_bytes),
        );

        let manifest = ModelManifest {
            schema_version: magnetar_runtime::model::MODEL_ARTIFACT_SCHEMA_VERSION,
            id,
            architecture: ModelArchitecture::new(
                "qwen",
                normalized_config.metadata.model_type.clone(),
            ),
            parts,
            storage_dtype,
            compute_dtype: None,
            supported_compute_dtypes,
            tensors,
            tokenizer: tokenizer_reference,
            tokenizer_config: tokenizer_config_metadata
                .as_ref()
                .and(Some("tokenizer".to_string())),
            chat_template: None,
            prompt_template: None,
            generation,
            quantization: None,
            shards,
            runtime_features: BTreeSet::new(),
            memory_features: BTreeSet::new(),
            provider_capabilities: Vec::new(),
            component: None,
            license: None,
            provenance: None,
            signatures: Vec::new(),
            source: Some(source.kind().clone()),
            architecture_config: Some(normalized_config.architecture_config),
        };

        Ok(ProductionIngestionResult {
            manifest,
            payload_source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::ModelArtifactSource;
    use magnetar_runtime::model::ModelTrustStore;
    use std::io::Write;

    fn qwen2_config_bytes() -> Vec<u8> {
        serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "float32",
            "bos_token_id": 0,
            "eos_token_id": 1
        })
        .to_string()
        .into_bytes()
    }

    fn write_tiny_safetensors(path: &std::path::Path) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for name in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            let start = data.len() as u64;
            data.extend_from_slice(&1.0f32.to_le_bytes());
            let end = data.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "F32", "shape": [1], "data_offsets": [start, end]}),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&data).unwrap();
    }

    #[test]
    fn ingests_a_complete_local_bundle() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), qwen2_config_bytes()).unwrap();
        write_tiny_safetensors(&dir.path().join("model.safetensors"));

        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let ingestor = HuggingFaceIngestor::new();
        let result = ingestor.ingest(&source).expect("bundle ingests");

        assert_eq!(result.manifest.tensors.len(), 3);
        assert!(result.manifest.architecture_config.is_some());
        let config = result.manifest.architecture_config.clone().unwrap();
        assert_eq!(config.hidden_size, 8);
        assert_eq!(config.num_hidden_layers, 1);

        // Parsing/normalizing alone never grants trust (Decision 2).
        let trust = ModelTrustStore::default().evaluate(&result.manifest);
        assert_eq!(
            trust.status(),
            magnetar_runtime::model::ModelTrustStatus::Unknown
        );

        // The manifest is independently well-formed per the existing
        // Model Artifact contract.
        result.manifest.validate().expect("manifest validates");
    }

    #[test]
    fn derives_lm_head_end_to_end_when_tied_and_absent() {
        // A real tied checkpoint (`tie_word_embeddings: true`) genuinely
        // omits `lm_head.weight` from its bundle -- this proves ingestion
        // still produces a usable `lm_head` tensor and correct bytes for
        // it, driven only by the real `token_embedding` tensor this
        // bundle does declare (task 10.5). Non-square
        // vocab_size/hidden_size (3 vs 2) so a transpose bug would not be
        // masked by a square shape.
        let dir = tempfile::tempdir().unwrap();
        let mut config =
            serde_json::from_slice::<serde_json::Value>(&qwen2_config_bytes()).unwrap();
        config["hidden_size"] = serde_json::json!(2);
        config["vocab_size"] = serde_json::json!(3);
        config["num_attention_heads"] = serde_json::json!(1);
        config["num_key_value_heads"] = serde_json::json!(1);
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        // token_embedding stored [vocab=3, hidden=2] row-major:
        // [[1,2],[3,4],[5,6]]. No lm_head.weight in this bundle at all.
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, values) in [
            (
                "model.embed_tokens.weight",
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            ("model.norm.weight", vec![1.0, 1.0]),
        ] {
            let start = data.len() as u64;
            for value in values {
                data.extend_from_slice(&(value as f32).to_le_bytes());
            }
            let end = data.len() as u64;
            let shape: Vec<u64> = if name == "model.embed_tokens.weight" {
                vec![3, 2]
            } else {
                vec![2]
            };
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = fs::File::create(dir.path().join("model.safetensors")).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&data).unwrap();

        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = HuggingFaceIngestor::new()
            .ingest(&source)
            .expect("bundle ingests");
        result.manifest.validate().expect("manifest validates");

        let lm_head = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "lm_head")
            .expect("a synthetic lm_head tensor was added");
        assert_eq!(lm_head.shape, vec![2, 3], "hidden x vocab, as expected");
        assert!(lm_head.digest.is_none());

        let range = magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
            identity: "lm_head".to_string(),
            offset: lm_head.offset_bytes.unwrap(),
            length: lm_head.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = result.payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        // Transpose of [[1,2],[3,4],[5,6]] (3x2) is (2x3):
        // [[1,3,5],[2,4,6]].
        assert_eq!(values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn rejects_a_bundle_missing_config() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_safetensors(&dir.path().join("model.safetensors"));
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = match HuggingFaceIngestor::new().ingest(&source) {
            Err(error) => error,
            Ok(_) => panic!("expected a required-part-missing error"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::RequiredPartMissing { .. }
        ));
    }

    #[test]
    fn rejects_a_bundle_missing_weights() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), qwen2_config_bytes()).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = match HuggingFaceIngestor::new().ingest(&source) {
            Err(error) => error,
            Ok(_) => panic!("expected a required-part-missing error"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::RequiredPartMissing { .. }
        ));
    }

    #[test]
    fn ingestor_id_is_stable() {
        assert_eq!(HuggingFaceIngestor::new().ingestor_id(), INGESTOR_ID);
    }
}
