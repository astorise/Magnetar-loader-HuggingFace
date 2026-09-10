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
mod naming;
mod tokenizer;
mod weight_layout;
mod weights;

pub use config::{NormalizedHfConfig, parse as parse_config};
pub use naming::normalize_tensor_name;
pub use tokenizer::{
    HuggingFaceTokenizer, parse_generation_config, parse_tokenizer_config,
    required_special_token_kinds,
};
pub use weight_layout::TransposingPayloadSource;
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
        let normalized_config = config::parse(&config_bytes)?;

        let (mut tensors, shards, payload_source) = weights::discover_and_parse_weights(source)?;
        if tensors.is_empty() {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: "no tensors were discovered in this bundle's weight files".into(),
            });
        }

        // Every 2D projection weight (q/k/v/o_proj, gate/up/down_proj,
        // lm_head) is stored (and therefore discovered) as `nn.Linear`'s
        // own [out_features, in_features] convention; the Component
        // expects [in_features, out_features] instead -- see
        // `weight_layout.rs`. `token_embedding` (a lookup table) and the
        // 1D normalization vectors are excluded structurally, never
        // transposed.
        let payload_source: Arc<dyn ProductionArtifactPayloadSource> = Arc::new(
            weight_layout::TransposingPayloadSource::new(payload_source, &tensors),
        );
        weight_layout::swap_declared_projection_shapes(&mut tensors);

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
