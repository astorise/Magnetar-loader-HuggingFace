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

mod awq;
mod bnb;
mod chat_template;
mod config;
mod derived_lm_head;
mod gptq;
mod naming;
mod tokenizer;
mod weight_layout;
mod weights;

pub use chat_template::HuggingFaceChatTemplateFormatter;
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
    ModelArtifactId, ModelArtifactKind, ModelArtifactPart, ModelDType, ModelDigest, ModelManifest,
    ModelName, ModelRevision,
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

/// The Model Artifact's architecture family (Tachyon integration audit
/// MAG-01, astorise/Magnetar#83): previously hardcoded to `"qwen"`
/// regardless of the real ingested `config.json`, which made every
/// downstream family-compatibility check (`FirstNativeModelConfig::
/// validate`) tautological -- it always compared a value this crate
/// invented against itself.
///
/// Currently a direct, unopinionated pass-through of the real `model_type`
/// HF declares (`"qwen2"`, `"llama"`, `"mistral"`, ...) -- deliberately
/// *not* grouping related model_types into a shared family. Whether, say,
/// `"qwen2"` and `"qwen2_moe"` should be treated as interchangeable is a
/// real Magnetar product decision (it depends on real capability
/// differences a loader has no basis to judge), not something this crate
/// should decide unilaterally. Until that decision is made, family ==
/// model_type exactly: every distinct source architecture gets its own
/// distinct, honest family value instead of a shared lie, which is already
/// enough for a real family-mismatch check (e.g. a Llama Artifact against
/// a Component that only declares Qwen support) to work.
fn architecture_family(model_type: &str) -> String {
    model_type.to_string()
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

        let (mut tensors, shards, quantized_projection_names, payload_source) =
            weights::discover_and_parse_weights(source)?;
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
        // transposed -- as are any GPTQ- or AWQ-dequantized projections
        // (`quantized_projection_names`), already `[in_features,
        // out_features]` by construction (`gptq.rs`/`awq.rs`).
        let transposing_source = weight_layout::TransposingPayloadSource::new(
            payload_source,
            &tensors,
            &quantized_projection_names,
        );
        weight_layout::swap_declared_projection_shapes(&mut tensors, &quantized_projection_names);

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

        // A real chat template (when the bundle declares one) is recorded
        // the same way "tokenizer"/"tokenizer_config" already are: a named
        // part carrying a real digest over its own content, referenced by
        // name from the manifest -- `ModelManifest::validate`'s `chat-
        // template` reference check requires exactly this shape (task
        // group 1). The raw template text itself is not carried on the
        // manifest (parts are identity/provenance, not payload, matching
        // "tokenizer"/"tokenizer_config"'s own convention) -- a caller
        // that wants to actually render it reads `tokenizer_config.json`
        // again through this same authorized `source` and calls
        // `chat_template::load_chat_template_formatter`, exactly how a
        // caller already re-reads `tokenizer.json` to build the real
        // `Tokenizer` today.
        let chat_template_reference = tokenizer_config_metadata
            .as_ref()
            .and_then(|metadata| metadata.chat_template_reference.as_deref())
            .map(|template| {
                parts.insert(
                    "chat_template".to_string(),
                    ModelArtifactPart {
                        name: "chat_template".to_string(),
                        kind: ModelArtifactKind::ChatTemplate,
                        digest: ModelDigest::sha256(template.as_bytes()),
                        size_bytes: Some(template.len() as u64),
                        required: false,
                    },
                );
                "chat_template".to_string()
            });

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

        let manifest =
            ModelManifest {
                schema_version: magnetar_runtime::model::MODEL_ARTIFACT_SCHEMA_VERSION,
                id,
                architecture: normalized_config.metadata.normalize_architecture(
                    architecture_family(&normalized_config.metadata.model_type),
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
                chat_template: chat_template_reference,
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

    /// [`qwen2_config_bytes`], parameterized over `architectures`/
    /// `model_type` (#83) -- everything else stays identical so only the
    /// architecture identity varies between cases.
    fn config_bytes_with_model_type(architectures: &str, model_type: &str) -> Vec<u8> {
        serde_json::json!({
            "architectures": [architectures],
            "model_type": model_type,
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
        // #83: family must be the real ingested model_type, never a
        // hardcoded "qwen" -- this bundle's own config.json declares
        // "qwen2", so that's what both family and identifier must read.
        assert_eq!(result.manifest.architecture.family, "qwen2");
        assert_eq!(result.manifest.architecture.identifier, "qwen2");
        assert!(
            result.manifest.chat_template.is_none(),
            "this bundle declares no tokenizer_config.json, so no chat template reference"
        );

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

    /// #83: the ingested architecture family must reflect the real
    /// `config.json` `model_type`, never a hardcoded `"qwen"` -- verified
    /// across several distinct real HF `model_type` values, plus one this
    /// crate has never seen before (`"some_future_arch"`), to prove this
    /// isn't an allowlist that would reject an unrecognized-but-valid
    /// architecture.
    #[test]
    fn architecture_family_reflects_the_real_ingested_model_type() {
        let cases = [
            ("Qwen2ForCausalLM", "qwen2"),
            ("LlamaForCausalLM", "llama"),
            ("MistralForCausalLM", "mistral"),
            ("SomeFutureArchForCausalLM", "some_future_arch"),
        ];
        for (architectures, model_type) in cases {
            let dir = tempfile::tempdir().unwrap();
            fs::write(
                dir.path().join("config.json"),
                config_bytes_with_model_type(architectures, model_type),
            )
            .unwrap();
            write_tiny_safetensors(&dir.path().join("model.safetensors"));

            let source = ProductionModelSource::authorized_local_bundle(
                ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
                dir.path().to_path_buf(),
            );
            let result = HuggingFaceIngestor::new()
                .ingest(&source)
                .unwrap_or_else(|error| panic!("{model_type} bundle must ingest: {error}"));

            assert_eq!(
                result.manifest.architecture.family, model_type,
                "family must be the real model_type, not a hardcoded value"
            );
            assert_eq!(result.manifest.architecture.identifier, model_type);
            assert_ne!(
                result.manifest.architecture.family, "qwen",
                "{model_type} must never be silently relabeled \"qwen\""
            );
        }
    }

    /// A real chat template, when declared, is threaded into the manifest
    /// as a named, digested part -- the same shape "tokenizer"/
    /// "tokenizer_config" already use -- not discarded (task 1.1).
    #[test]
    fn threads_a_declared_chat_template_into_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), qwen2_config_bytes()).unwrap();
        write_tiny_safetensors(&dir.path().join("model.safetensors"));
        let template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}";
        fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({"chat_template": template}).to_string(),
        )
        .unwrap();
        // `tokenizer_config`'s own reference in the manifest points at the
        // "tokenizer" part (see `ingest`'s own construction below), so a
        // real `tokenizer.json` must also be present for `validate` to
        // resolve it -- unrelated to this test's actual subject (the
        // chat-template reference), just a real bundle requirement.
        fs::write(
            dir.path().join("tokenizer.json"),
            serde_json::json!({
                "version": "1.0",
                "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"}
            })
            .to_string(),
        )
        .unwrap();

        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let result = HuggingFaceIngestor::new()
            .ingest(&source)
            .expect("bundle ingests");

        assert_eq!(
            result.manifest.chat_template.as_deref(),
            Some("chat_template")
        );
        let part = result
            .manifest
            .parts
            .get("chat_template")
            .expect("a chat_template part was inserted");
        assert_eq!(
            part.digest,
            magnetar_runtime::model::ModelDigest::sha256(template.as_bytes())
        );
        assert_eq!(part.size_bytes, Some(template.len() as u64));
        assert!(
            !part.required,
            "a declared chat template is optional metadata, not a mandatory artifact part"
        );

        // The reference is real: `validate` requires the named part to
        // actually exist, which it does.
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

    /// End-to-end proof that a GPTQ-quantized bundle ingests correctly
    /// through the full `ingest()` pipeline, not just the low-level
    /// dequantization math (`gptq.rs`'s own tests, including the real
    /// public-checkpoint comparison). Builds a small bundle with one
    /// GPTQ-quantized projection (`self_attn.q_proj`, hand-computed
    /// values, 1 quantization group) alongside two plain `F32` tensors,
    /// and asserts: the manifest carries `layers.0.self_attn.q_proj` at
    /// its real, already-Runtime-oriented `[in_features, out_features]`
    /// shape (never transposed a second time -- `weight_layout.rs`'s
    /// GPTQ exclusion), and reading its payload returns the correctly
    /// dequantized bytes.
    #[test]
    fn ingests_a_gptq_quantized_bundle_end_to_end() {
        fn pack_i32(nibbles: &[i32]) -> i32 {
            let mut value: i32 = 0;
            for (index, nibble) in nibbles.iter().enumerate() {
                value |= (nibble & 0xF) << (4 * index);
            }
            value
        }
        fn f16_bits_from_f32(value: f32) -> u16 {
            let bits = value.to_bits();
            let sign = (bits >> 16) & 0x8000;
            let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
            let mantissa = (bits >> 13) & 0x3FF;
            (sign | ((exponent as u32) << 10) | mantissa) as u16
        }

        let dir = tempfile::tempdir().unwrap();
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_config_bytes()).unwrap();
        config["hidden_size"] = serde_json::json!(8);
        config["intermediate_size"] = serde_json::json!(8);
        config["num_attention_heads"] = serde_json::json!(1);
        config["num_key_value_heads"] = serde_json::json!(1);
        config["vocab_size"] = serde_json::json!(4);
        config["tie_word_embeddings"] = serde_json::json!(false);
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        // One GPTQ group covering all 8 input channels, code pattern
        // [8,9,10,11,12,13,14,15] (zero_code=7, so quant_code-(zero+1) =
        // [0,1,...,7]) repeated identically for every one of the 8 output
        // columns, scale=0.1 -- the same hand-built shape `gptq.rs`'s own
        // unit test already verifies the arithmetic for, exercised here
        // through the real file-based ingestion path instead.
        let packed_row = pack_i32(&[8, 9, 10, 11, 12, 13, 14, 15]);
        let qweight_bytes: Vec<u8> = std::iter::repeat_n(packed_row, 8)
            .flat_map(i32::to_le_bytes)
            .collect();
        let qzeros_bytes: Vec<u8> = pack_i32(&[7; 8]).to_le_bytes().to_vec();
        let scales_bytes: Vec<u8> = std::iter::repeat_n(f16_bits_from_f32(0.1), 8)
            .flat_map(u16::to_le_bytes)
            .collect();
        let gidx_bytes: Vec<u8> = std::iter::repeat_n(0i32, 8)
            .flat_map(i32::to_le_bytes)
            .collect();

        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>,
                        name: &str,
                        dtype: &str,
                        shape: Vec<u64>,
                        bytes: &[u8]| {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            let end = data.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}),
            );
        };
        push(
            &mut header,
            "model.embed_tokens.weight",
            "F32",
            vec![4, 8],
            &[0u8; 4 * 8 * 4],
        );
        push(
            &mut header,
            "model.norm.weight",
            "F32",
            vec![8],
            &1.0f32.to_le_bytes().repeat(8),
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.qweight",
            "I32",
            vec![1, 8],
            &qweight_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.qzeros",
            "I32",
            vec![1, 1],
            &qzeros_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.scales",
            "F16",
            vec![1, 8],
            &scales_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.g_idx",
            "I32",
            vec![8],
            &gidx_bytes,
        );
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
            .expect("a GPTQ-quantized bundle ingests");

        let q_proj = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.self_attn.q_proj")
            .expect("the GPTQ projection is present under its canonical name");
        assert_eq!(
            q_proj.shape,
            vec![8, 8],
            "GPTQ's own packing already puts [in_features, out_features] first -- \
             the transpose exclusion must have kept this from being swapped a second time"
        );
        assert_eq!(q_proj.storage_dtype, ModelDType::F32);
        assert!(
            !result
                .manifest
                .tensors
                .iter()
                .any(|tensor| tensor.name.contains("qweight") || tensor.name.contains("qzeros")),
            "the four raw GPTQ siblings must not leak into the final manifest"
        );

        let range = magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
            identity: q_proj.name.clone(),
            offset: q_proj.offset_bytes.unwrap(),
            length: q_proj.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = result.payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values.len(), 64);
        // Row-major [in_features=8, out_features=8]: input channel i's
        // code is (i-th nibble of the shared packed value) - 8, scaled by
        // 0.1 -- identical across every output column, matching the
        // hand-built fixture above.
        for i in 0..8usize {
            let expected = 0.1 * (i as f32);
            for o in 0..8usize {
                let actual = values[i * 8 + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "value[{i}][{o}]: got {actual}, expected {expected}"
                );
            }
        }
    }

    /// End-to-end proof that an AWQ-quantized bundle ingests correctly
    /// through the full `ingest()` pipeline, not just the low-level
    /// dequantization math (`awq.rs`'s own tests, including the real
    /// public-checkpoint comparison). Also proves `extract_gptq_
    /// projections`'s own skip-when-no-`g_idx` behavior end to end: this
    /// bundle's `.qweight` tensor has no `.g_idx` sibling, so GPTQ
    /// extraction must leave it alone and AWQ extraction must claim it.
    #[test]
    fn ingests_an_awq_quantized_bundle_end_to_end() {
        const AWQ_REVERSE_ORDER: [i32; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
        fn pack_i32_awq_order(semantic_codes: &[i32; 8]) -> i32 {
            let mut value: i32 = 0;
            for (semantic_col, nibble) in semantic_codes.iter().enumerate() {
                let position = AWQ_REVERSE_ORDER[semantic_col];
                value |= (nibble & 0xF) << (4 * position);
            }
            value
        }
        fn f16_bits_from_f32(value: f32) -> u16 {
            let bits = value.to_bits();
            let sign = (bits >> 16) & 0x8000;
            let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
            let mantissa = (bits >> 13) & 0x3FF;
            (sign | ((exponent as u32) << 10) | mantissa) as u16
        }

        let dir = tempfile::tempdir().unwrap();
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_config_bytes()).unwrap();
        config["hidden_size"] = serde_json::json!(8);
        config["intermediate_size"] = serde_json::json!(8);
        config["num_attention_heads"] = serde_json::json!(1);
        config["num_key_value_heads"] = serde_json::json!(1);
        config["vocab_size"] = serde_json::json!(4);
        config["tie_word_embeddings"] = serde_json::json!(false);
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        // One AWQ group covering all 8 input channels, out_features=8
        // (one packed column-block): semantic codes [0,1,...,7], zero=3,
        // scale=0.1 -- identical for every one of the 8 input-channel
        // rows, matching `awq.rs`'s own hand-built unit test's shape.
        let packed_col_block = pack_i32_awq_order(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let qweight_bytes: Vec<u8> = std::iter::repeat_n(packed_col_block, 8)
            .flat_map(i32::to_le_bytes)
            .collect();
        let qzeros_bytes: Vec<u8> = pack_i32_awq_order(&[3; 8]).to_le_bytes().to_vec();
        let scales_bytes: Vec<u8> = std::iter::repeat_n(f16_bits_from_f32(0.1), 8)
            .flat_map(u16::to_le_bytes)
            .collect();

        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>,
                        name: &str,
                        dtype: &str,
                        shape: Vec<u64>,
                        bytes: &[u8]| {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            let end = data.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}),
            );
        };
        push(
            &mut header,
            "model.embed_tokens.weight",
            "F32",
            vec![4, 8],
            &[0u8; 4 * 8 * 4],
        );
        push(
            &mut header,
            "model.norm.weight",
            "F32",
            vec![8],
            &1.0f32.to_le_bytes().repeat(8),
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.qweight",
            "I32",
            vec![8, 1],
            &qweight_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.qzeros",
            "I32",
            vec![1, 1],
            &qzeros_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.scales",
            "F16",
            vec![1, 8],
            &scales_bytes,
        );
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
            .expect("an AWQ-quantized bundle ingests");

        let q_proj = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.self_attn.q_proj")
            .expect("the AWQ projection is present under its canonical name");
        assert_eq!(
            q_proj.shape,
            vec![8, 8],
            "AWQ's own packing already puts [in_features, out_features] first -- \
             the transpose exclusion must have kept this from being swapped a second time"
        );
        assert_eq!(q_proj.storage_dtype, ModelDType::F32);
        assert!(
            !result
                .manifest
                .tensors
                .iter()
                .any(|tensor| tensor.name.contains("qweight") || tensor.name.contains("qzeros")),
            "the three raw AWQ siblings must not leak into the final manifest"
        );

        let range = magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
            identity: q_proj.name.clone(),
            offset: q_proj.offset_bytes.unwrap(),
            length: q_proj.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = result.payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values.len(), 64);
        // Row-major [in_features=8, out_features=8]: every input channel
        // shares the same packed column-block, so the value depends only
        // on the output column: (semantic code - 3) * 0.1.
        for i in 0..8usize {
            for o in 0..8usize {
                let expected = 0.1 * (o as f32 - 3.0);
                let actual = values[i * 8 + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "value[{i}][{o}]: got {actual}, expected {expected}"
                );
            }
        }
    }

    /// End-to-end proof that a BitsAndBytes-quantized bundle ingests
    /// correctly through the full `ingest()` pipeline, not just the
    /// low-level dequantization math (`bnb.rs`'s own tests, including the
    /// real public-checkpoint comparison). This is the one quantization
    /// scheme whose packed weight tensor keeps the literal raw name
    /// `<prefix>.weight` -- the exact identity-relocation subtlety this
    /// test exercises (`extract_bnb_projections`'s own `weight_range.
    /// identity = canonical_name` fix) never comes up for GPTQ/AWQ, whose
    /// synthesized placeholders are pushed under a name distinct from any
    /// raw tensor. Also proves BnB projections are *not* excluded from
    /// this crate's `nn.Linear`-storage-convention transpose, unlike
    /// GPTQ/AWQ (uses a square 8x8 shape so a transpose bug would not be
    /// masked by symmetry in the byte layout itself, even though this
    /// particular fixture's dequantized values happen to be uniform).
    #[test]
    fn ingests_a_bnb_quantized_bundle_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_config_bytes()).unwrap();
        config["hidden_size"] = serde_json::json!(8);
        config["intermediate_size"] = serde_json::json!(8);
        config["num_attention_heads"] = serde_json::json!(1);
        config["num_key_value_heads"] = serde_json::json!(1);
        config["vocab_size"] = serde_json::json!(4);
        config["tie_word_embeddings"] = serde_json::json!(false);
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        // 64 elements (out_features=8, in_features=8), one first-level
        // block (blocksize=64) and one nested block (nested_blocksize=1):
        // every element dequantizes to the same value, matching `bnb.rs`'s
        // own hand-built double-quantized unit test's arithmetic
        // (quant_map[7] * (nested_quant_map[42]*nested_absmax[0] + 0.5) =
        // 7.0 * (3.0*2.0 + 0.5) = 45.5), exercised here through the real
        // file-based ingestion path instead.
        let quant_map_bytes: Vec<u8> = (0..16u32).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let weight_bytes = vec![0x77u8; 32]; // 64 elements, 2 per byte, both nibbles = 7
        let absmax_bytes = vec![42u8];
        let mut nested_quant_map = vec![0.0f32; 256];
        nested_quant_map[42] = 3.0;
        let nested_quant_map_bytes: Vec<u8> = nested_quant_map
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let nested_absmax_bytes: Vec<u8> = 2.0f32.to_le_bytes().to_vec();
        let quant_state_json = serde_json::json!({
            "quant_type": "nf4",
            "blocksize": 64,
            "dtype": "bfloat16",
            "shape": [8, 8],
            "nested_blocksize": 1,
            "nested_dtype": "float32",
            "nested_offset": 0.5
        })
        .to_string()
        .into_bytes();

        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut push = |header: &mut serde_json::Map<String, serde_json::Value>,
                        name: &str,
                        dtype: &str,
                        shape: Vec<u64>,
                        bytes: &[u8]| {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            let end = data.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}),
            );
        };
        push(
            &mut header,
            "model.embed_tokens.weight",
            "F32",
            vec![4, 8],
            &[0u8; 4 * 8 * 4],
        );
        push(
            &mut header,
            "model.norm.weight",
            "F32",
            vec![8],
            &1.0f32.to_le_bytes().repeat(8),
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.weight",
            "U8",
            vec![32, 1],
            &weight_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.weight.absmax",
            "U8",
            vec![1],
            &absmax_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.weight.quant_map",
            "F32",
            vec![16],
            &quant_map_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.weight.nested_absmax",
            "F32",
            vec![1],
            &nested_absmax_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.weight.nested_quant_map",
            "F32",
            vec![256],
            &nested_quant_map_bytes,
        );
        push(
            &mut header,
            "model.layers.0.self_attn.q_proj.weight.quant_state.bitsandbytes__nf4",
            "U8",
            vec![quant_state_json.len() as u64],
            &quant_state_json,
        );
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
            .expect("a BitsAndBytes-quantized bundle ingests");

        let q_proj = result
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.self_attn.q_proj")
            .expect("the BnB projection is present under its canonical name");
        assert_eq!(q_proj.shape, vec![8, 8]);
        assert_eq!(q_proj.storage_dtype, ModelDType::F32);
        assert!(
            !result
                .manifest
                .tensors
                .iter()
                .any(|tensor| tensor.name.contains("absmax") || tensor.name.contains("quant_map")),
            "the BnB sibling tensors must not leak into the final manifest"
        );

        let range = magnetar_runtime::production_model_ingestion::ProductionPayloadRange {
            identity: q_proj.name.clone(),
            offset: q_proj.offset_bytes.unwrap(),
            length: q_proj.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = result.payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values.len(), 64);
        for (index, value) in values.iter().enumerate() {
            assert!(
                (value - 45.5).abs() < 1e-3,
                "value[{index}]: got {value}, expected 45.5"
            );
        }
    }
}
