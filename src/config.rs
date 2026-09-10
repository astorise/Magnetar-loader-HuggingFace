//! Real Hugging Face-style `config.json` parsing, normalized into
//! `magnetar-runtime`'s generic [`ModelArchitectureConfig`] and
//! [`HfConfigMetadata`] contracts (`implement-production-qwen-model-loading`
//! task group 3, Decision 5).

use magnetar_runtime::model::ModelArchitectureConfig;
use magnetar_runtime::model_format_roadmap::{HfConfigMetadata, HfRopeMetadata};
use magnetar_runtime::production_model_ingestion::ProductionIngestionError;
use serde::Deserialize;
use std::collections::BTreeMap;

/// Raw `config.json` shape. Every architecture-critical field is
/// `Option` -- a real Hugging Face config's exact field set varies by
/// model family -- and validated explicitly by [`normalize`] rather than
/// relying on serde to enforce presence, so a missing field produces a
/// structured [`ProductionIngestionError::MalformedMetadata`] naming the
/// field, not an opaque deserialization error.
#[derive(Debug, Deserialize)]
struct RawHfConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: Option<String>,
    hidden_size: Option<u64>,
    intermediate_size: Option<u64>,
    num_hidden_layers: Option<u32>,
    num_attention_heads: Option<u32>,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u64>,
    vocab_size: Option<u64>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_scaling: Option<RawRopeScaling>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<RawEosTokenId>,
    #[serde(default)]
    max_position_embeddings: Option<u64>,
    #[serde(default)]
    hidden_act: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawRopeScaling {
    #[serde(default, rename = "type")]
    scaling_type: Option<String>,
    #[serde(default)]
    factor: Option<f64>,
}

/// `eos_token_id` is a single integer for most Hugging Face configs but a
/// list for some (multiple valid stop ids) -- the first is used for the
/// normalized single-id fields this profile carries; the full set is not
/// dropped silently, it is simply out of scope for this first profile's
/// normalized shape (documented non-goal, not a parsing bug).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawEosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl RawEosTokenId {
    fn first(&self) -> Option<u32> {
        match self {
            Self::Single(value) => Some(*value),
            Self::Multiple(values) => values.first().copied(),
        }
    }
}

/// The normalized result of parsing one `config.json`: the generic
/// architecture configuration a configurable Qwen Component queries
/// through `model-config`, plus the roadmap-shaped [`HfConfigMetadata`]
/// (architecture family/identifier normalization, `torch_dtype` preserved
/// as source annotation only).
#[derive(Debug)]
pub struct NormalizedHfConfig {
    pub architecture_config: ModelArchitectureConfig,
    pub metadata: HfConfigMetadata,
}

fn required<T>(value: Option<T>, field: &str) -> Result<T, ProductionIngestionError> {
    value.ok_or_else(|| ProductionIngestionError::MalformedMetadata {
        reason: format!("config.json is missing required field '{field}'"),
    })
}

/// Parses and normalizes `bytes` as a Hugging Face-style `config.json`.
/// Rejects missing/zero/internally-inconsistent architecture values with
/// structured errors (task 3.3) -- `head_dim`, when absent, is derived as
/// `hidden_size / num_attention_heads` and the derivation is validated to
/// divide evenly, matching the proposal's "head_dim (or validated
/// derivation)".
pub fn parse(bytes: &[u8]) -> Result<NormalizedHfConfig, ProductionIngestionError> {
    let raw: RawHfConfig = serde_json::from_slice(bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("config.json is not valid JSON matching the expected shape: {error}"),
        }
    })?;

    let hidden_size = required(raw.hidden_size, "hidden_size")?;
    let intermediate_size = required(raw.intermediate_size, "intermediate_size")?;
    let num_hidden_layers = required(raw.num_hidden_layers, "num_hidden_layers")?;
    let num_attention_heads = required(raw.num_attention_heads, "num_attention_heads")?;
    let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
    let vocab_size = required(raw.vocab_size, "vocab_size")?;
    let model_type = required(raw.model_type, "model_type")?;

    if hidden_size == 0 || intermediate_size == 0 || vocab_size == 0 {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json declares a zero-valued hidden_size/intermediate_size/vocab_size"
                .into(),
        });
    }
    if num_hidden_layers == 0 || num_attention_heads == 0 || num_key_value_heads == 0 {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json declares a zero-valued layer/head count".into(),
        });
    }
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "config.json's num_attention_heads ({num_attention_heads}) is not an exact \
                 multiple of num_key_value_heads ({num_key_value_heads})"
            ),
        });
    }

    let head_dim = match raw.head_dim {
        Some(declared) => declared,
        None => {
            if !hidden_size.is_multiple_of(u64::from(num_attention_heads)) {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "config.json declares no head_dim and hidden_size ({hidden_size}) is \
                         not evenly divisible by num_attention_heads ({num_attention_heads})"
                    ),
                });
            }
            hidden_size / u64::from(num_attention_heads)
        }
    };
    if head_dim == 0 {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json's head_dim (declared or derived) is zero".into(),
        });
    }

    let rms_norm_eps = raw.rms_norm_eps.unwrap_or(1e-6) as f32;
    let rope_theta = raw.rope_theta.unwrap_or(10_000.0);
    if rope_theta <= 0.0 {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json's rope_theta must be positive".into(),
        });
    }

    let architecture_config = ModelArchitectureConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        rope_scaling_factor: raw
            .rope_scaling
            .as_ref()
            .and_then(|scaling| scaling.factor)
            .map(|value| value as f32),
        tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        bos_token_id: raw.bos_token_id,
        eos_token_id: raw.eos_token_id.as_ref().and_then(RawEosTokenId::first),
    };
    architecture_config.validate().map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("config.json produced an inconsistent architecture config: {error}"),
        }
    })?;

    let mut annotations = BTreeMap::new();
    if let Some(dtype) = &raw.torch_dtype {
        annotations.insert("torch_dtype".to_string(), dtype.clone());
    }
    if let Some(act) = &raw.hidden_act {
        annotations.insert("hidden_act".to_string(), act.clone());
    }

    let metadata = HfConfigMetadata {
        architectures: raw.architectures,
        model_type,
        hidden_size: Some(hidden_size),
        num_hidden_layers: Some(num_hidden_layers),
        num_attention_heads: Some(num_attention_heads),
        num_key_value_heads: Some(num_key_value_heads),
        head_dim: Some(head_dim as u32),
        intermediate_size: Some(intermediate_size),
        vocab_size: Some(vocab_size),
        max_position_embeddings: raw.max_position_embeddings,
        hidden_act: raw.hidden_act,
        rope: Some(HfRopeMetadata {
            theta: Some(rope_theta as u64),
            scaling_type: raw
                .rope_scaling
                .as_ref()
                .and_then(|scaling| scaling.scaling_type.clone()),
            scaling_factor: raw
                .rope_scaling
                .as_ref()
                .and_then(|scaling| scaling.factor)
                .map(|value| value as u32),
        }),
        tie_word_embeddings: raw.tie_word_embeddings,
        torch_dtype: raw.torch_dtype,
        annotations,
    };

    Ok(NormalizedHfConfig {
        architecture_config,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen2_0_5b_config() -> Vec<u8> {
        br#"{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "max_position_embeddings": 32768
        }"#
        .to_vec()
    }

    #[test]
    fn parses_a_real_qwen2_config() {
        let normalized = parse(&qwen2_0_5b_config()).expect("valid Qwen2 config parses");
        let config = normalized.architecture_config;
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 64, "head_dim must be derived: 896 / 14");
        assert_eq!(config.vocab_size, 151936);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.bos_token_id, Some(151643));
        assert_eq!(config.eos_token_id, Some(151645));
        assert_eq!(normalized.metadata.model_type, "qwen2");
        assert_eq!(normalized.metadata.torch_dtype.as_deref(), Some("bfloat16"));
    }

    #[test]
    fn rejects_missing_required_field() {
        let error = parse(br#"{"model_type": "qwen2"}"#).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn rejects_invalid_head_kv_relationship() {
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        config["num_key_value_heads"] = serde_json::json!(3); // 14 % 3 != 0
        let bytes = serde_json::to_vec(&config).unwrap();
        let error = parse(&bytes).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn rejects_malformed_json() {
        let error = parse(b"not json").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn torch_dtype_is_preserved_as_annotation_only() {
        let normalized = parse(&qwen2_0_5b_config()).unwrap();
        assert_eq!(
            normalized
                .metadata
                .annotations
                .get("torch_dtype")
                .map(String::as_str),
            Some("bfloat16")
        );
    }
}
