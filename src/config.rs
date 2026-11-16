//! Real Hugging Face-style `config.json` parsing, normalized into
//! `magnetar-runtime`'s generic [`ModelArchitectureConfig`] and
//! [`HfConfigMetadata`] contracts (`implement-production-qwen-model-loading`
//! task group 3, Decision 5).

use magnetar_runtime::model::{
    LayerSpec, MixtureOfExpertsConfig, ModelArchitectureConfig, RopeFrequencyScaling,
};
use magnetar_runtime::model_format_roadmap::{HfConfigMetadata, HfRopeMetadata};
use magnetar_runtime::operator::RopeBandedFrequencyScaling;
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
    // Layered decoders (`per-layer-kv-geometry`): read by `layered_config`.
    /// One `"sliding_attention"` or `"full_attention"` per layer.
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    sliding_window: Option<u32>,
    /// The head width of `full_attention` layers, when it differs.
    #[serde(default)]
    global_head_dim: Option<u64>,
    /// The KV heads of `full_attention` layers, used with `attention_k_eq_v`.
    #[serde(default)]
    num_global_key_value_heads: Option<u32>,
    /// `full_attention` layers have no V projection: V is the raw K
    /// projection.
    #[serde(default)]
    attention_k_eq_v: Option<bool>,
    /// RoPE, either flat (`{"rope_theta": ..., "rope_type": ...}`, newer
    /// `transformers` releases) or per layer type
    /// (`{"sliding_attention": {...}, ...}`); read by [`rope_parameters`].
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    /// The last this many layers attend over the K/V of the last earlier
    /// layer of their type.
    #[serde(default)]
    num_kv_shared_layers: Option<u32>,
    /// KV-shared layers' feed-forward is twice `intermediate_size`.
    #[serde(default)]
    use_double_wide_mlp: Option<bool>,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    /// Per-layer input embeddings' width; `0` means none.
    #[serde(default)]
    hidden_size_per_layer_input: Option<u32>,
    /// `"all"` makes every token attend bidirectionally.
    #[serde(default)]
    use_bidirectional_attention: Option<String>,
    #[serde(default)]
    enable_moe_block: Option<bool>,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    top_k_experts: Option<u32>,
    #[serde(default)]
    moe_intermediate_size: Option<u64>,
    #[serde(default)]
    expert_intermediate_size: Option<u64>,
}

/// One layer type's RoPE in `rope_parameters`, or the whole of a flat one.
#[derive(Debug, Default, Deserialize)]
struct RawRopeParameters {
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawRopeScaling {
    #[serde(default, rename = "type")]
    scaling_type: Option<String>,
    /// The newer spelling of `type` (`transformers` >= 4.43).
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    factor: Option<f64>,
    #[serde(default)]
    low_freq_factor: Option<f64>,
    #[serde(default)]
    high_freq_factor: Option<f64>,
    #[serde(default)]
    original_max_position_embeddings: Option<u64>,
}

impl RawRopeScaling {
    fn kind(&self) -> Option<&str> {
        self.rope_type.as_deref().or(self.scaling_type.as_deref())
    }
}

/// The normalized RoPE scaling a `rope_scaling` object declares: a linear
/// position factor (`linear`, or a bare `factor` with no type -- this
/// loader's behaviour before types were read), a per-frequency scheme
/// (`llama3`), or nothing (`default`). Any other scheme (`yarn`, `dynamic`,
/// `longrope`, ...) is refused: ingesting it as unscaled or as a plain
/// factor would load a checkpoint whose every token is then wrong.
fn rope_scaling(
    raw: Option<&RawRopeScaling>,
) -> Result<(Option<f32>, Option<RopeFrequencyScaling>), ProductionIngestionError> {
    let Some(raw) = raw else {
        return Ok((None, None));
    };
    match raw.kind() {
        None | Some("linear") => Ok((raw.factor.map(|value| value as f32), None)),
        Some("default") => Ok((None, None)),
        Some("llama3") => {
            let field = |value: Option<f64>, name: &str| {
                value.ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                    reason: format!("config.json's llama3 rope_scaling has no '{name}'"),
                })
            };
            let scaling = RopeBandedFrequencyScaling {
                factor: field(raw.factor, "factor")?,
                low_freq_factor: field(raw.low_freq_factor, "low_freq_factor")?,
                high_freq_factor: field(raw.high_freq_factor, "high_freq_factor")?,
                original_context_length: raw.original_max_position_embeddings.ok_or_else(|| {
                    ProductionIngestionError::MalformedMetadata {
                        reason: "config.json's llama3 rope_scaling has no \
                                 'original_max_position_embeddings'"
                            .into(),
                    }
                })?,
            };
            scaling
                .validate()
                .map_err(|reason| ProductionIngestionError::MalformedMetadata { reason })?;
            Ok((None, Some(RopeFrequencyScaling::Banded(scaling))))
        }
        Some(other) => Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!("config.json's rope_scaling type '{other}' is not supported"),
        }),
    }
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

/// `rope_parameters` read as one entry per layer type (every value an
/// object), or as one flat entry under the key `""`.
fn rope_parameters(
    raw: &RawHfConfig,
) -> Result<BTreeMap<String, RawRopeParameters>, ProductionIngestionError> {
    let malformed = |error: serde_json::Error| ProductionIngestionError::MalformedMetadata {
        reason: format!("config.json's rope_parameters is malformed: {error}"),
    };
    let Some(value) = &raw.rope_parameters else {
        return Ok(BTreeMap::new());
    };
    let per_type = value
        .as_object()
        .is_some_and(|entries| !entries.is_empty() && entries.values().all(|v| v.is_object()));
    if per_type {
        serde_json::from_value(value.clone()).map_err(malformed)
    } else {
        Ok(BTreeMap::from([(
            String::new(),
            serde_json::from_value(value.clone()).map_err(malformed)?,
        )]))
    }
}

/// A layered decoder's per-layer geometry (`None` when every layer matches
/// the architecture-wide one), and the RoPE base of its sliding layers (the
/// architecture-wide `rope_theta`), kept even when the layout is dropped.
struct LayeredConfig {
    layout: Option<Vec<LayerSpec>>,
    rope_theta: f64,
}

/// The per-layer geometry of a decoder whose `config.json` declares
/// `layer_types` (`per-layer-kv-geometry`); `None` without `layer_types`.
/// Follows `transformers`' own derivation:
/// - a `sliding_attention` layer attends over `sliding_window` tokens, a
///   `full_attention` layer over the whole context;
/// - `full_attention` layers take `global_head_dim` when declared and, with
///   `attention_k_eq_v`, `num_global_key_value_heads` and no V projection;
/// - each layer type takes its `rope_parameters` entry: `default` rotates
///   the whole head, `proportional` its `partial_rotary_factor`;
/// - the last `num_kv_shared_layers` layers reuse the K/V of the last
///   earlier layer of their type, with a twice-wide feed-forward under
///   `use_double_wide_mlp`.
fn layered_config(
    raw: &RawHfConfig,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    head_dim: u64,
) -> Result<Option<LayeredConfig>, ProductionIngestionError> {
    let Some(layer_types) = &raw.layer_types else {
        return Ok(None);
    };
    let malformed = |reason: String| ProductionIngestionError::MalformedMetadata { reason };
    if layer_types.len() != num_hidden_layers as usize {
        return Err(malformed(format!(
            "config.json's layer_types has {} entries for {num_hidden_layers} layers",
            layer_types.len()
        )));
    }
    if raw.use_bidirectional_attention.as_deref() == Some("all") {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "config.json declares bidirectional attention for every token; only causal \
                     decoding is supported"
                .into(),
        });
    }
    let intermediate_size = raw.intermediate_size.unwrap_or_default();
    let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
    let k_eq_v = raw.attention_k_eq_v.unwrap_or(false);
    let rope_parameters = rope_parameters(raw)?;
    let flat_theta = rope_parameters.get("").and_then(|flat| flat.rope_theta);
    let default_theta = raw.rope_theta.or(flat_theta).unwrap_or(10_000.0);
    let rope = |layer_type: &str| -> Result<(f64, Option<f64>), ProductionIngestionError> {
        let Some(parameters) = rope_parameters.get(layer_type) else {
            return Ok((default_theta, None));
        };
        let theta = parameters.rope_theta.unwrap_or(default_theta);
        let partial = match parameters.rope_type.as_deref() {
            None | Some("default") => None,
            Some("proportional") => parameters.partial_rotary_factor,
            Some(other) => {
                return Err(ProductionIngestionError::UnsupportedFormat {
                    reason: format!(
                        "config.json's {layer_type} rope_type '{other}' is not supported"
                    ),
                });
            }
        };
        Ok((theta, partial.filter(|factor| *factor != 1.0)))
    };
    let shared = raw.num_kv_shared_layers.unwrap_or(0) as usize;
    let first_shared = layer_types.len().checked_sub(shared).ok_or_else(|| {
        malformed(format!(
            "config.json shares the K/V of {shared} of {num_hidden_layers} layers"
        ))
    })?;
    let mut layout = Vec::with_capacity(layer_types.len());
    for (index, layer_type) in layer_types.iter().enumerate() {
        let full = match layer_type.as_str() {
            "full_attention" => true,
            "sliding_attention" => false,
            other => {
                return Err(ProductionIngestionError::UnsupportedFormat {
                    reason: format!("config.json's layer type '{other}' is not supported"),
                });
            }
        };
        let sliding_window = if full {
            None
        } else {
            Some(raw.sliding_window.ok_or_else(|| {
                malformed("config.json has sliding layers but no sliding_window".into())
            })?)
        };
        let (rope_theta, partial_rotary_factor) = rope(layer_type)?;
        let kv_source_layer = if index >= first_shared {
            let producer = layer_types[..first_shared]
                .iter()
                .rposition(|producer| producer == layer_type)
                .ok_or_else(|| {
                    malformed(format!(
                        "config.json's layer {index} shares the K/V of an earlier \
                         {layer_type} layer, but none computes its own"
                    ))
                })?;
            Some(producer as u32)
        } else {
            None
        };
        let double_wide = kv_source_layer.is_some() && raw.use_double_wide_mlp.unwrap_or(false);
        layout.push(LayerSpec {
            sliding_window,
            head_dim: if full {
                raw.global_head_dim.unwrap_or(head_dim)
            } else {
                head_dim
            },
            num_key_value_heads: match raw.num_global_key_value_heads {
                Some(heads) if full && k_eq_v => heads,
                _ => num_key_value_heads,
            },
            rope_theta,
            partial_rotary_factor,
            value_from_key: full && k_eq_v,
            kv_source_layer,
            intermediate_size: intermediate_size * if double_wide { 2 } else { 1 },
        });
    }
    let rope_theta = layout
        .iter()
        .find(|layer| layer.sliding_window.is_some())
        .or(layout.first())
        .map_or(default_theta, |layer| layer.rope_theta);
    // Layers that all match the architecture-wide geometry need no layout:
    // e.g. a uniform decoder whose `layer_types` lists only full attention.
    let uniform = LayerSpec {
        sliding_window: None,
        head_dim,
        num_key_value_heads,
        rope_theta,
        partial_rotary_factor: None,
        value_from_key: false,
        kv_source_layer: None,
        intermediate_size,
    };
    let layout = (!layout.iter().all(|layer| *layer == uniform)).then_some(layout);
    Ok(Some(LayeredConfig { layout, rope_theta }))
}

/// The routed experts a layered decoder's `enable_moe_block` declares,
/// alongside its dense MLP: each token's top `top_k_experts` of
/// `num_experts`, renormalized, no shared expert.
fn mixture_of_experts(
    raw: &RawHfConfig,
) -> Result<Option<MixtureOfExpertsConfig>, ProductionIngestionError> {
    if !raw.enable_moe_block.unwrap_or(false) {
        return Ok(None);
    }
    let field = |value: Option<u64>, name: &str| {
        value.ok_or_else(|| ProductionIngestionError::MalformedMetadata {
            reason: format!("config.json enables a mixture of experts but has no '{name}'"),
        })
    };
    let count = |value: Option<u32>, name: &str| field(value.map(u64::from), name);
    Ok(Some(MixtureOfExpertsConfig {
        experts: count(raw.num_experts, "num_experts")? as u32,
        experts_per_token: count(raw.top_k_experts, "top_k_experts")? as u32,
        expert_intermediate_size: field(
            raw.moe_intermediate_size.or(raw.expert_intermediate_size),
            "moe_intermediate_size",
        )?,
        shared_expert_intermediate_size: None,
        normalize_top_k: true,
    }))
}

/// Parses and normalizes `bytes` as a Hugging Face-style `config.json`.
/// Rejects missing/zero/internally-inconsistent architecture values with
/// structured errors (task 3.3) -- `head_dim`, when absent, is derived as
/// `hidden_size / num_attention_heads` and the derivation is validated to
/// divide evenly, matching the proposal's "head_dim (or validated
/// derivation)".
pub fn parse(bytes: &[u8]) -> Result<NormalizedHfConfig, ProductionIngestionError> {
    let shape_error = |error: serde_json::Error| ProductionIngestionError::MalformedMetadata {
        reason: format!("config.json is not valid JSON matching the expected shape: {error}"),
    };
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(shape_error)?;
    // A multimodal bundle nests its text decoder's configuration under
    // `text_config`; the bundle's own `model_type` and `architectures` name
    // the checkpoint.
    let nested = value
        .get("text_config")
        .filter(|_| value.get("hidden_size").is_none())
        .cloned();
    let outer_model_type = nested
        .as_ref()
        .and(value.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let outer_architectures: Option<Vec<String>> = nested.as_ref().and_then(|_| {
        value
            .get("architectures")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    });
    let mut raw: RawHfConfig =
        serde_json::from_value(nested.unwrap_or(value)).map_err(shape_error)?;
    if let Some(model_type) = outer_model_type {
        raw.model_type = Some(model_type);
    }
    if let Some(architectures) = outer_architectures {
        raw.architectures = architectures;
    }

    let hidden_size = required(raw.hidden_size, "hidden_size")?;
    let intermediate_size = required(raw.intermediate_size, "intermediate_size")?;
    let num_hidden_layers = required(raw.num_hidden_layers, "num_hidden_layers")?;
    let num_attention_heads = required(raw.num_attention_heads, "num_attention_heads")?;
    let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
    let vocab_size = required(raw.vocab_size, "vocab_size")?;
    let model_type = required(raw.model_type.clone(), "model_type")?;

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
    let layered = layered_config(&raw, num_hidden_layers, num_attention_heads, head_dim)?;
    // `rope_theta`, else a flat `rope_parameters`' (newer `transformers`
    // releases moved it there), else a layered decoder's sliding layers'.
    let flat_rope_theta = rope_parameters(&raw)?
        .get("")
        .and_then(|flat| flat.rope_theta);
    let rope_theta = raw
        .rope_theta
        .or(flat_rope_theta)
        .or(layered.as_ref().map(|layered| layered.rope_theta))
        .unwrap_or(10_000.0);
    if rope_theta <= 0.0 {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json's rope_theta must be positive".into(),
        });
    }

    let (rope_scaling_factor, rope_frequency_scaling) = rope_scaling(raw.rope_scaling.as_ref())?;
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
        rope_scaling_factor,
        rope_frequency_scaling,
        tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
        // `config.json` never declares this for real Qwen2/2.5 checkpoints
        // (it's an architectural default of the model class, not a
        // config field) -- `false` here is a placeholder `lib.rs::ingest`
        // overwrites once real weight discovery has actually looked for
        // `self_attn.{q,k,v}_proj` bias tensors, the only real evidence
        // this ever needs.
        attention_bias: false,
        mixture_of_experts: mixture_of_experts(&raw)?,
        bos_token_id: raw.bos_token_id,
        eos_token_id: raw.eos_token_id.as_ref().and_then(RawEosTokenId::first),
        layer_layout: layered.and_then(|layered| layered.layout),
        final_logit_softcapping: raw.final_logit_softcapping,
        per_layer_input_size: raw.hidden_size_per_layer_input.filter(|size| *size > 0),
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
                .and_then(|scaling| scaling.kind().map(str::to_string)),
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

    /// Llama 3.2 1B Instruct's real `config.json` (rope and shape fields).
    fn llama3_2_1b_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "max_position_embeddings": 131072
        })
    }

    #[test]
    fn parses_llama3_rope_scaling_as_a_frequency_scheme_not_a_position_factor() {
        let bytes = serde_json::to_vec(&llama3_2_1b_config()).unwrap();
        let normalized = parse(&bytes).expect("Llama 3.2 config parses");
        let config = normalized.architecture_config;
        assert_eq!(config.rope_scaling_factor, None);
        assert_eq!(
            config.rope_frequency_scaling,
            Some(RopeFrequencyScaling::Banded(RopeBandedFrequencyScaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_context_length: 8192,
            }))
        );
        assert_eq!(normalized.metadata.model_type, "llama");
        assert_eq!(
            normalized
                .metadata
                .rope
                .and_then(|rope| rope.scaling_type)
                .as_deref(),
            Some("llama3")
        );
    }

    #[test]
    fn rejects_incomplete_or_unknown_rope_scaling() {
        let mut incomplete = llama3_2_1b_config();
        incomplete["rope_scaling"]
            .as_object_mut()
            .unwrap()
            .remove("low_freq_factor");
        assert!(matches!(
            parse(&serde_json::to_vec(&incomplete).unwrap()).unwrap_err(),
            ProductionIngestionError::MalformedMetadata { .. }
        ));
        let mut yarn = llama3_2_1b_config();
        yarn["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0});
        assert!(matches!(
            parse(&serde_json::to_vec(&yarn).unwrap()).unwrap_err(),
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
        let mut default = llama3_2_1b_config();
        default["rope_scaling"] = serde_json::json!({"rope_type": "default"});
        let config = parse(&serde_json::to_vec(&default).unwrap())
            .unwrap()
            .architecture_config;
        assert_eq!(
            (config.rope_scaling_factor, config.rope_frequency_scaling),
            (None, None)
        );
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
    fn rejects_wrong_field_type() {
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        config["hidden_size"] = serde_json::json!("not-a-number");
        let bytes = serde_json::to_vec(&config).unwrap();
        let error = parse(&bytes).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn rejects_zero_vocab_size() {
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        config["vocab_size"] = serde_json::json!(0);
        let bytes = serde_json::to_vec(&config).unwrap();
        let error = parse(&bytes).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn rejects_non_positive_rope_theta() {
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        config["rope_theta"] = serde_json::json!(-1.0);
        let bytes = serde_json::to_vec(&config).unwrap();
        let error = parse(&bytes).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn derives_head_dim_only_when_evenly_divisible() {
        let mut config: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        config["hidden_size"] = serde_json::json!(897); // not divisible by 14
        let bytes = serde_json::to_vec(&config).unwrap();
        let error = parse(&bytes).unwrap_err();
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

/// `per-layer-kv-geometry`: the real `config.json` files of the Gemma 4
/// checkpoints (`google/gemma-4-*-it`, Apache-2.0), whose layer geometry
/// the per-layer tensor shapes of the same checkpoints confirm.
#[cfg(test)]
mod layered_tests {
    use super::*;

    fn parse_testdata(name: &str) -> ModelArchitectureConfig {
        let path = format!(
            "{}/testdata/gemma4/{name}.config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        parse(&std::fs::read(path).unwrap())
            .unwrap()
            .architecture_config
    }

    /// `gemma-4-12B-it` declares `gemma4_unified`, whose text decoder is
    /// Gemma 4's dense one: five sliding layers to one full layer, full
    /// layers deriving V from K, no per-layer inputs, no experts.
    #[test]
    fn a_unified_bundle_takes_its_text_decoders_layout() {
        let path = format!(
            "{}/testdata/gemma4/gemma-4-12B-it.config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let parsed = parse(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(parsed.metadata.model_type, "gemma4_unified");
        let config = parsed.architecture_config;
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.hidden_size, 3840);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.final_logit_softcapping, Some(30.0));
        assert_eq!(config.per_layer_input_size, None);
        assert_eq!(config.mixture_of_experts, None);
        let layout = config.layer_layout.as_ref().unwrap();
        for (index, layer) in layout.iter().enumerate() {
            let full = index % 6 == 5;
            assert_eq!(layer.sliding_window.is_none(), full, "layer {index}");
            assert_eq!(layer.kv_source_layer, None, "layer {index}");
            assert_eq!(layer.intermediate_size, 15360, "layer {index}");
            if full {
                assert_eq!(layer.head_dim, 512);
                assert_eq!(layer.num_key_value_heads, 1);
                assert!(layer.value_from_key);
                assert_eq!(layer.rope_theta, 1_000_000.0);
                assert_eq!(layer.partial_rotary_factor, Some(0.25));
            } else {
                assert_eq!(layer.sliding_window, Some(1024));
                assert_eq!(layer.head_dim, 256);
                assert_eq!(layer.num_key_value_heads, 8);
                assert!(!layer.value_from_key);
                assert_eq!(layer.rope_theta, 10_000.0);
                assert_eq!(layer.partial_rotary_factor, None);
            }
        }
    }

    #[test]
    fn e2b_layers_slide_share_kv_and_widen_their_mlp() {
        let config = parse_testdata("gemma-4-E2B-it");
        assert_eq!(config.num_hidden_layers, 35);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.final_logit_softcapping, Some(30.0));
        assert_eq!(config.per_layer_input_size, Some(256));
        assert_eq!(config.mixture_of_experts, None);
        let layout = config.layer_layout.as_ref().unwrap();
        // A sliding layer: window 512, 256-wide heads, full rotation.
        let sliding = &layout[0];
        assert_eq!(sliding.sliding_window, Some(512));
        assert_eq!(sliding.head_dim, 256);
        assert_eq!(sliding.num_key_value_heads, 1);
        assert_eq!(sliding.rope_theta, 10_000.0);
        assert_eq!(sliding.partial_rotary_factor, None);
        assert_eq!(sliding.intermediate_size, 6144);
        // A full layer: whole context, 512-wide heads (q_norm `[512]`),
        // a quarter of each head rotated with base 1e6, its own V.
        let full = &layout[4];
        assert_eq!(full.sliding_window, None);
        assert_eq!(full.head_dim, 512);
        assert_eq!(full.num_key_value_heads, 1);
        assert_eq!(full.rope_theta, 1_000_000.0);
        assert_eq!(full.partial_rotary_factor, Some(0.25));
        assert!(!full.value_from_key);
        // The last 20 layers reuse the K/V of the last earlier layer of
        // their type (13 sliding, 14 full) and have a 12288-wide MLP.
        assert!(
            layout[..15]
                .iter()
                .all(|layer| layer.kv_source_layer.is_none())
        );
        assert_eq!(layout[15].kv_source_layer, Some(13));
        assert_eq!(layout[19].kv_source_layer, Some(14));
        assert_eq!(layout[34].kv_source_layer, Some(14));
        assert!(
            layout[15..]
                .iter()
                .all(|layer| layer.intermediate_size == 12288)
        );
        config.validate().unwrap();
    }

    #[test]
    fn full_layers_of_k_eq_v_checkpoints_take_global_kv_heads_and_no_v() {
        // 31B: `num_global_key_value_heads = 4`, `attention_k_eq_v`.
        let config = parse_testdata("gemma-4-31B-it");
        let layout = config.layer_layout.as_ref().unwrap();
        assert_eq!(layout.len(), 60);
        let full: Vec<usize> = (0..60)
            .filter(|i| layout[*i].sliding_window.is_none())
            .collect();
        assert_eq!(full, [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);
        for (index, layer) in layout.iter().enumerate() {
            let is_full = full.contains(&index);
            assert_eq!(layer.num_key_value_heads, if is_full { 4 } else { 16 });
            assert_eq!(layer.head_dim, if is_full { 512 } else { 256 });
            assert_eq!(layer.value_from_key, is_full);
            assert_eq!(layer.sliding_window, (!is_full).then_some(1024));
            assert_eq!(layer.kv_source_layer, None);
        }
        assert_eq!(config.per_layer_input_size, None);
    }

    #[test]
    fn a_mixture_of_experts_checkpoint_routes_renormalized_experts() {
        let config = parse_testdata("gemma-4-26B-A4B-it");
        assert_eq!(
            config.mixture_of_experts,
            Some(MixtureOfExpertsConfig {
                experts: 128,
                experts_per_token: 8,
                expert_intermediate_size: 704,
                shared_expert_intermediate_size: None,
                normalize_top_k: true,
            })
        );
        let layout = config.layer_layout.as_ref().unwrap();
        // 25 sliding layers have a V projection, 5 full ones derive V.
        assert_eq!(
            layout.iter().filter(|layer| !layer.value_from_key).count(),
            25
        );
        assert_eq!(layout[5].num_key_value_heads, 2);
        assert_eq!(layout[0].num_key_value_heads, 8);
    }

    #[test]
    fn e4b_shares_kv_without_widening_its_mlp() {
        let config = parse_testdata("gemma-4-E4B-it");
        let layout = config.layer_layout.as_ref().unwrap();
        assert_eq!(layout.len(), 42);
        // 18 shared layers; the last unshared ones are 22 (sliding) and
        // 23 (full).
        assert_eq!(layout[24].kv_source_layer, Some(22));
        assert_eq!(layout[29].kv_source_layer, Some(23));
        assert!(layout.iter().all(|layer| layer.intermediate_size == 10240));
    }

    #[test]
    fn unsupported_layered_forms_are_refused() {
        let mut value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!(
                "{}/testdata/gemma4/gemma-4-31B-it.config.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        )
        .unwrap();
        let text = value.get_mut("text_config").unwrap();
        text["use_bidirectional_attention"] = "all".into();
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            parse(&bytes),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
        let text = value.get_mut("text_config").unwrap();
        text["use_bidirectional_attention"] = serde_json::Value::Null;
        text["rope_parameters"]["full_attention"]["rope_type"] = "yarn".into();
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            parse(&bytes),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
        let text = value.get_mut("text_config").unwrap();
        text["rope_parameters"]["full_attention"]["rope_type"] = "proportional".into();
        text["layer_types"].as_array_mut().unwrap().pop();
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            parse(&bytes),
            Err(ProductionIngestionError::MalformedMetadata { .. })
        ));
    }

    /// A uniform decoder listing only full-attention layers, with a flat
    /// `rope_parameters` (newer `transformers` releases), keeps no layout
    /// and takes its RoPE base from `rope_parameters`.
    #[test]
    fn uniform_layer_types_and_flat_rope_parameters_add_no_layout() {
        let bytes = serde_json::to_vec(&serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 2,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "layer_types": ["full_attention", "full_attention"],
            "sliding_window": null,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}
        }))
        .unwrap();
        let config = parse(&bytes).unwrap().architecture_config;
        assert_eq!(config.layer_layout, None);
        assert_eq!(config.rope_theta, 1_000_000.0);
    }

    /// A uniform decoder whose only RoPE base is its per-type
    /// `rope_parameters` entry keeps that base when its layout is dropped.
    #[test]
    fn a_dropped_uniform_layout_keeps_its_per_type_rope_base() {
        let bytes = serde_json::to_vec(&serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 2,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "layer_types": ["full_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"}
            }
        }))
        .unwrap();
        let config = parse(&bytes).unwrap().architecture_config;
        assert_eq!(config.layer_layout, None);
        assert_eq!(config.rope_theta, 1_000_000.0);
    }

    #[test]
    fn a_nested_text_config_keeps_the_bundle_model_type() {
        let bytes = std::fs::read(format!(
            "{}/testdata/gemma4/gemma-4-31B-it.config.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed.metadata.model_type, "gemma4");
        assert_eq!(
            parsed.metadata.architectures,
            vec!["Gemma4ForConditionalGeneration".to_string()]
        );
        assert_eq!(parsed.architecture_config.hidden_size, 5376);
        assert!(parsed.architecture_config.tie_word_embeddings);
    }
}
