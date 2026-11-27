//! Real Hugging Face-style `config.json` parsing, normalized into
//! `magnetar-runtime`'s generic [`ModelArchitectureConfig`] and
//! [`HfConfigMetadata`] contracts (`implement-production-qwen-model-loading`
//! task group 3, Decision 5).

use magnetar_runtime::model::{
    DecoderExtensions, ExpertGroupScore, ExpertGroups, ExpertNormalization, ExpertRoutingConfig,
    ExpertScoring, HyperConnectionConfig, LatentAttentionConfig, LayerSpec, LinearAttentionConfig,
    MixtureOfExpertsConfig, ModelArchitectureConfig, NgramEmbeddingConfig, RopeFrequencyScaling,
    TokenSelectionConfig,
};
use magnetar_runtime::model_format_roadmap::{HfConfigMetadata, HfRopeMetadata};
use magnetar_runtime::operator::{RopeBandedFrequencyScaling, RopeYarnFrequencyScaling};
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
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    no_rope_layers: Option<Vec<u8>>,
    #[serde(default)]
    no_rope_layer_interval: Option<u32>,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    #[serde(default)]
    mlp_bias: Option<bool>,
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
    // Multi-head latent attention and its experts (`express-latent-
    // attention-forms`): read by `latent_attention` and `routed_experts`.
    #[serde(default)]
    q_lora_rank: Option<u32>,
    #[serde(default)]
    kv_lora_rank: Option<u32>,
    #[serde(default)]
    qk_nope_head_dim: Option<u64>,
    #[serde(default)]
    qk_rope_head_dim: Option<u64>,
    #[serde(default)]
    v_head_dim: Option<u64>,
    #[serde(default)]
    n_routed_experts: Option<u32>,
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
    #[serde(default)]
    n_shared_experts: Option<u64>,
    #[serde(default)]
    first_k_dense_replace: Option<u32>,
    #[serde(default)]
    moe_layer_freq: Option<u32>,
    /// Explicit sparse/dense MLP pattern used by newer GLM configs.
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
    /// `greedy`, `group_limited_greedy` (groups by their best expert) or
    /// `noaux_tc` (groups by their two best, with a selection bias).
    #[serde(default)]
    topk_method: Option<String>,
    #[serde(default)]
    n_group: Option<u32>,
    #[serde(default)]
    topk_group: Option<u32>,
    #[serde(default)]
    routed_scaling_factor: Option<f64>,
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    /// `softmax` (absent) or `sigmoid`.
    #[serde(default)]
    scoring_func: Option<String>,
    #[serde(default)]
    rope_interleave: Option<bool>,
    // Qwen3.5's hybrid decoder (`support-qwen38-flash-next`): read by
    // `qwen35_config`.
    #[serde(default)]
    linear_num_key_heads: Option<u32>,
    #[serde(default)]
    linear_num_value_heads: Option<u32>,
    #[serde(default)]
    linear_key_head_dim: Option<u64>,
    #[serde(default)]
    linear_value_head_dim: Option<u64>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<u32>,
    #[serde(default)]
    attn_output_gate: Option<bool>,
    #[serde(default)]
    full_attention_interval: Option<u32>,
    // Qwen4-Exp's residual streams, n-gram embeddings, indexer and linear
    // gate (`support-qwen38-flash-next`): read by `qwen35_config`.
    #[serde(default)]
    hc_count: Option<u32>,
    #[serde(default)]
    hc_lowrank: Option<u32>,
    #[serde(default)]
    ple_layer_ids: Option<Vec<u32>>,
    #[serde(default)]
    ple_embed_dim: Option<u64>,
    #[serde(default)]
    ple_conv_kernel_size: Option<u32>,
    #[serde(default)]
    ngram_size: Option<u32>,
    #[serde(default)]
    heads_per_ngram: Option<u32>,
    #[serde(default)]
    ngram_vocab_size_base: Option<u64>,
    #[serde(default)]
    make_ngram_vocab_size_divisible_by: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    indexer_n_heads: Option<u32>,
    #[serde(default)]
    indexer_kv_heads: Option<u32>,
    #[serde(default)]
    indexer_head_dim: Option<u64>,
    #[serde(default)]
    indexer_budget: Option<u32>,
    #[serde(default)]
    indexer_compress_ratio: Option<u32>,
    #[serde(default)]
    output_gate_type: Option<String>,
    #[serde(default)]
    shared_expert_intermediate_size: Option<u64>,
}

/// Qwen3.5's hybrid decoder, read from its text configuration: which layers
/// are gated-delta linear attention, their mixer geometry, and the layout of
/// its full-attention layers (partial rotary). Its norms are zero-centered
/// and its attention output is gated by default.
struct Qwen35Config {
    linear_attention: LinearAttentionConfig,
    layout: Vec<LayerSpec>,
    rope_theta: f64,
    attention_output_gate: bool,
    /// Qwen4-Exp's `1.14.0` forms; empty for Qwen3.5.
    extensions: DecoderExtensions,
    /// Qwen4-Exp's every-layer mixture of experts. Its routing is the
    /// `1.5.0` one (softmax over every expert, a gated shared expert), so
    /// it declares no `expert-routing`.
    experts: Option<MixtureOfExpertsConfig>,
}

/// [`Qwen35Config`] for `qwen3_5`/`qwen3_5_text` and, with its residual
/// streams, n-gram embeddings and indexer, `qwen4_exp`/`qwen4_exp_text`,
/// taking the `layer_types` it reads out of `raw` so the layered decoders'
/// reading never sees `linear_attention` entries. `None` for other model
/// types.
fn qwen35_config(
    raw: &mut RawHfConfig,
    model_type: &str,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    head_dim: u64,
    intermediate_size: u64,
) -> Result<Option<Qwen35Config>, ProductionIngestionError> {
    let qwen4 = matches!(model_type, "qwen4_exp" | "qwen4_exp_text");
    if !qwen4 && !matches!(model_type, "qwen3_5" | "qwen3_5_text") {
        return Ok(None);
    }
    let malformed = |reason: String| ProductionIngestionError::MalformedMetadata { reason };
    if num_hidden_layers > 4096 {
        return Err(malformed(
            "config.json declares too many decoder layers".into(),
        ));
    }
    let layer_types = match raw.layer_types.take() {
        Some(types) => types,
        None => {
            let interval = raw.full_attention_interval.unwrap_or(4);
            if interval == 0 {
                return Err(malformed(
                    "config.json's full_attention_interval must be positive".into(),
                ));
            }
            (0..num_hidden_layers)
                .map(|layer| {
                    if (layer + 1) % interval == 0 {
                        "full_attention"
                    } else {
                        "linear_attention"
                    }
                    .to_string()
                })
                .collect()
        }
    };
    if layer_types.len() != num_hidden_layers as usize {
        return Err(malformed(format!(
            "config.json's layer_types has {} entries for {num_hidden_layers} layers",
            layer_types.len()
        )));
    }
    let layers = layer_types
        .iter()
        .map(|kind| match kind.as_str() {
            "linear_attention" => Ok(true),
            "full_attention" => Ok(false),
            // Qwen4-Exp's full attention reads its tokens through an indexer.
            "indexed_attention" if qwen4 => Ok(false),
            other => Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!("config.json's Qwen3.5 layer type '{other}' is not supported"),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let required_u32 = |value: Option<u32>, field: &str| required(value, field);
    let linear_attention = LinearAttentionConfig {
        layers,
        key_heads: required_u32(raw.linear_num_key_heads, "linear_num_key_heads")?,
        value_heads: required_u32(raw.linear_num_value_heads, "linear_num_value_heads")?,
        key_head_dim: required(raw.linear_key_head_dim, "linear_key_head_dim")?,
        value_head_dim: required(raw.linear_value_head_dim, "linear_value_head_dim")?,
        conv_kernel: raw.linear_conv_kernel_dim.unwrap_or(4),
    };
    // Text positions are equal on every M-RoPE axis, so the interleaved
    // sections reduce to one rotation of the first
    // `partial_rotary_factor * head_dim` values.
    let parameters = rope_parameters(raw)?;
    let flat = parameters.get("");
    let rope_theta = raw
        .rope_theta
        .or(flat.and_then(|flat| flat.rope_theta))
        .unwrap_or(10_000_000.0);
    if let Some(kind) = flat.and_then(|flat| flat.rope_type.as_deref())
        && kind != "default"
    {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!("config.json's Qwen3.5 rope_type '{kind}' is not supported"),
        });
    }
    // Qwen3.5 rotates a quarter of each head by default, Qwen4-Exp all of it.
    let partial_rotary_factor = flat
        .and_then(|flat| flat.partial_rotary_factor)
        .unwrap_or(if qwen4 { 1.0 } else { 0.25 });
    if !(partial_rotary_factor > 0.0 && partial_rotary_factor <= 1.0) {
        return Err(malformed(format!(
            "config.json's partial_rotary_factor {partial_rotary_factor} must lie in (0, 1]"
        )));
    }
    let layer = LayerSpec {
        sliding_window: None,
        head_dim,
        num_key_value_heads,
        rope_theta,
        partial_rotary_factor: (partial_rotary_factor != 1.0).then_some(partial_rotary_factor),
        value_from_key: false,
        kv_source_layer: None,
        intermediate_size,
    };
    let (extensions, experts) = if qwen4 {
        let (extensions, experts) = qwen4_exp_forms(raw, &linear_attention, head_dim)?;
        (extensions, Some(experts))
    } else {
        (DecoderExtensions::default(), None)
    };
    Ok(Some(Qwen35Config {
        linear_attention,
        layout: vec![layer; num_hidden_layers as usize],
        rope_theta,
        // Qwen4-Exp's attention is always gated.
        attention_output_gate: qwen4 || raw.attn_output_gate.unwrap_or(true),
        extensions,
        experts,
    }))
}

/// Qwen4-Exp's decoder forms: its residual streams, n-gram embeddings,
/// indexer, the linear gate's activation, and its mixture of experts (every
/// layer, softmax routing, a shared expert behind a sigmoid gate).
fn qwen4_exp_forms(
    raw: &RawHfConfig,
    linear_attention: &LinearAttentionConfig,
    head_dim: u64,
) -> Result<(DecoderExtensions, MixtureOfExpertsConfig), ProductionIngestionError> {
    let malformed = |reason: String| ProductionIngestionError::MalformedMetadata { reason };
    let hyper_connections = HyperConnectionConfig {
        streams: raw.hc_count.unwrap_or(4),
        lowrank: raw.hc_lowrank.unwrap_or(320),
    };
    let layers = linear_attention.layers.len() as u32;
    let ple_layers = raw.ple_layer_ids.clone().unwrap_or_default();
    let ngram_embeddings = if ple_layers.is_empty() {
        None
    } else {
        let mut flags = vec![false; layers as usize];
        for id in ple_layers {
            // One-indexed in `config.json`.
            let slot = id
                .checked_sub(1)
                .filter(|slot| *slot < layers)
                .ok_or_else(|| {
                    malformed(format!(
                        "config.json's ple_layer_ids names layer {id} of {layers}"
                    ))
                })?;
            flags[slot as usize] = true;
        }
        let eos_token = raw
            .eos_token_id
            .as_ref()
            .and_then(RawEosTokenId::first)
            .ok_or_else(|| {
                malformed("config.json enables n-gram embeddings but has no eos_token_id".into())
            })?;
        Some(NgramEmbeddingConfig {
            layers: flags,
            embedding_dim: raw.ple_embed_dim.or(raw.hidden_size).ok_or_else(|| {
                malformed("config.json has neither ple_embed_dim nor hidden_size".into())
            })?,
            conv_kernel: raw.ple_conv_kernel_size.unwrap_or(4),
            ngram_size: raw.ngram_size.unwrap_or(3),
            heads_per_ngram: raw.heads_per_ngram.unwrap_or(8),
            vocabulary_base: raw.ngram_vocab_size_base.unwrap_or(20_000_000),
            vocabulary_multiple: raw.make_ngram_vocab_size_divisible_by.unwrap_or(128),
            seed: raw.seed.unwrap_or(1234),
            eos_token,
        })
    };
    let indexer = [
        raw.indexer_n_heads.is_some(),
        raw.indexer_kv_heads.is_some(),
        raw.indexer_head_dim.is_some(),
        raw.indexer_budget.is_some(),
        raw.indexer_compress_ratio.is_some(),
    ];
    let token_selection = if indexer.iter().any(|set| *set) {
        if !indexer.iter().all(|set| *set) {
            return Err(malformed(
                "config.json's indexer_* fields must all be set or all be absent".into(),
            ));
        }
        Some(TokenSelectionConfig {
            heads: raw.indexer_n_heads.unwrap_or_default(),
            key_heads: raw.indexer_kv_heads.unwrap_or_default(),
            head_dim: raw.indexer_head_dim.unwrap_or_default(),
            budget: raw.indexer_budget.unwrap_or_default(),
            compress_ratio: raw.indexer_compress_ratio.unwrap_or_default(),
        })
    } else {
        None
    };
    // The attention's rotary width must fit the indexer's head (it rotates
    // its queries and block keys the same way).
    if let Some(selection) = &token_selection {
        let factor = rope_parameters(raw)?
            .get("")
            .and_then(|flat| flat.partial_rotary_factor)
            .unwrap_or(1.0);
        if (head_dim as f64 * factor) as u64 > selection.head_dim {
            return Err(malformed(
                "config.json's rotary width does not fit the indexer's head width".into(),
            ));
        }
    }
    let sigmoid_linear_gate = match raw.output_gate_type.as_deref() {
        None | Some("silu") => false,
        Some("sigmoid") => true,
        Some(other) => {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!("config.json's output_gate_type '{other}' is not supported"),
            });
        }
    };
    let count = |value: Option<u32>, name: &str| {
        value.ok_or_else(|| {
            malformed(format!(
                "config.json declares a Qwen4-Exp decoder but no '{name}'"
            ))
        })
    };
    let experts = MixtureOfExpertsConfig {
        experts: count(raw.num_experts, "num_experts")?,
        experts_per_token: count(raw.num_experts_per_tok, "num_experts_per_tok")?,
        expert_intermediate_size: raw.moe_intermediate_size.ok_or_else(|| {
            malformed(
                "config.json declares a Qwen4-Exp decoder but no 'moe_intermediate_size'".into(),
            )
        })?,
        shared_expert_intermediate_size: Some(raw.shared_expert_intermediate_size.ok_or_else(
            || {
                malformed(
                    "config.json declares a Qwen4-Exp decoder but no \
                     'shared_expert_intermediate_size'"
                        .into(),
                )
            },
        )?),
        normalize_top_k: raw.norm_topk_prob.unwrap_or(true),
    };
    // New decoder forms must default to absent instead of preventing this
    // Qwen profile from compiling against a newer Runtime. On older Runtime
    // versions all fields happen to be specified, making the update redundant.
    #[allow(clippy::needless_update)]
    let extensions = DecoderExtensions {
        hyper_connections: Some(hyper_connections),
        ngram_embeddings,
        token_selection,
        sigmoid_linear_gate,
        ..Default::default()
    };
    Ok((extensions, experts))
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
    // YaRN.
    #[serde(default)]
    beta_fast: Option<f64>,
    #[serde(default)]
    beta_slow: Option<f64>,
    #[serde(default)]
    mscale: Option<f64>,
    #[serde(default)]
    mscale_all_dim: Option<f64>,
    #[serde(default)]
    attention_factor: Option<f64>,
    #[serde(default)]
    truncate: Option<bool>,
}

impl RawRopeScaling {
    fn kind(&self) -> Option<&str> {
        self.rope_type.as_deref().or(self.scaling_type.as_deref())
    }
}

/// The normalized RoPE scaling a `rope_scaling` object declares: a linear
/// position factor (`linear`, or a bare `factor` with no type -- this
/// loader's behaviour before types were read), a per-frequency scheme
/// (`llama3`, `yarn`), or nothing (`default`). Any other scheme (`dynamic`,
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
        Some("yarn") => Ok((None, Some(yarn_scaling(raw)?))),
        Some(other) => Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!("config.json's rope_scaling type '{other}' is not supported"),
        }),
    }
}

/// YaRN as `transformers` reads it (`_compute_yarn_parameters`): `beta_fast`
/// 32 and `beta_slow` 1 unless declared, a truncated correction range unless
/// `truncate` is false, and `cos`/`sin` multiplied by `attention_factor`,
/// else by `mscale(factor, mscale) / mscale(factor, mscale_all_dim)` when
/// both are declared, else by `mscale(factor)`, where `mscale(s, m) =
/// 0.1 m ln(s) + 1` for `s > 1`.
fn yarn_scaling(raw: &RawRopeScaling) -> Result<RopeFrequencyScaling, ProductionIngestionError> {
    let malformed = |what: &str| ProductionIngestionError::MalformedMetadata {
        reason: format!("config.json's yarn rope scaling has no '{what}'"),
    };
    let factor = raw.factor.ok_or_else(|| malformed("factor"))?;
    let original_context_length = raw
        .original_max_position_embeddings
        .ok_or_else(|| malformed("original_max_position_embeddings"))?;
    let mscale = |scale: f64, multiplier: f64| {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * multiplier * scale.ln() + 1.0
        }
    };
    let attention_factor = match (raw.attention_factor, raw.mscale, raw.mscale_all_dim) {
        (Some(factor), _, _) => factor,
        (None, Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
            mscale(factor, m) / mscale(factor, all)
        }
        _ => mscale(factor, 1.0),
    };
    let scaling = RopeYarnFrequencyScaling {
        factor,
        original_context_length,
        beta_fast: raw.beta_fast.unwrap_or(32.0),
        beta_slow: raw.beta_slow.unwrap_or(1.0),
        attention_factor,
        truncate: raw.truncate.unwrap_or(true),
    };
    scaling
        .validate()
        .map_err(|reason| ProductionIngestionError::MalformedMetadata { reason })?;
    Ok(RopeFrequencyScaling::Yarn {
        scaling,
        mscale_all_dim: raw.mscale_all_dim.filter(|value| *value != 0.0),
    })
}

/// Multi-head latent attention, when `config.json` declares a
/// `kv_lora_rank` (`express-latent-attention-forms`): the low ranks and the
/// three head widths, every one required.
fn latent_attention(
    raw: &RawHfConfig,
) -> Result<Option<LatentAttentionConfig>, ProductionIngestionError> {
    let Some(kv_lora_rank) = raw.kv_lora_rank else {
        return Ok(None);
    };
    let field = |value: Option<u64>, name: &str| {
        value.ok_or_else(|| ProductionIngestionError::MalformedMetadata {
            reason: format!("config.json declares a kv_lora_rank but no '{name}'"),
        })
    };
    Ok(Some(LatentAttentionConfig {
        q_lora_rank: raw.q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim: field(raw.qk_nope_head_dim, "qk_nope_head_dim")?,
        qk_rope_head_dim: field(raw.qk_rope_head_dim, "qk_rope_head_dim")?,
        v_head_dim: field(raw.v_head_dim, "v_head_dim")?,
        norm_eps: matches!(raw.model_type.as_deref(), Some("glm4_moe_lite" | "kimi_k2"))
            .then_some(1e-6),
    }))
}

/// The routed experts `n_routed_experts` declares (DeepSeekMoE-style
/// checkpoints): the experts, their always-active shared experts summed into
/// one of `n_shared_experts * moe_intermediate_size`, the layers whose MLP
/// they are (from layer `first_k_dense_replace` on, every `moe_layer_freq`
/// layers; the others dense), and their routing.
#[allow(clippy::type_complexity)]
fn routed_experts(
    raw: &RawHfConfig,
    layers: u32,
) -> Result<
    Option<(
        MixtureOfExpertsConfig,
        Option<Vec<bool>>,
        ExpertRoutingConfig,
    )>,
    ProductionIngestionError,
> {
    let Some(experts) = raw.n_routed_experts else {
        return Ok(None);
    };
    let glm = raw.model_type.as_deref() == Some("glm4_moe_lite");
    let kimi = raw.model_type.as_deref() == Some("kimi_k2");
    if kimi
        && (raw.scoring_func.as_deref() != Some("sigmoid")
            || raw.topk_method.as_deref() != Some("noaux_tc")
            || raw.norm_topk_prob.is_none()
            || raw.routed_scaling_factor.is_none()
            || raw.n_group.is_none()
            || raw.topk_group.is_none()
            || raw.n_group != raw.topk_group)
    {
        // Kimi's original router masks unselected groups to zero rather
        // than negative infinity. The shared graph is equivalent when all
        // groups are selected (as in the pinned checkpoint); other layouts
        // need a separately qualified masking mode.
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "Kimi routing requires explicit sigmoid/noaux settings and all groups selected"
                .into(),
        });
    }
    let malformed = |reason: String| ProductionIngestionError::MalformedMetadata { reason };
    let experts_per_token = raw.num_experts_per_tok.ok_or_else(|| {
        malformed("config.json declares routed experts but no 'num_experts_per_tok'".into())
    })?;
    let expert_intermediate_size = raw.moe_intermediate_size.ok_or_else(|| {
        malformed("config.json declares routed experts but no 'moe_intermediate_size'".into())
    })?;
    let moe = MixtureOfExpertsConfig {
        experts,
        experts_per_token,
        expert_intermediate_size,
        shared_expert_intermediate_size: raw
            .n_shared_experts
            .filter(|count| *count > 0)
            .map(|count| count * expert_intermediate_size),
        normalize_top_k: raw.norm_topk_prob.unwrap_or(glm),
    };
    let first_moe = raw.first_k_dense_replace.unwrap_or(u32::from(glm));
    let frequency = raw.moe_layer_freq.unwrap_or(1).max(1);
    let mut routed: Vec<bool> = (0..layers)
        .map(|layer| layer >= first_moe && layer % frequency == 0)
        .collect();
    if glm && let Some(types) = &raw.mlp_layer_types {
        if types.len() != layers as usize {
            return Err(malformed(
                "GLM mlp_layer_types must have one entry per layer".into(),
            ));
        }
        let explicit = types
            .iter()
            .map(|kind| match kind.as_str() {
                "dense" => Ok(false),
                "sparse" => Ok(true),
                _ => Err(malformed(format!(
                    "unsupported GLM MLP layer type '{kind}'"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if (raw.first_k_dense_replace.is_some() || raw.moe_layer_freq.is_some())
            && explicit != routed
        {
            return Err(malformed("GLM MLP layer declarations disagree".into()));
        }
        routed = explicit;
    }
    let moe_layers = (!routed.iter().all(|routed| *routed)).then_some(routed);
    let scoring = match raw.scoring_func.as_deref() {
        None => {
            if glm {
                ExpertScoring::Sigmoid
            } else {
                ExpertScoring::Softmax
            }
        }
        Some("softmax") if !glm => ExpertScoring::Softmax,
        Some("sigmoid") => ExpertScoring::Sigmoid,
        Some(other) => {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!("config.json's scoring_func '{other}' is not supported"),
            });
        }
    };
    let grouped = |score| -> Result<Option<ExpertGroups>, ProductionIngestionError> {
        let count = raw.n_group.or(glm.then_some(1)).ok_or_else(|| {
            malformed("config.json's group-limited routing has no 'n_group'".into())
        })?;
        let per_token = raw.topk_group.or(glm.then_some(1)).ok_or_else(|| {
            malformed("config.json's group-limited routing has no 'topk_group'".into())
        })?;
        Ok(Some(ExpertGroups {
            count,
            per_token,
            score,
        }))
    };
    let (groups, selection_bias) = match raw.topk_method.as_deref() {
        None => {
            if glm {
                (grouped(ExpertGroupScore::TopTwoSum)?, true)
            } else {
                (None, false)
            }
        }
        Some("greedy") if !glm => (None, false),
        Some("group_limited_greedy") if !glm => (grouped(ExpertGroupScore::Max)?, false),
        Some("noaux_tc") => (grouped(ExpertGroupScore::TopTwoSum)?, true),
        Some(other) => {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!("config.json's topk_method '{other}' is not supported"),
            });
        }
    };
    // DeepSeek-V2's own code renormalizes the chosen weights when
    // `norm_topk_prob` is set and scales them by `routed_scaling_factor`
    // otherwise; transformers' port ignores `norm_topk_prob` and always
    // scales. The two agree only without renormalization, as every published
    // V2 configuration declares, so the other case is refused rather than
    // guessed. `noaux_tc` (V3) renormalizes, then scales, in both.
    if raw.norm_topk_prob == Some(true) && !selection_bias {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "config.json renormalizes the top-{experts_per_token} weights \
                 ('norm_topk_prob') with topk_method '{}', whose reference \
                 implementations disagree on it",
                raw.topk_method.as_deref().unwrap_or("greedy")
            ),
        });
    }
    let routing = ExpertRoutingConfig {
        scoring,
        groups,
        routed_scaling_factor: raw
            .routed_scaling_factor
            .unwrap_or(if glm { 1.8 } else { 1.0 }),
        selection_bias,
        normalization: if glm {
            Some(if moe.normalize_top_k {
                ExpertNormalization::Always
            } else {
                ExpertNormalization::Never
            })
        } else if kimi {
            Some(if moe.normalize_top_k {
                ExpertNormalization::MultipleExperts
            } else {
                ExpertNormalization::Never
            })
        } else {
            None
        },
        shared_expert_gated: false,
    };
    Ok(Some((moe, moe_layers, routing)))
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
    /// The supported block FP8 checkpoint convention, when declared.
    pub fp8: Option<crate::fp8::CheckpointBlockSpecV1>,
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
    let malformed = |reason: String| ProductionIngestionError::MalformedMetadata { reason };
    if raw.use_bidirectional_attention.as_deref() == Some("all") {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "config.json declares bidirectional attention for every token; only causal \
                     decoding is supported"
                .into(),
        });
    }
    if raw.model_type.as_deref() == Some("mistral") && raw.sliding_window == Some(0) {
        return Err(malformed(
            "config.json's sliding_window must be positive".into(),
        ));
    }
    let Some(layer_types) = &raw.layer_types else {
        // Mistral v0.1 declares one window for every decoder layer without
        // a layer_types array. Other families can give sliding_window a
        // different meaning (for example, behind an enable flag), so do not
        // infer their attention semantics from that field alone.
        if raw.model_type.as_deref() != Some("mistral") || raw.sliding_window.is_none() {
            return Ok(None);
        }
        let window = raw.sliding_window.unwrap();
        // Avoid allocating an unbounded layout from untrusted metadata.
        if num_hidden_layers > 4096 {
            return Err(malformed(
                "config.json declares too many decoder layers".into(),
            ));
        }
        let flat_theta = rope_parameters(raw)?
            .get("")
            .and_then(|flat| flat.rope_theta);
        let rope_theta = raw.rope_theta.or(flat_theta).unwrap_or(10_000.0);
        let layer = LayerSpec {
            sliding_window: Some(window),
            head_dim,
            num_key_value_heads: raw.num_key_value_heads.unwrap_or(num_attention_heads),
            rope_theta,
            partial_rotary_factor: None,
            value_from_key: false,
            kv_source_layer: None,
            intermediate_size: raw.intermediate_size.unwrap_or_default(),
        };
        return Ok(Some(LayeredConfig {
            layout: Some(vec![layer; num_hidden_layers as usize]),
            rope_theta,
        }));
    };
    if layer_types.len() != num_hidden_layers as usize {
        return Err(malformed(format!(
            "config.json's layer_types has {} entries for {num_hidden_layers} layers",
            layer_types.len()
        )));
    }
    let intermediate_size = raw.intermediate_size.unwrap_or_default();
    let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
    let k_eq_v = raw.attention_k_eq_v.unwrap_or(false);
    let rope_parameters = rope_parameters(raw)?;
    let flat_theta = rope_parameters.get("").and_then(|flat| flat.rope_theta);
    let default_theta = raw
        .rope_theta
        .or(flat_theta)
        .unwrap_or(match raw.model_type.as_deref() {
            Some("olmo3") => 500_000.0,
            Some("smollm3") => 2_000_000.0,
            _ => 10_000.0,
        });
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
    let fp8 = crate::fp8::checkpoint_spec(value.get("quantization_config").or_else(|| {
        value
            .get("text_config")
            .and_then(|text| text.get("quantization_config"))
    }))?;
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
    let config_value = nested.unwrap_or(value);
    let window_missing = config_value.get("sliding_window").is_none();
    let rope_interval_is_null = config_value
        .get("no_rope_layer_interval")
        .is_some_and(serde_json::Value::is_null);
    let activation_is_null = config_value
        .get("hidden_act")
        .is_some_and(|value| value.is_null());
    let named_activation_is_null = config_value
        .get("hidden_activation")
        .is_some_and(|value| value.is_null());
    let mut raw: RawHfConfig = serde_json::from_value(config_value).map_err(shape_error)?;
    if let Some(model_type) = outer_model_type {
        raw.model_type = Some(model_type);
    }
    if let Some(architectures) = outer_architectures {
        raw.architectures = architectures;
    }
    // These hybrid families have ordered recurrent/attention blocks and
    // family-specific expert semantics not represented by this ingestor yet.
    // In particular, their similarly named MoE fields are not the generic
    // DeepSeek defaults below. Refuse the entire profile until ingestion can
    // carry and validate all of those semantics to its Component.
    if let Some(family @ ("olala" | "nemotron_h")) = raw.model_type.as_deref() {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "{family} hybrid recurrent/attention configuration is not supported by this ingestor"
            ),
        });
    }
    // MistralConfig's upstream default is 4096 when the field is absent;
    // an explicit null, as in v0.2/v0.3, disables sliding attention.
    if raw.model_type.as_deref() == Some("mistral") && window_missing {
        raw.sliding_window = Some(4096);
    }

    let hidden_size = required(raw.hidden_size, "hidden_size")?;
    // Qwen4-Exp's layers are all mixtures of experts: it declares no dense
    // MLP width, and the routed experts' width stands in for it.
    let intermediate_size = if matches!(
        raw.model_type.as_deref(),
        Some("qwen4_exp" | "qwen4_exp_text")
    ) {
        required(
            raw.intermediate_size.or(raw.moe_intermediate_size),
            "moe_intermediate_size",
        )?
    } else {
        required(raw.intermediate_size, "intermediate_size")?
    };
    let num_hidden_layers = required(raw.num_hidden_layers, "num_hidden_layers")?;
    let num_attention_heads = required(raw.num_attention_heads, "num_attention_heads")?;
    let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
    let vocab_size = required(raw.vocab_size, "vocab_size")?;
    let model_type = required(raw.model_type.clone(), "model_type")?;
    if model_type == "glm4_moe_lite" && raw.rope_interleave == Some(false) {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "GLM non-interleaved rotary weights are not supported".into(),
        });
    }

    // OLMo 3 alternates three sliding layers with a full-attention layer
    // when layer_types is omitted. Preserve that upstream default instead
    // of accidentally treating the whole model as full attention.
    if model_type == "olmo3" {
        if window_missing {
            raw.sliding_window = Some(4096);
        }
        if raw.layer_types.is_none() {
            if num_hidden_layers > 4096 {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: "config.json declares too many decoder layers".into(),
                });
            }
            raw.layer_types = Some(
                (0..num_hidden_layers)
                    .map(|index| {
                        if (index + 1) % 4 == 0 {
                            "full_attention"
                        } else {
                            "sliding_attention"
                        }
                        .to_string()
                    })
                    .collect(),
            );
        }
    }

    // The currently supported components for these families implement SiLU.
    // Reject alternatives until a Component implements and qualifies them.
    // An omitted field follows these model classes' SiLU default; explicit
    // null is malformed and must not be mistaken for an omitted field.
    if activation_is_null || named_activation_is_null {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json's activation must be a supported activation string".into(),
        });
    }
    if let (Some(short), Some(named)) = (&raw.hidden_act, &raw.hidden_activation)
        && short != named
    {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json declares conflicting activation fields".into(),
        });
    }
    let gemma = matches!(
        model_type.as_str(),
        "gemma4" | "gemma4_text" | "gemma4_unified"
    );
    let fixed_silu = matches!(
        model_type.as_str(),
        "llama"
            | "qwen2"
            | "qwen2moe"
            | "qwen2_moe"
            | "mistral"
            | "olmo3"
            | "smollm3"
            | "deepseek"
            | "deepseek_v2"
            | "deepseek_v3"
            | "deepseek2"
            | "glm4_moe_lite"
            | "kimi_k2"
            | "qwen3_5"
            | "qwen3_5_text"
            | "qwen4_exp"
            | "qwen4_exp_text"
    );
    let hidden_activation = raw
        .hidden_activation
        .clone()
        .or_else(|| raw.hidden_act.clone())
        .or_else(|| {
            if gemma {
                Some("gelu_pytorch_tanh".into())
            } else if fixed_silu {
                Some("silu".into())
            } else {
                None
            }
        });
    if fixed_silu
        && let Some(activation) = hidden_activation
            .as_deref()
            .filter(|value| *value != "silu")
    {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "config.json's hidden_act '{activation}' is not supported for {model_type}; only silu is available"
            ),
        });
    }
    if gemma && hidden_activation.as_deref() != Some("gelu_pytorch_tanh") {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "Gemma 4 requires the gelu_pytorch_tanh activation in this Component".into(),
        });
    }

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
    let rope_layers = if model_type == "smollm3" {
        if num_hidden_layers > 4096 {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: "config.json declares too many decoder layers".into(),
            });
        }
        if raw.use_sliding_window == Some(true) || raw.mlp_bias == Some(true) {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: "SmolLM3 sliding attention and MLP bias are not supported".into(),
            });
        }
        if let Some(types) = &raw.layer_types
            && (types.len() != num_hidden_layers as usize
                || types.iter().any(|kind| kind != "full_attention"))
        {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: "SmolLM3 requires one full_attention layer type per decoder layer".into(),
            });
        }
        let layers = if let Some(mask) = &raw.no_rope_layers {
            if mask.len() < num_hidden_layers as usize {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "config.json's no_rope_layers has {} entries for {num_hidden_layers} layers",
                        mask.len()
                    ),
                });
            }
            mask.iter()
                .take(num_hidden_layers as usize)
                .enumerate()
                .map(|(layer, &entry)| match entry {
                    0 => Ok(false),
                    1 => Ok(true),
                    _ => Err(ProductionIngestionError::MalformedMetadata {
                        reason: format!("config.json's no_rope_layers[{layer}] must be 0 or 1"),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            if rope_interval_is_null {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: "config.json's no_rope_layer_interval must be a positive integer"
                        .into(),
                });
            }
            let interval = raw.no_rope_layer_interval.unwrap_or(4);
            if interval == 0 {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: "config.json's no_rope_layer_interval must be positive".into(),
                });
            }
            (0..num_hidden_layers)
                .map(|layer| (layer + 1) % interval != 0)
                .collect()
        };
        Some(layers)
    } else {
        None
    };
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "config.json's num_attention_heads ({num_attention_heads}) is not an exact \
                 multiple of num_key_value_heads ({num_key_value_heads})"
            ),
        });
    }

    // Latent attention's query/key heads are its non-rotated and rotated
    // parts together, one K/V head per attention head
    // (`transformers` sets `head_dim` to the rotated width for its RoPE).
    let latent = latent_attention(&raw)?;
    let num_key_value_heads = if latent.is_some() {
        num_attention_heads
    } else {
        num_key_value_heads
    };
    let head_dim = match (latent, raw.head_dim) {
        (Some(mla), _) => mla.qk_nope_head_dim + mla.qk_rope_head_dim,
        (None, Some(declared)) => declared,
        (None, None) => {
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

    let rms_norm_eps = raw.rms_norm_eps.unwrap_or(
        if matches!(model_type.as_str(), "olmo3" | "glm4_moe_lite") {
            1e-5
        } else {
            1e-6
        },
    ) as f32;
    let qwen35 = qwen35_config(
        &mut raw,
        &model_type,
        num_hidden_layers,
        num_key_value_heads,
        head_dim,
        intermediate_size,
    )?;
    let layered = layered_config(&raw, num_hidden_layers, num_attention_heads, head_dim)?;
    // `rope_theta`, else a flat `rope_parameters`' (newer `transformers`
    // releases moved it there), else a layered decoder's sliding layers'.
    let flat_rope_theta = rope_parameters(&raw)?
        .get("")
        .and_then(|flat| flat.rope_theta);
    let rope_theta = raw
        .rope_theta
        .or(qwen35.as_ref().map(|qwen35| qwen35.rope_theta))
        .or(flat_rope_theta)
        .or(layered.as_ref().map(|layered| layered.rope_theta))
        .unwrap_or(if model_type == "smollm3" {
            2_000_000.0
        } else {
            10_000.0
        });
    if rope_theta <= 0.0 {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "config.json's rope_theta must be positive".into(),
        });
    }

    // `rope_scaling`, else a flat `rope_parameters` naming a scheme (newer
    // `transformers` releases moved it there).
    let flat_rope_scaling: Option<RawRopeScaling> = match (&raw.rope_scaling, &raw.rope_parameters)
    {
        (None, Some(value)) if value.get("rope_type").is_some() => {
            Some(serde_json::from_value(value.clone()).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("config.json's rope_parameters is malformed: {error}"),
                }
            })?)
        }
        _ => None,
    };
    let (rope_scaling_factor, rope_frequency_scaling) =
        rope_scaling(raw.rope_scaling.as_ref().or(flat_rope_scaling.as_ref()))?;
    let routed = routed_experts(&raw, num_hidden_layers)?;
    let (mixture_of_experts, moe_layers, expert_routing) = match routed {
        Some((moe, layers, routing)) => (Some(moe), layers, Some(routing)),
        None => match qwen35.as_ref().and_then(|qwen35| qwen35.experts) {
            Some(moe) => (Some(moe), None, None),
            None => (mixture_of_experts(&raw)?, None, None),
        },
    };
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
        rope_layers,
        hidden_activation,
        tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(model_type == "smollm3"),
        // `config.json` never declares this for real Qwen2/2.5 checkpoints
        // (it's an architectural default of the model class, not a
        // config field) -- `false` here is a placeholder `lib.rs::ingest`
        // overwrites once real weight discovery has actually looked for
        // `self_attn.{q,k,v}_proj` bias tensors, the only real evidence
        // this ever needs.
        attention_bias: false,
        mixture_of_experts,
        bos_token_id: raw.bos_token_id,
        eos_token_id: raw.eos_token_id.as_ref().and_then(RawEosTokenId::first),
        layer_layout: match &qwen35 {
            Some(qwen35) => Some(qwen35.layout.clone()),
            None => layered.and_then(|layered| layered.layout),
        },
        final_logit_softcapping: raw.final_logit_softcapping,
        per_layer_input_size: raw.hidden_size_per_layer_input.filter(|size| *size > 0),
        latent_attention: latent,
        moe_layers,
        expert_routing,
        attention_output_gate: qwen35
            .as_ref()
            .is_some_and(|qwen35| qwen35.attention_output_gate),
        // Qwen3.5's RMSNorms scale by `1 + weight`.
        norm_weight_offset: qwen35.as_ref().map(|_| 1.0),
        decoder_extensions: qwen35
            .as_ref()
            .map(|qwen35| qwen35.extensions.clone())
            .unwrap_or_default(),
        linear_attention: qwen35.map(|qwen35| qwen35.linear_attention),
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
        fp8,
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
        assert_eq!(config.hidden_activation.as_deref(), Some("silu"));
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
    fn refuses_unmodeled_hybrid_families_before_applying_decoder_defaults() {
        for family in ["olala", "nemotron_h"] {
            let mut value: serde_json::Value =
                serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
            value["model_type"] = family.into();
            value["hybrid_override_pattern"] = "M*".into();
            value["layers_config"] = "MV".into();
            assert!(
                matches!(
                    parse(&serde_json::to_vec(&value).unwrap()),
                    Err(ProductionIngestionError::UnsupportedFormat { .. })
                ),
                "{family} must not be ingested as a generic decoder"
            );
        }
    }

    #[test]
    fn mistral_uniform_window_is_carried_to_every_layer() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["model_type"] = "mistral".into();
        value["num_hidden_layers"] = 3.into();
        value["sliding_window"] = 4096.into();
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        let layout = config.layer_layout.unwrap();
        assert_eq!(layout.len(), 3);
        for layer in layout {
            assert_eq!(layer.sliding_window, Some(4096));
            assert_eq!(layer.head_dim, config.head_dim);
            assert_eq!(layer.num_key_value_heads, config.num_key_value_heads);
            assert_eq!(layer.intermediate_size, config.intermediate_size);
            assert_eq!(layer.rope_theta, config.rope_theta);
        }
    }

    #[test]
    fn mistral_missing_null_and_invalid_windows_are_distinguished() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["model_type"] = "mistral".into();
        let default = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        assert!(
            default
                .layer_layout
                .unwrap()
                .iter()
                .all(|layer| layer.sliding_window == Some(4096))
        );
        value["sliding_window"] = serde_json::Value::Null;
        assert!(
            parse(&serde_json::to_vec(&value).unwrap())
                .unwrap()
                .architecture_config
                .layer_layout
                .is_none()
        );
        for invalid in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("4096"),
        ] {
            value["sliding_window"] = invalid;
            assert!(matches!(
                parse(&serde_json::to_vec(&value).unwrap()),
                Err(ProductionIngestionError::MalformedMetadata { .. })
            ));
        }
    }

    #[test]
    fn explicit_mixed_layer_types_keep_their_windows() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["model_type"] = "mistral".into();
        value["num_hidden_layers"] = 2.into();
        value["layer_types"] = serde_json::json!(["full_attention", "sliding_attention"]);
        value["sliding_window"] = 64.into();
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        let layout = config.layer_layout.unwrap();
        assert_eq!(layout[0].sliding_window, None);
        assert_eq!(layout[1].sliding_window, Some(64));
    }

    #[test]
    fn olmo3_missing_layer_types_uses_upstream_attention_defaults() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["model_type"] = "olmo3".into();
        value["num_hidden_layers"] = 4.into();
        let object = value.as_object_mut().unwrap();
        object.remove("sliding_window");
        object.remove("rope_theta");
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        let layout = config.layer_layout.unwrap();
        assert_eq!(
            layout
                .iter()
                .map(|layer| layer.sliding_window)
                .collect::<Vec<_>>(),
            [Some(4096), Some(4096), Some(4096), None]
        );
        assert!(layout.iter().all(|layer| layer.rope_theta == 500_000.0));

        value["sliding_window"] = serde_json::Value::Null;
        assert!(matches!(
            parse(&serde_json::to_vec(&value).unwrap()),
            Err(ProductionIngestionError::MalformedMetadata { .. })
        ));
    }

    #[test]
    fn another_familys_window_is_not_assumed_to_be_enabled() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["sliding_window"] = 4096.into();
        value["use_sliding_window"] = false.into();
        assert!(
            parse(&serde_json::to_vec(&value).unwrap())
                .unwrap()
                .architecture_config
                .layer_layout
                .is_none()
        );
    }

    #[test]
    fn fixed_silu_components_reject_different_or_malformed_activations() {
        for family in [
            "llama",
            "qwen2",
            "qwen2_moe",
            "mistral",
            "olmo3",
            "smollm3",
            "deepseek_v2",
        ] {
            let mut value: serde_json::Value =
                serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
            value["model_type"] = family.into();
            // Absence means the upstream model class's SiLU default.
            assert!(
                parse(&serde_json::to_vec(&value).unwrap()).is_ok(),
                "{family}"
            );
            value["hidden_act"] = "silu".into();
            assert!(
                parse(&serde_json::to_vec(&value).unwrap()).is_ok(),
                "{family}"
            );
            for unsupported in ["gelu", "unknown", ""] {
                value["hidden_act"] = unsupported.into();
                assert!(
                    matches!(
                        parse(&serde_json::to_vec(&value).unwrap()),
                        Err(ProductionIngestionError::UnsupportedFormat { .. })
                    ),
                    "{family} {unsupported:?}"
                );
            }
            for malformed in [
                serde_json::Value::Null,
                serde_json::json!(42),
                serde_json::json!(true),
            ] {
                value["hidden_act"] = malformed;
                assert!(
                    matches!(
                        parse(&serde_json::to_vec(&value).unwrap()),
                        Err(ProductionIngestionError::MalformedMetadata { .. })
                    ),
                    "{family}"
                );
            }
        }
    }

    #[test]
    fn gemma4_named_activation_is_preserved_and_validated() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["model_type"] = "gemma4".into();
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        assert_eq!(
            config.hidden_activation.as_deref(),
            Some("gelu_pytorch_tanh")
        );
        value["hidden_activation"] = "gelu_pytorch_tanh".into();
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        assert_eq!(
            config.hidden_activation.as_deref(),
            Some("gelu_pytorch_tanh")
        );
        for invalid in [
            serde_json::json!("silu"),
            serde_json::json!("gelu"),
            serde_json::Value::Null,
            serde_json::json!(42),
        ] {
            value["hidden_activation"] = invalid;
            assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        value["hidden_activation"] = "gelu_pytorch_tanh".into();
        value["hidden_act"] = "silu".into();
        assert!(matches!(
            parse(&serde_json::to_vec(&value).unwrap()),
            Err(ProductionIngestionError::MalformedMetadata { .. })
        ));
    }

    #[test]
    fn smollm3_preserves_explicit_and_default_rope_plans() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["model_type"] = "smollm3".into();
        value["no_rope_layers"] = serde_json::json!([1, 1, 1, 0]);
        value["num_hidden_layers"] = 4.into();
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        assert_eq!(config.rope_layers, Some(vec![true, true, true, false]));

        value.as_object_mut().unwrap().remove("no_rope_layers");
        value["num_hidden_layers"] = 5.into();
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        assert_eq!(
            config.rope_layers,
            Some(vec![true, true, true, false, true])
        );
        value["no_rope_layer_interval"] = 2.into();
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        assert_eq!(
            config.rope_layers,
            Some(vec![true, false, true, false, true])
        );
    }

    #[test]
    fn smollm3_rejects_invalid_position_plans_and_unsupported_features() {
        let mut value: serde_json::Value = serde_json::from_slice(&qwen2_0_5b_config()).unwrap();
        value["model_type"] = "smollm3".into();
        value["num_hidden_layers"] = 2.into();
        for mask in [
            serde_json::json!([1]),
            serde_json::json!([1, 2]),
            serde_json::json!([1, true]),
        ] {
            value["no_rope_layers"] = mask;
            assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        value.as_object_mut().unwrap().remove("no_rope_layers");
        value["no_rope_layer_interval"] = 0.into();
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
        value["no_rope_layer_interval"] = serde_json::Value::Null;
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
        value["no_rope_layer_interval"] = 4.into();
        for (name, unsupported) in [
            ("use_sliding_window", serde_json::json!(true)),
            ("mlp_bias", serde_json::json!(true)),
            (
                "layer_types",
                serde_json::json!(["full_attention", "sliding_attention"]),
            ),
        ] {
            value[name] = unsupported;
            assert!(matches!(
                parse(&serde_json::to_vec(&value).unwrap()),
                Err(ProductionIngestionError::UnsupportedFormat { .. })
            ));
            value.as_object_mut().unwrap().remove(name);
        }
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
        // YaRN without its original context length is incomplete
        // (`express-latent-attention-forms` reads YaRN).
        let mut yarn = llama3_2_1b_config();
        yarn["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0});
        assert!(matches!(
            parse(&serde_json::to_vec(&yarn).unwrap()).unwrap_err(),
            ProductionIngestionError::MalformedMetadata { .. }
        ));
        let mut unknown = llama3_2_1b_config();
        unknown["rope_scaling"] = serde_json::json!({"rope_type": "longrope", "factor": 4.0});
        assert!(matches!(
            parse(&serde_json::to_vec(&unknown).unwrap()).unwrap_err(),
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

    /// Qwen3.5-0.8B-Base's `config.json` at revision
    /// dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 (Apache-2.0): a multimodal
    /// bundle whose `text_config` is the hybrid decoder.
    #[test]
    fn qwen35_declares_linear_layers_gated_attention_and_zero_centered_norms() {
        let path = format!(
            "{}/testdata/qwen35/Qwen3.5-0.8B-Base.config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let parsed = parse(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(parsed.metadata.model_type, "qwen3_5");
        let config = parsed.architecture_config;
        assert_eq!(
            (
                config.num_hidden_layers,
                config.num_attention_heads,
                config.num_key_value_heads
            ),
            (24, 8, 2)
        );
        assert_eq!((config.head_dim, config.hidden_size), (256, 1024));
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert!(config.tie_word_embeddings);
        assert!(config.attention_output_gate);
        assert_eq!(config.norm_weight_offset, Some(1.0));
        assert_eq!(config.hidden_activation.as_deref(), Some("silu"));
        let linear = config.linear_attention.unwrap();
        assert_eq!(
            (
                linear.key_heads,
                linear.value_heads,
                linear.key_head_dim,
                linear.value_head_dim
            ),
            (16, 16, 128, 128)
        );
        assert_eq!(linear.conv_kernel, 4);
        assert_eq!(
            linear.layers,
            (0..24)
                .map(|layer| (layer + 1) % 4 != 0)
                .collect::<Vec<_>>()
        );
        let layout = config.layer_layout.unwrap();
        assert_eq!(layout.len(), 24);
        assert!(
            layout
                .iter()
                .all(|layer| layer.partial_rotary_factor == Some(0.25)
                    && layer.head_dim == 256
                    && layer.num_key_value_heads == 2
                    && layer.sliding_window.is_none())
        );
    }

    /// A Qwen4-Exp text configuration (`qwen4_exp_text`): residual streams,
    /// one n-gram embedding layer, the indexer, a sigmoid linear gate and a
    /// mixture of experts with a gated shared expert.
    fn qwen4_exp_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen4ExpForCausalLM"],
            "model_type": "qwen4_exp_text",
            "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "indexed_attention",
                            "linear_attention", "indexed_attention"],
            "moe_intermediate_size": 12, "shared_expert_intermediate_size": 12,
            "num_experts": 4, "num_experts_per_tok": 2,
            "hc_count": 4, "hc_lowrank": 6,
            "ple_layer_ids": [1], "ple_embed_dim": 32, "ple_conv_kernel_size": 4,
            "ngram_size": 3, "heads_per_ngram": 8, "ngram_vocab_size_base": 1000,
            "make_ngram_vocab_size_divisible_by": 128, "seed": 1234,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
            "indexer_budget": 32, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid",
            "rope_parameters": {"rope_type": "default", "rope_theta": 100.0,
                                "partial_rotary_factor": 0.25},
            "rms_norm_eps": 1e-6, "bos_token_id": 1, "eos_token_id": 2,
            "tie_word_embeddings": false
        })
    }

    #[test]
    fn qwen4_exp_declares_streams_embeddings_indexer_gate_and_experts() {
        let parsed = parse(&serde_json::to_vec(&qwen4_exp_config()).unwrap()).unwrap();
        assert_eq!(parsed.metadata.model_type, "qwen4_exp_text");
        let config = parsed.architecture_config;
        assert!(config.attention_output_gate);
        assert_eq!(config.norm_weight_offset, Some(1.0));
        assert_eq!(config.hidden_activation.as_deref(), Some("silu"));
        assert_eq!(config.rope_theta, 100.0);
        let linear = config.linear_attention.as_ref().unwrap();
        assert_eq!(linear.layers, [true, false, true, false]);
        let layout = config.layer_layout.as_ref().unwrap();
        assert!(
            layout
                .iter()
                .all(|layer| layer.partial_rotary_factor == Some(0.25) && layer.head_dim == 16)
        );
        let extensions = &config.decoder_extensions;
        assert_eq!(
            extensions.hyper_connections,
            Some(HyperConnectionConfig {
                streams: 4,
                lowrank: 6
            })
        );
        let ngram = extensions.ngram_embeddings.as_ref().unwrap();
        assert_eq!(ngram.layers, [true, false, false, false]);
        assert_eq!(
            (
                ngram.embedding_dim,
                ngram.conv_kernel,
                ngram.ngram_size,
                ngram.heads_per_ngram
            ),
            (32, 4, 3, 8)
        );
        assert_eq!(
            (ngram.vocabulary_base, ngram.seed, ngram.eos_token),
            (1000, 1234, 2)
        );
        assert_eq!(
            extensions.token_selection,
            Some(TokenSelectionConfig {
                heads: 2,
                key_heads: 1,
                head_dim: 8,
                budget: 32,
                compress_ratio: 4
            })
        );
        assert!(extensions.sigmoid_linear_gate);
        let moe = config.mixture_of_experts.unwrap();
        assert_eq!((moe.experts, moe.experts_per_token), (4, 2));
        assert_eq!(moe.expert_intermediate_size, 12);
        assert_eq!(moe.shared_expert_intermediate_size, Some(12));
        assert!(moe.normalize_top_k);
        // The `1.5.0` routing: a softmax over every expert and a gated
        // shared expert, which a Component reads without `expert-routing`.
        assert!(config.expert_routing.is_none());
    }

    #[test]
    fn qwen4_exp_without_optional_forms_keeps_them_absent() {
        let mut value = qwen4_exp_config();
        for key in [
            "ple_layer_ids",
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
            "output_gate_type",
        ] {
            value.as_object_mut().unwrap().remove(key);
        }
        let config = parse(&serde_json::to_vec(&value).unwrap())
            .unwrap()
            .architecture_config;
        let extensions = config.decoder_extensions;
        assert!(extensions.hyper_connections.is_some());
        assert!(extensions.ngram_embeddings.is_none());
        assert!(extensions.token_selection.is_none());
        assert!(!extensions.sigmoid_linear_gate);
    }

    #[test]
    fn qwen4_exp_refuses_inconsistent_forms() {
        let refused = |change: fn(&mut serde_json::Value)| {
            let mut value = qwen4_exp_config();
            change(&mut value);
            assert!(
                parse(&serde_json::to_vec(&value).unwrap()).is_err(),
                "{value}"
            );
        };
        // A partial indexer, an unknown gate activation, a PLE layer past
        // the last, and no experts.
        refused(|value| {
            value.as_object_mut().unwrap().remove("indexer_budget");
        });
        refused(|value| value["output_gate_type"] = "gelu".into());
        refused(|value| value["ple_layer_ids"] = serde_json::json!([5]));
        refused(|value| {
            value.as_object_mut().unwrap().remove("num_experts");
        });
        // A budget that is not whole blocks.
        refused(|value| value["indexer_budget"] = 30.into());
        // A rotary width wider than the indexer's head.
        refused(|value| value["rope_parameters"]["partial_rotary_factor"] = 1.0.into());
    }

    #[test]
    fn qwen35_refuses_an_unknown_layer_type() {
        let path = format!(
            "{}/testdata/qwen35/Qwen3.5-0.8B-Base.config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        value["text_config"]["layer_types"][0] = "sliding_attention".into();
        assert!(matches!(
            parse(&serde_json::to_vec(&value).unwrap()),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
    }

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

/// `express-latent-attention-forms`: DeepSeek's published `config.json`s and
/// a `transformers` 5.23 export, read into latent attention, mixed expert
/// layers, routing and YaRN.
#[cfg(test)]
mod latent_attention_tests {
    use super::*;

    fn parse_deepseek(name: &str) -> NormalizedHfConfig {
        let path = format!(
            "{}/testdata/deepseek/{name}.config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        parse(&std::fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn glm_flash_defaults_to_grouped_sigmoid_and_top_one_normalization() {
        // Published GLM-4.7-Flash config: scoring_func is intentionally absent.
        let path = format!(
            "{}/testdata/glm/glm-4.7-flash.config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let parse_raw = |raw: &serde_json::Value| {
            parse(&serde_json::to_vec(raw).unwrap())
                .unwrap()
                .architecture_config
        };
        let config = parse_raw(&raw);
        assert_eq!(config.latent_attention.unwrap().norm_eps, Some(1e-6));
        let routing = config.expert_routing.unwrap();
        assert_eq!(routing.scoring, ExpertScoring::Sigmoid);
        assert_eq!(
            routing.groups,
            Some(ExpertGroups {
                count: 1,
                per_token: 1,
                score: ExpertGroupScore::TopTwoSum
            })
        );
        assert!(routing.selection_bias);
        assert_eq!(routing.normalization, Some(ExpertNormalization::Always));
        assert_eq!(routing.routed_scaling_factor, 1.8);
        let layers = config.moe_layers.unwrap();
        assert_eq!(layers.len(), 47);
        assert!(!layers[0] && layers[1..].iter().all(|value| *value));

        // These are class defaults even when omitted from a serialized config.
        let object = raw.as_object_mut().unwrap();
        for key in [
            "norm_topk_prob",
            "topk_method",
            "n_group",
            "topk_group",
            "first_k_dense_replace",
            "routed_scaling_factor",
        ] {
            object.remove(key);
        }
        let defaulted = parse_raw(&raw);
        assert_eq!(defaulted.expert_routing.unwrap(), routing);
        assert_eq!(defaulted.moe_layers.unwrap(), layers);

        raw["norm_topk_prob"] = serde_json::json!(false);
        assert_eq!(
            parse_raw(&raw).expert_routing.unwrap().normalization,
            Some(ExpertNormalization::Never)
        );
        raw["scoring_func"] = serde_json::json!("softmax");
        assert!(parse(&serde_json::to_vec(&raw).unwrap()).is_err());
        raw.as_object_mut().unwrap().remove("scoring_func");
        raw["rope_interleave"] = serde_json::json!(false);
        assert!(parse(&serde_json::to_vec(&raw).unwrap()).is_err());
        raw.as_object_mut().unwrap().remove("rope_interleave");
        raw["mlp_layer_types"] = serde_json::json!(["dense"]);
        assert!(parse(&serde_json::to_vec(&raw).unwrap()).is_err());
    }

    #[test]
    fn pinned_kimi_config_keeps_its_fp8_and_distinct_top_one_rules() {
        let path = format!(
            "{}/testdata/kimi-k2/config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(path).unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed.metadata.model_type, "kimi_k2");
        assert_eq!(
            parsed.fp8,
            Some(crate::fp8::CheckpointBlockSpecV1 {
                block_rows: 128,
                block_columns: 128
            })
        );
        let config = parsed.architecture_config;
        assert_eq!(config.latent_attention.unwrap().norm_eps, Some(1e-6));
        assert_eq!(
            config.expert_routing.unwrap().normalization,
            Some(ExpertNormalization::MultipleExperts)
        );
        assert_eq!(config.expert_routing.unwrap().groups.unwrap().count, 1);
        assert_eq!(config.mixture_of_experts.unwrap().experts, 384);
        let mut changed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        changed["n_group"] = serde_json::json!(2);
        assert!(parse(&serde_json::to_vec(&changed).unwrap()).is_err());
        changed["n_group"] = serde_json::json!(1);
        changed.as_object_mut().unwrap().remove("scoring_func");
        assert!(parse(&serde_json::to_vec(&changed).unwrap()).is_err());
    }

    fn yarn(config: &ModelArchitectureConfig) -> (RopeYarnFrequencyScaling, Option<f64>) {
        match config.rope_frequency_scaling {
            Some(RopeFrequencyScaling::Yarn {
                scaling,
                mscale_all_dim,
            }) => (scaling, mscale_all_dim),
            other => panic!("expected YaRN, got {other:?}"),
        }
    }

    #[test]
    fn deepseek_v2_lite_has_latent_attention_a_dense_first_layer_and_greedy_routing() {
        let parsed = parse_deepseek("DeepSeek-V2-Lite");
        assert_eq!(parsed.metadata.model_type, "deepseek_v2");
        let config = parsed.architecture_config;
        assert_eq!(
            config.latent_attention,
            Some(LatentAttentionConfig {
                q_lora_rank: None,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                norm_eps: None,
            })
        );
        assert_eq!(config.head_dim, 192);
        assert_eq!(config.num_key_value_heads, config.num_attention_heads);
        assert_eq!(config.intermediate_size, 10944);
        assert_eq!(
            config.mixture_of_experts,
            Some(MixtureOfExpertsConfig {
                experts: 64,
                experts_per_token: 6,
                expert_intermediate_size: 1408,
                shared_expert_intermediate_size: Some(2 * 1408),
                normalize_top_k: false,
            })
        );
        let layers = config.moe_layers.as_ref().unwrap();
        assert_eq!(layers.len(), 27);
        assert!(!layers[0] && layers[1..].iter().all(|routed| *routed));
        assert_eq!(
            config.expert_routing,
            Some(ExpertRoutingConfig {
                scoring: ExpertScoring::Softmax,
                groups: None,
                routed_scaling_factor: 1.0,
                selection_bias: false,
                normalization: None,
                shared_expert_gated: false,
            })
        );
        let (scaling, mscale_all_dim) = yarn(&config);
        assert_eq!(scaling.factor, 40.0);
        assert_eq!(scaling.original_context_length, 4096);
        assert_eq!((scaling.beta_fast, scaling.beta_slow), (32.0, 1.0));
        assert!(scaling.truncate);
        // mscale and mscale_all_dim are equal: cos and sin are unscaled.
        assert_eq!(scaling.attention_factor, 1.0);
        assert_eq!(mscale_all_dim, Some(0.707));
        config.validate().unwrap();
    }

    #[test]
    fn deepseek_v2_routes_within_groups_and_scales_the_weights() {
        let config = parse_deepseek("DeepSeek-V2").architecture_config;
        assert_eq!(config.latent_attention.unwrap().q_lora_rank, Some(1536));
        assert_eq!(
            config.expert_routing,
            Some(ExpertRoutingConfig {
                scoring: ExpertScoring::Softmax,
                groups: Some(ExpertGroups {
                    count: 8,
                    per_token: 3,
                    score: ExpertGroupScore::Max,
                }),
                routed_scaling_factor: 16.0,
                selection_bias: false,
                normalization: None,
                shared_expert_gated: false,
            })
        );
        config.validate().unwrap();
    }

    /// DeepSeek-V2's own code and transformers disagree on renormalized
    /// greedy or group-limited routing, so it is refused; V3's `noaux_tc`
    /// renormalizes in both.
    #[test]
    fn renormalized_v2_routing_is_refused() {
        let path = format!(
            "{}/testdata/deepseek/DeepSeek-V2-Lite.config.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        raw["norm_topk_prob"] = serde_json::json!(true);
        assert!(matches!(
            parse(&serde_json::to_vec(&raw).unwrap()),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
        raw["topk_method"] = serde_json::json!("group_limited_greedy");
        raw["n_group"] = serde_json::json!(8);
        raw["topk_group"] = serde_json::json!(3);
        assert!(matches!(
            parse(&serde_json::to_vec(&raw).unwrap()),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
        assert!(
            parse_deepseek("DeepSeek-V3")
                .architecture_config
                .mixture_of_experts
                .unwrap()
                .normalize_top_k
        );
    }

    #[test]
    fn deepseek_v3_scores_with_a_sigmoid_and_a_selection_bias() {
        let parsed = parse_deepseek("DeepSeek-V3");
        assert_eq!(parsed.metadata.model_type, "deepseek_v3");
        assert_eq!(
            parsed.fp8,
            Some(crate::fp8::CheckpointBlockSpecV1 {
                block_rows: 128,
                block_columns: 128,
            })
        );
        let config = parsed.architecture_config;
        let layers = config.moe_layers.as_ref().unwrap();
        assert!(
            layers[..3].iter().all(|routed| !routed) && layers[3..].iter().all(|routed| *routed)
        );
        let moe = config.mixture_of_experts.unwrap();
        assert!(moe.normalize_top_k);
        assert_eq!(
            moe.shared_expert_intermediate_size,
            Some(moe.expert_intermediate_size)
        );
        assert_eq!(
            config.expert_routing,
            Some(ExpertRoutingConfig {
                scoring: ExpertScoring::Sigmoid,
                groups: Some(ExpertGroups {
                    count: 8,
                    per_token: 4,
                    score: ExpertGroupScore::TopTwoSum,
                }),
                routed_scaling_factor: 2.5,
                selection_bias: true,
                normalization: None,
                shared_expert_gated: false,
            })
        );
        config.validate().unwrap();
    }

    /// `transformers` 5 writes RoPE (YaRN included) under a flat
    /// `rope_parameters` and `head_dim` as the rotated width; both are read.
    #[test]
    fn a_transformers_5_export_reads_yarn_from_rope_parameters() {
        let config = parse_deepseek("tiny-deepseek-v2").architecture_config;
        let mla = config.latent_attention.unwrap();
        assert_eq!(config.head_dim, mla.qk_nope_head_dim + mla.qk_rope_head_dim);
        assert_eq!(config.rope_theta, 100.0);
        let (scaling, mscale_all_dim) = yarn(&config);
        assert_eq!(scaling.factor, 4.0);
        assert_eq!((scaling.beta_fast, scaling.beta_slow), (2.0, 1.0));
        assert_eq!(scaling.attention_factor, 1.0);
        assert_eq!(mscale_all_dim, Some(0.8));
        config.validate().unwrap();
    }

    /// A YaRN `attention_factor` is taken as declared, else derived from the
    /// factor alone when the two mscales are not both declared.
    #[test]
    fn yarn_attention_factor_follows_transformers() {
        let parsed = |rope: &str| {
            let json = format!(
                r#"{{"model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
                    "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 32,
                    "rope_scaling": {rope}}}"#
            );
            yarn(&parse(json.as_bytes()).unwrap().architecture_config).0
        };
        let factor_only =
            parsed(r#"{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}"#);
        assert!((factor_only.attention_factor - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12);
        let declared = parsed(
            r#"{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                "attention_factor": 1.5, "truncate": false}"#,
        );
        assert_eq!(declared.attention_factor, 1.5);
        assert!(!declared.truncate);
        let missing = parse(
            br#"{"model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 32,
                "rope_scaling": {"type": "yarn", "factor": 4.0}}"#,
        );
        assert!(missing.is_err());
    }
}
