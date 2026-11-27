//! Strict metadata mapping for one pinned NVIDIA Nemotron-H configuration.
//!
//! This does not ingest a model or enable the generic ingestor's hybrid
//! path. Its weight mapping, checkpoint loading and reference profile still
//! need separate qualification before that guard can be removed.

use magnetar_runtime::model::{
    DecoderExtensions, ExpertGroupScore, ExpertGroups, ExpertNormalization, ExpertRoutingConfig,
    ExpertScoring, HybridDecoderBlock, Mamba2HybridConfig, MixtureOfExpertsConfig,
    ModelArchitectureConfig,
};
use magnetar_runtime::production_model_ingestion::ProductionIngestionError;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// SHA-256 of config.json at NVIDIA's immutable fe4143e revision.
pub const PINNED_NANO_CONFIG_SHA256: &str =
    "dd9fa380ac107b0477db5a26108db9febe6378e7bb3966a107944853ec4f76f8";

/// The graph-facing configuration and context limit of this exact artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct NanoProfile {
    pub architecture: ModelArchitectureConfig,
    pub max_positions: u64,
}

#[derive(Deserialize)]
struct SourceConfig {
    model_type: String,
    hybrid_override_pattern: String,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u64,
    vocab_size: u64,
    max_position_embeddings: u64,
    layer_norm_epsilon: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
    attention_bias: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    mamba_num_heads: u32,
    mamba_head_dim: u64,
    n_groups: u32,
    ssm_state_size: u64,
    conv_kernel: u32,
    use_conv_bias: bool,
    use_bias: bool,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u64,
    moe_shared_expert_intermediate_size: u64,
    norm_topk_prob: bool,
    n_group: u32,
    topk_group: u32,
    routed_scaling_factor: f64,
}

/// Map only the byte-exact published configuration. The fixed source digest
/// also prevents unknown or unmodeled configuration flags from being silently
/// interpreted as this profile. Initializer time-step bounds are not scan
/// limits: the pinned NVIDIA mixer's default inference limits are 0..infinity.
pub fn parse_pinned_nano_config(bytes: &[u8]) -> Result<NanoProfile, ProductionIngestionError> {
    let actual: String = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if actual != PINNED_NANO_CONFIG_SHA256 {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "Nemotron-H configuration is not the pinned Nano profile".into(),
        });
    }
    let raw: SourceConfig = serde_json::from_slice(bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("invalid pinned Nemotron-H configuration: {error}"),
        }
    })?;
    if raw.model_type != "nemotron_h" {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "pinned configuration has an unexpected model type".into(),
        });
    }
    let layers: Vec<_> = raw
        .hybrid_override_pattern
        .chars()
        .map(|kind| match kind {
            'M' => Ok(HybridDecoderBlock::Mamba2),
            '*' => Ok(HybridDecoderBlock::Attention),
            '-' => Ok(HybridDecoderBlock::DenseMlp),
            'E' => Ok(HybridDecoderBlock::RoutedMoe),
            _ => Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!("unsupported Nemotron-H block {kind:?}"),
            }),
        })
        .collect::<Result<_, _>>()?;
    let moe_layers = layers
        .iter()
        .map(|block| *block == HybridDecoderBlock::RoutedMoe)
        .collect();
    let architecture = ModelArchitectureConfig {
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        vocab_size: raw.vocab_size,
        rms_norm_eps: raw.layer_norm_epsilon as f32,
        rope_theta: raw.rope_theta,
        rope_layers: Some(vec![false; layers.len()]),
        hidden_activation: Some("relu_squared".into()),
        tie_word_embeddings: raw.tie_word_embeddings,
        attention_bias: raw.attention_bias,
        mixture_of_experts: Some(MixtureOfExpertsConfig {
            experts: raw.n_routed_experts,
            experts_per_token: raw.num_experts_per_tok,
            expert_intermediate_size: raw.moe_intermediate_size,
            shared_expert_intermediate_size: Some(raw.moe_shared_expert_intermediate_size),
            normalize_top_k: raw.norm_topk_prob,
        }),
        moe_layers: Some(moe_layers),
        expert_routing: Some(ExpertRoutingConfig {
            scoring: ExpertScoring::Sigmoid,
            groups: Some(ExpertGroups {
                count: raw.n_group,
                per_token: raw.topk_group,
                score: ExpertGroupScore::TopTwoSum,
            }),
            routed_scaling_factor: raw.routed_scaling_factor,
            selection_bias: true,
            normalization: Some(ExpertNormalization::Always),
            shared_expert_gated: false,
        }),
        decoder_extensions: DecoderExtensions {
            mamba2_hybrid: Some(Mamba2HybridConfig {
                layers,
                heads: raw.mamba_num_heads,
                head_dim: raw.mamba_head_dim,
                groups: raw.n_groups,
                state_dim: raw.ssm_state_size,
                conv_kernel: raw.conv_kernel,
                epsilon: raw.layer_norm_epsilon,
                dt_min: 0.0,
                dt_max: None,
                conv_bias: raw.use_conv_bias,
                linear_bias: raw.use_bias,
            }),
            ..Default::default()
        },
        bos_token_id: Some(raw.bos_token_id),
        eos_token_id: Some(raw.eos_token_id),
        ..Default::default()
    };
    architecture
        .validate()
        .map_err(|error| ProductionIngestionError::MalformedMetadata {
            reason: format!("inconsistent pinned Nemotron-H profile: {error}"),
        })?;
    Ok(NanoProfile {
        architecture,
        max_positions: raw.max_position_embeddings,
    })
}

#[cfg(test)]
mod tests;
