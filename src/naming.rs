//! Normalizes real Hugging Face Qwen checkpoint tensor names (e.g.
//! `model.layers.0.self_attn.q_proj.weight`) into the canonical Model
//! Artifact names `magnetar-runtime`'s Qwen Component's `weight-edge`
//! calls resolve directly (e.g. `layers.0.self_attn.q_proj`) -- see
//! `components/qwen/src/lib.rs`'s own `weight_tensor_name` doc comment:
//! "Supplying the canonical Model Artifact name directly ... is what lets
//! Model Loading resolve weight edges by a single, generic prefix strip
//! instead of carrying a Qwen-specific suffix-mapping table itself."
//! Ingestion is where that normalization happens for real checkpoint
//! names, so the Component and Runtime never see Hugging Face-specific
//! naming at all.

use magnetar_runtime::production_model_ingestion::ProductionIngestionError;

/// The prefixes of a multimodal bundle's non-text towers. Their tensors are
/// not part of the text decoder and are left out of the inventory.
const NON_TEXT_PREFIXES: [&str; 8] = [
    "model.vision_tower.",
    "model.audio_tower.",
    "model.embed_vision.",
    "model.embed_audio.",
    "model.multi_modal_projector.",
    // `gemma4_unified` bundles embed image patches directly.
    "model.vision_embedder.",
    // Qwen3.5's vision tower, and its multi-token prediction head, which
    // the text decoder's forward pass does not use.
    "model.visual.",
    "mtp.",
];

/// Whether `raw` belongs to a multimodal bundle's vision or audio tower
/// rather than to its text decoder.
pub fn is_non_text_tensor(raw: &str) -> bool {
    NON_TEXT_PREFIXES
        .iter()
        .any(|prefix| raw.starts_with(prefix))
        || is_derived_buffer(raw)
}

/// Qwen4-Exp's n-gram embedding stores the hash of its layer (the head
/// vocabulary sizes and offsets, the multipliers) as integer buffers. The
/// `ngram-embedding` operator derives them from the configuration, exactly
/// as Transformers does, so the stored ones are left out of the inventory.
fn is_derived_buffer(raw: &str) -> bool {
    raw.contains(".ple.ple_embedding.")
        && [
            ".layer_multipliers",
            ".ngram_heads_offsets",
            ".ngram_heads_vocab_sizes",
        ]
        .iter()
        .any(|suffix| raw.ends_with(suffix))
}

/// Normalizes one real Hugging Face Qwen tensor name into its canonical
/// Model Artifact name, or rejects it structurally:
/// - `self_attn.{q,k,v}_proj.bias` is accepted and canonicalized to
///   `layers.N.self_attn.{q,k,v}_bias`: a real Qwen2/2.5 architectural
///   default (every such checkpoint declares these three), which the
///   production Qwen Component's graph adds via a broadcast `add` node
///   after the matching projection's matmul.
/// - every other bias tensor (`o_proj.bias`, any MLP bias, `lm_head.bias`)
///   is rejected with `UnsupportedFormat`: the production Qwen Component's
///   graph does not add a bias term to those projections, so silently
///   dropping one would produce numerically wrong results rather than a
///   fail-closed error (a real, separate gap this change's non-goals do
///   not claim to close).
/// - an unrecognized name is rejected with `MalformedMetadata` naming it,
///   rather than passed through unrecognized (which would fail later,
///   opaquely, only when the Component's `weight-edge` call cannot find
///   it).
/// - a multimodal bundle's text decoder (`model.language_model.*`) takes
///   the same names, plus the layered decoders' own norms, `q_norm`/`k_norm`,
///   `layer_scalar` and per-layer input weights. Tables packing every
///   layer's weights and routed experts take internal names that
///   `layered_tensors.rs` replaces with the per-layer and stacked tensors.
/// - a latent-attention layer's fused projections and each routed
///   expert's own projections take internal names too: `layered_tensors.rs`
///   splits the former by rows and stacks the latter.
pub fn normalize_tensor_name(raw: &str) -> Result<String, ProductionIngestionError> {
    // A multimodal bundle nests its text decoder under `language_model`.
    let unnested;
    let raw = match raw.strip_prefix("model.language_model.") {
        Some(rest) => {
            unnested = format!("model.{rest}");
            unnested.as_str()
        }
        None => raw,
    };
    // Qwen3.5's gated-delta parameters have no `.weight` suffix.
    if let Some((layer_index, field)) = raw
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
        && layer_index.parse::<u64>().is_ok()
        && let Some(canonical) = match field {
            "linear_attn.A_log" => Some("linear_attn.a_log"),
            "linear_attn.dt_bias" => Some("linear_attn.dt_bias"),
            _ => None,
        }
    {
        return Ok(format!("layers.{layer_index}.{canonical}"));
    }
    // A per-layer scalar has no `.weight` suffix.
    if let Some(layer_index) = raw
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.strip_suffix(".layer_scalar"))
        && layer_index.parse::<u64>().is_ok()
    {
        return Ok(format!("layers.{layer_index}.layer_scalar"));
    }
    // Routed experts' stacked tensors and router scales carry no `.weight`
    // suffix either; `layered_tensors.rs` restacks the experts and folds
    // the router scale.
    if let Some((layer_index, field)) = raw
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
        && layer_index.parse::<u64>().is_ok()
    {
        let canonical = match field {
            "experts.gate_up_proj" | "mlp.experts.gate_up_proj" => {
                Some(crate::layered_tensors::EXPERTS_GATE_UP)
            }
            "experts.down_proj" | "mlp.experts.down_proj" => Some("moe.experts_down_proj"),
            "router.scale" => Some(crate::layered_tensors::ROUTER_SCALE),
            "router.per_expert_scale" => Some("moe.per_expert_scale"),
            // A sigmoid router's selection-only bias per expert.
            "mlp.gate.e_score_correction_bias" => Some("moe.selection_bias"),
            // Olala's learned scalar residual parameters and its separately
            // stored buffer have no `.weight` suffix. Keep mixer and MLP
            // identities distinct, including for nonzero layer indices.
            "geodesic_mixer.scale"
            | "geodesic_mixer.bias"
            | "geodesic_mixer.prosres_scalar"
            | "geodesic_mlp.scale"
            | "geodesic_mlp.bias"
            | "geodesic_mlp.prosres_scalar" => Some(field),
            _ => None,
        };
        if let Some(canonical) = canonical {
            return Ok(format!("layers.{layer_index}.{canonical}"));
        }
    }
    if let Some(suffix) = raw.strip_suffix(".bias") {
        if let Some(layer_suffix) = suffix.strip_prefix("model.layers.")
            && let Some((layer_index, rest)) = layer_suffix.split_once('.')
            && layer_index.parse::<u64>().is_ok()
        {
            let canonical_bias_suffix = match rest {
                "self_attn.q_proj" => Some("self_attn.q_bias"),
                "self_attn.k_proj" => Some("self_attn.k_bias"),
                "self_attn.v_proj" => Some("self_attn.v_bias"),
                _ => None,
            };
            if let Some(canonical_bias_suffix) = canonical_bias_suffix {
                return Ok(format!("layers.{layer_index}.{canonical_bias_suffix}"));
            }
        }
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "tensor '{raw}' is a bias term for '{suffix}'; the production Qwen Component \
                 graph only supports attention q/k/v_proj bias terms"
            ),
        });
    }
    let Some(name) = raw.strip_suffix(".weight") else {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!("tensor '{raw}' does not end in '.weight' or '.bias'"),
        });
    };

    if name == "model.embed_tokens" {
        return Ok("token_embedding".to_string());
    }
    if name == "model.norm" {
        return Ok("final_norm".to_string());
    }
    if name == "lm_head" {
        return Ok("lm_head".to_string());
    }
    // Qwen4-Exp's final residual-stream mixer: the same weights as a block's
    // gated residual, without the injection (`support-qwen38-flash-next`).
    // Their names leave out `proj`: the `gated-residual` operators take
    // them in checkpoint `[out, in]` order (`weight_layout.rs`).
    if let Some(field) = name.strip_prefix("model.hyper_connection_mixer.") {
        return match field {
            "hc_norm" => Ok("final_hyper.norm".to_string()),
            "input_mix_weight_down" => Ok("final_hyper.down".to_string()),
            "input_mix_weight_up" => Ok("final_hyper.up".to_string()),
            other => Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("tensor '{raw}' names an unrecognized mixer field '{other}'"),
            }),
        };
    }
    if name == "model.per_layer_projection_norm" {
        return Ok("per_layer_projection_norm".to_string());
    }
    // Each layer reads its own column (row) slice of these tables
    // (`layered_tensors.rs`).
    if name == "model.embed_tokens_per_layer" {
        return Ok(crate::layered_tensors::PER_LAYER_EMBEDDING_TABLE.to_string());
    }
    if name == "model.per_layer_model_projection" {
        return Ok(crate::layered_tensors::PER_LAYER_PROJECTION_TABLE.to_string());
    }
    if let Some(layer_suffix) = name.strip_prefix("model.layers.") {
        let (layer_index, rest) = layer_suffix.split_once('.').ok_or_else(|| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("tensor '{raw}' has an unrecognized per-layer name shape"),
            }
        })?;
        layer_index
            .parse::<u64>()
            .map_err(|_| ProductionIngestionError::MalformedMetadata {
                reason: format!("tensor '{raw}' has a non-numeric layer index '{layer_index}'"),
            })?;
        // One routed expert's own projection; `layered_tensors.rs` stacks
        // a layer's experts in expert order.
        if let Some((expert, projection)) = rest
            .strip_prefix("mlp.experts.")
            .and_then(|expert| expert.split_once('.'))
            && expert.parse::<u64>().is_ok()
            && matches!(projection, "gate_proj" | "up_proj" | "down_proj")
        {
            return Ok(format!(
                "layers.{layer_index}.{}{expert}.{projection}",
                crate::layered_tensors::EXPERT_PREFIX
            ));
        }
        // One shard of a layer's n-gram table; `layered_tensors.rs`
        // stacks a layer's shards in order.
        if let Some(shard) = rest.strip_prefix("ple.ple_embedding.ngram_embedding.shard_")
            && shard.parse::<u64>().is_ok()
        {
            return Ok(format!(
                "layers.{layer_index}.{}{shard}",
                crate::layered_tensors::NGRAM_SHARD_PREFIX
            ));
        }
        let canonical_suffix = match rest {
            "input_layernorm" => "input_norm",
            "post_attention_layernorm" => "post_attn_norm",
            "self_attn.q_proj" => "self_attn.q_proj",
            "self_attn.k_proj" => "self_attn.k_proj",
            "self_attn.v_proj" => "self_attn.v_proj",
            "self_attn.o_proj" => "self_attn.o_proj",
            "mlp.gate_proj" => "mlp.gate_proj",
            "mlp.up_proj" => "mlp.up_proj",
            "mlp.down_proj" => "mlp.down_proj",
            // Layered decoders (`per-layer-kv-geometry`).
            "pre_feedforward_layernorm" => "pre_ffn_norm",
            "post_feedforward_layernorm" => "post_ffn_norm",
            "pre_feedforward_layernorm_2" => "pre_ffn_norm_2",
            "post_feedforward_layernorm_1" => "post_ffn_norm_1",
            "post_feedforward_layernorm_2" => "post_ffn_norm_2",
            "self_attn.q_norm" => "self_attn.q_norm",
            "self_attn.k_norm" => "self_attn.k_norm",
            "per_layer_input_gate" => "per_layer_input_gate",
            "per_layer_projection" => "per_layer_projection",
            "post_per_layer_input_norm" => "post_per_layer_input_norm",
            "router.proj" => crate::layered_tensors::ROUTER_PROJECTION,
            // Qwen3.5's gated-delta mixer (`support-qwen38-flash-next`).
            // The `gated-delta-mixer` operator takes these projections in
            // checkpoint `[out, in]` order, so their names leave out
            // `proj`, which would transpose them (`weight_layout.rs`).
            "linear_attn.in_proj_qkv" => "linear_attn.in_qkv",
            "linear_attn.in_proj_z" => "linear_attn.in_z",
            "linear_attn.in_proj_a" => "linear_attn.in_a",
            "linear_attn.in_proj_b" => "linear_attn.in_b",
            "linear_attn.conv1d" => "linear_attn.conv1d",
            "linear_attn.norm" => "linear_attn.norm",
            "linear_attn.out_proj" => "linear_attn.out",
            // Latent attention (`express-latent-attention-forms`): the
            // fused projections take internal names that
            // `layered_tensors.rs` splits by rows.
            "self_attn.q_a_proj" => "self_attn.q_a_proj",
            "self_attn.q_a_layernorm" => "self_attn.q_a_norm",
            "self_attn.q_b_proj" => crate::layered_tensors::QUERY_UP_PROJECTION,
            "self_attn.kv_a_proj_with_mqa" => crate::layered_tensors::LATENT_DOWN_PROJECTION,
            "self_attn.kv_a_layernorm" => "self_attn.kv_a_norm",
            "self_attn.kv_b_proj" => crate::layered_tensors::LATENT_UP_PROJECTION,
            // Qwen4-Exp's residual streams (`support-qwen38-flash-next`).
            "attn_hyper_connection.hc_norm" => "attn_hyper.norm",
            "attn_hyper_connection.input_mix_weight_down" => "attn_hyper.down",
            "attn_hyper_connection.input_mix_weight_up" => "attn_hyper.up",
            "attn_hyper_connection.block_inject_weight" => "attn_hyper.inject",
            "mlp_hyper_connection.hc_norm" => "mlp_hyper.norm",
            "mlp_hyper_connection.input_mix_weight_down" => "mlp_hyper.down",
            "mlp_hyper_connection.input_mix_weight_up" => "mlp_hyper.up",
            "mlp_hyper_connection.block_inject_weight" => "mlp_hyper.inject",
            // Its token indexer: one projection for the queries and the
            // single key head, and their norms.
            "self_attn.indexer.index_qk_proj" => "self_attn.index_qk_proj",
            "self_attn.indexer.q_layernorm" => "self_attn.index_q_norm",
            "self_attn.indexer.k_layernorm" => "self_attn.index_k_norm",
            // Its n-gram embedding: the table in shards of rows, the
            // projections of the embedding into the streams' gate and
            // value, and the convolution (all in checkpoint `[out, in]`
            // order, so their names leave out `proj`).
            "ple.key_proj" => "ple.key",
            "ple.value_proj" => "ple.value",
            "ple.norm_key" => "ple.norm_key",
            "ple.norm_query" => "ple.norm_query",
            "ple.norm_conv" => "ple.norm_conv",
            "ple.conv1d" => "ple.conv1d",
            // A shared expert behind a sigmoid gate.
            "mlp.shared_expert.gate_proj" => "moe.shared_gate_proj",
            "mlp.shared_expert.up_proj" => "moe.shared_up_proj",
            "mlp.shared_expert.down_proj" => "moe.shared_down_proj",
            "mlp.shared_expert_gate" => "moe.shared_expert_gate",
            // An unscaled router and ungated shared experts.
            "mlp.gate" => "moe.router_proj",
            "mlp.shared_experts.gate_proj" => "moe.shared_gate_proj",
            "mlp.shared_experts.up_proj" => "moe.shared_up_proj",
            "mlp.shared_experts.down_proj" => "moe.shared_down_proj",
            other => {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "tensor '{raw}' names an unrecognized per-layer field '{other}'"
                    ),
                });
            }
        };
        return Ok(format!("layers.{layer_index}.{canonical_suffix}"));
    }

    Err(ProductionIngestionError::MalformedMetadata {
        reason: format!("tensor '{raw}' does not match any recognized Qwen tensor name shape"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_top_level_tensors() {
        assert_eq!(
            normalize_tensor_name("model.embed_tokens.weight").unwrap(),
            "token_embedding"
        );
        assert_eq!(
            normalize_tensor_name("model.norm.weight").unwrap(),
            "final_norm"
        );
        assert_eq!(normalize_tensor_name("lm_head.weight").unwrap(), "lm_head");
    }

    #[test]
    fn normalizes_per_layer_tensors() {
        assert_eq!(
            normalize_tensor_name("model.layers.0.self_attn.q_proj.weight").unwrap(),
            "layers.0.self_attn.q_proj"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.12.mlp.down_proj.weight").unwrap(),
            "layers.12.mlp.down_proj"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.3.input_layernorm.weight").unwrap(),
            "layers.3.input_norm"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.3.post_attention_layernorm.weight").unwrap(),
            "layers.3.post_attn_norm"
        );
    }

    #[test]
    fn normalizes_qkv_bias_tensors() {
        assert_eq!(
            normalize_tensor_name("model.layers.0.self_attn.q_proj.bias").unwrap(),
            "layers.0.self_attn.q_bias"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.0.self_attn.k_proj.bias").unwrap(),
            "layers.0.self_attn.k_bias"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.7.self_attn.v_proj.bias").unwrap(),
            "layers.7.self_attn.v_bias"
        );
    }

    #[test]
    fn rejects_bias_tensors_outside_qkv() {
        let error = normalize_tensor_name("model.layers.0.self_attn.o_proj.bias").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
        let error = normalize_tensor_name("model.layers.0.mlp.gate_proj.bias").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
        let error = normalize_tensor_name("lm_head.bias").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }

    #[test]
    fn rejects_unrecognized_names() {
        let error = normalize_tensor_name("model.layers.0.some_new_field.weight").unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
        let error2 = normalize_tensor_name("something_else_entirely").unwrap_err();
        assert!(matches!(
            error2,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }
}

#[cfg(test)]
mod layered_tests {
    use super::*;

    #[test]
    fn a_nested_text_decoder_takes_the_layered_logical_names() {
        let cases = [
            (
                "model.language_model.embed_tokens.weight",
                "token_embedding",
            ),
            ("model.language_model.norm.weight", "final_norm"),
            (
                "model.language_model.layers.4.self_attn.q_norm.weight",
                "layers.4.self_attn.q_norm",
            ),
            (
                "model.language_model.layers.4.self_attn.k_proj.weight",
                "layers.4.self_attn.k_proj",
            ),
            (
                "model.language_model.layers.7.pre_feedforward_layernorm.weight",
                "layers.7.pre_ffn_norm",
            ),
            (
                "model.language_model.layers.7.post_feedforward_layernorm_2.weight",
                "layers.7.post_ffn_norm_2",
            ),
            (
                "model.language_model.layers.7.layer_scalar",
                "layers.7.layer_scalar",
            ),
            (
                "model.language_model.layers.2.per_layer_input_gate.weight",
                "layers.2.per_layer_input_gate",
            ),
            (
                "model.language_model.per_layer_projection_norm.weight",
                "per_layer_projection_norm",
            ),
        ];
        for (raw, canonical) in cases {
            assert_eq!(normalize_tensor_name(raw).unwrap(), canonical, "{raw}");
        }
    }

    #[test]
    fn qwen35_mixer_tensors_keep_checkpoint_order_names() {
        for (raw, canonical) in [
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                "layers.0.linear_attn.in_qkv",
            ),
            (
                "model.language_model.layers.1.linear_attn.in_proj_z.weight",
                "layers.1.linear_attn.in_z",
            ),
            (
                "model.language_model.layers.2.linear_attn.in_proj_a.weight",
                "layers.2.linear_attn.in_a",
            ),
            (
                "model.language_model.layers.2.linear_attn.in_proj_b.weight",
                "layers.2.linear_attn.in_b",
            ),
            (
                "model.language_model.layers.4.linear_attn.conv1d.weight",
                "layers.4.linear_attn.conv1d",
            ),
            (
                "model.language_model.layers.5.linear_attn.A_log",
                "layers.5.linear_attn.a_log",
            ),
            (
                "model.language_model.layers.5.linear_attn.dt_bias",
                "layers.5.linear_attn.dt_bias",
            ),
            (
                "model.language_model.layers.6.linear_attn.norm.weight",
                "layers.6.linear_attn.norm",
            ),
            (
                "model.language_model.layers.6.linear_attn.out_proj.weight",
                "layers.6.linear_attn.out",
            ),
        ] {
            let name = normalize_tensor_name(raw).unwrap();
            assert_eq!(name, canonical);
            // The mixer takes these in checkpoint `[out, in]` order.
            assert!(!name.contains("proj"), "{name}");
        }
        assert!(is_non_text_tensor("model.visual.blocks.0.attn.qkv.weight"));
        assert!(is_non_text_tensor("mtp.layers.0.self_attn.q_proj.weight"));
    }

    /// Every tensor name of the pinned Qwen3.5-0.8B-Base safetensors header
    /// (revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68) is either left
    /// out (vision tower, MTP head) or takes a canonical name.
    #[test]
    fn every_qwen35_checkpoint_tensor_is_named_or_left_out() {
        let names = include_str!("../testdata/qwen35/Qwen3.5-0.8B-Base.tensor-names.txt");
        let names: Vec<&str> = names.lines().collect();
        assert_eq!(names.len(), 488);
        let text: Vec<&str> = names
            .iter()
            .copied()
            .filter(|name| !is_non_text_tensor(name))
            .collect();
        assert_eq!(names.len() - text.len(), 153 + 15);
        let canonical: std::collections::BTreeSet<String> = text
            .iter()
            .map(|name| normalize_tensor_name(name).unwrap())
            .collect();
        assert_eq!(canonical.len(), text.len());
        assert_eq!(
            canonical
                .iter()
                .filter(|name| name.contains(".linear_attn."))
                .count(),
            18 * 9
        );
    }

    #[test]
    fn non_text_towers_are_recognized() {
        assert!(is_non_text_tensor(
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight"
        ));
        assert!(is_non_text_tensor(
            "model.embed_audio.embedding_projection.weight"
        ));
        assert!(is_non_text_tensor(
            "model.vision_embedder.patch_dense.weight"
        ));
        assert!(!is_non_text_tensor("model.language_model.norm.weight"));
    }

    /// Packed per-layer tables and routed experts take the names
    /// `layered_tensors.rs` derives the requested tensors from.
    #[test]
    fn packed_tables_and_experts_take_their_derivation_names() {
        let cases = [
            (
                "model.language_model.embed_tokens_per_layer.weight",
                "packed.embed_tokens_per_layer",
            ),
            (
                "model.language_model.per_layer_model_projection.weight",
                "packed.per_layer_model_projection",
            ),
            (
                "model.language_model.layers.3.experts.gate_up_proj",
                "layers.3.packed.experts_gate_up",
            ),
            (
                "model.language_model.layers.3.experts.down_proj",
                "layers.3.moe.experts_down_proj",
            ),
            (
                "model.language_model.layers.3.router.proj.weight",
                "layers.3.packed.router_proj",
            ),
            (
                "model.language_model.layers.3.router.scale",
                "layers.3.packed.router_scale",
            ),
            (
                "model.language_model.layers.3.router.per_expert_scale",
                "layers.3.moe.per_expert_scale",
            ),
        ];
        for (raw, canonical) in cases {
            assert_eq!(normalize_tensor_name(raw).unwrap(), canonical, "{raw}");
        }
    }

    /// A latent-attention decoder's fused projections, per-expert weights,
    /// router and shared experts.
    #[test]
    fn latent_attention_and_per_expert_tensors_take_their_names() {
        let cases = [
            (
                "model.layers.2.self_attn.q_a_proj.weight",
                "layers.2.self_attn.q_a_proj",
            ),
            (
                "model.layers.2.self_attn.q_a_layernorm.weight",
                "layers.2.self_attn.q_a_norm",
            ),
            (
                "model.layers.2.self_attn.q_b_proj.weight",
                "layers.2.packed.q_b_proj",
            ),
            (
                "model.layers.2.self_attn.kv_a_proj_with_mqa.weight",
                "layers.2.packed.kv_a_proj_with_mqa",
            ),
            (
                "model.layers.2.self_attn.kv_a_layernorm.weight",
                "layers.2.self_attn.kv_a_norm",
            ),
            (
                "model.layers.2.self_attn.kv_b_proj.weight",
                "layers.2.packed.kv_b_proj",
            ),
            ("model.layers.2.mlp.gate.weight", "layers.2.moe.router_proj"),
            (
                "model.layers.2.mlp.gate.e_score_correction_bias",
                "layers.2.moe.selection_bias",
            ),
            (
                "model.layers.2.mlp.experts.17.down_proj.weight",
                "layers.2.packed.experts.17.down_proj",
            ),
            (
                "model.layers.2.mlp.shared_experts.up_proj.weight",
                "layers.2.moe.shared_up_proj",
            ),
            // A dense layer's MLP keeps its names.
            (
                "model.layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.gate_proj",
            ),
        ];
        for (raw, canonical) in cases {
            assert_eq!(normalize_tensor_name(raw).unwrap(), canonical, "{raw}");
        }
        assert!(normalize_tensor_name("model.layers.2.mlp.experts.x.up_proj.weight").is_err());
        assert!(normalize_tensor_name("model.layers.2.mlp.experts.3.fc1.weight").is_err());
    }

    /// `support-qwen38-flash-next`: Qwen4-Exp's residual-stream mixers,
    /// indexer and shared-expert gate take their own canonical names, none
    /// of the stream or injection weights with a `proj` that would
    /// transpose them.
    #[test]
    fn normalizes_qwen4_exp_tensors() {
        let cases = [
            (
                "model.layers.3.attn_hyper_connection.hc_norm.weight",
                "layers.3.attn_hyper.norm",
            ),
            (
                "model.layers.3.attn_hyper_connection.input_mix_weight_down.weight",
                "layers.3.attn_hyper.down",
            ),
            (
                "model.layers.3.attn_hyper_connection.input_mix_weight_up.weight",
                "layers.3.attn_hyper.up",
            ),
            (
                "model.layers.3.attn_hyper_connection.block_inject_weight.weight",
                "layers.3.attn_hyper.inject",
            ),
            (
                "model.layers.3.mlp_hyper_connection.block_inject_weight.weight",
                "layers.3.mlp_hyper.inject",
            ),
            (
                "model.hyper_connection_mixer.hc_norm.weight",
                "final_hyper.norm",
            ),
            (
                "model.hyper_connection_mixer.input_mix_weight_up.weight",
                "final_hyper.up",
            ),
            (
                "model.layers.1.self_attn.indexer.index_qk_proj.weight",
                "layers.1.self_attn.index_qk_proj",
            ),
            (
                "model.layers.1.self_attn.indexer.k_layernorm.weight",
                "layers.1.self_attn.index_k_norm",
            ),
            (
                "model.layers.2.mlp.shared_expert.up_proj.weight",
                "layers.2.moe.shared_up_proj",
            ),
            (
                "model.layers.2.mlp.shared_expert_gate.weight",
                "layers.2.moe.shared_expert_gate",
            ),
        ];
        for (raw, canonical) in cases {
            assert_eq!(normalize_tensor_name(raw).unwrap(), canonical, "{raw}");
        }
        assert!(normalize_tensor_name("model.hyper_connection_mixer.unknown.weight").is_err());
        // The n-gram embedding: its table shards and its projections; the
        // hash buffers derived from the configuration are left out.
        for (raw, canonical) in [
            (
                "model.layers.0.ple.ple_embedding.ngram_embedding.shard_3.weight",
                "layers.0.packed.ple_table.3",
            ),
            ("model.layers.0.ple.key_proj.weight", "layers.0.ple.key"),
            ("model.layers.0.ple.value_proj.weight", "layers.0.ple.value"),
            (
                "model.layers.0.ple.norm_conv.weight",
                "layers.0.ple.norm_conv",
            ),
            ("model.layers.0.ple.conv1d.weight", "layers.0.ple.conv1d"),
        ] {
            assert_eq!(normalize_tensor_name(raw).unwrap(), canonical, "{raw}");
        }
        for buffer in [
            "model.layers.0.ple.ple_embedding.layer_multipliers",
            "model.layers.0.ple.ple_embedding.ngram_heads_offsets",
            "model.layers.0.ple.ple_embedding.ngram_heads_vocab_sizes",
        ] {
            assert!(is_non_text_tensor(buffer), "{buffer}");
        }
        assert!(!is_non_text_tensor(
            "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
        ));
        // The shared expert's gate is an `nn.Linear` read as `[in, out]`;
        // the stream weights keep the checkpoint's order.
        assert!(crate::weight_layout::is_transposed_projection(
            "layers.2.moe.shared_expert_gate"
        ));
        assert!(!crate::weight_layout::is_transposed_projection(
            "layers.3.attn_hyper.down"
        ));
        assert!(!crate::weight_layout::is_transposed_projection(
            "final_hyper.up"
        ));
    }
}
