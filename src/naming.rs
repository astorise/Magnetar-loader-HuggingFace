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
const NON_TEXT_PREFIXES: [&str; 6] = [
    "model.vision_tower.",
    "model.audio_tower.",
    "model.embed_vision.",
    "model.embed_audio.",
    "model.multi_modal_projector.",
    // `gemma4_unified` bundles embed image patches directly.
    "model.vision_embedder.",
];

/// Whether `raw` belongs to a multimodal bundle's vision or audio tower
/// rather than to its text decoder.
pub fn is_non_text_tensor(raw: &str) -> bool {
    NON_TEXT_PREFIXES
        .iter()
        .any(|prefix| raw.starts_with(prefix))
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
            "experts.gate_up_proj" => Some(crate::layered_tensors::EXPERTS_GATE_UP),
            "experts.down_proj" => Some("moe.experts_down_proj"),
            "router.scale" => Some(crate::layered_tensors::ROUTER_SCALE),
            "router.per_expert_scale" => Some("moe.per_expert_scale"),
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
}
