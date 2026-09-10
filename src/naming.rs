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

/// Normalizes one real Hugging Face Qwen tensor name into its canonical
/// Model Artifact name, or rejects it structurally:
/// - a bias tensor (`*.bias`) is rejected with `UnsupportedFormat`: the
///   production Qwen Component's graph does not yet add a bias term to
///   any projection, so silently dropping one would produce numerically
///   wrong results rather than a fail-closed error (a real, separate gap
///   this change's non-goals do not claim to close).
/// - an unrecognized name is rejected with `MalformedMetadata` naming it,
///   rather than passed through unrecognized (which would fail later,
///   opaquely, only when the Component's `weight-edge` call cannot find
///   it).
pub fn normalize_tensor_name(raw: &str) -> Result<String, ProductionIngestionError> {
    if let Some(suffix) = raw.strip_suffix(".bias") {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "tensor '{raw}' is a bias term for '{suffix}'; the production Qwen Component \
                 graph does not yet support attention/MLP bias terms"
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
    fn rejects_bias_tensors() {
        let error = normalize_tensor_name("model.layers.0.self_attn.q_proj.bias").unwrap_err();
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
