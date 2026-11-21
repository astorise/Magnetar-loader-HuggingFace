use super::*;
use crate::weight_layout::{TransposingPayloadSource, swap_declared_projection_shapes};
use crate::weights::discover_and_parse_weights;
use magnetar_runtime::model::MixtureOfExpertsConfig;
use magnetar_runtime::{ModelArtifactSource, production_model_ingestion::ProductionModelSource};
use std::collections::BTreeSet;
use std::io::Write;

/// A tensor to store: its name, safetensors dtype, shape, and values.
struct Stored {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<u64>,
    values: Vec<f32>,
}

fn stored(name: &'static str, dtype: &'static str, shape: &[u64]) -> Stored {
    let count = shape.iter().product::<u64>();
    Stored {
        name,
        dtype,
        shape: shape.to_vec(),
        values: (0..count).map(|index| index as f32 * 0.5 - 3.0).collect(),
    }
}

fn encode(values: &[f32], dtype: &str) -> Vec<u8> {
    match dtype {
        "F32" => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        // Every test value is exact in bf16.
        "BF16" => values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        other => panic!("no encoder for {other}"),
    }
}

/// Writes `tensors` as a one-file bundle, discovers it, derives the layered
/// tensors and transposes the projections, as `ingest` does.
fn ingest(
    tensors: &[Stored],
    config: &ModelArchitectureConfig,
) -> Result<
    (
        Vec<ModelTensorMetadata>,
        TransposingPayloadSource<LayeredTensorSource>,
    ),
    ProductionIngestionError,
> {
    let dir = tempfile::tempdir().unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let start = data.len();
        data.extend(encode(&tensor.values, tensor.dtype));
        header.insert(
            tensor.name.to_string(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
        dir.path().to_path_buf(),
    );
    let (mut inventory, _shards, _quantized, payload) = discover_and_parse_weights(&source)?;
    let payload = derive(&mut inventory, payload, config)?;
    let transposing = TransposingPayloadSource::new(payload, &inventory, &BTreeSet::new());
    swap_declared_projection_shapes(&mut inventory, &BTreeSet::new());
    // Keep the files alive for the reads below.
    std::mem::forget(dir);
    Ok((inventory, transposing))
}

/// Reads tensor `name` as `f32` values, with its logical shape.
fn read(
    inventory: &[ModelTensorMetadata],
    payload: &impl ProductionArtifactPayloadSource,
    name: &str,
) -> (Vec<u64>, Vec<f32>) {
    let tensor = inventory
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor '{name}'"));
    let bytes = payload
        .read_payload(&ProductionPayloadRange {
            identity: tensor.name.clone(),
            offset: tensor.offset_bytes.unwrap(),
            length: tensor.size_bytes.unwrap(),
            digest: tensor.digest.clone(),
        })
        .unwrap();
    (tensor.shape.clone(), decode(&bytes, tensor.storage_dtype))
}

fn layered_config() -> ModelArchitectureConfig {
    ModelArchitectureConfig {
        hidden_size: 3,
        num_hidden_layers: 2,
        per_layer_input_size: Some(2),
        ..Default::default()
    }
}

#[test]
fn each_layer_reads_its_slice_of_the_per_layer_tables() {
    // vocab 4, 2 layers of width 2, hidden 3.
    let embedding = stored(
        "model.language_model.embed_tokens_per_layer.weight",
        "BF16",
        &[4, 4],
    );
    let projection = stored(
        "model.language_model.per_layer_model_projection.weight",
        "F32",
        &[4, 3],
    );
    let (inventory, payload) = ingest(
        &[
            stored("model.language_model.embed_tokens.weight", "F32", &[4, 3]),
            embedding,
            projection,
        ],
        &layered_config(),
    )
    .unwrap();
    assert!(
        !inventory
            .iter()
            .any(|tensor| tensor.name.contains("packed."))
    );
    let embedding = stored("", "BF16", &[4, 4]).values;
    let projection = stored("", "F32", &[4, 3]).values;
    for layer in 0..2 {
        let (shape, values) = read(
            &inventory,
            &payload,
            &format!("layers.{layer}.per_layer_embedding"),
        );
        assert_eq!(shape, vec![4, 2]);
        let expected: Vec<f32> = (0..4)
            .flat_map(|token| (0..2).map(move |column| (token, column)))
            .map(|(token, column)| embedding[token * 4 + layer * 2 + column])
            .collect();
        assert_eq!(values, expected, "layer {layer} embedding");

        // `[hidden, width]`: element (h, p) is stored row `layer * 2 + p`.
        let (shape, values) = read(
            &inventory,
            &payload,
            &format!("layers.{layer}.per_layer_model_projection"),
        );
        assert_eq!(shape, vec![3, 2]);
        let expected: Vec<f32> = (0..3)
            .flat_map(|hidden| (0..2).map(move |p| (hidden, p)))
            .map(|(hidden, p)| projection[(layer * 2 + p) * 3 + hidden])
            .collect();
        assert_eq!(values, expected, "layer {layer} projection");
    }
}

#[test]
fn experts_are_restacked_and_the_router_scale_folded() {
    // 2 experts of width 2, hidden 3.
    let config = ModelArchitectureConfig {
        hidden_size: 3,
        num_hidden_layers: 1,
        mixture_of_experts: Some(MixtureOfExpertsConfig {
            experts: 2,
            experts_per_token: 1,
            expert_intermediate_size: 2,
            shared_expert_intermediate_size: None,
            normalize_top_k: true,
        }),
        ..Default::default()
    };
    let tensors = [
        stored(
            "model.language_model.layers.0.experts.gate_up_proj",
            "BF16",
            &[2, 4, 3],
        ),
        stored(
            "model.language_model.layers.0.experts.down_proj",
            "F32",
            &[2, 3, 2],
        ),
        stored(
            "model.language_model.layers.0.router.proj.weight",
            "BF16",
            &[2, 3],
        ),
        stored("model.language_model.layers.0.router.scale", "F32", &[3]),
        stored(
            "model.language_model.layers.0.router.per_expert_scale",
            "F32",
            &[2],
        ),
    ];
    let (inventory, payload) = ingest(&tensors, &config).unwrap();
    let gate_up = &tensors[0].values;
    let down = &tensors[1].values;
    let router = &tensors[2].values;
    let scale = &tensors[3].values;

    // `[hidden, experts * intermediate]`: (h, e * 2 + i) is expert e's
    // output row i (gate) or 2 + i (up), input h.
    for (name, first_row) in [("experts_gate_proj", 0), ("experts_up_proj", 2)] {
        let (shape, values) = read(&inventory, &payload, &format!("layers.0.moe.{name}"));
        assert_eq!(shape, vec![3, 4], "{name}");
        let expected: Vec<f32> = (0..3)
            .flat_map(|hidden| (0..4).map(move |column| (hidden, column)))
            .map(|(hidden, column)| {
                let (expert, row) = (column / 2, first_row + column % 2);
                gate_up[(expert * 4 + row) * 3 + hidden]
            })
            .collect();
        assert_eq!(values, expected, "{name}");
    }
    // `[intermediate, experts * hidden]`: (i, e * 3 + h) is expert e's
    // output h, input i.
    let (shape, values) = read(&inventory, &payload, "layers.0.moe.experts_down_proj");
    assert_eq!(shape, vec![2, 6]);
    let expected: Vec<f32> = (0..2)
        .flat_map(|i| (0..6).map(move |column| (i, column)))
        .map(|(i, column)| down[((column / 3) * 3 + column % 3) * 2 + i])
        .collect();
    assert_eq!(values, expected);
    // `[hidden, experts]`, in f32, with `scale[h] / sqrt(hidden)` folded in.
    let router_tensor = inventory
        .iter()
        .find(|tensor| tensor.name == "layers.0.moe.router_proj")
        .unwrap();
    assert_eq!(router_tensor.storage_dtype, ModelDType::F32);
    let (shape, values) = read(&inventory, &payload, "layers.0.moe.router_proj");
    assert_eq!(shape, vec![3, 2]);
    let factor = 3f32.powf(-0.5);
    let expected: Vec<f32> = (0..3)
        .flat_map(|hidden| (0..2).map(move |expert| (hidden, expert)))
        .map(|(hidden, expert)| router[expert * 3 + hidden] * scale[hidden] * factor)
        .collect();
    assert_eq!(values, expected);
    let (shape, values) = read(&inventory, &payload, "layers.0.moe.per_expert_scale");
    assert_eq!(shape, vec![2]);
    assert_eq!(values, tensors[4].values);
    assert!(
        !inventory
            .iter()
            .any(|tensor| tensor.name.contains("packed."))
    );
}

#[test]
fn inconsistent_packed_tensors_are_refused() {
    let malformed = |result: Result<_, ProductionIngestionError>| {
        matches!(
            result.map(|_| ()),
            Err(ProductionIngestionError::MalformedMetadata { .. })
        )
    };
    let embedding = || {
        stored(
            "model.language_model.embed_tokens_per_layer.weight",
            "F32",
            &[4, 4],
        )
    };
    // No declared per-layer input width.
    let mut config = layered_config();
    config.per_layer_input_size = None;
    assert!(malformed(ingest(&[embedding()], &config)));
    // A table that is not `layers * width` wide.
    let mut config = layered_config();
    config.per_layer_input_size = Some(3);
    assert!(malformed(ingest(&[embedding()], &config)));
    // A router projection without its scale.
    assert!(malformed(ingest(
        &[stored(
            "model.language_model.layers.0.router.proj.weight",
            "F32",
            &[2, 3]
        )],
        &layered_config(),
    )));
    // Experts that disagree with config.json.
    let mut config = layered_config();
    config.mixture_of_experts = Some(MixtureOfExpertsConfig {
        experts: 4,
        experts_per_token: 1,
        expert_intermediate_size: 2,
        shared_expert_intermediate_size: None,
        normalize_top_k: true,
    });
    assert!(malformed(ingest(
        &[stored(
            "model.language_model.layers.0.experts.gate_up_proj",
            "F32",
            &[2, 4, 3]
        )],
        &config,
    )));
}

/// `stored`, with values starting at `first` so that tensors differ.
fn stored_from(name: &'static str, shape: &[u64], first: f32) -> Stored {
    let mut tensor = stored(name, "F32", shape);
    for value in &mut tensor.values {
        *value += first;
    }
    tensor
}

/// The transposed `[columns, rows.len()]` matrix whose column `r` is row
/// `rows[r]` of `values`, a stored `[_, columns]` matrix.
fn transposed_rows(values: &[f32], columns: usize, rows: &[usize]) -> Vec<f32> {
    (0..columns)
        .flat_map(|column| rows.iter().map(move |row| values[row * columns + column]))
        .collect()
}

/// Each head's rows `[offset, offset + width)` of a `stride`-row block.
fn head_rows(heads: usize, stride: usize, offset: usize, width: usize) -> Vec<usize> {
    (0..heads)
        .flat_map(|head| (0..width).map(move |row| head * stride + offset + row))
        .collect()
}

/// 2 heads, position-independent width 3, rotated width 2, value width 4,
/// latent rank 5, hidden 6.
fn latent_config(q_lora_rank: Option<u32>) -> ModelArchitectureConfig {
    ModelArchitectureConfig {
        hidden_size: 6,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        latent_attention: Some(magnetar_runtime::model::LatentAttentionConfig {
            q_lora_rank,
            kv_lora_rank: 5,
            qk_nope_head_dim: 3,
            qk_rope_head_dim: 2,
            v_head_dim: 4,
            norm_eps: None,
        }),
        ..Default::default()
    }
}

#[test]
fn latent_attention_projections_are_split_by_rows() {
    for q_lora_rank in [None, Some(4)] {
        let query_input = q_lora_rank.map_or(6, |rank| rank as usize);
        let mut tensors = vec![
            stored_from(
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                &[7, 6],
                100.0,
            ),
            stored_from("model.layers.0.self_attn.kv_b_proj.weight", &[14, 5], 200.0),
            stored_from(
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                &[5],
                300.0,
            ),
        ];
        tensors.push(match q_lora_rank {
            None => stored_from("model.layers.0.self_attn.q_proj.weight", &[10, 6], 400.0),
            Some(_) => stored_from("model.layers.0.self_attn.q_b_proj.weight", &[10, 4], 400.0),
        });
        let (inventory, payload) = ingest(&tensors, &latent_config(q_lora_rank)).unwrap();
        let down = &tensors[0].values;
        let up = &tensors[1].values;
        let query = &tensors[3].values;
        let cases = [
            ("q_nope_proj", query, query_input, head_rows(2, 5, 0, 3)),
            ("q_rope_proj", query, query_input, head_rows(2, 5, 3, 2)),
            ("kv_a_proj", down, 6, (0..5).collect()),
            ("k_rope_proj", down, 6, (5..7).collect()),
            ("k_nope_proj", up, 5, head_rows(2, 7, 0, 3)),
            ("v_proj", up, 5, head_rows(2, 7, 3, 4)),
        ];
        for (name, values, columns, rows) in cases {
            let (shape, derived) =
                read(&inventory, &payload, &format!("layers.0.self_attn.{name}"));
            assert_eq!(
                shape,
                vec![columns as u64, rows.len() as u64],
                "{name}, {q_lora_rank:?}"
            );
            assert_eq!(
                derived,
                transposed_rows(values, columns, &rows),
                "{name}, {q_lora_rank:?}"
            );
        }
        // The absorbed decode's tables: the key and value rows as stored.
        for (name, rows) in [
            ("k_absorb", head_rows(2, 7, 0, 3)),
            ("v_absorb", head_rows(2, 7, 3, 4)),
        ] {
            let (shape, derived) =
                read(&inventory, &payload, &format!("layers.0.self_attn.{name}"));
            assert_eq!(shape, vec![rows.len() as u64, 5], "{name}, {q_lora_rank:?}");
            let expected: Vec<f32> = rows
                .iter()
                .flat_map(|row| up[row * 5..(row + 1) * 5].iter().copied())
                .collect();
            assert_eq!(derived, expected, "{name}, {q_lora_rank:?}");
        }
        let (shape, values) = read(&inventory, &payload, "layers.0.self_attn.kv_a_norm");
        assert_eq!(shape, vec![5]);
        assert_eq!(values, tensors[2].values);
        assert!(
            !inventory
                .iter()
                .any(|tensor| tensor.name.contains("packed.") || tensor.name.ends_with("q_proj"))
        );
    }
}

#[test]
fn experts_stored_one_by_one_are_stacked_in_expert_order() {
    // 2 experts of width 2, hidden 3, one shared expert of width 4.
    let config = ModelArchitectureConfig {
        hidden_size: 3,
        num_hidden_layers: 1,
        mixture_of_experts: Some(MixtureOfExpertsConfig {
            experts: 2,
            experts_per_token: 1,
            expert_intermediate_size: 2,
            shared_expert_intermediate_size: Some(4),
            normalize_top_k: false,
        }),
        ..Default::default()
    };
    let tensors = [
        // Expert 1 first: the stack follows the expert index, not the file.
        stored_from(
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            &[2, 3],
            10.0,
        ),
        stored_from(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            &[2, 3],
            20.0,
        ),
        stored_from("model.layers.0.mlp.experts.0.up_proj.weight", &[2, 3], 30.0),
        stored_from("model.layers.0.mlp.experts.1.up_proj.weight", &[2, 3], 40.0),
        stored_from(
            "model.layers.0.mlp.experts.0.down_proj.weight",
            &[3, 2],
            50.0,
        ),
        stored_from(
            "model.layers.0.mlp.experts.1.down_proj.weight",
            &[3, 2],
            60.0,
        ),
        stored_from("model.layers.0.mlp.gate.weight", &[2, 3], 70.0),
        stored_from(
            "model.layers.0.mlp.gate.e_score_correction_bias",
            &[2],
            80.0,
        ),
        stored_from(
            "model.layers.0.mlp.shared_experts.up_proj.weight",
            &[4, 3],
            90.0,
        ),
    ];
    let (inventory, payload) = ingest(&tensors, &config).unwrap();
    let concat =
        |first: &Stored, second: &Stored| [first.values.clone(), second.values.clone()].concat();
    let cases = [
        ("experts_gate_proj", concat(&tensors[1], &tensors[0]), 3, 4),
        ("experts_up_proj", concat(&tensors[2], &tensors[3]), 3, 4),
        ("experts_down_proj", concat(&tensors[4], &tensors[5]), 2, 6),
        ("router_proj", tensors[6].values.clone(), 3, 2),
        ("shared_up_proj", tensors[8].values.clone(), 3, 4),
    ];
    for (name, stacked, columns, rows) in cases {
        let (shape, values) = read(&inventory, &payload, &format!("layers.0.moe.{name}"));
        assert_eq!(shape, vec![columns as u64, rows as u64], "{name}");
        assert_eq!(
            values,
            transposed_rows(&stacked, columns, &(0..rows).collect::<Vec<_>>()),
            "{name}"
        );
    }
    let (shape, values) = read(&inventory, &payload, "layers.0.moe.selection_bias");
    assert_eq!(shape, vec![2]);
    assert_eq!(values, tensors[7].values);
    assert!(
        !inventory
            .iter()
            .any(|tensor| tensor.name.contains("packed."))
    );
}

#[test]
fn inconsistent_latent_or_per_expert_tensors_are_refused() {
    let malformed = |result: Result<_, ProductionIngestionError>| {
        matches!(
            result.map(|_| ()),
            Err(ProductionIngestionError::MalformedMetadata { .. })
        )
    };
    let latent = || {
        vec![
            stored("model.layers.0.self_attn.q_proj.weight", "F32", &[10, 6]),
            stored(
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "F32",
                &[7, 6],
            ),
            stored("model.layers.0.self_attn.kv_b_proj.weight", "F32", &[14, 5]),
        ]
    };
    assert!(ingest(&latent(), &latent_config(None)).is_ok());
    // No declared latent attention.
    let mut config = latent_config(None);
    config.latent_attention = None;
    assert!(malformed(ingest(&latent(), &config)));
    // A low-rank query declared, but the full query projection stored.
    assert!(malformed(ingest(&latent(), &latent_config(Some(4)))));
    // A key and value projection of the wrong height.
    let mut tensors = latent();
    tensors[2] = stored("model.layers.0.self_attn.kv_b_proj.weight", "F32", &[12, 5]);
    assert!(malformed(ingest(&tensors, &latent_config(None))));

    let config = ModelArchitectureConfig {
        hidden_size: 3,
        num_hidden_layers: 1,
        mixture_of_experts: Some(MixtureOfExpertsConfig {
            experts: 2,
            experts_per_token: 1,
            expert_intermediate_size: 2,
            shared_expert_intermediate_size: None,
            normalize_top_k: false,
        }),
        ..Default::default()
    };
    let expert = |name| stored(name, "F32", &[2, 3]);
    // Expert 1's projection missing.
    assert!(malformed(ingest(
        &[expert("model.layers.0.mlp.experts.0.gate_proj.weight")],
        &config,
    )));
    // An expert of another width.
    assert!(malformed(ingest(
        &[
            expert("model.layers.0.mlp.experts.0.gate_proj.weight"),
            stored(
                "model.layers.0.mlp.experts.1.gate_proj.weight",
                "F32",
                &[3, 3]
            ),
        ],
        &config,
    )));
    // No declared mixture of experts.
    let mut no_experts = config.clone();
    no_experts.mixture_of_experts = None;
    assert!(malformed(ingest(
        &[
            expert("model.layers.0.mlp.experts.0.gate_proj.weight"),
            expert("model.layers.0.mlp.experts.1.gate_proj.weight"),
        ],
        &no_experts,
    )));
}
