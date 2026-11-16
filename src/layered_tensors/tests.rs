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
