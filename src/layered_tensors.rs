//! A layered decoder's packed tensors, cut into the tensors its Model
//! Component requests (`load-layered-checkpoint-tensors`):
//! - `embed_tokens_per_layer` (`[vocab, layers * width]`) packs every
//!   layer's per-layer input embedding; layer `N` reads columns
//!   `[N * width, (N + 1) * width)`, as `layers.N.per_layer_embedding`.
//! - `per_layer_model_projection` (`nn.Linear`, `[layers * width, hidden]`)
//!   packs every layer's context projection; layer `N` reads rows
//!   `[N * width, (N + 1) * width)`, as `layers.N.per_layer_model_projection`.
//! - `experts.gate_up_proj` (`[experts, 2 * intermediate, hidden]`, each
//!   expert's gate rows before its up rows) becomes the stacked
//!   `moe.experts_gate_proj` and `moe.experts_up_proj` (`[experts *
//!   intermediate, hidden]`), and `experts.down_proj` (`[experts, hidden,
//!   intermediate]`) the stacked `moe.experts_down_proj` (`[experts * hidden,
//!   intermediate]`, the same bytes): expert `e` owns rows `[e * out, (e + 1)
//!   * out)`, which the projection transpose turns into columns.
//! - `router.proj` with `router.scale` and `hidden^-0.5` folded into its
//!   input columns becomes `moe.router_proj`, in `f32`: the router computes
//!   `proj(norm(x) * scale * hidden^-0.5)`, and this fold is exact for a
//!   float projection.
//! - A latent-attention layer's fused projections are split by rows
//!   (`express-latent-attention-forms`): the query projection (`q_proj`, or
//!   `q_b_proj` after a low-rank query) stores each head's
//!   position-independent rows before its rotated rows, and becomes
//!   `self_attn.q_nope_proj` and `self_attn.q_rope_proj`;
//!   `kv_a_proj_with_mqa` stores the latent's rows before the shared
//!   rotated key's, and becomes `self_attn.kv_a_proj` and
//!   `self_attn.k_rope_proj`; `kv_b_proj` stores each head's key rows
//!   before its value rows, and becomes `self_attn.k_nope_proj` and
//!   `self_attn.v_proj`. Row splits copy bytes, so they are exact. The
//!   same key and value rows are also `self_attn.k_absorb` and
//!   `self_attn.v_absorb` (`cache-latent-attention-history`): not
//!   projections, so they keep the stored `[out, in]` orientation, which is
//!   what absorbed decode attention multiplies by head.
//! - Routed experts stored one by one (`mlp.experts.E.{gate,up,down}_proj`)
//!   are stacked in expert order into `moe.experts_{gate,up,down}_proj`, the
//!   same layout as the restacked `experts.gate_up_proj` above.
//!
//! The derived tensors keep `nn.Linear`'s `[out, in]` orientation, so
//! `weight_layout.rs` transposes them like every stored projection. No
//! other weight is fabricated, and no factor is folded into a weight that
//! a quantized checkpoint could store.

use crate::weights::WeightsPayloadSource;
use magnetar_runtime::model::{ModelArchitectureConfig, ModelDType, ModelTensorMetadata};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::collections::BTreeMap;

/// `embed_tokens_per_layer`, as discovered.
pub(crate) const PER_LAYER_EMBEDDING_TABLE: &str = "packed.embed_tokens_per_layer";
/// `per_layer_model_projection`, as discovered.
pub(crate) const PER_LAYER_PROJECTION_TABLE: &str = "packed.per_layer_model_projection";
/// A layer's `experts.gate_up_proj`, as discovered (after `layers.N.`).
pub(crate) const EXPERTS_GATE_UP: &str = "packed.experts_gate_up";
/// A layer's `router.proj`, as discovered (after `layers.N.`).
pub(crate) const ROUTER_PROJECTION: &str = "packed.router_proj";
/// A layer's `router.scale`, as discovered (after `layers.N.`).
pub(crate) const ROUTER_SCALE: &str = "packed.router_scale";
/// A latent-attention layer's `q_b_proj`, as discovered (after `layers.N.`).
pub(crate) const QUERY_UP_PROJECTION: &str = "packed.q_b_proj";
/// A latent-attention layer's `kv_a_proj_with_mqa`, as discovered.
pub(crate) const LATENT_DOWN_PROJECTION: &str = "packed.kv_a_proj_with_mqa";
/// A latent-attention layer's `kv_b_proj`, as discovered.
pub(crate) const LATENT_UP_PROJECTION: &str = "packed.kv_b_proj";
/// One routed expert's projection, as discovered: `layers.N.` then this
/// prefix, the expert's index and `.{gate,up,down}_proj`.
pub(crate) const EXPERT_PREFIX: &str = "packed.experts.";
/// One shard of a layer's n-gram embedding table, as discovered: `layers.N.`
/// then this prefix and the shard's index.
pub(crate) const NGRAM_SHARD_PREFIX: &str = "packed.ple_table.";

/// How one derived tensor's bytes come from a stored one.
#[derive(Debug)]
enum Derivation {
    /// `width` bytes at `start` of each of `rows` rows of `row_bytes`.
    Columns {
        source: String,
        rows: u64,
        row_bytes: u64,
        start: u64,
        width: u64,
    },
    /// These `(offset, length)` byte ranges, concatenated.
    Ranges {
        source: String,
        ranges: Vec<(u64, u64)>,
    },
    /// These whole tensors, concatenated.
    Stack { sources: Vec<String> },
    /// The `[experts, hidden]` router projection with `scale[h] * factor`
    /// folded into its column `h`, as `f32`.
    Router {
        projection: String,
        projection_dtype: ModelDType,
        scale: String,
        scale_dtype: ModelDType,
        hidden: u64,
        factor: f32,
    },
}

/// Answers each derived tensor's identity from the stored tensors, and
/// every other identity from `inner`.
#[derive(Debug)]
pub struct LayeredTensorSource {
    inner: WeightsPayloadSource,
    derived: BTreeMap<String, (Derivation, u64)>,
}

fn malformed(reason: String) -> ProductionIngestionError {
    ProductionIngestionError::MalformedMetadata { reason }
}

fn element_bytes(tensor: &ModelTensorMetadata) -> Result<u64, ProductionIngestionError> {
    match tensor.storage_dtype {
        ModelDType::F32 => Ok(4),
        ModelDType::F16 | ModelDType::Bf16 => Ok(2),
        other => Err(ProductionIngestionError::UnsupportedFormat {
            reason: format!(
                "tensor '{}' is stored as {other:?}; slicing or restacking it needs f32, f16 \
                 or bf16",
                tensor.name
            ),
        }),
    }
}

fn decode(bytes: &[u8], dtype: ModelDType) -> Vec<f32> {
    match dtype {
        ModelDType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        ModelDType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
            .collect(),
        _ => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
            .collect(),
    }
}

/// IEEE 754 binary16 to binary32, subnormals included.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Removes and returns the tensor named `name`.
fn take(tensors: &mut Vec<ModelTensorMetadata>, name: &str) -> Option<ModelTensorMetadata> {
    let index = tensors.iter().position(|tensor| tensor.name == name)?;
    Some(tensors.remove(index))
}

/// The layer indices of the tensors named `layers.N.{suffix}`.
fn layers_with(tensors: &[ModelTensorMetadata], suffix: &str) -> Vec<String> {
    tensors
        .iter()
        .filter_map(|tensor| {
            let (layer, rest) = tensor.name.strip_prefix("layers.")?.split_once('.')?;
            (rest == suffix).then(|| layer.to_string())
        })
        .collect()
}

/// A derived tensor's metadata: `source`'s, renamed and reshaped, with no
/// declared digest (its bytes are not the file's).
fn derived_tensor(
    source: &ModelTensorMetadata,
    name: String,
    shape: Vec<u64>,
    storage_dtype: ModelDType,
    size_bytes: u64,
) -> ModelTensorMetadata {
    let mut tensor = source.clone();
    tensor.name = name;
    tensor.shape = shape;
    tensor.storage_dtype = storage_dtype;
    tensor.size_bytes = Some(size_bytes);
    tensor.digest = None;
    tensor.layout = None;
    tensor
}

/// Replaces the packed tensors in `tensors` with the tensors derived from
/// them, and wraps `inner` to read the derived ones. A bundle without
/// packed tensors is returned unchanged.
pub fn derive(
    tensors: &mut Vec<ModelTensorMetadata>,
    inner: WeightsPayloadSource,
    config: &ModelArchitectureConfig,
) -> Result<LayeredTensorSource, ProductionIngestionError> {
    let mut derived = BTreeMap::new();
    let mut added = Vec::new();
    let layers = u64::from(config.num_hidden_layers);

    let tables = [
        take(tensors, PER_LAYER_EMBEDDING_TABLE),
        take(tensors, PER_LAYER_PROJECTION_TABLE),
    ];
    if tables.iter().any(Option::is_some) {
        let width = u64::from(config.per_layer_input_size.ok_or_else(|| {
            malformed(
                "the bundle packs per-layer input tables, but config.json declares no \
                 hidden_size_per_layer_input"
                    .into(),
            )
        })?);
        let [embedding, projection] = tables;
        if let Some(table) = embedding {
            let [vocab, packed] = table.shape[..] else {
                return Err(malformed(format!(
                    "'{}' is not a matrix: {:?}",
                    table.name, table.shape
                )));
            };
            if packed != layers * width {
                return Err(malformed(format!(
                    "'{}' has {packed} columns, not {layers} layers of {width}",
                    table.name
                )));
            }
            let element = element_bytes(&table)?;
            for layer in 0..layers {
                let name = format!("layers.{layer}.per_layer_embedding");
                let size = vocab * width * element;
                derived.insert(
                    name.clone(),
                    (
                        Derivation::Columns {
                            source: table.name.clone(),
                            rows: vocab,
                            row_bytes: packed * element,
                            start: layer * width * element,
                            width: width * element,
                        },
                        size,
                    ),
                );
                added.push(derived_tensor(
                    &table,
                    name,
                    vec![vocab, width],
                    table.storage_dtype,
                    size,
                ));
            }
        }
        if let Some(table) = projection {
            let [packed, hidden] = table.shape[..] else {
                return Err(malformed(format!(
                    "'{}' is not a matrix: {:?}",
                    table.name, table.shape
                )));
            };
            if packed != layers * width {
                return Err(malformed(format!(
                    "'{}' has {packed} rows, not {layers} layers of {width}",
                    table.name
                )));
            }
            let element = element_bytes(&table)?;
            let size = width * hidden * element;
            for layer in 0..layers {
                let name = format!("layers.{layer}.per_layer_model_projection");
                derived.insert(
                    name.clone(),
                    (
                        Derivation::Ranges {
                            source: table.name.clone(),
                            ranges: vec![(layer * size, size)],
                        },
                        size,
                    ),
                );
                added.push(derived_tensor(
                    &table,
                    name,
                    vec![width, hidden],
                    table.storage_dtype,
                    size,
                ));
            }
        }
    }

    for layer in layers_with(tensors, EXPERTS_GATE_UP) {
        let stored = take(tensors, &format!("layers.{layer}.{EXPERTS_GATE_UP}"))
            .expect("listed by layers_with");
        let [experts, gate_up, hidden] = stored.shape[..] else {
            return Err(malformed(format!(
                "'{}' is not [experts, 2 * intermediate, hidden]: {:?}",
                stored.name, stored.shape
            )));
        };
        if !gate_up.is_multiple_of(2) {
            return Err(malformed(format!(
                "'{}' has an odd gate and up width {gate_up}",
                stored.name
            )));
        }
        let intermediate = gate_up / 2;
        if let Some(moe) = &config.mixture_of_experts
            && (experts != u64::from(moe.experts) || intermediate != moe.expert_intermediate_size)
        {
            return Err(malformed(format!(
                "'{}' holds {experts} experts of width {intermediate}, but config.json \
                 declares {} of width {}",
                stored.name, moe.experts, moe.expert_intermediate_size
            )));
        }
        let element = element_bytes(&stored)?;
        let half = intermediate * hidden * element;
        for (projection, first) in [("experts_gate_proj", 0), ("experts_up_proj", half)] {
            let name = format!("layers.{layer}.moe.{projection}");
            derived.insert(
                name.clone(),
                (
                    Derivation::Ranges {
                        source: stored.name.clone(),
                        ranges: (0..experts)
                            .map(|expert| (expert * 2 * half + first, half))
                            .collect(),
                    },
                    experts * half,
                ),
            );
            added.push(derived_tensor(
                &stored,
                name,
                vec![experts * intermediate, hidden],
                stored.storage_dtype,
                experts * half,
            ));
        }
    }

    // `experts.down_proj` is already stacked: the same bytes as a matrix.
    for tensor in tensors.iter_mut() {
        if tensor.name.ends_with(".moe.experts_down_proj")
            && let [experts, hidden, intermediate] = tensor.shape[..]
        {
            tensor.shape = vec![experts * hidden, intermediate];
        }
    }

    for layer in layers_with(tensors, ROUTER_PROJECTION) {
        let projection = take(tensors, &format!("layers.{layer}.{ROUTER_PROJECTION}"))
            .expect("listed by layers_with");
        let scale = take(tensors, &format!("layers.{layer}.{ROUTER_SCALE}")).ok_or_else(|| {
            malformed(format!(
                "layer {layer}'s router has a projection but no scale"
            ))
        })?;
        let [experts, hidden] = projection.shape[..] else {
            return Err(malformed(format!(
                "'{}' is not [experts, hidden]: {:?}",
                projection.name, projection.shape
            )));
        };
        if scale.shape != [hidden] {
            return Err(malformed(format!(
                "'{}' is {:?}, not [{hidden}]",
                scale.name, scale.shape
            )));
        }
        element_bytes(&projection)?;
        element_bytes(&scale)?;
        let name = format!("layers.{layer}.moe.router_proj");
        let size = experts * hidden * 4;
        derived.insert(
            name.clone(),
            (
                Derivation::Router {
                    projection: projection.name.clone(),
                    projection_dtype: projection.storage_dtype,
                    scale: scale.name.clone(),
                    scale_dtype: scale.storage_dtype,
                    hidden,
                    factor: (hidden as f32).powf(-0.5),
                },
                size,
            ),
        );
        added.push(derived_tensor(
            &projection,
            name,
            vec![experts, hidden],
            ModelDType::F32,
            size,
        ));
    }
    derive_latent_attention(tensors, config, &mut derived, &mut added)?;
    stack_experts(tensors, config, &mut derived, &mut added)?;
    stack_ngram_shards(tensors, &mut derived, &mut added)?;
    // GPTQ, AWQ and BitsAndBytes projections are still refused here.
    // FP8 projections can be derived after conversion to their original
    // row-major F32 orientation, before the normal projection transpose.
    for (name, (derivation, _)) in &derived {
        let sources: Vec<&String> = match derivation {
            Derivation::Columns { source, .. } | Derivation::Ranges { source, .. } => {
                vec![source]
            }
            Derivation::Stack { sources } => sources.iter().collect(),
            Derivation::Router {
                projection, scale, ..
            } => vec![projection, scale],
        };
        if let Some(source) = sources
            .into_iter()
            .find(|source| inner.dequantizes(source) && !inner.is_fp8_projection(source))
        {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!(
                    "'{name}' would be derived from '{source}', a quantized projection; \
                     splitting or stacking GPTQ, AWQ or BitsAndBytes projections is not \
                     supported"
                ),
            });
        }
    }
    if let Some(stray) = tensors
        .iter()
        .find(|tensor| tensor.name.contains("packed."))
    {
        return Err(malformed(format!(
            "tensor '{}' has no matching tensor to derive weights from",
            stray.name
        )));
    }
    tensors.extend(added);
    Ok(LayeredTensorSource { inner, derived })
}

/// `rows` of `stored`, by `(first row, row count)`, as the derived tensor
/// `layers.{layer}.{name}`: a `[rows, in]` matrix of `stored`'s bytes.
fn insert_rows(
    derived: &mut BTreeMap<String, (Derivation, u64)>,
    added: &mut Vec<ModelTensorMetadata>,
    stored: &ModelTensorMetadata,
    layer: &str,
    name: &str,
    rows: &[(u64, u64)],
) -> Result<(), ProductionIngestionError> {
    let columns = stored.shape[1];
    let row_bytes = columns * element_bytes(stored)?;
    let count: u64 = rows.iter().map(|(_, count)| count).sum();
    let name = format!("layers.{layer}.{name}");
    let size = count * row_bytes;
    derived.insert(
        name.clone(),
        (
            Derivation::Ranges {
                source: stored.name.clone(),
                ranges: rows
                    .iter()
                    .map(|(first, count)| (first * row_bytes, count * row_bytes))
                    .collect(),
            },
            size,
        ),
    );
    added.push(derived_tensor(
        stored,
        name,
        vec![count, columns],
        stored.storage_dtype,
        size,
    ));
    Ok(())
}

/// Each head's `first` rows then its `second` rows, for `heads` heads: the
/// `(first row, row count)` runs of either part.
fn per_head_rows(heads: u64, first: u64, second: u64) -> [Vec<(u64, u64)>; 2] {
    let stride = first + second;
    [
        (0..heads).map(|head| (head * stride, first)).collect(),
        (0..heads)
            .map(|head| (head * stride + first, second))
            .collect(),
    ]
}

/// Takes `layers.{layer}.{suffix}`, checked to be a `[rows, columns]`
/// matrix.
fn take_matrix(
    tensors: &mut Vec<ModelTensorMetadata>,
    layer: &str,
    suffix: &str,
    rows: u64,
    columns: u64,
) -> Result<ModelTensorMetadata, ProductionIngestionError> {
    let name = format!("layers.{layer}.{suffix}");
    let stored = take(tensors, &name)
        .ok_or_else(|| malformed(format!("latent-attention layer {layer} has no '{name}'")))?;
    if stored.shape != [rows, columns] {
        return Err(malformed(format!(
            "'{name}' is {:?}, not [{rows}, {columns}]",
            stored.shape
        )));
    }
    Ok(stored)
}

/// Splits each latent-attention layer's fused projections by rows (see the
/// module documentation).
fn derive_latent_attention(
    tensors: &mut Vec<ModelTensorMetadata>,
    config: &ModelArchitectureConfig,
    derived: &mut BTreeMap<String, (Derivation, u64)>,
    added: &mut Vec<ModelTensorMetadata>,
) -> Result<(), ProductionIngestionError> {
    let latent_layers = layers_with(tensors, LATENT_DOWN_PROJECTION);
    let Some(mla) = &config.latent_attention else {
        if let Some(layer) = latent_layers.first() {
            return Err(malformed(format!(
                "layer {layer} stores latent-attention projections, but config.json \
                 declares no kv_lora_rank"
            )));
        }
        return Ok(());
    };
    let heads = u64::from(config.num_attention_heads);
    let (nope, rope, value) = (mla.qk_nope_head_dim, mla.qk_rope_head_dim, mla.v_head_dim);
    let latent = u64::from(mla.kv_lora_rank);
    let hidden = config.hidden_size;
    for layer in 0..u64::from(config.num_hidden_layers) {
        let layer = layer.to_string();
        let (query_name, query_input) = match mla.q_lora_rank {
            Some(rank) => (QUERY_UP_PROJECTION, u64::from(rank)),
            None => ("self_attn.q_proj", hidden),
        };
        let query = take_matrix(
            tensors,
            &layer,
            query_name,
            heads * (nope + rope),
            query_input,
        )?;
        let [query_nope, query_rope] = per_head_rows(heads, nope, rope);
        insert_rows(
            derived,
            added,
            &query,
            &layer,
            "self_attn.q_nope_proj",
            &query_nope,
        )?;
        insert_rows(
            derived,
            added,
            &query,
            &layer,
            "self_attn.q_rope_proj",
            &query_rope,
        )?;

        let down = take_matrix(
            tensors,
            &layer,
            LATENT_DOWN_PROJECTION,
            latent + rope,
            hidden,
        )?;
        insert_rows(
            derived,
            added,
            &down,
            &layer,
            "self_attn.kv_a_proj",
            &[(0, latent)],
        )?;
        insert_rows(
            derived,
            added,
            &down,
            &layer,
            "self_attn.k_rope_proj",
            &[(latent, rope)],
        )?;

        let up = take_matrix(
            tensors,
            &layer,
            LATENT_UP_PROJECTION,
            heads * (nope + value),
            latent,
        )?;
        let [key_nope, values] = per_head_rows(heads, nope, value);
        insert_rows(
            derived,
            added,
            &up,
            &layer,
            "self_attn.k_nope_proj",
            &key_nope,
        )?;
        insert_rows(derived, added, &up, &layer, "self_attn.v_proj", &values)?;
        // The absorbed decode's tables: the same rows, kept `[out, in]`.
        insert_rows(derived, added, &up, &layer, "self_attn.k_absorb", &key_nope)?;
        insert_rows(derived, added, &up, &layer, "self_attn.v_absorb", &values)?;
    }
    Ok(())
}

/// Stacks each layer's routed experts, stored one by one, in expert order
/// (see the module documentation).
/// Stacks each layer's n-gram embedding table shards (`layers.N.
/// packed.ple_table.K`, in shard order) into `layers.N.ple.table`: the
/// shards hold consecutive rows of one `[rows, head_dim]` table.
fn stack_ngram_shards(
    tensors: &mut Vec<ModelTensorMetadata>,
    derived: &mut BTreeMap<String, (Derivation, u64)>,
    added: &mut Vec<ModelTensorMetadata>,
) -> Result<(), ProductionIngestionError> {
    let mut stored: BTreeMap<String, BTreeMap<u64, ModelTensorMetadata>> = BTreeMap::new();
    let mut index = 0;
    while index < tensors.len() {
        let parsed = tensors[index]
            .name
            .strip_prefix("layers.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(layer, rest)| Some((layer, rest.strip_prefix(NGRAM_SHARD_PREFIX)?)))
            .and_then(|(layer, shard)| Some((layer.to_string(), shard.parse::<u64>().ok()?)));
        let Some((layer, shard)) = parsed else {
            index += 1;
            continue;
        };
        let tensor = tensors.remove(index);
        stored.entry(layer).or_default().insert(shard, tensor);
    }
    for (layer, shards) in stored {
        if shards.keys().copied().ne(0..shards.len() as u64) {
            return Err(malformed(format!(
                "layer {layer} stores n-gram table shards {:?}, not shards 0 to {}",
                shards.keys().collect::<Vec<_>>(),
                shards.len() - 1
            )));
        }
        let first = &shards[&0];
        let [_, width] = first.shape[..] else {
            return Err(malformed(format!(
                "'{}' is not a matrix: {:?}",
                first.name, first.shape
            )));
        };
        let mut rows = 0;
        for tensor in shards.values() {
            if tensor.shape.len() != 2
                || tensor.shape[1] != width
                || tensor.storage_dtype != first.storage_dtype
            {
                return Err(malformed(format!(
                    "'{}' is {:?} {:?}, not {:?} rows of {width} like the layer's other shards",
                    tensor.name, tensor.storage_dtype, tensor.shape, first.storage_dtype
                )));
            }
            rows += tensor.shape[0];
        }
        let size = rows * width * element_bytes(first)?;
        let name = format!("layers.{layer}.ple.table");
        derived.insert(
            name.clone(),
            (
                Derivation::Stack {
                    sources: shards.values().map(|tensor| tensor.name.clone()).collect(),
                },
                size,
            ),
        );
        added.push(derived_tensor(
            first,
            name,
            vec![rows, width],
            first.storage_dtype,
            size,
        ));
    }
    Ok(())
}

fn stack_experts(
    tensors: &mut Vec<ModelTensorMetadata>,
    config: &ModelArchitectureConfig,
    derived: &mut BTreeMap<String, (Derivation, u64)>,
    added: &mut Vec<ModelTensorMetadata>,
) -> Result<(), ProductionIngestionError> {
    // (layer, projection) -> expert -> stored tensor.
    let mut stored: BTreeMap<(String, String), BTreeMap<u64, ModelTensorMetadata>> =
        BTreeMap::new();
    let mut index = 0;
    while index < tensors.len() {
        let parsed = tensors[index]
            .name
            .strip_prefix("layers.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(layer, rest)| Some((layer, rest.strip_prefix(EXPERT_PREFIX)?)))
            .and_then(|(layer, rest)| {
                let (expert, projection) = rest.split_once('.')?;
                Some((
                    layer.to_string(),
                    expert.parse::<u64>().ok()?,
                    projection.to_string(),
                ))
            });
        let Some((layer, expert, projection)) = parsed else {
            index += 1;
            continue;
        };
        let tensor = tensors.remove(index);
        stored
            .entry((layer, projection))
            .or_default()
            .insert(expert, tensor);
    }
    if stored.is_empty() {
        return Ok(());
    }
    let moe = config.mixture_of_experts.as_ref().ok_or_else(|| {
        malformed(
            "the bundle stores routed experts, but config.json declares no mixture of experts"
                .into(),
        )
    })?;
    let experts = u64::from(moe.experts);
    let intermediate = moe.expert_intermediate_size;
    let hidden = config.hidden_size;
    for ((layer, projection), by_expert) in stored {
        let shape = match projection.as_str() {
            "gate_proj" | "up_proj" => [intermediate, hidden],
            _ => [hidden, intermediate],
        };
        if by_expert.keys().copied().ne(0..experts) {
            return Err(malformed(format!(
                "layer {layer} stores the {projection} of experts {:?}, not of experts 0 to {}",
                by_expert.keys().collect::<Vec<_>>(),
                experts - 1
            )));
        }
        let first = &by_expert[&0];
        for tensor in by_expert.values() {
            if tensor.shape != shape || tensor.storage_dtype != first.storage_dtype {
                return Err(malformed(format!(
                    "'{}' is {:?} {:?}, not {:?} {shape:?} like the layer's other experts",
                    tensor.name, tensor.storage_dtype, tensor.shape, first.storage_dtype
                )));
            }
        }
        let size = experts * shape[0] * shape[1] * element_bytes(first)?;
        let name = format!("layers.{layer}.moe.experts_{projection}");
        derived.insert(
            name.clone(),
            (
                Derivation::Stack {
                    sources: by_expert
                        .values()
                        .map(|tensor| tensor.name.clone())
                        .collect(),
                },
                size,
            ),
        );
        added.push(derived_tensor(
            first,
            name,
            vec![experts * shape[0], shape[1]],
            first.storage_dtype,
            size,
        ));
    }
    Ok(())
}

impl ProductionArtifactPayloadSource for LayeredTensorSource {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let Some((derivation, length)) = self.derived.get(&range.identity) else {
            return self.inner.read_payload(range);
        };
        if range.length != *length {
            return Err(ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            });
        }
        let plain = self.inner.plain();
        let capacity = usize::try_from(*length)
            .map_err(|_| malformed(format!("'{}' output is too large", range.identity)))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| malformed(format!("'{}' output cannot be allocated", range.identity)))?;
        let append_slice = |output: &mut Vec<u8>, source: &[u8], offset: u64, length: u64| {
            let start = usize::try_from(offset).ok();
            let end = offset
                .checked_add(length)
                .and_then(|end| usize::try_from(end).ok());
            let slice = start
                .zip(end)
                .and_then(|(start, end)| source.get(start..end))
                .ok_or_else(|| ProductionIngestionError::PayloadOutOfBounds {
                    identity: range.identity.clone(),
                })?;
            output.extend_from_slice(slice);
            Ok::<(), ProductionIngestionError>(())
        };
        match derivation {
            Derivation::Columns {
                source,
                rows,
                row_bytes,
                start,
                width,
            } => {
                if let Some(converted) = self.inner.read_fp8_projection(source)? {
                    for row in 0..*rows {
                        let offset = row
                            .checked_mul(*row_bytes)
                            .and_then(|value| value.checked_add(*start))
                            .ok_or_else(|| {
                                malformed(format!("'{}' slice offset overflows", range.identity))
                            })?;
                        append_slice(&mut bytes, &converted, offset, *width)?;
                    }
                } else {
                    bytes = plain.read_rows(source, *start, *row_bytes, *width, *rows)?;
                }
            }
            Derivation::Ranges { source, ranges } => {
                if let Some(converted) = self.inner.read_fp8_projection(source)? {
                    for (offset, length) in ranges {
                        append_slice(&mut bytes, &converted, *offset, *length)?;
                    }
                } else {
                    for (offset, length) in ranges {
                        bytes.extend(plain.read_within(source, *offset, *length)?);
                    }
                }
            }
            Derivation::Stack { sources } => {
                for source in sources {
                    if let Some(converted) = self.inner.read_fp8_projection(source)? {
                        bytes.extend(converted);
                    } else {
                        let total = plain.tensor_length(source)?;
                        bytes.extend(plain.read_within(source, 0, total)?);
                    }
                }
            }
            Derivation::Router {
                projection,
                projection_dtype,
                scale,
                scale_dtype,
                hidden,
                factor,
            } => {
                let read_all = |identity: &str| {
                    if let Some(converted) = self.inner.read_fp8_projection(identity)? {
                        Ok(converted)
                    } else {
                        let total = plain.tensor_length(identity)?;
                        plain.read_within(identity, 0, total)
                    }
                };
                let weights = decode(&read_all(projection)?, *projection_dtype);
                let scale = decode(&read_all(scale)?, *scale_dtype);
                for (index, weight) in weights.iter().enumerate() {
                    let folded = weight * scale[index % *hidden as usize] * factor;
                    bytes.extend_from_slice(&folded.to_le_bytes());
                }
            }
        }
        if bytes.len() as u64 != *length {
            return Err(malformed(format!(
                "'{}' derived {} bytes, not {length}",
                range.identity,
                bytes.len()
            )));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests;
