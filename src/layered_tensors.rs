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
        let mut bytes = Vec::with_capacity(usize::try_from(*length).unwrap_or(0));
        match derivation {
            Derivation::Columns {
                source,
                rows,
                row_bytes,
                start,
                width,
            } => {
                bytes = plain.read_rows(source, *start, *row_bytes, *width, *rows)?;
            }
            Derivation::Ranges { source, ranges } => {
                for (offset, length) in ranges {
                    bytes.extend(plain.read_within(source, *offset, *length)?);
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
                    let total = plain.tensor_length(identity)?;
                    plain.read_within(identity, 0, total)
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
