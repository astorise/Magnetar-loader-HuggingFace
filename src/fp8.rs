//! Reference conversion for block-scaled FP8 E4M3FN weights.
//!
//! The ingestor validates FP8 checkpoint configuration and associates each
//! weight with its scale tensor before converting it on demand. Unsupported
//! FP8 formats and incomplete or inconsistent metadata are rejected.

use magnetar_runtime::model::{ModelDType, ModelTensorMetadata};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionPayloadRange,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

/// Version 1 of the supported checkpoint convention: E4M3FN weights,
/// row-major block scales and dynamic activation quantization metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointBlockSpecV1 {
    pub block_rows: u32,
    pub block_columns: u32,
}

pub(crate) fn checkpoint_spec(
    value: Option<&serde_json::Value>,
) -> Result<
    Option<CheckpointBlockSpecV1>,
    magnetar_runtime::production_model_ingestion::ProductionIngestionError,
> {
    use magnetar_runtime::production_model_ingestion::ProductionIngestionError;
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let Some(method) = value.get("quant_method").and_then(|method| method.as_str()) else {
        return Ok(None);
    };
    if method != "fp8" {
        return Ok(None);
    }
    if value.get("fmt").and_then(|fmt| fmt.as_str()) != Some("e4m3") {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "FP8 checkpoint must declare the supported e4m3 format".into(),
        });
    }
    if value
        .get("activation_scheme")
        .and_then(|scheme| scheme.as_str())
        != Some("dynamic")
    {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "FP8 checkpoint must declare dynamic activation scaling".into(),
        });
    }
    let dimensions = value
        .get("weight_block_size")
        .and_then(|size| size.as_array())
        .filter(|size| size.len() == 2)
        .and_then(|size| Some((size[0].as_u64()?, size[1].as_u64()?)))
        .and_then(|(rows, columns)| Some((u32::try_from(rows).ok()?, u32::try_from(columns).ok()?)))
        .filter(|(rows, columns)| *rows > 0 && *columns > 0)
        .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
            reason: "FP8 weight_block_size must contain two positive u32 dimensions".into(),
        })?;
    Ok(Some(CheckpointBlockSpecV1 {
        block_rows: dimensions.0,
        block_columns: dimensions.1,
    }))
}

/// Row-major weight and scale geometry. Each scale multiplies the weight
/// values in one `block_rows` by `block_columns` tile, including edge tiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockLayout {
    pub rows: usize,
    pub columns: usize,
    pub block_rows: usize,
    pub block_columns: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8ConversionError {
    EmptyDimension,
    SizeOverflow,
    WeightLength,
    ScaleLength,
    NonFiniteScale,
    AllocationFailed,
}

impl fmt::Display for Fp8ConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyDimension => "weight and block dimensions must be nonzero",
            Self::SizeOverflow => "weight or scale dimensions overflow",
            Self::WeightLength => "FP8 weight length does not match its shape",
            Self::ScaleLength => "scale count does not match the block geometry",
            Self::NonFiniteScale => "FP8 scales must be finite",
            Self::AllocationFailed => "cannot allocate the dequantized weight",
        })
    }
}

impl Error for Fp8ConversionError {}

/// Decode an E4M3FN byte. This is the finite-only IEEE-style variant used by
/// PyTorch's `float8_e4m3fn`: exponent 15 remains finite except for mantissa
/// 7, which represents NaN. The signed zeros are retained.
pub fn decode_e4m3fn(byte: u8) -> f32 {
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    let value = match (exponent, mantissa) {
        (0, _) => f32::from(mantissa) * (1.0 / 512.0),
        (15, 7) => f32::NAN,
        _ => (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7),
    };
    if byte & 0x80 != 0 { -value } else { value }
}

/// Convert a row-major E4M3FN matrix into `f32`, multiplying each value by
/// its tile's `f32` dequantization scale. `weight_scale_inv` in the pinned
/// DeepSeek/Kimi convention is multiplied, despite its name.
pub fn dequantize_blocks(
    weight: &[u8],
    scales: &[f32],
    layout: BlockLayout,
) -> Result<Vec<f32>, Fp8ConversionError> {
    let BlockLayout {
        rows,
        columns,
        block_rows,
        block_columns,
    } = layout;
    if rows == 0 || columns == 0 || block_rows == 0 || block_columns == 0 {
        return Err(Fp8ConversionError::EmptyDimension);
    }
    let elements = rows
        .checked_mul(columns)
        .ok_or(Fp8ConversionError::SizeOverflow)?;
    if weight.len() != elements {
        return Err(Fp8ConversionError::WeightLength);
    }
    let scale_rows = rows.div_ceil(block_rows);
    let scale_columns = columns.div_ceil(block_columns);
    let scale_count = scale_rows
        .checked_mul(scale_columns)
        .ok_or(Fp8ConversionError::SizeOverflow)?;
    if scales.len() != scale_count {
        return Err(Fp8ConversionError::ScaleLength);
    }
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err(Fp8ConversionError::NonFiniteScale);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| Fp8ConversionError::AllocationFailed)?;
    for (index, byte) in weight.iter().copied().enumerate() {
        let row = index / columns;
        let column = index % columns;
        let scale_index = (row / block_rows) * scale_columns + column / block_columns;
        output.push(decode_e4m3fn(byte) * scales[scale_index]);
    }
    Ok(output)
}

#[derive(Debug)]
pub(crate) struct Projection {
    canonical_name: String,
    weight: ProductionPayloadRange,
    scale: ProductionPayloadRange,
    output_offset: u64,
    output_length: u64,
    layout: BlockLayout,
}

fn range(tensor: &ModelTensorMetadata) -> Result<ProductionPayloadRange, ProductionIngestionError> {
    Ok(ProductionPayloadRange {
        identity: tensor.name.clone(),
        offset: tensor
            .offset_bytes
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' has no offset", tensor.name),
            })?,
        length: tensor
            .size_bytes
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' has no byte length", tensor.name),
            })?,
        digest: tensor.digest.clone(),
    })
}

/// Associate every FP8 tensor with the exact raw scale tensor before names
/// are normalized. The raw source may locate those tensors in different shards.
pub(crate) fn extract_projections(
    tensors: &mut Vec<ModelTensorMetadata>,
    spec: Option<CheckpointBlockSpecV1>,
) -> Result<Vec<Projection>, ProductionIngestionError> {
    let inventory: BTreeMap<String, ModelTensorMetadata> = tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.clone()))
        .collect();
    let fp8: Vec<_> = tensors
        .iter()
        .filter(|tensor| tensor.storage_dtype == ModelDType::F8E4M3Fn)
        .cloned()
        .collect();
    if fp8.is_empty() {
        if spec.is_some()
            || tensors
                .iter()
                .any(|tensor| tensor.name.ends_with(".weight_scale_inv"))
        {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: "FP8 configuration or scale tensors have no E4M3 weights".into(),
            });
        }
        return Ok(Vec::new());
    }
    let spec = spec.ok_or_else(|| ProductionIngestionError::UnsupportedFormat {
        reason: "E4M3 weights require an explicit supported FP8 quantization configuration".into(),
    })?;
    if spec.block_rows == 0 || spec.block_columns == 0 {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: "FP8 block dimensions must be positive".into(),
        });
    }
    let mut used_scales = BTreeSet::new();
    let mut output_lengths = BTreeMap::new();
    let mut projections = Vec::with_capacity(fp8.len());
    for weight in fp8 {
        if !weight.name.ends_with(".weight") || weight.shape.len() != 2 {
            return Err(ProductionIngestionError::UnsupportedFormat {
                reason: format!(
                    "FP8 tensor '{}' must be a two-dimensional weight",
                    weight.name
                ),
            });
        }
        let rows = usize::try_from(weight.shape[0]).map_err(|_| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' has an oversized row count", weight.name),
            }
        })?;
        let columns = usize::try_from(weight.shape[1]).map_err(|_| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' has an oversized column count", weight.name),
            }
        })?;
        if rows == 0 || columns == 0 {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' has an empty dimension", weight.name),
            });
        }
        let elements = rows.checked_mul(columns).ok_or_else(|| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' dimensions overflow", weight.name),
            }
        })?;
        let output_length = elements
            .checked_mul(4)
            .and_then(|length| u64::try_from(length).ok())
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' output size overflows", weight.name),
            })?;
        let scale_name = format!("{}_scale_inv", weight.name);
        let scale = inventory.get(&scale_name).ok_or_else(|| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 tensor '{}' is missing '{scale_name}'", weight.name),
            }
        })?;
        let layout = BlockLayout {
            rows,
            columns,
            block_rows: spec.block_rows as usize,
            block_columns: spec.block_columns as usize,
        };
        let scale_shape = [
            rows.div_ceil(layout.block_rows) as u64,
            columns.div_ceil(layout.block_columns) as u64,
        ];
        if scale.storage_dtype != ModelDType::F32 || scale.shape.as_slice() != scale_shape {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 scale '{scale_name}' must be F32 with shape {scale_shape:?}"),
            });
        }
        let mut weight_range = range(&weight)?;
        if weight_range.length != elements as u64 {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "FP8 tensor '{}' byte length does not match its shape",
                    weight.name
                ),
            });
        }
        let scale_range = range(scale)?;
        let expected_scale_length = scale_shape[0]
            .checked_mul(scale_shape[1])
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 scale '{scale_name}' size overflows"),
            })?;
        if scale_range.length != expected_scale_length {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 scale '{scale_name}' byte length does not match its shape"),
            });
        }
        let canonical_name = crate::naming::normalize_tensor_name(&weight.name)?;
        weight_range.identity = canonical_name.clone();
        let output_offset = weight_range.offset;
        projections.push(Projection {
            canonical_name,
            weight: weight_range,
            scale: scale_range,
            output_offset,
            output_length,
            layout,
        });
        used_scales.insert(scale_name);
        output_lengths.insert(weight.name, output_length);
    }
    if let Some(unmatched) = tensors.iter().find(|tensor| {
        tensor.name.ends_with(".weight_scale_inv") && !used_scales.contains(&tensor.name)
    }) {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!("unmatched FP8 scale '{}'", unmatched.name),
        });
    }
    for entry in tensors.iter_mut() {
        if let Some(length) = output_lengths.get(&entry.name) {
            entry.storage_dtype = ModelDType::F32;
            entry.size_bytes = Some(*length);
            entry.digest = None;
            entry.quantization = None;
        }
    }
    tensors.retain(|tensor| !used_scales.contains(&tensor.name));
    Ok(projections)
}

#[derive(Debug)]
pub struct DequantizingPayloadSource<S> {
    inner: S,
    projections: BTreeMap<String, Projection>,
}

impl<S> DequantizingPayloadSource<S> {
    pub(crate) fn new(inner: S, projections: Vec<Projection>) -> Self {
        Self {
            inner,
            projections: projections
                .into_iter()
                .map(|projection| (projection.canonical_name.clone(), projection))
                .collect(),
        }
    }
    pub(crate) fn inner(&self) -> &S {
        &self.inner
    }
    pub(crate) fn dequantizes(&self, identity: &str) -> bool {
        self.projections.contains_key(identity)
    }
}

impl<S: ProductionArtifactPayloadSource> DequantizingPayloadSource<S> {
    pub(crate) fn read_projection(
        &self,
        identity: &str,
    ) -> Result<Option<Vec<u8>>, ProductionIngestionError> {
        let Some(projection) = self.projections.get(identity) else {
            return Ok(None);
        };
        self.read_payload(&ProductionPayloadRange {
            identity: identity.to_string(),
            offset: projection.output_offset,
            length: projection.output_length,
            digest: None,
        })
        .map(Some)
    }
}

impl<S: ProductionArtifactPayloadSource> ProductionArtifactPayloadSource
    for DequantizingPayloadSource<S>
{
    fn read_payload(
        &self,
        requested: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let Some(projection) = self.projections.get(&requested.identity) else {
            return self.inner.read_payload(requested);
        };
        if requested.offset != projection.output_offset
            || requested.length != projection.output_length
            || requested.digest.is_some()
        {
            return Err(ProductionIngestionError::PayloadOutOfBounds {
                identity: requested.identity.clone(),
            });
        }
        let weight = self.inner.read_payload(&projection.weight)?;
        let scale = self.inner.read_payload(&projection.scale)?;
        let (chunks, remainder) = scale.as_chunks::<4>();
        if !remainder.is_empty() {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: "FP8 scale bytes are not F32-aligned".into(),
            });
        }
        let mut scales = Vec::new();
        scales.try_reserve_exact(chunks.len()).map_err(|_| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "FP8 scales for '{}' cannot be allocated",
                    requested.identity
                ),
            }
        })?;
        scales.extend(chunks.iter().copied().map(f32::from_le_bytes));
        let values = dequantize_blocks(&weight, &scales, projection.layout).map_err(|error| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 conversion of '{}' failed: {error}", requested.identity),
            }
        })?;
        let output_length = usize::try_from(projection.output_length).map_err(|_| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 output '{}' is too large", requested.identity),
            }
        })?;
        let mut output = Vec::new();
        output.try_reserve_exact(output_length).map_err(|_| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("FP8 output '{}' cannot be allocated", requested.identity),
            }
        })?;
        for value in values {
            output.extend_from_slice(&value.to_le_bytes());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
