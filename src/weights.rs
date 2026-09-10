//! Single-file and Hugging Face-style sharded Safetensors weight discovery
//! (`implement-production-qwen-model-loading` task groups 2 and 4): reuses
//! the real `magnetar-format-safetensors` parser (Decision 1 -- no second
//! Safetensors parser in this crate), cross-checks a sharded index against
//! real parsed shard inventories (Decision 10), and exposes bounded,
//! on-demand payload access instead of holding every weight file's bytes
//! in memory at once.

use magnetar_runtime::model::{ModelDigest, ModelShard, ModelShardId, ModelTensorMetadata};
use magnetar_runtime::production_model_ingestion::{
    ProductionArtifactPayloadSource, ProductionIngestionError, ProductionModelSource,
    ProductionPayloadRange,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

const SINGLE_FILE_NAME: &str = "model.safetensors";
const INDEX_FILE_NAME: &str = "model.safetensors.index.json";

#[derive(Debug, Deserialize)]
struct RawShardIndex {
    weight_map: BTreeMap<String, String>,
}

/// Where one tensor's bytes physically live: which shard file, and the
/// byte range within that *file's own data section* (relative, matching
/// every format parser's own construction -- see `magnetar-runtime`'s
/// `host_tensors_from_artifact_bytes` doc comment).
#[derive(Debug)]
struct TensorLocation {
    shard_path: PathBuf,
    data_section_start: u64,
    offset: u64,
    length: u64,
}

/// Bounded, on-demand [`ProductionArtifactPayloadSource`] for Hugging
/// Face-style weight files: `read_payload` opens exactly the shard file a
/// tensor lives in, seeks to its byte range, and reads exactly that many
/// bytes -- it never holds a whole weight file's bytes in memory across
/// calls (`implement-production-qwen-model-loading` task "Production
/// Ingestion Exposes Bounded Payload Access").
#[derive(Debug)]
pub struct SafetensorsPayloadSource {
    locations: BTreeMap<String, TensorLocation>,
}

impl ProductionArtifactPayloadSource for SafetensorsPayloadSource {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let location = self.locations.get(&range.identity).ok_or_else(|| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            }
        })?;
        if range.offset != location.offset || range.length != location.length {
            return Err(ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            });
        }
        let mut file = fs::File::open(&location.shard_path).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        let start = location
            .data_section_start
            .checked_add(location.offset)
            .ok_or_else(|| ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            })?;
        file.seek(SeekFrom::Start(start)).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        let length = usize::try_from(location.length).map_err(|_| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: range.identity.clone(),
            }
        })?;
        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer).map_err(|error| {
            ProductionIngestionError::PayloadUnavailable {
                identity: format!("{}: {error}", range.identity),
            }
        })?;
        if let Some(expected_digest) = &range.digest {
            expected_digest.verify_bytes(&buffer).map_err(|error| {
                ProductionIngestionError::IntegrityMismatch {
                    identity: format!("{}: {error}", range.identity),
                }
            })?;
        }
        Ok(buffer)
    }
}

/// Reads the 8-byte little-endian header-length prefix every Safetensors
/// file begins with and returns the data section's start offset
/// (`8 + header_length`), without re-parsing the header itself (the
/// caller already has, or is about to get, the parsed tensor inventory
/// from `magnetar_format_safetensors::parse`).
fn data_section_start(bytes: &[u8]) -> Result<u64, ProductionIngestionError> {
    let header_len_bytes: [u8; 8] = bytes
        .get(..8)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
            reason: "safetensors file is too short for its header-length prefix".into(),
        })?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    header_len
        .checked_add(8)
        .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
            reason: "safetensors header length overflows".into(),
        })
}

fn parse_shard_file(
    path: &std::path::Path,
    shard_id: Option<&str>,
) -> Result<
    (
        Vec<u8>,
        magnetar_format_safetensors::SafetensorsArtifact,
        u64,
    ),
    ProductionIngestionError,
> {
    let bytes = fs::read(path).map_err(|error| ProductionIngestionError::RequiredPartMissing {
        part: format!("{} ({error})", path.display()),
    })?;
    let artifact = magnetar_format_safetensors::parse(&bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "{} failed to parse as Safetensors: {error}",
                shard_id.unwrap_or_else(|| path.to_str().unwrap_or("<weights>"))
            ),
        }
    })?;
    let start = data_section_start(&bytes)?;
    Ok((bytes, artifact, start))
}

/// Discovers and parses this bundle's weight files -- single-file
/// `model.safetensors` if present, else Hugging Face-style sharded
/// `model.safetensors.index.json` plus `model-*-of-*.safetensors` shards
/// -- returning the normalized tensor inventory, shard metadata (empty
/// for the single-file case), and a bounded payload source.
pub fn discover_and_parse_weights(
    source: &ProductionModelSource,
) -> Result<
    (
        Vec<ModelTensorMetadata>,
        Vec<ModelShard>,
        SafetensorsPayloadSource,
    ),
    ProductionIngestionError,
> {
    let (mut tensors, shards, mut payload_source) =
        if let Ok(index_path) = source.resolve(INDEX_FILE_NAME) {
            discover_sharded(source, &index_path)?
        } else if let Ok(single_path) = source.resolve(SINGLE_FILE_NAME) {
            discover_single_file(&single_path)?
        } else {
            return Err(ProductionIngestionError::RequiredPartMissing {
                part: format!("{SINGLE_FILE_NAME} or {INDEX_FILE_NAME}"),
            });
        };
    // Real Hugging Face tensor names (e.g. "model.layers.0.self_attn.
    // q_proj.weight") are normalized into the canonical Model Artifact
    // names the Qwen Component's weight-edge calls resolve directly --
    // both the returned inventory and the payload source's own lookup
    // keys are renamed together so `read_payload` stays reachable by the
    // canonical identity Model Loading will actually request.
    let mut renamed_locations = BTreeMap::new();
    for tensor in &mut tensors {
        let canonical = crate::naming::normalize_tensor_name(&tensor.name)?;
        if let Some(location) = payload_source.locations.remove(&tensor.name) {
            renamed_locations.insert(canonical.clone(), location);
        }
        tensor.name = canonical;
    }
    payload_source.locations = renamed_locations;
    Ok((tensors, shards, payload_source))
}

fn discover_single_file(
    path: &std::path::Path,
) -> Result<
    (
        Vec<ModelTensorMetadata>,
        Vec<ModelShard>,
        SafetensorsPayloadSource,
    ),
    ProductionIngestionError,
> {
    let (_bytes, artifact, start) = parse_shard_file(path, None)?;
    let mut locations = BTreeMap::new();
    let mut tensors = Vec::with_capacity(artifact.tensors.len());
    for tensor in artifact.tensors {
        let (offset, length) = match (tensor.offset_bytes, tensor.size_bytes) {
            (Some(offset), Some(length)) => (offset, length),
            _ => {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!("tensor '{}' has no declared byte range", tensor.name),
                });
            }
        };
        locations.insert(
            tensor.name.clone(),
            TensorLocation {
                shard_path: path.to_path_buf(),
                data_section_start: start,
                offset,
                length,
            },
        );
        // No content digest: neither the Safetensors header nor a
        // single-file bundle's own metadata declares a per-tensor digest
        // -- inventing one here would let it silently "verify" against
        // itself, proving nothing (`ModelTensorMetadata::digest`'s own
        // "None means unknown" precedent).
        tensors.push(tensor);
    }
    Ok((tensors, Vec::new(), SafetensorsPayloadSource { locations }))
}

fn discover_sharded(
    source: &ProductionModelSource,
    index_path: &std::path::Path,
) -> Result<
    (
        Vec<ModelTensorMetadata>,
        Vec<ModelShard>,
        SafetensorsPayloadSource,
    ),
    ProductionIngestionError,
> {
    let index_bytes =
        fs::read(index_path).map_err(|error| ProductionIngestionError::RequiredPartMissing {
            part: format!("{} ({error})", index_path.display()),
        })?;
    let index: RawShardIndex = serde_json::from_slice(&index_bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("{INDEX_FILE_NAME} is not valid JSON: {error}"),
        }
    })?;

    let mut shard_filenames: Vec<&String> = index.weight_map.values().collect();
    shard_filenames.sort();
    shard_filenames.dedup();

    let mut shards = Vec::with_capacity(shard_filenames.len());
    let mut shard_inventories: BTreeMap<
        String,
        (
            magnetar_format_safetensors::SafetensorsArtifact,
            u64,
            PathBuf,
        ),
    > = BTreeMap::new();
    for (order, shard_filename) in shard_filenames.iter().enumerate() {
        let shard_path = source.resolve(shard_filename).map_err(|_| {
            ProductionIngestionError::RequiredPartMissing {
                part: (*shard_filename).clone(),
            }
        })?;
        let (bytes, artifact, start) = parse_shard_file(&shard_path, Some(shard_filename))?;
        let digest = ModelDigest::sha256(&bytes);
        shards.push(ModelShard {
            id: ModelShardId::new((*shard_filename).clone()).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("shard filename '{shard_filename}' is invalid: {error}"),
                }
            })?,
            digest,
            size_bytes: bytes.len() as u64,
            order: order as u32,
        });
        shard_inventories.insert((*shard_filename).clone(), (artifact, start, shard_path));
    }

    // Every shard's *real* parsed inventory (not merely the index's own
    // mapping) must not declare the same tensor name twice across shards
    // -- reuses the existing generic duplicate-tensor detection rather
    // than a parallel check (Decision 1's "compose, don't reimplement").
    let all_real_tensors: Vec<ModelTensorMetadata> = shard_inventories
        .values()
        .flat_map(|(artifact, _, _)| artifact.tensors.clone())
        .collect();
    magnetar_runtime::model_format_roadmap::detect_duplicate_tensor_names(&all_real_tensors)
        .map_err(|error| ProductionIngestionError::MalformedMetadata {
            reason: error.to_string(),
        })?;

    let mut locations = BTreeMap::new();
    let mut tensors = Vec::with_capacity(index.weight_map.len());
    let mut seen_tensor_names = std::collections::BTreeSet::new();
    for (tensor_name, shard_filename) in &index.weight_map {
        if !seen_tensor_names.insert(tensor_name.clone()) {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!("tensor '{tensor_name}' is mapped more than once in the index"),
            });
        }
        let (artifact, start, shard_path) =
            shard_inventories.get(shard_filename).ok_or_else(|| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!(
                        "tensor '{tensor_name}' maps to shard '{shard_filename}', which was not \
                     discovered"
                    ),
                }
            })?;
        let real_entry = artifact
            .tensors
            .iter()
            .find(|entry| entry.name == *tensor_name)
            .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "the index maps tensor '{tensor_name}' to shard '{shard_filename}', but \
                     that shard's real Safetensors inventory does not contain it"
                ),
            })?;
        let (offset, length) = match (real_entry.offset_bytes, real_entry.size_bytes) {
            (Some(offset), Some(length)) => (offset, length),
            _ => {
                return Err(ProductionIngestionError::MalformedMetadata {
                    reason: format!("tensor '{tensor_name}' has no declared byte range"),
                });
            }
        };
        locations.insert(
            tensor_name.clone(),
            TensorLocation {
                shard_path: shard_path.clone(),
                data_section_start: *start,
                offset,
                length,
            },
        );
        let mut tensor = real_entry.clone();
        tensor.shard = Some(ModelShardId::new(shard_filename.clone()).map_err(|error| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("shard filename '{shard_filename}' is invalid: {error}"),
            }
        })?);
        tensors.push(tensor);
    }

    Ok((tensors, shards, SafetensorsPayloadSource { locations }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_runtime::ModelArtifactSource;
    use std::io::Write;

    fn write_safetensors_file(path: &std::path::Path, tensors: &[(&str, &[f32])]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, values) in tensors {
            let start = data.len() as u64;
            for value in *values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            let end = data.len() as u64;
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [values.len()],
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&data).unwrap();
    }

    #[test]
    fn discovers_and_reads_a_single_file_bundle() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join(SINGLE_FILE_NAME),
            &[
                ("model.embed_tokens.weight", &[1.0, 2.0]),
                ("model.norm.weight", &[3.0]),
            ],
        );
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, shards, payload_source) = discover_and_parse_weights(&source).unwrap();
        assert_eq!(tensors.len(), 2);
        assert!(shards.is_empty());

        let tensor_a = tensors
            .iter()
            .find(|t| t.name == "token_embedding")
            .unwrap();
        let range = ProductionPayloadRange {
            identity: tensor_a.name.clone(),
            offset: tensor_a.offset_bytes.unwrap(),
            length: tensor_a.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn discovers_and_cross_checks_a_sharded_bundle() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[("model.embed_tokens.weight", &[1.0])],
        );
        write_safetensors_file(
            &dir.path().join("model-00002-of-00002.safetensors"),
            &[("model.norm.weight", &[2.0, 3.0])],
        );
        let index = serde_json::json!({
            "metadata": {"total_size": 12},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors",
            }
        });
        fs::write(
            dir.path().join(INDEX_FILE_NAME),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, shards, payload_source) = discover_and_parse_weights(&source).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(shards.len(), 2);

        let tensor_b = tensors.iter().find(|t| t.name == "final_norm").unwrap();
        assert_eq!(
            tensor_b.shard.as_ref().unwrap().as_str(),
            "model-00002-of-00002.safetensors"
        );
        let range = ProductionPayloadRange {
            identity: tensor_b.name.clone(),
            offset: tensor_b.offset_bytes.unwrap(),
            length: tensor_b.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = payload_source.read_payload(&range).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values, vec![2.0, 3.0]);
    }

    #[test]
    fn rejects_a_tensor_duplicated_across_two_real_shards() {
        let dir = tempfile::tempdir().unwrap();
        // Both shards genuinely declare "model.embed_tokens.weight" in
        // their own real Safetensors header -- a corrupt/inconsistent
        // shard pair, not merely an index mapping mistake.
        write_safetensors_file(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[("model.embed_tokens.weight", &[1.0])],
        );
        // Shard 2 also physically declares "model.embed_tokens.weight" in
        // its own real header, even though the index below only ever
        // points at it for "model.norm.weight" -- both shards still get
        // discovered (the index references each of them for some
        // tensor), so the cross-shard scan over their *real* inventories
        // is what must catch this.
        write_safetensors_file(
            &dir.path().join("model-00002-of-00002.safetensors"),
            &[
                ("model.embed_tokens.weight", &[2.0]),
                ("model.norm.weight", &[3.0]),
            ],
        );
        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors",
            }
        });
        fs::write(
            dir.path().join(INDEX_FILE_NAME),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = match discover_and_parse_weights(&source) {
            Err(error) => error,
            Ok(_) => panic!("expected a duplicate-tensor-across-shards error"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn rejects_a_tensor_digest_mismatch_at_payload_read_time() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join(SINGLE_FILE_NAME),
            &[("model.embed_tokens.weight", &[1.0])],
        );
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, _shards, payload_source) = discover_and_parse_weights(&source).unwrap();
        let tensor_a = tensors
            .iter()
            .find(|t| t.name == "token_embedding")
            .unwrap();
        let range = ProductionPayloadRange {
            identity: tensor_a.name.clone(),
            offset: tensor_a.offset_bytes.unwrap(),
            length: tensor_a.size_bytes.unwrap(),
            digest: Some(ModelDigest::sha256(b"not the real tensor bytes")),
        };
        let error = match payload_source.read_payload(&range) {
            Err(error) => error,
            Ok(_) => panic!("expected an integrity mismatch error"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::IntegrityMismatch { .. }
        ));
    }

    #[test]
    fn rejects_index_referencing_a_missing_shard() {
        let dir = tempfile::tempdir().unwrap();
        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {"model.embed_tokens.weight": "does-not-exist.safetensors"}
        });
        fs::write(
            dir.path().join(INDEX_FILE_NAME),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = discover_and_parse_weights(&source).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::RequiredPartMissing { .. }
        ));
    }

    #[test]
    fn rejects_no_weight_files_present() {
        let dir = tempfile::tempdir().unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = discover_and_parse_weights(&source).unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::RequiredPartMissing { .. }
        ));
    }

    #[test]
    fn normalizes_per_layer_tensor_names_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join(SINGLE_FILE_NAME),
            &[
                ("model.embed_tokens.weight", &[1.0]),
                ("model.layers.0.self_attn.q_proj.weight", &[2.0]),
                ("model.layers.0.mlp.down_proj.weight", &[3.0]),
                ("model.norm.weight", &[4.0]),
                ("lm_head.weight", &[5.0]),
            ],
        );
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, _shards, payload_source) = discover_and_parse_weights(&source).unwrap();
        let names: std::collections::BTreeSet<String> =
            tensors.iter().map(|t| t.name.clone()).collect();
        assert_eq!(
            names,
            std::collections::BTreeSet::from([
                "token_embedding".to_string(),
                "layers.0.self_attn.q_proj".to_string(),
                "layers.0.mlp.down_proj".to_string(),
                "final_norm".to_string(),
                "lm_head".to_string(),
            ])
        );
        // The payload source's own lookup keys were renamed in lockstep --
        // a canonical identity actually reads real bytes, not just the
        // returned tensor inventory's names.
        let q_proj = tensors
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj")
            .unwrap();
        let range = ProductionPayloadRange {
            identity: q_proj.name.clone(),
            offset: q_proj.offset_bytes.unwrap(),
            length: q_proj.size_bytes.unwrap(),
            digest: None,
        };
        let bytes = payload_source.read_payload(&range).unwrap();
        assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), 2.0);
    }

    #[test]
    fn rejects_a_bias_tensor() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join(SINGLE_FILE_NAME),
            &[
                ("model.embed_tokens.weight", &[1.0]),
                ("model.layers.0.self_attn.q_proj.bias", &[0.0]),
            ],
        );
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let error = match discover_and_parse_weights(&source) {
            Err(error) => error,
            Ok(_) => panic!("expected a bias-tensor rejection"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::UnsupportedFormat { .. }
        ));
    }
}
