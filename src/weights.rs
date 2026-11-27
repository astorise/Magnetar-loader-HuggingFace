//! Single-file and Hugging Face-style sharded Safetensors weight discovery
//! (`implement-production-qwen-model-loading` task groups 2 and 4): reuses
//! the real `magnetar-format-safetensors` parser (Decision 1 -- no second
//! Safetensors parser in this crate), cross-checks a sharded index against
//! real parsed shard inventories (Decision 10), and exposes bounded,
//! on-demand payload access instead of holding every weight file's bytes
//! in memory at once.

use crate::mapping::{FileMapping, verify_digest};
use magnetar_runtime::model::{
    ModelDType, ModelDigest, ModelShard, ModelShardId, ModelTensorMetadata,
};
use magnetar_runtime::production_model_ingestion::{
    MappedPayload, ProductionArtifactPayloadSource, ProductionIngestionError,
    ProductionModelSource, ProductionPayloadRange,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::Arc,
};

const SINGLE_FILE_NAME: &str = "model.safetensors";
const INDEX_FILE_NAME: &str = "model.safetensors.index.json";
// Matches the Safetensors format's practical header limit and prevents an
// untrusted prefix from requesting an arbitrarily large allocation.
const MAX_HEADER_BYTES: u64 = 100_000_000;

#[derive(Debug, Deserialize)]
struct RawShardIndex {
    #[serde(deserialize_with = "deserialize_unique_weight_map")]
    weight_map: BTreeMap<String, String>,
}

fn deserialize_unique_weight_map<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct UniqueMap;
    impl<'de> serde::de::Visitor<'de> for UniqueMap {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a tensor-to-shard object with unique names")
        }
        fn visit_map<A>(self, mut source: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut result = BTreeMap::new();
            while let Some((name, shard)) = source.next_entry::<String, String>()? {
                if result.insert(name.clone(), shard).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate tensor '{name}' in weight_map"
                    )));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(UniqueMap)
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

    fn map_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Option<Arc<dyn MappedPayload>>, ProductionIngestionError> {
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
        let mapping = self.map_tensor(&range.identity)?;
        if let Some(expected) = &range.digest {
            verify_digest(&mapping, expected, &range.identity)?;
        }
        Ok(Some(Arc::new(mapping)))
    }
}

impl SafetensorsPayloadSource {
    /// Tensor `identity`'s stored bytes, mapped read-only.
    pub(crate) fn map_tensor(
        &self,
        identity: &str,
    ) -> Result<FileMapping, ProductionIngestionError> {
        let location = self.locations.get(identity).ok_or_else(|| {
            ProductionIngestionError::PayloadOutOfBounds {
                identity: identity.to_string(),
            }
        })?;
        let start = location
            .data_section_start
            .checked_add(location.offset)
            .ok_or_else(|| ProductionIngestionError::PayloadOutOfBounds {
                identity: identity.to_string(),
            })?;
        FileMapping::open(&location.shard_path, start, location.length, identity)
    }

    /// The stored byte length of tensor `identity`.
    pub(crate) fn tensor_length(&self, identity: &str) -> Result<u64, ProductionIngestionError> {
        self.locations
            .get(identity)
            .map(|location| location.length)
            .ok_or_else(|| ProductionIngestionError::PayloadOutOfBounds {
                identity: identity.to_string(),
            })
    }

    /// Reads `length` bytes at `offset + row * stride` within the stored
    /// tensor `identity` for each of `rows` rows, concatenated: a column
    /// slice, read without the other columns (`layered_tensors.rs`).
    pub(crate) fn read_rows(
        &self,
        identity: &str,
        offset: u64,
        stride: u64,
        length: u64,
        rows: u64,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let out_of_bounds = || ProductionIngestionError::PayloadOutOfBounds {
            identity: identity.to_string(),
        };
        let location = self.locations.get(identity).ok_or_else(out_of_bounds)?;
        let end = rows
            .checked_sub(1)
            .and_then(|last| last.checked_mul(stride))
            .and_then(|last| last.checked_add(offset))
            .and_then(|last| last.checked_add(length));
        if end.is_some_and(|end| end > location.length) || (rows > 0 && end.is_none()) {
            return Err(out_of_bounds());
        }
        let start = location
            .data_section_start
            .checked_add(location.offset)
            .and_then(|start| start.checked_add(offset))
            .ok_or_else(out_of_bounds)?;
        let unavailable = |error: std::io::Error| ProductionIngestionError::PayloadUnavailable {
            identity: format!("{identity}: {error}"),
        };
        let width = usize::try_from(length).map_err(|_| out_of_bounds())?;
        let total = rows
            .checked_mul(length)
            .and_then(|total| usize::try_from(total).ok())
            .ok_or_else(out_of_bounds)?;
        let mut buffer = vec![0u8; total];
        if width == 0 {
            return Ok(buffer);
        }
        let mut file = fs::File::open(&location.shard_path).map_err(unavailable)?;
        for (row, slice) in buffer.chunks_exact_mut(width).enumerate() {
            file.seek(SeekFrom::Start(start + row as u64 * stride))
                .map_err(unavailable)?;
            file.read_exact(slice).map_err(unavailable)?;
        }
        Ok(buffer)
    }

    /// Reads `length` bytes at `offset` within the stored tensor
    /// `identity`, for tensors derived from part of a stored one
    /// (`layered_tensors.rs`).
    pub(crate) fn read_within(
        &self,
        identity: &str,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        let out_of_bounds = || ProductionIngestionError::PayloadOutOfBounds {
            identity: identity.to_string(),
        };
        let location = self.locations.get(identity).ok_or_else(out_of_bounds)?;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > location.length)
        {
            return Err(out_of_bounds());
        }
        let start = location
            .data_section_start
            .checked_add(location.offset)
            .and_then(|start| start.checked_add(offset))
            .ok_or_else(out_of_bounds)?;
        let unavailable = |error: std::io::Error| ProductionIngestionError::PayloadUnavailable {
            identity: format!("{identity}: {error}"),
        };
        let mut file = fs::File::open(&location.shard_path).map_err(unavailable)?;
        file.seek(SeekFrom::Start(start)).map_err(unavailable)?;
        let mut buffer = vec![0u8; usize::try_from(length).map_err(|_| out_of_bounds())?];
        file.read_exact(&mut buffer).map_err(unavailable)?;
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

fn digest_from_hasher(hasher: Sha256) -> ModelDigest {
    let mut value = String::with_capacity(71);
    value.push_str("sha256:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in hasher.finalize() {
        value.push(HEX[(byte >> 4) as usize] as char);
        value.push(HEX[(byte & 15) as usize] as char);
    }
    ModelDigest {
        algorithm: "sha256".into(),
        value,
    }
}

/// Hashes a bounded section without retaining it in memory. The shard hash
/// includes every byte; the optional tensor hash binds F32/FP8 payloads.
fn hash_section(
    file: &mut fs::File,
    mut length: u64,
    shard: &mut Sha256,
    tensor: &mut Option<Sha256>,
    buffer: &mut [u8],
) -> std::io::Result<()> {
    while length != 0 {
        let count = length.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..count])?;
        shard.update(&buffer[..count]);
        if let Some(tensor) = tensor {
            tensor.update(&buffer[..count]);
        }
        length -= count as u64;
    }
    Ok(())
}

/// What [`parse_shard_file`] returns for a shard.
type ParsedShard = Result<
    (
        ModelDigest,
        magnetar_format_safetensors::SafetensorsArtifact,
        u64,
        u64,
    ),
    ProductionIngestionError,
>;

/// Parses and hashes every shard, each on one of at most as many threads as
/// the host has cores, and returns the results in shard order: the first
/// error a caller meets is the first shard's, whichever finished first.
///
/// Hashing a shard is one sequential pass over its bytes, so a single shard
/// gains nothing; a checkpoint of many shards reads and hashes them
/// side by side (measured with a release build on this development VM: one
/// core hashes about 1 GB/s, four cores about 2.2 GB/s from a cold cache).
fn parse_shards(names: &[&String], paths: &[PathBuf]) -> Vec<ParsedShard> {
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(paths.len());
    if workers <= 1 {
        return paths
            .iter()
            .zip(names)
            .map(|(path, name)| parse_shard_file(path, Some(name)))
            .collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<ParsedShard>>> =
        paths.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(path) = paths.get(index) else { break };
                    let parsed = parse_shard_file(path, Some(names[index]));
                    *slots[index]
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(parsed);
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .expect("every shard was parsed")
        })
        .collect()
}

fn parse_shard_file(
    path: &std::path::Path,
    shard_id: Option<&str>,
) -> Result<
    (
        ModelDigest,
        magnetar_format_safetensors::SafetensorsArtifact,
        u64,
        u64,
    ),
    ProductionIngestionError,
> {
    let unavailable = |error: std::io::Error| ProductionIngestionError::RequiredPartMissing {
        part: format!("{} ({error})", path.display()),
    };
    let mut file = fs::File::open(path).map_err(unavailable)?;
    let file_length = file.metadata().map_err(unavailable)?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).map_err(unavailable)?;
    let header_length = u64::from_le_bytes(prefix);
    let header_end = header_length
        .checked_add(8)
        .filter(|end| *end <= file_length)
        .filter(|_| header_length <= MAX_HEADER_BYTES)
        .ok_or_else(|| ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "{} has an invalid or oversized Safetensors header",
                path.display()
            ),
        })?;
    let header_size =
        usize::try_from(header_end).map_err(|_| ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "{} has a Safetensors header too large for this host",
                path.display()
            ),
        })?;
    let mut header = Vec::new();
    header.try_reserve_exact(header_size).map_err(|_| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("{} cannot allocate its Safetensors header", path.display()),
        }
    })?;
    header.extend_from_slice(&prefix);
    header.resize(header_size, 0);
    file.read_exact(&mut header[8..]).map_err(unavailable)?;
    let mut artifact =
        magnetar_format_safetensors::parse_header(&header, file_length).map_err(|error| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "{} failed to parse as Safetensors: {error}",
                    shard_id.unwrap_or_else(|| path.to_str().unwrap_or("<weights>"))
                ),
            }
        })?;
    let start = data_section_start(&header)?;

    // Preserve the same shard and per-tensor digests as parsing the entire
    // file, but keep memory bounded by the header plus one small buffer.
    let mut shard_hasher = Sha256::new();
    shard_hasher.update(&header);
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut order: Vec<usize> = (0..artifact.tensors.len()).collect();
    order.sort_by_key(|&index| {
        (
            artifact.tensors[index].offset_bytes,
            artifact.tensors[index].size_bytes,
        )
    });
    let mut cursor = 0_u64;
    for index in order {
        let tensor = &mut artifact.tensors[index];
        let offset = tensor.offset_bytes.expect("format parser supplies offsets");
        let length = tensor.size_bytes.expect("format parser supplies sizes");
        let gap = offset.checked_sub(cursor).ok_or_else(|| {
            ProductionIngestionError::MalformedMetadata {
                reason: "Safetensors tensor ranges overlap".into(),
            }
        })?;
        hash_section(&mut file, gap, &mut shard_hasher, &mut None, &mut buffer)
            .map_err(unavailable)?;
        let mut tensor_hasher =
            matches!(tensor.storage_dtype, ModelDType::F32 | ModelDType::F8E4M3Fn)
                .then(Sha256::new);
        hash_section(
            &mut file,
            length,
            &mut shard_hasher,
            &mut tensor_hasher,
            &mut buffer,
        )
        .map_err(unavailable)?;
        tensor.digest = tensor_hasher.map(digest_from_hasher);
        cursor = offset + length;
    }
    let tail = (file_length - start).checked_sub(cursor).ok_or_else(|| {
        ProductionIngestionError::MalformedMetadata {
            reason: "Safetensors tensor range exceeds its data section".into(),
        }
    })?;
    hash_section(&mut file, tail, &mut shard_hasher, &mut None, &mut buffer)
        .map_err(unavailable)?;
    Ok((
        digest_from_hasher(shard_hasher),
        artifact,
        start,
        file_length,
    ))
}

/// Either a plain Safetensors payload source, or one wrapping it to
/// dequantize GPTQ-quantized (`gptq.rs`), AWQ-quantized (`awq.rs`), or
/// BitsAndBytes-quantized (`bnb.rs`) or block-FP8 (`fp8.rs`) projections on demand -- the one
/// caller (`discover_and_parse_weights`) picks the variant, so every
/// other caller (`lib.rs::ingest`) just sees one
/// [`ProductionArtifactPayloadSource`] regardless of which real bundle
/// shape it read. A bundle declaring more than one quantization method is
/// not a real shape any actual checkpoint takes (one method per
/// checkpoint), so this stays a plain enum rather than supporting
/// simultaneous variants.
#[derive(Debug)]
pub enum WeightsPayloadSource {
    Plain(SafetensorsPayloadSource),
    Gptq(crate::gptq::GptqDequantizingPayloadSource<SafetensorsPayloadSource>),
    Awq(crate::awq::AwqDequantizingPayloadSource<SafetensorsPayloadSource>),
    Bnb(crate::bnb::BnbDequantizingPayloadSource<SafetensorsPayloadSource>),
    Fp8(crate::fp8::DequantizingPayloadSource<SafetensorsPayloadSource>),
}

impl WeightsPayloadSource {
    /// The source the stored tensors are read from, whatever dequantizes
    /// the quantized projections on top of it.
    pub(crate) fn plain(&self) -> &SafetensorsPayloadSource {
        match self {
            Self::Plain(source) => source,
            Self::Gptq(source) => source.inner(),
            Self::Awq(source) => source.inner(),
            Self::Bnb(source) => source.inner(),
            Self::Fp8(source) => source.inner(),
        }
    }

    /// Whether `identity` is a quantized projection, whose stored bytes are
    /// not its values: it is dequantized on read.
    pub(crate) fn dequantizes(&self, identity: &str) -> bool {
        match self {
            Self::Plain(_) => false,
            Self::Gptq(source) => source.dequantizes(identity),
            Self::Awq(source) => source.dequantizes(identity),
            Self::Bnb(source) => source.dequantizes(identity),
            Self::Fp8(source) => source.dequantizes(identity),
        }
    }

    /// FP8 conversion preserves the stored matrix orientation, unlike some
    /// other quantizers; its converted bytes can be safely sliced or stacked.
    pub(crate) fn is_fp8_projection(&self, identity: &str) -> bool {
        matches!(self, Self::Fp8(source) if source.dequantizes(identity))
    }

    pub(crate) fn read_fp8_projection(
        &self,
        identity: &str,
    ) -> Result<Option<Vec<u8>>, ProductionIngestionError> {
        match self {
            Self::Fp8(source) => source.read_projection(identity),
            _ => Ok(None),
        }
    }
}

impl ProductionArtifactPayloadSource for WeightsPayloadSource {
    fn read_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Vec<u8>, ProductionIngestionError> {
        match self {
            Self::Plain(source) => source.read_payload(range),
            Self::Gptq(source) => source.read_payload(range),
            Self::Awq(source) => source.read_payload(range),
            Self::Bnb(source) => source.read_payload(range),
            Self::Fp8(source) => source.read_payload(range),
        }
    }

    /// A tensor stored as it is read is mapped; a quantized projection is
    /// dequantized on read, so it has no bytes to map.
    fn map_payload(
        &self,
        range: &ProductionPayloadRange,
    ) -> Result<Option<Arc<dyn MappedPayload>>, ProductionIngestionError> {
        if self.dequantizes(&range.identity) {
            return Ok(None);
        }
        self.plain().map_payload(range)
    }
}

/// The tensor inventory, shard metadata (empty for the single-file case),
/// the set of canonical names already in the Runtime's expected
/// `[in_features, out_features]` orientation (GPTQ- or AWQ-dequantized
/// projections -- `weight_layout.rs`'s transpose must skip these), and a
/// bounded payload source -- [`discover_and_parse_weights`]'s own return
/// shape, named so its signature stays readable.
pub type DiscoveredWeights = (
    Vec<ModelTensorMetadata>,
    Vec<ModelShard>,
    std::collections::BTreeSet<String>,
    WeightsPayloadSource,
);

/// Parsed checkpoint-order tensor metadata and bounded access to its bytes.
/// This is not a trusted Model Artifact: callers must authenticate the
/// checkpoint and validate its model-specific tensor inventory before use.
/// The caller must keep the source files unchanged while the payloads are read;
/// measured file digests describe inspection, not a locked file snapshot.
#[derive(Debug)]
pub struct RawSafetensors {
    pub tensors: Vec<ModelTensorMetadata>,
    pub shards: Vec<ModelShard>,
    pub payloads: SafetensorsPayloadSource,
    /// Observed whole-file digest for a single file; compare it with an
    /// independently authorized digest. Indexed shards carry their own
    /// observed digests in `shards`. This value is not a trust decision.
    pub single_file_digest: Option<ModelDigest>,
}

/// Inspect a caller-selected Safetensors file within an authorized bundle.
/// Names, shapes and storage dtypes remain exactly as declared in the file;
/// no model-specific renaming, transposition or dequantization is performed.
pub fn inspect_safetensors_file(
    source: &ProductionModelSource,
    relative: &str,
) -> Result<RawSafetensors, ProductionIngestionError> {
    let path = source.resolve(relative)?;
    let (tensors, shards, payloads, digest) = discover_single_file(&path)?;
    Ok(RawSafetensors {
        tensors,
        shards,
        payloads,
        single_file_digest: Some(digest),
    })
}

/// Discover conventional single-file or indexed sharded weights without
/// changing tensor names or their physical checkpoint layout.
pub fn discover_raw_safetensors(
    source: &ProductionModelSource,
) -> Result<RawSafetensors, ProductionIngestionError> {
    let (tensors, shards, payloads, single_file_digest) = match source.resolve(INDEX_FILE_NAME) {
        Ok(index) => {
            let (tensors, shards, payloads) = discover_sharded(source, &index)?;
            (tensors, shards, payloads, None)
        }
        Err(ProductionIngestionError::RequiredPartMissing { .. }) => {
            match source.resolve(SINGLE_FILE_NAME) {
                Ok(path) => {
                    let (tensors, shards, payloads, digest) = discover_single_file(&path)?;
                    (tensors, shards, payloads, Some(digest))
                }
                Err(ProductionIngestionError::RequiredPartMissing { .. }) => {
                    return Err(ProductionIngestionError::RequiredPartMissing {
                        part: format!("{SINGLE_FILE_NAME} or {INDEX_FILE_NAME}"),
                    });
                }
                Err(error) => return Err(error),
            }
        }
        Err(error) => return Err(error),
    };
    Ok(RawSafetensors {
        tensors,
        shards,
        payloads,
        single_file_digest,
    })
}

/// Discovers and parses this bundle's weight files -- single-file
/// `model.safetensors` if present, else Hugging Face-style sharded
/// `model.safetensors.index.json` plus `model-*-of-*.safetensors` shards.
#[cfg(test)]
pub fn discover_and_parse_weights(
    source: &ProductionModelSource,
) -> Result<DiscoveredWeights, ProductionIngestionError> {
    discover_and_parse_weights_with_fp8(source, None)
}

#[cfg(test)]
pub fn discover_and_parse_weights_with_fp8(
    source: &ProductionModelSource,
    fp8_spec: Option<crate::fp8::CheckpointBlockSpecV1>,
) -> Result<DiscoveredWeights, ProductionIngestionError> {
    discover_and_parse_weights_for_model(source, fp8_spec, None)
}

pub(crate) fn discover_and_parse_weights_for_model(
    source: &ProductionModelSource,
    fp8_spec: Option<crate::fp8::CheckpointBlockSpecV1>,
    model: Option<(&str, u32)>,
) -> Result<DiscoveredWeights, ProductionIngestionError> {
    let RawSafetensors {
        mut tensors,
        shards,
        payloads: mut payload_source,
        ..
    } = discover_raw_safetensors(source)?;

    if let Some(("kimi_k2", layer_count)) = model {
        // The pinned Kimi implementation registers this derived frequency
        // buffer with `persistent=False`; it recomputes it from the RoPE
        // configuration. The published shards contain stale BF16 [56]
        // copies, which upstream does not use. Recognize exactly that
        // documented checkpoint form and fail closed on other shapes.
        let mut ignored = std::collections::BTreeSet::new();
        for tensor in &tensors {
            if let Some(layer) = tensor
                .name
                .strip_prefix("model.layers.")
                .and_then(|name| name.strip_suffix(".self_attn.rotary_emb.inv_freq"))
            {
                let valid_layer = layer.parse::<u32>().is_ok_and(|layer| layer < layer_count);
                if !valid_layer || tensor.storage_dtype != ModelDType::Bf16 || tensor.shape != [56]
                {
                    return Err(ProductionIngestionError::MalformedMetadata {
                        reason: format!(
                            "invalid Kimi nonpersistent rotary buffer '{}'",
                            tensor.name
                        ),
                    });
                }
                ignored.insert(tensor.name.clone());
            }
        }
        tensors.retain(|tensor| !ignored.contains(&tensor.name));
        for name in ignored {
            payload_source.locations.remove(&name);
        }
    }

    // GPTQ-quantized projections replace their raw `.qweight`/`.qzeros`/
    // `.scales`/`.g_idx` quadruple with one synthesized `<prefix>.weight`
    // placeholder here, *before* the renaming loop below runs (`naming.rs`
    // does not recognize those raw suffixes and would reject them). AWQ
    // detection runs second, over whatever `.qweight` tensors GPTQ left
    // untouched (no `.g_idx` sibling -- GPTQ's own disambiguating signal).
    let gptq_projections = crate::gptq::extract_gptq_projections(&mut tensors)?;
    let awq_projections = crate::awq::extract_awq_projections(&mut tensors)?;
    // BitsAndBytes detection is independent of GPTQ/AWQ (a real
    // `<prefix>.weight` tensor with a `<prefix>.weight.absmax` sibling --
    // a naming shape neither other scheme produces) but, uniquely, needs
    // to *read* a small real JSON blob (`<prefix>.weight.quant_state.
    // bitsandbytes__nf4`) to learn its real logical shape -- so it must
    // run here too, while `payload_source` still holds every tensor's
    // real raw-named location, before the renaming loop below.
    let bnb_projections = crate::bnb::extract_bnb_projections(&mut tensors, &payload_source)?;
    let fp8_projections = crate::fp8::extract_projections(&mut tensors, fp8_spec)?;
    if !fp8_projections.is_empty()
        && (!gptq_projections.is_empty()
            || !awq_projections.is_empty()
            || !bnb_projections.is_empty())
    {
        return Err(ProductionIngestionError::UnsupportedFormat {
            reason: "mixing FP8 with other quantization methods is not supported".into(),
        });
    }

    // Real Hugging Face tensor names (e.g. "model.layers.0.self_attn.
    // q_proj.weight") are normalized into the canonical Model Artifact
    // names the Qwen Component's weight-edge calls resolve directly --
    // both the returned inventory and the payload source's own lookup
    // keys are renamed together so `read_payload` stays reachable by the
    // canonical identity Model Loading will actually request. A GPTQ
    // placeholder has no location entry (its real bytes come from its
    // four raw siblings' own, untouched locations instead), so its
    // `.remove()` below is a harmless no-op.
    // A multimodal bundle's vision and audio towers are not part of the
    // text decoder.
    tensors.retain(|tensor| {
        let text = !crate::naming::is_non_text_tensor(&tensor.name);
        if !text {
            payload_source.locations.remove(&tensor.name);
        }
        text
    });
    let mut renamed_locations = BTreeMap::new();
    for tensor in &mut tensors {
        let canonical = crate::naming::normalize_tensor_name(&tensor.name)?;
        if let Some(location) = payload_source.locations.remove(&tensor.name) {
            renamed_locations.insert(canonical.clone(), location);
        }
        // A depthwise `nn.Conv1d` weight `[channels, 1, kernel]` is the
        // mixer's `[channels, kernel]`: the same bytes.
        if (canonical.ends_with(".linear_attn.conv1d") || canonical.ends_with(".ple.conv1d"))
            && let [channels, 1, kernel] = tensor.shape[..]
        {
            tensor.shape = vec![channels, kernel];
        }
        tensor.name = canonical;
    }
    // `payload_source.locations` at this point holds only whatever the
    // loop above never `.remove()`'d -- exactly the raw GPTQ/AWQ sibling
    // entries per quantized projection (never present in `tensors`, so
    // never targeted by `.remove()`) plus, for BnB, every sibling except
    // the packed weight tensor itself (which *is* still present in
    // `tensors`, relocated to its canonical key like any other tensor) --
    // still keyed by their real raw names.
    // `GptqDequantizingPayloadSource`/`AwqDequantizingPayloadSource`/
    // `BnbDequantizingPayloadSource` read them by those raw identities, so
    // they must survive into the final map alongside the newly
    // canonical-keyed ones -- `extend`, not an overwriting assignment,
    // which would silently drop them and make every quantized read fail
    // closed with `PayloadOutOfBounds` (a real bug caught by this crate's
    // own end-to-end GPTQ ingestion test, not merely a theoretical risk).
    payload_source.locations.extend(renamed_locations);

    // GPTQ/AWQ dequantize directly to the Runtime's expected
    // `[in_features, out_features]` orientation and so must be *excluded*
    // from `weight_layout.rs`'s transpose; BnB dequantizes to `nn.
    // Linear`'s own `[out_features, in_features]` storage convention (see
    // `bnb.rs`'s own doc comment) and so must still go through it like
    // any plain weight -- `bnb_projections`' own canonical names are
    // deliberately not added here.
    let quantized_canonical_names: std::collections::BTreeSet<String> = gptq_projections
        .iter()
        .map(crate::gptq::GptqProjection::canonical_name)
        .chain(
            awq_projections
                .iter()
                .map(crate::awq::AwqProjection::canonical_name),
        )
        .map(str::to_string)
        .collect();
    let payload_source = if !gptq_projections.is_empty() {
        WeightsPayloadSource::Gptq(crate::gptq::GptqDequantizingPayloadSource::new(
            payload_source,
            gptq_projections,
        ))
    } else if !awq_projections.is_empty() {
        WeightsPayloadSource::Awq(crate::awq::AwqDequantizingPayloadSource::new(
            payload_source,
            awq_projections,
        ))
    } else if !bnb_projections.is_empty() {
        WeightsPayloadSource::Bnb(crate::bnb::BnbDequantizingPayloadSource::new(
            payload_source,
            bnb_projections,
        ))
    } else if !fp8_projections.is_empty() {
        WeightsPayloadSource::Fp8(crate::fp8::DequantizingPayloadSource::new(
            payload_source,
            fp8_projections,
        ))
    } else {
        WeightsPayloadSource::Plain(payload_source)
    };
    Ok((tensors, shards, quantized_canonical_names, payload_source))
}

fn discover_single_file(
    path: &std::path::Path,
) -> Result<
    (
        Vec<ModelTensorMetadata>,
        Vec<ModelShard>,
        SafetensorsPayloadSource,
        ModelDigest,
    ),
    ProductionIngestionError,
> {
    let (digest, artifact, start, _) = parse_shard_file(path, None)?;
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
        // Any digest here was measured while reading this file, not
        // declared by Safetensors. It detects later byte changes but does
        // not authenticate the checkpoint.
        tensors.push(tensor);
    }
    Ok((
        tensors,
        Vec::new(),
        SafetensorsPayloadSource { locations },
        digest,
    ))
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
    let shard_paths = shard_filenames
        .iter()
        .map(|shard_filename| {
            source.resolve(shard_filename).map_err(|_| {
                ProductionIngestionError::RequiredPartMissing {
                    part: (*shard_filename).clone(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Reading and hashing a shard is the one full read of the weights a load
    // makes before it maps or stages anything: the shards go to as many
    // threads as the host has cores.
    let parsed = parse_shards(&shard_filenames, &shard_paths);
    for (order, ((shard_filename, shard_path), parsed)) in shard_filenames
        .iter()
        .zip(shard_paths)
        .zip(parsed)
        .enumerate()
    {
        let (digest, artifact, start, size_bytes) = parsed?;
        shards.push(ModelShard {
            id: ModelShardId::new((*shard_filename).clone()).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("shard filename '{shard_filename}' is invalid: {error}"),
                }
            })?,
            digest,
            size_bytes,
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
    if all_real_tensors.len() != index.weight_map.len() {
        return Err(ProductionIngestionError::MalformedMetadata {
            reason: format!(
                "the shard index lists {} tensors but the shard headers contain {}",
                index.weight_map.len(),
                all_real_tensors.len()
            ),
        });
    }

    let mut locations = BTreeMap::new();
    let mut tensors = Vec::with_capacity(index.weight_map.len());
    for (tensor_name, shard_filename) in &index.weight_map {
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
            .binary_search_by(|entry| entry.name.as_str().cmp(tensor_name.as_str()))
            .ok()
            .map(|index| &artifact.tensors[index])
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
    use std::io::{Seek, SeekFrom, Write};

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
    fn streamed_shard_parsing_preserves_full_and_per_tensor_digests() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("weights.safetensors");
        let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[2,6]},"b":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[8,11]},"c":{"dtype":"BF16","shape":[1],"data_offsets":[11,13]}}"#;
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[9, 8, 0, 0, 0x80, 0x3f, 7, 6, 0x38, 0x40, 0x48, 1, 2, 5, 4])
            .unwrap();
        drop(file);
        let bytes = fs::read(&path).unwrap();
        let expected = magnetar_format_safetensors::parse(&bytes).unwrap();
        let (digest, actual, start, length) = parse_shard_file(&path, None).unwrap();
        assert_eq!(digest, ModelDigest::sha256(&bytes));
        assert_eq!(actual, expected);
        assert_eq!(start, 8 + header.len() as u64);
        assert_eq!(length, bytes.len() as u64);
    }

    #[test]
    fn shard_index_rejects_duplicate_names_and_unlisted_header_tensors() {
        let duplicate = br#"{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}"#;
        assert!(serde_json::from_slice::<RawShardIndex>(duplicate).is_err());
        let directory = tempfile::tempdir().unwrap();
        let shard = directory.path().join("one.safetensors");
        write_safetensors_file(&shard, &[("a", &[1.0]), ("b", &[2.0])]);
        let index = directory.path().join(INDEX_FILE_NAME);
        fs::write(&index, br#"{"weight_map":{"a":"one.safetensors"}}"#).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(directory.path().to_path_buf()),
            directory.path().to_path_buf(),
        );
        assert!(matches!(
            discover_sharded(&source, &index),
            Err(ProductionIngestionError::MalformedMetadata { .. })
        ));
    }

    #[test]
    fn pinned_kimi_index_and_real_headers_agree_on_fp8_scales() {
        use flate2::read::GzDecoder;
        let mut index_bytes = Vec::new();
        GzDecoder::new(&include_bytes!("../testdata/kimi-k2/model.safetensors.index.json.gz")[..])
            .read_to_end(&mut index_bytes)
            .unwrap();
        assert_eq!(index_bytes.len(), 12_529_626);
        assert_eq!(
            ModelDigest::sha256(&index_bytes).value,
            "sha256:78957a08bb224e141dbf63feba25b8c6f2b9825a89434054e2856a20c45a1eb7"
        );
        let index: RawShardIndex = serde_json::from_slice(&index_bytes).unwrap();
        assert_eq!(index.weight_map.len(), 139_644);
        assert_eq!(
            index
                .weight_map
                .values()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            61
        );
        let headers: [(&str, &[u8], u64, &str, usize); 3] = [
            (
                "model-1-of-61.safetensors",
                include_bytes!("../testdata/kimi-k2/model-1-of-61.header"),
                2_846_451_040,
                "sha256:0b61acf328951e0f0efc25d9cd4b436d4db41e7c31e290b7ad26ddcd647fdecd",
                8,
            ),
            (
                "model-2-of-61.safetensors",
                include_bytes!("../testdata/kimi-k2/model-2-of-61.header"),
                17_066_593_104,
                "sha256:fbf0cc5f42beeab55a295845073300a1517de025b13ebe33c4902b37017a3c2f",
                1160,
            ),
            (
                "model-61-of-61.safetensors",
                include_bytes!("../testdata/kimi-k2/model-61-of-61.header"),
                19_415_420_696,
                "sha256:079dcd832e9e1de321e56a8ad3be5ace680dcb6f773d563832b3f0480237565b",
                1160,
            ),
        ];
        for (filename, header, file_length, digest, expected_fp8) in headers {
            assert_eq!(ModelDigest::sha256(header).value, digest);
            let artifact = magnetar_format_safetensors::parse_header(header, file_length).unwrap();
            let declared: std::collections::BTreeSet<_> = index
                .weight_map
                .iter()
                .filter(|(_, shard)| shard.as_str() == filename)
                .map(|(name, _)| name.as_str())
                .collect();
            let actual: std::collections::BTreeSet<_> = artifact
                .tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect();
            assert_eq!(declared, actual, "{filename}");
            let by_name: BTreeMap<_, _> = artifact
                .tensors
                .iter()
                .map(|tensor| (tensor.name.as_str(), tensor))
                .collect();
            let mut fp8_count = 0;
            for tensor in &artifact.tensors {
                if tensor.storage_dtype != ModelDType::F8E4M3Fn {
                    continue;
                }
                fp8_count += 1;
                assert_eq!(tensor.shape.len(), 2, "{}", tensor.name);
                let scale_name = format!("{}_scale_inv", tensor.name);
                let scale = by_name
                    .get(scale_name.as_str())
                    .expect("real FP8 weight has a scale");
                assert_eq!(scale.storage_dtype, ModelDType::F32, "{scale_name}");
                assert_eq!(
                    scale.shape,
                    tensor
                        .shape
                        .iter()
                        .map(|dimension| dimension.div_ceil(128))
                        .collect::<Vec<_>>(),
                    "{scale_name}"
                );
            }
            assert_eq!(fp8_count, expected_fp8);
            let mut inventory = artifact.tensors.clone();
            let projections = crate::fp8::extract_projections(
                &mut inventory,
                Some(crate::fp8::CheckpointBlockSpecV1 {
                    block_rows: 128,
                    block_columns: 128,
                }),
            )
            .expect("real Kimi metadata is accepted by FP8 projection extraction");
            assert_eq!(projections.len(), expected_fp8);
            assert!(
                inventory
                    .iter()
                    .all(|tensor| tensor.storage_dtype != ModelDType::F8E4M3Fn)
            );
        }
        let scale_names: Vec<_> = index
            .weight_map
            .keys()
            .filter(|name| name.ends_with("_scale_inv"))
            .collect();
        assert_eq!(scale_names.len(), 69_608);
        for scale in scale_names {
            assert!(
                index
                    .weight_map
                    .contains_key(scale.strip_suffix("_scale_inv").unwrap())
            );
        }
        let unknown: Vec<_> = index
            .weight_map
            .keys()
            .filter(|name| {
                !name.ends_with("_scale_inv") && !name.ends_with(".self_attn.rotary_emb.inv_freq")
            })
            .filter_map(|name| {
                crate::naming::normalize_tensor_name(name)
                    .err()
                    .map(|_| name.clone())
            })
            .take(10)
            .collect();
        assert!(
            unknown.is_empty(),
            "unrecognized real Kimi tensor names: {unknown:?}"
        );
    }

    #[test]
    fn pinned_olala_header_keeps_separate_geodesic_parameter_identities() {
        let header = include_bytes!("../testdata/olala-mini/model.header");
        assert_eq!(
            ModelDigest::sha256(header).value,
            "sha256:81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5"
        );
        let artifact = magnetar_format_safetensors::parse_header(header, 13_579_677_592).unwrap();
        let by_name: BTreeMap<_, _> = artifact
            .tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect();
        let mut canonical_names = std::collections::BTreeSet::new();
        for layer in 0..36 {
            for branch in ["mixer", "mlp"] {
                for parameter in ["scale", "bias", "prosres_scalar"] {
                    let raw = format!("model.layers.{layer}.geodesic_{branch}.{parameter}");
                    let tensor = by_name
                        .get(raw.as_str())
                        .expect("published geodesic scalar");
                    assert_eq!(tensor.storage_dtype, ModelDType::F32);
                    assert!(tensor.shape.is_empty());
                    let canonical = crate::naming::normalize_tensor_name(&raw).unwrap();
                    assert_eq!(
                        canonical,
                        format!("layers.{layer}.geodesic_{branch}.{parameter}")
                    );
                    assert!(canonical_names.insert(canonical));
                }
            }
        }
        assert_eq!(canonical_names.len(), 216);
        let scalars = include_bytes!("../testdata/olala-mini/layer0-geodesic.bin");
        assert_eq!(
            ModelDigest::sha256(scalars).value,
            "sha256:b14815a9aad6ed3d1a74a05e4a717eb8a40c9daa24c77078d7f3ec125706d75e"
        );
        let values: Vec<_> = scalars
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect();
        assert_eq!(
            values,
            [0.80078125, 1.0, 1.546875, 0.31252673, 1.0, 0.36721435]
        );
    }

    #[test]
    fn fp8_without_a_declared_configuration_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let header = br#"{"model.layers.0.mlp.up_proj.weight":{"dtype":"F8_E4M3","shape":[1,1],"data_offsets":[0,1]}}"#;
        let mut file = fs::File::create(dir.path().join(SINGLE_FILE_NAME)).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0x38]).unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        assert!(matches!(
            discover_and_parse_weights(&source),
            Err(ProductionIngestionError::UnsupportedFormat { .. })
        ));
    }

    fn write_raw_tensor(
        path: &std::path::Path,
        name: &str,
        dtype: &str,
        shape: &[u64],
        data: &[u8],
    ) {
        let header = serde_json::to_vec(&serde_json::json!({
            name: {"dtype": dtype, "shape": shape, "data_offsets": [0, data.len()]}
        }))
        .unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(data).unwrap();
    }

    #[test]
    fn kimi_nonpersistent_rotary_buffers_are_discarded_only_for_the_known_form() {
        let directory = tempfile::tempdir().unwrap();
        let weight = "model.layers.0.mlp.up_proj.weight";
        let buffer = "model.layers.0.self_attn.rotary_emb.inv_freq";
        write_raw_tensor(
            &directory.path().join("weight.safetensors"),
            weight,
            "F32",
            &[1, 1],
            &1.0_f32.to_le_bytes(),
        );
        write_raw_tensor(
            &directory.path().join("buffer.safetensors"),
            buffer,
            "BF16",
            &[56],
            &[0; 112],
        );
        fs::write(
            directory.path().join(INDEX_FILE_NAME),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {(weight): "weight.safetensors", (buffer): "buffer.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(directory.path().to_path_buf()),
            directory.path().to_path_buf(),
        );
        assert!(discover_and_parse_weights_with_fp8(&source, None).is_err());
        let (tensors, _, _, _) =
            discover_and_parse_weights_for_model(&source, None, Some(("kimi_k2", 1))).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "layers.0.mlp.up_proj");
        assert!(discover_and_parse_weights_for_model(&source, None, Some(("kimi_k2", 0))).is_err());
    }

    #[test]
    fn fp8_weight_reads_its_scale_from_a_different_shard() {
        let dir = tempfile::tempdir().unwrap();
        let weight_name = "model.layers.0.mlp.up_proj.weight";
        let scale_name = "model.layers.0.mlp.up_proj.weight_scale_inv";
        write_raw_tensor(
            &dir.path().join("weight.safetensors"),
            weight_name,
            "F8_E4M3",
            &[2, 3],
            &[0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c],
        );
        let scale_bytes: Vec<u8> = [0.5_f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        write_raw_tensor(
            &dir.path().join("scale.safetensors"),
            scale_name,
            "F32",
            &[1, 2],
            &scale_bytes,
        );
        fs::write(
            dir.path().join(INDEX_FILE_NAME),
            serde_json::to_vec(&serde_json::json!({"weight_map": {
                weight_name: "weight.safetensors", scale_name: "scale.safetensors"
            }}))
            .unwrap(),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let spec = Some(crate::fp8::CheckpointBlockSpecV1 {
            block_rows: 2,
            block_columns: 2,
        });
        let (tensors, shards, _, payload) =
            discover_and_parse_weights_with_fp8(&source, spec).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(tensors.len(), 1);
        let tensor = &tensors[0];
        assert_eq!(tensor.name, "layers.0.mlp.up_proj");
        assert_eq!(
            tensor.storage_dtype,
            magnetar_runtime::model::ModelDType::F32
        );
        let bytes = payload
            .read_payload(&ProductionPayloadRange {
                identity: tensor.name.clone(),
                offset: tensor.offset_bytes.unwrap(),
                length: tensor.size_bytes.unwrap(),
                digest: None,
            })
            .unwrap();
        let (chunks, remainder) = bytes.as_chunks::<4>();
        assert!(remainder.is_empty());
        let actual: Vec<f32> = chunks.iter().copied().map(f32::from_le_bytes).collect();
        assert_eq!(actual, vec![0.5, 1.0, 6.0, 2.0, 2.5, 12.0]);
        let mut stored = fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join("weight.safetensors"))
            .unwrap();
        stored.seek(SeekFrom::End(-1)).unwrap();
        stored.write_all(&[0x00]).unwrap();
        let error = payload
            .read_payload(&ProductionPayloadRange {
                identity: tensor.name.clone(),
                offset: tensor.offset_bytes.unwrap(),
                length: tensor.size_bytes.unwrap(),
                digest: None,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            ProductionIngestionError::IntegrityMismatch { .. }
        ));
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
        let (tensors, shards, _gptq_names, payload_source) =
            discover_and_parse_weights(&source).unwrap();
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

    /// A multimodal bundle's vision and audio towers stay out of the
    /// inventory; its text decoder takes the canonical names.
    #[test]
    fn a_multimodal_bundle_keeps_only_its_text_decoder() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join(SINGLE_FILE_NAME),
            &[
                ("model.language_model.embed_tokens.weight", &[1.0, 2.0]),
                ("model.language_model.norm.weight", &[3.0]),
                ("model.vision_tower.patch_embedder.weight", &[4.0]),
                ("model.embed_audio.embedding_projection.weight", &[5.0]),
            ],
        );
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, _shards, _names, payload_source) =
            discover_and_parse_weights(&source).unwrap();
        let mut names: Vec<&str> = tensors.iter().map(|tensor| tensor.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, ["final_norm", "token_embedding"]);
        let norm = tensors.iter().find(|t| t.name == "final_norm").unwrap();
        let bytes = payload_source
            .read_payload(&ProductionPayloadRange {
                identity: norm.name.clone(),
                offset: norm.offset_bytes.unwrap(),
                length: norm.size_bytes.unwrap(),
                digest: None,
            })
            .unwrap();
        assert_eq!(bytes, 3.0f32.to_le_bytes());
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
        let (tensors, shards, _gptq_names, payload_source) =
            discover_and_parse_weights(&source).unwrap();
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

    /// Many shards are read and hashed side by side and come out in shard
    /// order, each with the digest of its own bytes.
    #[test]
    fn many_shards_are_parsed_in_parallel_and_keep_their_order() {
        let dir = tempfile::tempdir().unwrap();
        let count = 13;
        let mut weight_map = serde_json::Map::new();
        let mut names = Vec::new();
        for shard in 0..count {
            let file = format!("model-{shard:05}-of-{count:05}.safetensors");
            let tensor = format!("model.layers.{shard}.input_layernorm.weight");
            let values: Vec<f32> = (0..64 + shard).map(|v| (v * (shard + 1)) as f32).collect();
            write_safetensors_file(&dir.path().join(&file), &[(tensor.as_str(), &values)]);
            weight_map.insert(tensor, serde_json::Value::String(file.clone()));
            names.push(file);
        }
        fs::write(
            dir.path().join(INDEX_FILE_NAME),
            serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": weight_map}))
                .unwrap(),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, shards, _, _) = discover_and_parse_weights(&source).unwrap();
        assert_eq!(tensors.len(), count);
        assert_eq!(shards.len(), count);
        for (order, shard) in shards.iter().enumerate() {
            assert_eq!(shard.order as usize, order);
            assert_eq!(shard.id.as_str(), names[order]);
            let bytes = fs::read(dir.path().join(&names[order])).unwrap();
            assert_eq!(
                shard.digest,
                ModelDigest::sha256(&bytes),
                "{}",
                names[order]
            );
            assert_eq!(shard.size_bytes, bytes.len() as u64);
        }
    }

    /// Whichever of several bad shards a thread meets first, the error is
    /// the first shard's in shard order.
    #[test]
    fn the_error_of_several_bad_shards_is_the_first_in_shard_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut weight_map = serde_json::Map::new();
        for shard in 0..8 {
            let file = format!("model-{shard:05}-of-00008.safetensors");
            let tensor = format!("model.layers.{shard}.input_layernorm.weight");
            if shard == 2 || shard == 5 {
                // Not a safetensors file: a header length past the file.
                fs::write(dir.path().join(&file), [0xff_u8; 16]).unwrap();
            } else {
                write_safetensors_file(&dir.path().join(&file), &[(tensor.as_str(), &[1.0])]);
            }
            weight_map.insert(tensor, serde_json::Value::String(file));
        }
        fs::write(
            dir.path().join(INDEX_FILE_NAME),
            serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": weight_map}))
                .unwrap(),
        )
        .unwrap();
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        for _ in 0..20 {
            let error = discover_and_parse_weights(&source).unwrap_err();
            let ProductionIngestionError::MalformedMetadata { reason } = error else {
                panic!("a malformed header is malformed metadata");
            };
            assert!(
                reason.contains("model-00002-of-00008.safetensors"),
                "the first bad shard in order is reported, not another: {reason}"
            );
        }
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
        let (tensors, _shards, _gptq_names, payload_source) =
            discover_and_parse_weights(&source).unwrap();
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
        let (tensors, _shards, _gptq_names, payload_source) =
            discover_and_parse_weights(&source).unwrap();
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
    fn rejects_a_bias_tensor_outside_qkv() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join(SINGLE_FILE_NAME),
            &[
                ("model.embed_tokens.weight", &[1.0]),
                ("model.layers.0.self_attn.o_proj.bias", &[0.0]),
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

    /// Real Qwen2/2.5 checkpoints declare `self_attn.{q,k,v}_proj.bias`
    /// (task 10.5's sibling QKV bias support) -- these must be accepted
    /// and canonicalized, not rejected like every other bias tensor.
    #[test]
    fn accepts_qkv_bias_tensors() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors_file(
            &dir.path().join(SINGLE_FILE_NAME),
            &[
                ("model.embed_tokens.weight", &[1.0]),
                ("model.layers.0.self_attn.q_proj.bias", &[2.0]),
                ("model.layers.0.self_attn.k_proj.bias", &[3.0]),
                ("model.layers.0.self_attn.v_proj.bias", &[4.0]),
            ],
        );
        let source = ProductionModelSource::authorized_local_bundle(
            ModelArtifactSource::LocalPath(dir.path().to_path_buf()),
            dir.path().to_path_buf(),
        );
        let (tensors, _shards, _gptq_names, _payload_source) =
            discover_and_parse_weights(&source).unwrap();
        let names: std::collections::BTreeSet<String> =
            tensors.iter().map(|t| t.name.clone()).collect();
        assert!(names.contains("layers.0.self_attn.q_bias"));
        assert!(names.contains("layers.0.self_attn.k_bias"));
        assert!(names.contains("layers.0.self_attn.v_bias"));
    }
}
