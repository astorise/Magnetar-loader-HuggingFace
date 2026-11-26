//! Read-only mappings of weight ranges (`support-qwen38-flash-next` task 4.4,
//! design D8).
//!
//! A table too large to hold, such as Qwen4-Exp's n-gram embedding table,
//! is never read whole: the Runtime asks this source for a
//! [`MappedPayload`] and copies out the few rows a step names. The mapping
//! belongs to this crate, which already owns the weight files: the Runtime
//! never sees a path, a handle or a pointer, and has no `unsafe`.
//!
//! This module holds the crate's only `unsafe` code: the memory map itself.

use magnetar_runtime::model::ModelDigest;
use magnetar_runtime::production_model_ingestion::{MappedPayload, ProductionIngestionError};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::{fs::File, path::Path, sync::Arc};

/// Bytes hashed at a time when a mapping's digest is verified: the table is
/// read once, in blocks, never held.
const DIGEST_CHUNK: usize = 1 << 20;

fn out_of_bounds(identity: &str) -> ProductionIngestionError {
    ProductionIngestionError::PayloadOutOfBounds {
        identity: identity.to_string(),
    }
}

/// A byte range of one weight file, mapped read-only.
#[derive(Debug)]
pub(crate) struct FileMapping {
    map: Mmap,
    start: usize,
    len: usize,
}

impl FileMapping {
    /// Maps `length` bytes at `start` of the file at `path`.
    #[allow(unsafe_code)]
    pub(crate) fn open(
        path: &Path,
        start: u64,
        length: u64,
        identity: &str,
    ) -> Result<Self, ProductionIngestionError> {
        let unavailable = |error: std::io::Error| ProductionIngestionError::PayloadUnavailable {
            identity: format!("{identity}: {error}"),
        };
        let file = File::open(path).map_err(unavailable)?;
        // SAFETY: the file is opened read-only and the mapping is never
        // written through. The one way to break a read-only mapping is for
        // another process to truncate the file while it is mapped (a read
        // past the new end is a bus error), which is the same race any
        // reader of a weight file has with a writer replacing it under it;
        // the weight files of an authorized bundle are not modified while a
        // model is loaded.
        let map = unsafe { Mmap::map(&file) }.map_err(unavailable)?;
        let range = usize::try_from(start)
            .ok()
            .zip(usize::try_from(length).ok())
            .filter(|(start, length)| {
                start
                    .checked_add(*length)
                    .is_some_and(|end| end <= map.len())
            });
        let Some((start, len)) = range else {
            return Err(out_of_bounds(identity));
        };
        Ok(Self { map, start, len })
    }
}

impl MappedPayload for FileMapping {
    fn len(&self) -> u64 {
        self.len as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), ProductionIngestionError> {
        let span = usize::try_from(offset)
            .ok()
            .and_then(|offset| Some((offset, offset.checked_add(out.len())?)))
            .filter(|&(_, end)| end <= self.len)
            .ok_or_else(|| out_of_bounds("mapped payload"))?;
        out.copy_from_slice(&self.map[self.start + span.0..self.start + span.1]);
        Ok(())
    }

    fn prefetch(&self, spans: &[(u64, u64)]) {
        // A hint: it never blocks, and a failure only means no read-ahead.
        #[cfg(unix)]
        for &(offset, length) in spans {
            let span = usize::try_from(offset)
                .ok()
                .zip(usize::try_from(length).ok())
                .filter(|(offset, length)| {
                    offset
                        .checked_add(*length)
                        .is_some_and(|end| end <= self.len)
                });
            if let Some((offset, length)) = span {
                let _ =
                    self.map
                        .advise_range(memmap2::Advice::WillNeed, self.start + offset, length);
            }
        }
        #[cfg(not(unix))]
        let _ = spans;
    }
}

/// Several mappings read as one, one after the other: the shards of a table
/// stacked in order.
#[derive(Debug)]
pub(crate) struct ConcatMapping {
    /// Each part with the offset it starts at.
    parts: Vec<(u64, Arc<dyn MappedPayload>)>,
    len: u64,
}

impl ConcatMapping {
    pub(crate) fn new(parts: Vec<Arc<dyn MappedPayload>>) -> Self {
        let mut len = 0;
        let parts = parts
            .into_iter()
            .filter(|part| !part.is_empty())
            .map(|part| {
                let start = len;
                len += part.len();
                (start, part)
            })
            .collect();
        Self { parts, len }
    }
}

impl MappedPayload for ConcatMapping {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), ProductionIngestionError> {
        let end = offset
            .checked_add(out.len() as u64)
            .filter(|&end| end <= self.len)
            .ok_or_else(|| out_of_bounds("mapped payload"))?;
        let mut position = offset;
        let mut done = 0;
        while position < end {
            let index = self
                .parts
                .partition_point(|(start, _)| *start <= position)
                .saturating_sub(1);
            let (start, part) = &self.parts[index];
            let within = position - start;
            let take = (part.len() - within).min(end - position) as usize;
            part.read_at(within, &mut out[done..done + take])?;
            position += take as u64;
            done += take;
        }
        Ok(())
    }

    fn prefetch(&self, spans: &[(u64, u64)]) {
        for (start, part) in &self.parts {
            let part_end = start + part.len();
            let clipped: Vec<(u64, u64)> = spans
                .iter()
                .filter_map(|&(offset, length)| {
                    let from = offset.max(*start);
                    let to = offset.saturating_add(length).min(part_end);
                    (from < to).then(|| (from - start, to - from))
                })
                .collect();
            if !clipped.is_empty() {
                part.prefetch(&clipped);
            }
        }
    }
}

/// Checks `mapping` against the digest its range declares, hashing it in
/// blocks of [`DIGEST_CHUNK`] bytes: the same check
/// [`ModelDigest::verify_bytes`] makes of bytes held whole.
pub(crate) fn verify_digest(
    mapping: &dyn MappedPayload,
    expected: &ModelDigest,
    identity: &str,
) -> Result<(), ProductionIngestionError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; DIGEST_CHUNK.min(mapping.len() as usize)];
    let mut offset = 0u64;
    while offset < mapping.len() {
        let take = (mapping.len() - offset).min(buffer.len() as u64) as usize;
        mapping.read_at(offset, &mut buffer[..take])?;
        hasher.update(&buffer[..take]);
        offset += take as u64;
    }
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let computed = ModelDigest {
        algorithm: "sha256".into(),
        value: format!("sha256:{hex}"),
    };
    if &computed == expected {
        Ok(())
    } else {
        Err(ProductionIngestionError::IntegrityMismatch {
            identity: format!(
                "{identity}: declared {}, computed {}",
                expected.value, computed.value
            ),
        })
    }
}

#[cfg(test)]
mod tests;
