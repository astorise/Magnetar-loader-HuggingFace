//! Unit tests for the parent module.
//!
//! Kept in its own file so coverage tooling classifies it as test
//! source rather than loader implementation source.

use super::*;
use magnetar_runtime::model::ModelDigest;
use std::io::Write;
use std::sync::Mutex;

/// A mapping over memory that records the prefetch hints it receives.
#[derive(Debug)]
struct Recording {
    bytes: Vec<u8>,
    hints: Mutex<Vec<Vec<(u64, u64)>>>,
}

impl Recording {
    fn new(bytes: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            hints: Mutex::new(Vec::new()),
        })
    }
}

impl MappedPayload for Recording {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), ProductionIngestionError> {
        let start = offset as usize;
        out.copy_from_slice(
            self.bytes
                .get(start..start + out.len())
                .ok_or_else(|| out_of_bounds("recording"))?,
        );
        Ok(())
    }

    fn prefetch(&self, spans: &[(u64, u64)]) {
        self.hints.lock().unwrap().push(spans.to_vec());
    }
}

/// A file of `bytes` that outlives the test's directory handle.
fn file_of(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.bin");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    (dir, path)
}

fn pattern(length: usize, seed: u8) -> Vec<u8> {
    (0..length)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// A mapping of a range of a file reads exactly that range, wherever it
/// starts, and refuses a read that leaves it.
#[test]
fn a_file_range_is_read_exactly() {
    let bytes = pattern(10_000, 7);
    let (_dir, path) = file_of(&bytes);
    let mapping = FileMapping::open(&path, 123, 4_000, "table").unwrap();
    assert_eq!(mapping.len(), 4_000);
    let mut out = vec![0u8; 700];
    mapping.read_at(250, &mut out).unwrap();
    assert_eq!(out, bytes[123 + 250..123 + 250 + 700]);
    // The last byte of the range, then one past it.
    let mut last = [0u8; 1];
    mapping.read_at(3_999, &mut last).unwrap();
    assert_eq!(last[0], bytes[123 + 3_999]);
    assert!(mapping.read_at(3_999, &mut [0u8; 2]).is_err());
    assert!(mapping.read_at(u64::MAX, &mut [0u8; 1]).is_err());
}

/// A range past the end of its file is refused when it is mapped, and a
/// missing file is unavailable, not a panic.
#[test]
fn a_range_past_its_file_is_refused() {
    let (_dir, path) = file_of(&pattern(100, 0));
    assert!(matches!(
        FileMapping::open(&path, 50, 51, "table"),
        Err(ProductionIngestionError::PayloadOutOfBounds { .. })
    ));
    assert!(matches!(
        FileMapping::open(&path, u64::MAX, 2, "table"),
        Err(ProductionIngestionError::PayloadOutOfBounds { .. })
    ));
    assert!(matches!(
        FileMapping::open(&path.with_file_name("absent.bin"), 0, 1, "table"),
        Err(ProductionIngestionError::PayloadUnavailable { .. })
    ));
}

/// A prefetch hint is only a hint: spans in, past and across the range
/// change nothing that is read afterwards.
#[test]
fn a_prefetch_hint_changes_nothing_that_is_read() {
    let bytes = pattern(40_000, 3);
    let (_dir, path) = file_of(&bytes);
    let mapping = FileMapping::open(&path, 4_097, 30_000, "table").unwrap();
    mapping.prefetch(&[(0, 4_096), (10_000, 100), (29_900, 10_000), (u64::MAX, 4)]);
    let mut out = vec![0u8; 30_000];
    mapping.read_at(0, &mut out).unwrap();
    assert_eq!(out, bytes[4_097..4_097 + 30_000]);
}

/// Mappings read as one are their bytes one after the other, whichever
/// way a read crosses their seams, and an empty part is skipped.
#[test]
fn stacked_mappings_read_as_one() {
    let parts: Vec<Vec<u8>> = vec![
        pattern(5, 1),
        vec![],
        pattern(9, 2),
        pattern(1, 3),
        pattern(20, 4),
    ];
    let all = parts.concat();
    let stacked = ConcatMapping::new(
        parts
            .iter()
            .map(|part| Recording::new(part.clone()) as Arc<dyn MappedPayload>)
            .collect(),
    );
    assert_eq!(stacked.len(), all.len() as u64);
    for start in 0..all.len() {
        for end in start..=all.len() {
            let mut out = vec![0u8; end - start];
            stacked.read_at(start as u64, &mut out).unwrap();
            assert_eq!(out, all[start..end], "bytes {start}..{end}");
        }
    }
    assert!(stacked.read_at(all.len() as u64, &mut [0u8; 1]).is_err());
    assert!(stacked.read_at(u64::MAX, &mut [0u8; 1]).is_err());
}

/// A hint on the stacked mapping reaches each part as the spans that fall
/// in it, relative to the part.
#[test]
fn a_hint_on_stacked_mappings_is_split_across_their_parts() {
    let first = Recording::new(vec![0; 10]);
    let second = Recording::new(vec![0; 10]);
    let third = Recording::new(vec![0; 10]);
    let stacked = ConcatMapping::new(vec![first.clone(), second.clone(), third.clone()]);
    // 8..13 crosses the first seam, 21..22 is inside the third part.
    stacked.prefetch(&[(8, 5), (21, 1)]);
    assert_eq!(*first.hints.lock().unwrap(), vec![vec![(8, 2)]]);
    assert_eq!(*second.hints.lock().unwrap(), vec![vec![(0, 3)]]);
    assert_eq!(*third.hints.lock().unwrap(), vec![vec![(1, 1)]]);
}

/// The digest of a mapping is checked in blocks, and equals the digest of
/// its bytes held whole; a block boundary in the middle changes nothing.
#[test]
fn a_mappings_digest_is_checked_in_blocks() {
    // More than three blocks, ending inside one.
    let bytes = pattern(3 * DIGEST_CHUNK + 12_345, 9);
    let mapping = Recording::new(bytes.clone());
    let expected = ModelDigest::sha256(&bytes);
    verify_digest(mapping.as_ref(), &expected, "table").unwrap();

    let mut wrong = bytes.clone();
    *wrong.last_mut().unwrap() ^= 1;
    let error = verify_digest(Recording::new(wrong).as_ref(), &expected, "table").unwrap_err();
    assert!(
        matches!(error, ProductionIngestionError::IntegrityMismatch { ref identity } if identity.starts_with("table:")),
        "{error:?}"
    );

    // A digest of another algorithm cannot match.
    let other = ModelDigest {
        algorithm: "md5".into(),
        value: expected.value.clone(),
    };
    assert!(verify_digest(mapping.as_ref(), &other, "table").is_err());

    // An empty range has the empty digest.
    verify_digest(
        Recording::new(Vec::new()).as_ref(),
        &ModelDigest::sha256(&[]),
        "empty",
    )
    .unwrap();
}
