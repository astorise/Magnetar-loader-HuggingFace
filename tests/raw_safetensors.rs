use std::{collections::BTreeMap, fs, path::Path};

use magnetar_format_safetensors::{TensorToSerialize, serialize};
use magnetar_loader_huggingface::{
    discover_authenticated_sharded_safetensors, discover_raw_safetensors, inspect_safetensors_file,
};
use magnetar_runtime::{
    ModelArtifactSource, ModelDType, ModelDigest, ProductionArtifactPayloadSource,
    ProductionIngestionError, ProductionModelSource, ProductionPayloadRange,
};

fn source(root: &Path) -> ProductionModelSource {
    ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::LocalPath(root.to_path_buf()),
        root.to_path_buf(),
    )
}

fn write(path: &Path, tensors: &[TensorToSerialize<'_>]) {
    fs::write(path, serialize(tensors, &BTreeMap::new()).unwrap()).unwrap();
}

#[test]
fn raw_single_file_preserves_laya_and_adapter_names_and_reads_bounded_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let f32_bytes = [1.0_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat();
    let f16_bytes = [0x00, 0x3c, 0x00, 0xc0];
    let tensors = [
        TensorToSerialize {
            name: "encoder.layers.0.attn.Wqkv.weight",
            dtype: ModelDType::F32,
            shape: vec![2],
            data: &f32_bytes,
        },
        TensorToSerialize {
            name: "scorer.3.bias",
            dtype: ModelDType::F16,
            shape: vec![2],
            data: &f16_bytes,
        },
    ];
    write(&directory.path().join("model.safetensors"), &tensors);
    write(
        &directory.path().join("adapter_model.safetensors"),
        &tensors,
    );
    let expected_digest =
        ModelDigest::sha256(&fs::read(directory.path().join("model.safetensors")).unwrap());
    let source = source(directory.path());
    for raw in [
        discover_raw_safetensors(&source).unwrap(),
        inspect_safetensors_file(&source, "adapter_model.safetensors").unwrap(),
    ] {
        assert!(raw.shards.is_empty());
        assert_eq!(raw.single_file_digest.as_ref(), Some(&expected_digest));
        assert_eq!(raw.tensors.len(), 2);
        for (name, expected, dtype) in [
            (
                "encoder.layers.0.attn.Wqkv.weight",
                f32_bytes.as_slice(),
                ModelDType::F32,
            ),
            ("scorer.3.bias", f16_bytes.as_slice(), ModelDType::F16),
        ] {
            let tensor = raw
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap();
            assert_eq!(tensor.shape, [2]);
            assert_eq!(tensor.storage_dtype, dtype);
            let range = ProductionPayloadRange {
                identity: name.into(),
                offset: tensor.offset_bytes.unwrap(),
                length: tensor.size_bytes.unwrap(),
                digest: tensor.digest.clone(),
            };
            assert_eq!(raw.payloads.read_payload(&range).unwrap(), expected);
            let mut wrong = range.clone();
            wrong.length += 1;
            assert!(matches!(
                raw.payloads.read_payload(&wrong),
                Err(ProductionIngestionError::PayloadOutOfBounds { .. })
            ));
        }
    }
    assert!(matches!(
        inspect_safetensors_file(&source, "../outside.safetensors"),
        Err(ProductionIngestionError::UnauthorizedSource { .. })
    ));
    assert!(matches!(
        inspect_safetensors_file(&source, "missing.safetensors"),
        Err(ProductionIngestionError::RequiredPartMissing { .. })
    ));
}

#[test]
fn raw_sharded_index_keeps_original_names_and_shard_locations() {
    let directory = tempfile::tempdir().unwrap();
    let first = 3.0_f32.to_le_bytes();
    let second = (-4.0_f32).to_le_bytes();
    write(
        &directory.path().join("one.safetensors"),
        &[TensorToSerialize {
            name: "encoder.embeddings.norm.weight",
            dtype: ModelDType::F32,
            shape: vec![1],
            data: &first,
        }],
    );
    write(
        &directory.path().join("two.safetensors"),
        &[TensorToSerialize {
            name: "head.layers.0.norm1.weight",
            dtype: ModelDType::F32,
            shape: vec![1],
            data: &second,
        }],
    );
    fs::write(directory.path().join("model.safetensors.index.json"), br#"{"weight_map":{"encoder.embeddings.norm.weight":"one.safetensors","head.layers.0.norm1.weight":"two.safetensors"}}"#).unwrap();
    let raw = discover_raw_safetensors(&source(directory.path())).unwrap();
    assert_eq!(raw.shards.len(), 2);
    assert!(raw.single_file_digest.is_none());
    for (name, expected) in [
        ("encoder.embeddings.norm.weight", first),
        ("head.layers.0.norm1.weight", second),
    ] {
        let tensor = raw
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap();
        let range = ProductionPayloadRange {
            identity: name.into(),
            offset: tensor.offset_bytes.unwrap(),
            length: tensor.size_bytes.unwrap(),
            digest: tensor.digest.clone(),
        };
        assert_eq!(raw.payloads.read_payload(&range).unwrap(), expected);
    }
}

#[test]
fn authenticated_shards_bind_half_precision_payloads_and_refuse_mutation() {
    let directory = tempfile::tempdir().unwrap();
    let first = [0x80, 0x3f, 0x00, 0xc0];
    let second = [0x00, 0x3c, 0x00, 0xc0];
    for (file, name, dtype, bytes) in [
        ("one.safetensors", "mixer.weight", ModelDType::Bf16, first),
        ("two.safetensors", "head.weight", ModelDType::F16, second),
    ] {
        write(
            &directory.path().join(file),
            &[TensorToSerialize {
                name,
                dtype,
                shape: vec![2],
                data: &bytes,
            }],
        );
    }
    fs::write(
        directory.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"mixer.weight":"one.safetensors","head.weight":"two.safetensors"}}"#,
    )
    .unwrap();
    let expected: BTreeMap<_, _> = ["one.safetensors", "two.safetensors"]
        .into_iter()
        .map(|name| {
            (
                name.to_string(),
                ModelDigest::sha256(&fs::read(directory.path().join(name)).unwrap()),
            )
        })
        .collect();
    let source = source(directory.path());
    let raw = discover_authenticated_sharded_safetensors(&source, &expected).unwrap();
    assert_eq!(raw.shards.len(), 2);
    for (name, bytes) in [("mixer.weight", first), ("head.weight", second)] {
        let tensor = raw
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap();
        assert_eq!(tensor.digest.as_ref(), Some(&ModelDigest::sha256(&bytes)));
        let range = ProductionPayloadRange {
            identity: name.into(),
            offset: tensor.offset_bytes.unwrap(),
            length: tensor.size_bytes.unwrap(),
            digest: None,
        };
        assert_eq!(raw.payloads.read_payload(&range).unwrap(), bytes);
        assert!(raw.payloads.map_payload(&range).unwrap().is_none());
    }
    let mut missing = expected.clone();
    missing.remove("two.safetensors");
    assert!(matches!(
        discover_authenticated_sharded_safetensors(&source, &missing),
        Err(ProductionIngestionError::IntegrityMismatch { .. })
    ));
    let mut wrong = expected.clone();
    wrong.insert("one.safetensors".into(), ModelDigest::sha256(b"wrong"));
    assert!(matches!(
        discover_authenticated_sharded_safetensors(&source, &wrong),
        Err(ProductionIngestionError::IntegrityMismatch { .. })
    ));

    // A caller cannot bypass authentication by omitting the range digest or
    // by changing the file after its whole-shard digest was verified.
    let tensor = raw
        .tensors
        .iter()
        .find(|tensor| tensor.name == "mixer.weight")
        .unwrap();
    let range = ProductionPayloadRange {
        identity: tensor.name.clone(),
        offset: tensor.offset_bytes.unwrap(),
        length: tensor.size_bytes.unwrap(),
        digest: None,
    };
    let path = directory.path().join("one.safetensors");
    let mut mutated = fs::read(&path).unwrap();
    *mutated.last_mut().unwrap() ^= 1;
    fs::write(&path, mutated).unwrap();
    assert!(matches!(
        raw.payloads.read_payload(&range),
        Err(ProductionIngestionError::IntegrityMismatch { .. })
    ));
    assert!(matches!(
        discover_authenticated_sharded_safetensors(&source, &expected),
        Err(ProductionIngestionError::IntegrityMismatch { .. })
    ));
}

#[cfg(unix)]
#[test]
fn unauthorized_index_is_not_ignored_in_favor_of_a_single_file() {
    use std::os::unix::fs::symlink;
    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let external_index = outside.path().join("index.json");
    fs::write(&external_index, b"{}").unwrap();
    symlink(
        &external_index,
        directory.path().join("model.safetensors.index.json"),
    )
    .unwrap();
    let value = 1.0_f32.to_le_bytes();
    write(
        &directory.path().join("model.safetensors"),
        &[TensorToSerialize {
            name: "x",
            dtype: ModelDType::F32,
            shape: vec![1],
            data: &value,
        }],
    );
    assert!(matches!(
        discover_raw_safetensors(&source(directory.path())),
        Err(ProductionIngestionError::UnauthorizedSource { .. })
    ));
}
