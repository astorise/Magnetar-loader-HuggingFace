# magnetar-loader-huggingface

## Purpose

The first external, pinned production Model Artifact ingestor for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime
(`implement-production-qwen-model-loading`). Implements
`magnetar-runtime`'s generic `ProductionModelArtifactIngestor` contract for
Hugging Face-style Qwen bundles: real `config.json` parsing, real
`tokenizer.json`-backed tokenization, single-file and sharded Safetensors
weight discovery (composing the real `magnetar-format-safetensors`
parser), and `tokenizer_config.json`/`generation_config.json`
normalization.

## Status

**Real implementation**, not a fixture. `src/config.rs` parses real
Hugging Face `config.json` bytes into `magnetar-runtime`'s generic
`ModelArchitectureConfig`, rejecting missing/zero/inconsistent
architecture values structurally (invalid head/KV-head relationships,
non-dividing `head_dim` derivation, non-positive `rope_theta`, ...).
`src/weights.rs` reuses the real `magnetar-format-safetensors` parser for
both single-file `model.safetensors` and Hugging Face-style sharded
`model.safetensors.index.json` + `model-*-of-*.safetensors` bundles,
cross-checking the index against real parsed shard inventories, and
exposes a bounded, on-demand `ProductionArtifactPayloadSource` that reads
exactly one tensor's bytes per call rather than holding a whole weight
file in memory. `src/tokenizer.rs` loads a real tokenizer through the
`tokenizers` crate (kept out of `magnetar-runtime` entirely -- this crate
is the only place in the workspace that depends on it), validating
vocabulary size against the model's declared `vocab_size`.

Parsing/normalizing a bundle never grants trust: the `ModelManifest` this
crate produces still goes through `magnetar-runtime`'s own
`ModelManifest::validate` and `ModelTrustStore::evaluate` like any other
manifest before Model Loading may materialize anything from it.

## Governing contract

[`production-model-ingestion`](https://github.com/astorise/Magnetar/blob/main/openspec/changes/implement-production-qwen-model-loading/specs/production-model-ingestion/spec.md)
in the main Magnetar repository's OpenSpec change set defines the generic
Runtime-owned contract this crate implements.

## Relationship to magnetar-runtime

`magnetar-runtime` never imports this crate (enforced by CI's
`submodule-integration` dependency guard, extended to cover
`magnetar-loader-*` crates). An embedder registers a `HuggingFaceIngestor`
instance with `magnetar-runtime`'s generic `ProductionIngestionRegistry`;
Runtime performs every trust, memory, component, residency,
materialization, and readiness decision from there. It is pinned into the
main Magnetar repository as a git submodule at `loaders/huggingface`.
