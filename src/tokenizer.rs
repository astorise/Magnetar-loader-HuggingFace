//! Real `tokenizer.json`-backed [`Tokenizer`] implementation
//! (`implement-production-qwen-model-loading` task group 9), plus
//! `tokenizer_config.json`/`generation_config.json` normalization into
//! `magnetar-runtime`'s existing roadmap contracts. Kept entirely outside
//! `magnetar-runtime` -- this crate is the only place in the workspace
//! that depends on the `tokenizers` crate.

use magnetar_runtime::ModelDigest;
use magnetar_runtime::model_format_roadmap::{
    GenerationConfigMetadata, PaddingSide, TokenizerConfigMetadata, TruncationSide,
};
use magnetar_runtime::production_model_ingestion::ProductionIngestionError;
use magnetar_runtime::tokenizer::{
    DecodeInput, DecodeOutput, EncodeInput, EncodeOutput, SpecialToken, SpecialTokenKind,
    TokenIdRange, TokenOffset, Tokenizer, TokenizerArtifactId, TokenizerError, TokenizerFamily,
    TokenizerId, TokenizerMetadata, TokenizerRevision, TruncationPolicy,
};
use serde::Deserialize;
use std::collections::BTreeSet;
use tokenizers::Tokenizer as HfTokenizerImpl;

#[derive(Debug, Default, Deserialize)]
struct RawTokenizerConfig {
    #[serde(default)]
    tokenizer_class: Option<String>,
    #[serde(default)]
    model_max_length: Option<f64>,
    #[serde(default)]
    padding_side: Option<String>,
    #[serde(default)]
    truncation_side: Option<String>,
    #[serde(default)]
    chat_template: Option<String>,
    #[serde(default)]
    bos_token: Option<TokenField>,
    #[serde(default)]
    eos_token: Option<TokenField>,
    #[serde(default)]
    pad_token: Option<TokenField>,
    #[serde(default)]
    clean_up_tokenization_spaces: Option<bool>,
}

/// A Hugging Face `*_token` field is either a bare string or an object
/// with a `content` field (`AddedToken`-shaped) -- both are accepted.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenField {
    Plain(String),
    Structured { content: String },
}

impl TokenField {
    fn text(&self) -> &str {
        match self {
            Self::Plain(value) => value,
            Self::Structured { content } => content,
        }
    }
}

/// Parses and normalizes `tokenizer_config.json` into
/// [`TokenizerConfigMetadata`] (task 9.2).
pub fn parse_tokenizer_config(
    bytes: &[u8],
) -> Result<TokenizerConfigMetadata, ProductionIngestionError> {
    let raw: RawTokenizerConfig = serde_json::from_slice(bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("tokenizer_config.json is not valid JSON: {error}"),
        }
    })?;
    Ok(TokenizerConfigMetadata {
        tokenizer_class: raw.tokenizer_class,
        model_max_length: raw
            .model_max_length
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(|value| value as u32),
        padding_side: raw.padding_side.as_deref().and_then(|side| match side {
            "left" => Some(PaddingSide::Left),
            "right" => Some(PaddingSide::Right),
            _ => None,
        }),
        truncation_side: raw.truncation_side.as_deref().and_then(|side| match side {
            "left" => Some(TruncationSide::Left),
            "right" => Some(TruncationSide::Right),
            _ => None,
        }),
        chat_template_reference: raw.chat_template,
        bos_token: raw.bos_token.as_ref().map(|token| token.text().to_string()),
        eos_token: raw.eos_token.as_ref().map(|token| token.text().to_string()),
        pad_token: raw.pad_token.as_ref().map(|token| token.text().to_string()),
        added_special_tokens: Vec::new(),
        clean_up_tokenization_spaces: raw.clean_up_tokenization_spaces,
    })
}

#[derive(Debug, Default, Deserialize)]
struct RawGenerationConfig {
    #[serde(default)]
    max_length: Option<u32>,
    #[serde(default)]
    max_new_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    do_sample: Option<bool>,
}

/// Parses and normalizes `generation_config.json` into
/// [`GenerationConfigMetadata`] (task 9.5) -- these values are defaults
/// only; `apply_generation_override` (in `magnetar-runtime`) is what a
/// caller uses to let an explicit request value win.
pub fn parse_generation_config(
    bytes: &[u8],
) -> Result<GenerationConfigMetadata, ProductionIngestionError> {
    let raw: RawGenerationConfig = serde_json::from_slice(bytes).map_err(|error| {
        ProductionIngestionError::MalformedMetadata {
            reason: format!("generation_config.json is not valid JSON: {error}"),
        }
    })?;
    Ok(GenerationConfigMetadata {
        max_length: raw.max_length,
        max_new_tokens: raw.max_new_tokens,
        temperature: raw.temperature,
        top_k: raw.top_k,
        top_p: raw.top_p,
        repetition_penalty: raw.repetition_penalty,
        eos_token_id: None,
        bos_token_id: None,
        pad_token_id: None,
        do_sample: raw.do_sample,
        stop_strings: Vec::new(),
    })
}

/// Real, `tokenizer.json`-backed [`Tokenizer`] implementation
/// (`Tokenizer Contract`). Encode/decode delegate to the real
/// `tokenizers::Tokenizer`; everything else (offsets, attention mask,
/// truncation, special-token policy) is normalized into the same shapes
/// `magnetar-runtime`'s `FixtureTokenizer` already produces, so callers do
/// not need to special-case a real tokenizer's output.
pub struct HuggingFaceTokenizer {
    inner: HfTokenizerImpl,
    metadata: TokenizerMetadata,
}

impl HuggingFaceTokenizer {
    /// Loads a real tokenizer from `tokenizer.json` bytes plus the model
    /// identity metadata needed to construct a [`TokenizerMetadata`].
    /// Validates vocabulary size and BOS/EOS/PAD special-token
    /// compatibility against `expected_vocab_size` (task 9.3).
    ///
    /// `expected_vocab_size` is the embedding table's row count (`config.
    /// json`'s `vocab_size`, i.e. `token_embedding`'s declared shape), not
    /// necessarily the tokenizer's own vocabulary size exactly: real Qwen2/
    /// 2.5 checkpoints pad the embedding table to a hardware-friendly
    /// round number (e.g. Qwen2.5-0.5B-Instruct declares `vocab_size:
    /// 151936` while its real `tokenizer.json` vocabulary is `151665` --
    /// the extra rows are simply unused padding, never referenced by any
    /// real token id). A tokenizer vocabulary *larger* than the embedding
    /// table is the genuine incompatibility this rejects: a token id it
    /// could produce would then index past `token_embedding`'s real rows.
    pub fn from_bytes(
        tokenizer_json: &[u8],
        tokenizer_config: Option<&TokenizerConfigMetadata>,
        artifact_id: impl Into<String>,
        expected_vocab_size: Option<u64>,
    ) -> Result<Self, ProductionIngestionError> {
        let inner = HfTokenizerImpl::from_bytes(tokenizer_json).map_err(|error| {
            ProductionIngestionError::MalformedMetadata {
                reason: format!("tokenizer.json failed to load: {error}"),
            }
        })?;
        let vocabulary_size = inner.get_vocab_size(true) as u32;
        if let Some(expected) = expected_vocab_size
            && u64::from(vocabulary_size) > expected
        {
            return Err(ProductionIngestionError::MalformedMetadata {
                reason: format!(
                    "tokenizer.json vocabulary size {vocabulary_size} exceeds the model's \
                     declared vocab_size {expected} (embedding table row count)"
                ),
            });
        }

        let mut special_tokens = Vec::new();
        let mut push_special = |kind: SpecialTokenKind, text: Option<&str>| {
            let Some(text) = text else { return };
            if let Some(id) = inner.token_to_id(text) {
                special_tokens.push(SpecialToken::new(kind, text, id));
            }
        };
        push_special(
            SpecialTokenKind::Bos,
            tokenizer_config.and_then(|config| config.bos_token.as_deref()),
        );
        push_special(
            SpecialTokenKind::Eos,
            tokenizer_config.and_then(|config| config.eos_token.as_deref()),
        );
        push_special(
            SpecialTokenKind::Pad,
            tokenizer_config.and_then(|config| config.pad_token.as_deref()),
        );

        let max_id = special_tokens
            .iter()
            .map(|token| token.id)
            .max()
            .unwrap_or(0)
            .max(vocabulary_size.saturating_sub(1));

        let artifact_id = artifact_id.into();
        let metadata = TokenizerMetadata {
            id: TokenizerId::new(&artifact_id).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("invalid tokenizer id '{artifact_id}': {error}"),
                }
            })?,
            artifact: TokenizerArtifactId::new(&artifact_id).map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: format!("invalid tokenizer artifact id '{artifact_id}': {error}"),
                }
            })?,
            digest: ModelDigest::sha256(tokenizer_json),
            family: TokenizerFamily::new("huggingface").map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: error.to_string(),
                }
            })?,
            revision: TokenizerRevision::new("1").map_err(|error| {
                ProductionIngestionError::MalformedMetadata {
                    reason: error.to_string(),
                }
            })?,
            vocabulary_size,
            added_token_count: inner.get_added_tokens_decoder().len() as u32,
            token_id_range: TokenIdRange::new(0, max_id),
            model_max_length: tokenizer_config.and_then(|config| config.model_max_length),
            special_tokens,
            additional_special_tokens: Vec::new(),
            byte_fallback: false,
            normalization: None,
            pre_tokenizer: None,
            supports_offsets: true,
            supports_token_type_ids: false,
            supports_browser: false,
        };
        metadata
            .validate()
            .map_err(|error| ProductionIngestionError::MalformedMetadata {
                reason: format!("tokenizer metadata failed validation: {error}"),
            })?;

        Ok(Self { inner, metadata })
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn metadata(&self) -> &TokenizerMetadata {
        &self.metadata
    }

    fn encode(&self, input: EncodeInput) -> Result<EncodeOutput, TokenizerError> {
        if input.return_offsets && !self.metadata.supports_offsets {
            return Err(TokenizerError::OffsetsUnsupported);
        }
        let encoding = self
            .inner
            .encode(input.text.as_str(), input.add_special_tokens)
            .map_err(|error| TokenizerError::BatchInputInvalid {
                message: error.to_string(),
            })?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut offsets: Option<Vec<TokenOffset>> = input.return_offsets.then(|| {
            encoding
                .get_offsets()
                .iter()
                .map(|(start, end)| TokenOffset {
                    byte_start: *start as u32,
                    byte_end: *end as u32,
                    char_start: Some(*start as u32),
                    char_end: Some(*end as u32),
                })
                .collect()
        });

        let limit = input
            .max_tokens
            .or(self.metadata.model_max_length.map(|value| value as usize));
        if let Some(limit) = limit
            && token_ids.len() > limit
        {
            match input.truncation {
                TruncationPolicy::None => {
                    return Err(TokenizerError::PromptTooLong {
                        token_count: token_ids.len(),
                        limit,
                    });
                }
                TruncationPolicy::Left | TruncationPolicy::ClientPolicy => {
                    let drop = token_ids.len() - limit;
                    token_ids.drain(0..drop);
                    if let Some(offsets) = &mut offsets {
                        offsets.drain(0..drop);
                    }
                }
                TruncationPolicy::Right | TruncationPolicy::ModelDefault => {
                    token_ids.truncate(limit);
                    if let Some(offsets) = &mut offsets {
                        offsets.truncate(limit);
                    }
                }
                TruncationPolicy::Middle => {
                    let left = limit / 2;
                    let right = limit - left;
                    let tail_start = token_ids.len() - right;
                    let mut truncated = token_ids[..left].to_vec();
                    truncated.extend_from_slice(&token_ids[tail_start..]);
                    token_ids = truncated;
                    if let Some(offsets) = &mut offsets {
                        let mut truncated = offsets[..left].to_vec();
                        truncated.extend_from_slice(&offsets[tail_start..]);
                        *offsets = truncated;
                    }
                }
            }
        }

        Ok(EncodeOutput {
            token_count: token_ids.len(),
            attention_mask: Some(vec![1u8; token_ids.len()]),
            token_type_ids: None,
            token_ids,
            offsets,
            diagnostics: Vec::new(),
        })
    }

    fn decode(&self, input: DecodeInput) -> Result<DecodeOutput, TokenizerError> {
        for id in &input.token_ids {
            if !self.metadata.token_id_range.contains(*id) {
                return Err(TokenizerError::InvalidTokenId { token_id: *id });
            }
        }
        let text = self
            .inner
            .decode(&input.token_ids, input.skip_special_tokens)
            .map_err(|error| TokenizerError::InvalidUtf8 {
                message: error.to_string(),
            })?;
        Ok(DecodeOutput {
            text,
            consumed_token_count: input.token_ids.len(),
            pending_partial_state: None,
            diagnostics: Vec::new(),
        })
    }
}

/// Compatibility-relevant special-token kinds a production tokenizer
/// bundle is expected to carry, used by callers validating a loaded
/// tokenizer against a model's requirements.
pub fn required_special_token_kinds() -> BTreeSet<SpecialTokenKind> {
    BTreeSet::from([SpecialTokenKind::Bos, SpecialTokenKind::Eos])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny, real `tokenizer.json` (byte-level BPE over a 6-token
    /// vocabulary) -- valid tokenizers-crate input, not a fixture stand-in.
    fn tiny_tokenizer_json() -> Vec<u8> {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0, "content": "<bos>", "special": true,
                    "single_word": false, "lstrip": false, "rstrip": false, "normalized": false
                },
                {
                    "id": 1, "content": "<eos>", "special": true,
                    "single_word": false, "lstrip": false, "rstrip": false, "normalized": false
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<bos>": 0,
                    "<eos>": 1,
                    "hello": 2,
                    "world": 3,
                    "foo": 4,
                    "bar": 5
                },
                "unk_token": "hello"
            }
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn loads_a_real_tokenizer_and_encodes_decodes() {
        let tokenizer_config = TokenizerConfigMetadata {
            bos_token: Some("<bos>".into()),
            eos_token: Some("<eos>".into()),
            ..TokenizerConfigMetadata::default()
        };
        let tokenizer = HuggingFaceTokenizer::from_bytes(
            &tiny_tokenizer_json(),
            Some(&tokenizer_config),
            "test-tokenizer",
            Some(6),
        )
        .expect("real tokenizer.json loads");

        assert_eq!(tokenizer.metadata().vocabulary_size, 6);
        assert!(
            tokenizer
                .metadata()
                .special_token(SpecialTokenKind::Bos)
                .is_some()
        );
        assert!(
            tokenizer
                .metadata()
                .special_token(SpecialTokenKind::Eos)
                .is_some()
        );

        let output = tokenizer
            .encode(EncodeInput {
                add_special_tokens: false,
                ..EncodeInput::new("hello world")
            })
            .expect("encode succeeds");
        assert_eq!(output.token_ids, vec![2, 3]);

        let decoded = tokenizer
            .decode(DecodeInput {
                token_ids: output.token_ids,
                skip_special_tokens: true,
                clean_up_tokenization_spaces: false,
                streaming_state: None,
            })
            .expect("decode succeeds");
        assert_eq!(decoded.text, "hello world");
    }

    #[test]
    fn rejects_a_tokenizer_vocabulary_larger_than_the_declared_embedding_table() {
        // tiny_tokenizer_json declares 6 real token ids (0..=5); a
        // declared embedding table of only 3 rows cannot hold them all --
        // a genuine incompatibility, not padding.
        let error = match HuggingFaceTokenizer::from_bytes(
            &tiny_tokenizer_json(),
            None,
            "test-tokenizer",
            Some(3),
        ) {
            Err(error) => error,
            Ok(_) => panic!("expected a vocabulary size mismatch error"),
        };
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    /// Real Qwen2/2.5 checkpoints pad the embedding table beyond the
    /// tokenizer's real vocabulary (e.g. Qwen2.5-0.5B-Instruct: tokenizer
    /// vocabulary 151665, declared `vocab_size` 151936) -- a tokenizer
    /// vocabulary *smaller* than the declared embedding table is real,
    /// common, and must load successfully, not be rejected as a mismatch.
    #[test]
    fn accepts_a_tokenizer_vocabulary_smaller_than_the_declared_embedding_table() {
        HuggingFaceTokenizer::from_bytes(&tiny_tokenizer_json(), None, "test-tokenizer", Some(999))
            .expect("a padded embedding table larger than the real tokenizer vocabulary is valid");
    }

    #[test]
    fn parses_a_real_tokenizer_config() {
        let bytes = serde_json::json!({
            "tokenizer_class": "Qwen2Tokenizer",
            "model_max_length": 32768,
            "bos_token": null,
            "eos_token": {"content": "<|endoftext|>"},
            "chat_template": "{{ messages }}"
        })
        .to_string()
        .into_bytes();
        let config = parse_tokenizer_config(&bytes).unwrap();
        assert_eq!(config.tokenizer_class.as_deref(), Some("Qwen2Tokenizer"));
        assert_eq!(config.model_max_length, Some(32768));
        assert_eq!(config.eos_token.as_deref(), Some("<|endoftext|>"));
        assert!(config.bos_token.is_none());
    }

    #[test]
    fn parses_a_real_generation_config() {
        let bytes = serde_json::json!({
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.8,
            "do_sample": true
        })
        .to_string()
        .into_bytes();
        let config = parse_generation_config(&bytes).unwrap();
        assert_eq!(config.max_new_tokens, Some(2048));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.do_sample, Some(true));
    }
}
