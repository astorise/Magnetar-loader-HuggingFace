# huggingface-model-ingestion Specification

## Purpose
TBD - created by archiving change support-gptq-quantized-huggingface-ingestion. Update Purpose after archive.
## Requirements
### Requirement: Hugging Face Ingestion Dequantizes Supported GPTQ Projections

The Hugging Face ingestor SHALL dequantize a 4-bit GPTQ-quantized projection (a raw tensor quadruple sharing one module prefix: `<prefix>.qweight`, `<prefix>.qzeros`, `<prefix>.scales`, `<prefix>.g_idx`) to `F32`, at its already Runtime-oriented `[in_features, out_features]` logical shape, before that projection's shape/layout normalization (including this ingestor's own `nn.Linear`-storage-convention transpose) runs, and SHALL report that projection's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact -- never the original packed values. A bit width other than 4 SHALL be rejected with a structured error rather than silently mis-decoded.

#### Scenario: A GPTQ-quantized projection dequantizes before materialization

- **WHEN** a Hugging Face bundle declares a projection as a GPTQ-quantized tensor quadruple
- **THEN** the returned tensor's storage dtype is `F32`, its shape is `[in_features, out_features]`, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: A GPTQ-dequantized projection is not transposed a second time

- **WHEN** a GPTQ-quantized projection is also a name this ingestor's existing `nn.Linear`-storage-convention transpose would otherwise match
- **THEN** dequantization's own already-correct `[in_features, out_features]` orientation is preserved, not swapped again

#### Scenario: An unsupported GPTQ bit width fails closed

- **WHEN** a GPTQ-quantized projection's declared packing implies a bit width other than 4
- **THEN** ingestion fails with a structured error naming the unsupported bit width, rather than producing silently incorrect dequantized values

#### Scenario: A bundle with no GPTQ tensors ingests unaffected

- **WHEN** a Hugging Face bundle's tensors are entirely unquantized
- **THEN** ingestion proceeds normally, unaffected by GPTQ dequantization support existing for other bundles

### Requirement: Hugging Face Ingestion Dequantizes Supported AWQ Projections

The Hugging Face ingestor SHALL dequantize a 4-bit AWQ (GEMM-kernel) quantized projection (a raw tensor triple sharing one module prefix: `<prefix>.qweight`, `<prefix>.qzeros`, `<prefix>.scales`, with no `<prefix>.g_idx` sibling) to `F32`, at its already Runtime-oriented `[in_features, out_features]` logical shape, before that projection's shape/layout normalization (including this ingestor's own `nn.Linear`-storage-convention transpose) runs, and SHALL report that projection's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact -- never the original packed values. A bit width other than 4 SHALL be rejected with a structured error rather than silently mis-decoded.

#### Scenario: An AWQ-quantized projection dequantizes before materialization

- **WHEN** a Hugging Face bundle declares a projection as an AWQ-quantized `.qweight`/`.qzeros`/`.scales` triple with no `.g_idx` sibling
- **THEN** the returned tensor's storage dtype is `F32`, its shape is `[in_features, out_features]`, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: An AWQ-dequantized projection is not transposed a second time

- **WHEN** an AWQ-quantized projection is also a name this ingestor's existing `nn.Linear`-storage-convention transpose would otherwise match
- **THEN** dequantization's own already-correct `[in_features, out_features]` orientation is preserved, not swapped again

#### Scenario: A GPTQ-quantized bundle is not misinterpreted as AWQ

- **GIVEN** a `.qweight` tensor with a `.g_idx` sibling (GPTQ's own disambiguating signal)
- **WHEN** the bundle is ingested
- **THEN** it is dequantized as GPTQ, not AWQ, regardless of the two schemes' otherwise-identical `.qweight`/`.qzeros`/`.scales` naming

---

### Requirement: Hugging Face Ingestion Dequantizes Supported BitsAndBytes Projections

The Hugging Face ingestor SHALL dequantize a 4-bit BitsAndBytes (NF4/FP4) quantized projection -- identified by a `<prefix>.weight` tensor with a `<prefix>.weight.absmax` sibling, and a real `<prefix>.weight.quant_state.bitsandbytes__nf4` JSON blob declaring the projection's true logical shape and block size(s) -- to `F32`, at its real `[out_features, in_features]` logical shape (`nn.Linear`'s own storage convention), before that projection's shape/layout normalization runs, and SHALL report that projection's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact. A codebook other than 16 values (i.e. not 4-bit) SHALL be rejected with a structured error rather than silently mis-decoded. Both single-level and double-quantized (`absmax` itself quantized) encodings SHALL be supported.

#### Scenario: A BitsAndBytes-quantized projection dequantizes before materialization

- **WHEN** a Hugging Face bundle declares a projection as a BitsAndBytes-quantized `.weight` tensor with an `.absmax` sibling and a real `quant_state` JSON blob
- **THEN** the returned tensor's storage dtype is `F32`, its shape matches the real logical shape the `quant_state` blob declares, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: A BitsAndBytes-dequantized projection still transposes normally

- **WHEN** a BitsAndBytes-quantized projection is also a name this ingestor's existing `nn.Linear`-storage-convention transpose matches
- **THEN** it is transposed the same way a plain, unquantized weight of the same name would be -- unlike a GPTQ- or AWQ-dequantized projection, which is excluded from that transpose

#### Scenario: A bundle with no quantized tensors of any supported scheme ingests unaffected

- **WHEN** a Hugging Face bundle's tensors are entirely unquantized
- **THEN** ingestion proceeds normally, unaffected by GPTQ/AWQ/BitsAndBytes dequantization support existing for other bundles

### Requirement: Hugging Face Ingestion Derives A Layered Decoder's Packed Tensors

Hugging Face ingestion SHALL derive the tensors a layered decoder's Model Component requests from its packed tensors:
- **Per-layer input tables:** each layer's slice of `embed_tokens_per_layer` (columns) and `per_layer_model_projection` (rows).
- **Routed experts:** the stacked gate and up projections from `experts.gate_up_proj`, and the stacked down projection from `experts.down_proj`.
- **Router:** `router.proj` with `router.scale` and `hidden^-0.5` folded in, as `f32`, and `router.per_expert_scale`.

It SHALL refuse packed tensors that disagree with `config.json`, and SHALL fold no factor into any other weight.

#### Scenario: Per-layer embedding slice
- **WHEN** a bundle packs 2 layers of 2 per-layer input values into `embed_tokens_per_layer`
- **THEN** `layers.1.per_layer_embedding` holds columns 2 and 3 of every row

#### Scenario: Experts that disagree with the configuration
- **WHEN** `experts.gate_up_proj` holds a different expert count from `config.json`'s
- **THEN** ingestion fails with malformed metadata

### Requirement: Hugging Face Ingestion Reads Latent Attention, YaRN And Routed Expert Layers

Hugging Face ingestion SHALL read a `config.json` that declares a `kv_lora_rank` as multi-head latent attention, a `yarn` rope scaling as YaRN, and routed experts after `first_k_dense_replace` dense layers with their scoring, groups, scaling factor and selection bias, and SHALL split the checkpoint's fused latent-attention projections by rows and stack its experts stored one by one into the tensors the Model Component requests.

The query projection (`q_proj`, or `q_b_proj` after a low-rank query) stores each head's position-independent rows before its rotated rows; `kv_a_proj_with_mqa` stores the latent's rows before the shared rotated key's; `kv_b_proj` stores each head's key rows before its value rows. Row splits copy bytes. A layer missing a projection, a projection of the wrong shape, latent projections without a declared `kv_lora_rank`, and experts that are missing or differ in shape are refused.

#### Scenario: A DeepSeek-V2-Lite configuration
- **WHEN** the loader parses DeepSeek-V2-Lite's `config.json`
- **THEN** it declares latent attention with no low-rank query, a 512-wide latent, 128 + 64-wide query and key heads and 128-wide values, 16 K/V heads of 192, YaRN with factor 40 over 4096 positions and `mscale_all_dim` 0.707, a dense first layer, 64 experts of width 1408 with 6 per token and two shared experts summed to 2816, and softmax routing without groups

#### Scenario: Split projections reproduce the checkpoint's own computation
- **WHEN** a random tiny `DeepseekV2ForCausalLM` saved by transformers 5.23, with or without a low-rank query, is ingested
- **THEN** each derived projection gives the model's own query, latent, key, value, output, expert, shared-expert and router results on the same input (equal, or within 1e-10 for the summed experts)

### Requirement: Hugging Face Ingestion Keeps Only A Multimodal Bundle's Text Decoder

Hugging Face ingestion SHALL ingest a bundle whose `config.json` nests the decoder under `text_config` as that text decoder: the layout comes from `text_config`, the Artifact family is the bundle's own `model_type`, and tensors under the vision and audio towers, their embedders (`model.embed_vision.`, `model.embed_audio.`, `model.vision_embedder.`) and the multimodal projector stay out of the text inventory.

#### Scenario: A unified bundle
- **WHEN** the loader parses the `config.json` of `gemma-4-12B-it`, which declares `gemma4_unified`
- **THEN** the family is `gemma4_unified`, every sixth layer attends over the whole context with 512-wide heads, one K/V head, V derived from K, RoPE base 1e6 and a quarter of the head rotated, and the other layers slide over 1024 tokens with 256-wide heads, eight K/V heads and RoPE base 1e4

#### Scenario: The image-patch embedder
- **WHEN** the bundle stores `model.vision_embedder.patch_dense.weight`
- **THEN** the tensor is not part of the text inventory
