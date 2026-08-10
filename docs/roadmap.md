# Roadmap

Planned features beyond the 0.1.0 scope (dense llama-family GGUF Q8_0 on a
single RTX 5090). Ordered roughly by value-per-effort for a single-GPU
inference box; each item is independent unless a dependency is noted.

Estimates: **S** ≈ days, **M** ≈ 1–2 weeks, **L** ≈ several weeks of
agent-driven development including tests and coherence gates.

## 1. K-quant support (Q4_K / Q5_K / Q6_K) — M

Q8_0 keeps quality but costs ~13 GB for a 12B model. K-quants roughly halve
that, fitting 30B-class dense models on one 5090 and leaving VRAM for KV.

- Dense prefill GEMMs per format (`gemm_q4k.cu`, `gemm_q6k.cu`) plus
  tile/layout variants for the IMMA and HMMA paths
  (`mmq_q4k_imma_tile.cu`, `mmq_q4k_imma_layout.cu`, `mmq_q4k_hmma.cu`).
- Decode reuses the existing NVFP4 decode-cache path — only the source-dequant
  step needs the new formats, and the GEMV fallback family already parses
  K-quant blocks.
- Gate: perplexity parity harness vs Q8_0 baseline; coherence suite.

## 2. KV-cache quantization — M

FP16 KV is the context-length ceiling: ~1 MiB per token for a 12B GQA model.
FP8 KV doubles usable context; INT8/INT4/NVFP4 go further at increasing
quality risk.

- Paged-attention decode variants per KV dtype: `attention_paged_fp8.cu`
  (+ split-K tile variant), `attention_paged_int8.cu`,
  `attention_paged_int4.cu`, `attention_paged_nvfp4.cu` (+ tensor-core
  variant). All share the online-softmax core via
  `attention_paged_common.cuh` — only the K/V dequant differs.
- Config: `kv.dtype` key + CLI flags; auto-fallback to FP16 for
  unsupported head dims.
- Gate: long-context NIAH-style retrieval check (see item 10) + degen suite.

## 3. SafeTensors / Hugging Face checkpoint loading — M

GGUF-only excludes checkpoints published without a GGUF conversion.

- `safetensors_loader.cpp` / `safetensors_raw.cpp` (mmap, BF16/FP16/FP8
  tensors), `hf_config_loader.cpp` (config.json → ModelConfig),
  `sentencepiece_loader.cpp` for non-BPE tokenizers,
  llm-compressor recipe support for pre-quantized checkpoints.
- Reuses the existing weight-map name ladder; the FP8/NVFP4 caches build the
  same way once tensors are resident.
- Gate: same-model GGUF vs SafeTensors logit parity.

## 4. Mixture-of-Experts architectures — L

Unlocks Mixtral, Qwen3-MoE, and gpt-oss-class models, which are the strongest
open checkpoints per active-parameter count.

- Routing: `moe_routing.cu` (top-k gate + permute), fused
  gate-topk kernel.
- Expert GEMMs: CUTLASS grouped GEMM (`gemm_cutlass_grouped_3x.cu`,
  `gemm_grouped.cu`), fused MoE GEMV/GEMM for decode
  (`gemm_moe_fused.cu`, `gemm_moe_gemv.cu`), small-M NVFP4 grouped variant.
- Memory: expert LRU cache with host-offload (`expert_cache.cu`,
  `moe_workspace.cu`) so total experts may exceed VRAM.
- Executor: MoE forward path (`executor_forward_moe*.cu`) + MoE pre-dequant
  phase; NVFP4 decode-cache build for expert tensors.
- Depends on: nothing, but item 5 (MXFP4) shares the gpt-oss target.
- Gate: coherence suite per MoE model; expert-cache thrash benchmark.

## 5. MXFP4 weight format (gpt-oss family) — L

gpt-oss ships MXFP4-native weights; supporting them without requantization
preserves the published quality.

- MXFP4 GEMV/GEMM (`mxfp4_gemm.cu`, CUTLASS `gemm_cutlass_mxfp4_sm120.cu`),
  MXFP4-aware FMHA prefill, format conversion (`gpt_oss_mxfp4_convert.cu`),
  Hadamard transform for outlier smoothing (`hadamard.cu`), FFN sparsity
  probe/mask for the interleaved-sparse variants.
- Depends on: item 4 (gpt-oss is MoE).

## 6. Speculative-quality and offline quantization tooling — M

An offline quantizer turns any FP16/BF16 checkpoint into the engine's
preferred formats with calibration.

- `quench-quantize` tool: AWQ scale search (`awq.cu`, `awq_plan.cpp`),
  per-tensor policy (`tensor_policy.cpp`), calibration statistics capture,
  GPTQ-format dequant import, SafeTensors writer for the output.
- Depends on: item 3 (SafeTensors read/write).
- Gate: perplexity delta vs source checkpoint below threshold.

## 7. Multi-head latent attention (DeepSeek family) — L

MLA compresses KV into a latent space — DeepSeek-V3-class models need it.

- Latent absorption (`mla_absorb.cu`), KV assembly (`mla_kv_assemble.cu`),
  profile plumbing for the split q/kv projection geometry.
- Depends on: item 2 helps (MLA's KV savings compound with KV quant).

## 8. Hybrid SSM / gated-delta-net architectures — L

Qwen3-Next-class hybrids interleave attention with linear-time mixers.

- SSM scan (`ssm.cu`), gated delta net (`gdn.cu`, `gdn_scan.cu`,
  tensor-core scan variant), executor dispatch for mixed layer types,
  recurrent-state snapshot/restore integration for prefix caching (the
  snapshot store already exists in the runtime).
- Gate: long-generation coherence (hybrids fail late, not early).

## 9. Vision / multimodal input — L

Image understanding for Qwen3-VL-class checkpoints: mmproj GGUF loading,
image preprocessing, vision encoder, M-RoPE.

- `src/vision/` stack: image processor, ViT encoder + kernels, patch-grid
  handling, DeepStack feature injection, weight load/map/upload for the
  vision tower.
- Model glue: image placeholder expansion in the chat template, M-RoPE
  position generation, engine-side pipeline (`engine_qwen3vl.cpp`).
- API: `quench_set_image` / `quench_add_image` C API, base64 image parts in
  the OpenAI/Anthropic server, `--image` / `--mmproj` CLI flags.
- Gate: sight-check harness (describe a known image set).

## 10. Evaluation & benchmarking harness — S

Make performance claims publishable (the README currently promises numbers).

- `quench-bench` tool (attention / GEMM / end-to-end token throughput),
  NIAH long-context retrieval eval, multiturn + agent-loop latency
  benchmarks, roofline pipeline (NCU/NSYS capture → classify → plot →
  regress) with CI perf-baseline gate.
- No dependencies; do first if publishing numbers matters sooner.

## 11. LoRA adapters — S

Runtime-applied low-rank deltas for cheap per-task specialization.

- Adapter loader (`lora_adapter.cpp`), executor-side delta application
  (`executor_lora.cu`), per-request adapter selection in the server.

## 12. Embeddings & rerank endpoint — S

Serve encoder-style models next to the generator for RAG stacks.

- Encoder forward pass (`encoder_forward.cu`), `/v1/rerank` server handler,
  pooling + scoring config.

## Sequencing note

Items 1–3 extend the current dense pipeline and are safe to run in parallel.
Items 4–5 and 7–9 each introduce a new forward-pass topology — land them one
at a time behind the coherence gate. Item 10 is independent and unblocks
publishing benchmarks for everything else.
