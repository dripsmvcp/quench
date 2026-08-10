# Quant / dequant pipeline

Explains the two parallel layers that handle quantized weights and the boundary between them.

## Two layers, one direction

| Layer | Where | When | What |
|---|---|---|---|
| **Hot-path dequant** | `src/quant/dequant_*.cu` | Per-forward-pass, per-token | In-place dequant for kernels that take FP16/FP8 inputs and need to materialize from a GGUF block on demand. Tiny scratch buffers, no persistent state. |
| **Init-time pre-dequant** | `src/exec/pre_dequant_phase*.cu` | Once at `Engine::init` | Multi-phase orchestration that decides which weights live in which device tier (FP16 cache, FP8 cache, NVFP4 decode cache) and runs the actual format conversions. |

Both layers ultimately call kernels from `src/quant/` (`dequant_q8_to_fp16`-class row-block kernels, `quantize_fp16_nvfp4_cutlass`, `calibrate_quantize_fp8_async`, etc.). The difference is **lifetime**: hot-path dequant is per-call ephemeral; pre-dequant produces device tensors that live as long as the engine.

## Files

### `src/quant/` — kernels + hot-path orchestration
- `dequant_fp16.cu` — generic GGUF-block → FP16 row-block kernels
- `dequant_gpu.cu` — small device-side dequant helpers
- `fp8_quant.cu`, `fp8_utils.cu`, `fp8_utils.cuh` — FP8 (E4M3) quant/dequant + calibration
- `nvfp4_quant.cu`, `nvfp4_gemm.cu`, `nvfp4_gemv_dense.cu`, `nvfp4_gemv_fused.cu` — NVFP4 quant + GEMM/GEMV
- `quant_gemm.cu`, `quant_gemm.h`, `quant_types.h` — shared types + dispatch shims

### `src/exec/pre_dequant_*.cu` — init-time pipeline
Pure orchestration; calls `src/quant/` kernels for the actual format work.

- `executor_pre_dequant.cu` — small orchestrator that calls each phase in order
- `pre_dequant_internal.h` — shared helpers (`borrow_payload_from_wcache`, `for_each_dense_weight`, etc.)
- `pre_dequant_phase1_fp16_cache.cu` — Phase 1: GGUF Q8_0 → FP16 device cache
- `pre_dequant_phase2_fp8_cache.cu` — Phase 2: FP16 → FP8 device tensors for the `fp8_prefill` path
- `pre_dequant_phase3_nvfp4_decode.cu` — Phase 3: NVFP4 decode-cache quantization (the bulk), split further into `pre_dequant_phase3_fp8.cu` and `pre_dequant_phase3_cutlass.cu`
- `pre_dequant_phase4_tensor_registry.cu` — Phase 4: WeightMap → role/tier registration

## GEMM dispatch — the registry pattern

`src/exec/gemm_kernel_registry.{cu,h}` is the unconditional dispatch path.

Adding a new quant tier is a single-file change: implement the kernel in `src/exec/gemm_kernel_<format>.cu`, call `register_gemm_kernel(strategy_key, fn_ptr)` from a static initializer or registration helper, and the dispatcher picks it up. Existing tiers:

- `gemm_kernel_cutlass_nvfp4.cu` — CUTLASS NVFP4 (default for NVFP4 prefill)
- `gemm_kernel_nvfp4_gemm.cu`, `gemm_kernel_nvfp4_gemv.cu` — non-CUTLASS NVFP4 fallbacks
- `gemm_kernel_fp8.cu` — FP8 paths
- `gemm_kernel_gguf.cu` — GGUF small-M path (dp4a + mmvq)

## Boundary rules

When adding a new quant format:

- If the format needs **per-call decode** during forward pass: add the kernel to `src/quant/` and a dispatch entry to `src/exec/gemm_kernel_<format>.cu`.
- If the format needs **per-engine setup** (allocating tiered device tensors, computing a quant cache at load time): add a new `src/exec/pre_dequant_phase<N>_<name>.cu` file and one call from `executor_pre_dequant.cu`.
- The boundary stays clean as long as `src/quant/` stays kernels-and-helpers and `src/exec/pre_dequant_*.cu` stays orchestration.

## Where the two layers meet at runtime

1. `Engine::init_kv_cache()` (`src/runtime/engine_kv_cache_init.cpp`) → `executor_pre_dequant.cu::pre_dequant_weights()` runs all phases sequentially, producing device tensors in the right tiers. **The call site is load-bearing, not incidental**: the whole pipeline runs *before* the KV pool is sized, so the pool takes the measured residual instead of the caches being sized against an estimate.
2. Per-forward-pass: `gemm_kernel_registry` dispatches to the right `gemm_kernel_<format>.cu`, which reads the pre-dequant tier or calls `src/quant/dequant_*.cu` for an on-demand decode.

Both paths share kernels in `src/quant/`. The split between `src/quant/` and `src/exec/pre_dequant_*.cu` is **when the work happens**, not what work it is.
