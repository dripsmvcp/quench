# Quantization

quench loads GGUF weights and builds its own device-side compute caches at engine
init. This page explains each format in play, where it is used inside the
engine, and what the trade-offs are.

For per-model picks see [`supported-models.md`](supported-models.md). For the
init-time pipeline that produces the caches see
[`quant-pipeline.md`](quant-pipeline.md).

## Formats and where they show up

| Format | Bits / weight | Source | Used for |
|---|---:|---|---|
| Q8_0 | 8.0 | GGUF | source format on disk; dp4a GEMV decode fallback |
| FP16 | 16.0 | runtime | dequant target for prefill weight cache and reference paths; KV cache |
| FP8 E4M3 | 8.0 | runtime | prefill weight cache (~2× prefill throughput vs FP16) |
| NVFP4 (FP4 E2M1) | 4.0 | runtime | decode weight cache (Blackwell-native block-scaled MMA) |

**Q8_0 is the verified GGUF quantization.** The loader also carries generic
dequant paths for other GGUF quant types, but those paths are unverified —
stage Q8_0 files.

## The two runtime caches

GGUF payloads are mmap'd from disk and uploaded to the GPU; the engine then
quantizes its own compute caches once, at init:

- **FP8 E4M3 prefill cache** — per-tensor scale, consumed by cuBLASLt FP8 GEMM.
  Prefill is compute-bound, and FP8 tensor cores run at 2× FP16 rate on
  `sm_120a`. Controlled by `attention.fp8_prefill` / `--prefill-fp8` /
  `--no-fp8-prefill`.
- **NVFP4 decode cache** — FP4 E2M1 payload with FP8 E4M3 micro-scales (per 16
  elements) and an FP32 tensor scale. Decode is memory-bound, so a 4-bit weight
  read is a near-proportional decode win. Two modes:
  - **mode 1 (additive, `--decode-nvfp4`)** — FP8 prefill cache + NVFP4 decode
    cache both resident. Fastest overall; costs the extra VRAM of the duplicate
    cache.
  - **mode 2 (replacement, `--decode-nvfp4-only`)** — NVFP4 only. Saves VRAM,
    slightly slower decode; prefill runs from the NVFP4 cache.
  `--no-nvfp4` disables the decode cache entirely.

Micro-scales are chosen by `absmax` per 16-value block. The dominant NVFP4
error is the FP4 grid itself (eight magnitudes), which no scale choice
improves — quality cost against the Q8_0 source is small and is paid only on
the decode path.

## KV cache element type

The KV cache is **FP16**:

```toml
[kv_cache]
dtype = "fp16"
```

Paged, block size 16 tokens, with prefix caching. See
[`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md) for how the pool is sized.

## Choosing a quant

- **Q8_0** is the supported baseline: quality is effectively indistinguishable
  from FP16 on the models in [`supported-models.md`](supported-models.md), and
  the engine's own FP8/NVFP4 caches supply the speed.
- The engine's cache mode is the real knob: mode 1 (`--decode-nvfp4`) for
  decode-heavy interactive use, mode 2 (`--decode-nvfp4-only`) when VRAM is
  tight or prompts dominate wallclock.
