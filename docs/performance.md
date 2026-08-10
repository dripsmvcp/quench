# Performance

**No measured numbers are published in this document.** Measure on your own
hardware with `quench-cli --bench`; this page documents methodology and known
behaviour only.

## Methodology

| | |
|---|---|
| Hardware | Single NVIDIA RTX 5090, 32 GB GDDR7, `sm_120a` |
| Toolchain | CUDA 13.3, CUTLASS, GCC 13.3, Release Docker build |
| quench config | FP8 prefill weight cache + NVFP4 decode cache, FP16 KV, CUDA Graphs on |
| Sampling | Greedy (temp = 0) |
| Repetitions | ≥3 (decode); prefill varies up to ±2.6× across container restarts (cuBLAS algo selection) |
| Reported | Mean; decode (`tg256`) is the reliable A/B signal |

Compare benchmark results only within one session: cuBLAS algorithm selection
and host/driver state make cross-session prefill numbers unreliable.

**Bench-mode caveat**: `--bench --max-tokens 128` sizes the engine to the bench
workload, so it does not measure the served regime (`quench-server` defaults to the
model's full context). Use `quench-cli --prompt` or a real server request for production
numbers. Bench-mode KV sizing does not change what is left for the NVFP4 cache:
the weight caches are built *before* the KV pool and the pool takes the measured
residual, so the cache budget does not depend on how much KV was allocated first.

## Known behaviour (qualitative, verify by measurement)

- **Prefill variance**: cuBLAS autotuning can cause up to 2.6× variance in prefill numbers between container restarts. Decode is stable — compare decode only for reliable A/B testing.
- **Decode is memory-bound** at batch=1: the NVFP4 decode cache raises throughput by shrinking weight traffic, not by adding compute. See [`sm120.md`](sm120.md).
- **CUDA Graphs** deliver an integer-factor decode win over per-step launch; `scripts/verify.sh` gates the ON/OFF ratio so a silently broken capture is caught.
