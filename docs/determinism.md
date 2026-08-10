# Determinism

What quench guarantees about run-to-run reproducibility, what `[runtime]
deterministic` adds, and the documented limits.

## Shipped guarantees

`[runtime] deterministic = true` (legacy env `QUENCH_DETERMINISTIC=1`) is the
opt-in full-reproducibility mode for temperature=0 evals. It eliminates the
known run-to-run non-determinism sources by selecting deterministic kernel
variants:

- **Top-k sampling** — atomicMax/atomicAdd softmax-stat races (single-block
  path, `top_k <= 128`).
- **GEMM** — implies `deterministic_gemm` (cuBLASLt `no_reduce_split`;
  timing-based algo selection is itself a non-determinism source).

With it ON, the gated guarantees (`DetEvalE2ETest`) are:

- **Greedy output bit-identical** across runs in the *same context* and
  across *fresh processes*.
- **Perplexity NLL bit-stable** (`quench_perplexity` / `quench-cli --perplexity`)
  — the teacher-forced harness is the determinism-proof A/B instrument.

The mode is applied engine-side (effective through the C API and server, not
just the CLI tools). Costs a little throughput; strictly OFF by default —
the default path runs the exact same kernels with zero overhead.

## Default-mode guarantee

Without `[runtime] deterministic`, the DEFAULT path guarantees greedy
**request-order independence within a process**: identical greedy requests
produce identical output no matter how many requests preceded them. Three
pieces make this hold:

- `runtime.warmup` defaults to **true**: engine warmup pre-arms the decode
  graph pool, so the first real request starts with the same graph state as
  every later one.
- `CudaGraphRunner::mark_process_warm()` — keeps warmup's teardown from
  resetting the per-runner eager pre-capture step; a reset step would
  execute only in the FIRST real request: one step on a numerically
  different kernel mix (eager vs captured graph), flipping greedy output
  on near-tie logits.
- Scheduler gates use `graph_path_available()` instead of `is_ready()` —
  gating loop/pipeline entry on `is_captured()` would defer those paths by
  one step on the first request only.

The failure mode this protects against is greedy nondeterminism at temp=0
that shows as 1 divergent + N identical runs, always the first request
of a process. Verified: 3 fresh server processes x 12 greedy
requests — 36/36 byte-identical (even across processes, though
cross-process stability additionally depends on cuBLAS algo selection; see
`deterministic_gemm` for the hard guarantee).

`runtime.warmup=false` skips the warmup cost and reintroduces the
first-request asymmetry — acceptable for dev/CI, not for evals.

## Known limits

These are the documented boundaries of the guarantee. They are deliberate
(perf or upstream-API constraints) and live here so
they are not only discoverable as code comments.

### 1. Dense greedy logit ties

Exactly-tied logits can resolve to different argmax tokens across kernel
paths and runs — the FP values are bit-identical, but tie-breaking is not
specified across paths. Greedy-token A/B comparisons on tie-heavy prompts
(synthetic lists, repetitive corpora) are therefore **invalid as a
correctness signal**; use teacher-forced NLL instead (this is why
`ChunkedPrefillTest` gates on NLL rather than byte-equality).

### 2. CUB top-k is not tie-stable for `top_k > 128`

`src/compute/sampling.cu` (`DeviceTopK::MaxPairs`): the CUB path runs with
`determinism::not_guaranteed`, and the descending radix sort is not
guaranteed stable on the token index for bit-identical probabilities — two
tied tokens can swap order between runs. The single-block path
(`top_k <= 128`) tie-breaks by index and is fully deterministic. Fix path if
ever needed: fold the vocab index into the sort key (`(prob, -index)`) or
request `determinism::guaranteed`.

### 3. `typical_p` shared-memory FP atomicAdd

`src/compute/sampling.cu` (bucket histogram): per-bucket probability mass is
accumulated via shared-memory FP `atomicAdd`, whose ordering is
scheduling-dependent. Under deterministic mode this remains a documented
exception — `typical_p` is not part of the temp=0 eval surface.

## Recipe: reproducible evals

```ini
# quench.conf
[runtime]
deterministic = true
```

- Compare **teacher-forced NLL** (`quench-cli --perplexity`), not greedy bytes.
- temperature=0 / greedy only on prompts without logit ties, `top_k <= 128`.
- `quench-cli` logs to stdout — strip log lines before hashing output.
