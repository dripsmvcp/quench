# The KV clamp and the pending library reserve

Incident analysis behind the fix in `engine_kv_cache_init.cpp` /
`vram_query.h` (branch `fix/kv-clamp-library-reserve`). Found onboarding
Mistral-Nemo-12B Q8_0 (GGUF, dense, 12.4 GiB weights) on a rented RTX 5090
running **native Linux** — the first host outside WSL2/WDDM this engine was
exercised on, which is why both defects had stayed invisible.

## Symptom

Default configuration (`quench-cli --model <nemo-q8_0>`, no flags): the card
is driven to **2 MiB free**, warmup graph capture OOMs
(`dequant_gpu.cu:781 … out of memory`), the CUDA context is poisoned, and
output degenerates (`? I color, it is blue. I color…`). `quench-server`
collapses harder: thousands of `attention_paged.cu … unknown error` per
second, every request `internal_error`. `vram.reserve_floor_pct=20` changes
nothing. `--vram-budget 28000` serves cleanly (31/32 degen-suite checks).

## Root cause 1: the residual clamp sizes the pool into the pending claim

The KV pool takes the measured post-cache residual, clamped by
`kv_blocks_from_residual(free_now, headroom, …)` — where `headroom` was only
`vram_allocator_headroom` (5 % ≈ 1605 MiB on 32 GiB). But the **library
reserve is claimed AFTER the clamp**, on the first forward
(cuBLAS/CUTLASS, A1.5), and `free_now` still contains those bytes. Whenever
the clamp binds, KV consumes the claim's space:

```
free after caches   11305 MiB
clamp kept          1605 MiB   (allocator headroom only)
KV pool             9700 MiB   (3880 blocks)
first forward claim 2297 MiB   → deficit → warmup OOM → poisoned context
```

The clamp binds here because the shadow plan (A7 step 2, APPLIED) grants the
optional weight caches only the residual above one full-`max_seq_len`
sequence; with the auto `max_seq_len` (itself sized from VRAM) that grant is
~0, the cache phases build the full 6.9 GiB anyway from their own live-free
budgets, and the plan's KV number (5860 blocks) lands far above what fits.
The plan/build disagreement is pre-existing and acknowledged in
`plan.h`'s header; the clamp is the designed backstop — it just protected
the wrong amount.

**Fix**: `kv_blocks_from_residual` takes an explicit `pending_reserve_bytes`
(non-defaulted, so no future call site can silently skip the question) and
the call site passes the same library-reserve figure the shadow plan
charges. Regression tests with the incident's numbers:
`tests/test_kv_residual_sizing.cpp` (2961 blocks, not 3880).

## Root cause 2: the measurement ratchet

The library-reserve measurement (persisted per model/quant/CUDA key) records
`max(forward-window, whole-init-unattributed)` — **measured during the
errored warmup**. The claim is opportunistic: under pressure the first
forward claims whatever is left, so the recording grows every start:

```
start 1: records 2297 → start 2 reserves more, frees more, records 7348
start 3: records 9647 → KV pool collapses (940 blocks on a 32 GiB card)
```

**Fix**: recording is gated on `log_error_count() == 0`
(`core/log_stats.h`, counted in the logging sink) — a measurement taken
during an errored init reflects a degraded forward, not the libraries'
demand, and is discarded loudly.

## Verified

CPU lane green (`ctest -L unit`, 16/16 in the sizing suite). On the 5090:
default CLI now answers correctly at full speed (tg 185 tok/s, graphs on),
the clamp log shows `1605 MiB allocator headroom + 3900 MiB pending library
reserve kept`, and the gate refuses to persist errored measurements.

## Still open on native Linux (out of scope here)

The warmup error spam itself remains on this host class. Evidence:

- The engine's own mismatch probe measured the first forward's settled claim
  at **6826 MiB** on native Linux CUDA 13.3 (`ubuntu2204` apt runtime) vs
  the 3900 MiB constant measured on WSL2 — but no pin silences the spam:
  errors scale inversely with the reserve (493 at the 3900 constant, 287 at
  7000, 92 at 11 000) and never reach zero. The transient demand during the
  claim exceeds any settled value; the failed probes surface as sticky
  launch errors (WDDM absorbs them by paging; native Linux does not); the
  engine clears them and batch-1 CLI output stays correct at full speed.
- The one historically "clean" server run (`--vram-budget 28000`,
  `max_seq_len=8192`, 0 errors, 31/32 checks) turns out to have been the
  **measurement ratchet in disguise**: its plan charged a poisoned-high
  10 715 MiB library reserve accumulated by earlier errored starts, which
  accidentally held back enough VRAM for the whole transient grab. On a
  fresh measurement cache the same config spams (170 errors) and the suite
  degrades under load — there is currently NO reliably clean server
  configuration on this host class, only correct-but-noisy batch-1 CLI.
- The mismatch warning's advice to pin the *settled* claim (6826) is
  therefore misleading on native Linux: the transient peak is what needs
  the room, and it is larger than anything the probe reports.

The structural remedy worth discussing: claim the libraries **eagerly at
init, before the KV pool is sized** (a dummy forward-shaped GEMM per path),
which turns the pending-reserve bookkeeping — constant, measurement cache,
and this clamp parameter — into a measured fact the residual simply
reflects. That is a maintainer decision; this fix stops the two ways the
pending claim was being destroyed.
