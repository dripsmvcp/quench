# sm_120a Known Issues, Dead Ends, Root-Cause Reference

Heavy reference for the `sm120-cuda-expert` skill. Things that don't work, version-dependent gotchas, and root-cause constraints the current hot-path kernels depend on.

---

## Pre-flight before non-trivial kernel work

1. Check the dead-ends list below — many "obvious" optimizations are proven failures on sm_120.
2. If the installed CUDA version differs from when a dead end was tested (check `nvcc --version`), see "Version-dependent dead ends" — some are worth retrying.

For small edits (parameter tweak, kernel-signature change, fusing two existing kernels), skip pre-flight and go straight to the patterns.

---

## Version-dependent dead ends (worth retrying when CUDA version changes)

> **CUDA 13.3 re-test (PTX ISA 9.3): NO new sm_120a capability.** Full
> `ptx_survey_all.sh` at `compute_120a` under 13.2 vs 13.3 = **0 of 247 instructions
> flipped** (none unlocked, none regressed). The "retry on CUDA 13.3+" rows below were
> re-probed and stay ❌. sm_120's ISA surface is silicon-fixed; toolkit bumps don't add
> tcgen05/wgmma/TMA.
> 13.3's value is tooling (CUDA Tile C++, CompileIQ) + cuBLAS perf, not instructions.

| Dead end | Blocked by | Retry on |
|----------|-----------|----------|
| cuBLASLt grouped layout sm_120 | Zero algorithms for consumer Blackwell | New cuBLAS release (check algorithm count) |
| CUTLASS TC GEMM at M=1 | Activation quant + TMA overhead | ~~CUTLASS 4.5+~~ pin is v4.5.2 (see `cmake/quench-deps.cmake`) — retry condition met but NOT re-probed yet. Note: roofline measurement puts decode GEMV at 66–70% HBM with a 4-bit-dequant co-limit, so the upside is bounded. |
| `cp.async.bulk` with `.ignore_oob` | Requires TMA descriptor rewrite | ~~CUDA 13.3+~~ still ❌ on 13.3 — TMA not on sm_120; next major |
| `st.async .b128` to global | PTX 9.2 only targets `shared::cluster` | ~~New PTX ISA~~ still ❌ on PTX ISA 9.3 (13.3) |
| CUTLASS NVFP4 sm_120 graph-determinism | Universally non-deterministic for `cudaGraphExecUpdate` re-capture | Future CUTLASS NVFP4 deterministic mode |
| Native FP4 GEMM faster than dequant→cuBLAS on sm_120 | Marlin-style dequant→cuBLAS measured *faster* than FlashInfer-CUTLASS-NVFP4 on consumer Blackwell. Storage-format wins (4× VRAM) but compute-speed parity is open. | Future custom kernel or upstream CUTLASS fix |

---

## Resolved (no longer dead ends)

- **PTX `cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2}`** and reverse direction work on both `sm_120f` and `sm_120a` under CUDA 13.2.1. Correct usage routes the FP4 byte through a `.b8` register — see `references/ptx-patterns.md` "FP4 ↔ FP16/FP32/BF16 packed conversion". SASS confirms hardware emission: `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`.

- **Build target `sm_120a`** — under older toolkits a `ptxas` C7600 bug forced the `120f` workaround. As of CUDA 13.2.1 the `a` arch suffix is the correct target — superset of `120f`, adds `mma.sync.kind::mxf4nvf4.block_scale` and TMA-WS-Grouped-GEMM.

- **CUDA Graphs + prequant-NVFP4 MoE work.** The MoE decode fast-path (`executor_forward_moe.cu:524`) is fully device-side, no D2H sync — graph-safe. Verified across Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 (all +193%–234% decode vs `--no-cuda-graphs`). GGUF MoE prefill paths still use D2H sync, but prefill isn't graph-captured anyway. Hybrid Mamba2 (Nemotron-H) does NOT benefit yet — SSM layers don't fast-path.

- **SSM dispatch via the NVFP4 cache.** `ssm_in`/`ssm_out` are registered in the `cutlass_nvfp4_cache` so GDN/SSM weights hit the fast NVFP4 GEMM path. Shows +95–376% decode on Qwen3.5/3.6 GDN families — but the gain comes from CUDA Graph capture *enabled by* the faster GEMM, not the GEMM speedup itself. **Always re-bench graphs ON after a hot-path kernel change.**

---

## Load-bearing root-cause fixes (don't regress these)

The current kernels assume these fixes are in place.

| Fix | Symptom if regressed | Where |
|-----|----------------------|-------|
| **FP8 FMHA S_tile pointer advance** | Long-context cliff at prompt > 1024 tokens | `attention_fmha_sm120.cu` — pointer must advance with `sizeof(half)`. Regression test in tree. |
| **Qwen3.5/3.6 GDN `__launch_bounds__(HD,1)` not `(HD,2)`** | HD=128 GDN miscompile, garbage output | GDN kernel — keep `(HD,1)`. |
| **Qwen3.5 partial RoPE pair offset `+ rope_pairs` (not `+ head_dim/2`)** | Sister bug to launch_bounds; partial-RoPE corruption | RoPE kernel. |
| **Qwen3.5 Q8 α/β qtype consistency** | Pre-dequanted Q8→FP16 without updating qtype → dispatcher mis-interprets bytes → state collapse | `upload_weight` path — keep qtype tag in sync with stored bytes. |
| **Qwen 3.6 h_state FP32 gate + PyTorch L2 norm** | NaN at L38 in GDN | h_state must be FP32, not FP16/BF16. |
| **Gemma-4 per-layer `rope_freqs` for non-SWA layers, `n_rot=hd`** | L13/L14 drift 11–15% (vs <2% correct) | Pass per-layer rope_freqs through. |
| **MoE expert-offload auto-probe at 10% before falling back to 30%** | Qwen3-Coder-30B Q6_K decode 234 → 77 tok/s | MoE offload path (`src/exec/executor_forward_moe*.cu` / `expert_cache.cu`, config `moe.expert_overhead_pct=10`) — keep the 10% probe. |
| **L2 access-policy window `num_bytes` clamp to `cudaDevAttrMaxAccessPolicyWindowSize`** | Silent CUDA error / IMA on 5090 (128 MiB max) | `set_l2_streaming` / `set_l2_persist_kv` in `runtime/`. |
| **NVFP4 dequant graph-safe fallback** | `cudaMallocAsync` inside captured graph crash | `set_nvfp4_dequant_workspace()` + capture-guard in `ensure_dequant_buffer`. |
| **Weight caches built BEFORE the KV pool** | Card ends at ~0 MiB free → WSL2/WDDM spills into host memory → ~7× decode collapse with no error. Nothing fails; bandwidth just drops ~1530 → ~237 GB/s | `src/runtime/engine_kv_cache_init.cpp` — caches (bounded by the model) first, KV pool takes the **measured** residual. Sizing KV from an *estimate* of cache demand breaks this. |
| **No D2H of MoE host-args under graph capture** | IMA on capture; WSL2 compute-sanitizer can't diagnose it | Hybrid-capture foundation — MoE args must stay device-side in captured regions. Sister hazard: da_cache stack-UAF. |
| **Deterministic-GEMM cuBLAS algo is warmup-validated** | Intermittent `status 14` mid-run; void-GEMM continues on garbage | Det path must not take `results[0]` blindly; total algo failure THROWS. |
| **Explicit MXFP4→FP16 decode-fallback VRAM reserve on GDN hybrids** | token-0 `!` garbage (silent alloc failure) | VRAM planner reserves the fallback up front + fail-loud — see `quant-formats`. |

---

## Performance-relevant scaling rules

| Rule | Source |
|------|--------|
| Decode at batch=1: launch overhead first, memory second | Three Laws #1 in main SKILL.md |
| `__launch_bounds__` cost on regular paths: -4.5% to -20% | Repeated benchmarks |
| `mxf4nvf4.block_scale` raw MMA: 2.60× over f8f6f4 | `mxf4nvf4_mma_bench` |
| CUDA Graph decode on prequant NVFP4 MoE: +193% to +234% | Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 |
| pp512 cuBLAS-autotune variance: up to 2.6× across container restarts | Use `tg256` for A/B; ≤5% prefill-kernel deltas need nsys per-kernel sums, not end-to-end pp — see `benchmark-cuda` skill |
| FP4 `mma.sync` measured peak ≈ 2,019 TOPS (~½ datasheet); f32-accumulate = ¼ rate | TC-rate calibration |

---

## Negative results (don't repeat)

- **Generic `compute_120` PTX fallback.** Lacks FP8 MMA + block-scale. Always pin `compute_120a/sm_120a`.
- **FP8×FP8 cuBLAS prefill on sm_120.** Disabled by default: cuBLAS FP8 returns `NOT_SUPPORTED` at non-aligned M on consumer Blackwell (`engine_init_resolver.cpp:156`, config `attention.fp8_prefill`). Prefill levers are the FA2 family instead.
- **NVFP4 on GDN in/out projections.** REGRESSES −9 to −20% on wide GDN shapes — FP16 wins there; the byte-aligned `gemm.fp8_ssm_proj` sidecar is the shipped answer (native +19%, GGUF-Q8_0 +21%); `gemm.nvfp4_attn_proj` remains a measured opt-in exception.
- **Occupancy raise / KPAR→MR reroute on the NVFP4 decode GEMV path.** Refuted by an nsys+ncu roofline sweep — decode plateau is a 4-bit-dequant co-limit (L1TEX 91%), not occupancy.
- **Batch-1 MoE decode GEMV beyond 30% roofline.** Structural: shallow grids (1.5–2 waves) + tiny K (1.5–4 loads/lane); occupancy is already HIGHER than dense. `moe.mr_nr` is saturated — NR=4 +0.9%, NR≥16 regresses. Don't re-pursue.
- **MoE grouped-GEMM (NVFP4 prefill) beyond 41% roofline.** Structural: grid=170 (1 wave), 23% occupancy, per-expert M≈32. `moe.nvfp4_smallM` REGRESSES vs the device-args default (which is +25–32%) — keep it OFF. The NVFP4 prefill lever is attention, not grouped GEMM.
- **Q8-IMMA occupancy/fetch tuning.** Three refuted attempts. Also: per-launch **workspace memos poison IMMA perf**, and f32-accumulate quarters the TC rate.
- **Re-enabling tiled SWA at hd=256 (gemma-3).** Tiled kernels DO support hd=256, but gemma-3-on-cuBLAS is a deliberate SWA-correctness anchor (tiled hd256+window = PPL 42 vs 1.0). Blocked on correctness, not wiring.
- **FA2 occupancy work without smem surgery (Bq=64 etc.).** FA2 is tensor-pipe-busiest and smem-capped at 16.7% occupancy; the shipped levers are `fa2_f16acc` and the Bkv=32 underfill variant.
- **Split-D warp-pairing for the HD=256 FA2 instance.** Two warps per 16-row tile, each owning one D half: a_frag/O regs halve (228→138, zero spills), warps/SM double (4→8), the partial-S exchange even rides the existing TWOSLOT mid-loop barrier (zero extra syncs). Measured (Sq sweep 512/1024/2048/4096, 8Q/2KV): **slower EVERYWHERE, +10–16% — including the grid-underfill band** (64 CTAs on 170 SMs). The shipped 4-warp/228-reg instance is not latency-limited: the long unrolled MMA chains with software-pipelined ldsm supply enough ILP per warp; split-D pays replicated softmax (×2/CTA), smem exchange traffic, and HALVED per-warp MMA chain length (less ILP) with no offsetting win. Occupancy raises on this kernel are dead — the remaining hd=256 levers are algorithmic (Bkv tiling, smem layout), not warp count.
- **FP4-precision attention — the whole family is CLOSED.** Three independent refutations: an NVFP4-attention spike, a ThriftAttention promotion gate, and paged FP4-QK KV-append quant. Attention stays FP16/FP8; don't reopen without new hardware. (Related: hd=128 NVFP4 A/Bs need `attention.fa2_fp16qk=never` or you measure the wrong path.)
- **KPAR-GEMV paired-microblocks tuning.** PDL (programmatic dependent launch) already overlaps the grid-end/launch latency the pairing tried to hide — measured ±0. Don't re-derive.
- **Launch-elimination levers on the graphs+PDL decode loop — the whole class.** The roofline lever list is built from `--no-cuda-graphs` profiles; under the shipped conditional-graph+PDL decode the grid-(1,1,1)/launch-latency classes (moe_routing, rmsnorm, rope, kv_write, elementwise) largely overlap away — on Qwen3-30B-A3B the no-graphs kernel-time sum is ~1.8× the real graphs-ON step. Two direct refutations: (a) a multi-block fused gate-GEMV+top-k kernel (bit-identical outputs, −2 launches/layer, killed the 6.9%-share topk_gating launch) measured **0% e2e**; (b) capping decode split-K to kill the "5.8 µs reduce at 40 GB/s" REGRESSED −21…−35% (Q4KM 324→211 at cap=1) — split parallelism feeds 170 SMs from batch×heads=32; the auto policies are right even at ctx 512. Rule: a decode lever must hold real BYTES or critical-path math (dtype shrink, fewer bytes moved, faster big-GEMV) — validate launch/latency-class ideas with a graphs-ON e2e A/B before writing any kernel.
- **C++23 `[[assume]]` in NVFP4 GEMV.** Byte-identical SASS = provably inert. General rule: **SASS-diff (`cuobjdump -sass`) before any "perf-neutral" or "should help" claim** on compiler-hint changes — it settles the question in seconds.
- **Async `wgmma` / `tcgen05` / TMEM on consumer Blackwell.** Not available — SM100 (B200) exclusives. sm_120 peak path is register `mma.sync`. (Note: the *synchronous* `nvcuda::wmma` API *does* compile on sm_120 but lowers to **HMMA** — it is not async wgmma and not the peak path; it costs extra smem traffic and a smem round-trip vs hand-written `mma.sync` with register-resident fragments.)
- **Materializing the attention score tile (S/P) in shared memory.** A FA-style kernel that writes S to smem, runs softmax over smem, then reads P back for the PV MMA becomes **barrier- / L1-TEX-bound** (tensor cores idle, compute util in the teens) — the smem round-trip + `__syncthreads` dominate, not the MMAs. True FA2 keeps row max/sum and the S/P fragments **register-resident** and fuses softmax into the QK→PV handoff. Don't trust a kernel header that *claims* register-based softmax — verify against the code (some in-tree kernels are mislabeled).
- **`__noinline__` on device inner-loop helpers.** Spills to Local Memory (DRAM). Use `__forceinline__`.
- **`reinterpret_cast` on Q8_0 blocks.** 34-byte blocks NOT 4-aligned. Use `memcpy()`.
- **Skipping graph re-bench after a hot-path patch.** Compute speedup alone often shows ~0% in tok/s — the win is graph-replay-mediated. Always re-bench graphs ON.
- **Increasing SMEM beyond `cudaDeviceProp::sharedMemPerBlockOptin`** assuming H100's 228 KB. RTX 5090 max is ~99 KB.
