# quench

## What is quench

A C++23/CUDA inference engine that targets exactly one architecture: the NVIDIA RTX 5090 / RTX PRO 6000 (GB202, `sm_120a`). The build emits raw `sm_120a` SASS via direct gencode: no portability layer, no wrapper around llama.cpp or vLLM. A `compute_120f` PTX fallback covers the other Blackwell consumer SKUs (RTX 5080 / 5070 Ti).

The scope is deliberately narrow: **dense llama-family transformer models from GGUF checkpoints** (Llama, Mistral, Qwen3-dense and friends). quench ships its own GGUF parser, BPE tokenizer, Jinja2 chat-template engine, paged KV cache, hand-written attention kernels, an NVFP4 decode weight cache, CUDA-Graph decode loop, speculative decoding, and an OpenAI/Anthropic-compatible HTTP server with tool calling and constrained decoding.

## Why this exists

The RTX 5090 shipped with native FP4 tensor cores, 32 GB GDDR7 at 1.8 TB/s, and a new ISA (`sm_120a`) that no existing inference engine fully exploits. llama.cpp targets broad hardware compatibility and runs GGUF through dequant-to-FP16 paths: fast everywhere, but leaving consumer-Blackwell-specific features (NVFP4 block-scaled `mma.sync` MMA, FP8 `f8f6f4` scores) on the table. vLLM targets datacenter Blackwell (B200/B300, `sm_100`) and gates key backends on `tcgen05`, an opcode family that consumer Blackwell (`sm_120`) doesn't have.

Consumer Blackwell is **not** a smaller datacenter Blackwell: `sm_120a` has no `tcgen05`, no TMEM, no `wgmma`, and no TMA warp-specialized grouped GEMM. quench is built against that reality: the FP4 path is `mma.sync mxf4nvf4` with FlashAttention-2-style block-scaling, not the Hopper/B200 kernel designs.

For GGUF models, quench builds an **NVFP4 decode cache** that converts Q8_0 weights to FP4 at init time, getting the bandwidth benefit of sub-byte weights on the decode hot path while keeping full-precision prefill via an FP8 weight cache. Decode is bandwidth-bound; halving the bytes the GEMV reads is the whole game on one card.

## Performance

First comparative measurement — Mistral-Nemo-12B-Instruct, batch 1,
synthetic pp512/tg128, 3-rep averages, each run verified exclusive on the
GPU (llama-bench methodology across all three engines):

Percentages are against llama.cpp on the identical GGUF file.

| Engine | Weights | Prefill pp512 | Decode tg128 |
|---|---|---:|---:|
| **quench** (default: n-gram speculator on) | Q8_0 GGUF → NVFP4 decode cache | 9,315 tok/s (−6%) | **226.6 tok/s (+92%)** |
| **quench** (`--set speculative.ngram=false`) | Q8_0 GGUF → NVFP4 decode cache | 9,553 tok/s (−4%) | **188.5 tok/s (+59%)** |
| llama.cpp (master `030ebb5`, `-fa 1`) | Q8_0 GGUF (same file) | 9,942 ± 1,133 tok/s (baseline) | 118.2 ± 0.2 tok/s (baseline) |
| vLLM 0.27.0 (FLASH_ATTN backend) | BF16 safetensors | ~8,438 tok/s (−15%) | ~65.5 tok/s (−45%) |

**+59% decode on the same file — and −4% prefill.** quench is faster where it
was built to be (decode, the bandwidth-bound path an NVFP4 cache attacks) and
marginally slower at prefill, which is compute-bound and where llama.cpp's
cuBLAS path is already well tuned. Prefill here also carries ±11% run-to-run
variance, so treat that 4% as a tie rather than a loss.

Measured 2026-08-11 on an RTX 5090 (32 GB, vast.ai), quench commit `4e958fd`
built with CUDA 13.3:

```bash
# quench (add --set speculative.ngram=false for the pure-decode row)
quench-cli --model models/Mistral-Nemo-Instruct-2407-Q8_0.gguf \
  --bench --bench-pp 512 --max-tokens 128 --bench-reps 3
# llama.cpp
llama-bench -m Mistral-Nemo-Instruct-2407-Q8_0.gguf -p 512 -n 128 -r 3 -fa 1
# vLLM (CUDA 13.0 torch; decode derived as 127/(t_out128 - t_out1))
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm bench latency --model unsloth/Mistral-Nemo-Instruct-2407 \
  --dtype bfloat16 --batch-size 1 --input-len 512 --output-len 128 \
  --num-iters 3 --num-iters-warmup 1 --max-model-len 2048
```

Read it honestly. Only the llama.cpp row is like-for-like — same GGUF file,
same harness. Two caveats on the other numbers:

- The **+92%** row has the n-gram speculator on. It is the default and is
  token-identical to greedy, but synthetic bench output is unusually
  draftable, so that figure will not hold on real prompts. **+59% is the
  number to quote.**
- **vLLM is not a like-for-like comparison** and is shown for context only. It
  has no usable Q8_0 GGUF path on `sm_120`, so it ran BF16 — a different
  weight format — and batch 1 on one consumer card is the regime it is least
  suited to. Its strength is throughput under concurrency, which this
  benchmark does not measure.

Most of the gap is bytes moved per token, which is the point of the NVFP4
decode cache — but the speedups are *smaller* than the byte reductions:
quench's decode overlay measures 5,850 MiB against llama.cpp's 12.12 GiB
Q8_0 and vLLM's 24.5 GB BF16, i.e. 2.1× and 4.0× fewer bytes for 1.6× and
2.9× the throughput. quench is the fastest of the three here and also the
furthest from its own roofline, so there is decode headroom left.

Decode is the reliable signal; prefill varies with cuBLAS autotuning across
container restarts (see Known limitations). This is one model on one card at
batch 1 — a reproducible harness is on the [roadmap](docs/roadmap.md).

## Should I use this?

**Use [llama.cpp](https://github.com/ggerganov/llama.cpp) if** you want a stable, mature engine with broad model/hardware support and CPU fallback. It runs everywhere and supports everything.

**Use [vLLM](https://github.com/vllm-project/vllm) if** you serve high-concurrency batched workloads on datacenter GPUs (H100/B200).

**Use quench if** you run a Blackwell consumer/workstation card (RTX 5090, RTX PRO 6000), your model is a dense llama-family GGUF, and you want a backend built specifically for that card: native NVFP4 decode, hand-written `sm_120a` attention, tool calling, constrained decoding, and prefix-cached multi-turn on one GPU. The trade-off is scope: single GPU, one model family, GGUF only. It is also young and only lightly measured — llama.cpp and vLLM are mature and extensively benchmarked, and this is not.

## Known limitations

- **Single GPU only.** No tensor parallelism, no multi-GPU.
- **Consumer Blackwell only.** `sm_120a` SASS + `compute_120f` PTX fallback. Only the 32 GB RTX 5090 is tested. No Hopper, Ada, Ampere, datacenter Blackwell. No AMD, Intel, Apple, or CPU paths.
- **Dense llama-family GGUF only.** No MoE, no state-space/hybrid architectures, no vision, no SafeTensors loading. A checkpoint outside the supported list may load but is not verified.
- **Minimal benchmarking.** One comparative measurement is published (see Performance) — single model, single GPU, batch 1, no reproducible harness yet.
- **Prefill timing is noisy.** cuBLAS autotuning causes substantial variance across container restarts, so prefill is a weak A/B signal; decode is the reliable one.
- **Single-user / agentic scope.** Built and tuned for single-user and moderate-concurrency inference on a 5090, not datacenter serving.

## Quickstart

Everything runs in Docker, no local CUDA toolkit needed:

```bash
git clone <this-repo> quench && cd quench

# Drop a GGUF model into ./models/
mkdir -p models

# Build and run the server. The cache volume is optional but pays for itself
# on the second start: it holds the transformed weights and the measured
# library reserve.
docker compose build quench-server
docker run --gpus all -v ./models:/models -v quench-cache:/home/quench/.cache/quench \
  -p 8080:8080 \
  quench:latest --model /models/your-model.gguf

# Hit the OpenAI-compatible endpoint (the model id is the file basename;
# GET /v1/models lists it, and the `model` field is required).
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model.gguf","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

Or open <http://localhost:8080> in a browser: the server ships a small chat UI
that streams the answer and plots inter-token latency while it is written.

CLI reference, server flags, config, and C API: [`docs/usage.md`](docs/usage.md).

## Supported models

| Family | Variants | Quantizations |
|---|---|---|
| Mistral | Nemo-12B, 7B | GGUF Q8_0 |
| Llama 3.x | dense | GGUF Q8_0 |
| Qwen3 | dense (4B/8B/14B/32B) | GGUF Q8_0 |

Q8_0 is the verified quantization; other GGUF quants may load through the
dequant paths but are not part of the tested set. VRAM and per-model notes:
[`docs/supported-models.md`](docs/supported-models.md).

## Features

| | |
|---|---|
| **Quantization** | GGUF Q8_0 source weights; FP8 prefill weight cache + NVFP4 decode weight cache built at init time. FP16 KV cache, paged (block_size 16). Auto context ceiling, VRAM- and model-bounded. |
| **Attention** | Prefill: FP16 cuBLAS below the auto `fmha_prefill_threshold`, then a register-resident FlashAttention-2 `mma.sync` kernel (head_dim 128) above it. Decode: paged attention with split-K. Auto-dispatch per phase × layer. |
| **`sm_120a` kernels** | NVFP4 block-scaled `mma.sync mxf4nvf4` GEMV/GEMM (CUTLASS), packed `cvt.e2m1`/`cvt.e4m3x2` dequant, PDL, Green Contexts. **No** `tcgen05`/TMEM/`wgmma`; those are datacenter Blackwell only. |
| **Server** | OpenAI `/v1/chat/completions` + `/v1/responses` + `/v1/completions`, Anthropic `/v1/messages` with per-token SSE streaming and `cache_control` prompt caching; tool/function calling; constrained decoding (`json_object`, `json_schema`, regex, GBNF grammars via a pushdown simulator); `reasoning_content` separation + think budget; prefix caching (model-fingerprint-gated on disk); Prometheus `/metrics`; API-key auth, rate limiting, JSONL request logging; client-disconnect cancellation; suspend-to-RAM (`/admin/suspend` / `/admin/resume`); model swapping on request. |
| **Speculative decoding** | n-gram / suffix / token-recycling drafters with a verify-chunk decode loop; per-request override (`"speculative": true/false`). |
| **Runtime** | CUDA Graphs (auto), `quench.conf` + CLI config, Jinja2 chat templates with macro support. `degen_suite.py` is the coherence quality-gate after hot-path changes. |

## Documentation

| Document | |
|---|---|
| [Memory architecture](docs/MEMORY_ARCHITECTURE.md) | Lifetime tiers, allocators, ownership invariants, VRAM planning |
| [Supported models](docs/supported-models.md) | Tested families with VRAM + notes |
| [Usage & reference](docs/usage.md) | Build, server, CLI, C API |
| [Quantization](docs/quantization.md) | GGUF Q8_0, FP8, NVFP4: formats and trade-offs |
| [sm_120a kernels](docs/sm120.md) | Kernel optimization notes |
| [Determinism](docs/determinism.md) | Reproducibility guarantees + known limits |
| [quench.conf reference](quench.conf.example) | All runtime configuration keys |
| [Roadmap](docs/roadmap.md) | Planned features beyond the 0.1.0 scope |
| [Changelog](CHANGELOG.md) | Per-release notes |

## Building from source

```bash
# With CUDA 13.3+ on the host:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Or via Docker (canonical):
make build
```

Full build options and test commands: [`docs/usage.md`](docs/usage.md). Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT, see [`LICENSE`](LICENSE).

## Acknowledgements

Stands on the shoulders of [llama.cpp](https://github.com/ggerganov/llama.cpp). The GGUF format, GGML quantization schemes, and most practical conventions for local LLM inference were established there.
