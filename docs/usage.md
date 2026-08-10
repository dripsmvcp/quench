# quench — Usage & Reference

Build instructions, CLI/server usage, configuration, C API, project structure.

---

## Requirements

- **NVIDIA Blackwell GB202** (sm_120a) — RTX 5090, RTX PRO 5000 Blackwell, or RTX PRO 6000 Blackwell. Same binary, same kernels; the workstation cards just have more VRAM (48 / 96 GB) for bigger models and longer context.
- **CUDA Toolkit 13.3** (13.2 minimum enforced by CMake; 13.3 is the canonical toolchain Docker and CI build with) — `cudart`, `cuda_driver`, `cublas`, `cublasLt`
- **CMake 3.25+**
- **C++20 compiler** (GCC 11+, Clang 14+)

CUTLASS v4.5.1 and Google Test v1.17.0 are fetched automatically via
`FetchContent`.

## Build

The canonical workflow is Docker via the Makefile (`make build` →
`quench:test`). Host builds also work when CUDA 13.2+ is installed natively.

```bash
# Host build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Docker build (canonical)
make build       # → quench:test image with full GPU passthrough
make verify-fast # build + filtered tests + graphs gate + smoke prompt
make verify      # full pre-merge gate (~5 min)
```

| CMake option | Default | Description |
|---|---|---|
| `QUENCH_BUILD_TESTS` | ON | GTest suite |
| `QUENCH_BUILD_TOOLS` | ON | quench-cli |
| `QUENCH_BUILD_SERVER` | ON | quench-server |
| `QUENCH_SANITIZERS` | OFF | ASAN + UBSAN (host C++ code only) |
| `QUENCH_ALLOC_INTERPOSE` | OFF | Wrap `cudaMalloc`/`cudaMallocAsync` to attribute steady-state allocations (diagnostic; costs ~3% decode, so never benchmark with it on) |
| `CMAKE_CUDA_ARCHITECTURES` | hard-pinned `sm_120a` | RTX 5090 / RTX PRO 6000 |

`sm_120a` SASS + `compute_120f` PTX fallback are set via raw `--generate-code`
in `CMakeLists.txt` (CMake < 3.31 workaround). Don't override
`CMAKE_CUDA_ARCHITECTURES`.

## Configuration — `quench.conf`

`quench.conf` is the runtime configuration interface — a sectioned
TOML-subset file.
See `quench.conf.example` in the repo root for the full schema with defaults
and inline comments.

**Loading precedence** (first non-empty wins):

1. `--config <path>` CLI flag
2. `$QUENCH_CONFIG` environment variable
3. `./quench.conf` (working directory)
4. `~/.config/quench/quench.conf`
5. embedded defaults (no file)

**Per-run overrides** on top of the loaded config:

```bash
quench-cli --set runtime.cuda_graphs=never --set speculative.ngram=false \
        --model X.gguf --prompt "..."
```

The most common keys are also exposed as named CLI flags
(`--no-cuda-graphs`, `--prefill-fp8`, …) for convenience.

## CLI — quench-cli

```bash
# Single prompt (GGUF)
./build/quench-cli --model model.gguf --prompt "Hello, world!"

# Interactive chat
./build/quench-cli --model model.gguf --interactive

# NVFP4 decode cache mode 1 (additive; usually the auto-default)
./build/quench-cli --model model.gguf --decode-nvfp4 --interactive

# Long-context prompt (trade weight-cache VRAM for KV headroom)
./build/quench-cli --model model.gguf \
                --min-kv-tokens 14000 --prompt "$(cat long.txt)"

# Benchmark (matches llama-bench methodology)
./build/quench-cli --model model.gguf --bench --bench-pp 512 \
                --max-tokens 128 --bench-reps 5
```

`--max-seq-len` and `--min-kv-tokens` control KV-cache VRAM reservation.
Auto defaults size the KV pool from the measured post-cache residual.
`--min-kv-tokens` overrides the defensive cap and trades weight-cache
capacity for more context. The budget planner's envelope itself is tunable
via quench.conf `[vram]`: `kv_fraction` (default 0.8 — the KV share of
post-reserve VRAM) and `reserve_floor_pct` (default 10 — the free-VRAM
headroom floor as % of total). See `quench.conf.example`.

To serve a context longer than the model's native window, inject RoPE
scaling at load via quench.conf `[rope]` (or `--set`), e.g. a native-32k
model at 128k:

```bash
quench-server --model model.gguf --set rope.scaling=yarn --set rope.factor=4
```

The override mirrors model-declared `rope_scaling` metadata (YaRN or
linear), raises the detected context window to `factor × orig_ctx`, and
is refused for per-dimension scaling tables and NoPE models. Quality past
the native window is the checkpoint's YaRN extrapolation quality — validate
on your workload.

`--vram-budget <mb>` (also `[runtime] vram_budget_mb` in quench.conf) hard-caps
this process's VRAM: every sizing decision — weight caches, KV clamp,
workspaces, upload gates — sees a virtual GPU of that size, so
multiple quench-server processes can share one card:

```bash
quench-server --model Qwen3-4B-Instruct-2507-Q8_0.gguf --port 8080 --vram-budget 9000 &
quench-server --model Llama-3.2-3B-Instruct-Q8_0.gguf --port 8081 --vram-budget 8000 &
```

The cap binds, but it is not exact — and the overshoot is measured rather than
guessed. Two charges sit outside the sizing
gates: the CUDA primary context (~1.7 GiB on this host, allocated before quench
takes its baseline snapshot, so no budget can cover it) and ~1.8 GiB of
dequant scratch, CUTLASS scale-factor buffers, pinned staging and workspaces
whose allocation sites don't consult the budget. Measured on Qwen3-8B-Q8_0,
`--vram-budget 16000` peaks at 19468 MiB — so leave ~3.5 GiB of real headroom
between the sum of budgets and the card. `--mem-report` prints the peak
against the cap and marks it `[OVER BUDGET]` when it exceeds it, so the gap is
visible rather than inferred.

Every term of the planner's reserve is a percentage of the (virtual) card,
while the cuBLAS/CUTLASS reserve claimed on the first forward pass is a
~3.9 GiB constant — so the reserve is floored at that measured charge (tune
with `[vram] library_reserve_mb`). A model whose weights don't fit the budget
fails cleanly at load instead of starving the neighbour, and a budget that
cannot hold a single `max_seq_len` sequence is refused at init — naming the
blocks available, the blocks needed and the MiB to add — rather than loading
and then failing every request.

<details>
<summary>Full CLI options</summary>

```
Model:
  --model <path>           Path to a GGUF model file, or a HuggingFace repo id
  --revision <rev>         HuggingFace revision when --model is a hub repo id
  --device <n>             CUDA device ID (default: 0)
  --gpu-layers <n>         Layers on GPU, -1 = all (default: -1)
  --config <path>          Path to quench.conf (overrides search-path)
  --set section.key=value  Per-run override (repeatable)

Generation:
  --prompt <text>          Input prompt
  --max-tokens <n>         Max tokens to generate (default: 256)
  --max-seq-len <n>        KV context ceiling in tokens (default: auto)
  --min-kv-tokens <n>      Minimum KV capacity in tokens (default: auto)
  --vram-budget <mb>       Hard per-process VRAM cap in MiB (default: 0 = uncapped)
  --mem-report             Print the full VRAM attribution table at init
                           (lifecycle checkpoints, per-pool notes, named
                           charges, own_peak vs the cap, residual)
  --interactive            Interactive chat mode
  --stop <str>             Stop sequence (repeatable, up to 4)
  --chat-template <t>      auto|none|chatml|llama2|llama3|nemotron|gemma|deepseek_r1|phi

Sampling:
  --temperature <f>        (default: 0.7)
  --top-p <f>              (default: 0.9)
  --top-k <n>              (default: 40)
  --min-p <f>              (default: 0.0, disabled)
  --typical-p <f>          (default: 1.0, disabled)
  --repeat-penalty <f>     (default: 1.0, disabled)
  --repeat-last-n <n>      Penalty window (default: 0, all tokens)
  --frequency-penalty <f>  (default: 0.0)
  --presence-penalty <f>   (default: 0.0)
  --seed <n>               -1 for random (default: -1)
  --dry-multiplier <f>     DRY penalty scale (default: 0.0, disabled)
  --dry-base <f>           DRY exponential base (default: 1.75)
  --dry-allowed-length <n> (default: 2)
  --dry-penalty-last-n <n> (default: 0, all)
  --mirostat <n>           0=off, 2=v2 (default: 0)

Performance:
  --kv-fp16                Force FP16 KV cache (the default)
  --prefill-fp8            FP8 weight cache for prefill
  --no-fp8-prefill         Disable the auto FP8 prefill weight cache
  --prefill-chunk-size <n> Max tokens per prefill chunk (default: per-arch)
  --decode-nvfp4           NVFP4 decode cache, mode 1 (FP8 prefill + NVFP4 decode)
  --decode-nvfp4-only      NVFP4 decode cache, mode 2 (saves VRAM, slower prefill)
  --no-nvfp4               Disable the NVFP4 decode cache
  --no-cuda-graphs         Disable CUDA Graphs
  --prefix-caching         Enable prefix caching in the CLI engine
  --streaming-kv           Streaming-KV attention (sinks + sliding window)
  --no-streaming-kv-auto   Disable streaming-KV auto-enable heuristic
  --stream-sinks <n>       Streaming-KV: number of attention-sink tokens
  --stream-window <n>      Streaming-KV: sliding-window size in tokens

Benchmark / eval:
  --bench                  Synthetic benchmark mode (warmup + timed reps)
  --bench-pp <n>           Prompt tokens (default: 512)
  --bench-reps <n>         Repetitions (default: 3)
  --perplexity <file>      Teacher-forced perplexity over a text file
                           (deterministic eval harness)
```

</details>

## Server — quench-server (OpenAI + Anthropic compatible)

`--model` is required at startup (a GGUF file path or a HF repo id).

```bash
./build/quench-server --model model.gguf --port 8080
```

Endpoints: `/v1/chat/completions`, `/v1/responses` (OpenAI Responses API —
the Agents SDK / Codex dialect; stateless, so use `store: false` and resend
the transcript in `input`), `/v1/completions`,
`/v1/models`, `/v1/messages` + `/v1/messages/count_tokens`
(Anthropic-compatible, streaming + non-streaming), `/tokenize`,
`/detokenize`, `/health`, `/props`, `/info`, `/metrics` (Prometheus),
`/admin/suspend`, `/admin/resume`, and a web UI at `/`.
Tool/function calling, constrained decoding (`json_schema` / regex / GBNF),
reasoning-content split, streaming usage stats, logprobs, and API-key auth
(`--api-key`) supported.

**Warm weight cache.** The first cold load of a model writes a cache file
(`<model-name>-<hash>.impwcache`) into `~/.cache/quench/warm` (override with
`[warm_cache] dir`) holding the converted weight buffers; subsequent starts
mmap it and skip the conversion work. Raw quant payloads are never
duplicated, so the cache stays small for raw-served GGUF quants. On by
default (`[warm_cache] enabled = false` to opt out); stale caches (changed
model file) are detected and ignored — delete the directory at any time. In
containers, mount a persistent volume at the cache dir; if it is not
writable, loads simply stay cold (INFO log).

**Suspend to RAM.** `POST /admin/suspend` drains in-flight requests, parks
the model weights in host RAM, and frees the GPU completely (with
`[suspend] device_reset` — the default — the CUDA context is reset too, so
`nvidia-smi` shows ~0 MiB for the process). `POST /admin/resume` restores
the weights from RAM (no mmap re-read, no requantization) and serves again
in seconds. Sessions/KV do not survive — only the weights stay warm. While
suspended, inference endpoints answer 503 and `/health` reports
`"suspended": true` (HTTP 200 — it is a deliberate operator state, not a
fault). Capture fails cleanly (507) when host `MemAvailable` is below the
snapshot size + `[suspend] host_ram_headroom_mb`.

**Model selection.** `/v1/models` lists the model the server is serving
(OpenAI semantics: the server exposes exactly what it can serve). Requests
must name that model — any other `model` value gets `404 model_not_found`.
With `--models-dir <path>` the server scans that directory for `.gguf` files
and can swap the loaded model on selection; without it, restart the server
with a different `--model` to switch.

**Context-window auto-detection.** The served context length is exposed in
the three conventions OpenAI-compatible clients already probe, so no
hard-coded table is needed: `/v1/models` carries vLLM's `max_model_len` and
llama.cpp's `meta.n_ctx_train` on the model object, `GET /props` returns the
llama.cpp `n_ctx` (top-level and under `default_generation_settings`), and
`GET /info` returns TGI's `max_total_tokens` / `max_input_tokens`. All three
report the same window (the engine-detected `max_seq_len`).

Server-only flags (not on `quench-cli`):

| Flag | Effect |
|---|---|
| `--host <addr>` | Listen address (default `127.0.0.1`) |
| `--port <n>` | Listen port (default `8080`) |
| `--max-batch <n>` | Decode batch / KV+workspace sizing (default 0 = auto) |
| `--models-dir <path>` | Directory to scan for `.gguf` models (auto-load on select) |
| `--api-key <key>` | Require `Authorization: Bearer <key>` on requests |
| `--metrics-require-auth` | Also gate `/metrics` behind `--api-key` |
| `--max-concurrent <n>` | Max simultaneous requests (default 64, 0 = unlimited) |
| `--rate-limit <n>` | Max requests/min per IP (default 0 = unlimited) |
| `--log-requests <path>` | Append per-request JSONL with prompt + response content + timing to `<path>` (opt-in; off by default) |
| `--reasoning-format <f>` | `deepseek` (default) or `none` — controls `<think>` channel handling |
| `--think-budget <f>` | Fraction of `max_tokens` reserved for reasoning (default 0.5, 0 = disabled) |
| `--request-timeout <s>` | Per-request timeout in seconds (default 300, 0 = unlimited) |
| `--max-input-tokens <n>` | Reject prompts longer than n tokens with HTTP 400 (default 0 = unlimited) |
| `--prefix-cache <path>` | Persist the prefix cache to `<path>` across restarts |

```bash
# The model id is the served model's name (= /v1/models data[0].id)
MODEL=$(curl -s http://localhost:8080/v1/models | jq -r '.data[0].id')

# OpenAI chat completion
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":64}"

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":true}"
```

Works with the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
model = client.models.list().data[0].id
for chunk in client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Hi"}],
    stream=True, max_tokens=64
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## C API

> The C API is consumable only from a **source build**: `cmake --install` stages
> `libimp.a` and the `include/quench/` headers, which you link against. The prebuilt
> runtime image ships only the `quench-server` / `quench-cli` binaries — not the
> static library or headers — so embedding the C API means building from source
> (or copying the lib/headers out of the Docker `builder` stage).

```c
#include <quench/quench.h>

QuenchModel model;
quench_model_load("model.gguf", QUENCH_FORMAT_GGUF, &model);

QuenchConfig cfg = quench_config_default();
QuenchContext ctx;
quench_context_create(model, &cfg, &ctx);

QuenchGenerateParams params = quench_generate_params_default();
params.max_tokens = 128;

char output[4096];
size_t output_len;
quench_generate(ctx, "The capital of France is", &params,
             output, sizeof(output), &output_len);
printf("%.*s\n", (int)output_len, output);

quench_context_free(ctx);
quench_model_free(model);
```

Token-level control via `quench_prefill` / `quench_decode_step`.

## Project Structure

```
quench/
├── include/quench/     Public C API (quench.h, config.h, types.h, error.h)
├── src/
│   ├── core/           Tensor, Buffer, Logging, Threading
│   ├── compute/        CUDA kernels (GEMM, attention, RoPE, LayerNorm, sampling)
│   ├── memory/         Driver backend, tier allocators (arena, block pool,
│   │                   scratch stack, graph slots), capacity planner,
│   │                   KV cache (paged)
│   ├── model/          GGUF loading, tokenizer, chat templates, weight upload
│   ├── quant/          FP8 / NVFP4 quant + dequant, quantised GEMM
│   ├── exec/           Executor (hardcoded transformer forward pass)
│   ├── runtime/        Engine, Scheduler, CUDA Graphs, PDL, Green Contexts,
│   │                   RuntimeConfig (quench.conf parser)
│   └── api/            C API implementation
├── tools/
│   ├── quench-cli/     CLI (interactive + single-prompt + benchmark)
│   └── quench-server/  OpenAI + Anthropic-compatible HTTP server
└── tests/              Google Test suite
```
