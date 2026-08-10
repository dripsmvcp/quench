# Changelog

All notable changes to quench are documented here. One to three lines per
entry: what changed for the reader, plus the number that makes it checkable.

## [Unreleased]

- feat(eval): trust-minimized PR evaluation — same-session A/B on the rented
  5090 with a base-pinned harness (`bench/`, `tests/`) and taint policy, a
  deterministic scorer whose verdict is recomputed inside an Intel TDX machine
  (polaris.computer DCAP receipt), and durable evidence via `quench/eval`
  commit statuses + `refs/notes/quench-eval`. Gates on `tg` only, ±2% bar.
  Design: `scripts/pr_eval_bot.py` docstring; offline check:
  `scripts/verify_receipt.py`; contributor contract: CONTRIBUTING.md
  ("What happens to your PR").

- fix(vram): the measured-residual KV clamp now keeps the pending library
  reserve free alongside the allocator headroom — on Mistral-Nemo-12B Q8_0 /
  32 GiB the pool no longer sizes into the first forward's claim (was: card
  driven to 2 MiB free, warmup graph capture OOM, poisoned context, garbage
  output). The library-reserve measurement is no longer persisted from an
  errored init (was: ratcheted 2297→7348→9647 MiB across starts).
  Analysis: docs/vram-clamp-library-reserve.md.

## [0.1.0]

Initial release.

- C++23/CUDA inference engine targeting NVIDIA consumer Blackwell (`sm_120a`,
  RTX 5090 / GB202) exclusively — native NVFP4 `mma.sync` GEMV/GEMM via
  CUTLASS, FA2 prefill attention, paged FP16 KV cache, CUDA-graph decode
  loop, speculative decoding (n-gram / suffix / token-recycling drafters).
- GGUF loader (dense llama-family: Mistral, Llama, Qwen3-dense) with an FP8
  prefill weight cache and an NVFP4 decode cache built from Q8_0 at init.
- OpenAI- and Anthropic-compatible HTTP server (`quench-server`): chat
  completions, responses, messages, tool calling, `json_schema` / regex /
  GBNF constrained decoding, reasoning-channel separation, prefix caching,
  model swapping, suspend/resume, Prometheus metrics, built-in live web UI.
- CLI (`quench-cli`).
