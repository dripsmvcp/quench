# Changelog

All notable changes to quench are documented here. One to three lines per
entry: what changed for the reader, plus the number that makes it checkable.

## [Unreleased]

- docs(readme): publish first comparative benchmark — Mistral-Nemo-12B on RTX
  5090, decode tg128: quench 188.5 tok/s (226.6 with n-gram spec) vs llama.cpp
  118.2 vs vLLM-BF16 ~65.5; same Q8_0 GGUF for quench/llama.cpp, pp512/tg128 ×3.
- feat(eval): trust-minimized PR evaluation — same-session A/B on the rented
  5090 with a base-pinned harness (`bench/`, `tests/`) and taint policy, a
  deterministic scorer whose verdict is recomputed inside an Intel TDX machine
  (polaris.computer DCAP receipt), and durable evidence via `quench/eval`
  commit statuses + `refs/notes/quench-eval`. Gates on `tg` only, ±1.5% bar.
  Design: `scripts/pr_eval_bot.py` docstring; offline check:
  `scripts/verify_receipt.py`; contributor contract: CONTRIBUTING.md
  ("What happens to your PR").
- fix(eval): the model-integrity check was a no-op. `file_hash` ran
  `sha256sum X | cut -f1`, so the remote shell reported *cut's* status — a
  missing or unreadable model returned rc=0 and an empty digest, and the
  before/after comparison then asked `"" != ""` and passed. Verified against a
  box with no model staged. Now unpiped, with a 64-hex-char shape check.
- feat(eval): PRs from known contributors are evaluated with no human in the
  loop; the 5090 opt-in tick is now needed only on a first PR (a spend guard —
  the bot is cron, not a workflow, so GitHub's first-time-contributor approval
  never applied to it). The PR template now offers that tick, which it never
  did, so the opt-in was previously unreachable.
- fix(eval): `pr_eval_cron.sh`'s own install snippet pointed at a different
  repo's path and filtered on the bare basename, so running it re-installed
  that repo's cron line and never quench's. No quench PR was ever evaluated on
  a schedule.
- feat(eval): the measured verdict now decides the merge. `eval:pass` and
  `eval:noise` arm GitHub squash auto-merge for any author; the bar moves
  2.0% → 1.5% (`scorer.NOISE_PCT`, bidirectional — it is the reject bar too).
  A taint now caps `eval:noise` as well as `eval:pass`, and `cmake/`,
  `Dockerfile`, `docker-entrypoint.sh`, `docker-compose.yml` join the tainting
  set: unmeasured by the A/B, so they must not ride a green status into an
  unattended merge.

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
