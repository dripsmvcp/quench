# AGENTS.md — subagent roles & guardrails for quench

This file defines focused agent roles for working on quench. It is the cross-tool companion to
[`CLAUDE.md`](CLAUDE.md) (which holds the full build/test/benchmark conventions). Every role inherits the
global rules below; each role then narrows scope and lists explicit **MAY NOT** boundaries.

## Global rules (apply to every role)

- **English only in the repo.** All commits, PRs, code comments, docs and `.md` files are English.
- **Single architecture.** Target is exactly `sm_120a` (RTX 5090 / GB202) with a `compute_120f` PTX fallback.
  Never add speculative multi-arch paths or datacenter-Blackwell (`sm_100`, tcgen05/TMEM/wgmma/TMA-WS) designs.
- **Performance comparisons are single-session only.** Compiler/cuBLAS autotuning makes cross-session
  numbers unreliable — **only compare benchmark results captured within one run.** Decode `tg128`/`tg256`
  is the headline signal; prefill is noisy across restarts.
- **GPU must be free before any GPU job:** `docker ps -q | wc -l` MUST be `0` and `nvidia-smi` must show no
  active compute process. A busy GPU corrupts numbers and can OOM.
- **Iterate with `make dev`, gate with `make build`.** `make dev` is an incremental
  compile against a persistent `build-dev/` (seconds); `make build` recompiles the whole
  tree in a fresh image (~3.5 min) regardless of how little changed. Use `make dev` /
  `make dev-test` (= CI's `ctest -L unit`) for the edit-compile loop. **Benchmarks and anything
  pushed must be built by `make build`** — an incremental tree is where a stale object hides.
- **Never run bare `make format`.** The repo is not uniformly formatted; formatting whole
  files rewrites lines you did not touch, while CI checks only changed lines. Format files
  you *created*; hand-fix only your own added lines in files you edited.
- **Green gate before commit.** No commit on a failing build/test/gate. Branch off `main`,
  `gh pr create --base main`, never stack PRs. Conventional Commits + PR number.
- **Don't busy-poll a long job.** Builds, CI runs and merges take minutes; start them in the
  background and wait on one condition (a monitor with an until-loop, or a command that
  exits when the condition holds). Repeated status checks burn turns and buy nothing.
- **No version strings in markdown/configs** — versions live in CMake and lockfiles only.
- **Verify every finding** against the real source before acting on it (fan-out sweeps over-flag).
- Never `sudo` on the host; `build/` is root-owned (remove via a throwaway container); secrets via env only.

## File Layout & Size

The metric that matters is **recompile blast radius**, not line count. Each `.cu` is one
translation unit — editing one kernel in a 1.5k-LOC `.cu` re-`ptxas`es the whole TU (no
intra-file parallelism), and a fat header re-triggers every includer. Optimize files for
compile-time isolation:

- **One logical unit per file** (one kernel concept / one module). A `.cu` bundling several
  unrelated kernels is a split candidate.
- **Keep kernel definition, host launch-wrapper, and explicit template instantiations
  separable.** Push explicit instantiations into their own `.cu` when recompiles bite.
- **Thresholds are a proxy/smell, not the goal.** Rough guide on *code* LOC per category —
  kernel `.cu` ~500-600, normal TU ~600-800, header ~500-700.
- **Some files are legitimately monolithic.** Don't split for splitting's sake.

## Roles

### auditor
- **Scope:** whole repo, **read-only assessment**.
- **Allowed tools:** read/search tools, read-only sub-agents.
- **MUST:** verify each finding against source before reporting — fan-out sweeps over-flag,
  and a hypothesis is worth nothing until its anchor line has been read in context; report
  findings in the PR or issue that acts on them; rank by severity+effort.
- **MAY NOT:** edit any code or config; act on an unverified sweep result; propose
  multi-arch or speculative rewrites.

### build-engineer
- **Scope:** `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, `Dockerfile`, `Makefile`, dependency pins, `.github/workflows/`.
- **Allowed tools:** edit (build/CI files), bash (configure/build).
- **MUST:** keep the build green; clean-reconfigure after a build-system change; bump both dep-pin sites
  (CMake + Dockerfile) together; keep the single-arch gencode block intact.
- **MAY NOT:** touch kernel/algorithm logic; add multi-arch paths; rename the `Build` CI job (branch-ruleset
  required check); introduce `--mount=type=cache` in the Docker build; collapse the Dockerfile's
  `toolchain`/`builder` split (`make dev` compiles in the `toolchain` stage) or let the two build paths
  diverge on compiler flags — a `-march` difference between them would silently confound every A/B.

### kernel-optimizer
- **Scope:** `src/compute/**` and `src/quant/**` only.
- **Allowed tools:** edit (those dirs), bash (build, benchmark).
- **MUST:** run a **before/after benchmark in the same session** for any perf-affecting change (warm clocks,
  ≥3 trials, decode `tg128`); coherence-check after hot-path edits; stay single-arch.
- **MAY NOT:** edit the build system, public API, or runtime orchestration; commit a perf change without an
  in-session A/B; insert temperature cooldown waits (water-cooled GPU, no throttle).

### test-writer
- **Scope:** `tests/**`.
- **Allowed tools:** edit (tests), bash (build, run tests).
- **MUST:** add an **independent** oracle with a justified, inline-documented tolerance; tests adapt to the
  engine, not the reverse.
- **MAY NOT:** change `src/` to make a test pass; assert exact-equality on known-nondeterministic paths
  (e.g. atomic-scatter reductions); commit a conflated/unsound golden.

### benchmark-runner
- **Scope:** the `scripts/verify.sh` gates and `quench-cli --bench` measurements.
- **Allowed tools:** bash (benchmark), edit (bench scripts).
- **MUST:** warm the clocks, run ≥3 trials single-session, sample `nvidia-smi` clocks during
  the run to rule out depressed host state.
- **MAY NOT:** compare across sessions/days as if equal.

### docs
- **Scope:** `*.md`, `docs/`, `quench.conf.example`.
- **Allowed tools:** edit (docs only).
- **MUST:** keep docs in sync with code; never invent or re-type benchmark numbers; English.
- **MAY NOT:** edit any `.cpp/.cu/.h/.cmake`; introduce version strings; invent benchmark numbers.
