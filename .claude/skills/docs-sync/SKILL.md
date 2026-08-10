---
name: docs-sync
description: Use when keeping quench's docs and config examples coherent after a change — updating architecture.md / README / docs/GOAL.md / supported-models.md / quench.conf.example / CHANGELOG, or "is this doc stale", "document this change", "the example config is out of date", "the README says X but the code does Y". Do NOT use for structural code audits (codebase-audit), measuring perf or refreshing the perf baseline (benchmark-cuda), or the agent's private memory (MEMORY.md).
---

# Docs sync — quench

## Hard rules

1. **English only in the repo.** Every committed artifact — PRs, commits, code
   comments, `.md` docs — is written entirely in English.
2. **`quench.conf.example` MUST match the parser.** Every key in the example has to be
   a real `B/I/F/S("...")` binder in `src/runtime/config.cpp`; a key the parser
   dropped logs `quench.conf: unknown key` at load. When you add/remove/rename a config
   field: update `config.h`, `config.cpp`, AND `quench.conf.example` together, with the
   real default.
3. **Numbers are commit-anchored.** `tests/perf_baseline.json` is the canonical gate
   (3% decode / 5% prefill, plus 10% peak VRAM over `metrics.memory_mb.own_peak_mb` —
   that one is evaluated by `scripts/verify.sh`). Never publish a tok/s number without
   the commit SHA + CUDA version + the exact reproducing command.
4. **Verify before you claim.** Docs drift; grep the tree before citing a doc fact —
   half of all staleness reports describe something already fixed.
5. **Never document `QUENCH_*` env vars as config.** The only live env vars are
   `QUENCH_DETERMINISTIC` and `QUENCH_FMHA_FA2` (plus the trace knobs backed by
   `diagnostics.*` config keys); everything else is `quench.conf` / `--config` /
   `--set`. A doc or example suggesting another `QUENCH_*` var is a bug.

## The doc set + what each owns

| Doc | Owns | Touch it when |
|---|---|---|
| `docs/architecture.md` | Canonical narrative (the source of truth) | a refactor changes the high-level structure / data flow |
| `docs/MEMORY_ARCHITECTURE.md` | Memory subsystem: lifetime tiers, allocators, invariants I1–I7, acceptance criteria | ownership, lifetime, capacity or VRAM behaviour changes. `architecture.md` defers to it for anything memory-shaped, so don't re-narrate it there |
| `README.md` | User-facing pitch + quickstart | supported models, build steps, or a published headline changes |
| `docs/GOAL.md` | Mission, release bar, north-star metric + hardware scope | the goal or hardware scope changes |
| `docs/roadmap.md` | Current focus + open gaps | an open item closes or a new gap is identified |
| `tests/perf_baseline*.json` | The perf **and peak-VRAM** gate (canonical) | a change intentionally moves perf or peak VRAM — refresh via `scripts/gen_perf_baseline.sh` (it re-pins `own_peak_mb` too) |
| `docs/supported-models.md` | Supported architectures/models | a new arch/model lands or a quant is dropped |
| `CHANGELOG.md` | Notable changes | user-visible behavior changes |

## When a change lands, sync the matching doc

- **Perf moved** (intentionally) → refresh `perf_baseline.json` via
  `scripts/gen_perf_baseline.sh` and **say so in the PR**.
  Re-bench properly first (REQUIRED SUB-SKILL: benchmark-cuda — clock ramp + host
  drift make cold single shots lie).
- **Config flag added/removed** → `quench.conf.example` + the `config.h`/`config.cpp` pair.
- **New arch / model / quant** → `docs/supported-models.md` + README if headline.
- **Structural refactor** → `docs/architecture.md` if the narrative no longer matches.

## Common mistakes

- Publishing a perf number from a cold/single-shot run (use benchmark-cuda).
- Adding a `quench.conf.example` key without a parser binder (or vice versa) → silent drift.
- "Fixing" a doc from memory without grepping — half the staleness reports are already fixed.
- Confusing repo docs with the agent's private `MEMORY.md` (that's the auto-memory system, not a repo doc).
