<!--
Thanks for the PR. Keep it short — what changed, why, and how it was tested.
For perf-sensitive changes, paste before/after numbers (model, quant,
tg256 / pp512, hardware).
-->

## Summary

<!-- 1–3 bullets on what this changes and why. -->

## Test plan

- [ ] `make verify-fast` (or `make verify`) green
- [ ] For perf changes: tg256 / pp512 before/after on at least one model
- [ ] For new model architectures: smoke prompt + degeneration check
- [ ] For C-API changes: every `include/quench/` caller updated

## Measured evaluation

- [ ] Run the automated A/B on the maintainer's **RTX 5090** (see CONTRIBUTING.md,
      "What happens to your PR")

<!--
Tick the box above to opt in. Returning contributors are evaluated
automatically and can leave it unticked; for a first PR here it is what
starts the eval, because a run costs real GPU time on rented hardware.
The verdict decides the merge: eval:pass and eval:noise auto-merge once the
required checks are green, eval:reject and eval:tainted hold the PR.
-->

