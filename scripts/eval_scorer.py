#!/usr/bin/env python3
"""quench eval scorer — the verdict policy as a pure, TEE-runnable program.

This file is the single source of truth for how a measured evaluation
becomes a verdict. `scripts/pr_eval_bot.py` imports its policy from here,
and the same file is shipped byte-for-byte into a polaris.computer Intel
TDX machine per evaluation (`/v1/attest`, egress=none), where it re-derives
the verdict from the canonical evidence bundle. The receipt's `files_sha256`
binds this file's hash and the bundle's hash into the TD quote's
report_data, and `result_sha256` binds the verdict printed on stdout — so a
third party can check that THIS policy, at the committed revision, produced
THAT verdict from THOSE measurements, without trusting the maintainer's
machine. `scripts/verify_receipt.py` performs the recomputation.

Policy specifics for quench:

  * The verdict gates on decode throughput ("tg") only. Prefill ("pp") is
    measured and reported, but cuBLAS algorithm selection makes prefill
    vary up to 2.6x across container restarts (see CONTRIBUTING.md,
    "Benchmark") — decode is the reliable A/B signal on this box.
  * The +-1.5% noise bar is inherited from the same-session A/B discipline
    (both arms run in one box session, arm order alternated per rep,
    median of reps). It must be re-validated by this repo's own null test
    (a no-op PR must come back eval:noise, not eval:pass). The bar is
    bidirectional: it is the reject bar as well as the pass bar, so
    lowering it makes the bot quicker to certify a speedup AND quicker to
    call a regression. Do not lower it again without re-running the null
    test on the box.
  * Verdicts feed auto-merge: eval:pass and eval:noise map to a `success`
    quench/eval commit status, which main's branch protection requires.
    A path whose diff needs human eyes must therefore taint (see
    classify_taint) rather than merely be reported.

Rules for editing: stdlib only, deterministic (no clocks, no randomness, no
network — egress is sealed anyway), and every printed byte canonical
(sorted keys, fixed separators). Any behavior change here is a policy
change and lands with tests in tests/eval/test_pr_eval_bot.py.

Usage:  python3 eval_scorer.py bundle.json    # or '-' for stdin
Prints the canonical verdict JSON on stdout; exits non-zero on a malformed
bundle.
"""
from __future__ import annotations

import json
import statistics
import sys

SCHEMA_IN = "quench-eval-bundle/1"
SCHEMA_OUT = "quench-eval-verdict/1"
NOISE_PCT = 1.5                            # the same-session bar; also the verdict bar
GATED = ("tg",)                            # workloads that decide the verdict
PINNED_DIRS = ("bench", "tests")           # harness: taken from base
INFRA_FILES = ("scripts/pr_eval_bot.py", "scripts/pr_eval_cron.sh",
               "scripts/eval_scorer.py", "scripts/provision_eval_box.sh")
INFRA_PREFIXES = (".github/",)
# Supply-chain paths. Not eval infrastructure and not measured by the A/B —
# they are how the build gets its dependencies and how the container starts,
# which makes them the highest-value target in the repo for anyone who wants
# code on main without writing it into src/. touches_runtime() does not cover
# them, so a PR editing only these would otherwise be stamped "eval not
# required" -> success -> auto-mergeable unreviewed. Tainting them means the
# best reachable verdict is eval:tainted (a `failure` status), which holds the
# merge until a human has read the diff.
SUPPLY_CHAIN_FILES = ("Dockerfile", "docker-entrypoint.sh", "docker-compose.yml")
SUPPLY_CHAIN_PREFIXES = ("cmake/",)


def median(xs: list[float]) -> float:
    return statistics.median(xs)


def _pinned(path: str) -> bool:
    return any(path == d or path.startswith(d + "/") for d in PINNED_DIRS)


def classify_taint(name_status) -> tuple[list[str], list[str]]:
    """Split a base..PR name-status diff into (tainted, unexercised).

    tainted: harness files the PR modifies or deletes, plus any touch at all
    of the eval infrastructure (this file included) or of the supply-chain
    paths. The eval still runs — with the pinned harness, so the numbers stay
    honest — but the best reachable verdict is eval:tainted; that diff needs
    human eyes.

    unexercised: files ADDED under pinned dirs. An addition cannot weaken
    the pinned gate, so it does not taint — but the overlay means it never
    runs during the eval either (a new test joins the gate once merged).
    """
    tainted, unexercised = [], []
    for status, path in (tuple(p) for p in name_status):
        if (path in INFRA_FILES or path.startswith(INFRA_PREFIXES)
                or path in SUPPLY_CHAIN_FILES
                or path.startswith(SUPPLY_CHAIN_PREFIXES)):
            tainted.append(path)
        elif _pinned(path):
            (unexercised if status == "A" else tainted).append(path)
    return tainted, unexercised


def verdict(tests_ok: bool, deltas_pct: dict[str, float],
            tainted=(), noise: float = NOISE_PCT) -> tuple[str, str]:
    """(label, reason). Ladder, strongest first: suite failure, then a measured
    regression (a regression anywhere outranks an improvement anywhere), then a
    taint, then pass/noise.

    A taint caps BOTH a would-be pass and a would-be noise at eval:tainted. It
    used to cap only a pass, which was safe while merging was manual: a tainted
    no-delta PR read eval:noise and a human still clicked merge. Now that
    eval:noise is a `success` status and arms auto-merge, that would hand
    unattended merges to exactly the diffs the taint exists to hold back — a
    Dockerfile or cmake/ edit has no reason to move decode throughput, so it
    would land on the noise path every time. The measurement itself stays
    trustworthy either way (it used the pinned harness); what needs eyes is the
    diff. deltas_pct holds only the GATED workloads."""
    if not tests_ok:
        return "eval:reject", "pinned test suite failed on the merged tree"
    worst = min(deltas_pct.values())
    best = max(deltas_pct.values())
    if worst < -noise:
        w = min(deltas_pct, key=deltas_pct.get)
        return "eval:reject", f"measured regression at {w}: {deltas_pct[w]:+.1f}%"
    if tainted:
        w = max(deltas_pct, key=deltas_pct.get)
        return "eval:tainted", (
            f"no regression (best {deltas_pct[w]:+.1f}% at {w}) — but the PR "
            f"modifies harness or supply-chain files; review them by hand "
            f"before merging")
    if best > noise:
        w = max(deltas_pct, key=deltas_pct.get)
        return "eval:pass", f"measured speedup at {w}: {deltas_pct[w]:+.1f}%"
    return "eval:noise", (f"all gated deltas within the ±{noise:.1f}% same-session "
                          f"bar (best {best:+.1f}%)")


def canonical(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def score_bundle(bundle: dict) -> dict:
    """Canonical bundle in, verdict document out. Deterministic; the bot and
    the TEE both call exactly this, so their outputs must be byte-identical."""
    if bundle.get("schema") != SCHEMA_IN:
        raise ValueError(f"unknown bundle schema: {bundle.get('schema')!r}")
    workloads = [str(w) for w in bundle["workloads"]]
    tests_ok = bool(bundle["tests_ok"])
    tainted, unexercised = classify_taint(bundle.get("name_status", []))

    medians: dict[str, dict[str, float]] = {}
    deltas: dict[str, float] = {}
    if tests_ok:
        missing = [w for w in GATED if w not in workloads]
        if missing:
            raise ValueError(f"tests passed but gated workloads missing: {missing}")
        m = {arm: {w: median([float(v) for v in bundle["samples"][arm][w]])
                   for w in workloads} for arm in ("pr", "main")}
        deltas = {w: (m["pr"][w] - m["main"][w]) / m["main"][w] * 100 for w in workloads}
        medians = {arm: {w: round(v, 3) for w, v in per.items()}
                   for arm, per in m.items()}
    gated_deltas = {w: deltas[w] for w in GATED} if tests_ok else {}
    label, reason = verdict(tests_ok, gated_deltas, tainted)

    return {
        "schema": SCHEMA_OUT,
        "pr": bundle["pr"], "head": bundle["head"],
        "eval_sha": bundle["eval_sha"], "base_sha": bundle["base_sha"],
        "label": label, "reason": reason, "noise_pct": NOISE_PCT,
        "gated": list(GATED),
        "tests_ok": tests_ok,
        "medians": medians,
        "deltas_pct": {w: round(v, 3) for w, v in sorted(deltas.items())},
        "tainted": tainted, "unexercised": unexercised,
    }


def main() -> int:
    src = sys.stdin if len(sys.argv) < 2 or sys.argv[1] == "-" else open(sys.argv[1])
    with src:
        bundle = json.load(src)
    print(canonical(score_bundle(bundle)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
