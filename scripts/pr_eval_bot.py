#!/usr/bin/env python3
"""quench PR auto-evaluator: does the PR *really* improve decode throughput?

The CI gates check the claim's paperwork; this bot checks the claim. For each
eligible PR head it starts the rented RTX 5090, builds the PR's merged tree
AND origin/main from source on the box, runs the pinned test suite on the PR
tree, benches both arms **in the same box session**, posts the measured table
as a PR comment, applies an eval:* label, and stops the box. Ported from
braid's pr_eval_bot (see that file's history for the design rationale), with
quench's compiled-engine reality built in:

  * Same-session A/B, arms alternated per rep, median of --reps. The noise
    rule (deltas under the bar are not attributable to code on this box) is
    satisfied by construction: both arms run in one session, and the verdict
    bar is that same figure. The bar itself lives in one place,
    scorer.NOISE_PCT — currently 1.5%.
  * The measured figure is `quench-cli --bench` decode throughput ("tg",
    batch 1, greedy, ignore-EOS) via the pinned bench/decode_speed.py.
    Prefill ("pp") is recorded but does not gate: cuBLAS algorithm selection
    makes prefill vary up to 2.6x across container restarts
    (CONTRIBUTING.md, "Benchmark") — decode is the reliable A/B signal.
  * A no-op PR must come back `eval:noise`, not `eval:pass`. That is the
    bot's own null test.

Trust boundary — the part that must never be relaxed:

  * PR code executes ONLY on the rented box — including at BUILD time:
    CMake configure runs arbitrary code from the tree, so the arms are
    compiled on the box, never on this machine. Locally the PR is
    materialized via `git archive` (nothing imported, built, or executed
    here) and rsynced to /root/quench_eval/, outside any dev tree.
  * The gh and vast credentials live on this machine and are never copied
    to the box. The bot itself always runs from the local main tree.
  * The PR does not grade itself. The harness — `bench/` and `tests/` — is
    pinned: those directories are taken from the BASE sha and overlaid onto
    the PR tree before it is pushed to the box, so the PR's engine code is
    measured by main's bench and gated by main's tests against main's
    oracles (tests/refs). A PR that modifies any pinned path (or this bot,
    or the workflows) is evaluated anyway but its best verdict is capped at
    `eval:tainted` — the numbers are trustworthy, the harness diff still
    needs human eyes before merge. New files ADDED under pinned paths are
    not tainting (they cannot weaken the gate) but they do not run during
    the eval; the comment lists them as unexercised.
  * The suite is compiled by the PR's own CMakeLists (build files are
    engine-side: they produce the artifact being measured), which braid's
    pytest world did not have to worry about. The hole — a CMake edit that
    quietly drops or renames a pinned test — is closed by a suite-shape
    check: `ctest -N` on the PR arm must list a superset of the base arm's
    test names, or the eval aborts as TAMPER SUSPECTED. Residual: a CMake
    edit that keeps every test name but points a target at hollowed-out
    sources survives the check; it is overt sabotage in the diff and human
    review of build-file changes remains part of the merge decision.
  * The arms are isolated as far as software can manage on a shared root
    box: per-eval wipe of both arm trees (build dirs included — nothing
    survives across PRs), MAIN IS BUILT FIRST so PR configure/build code
    cannot poison the baseline's toolchain or build tree, main's build dir
    is hash-locked after that build, the main source tree is
    checksum-re-pushed before every main rep (PR code runs first in the
    bench loop and could otherwise edit the baseline), a GPU-idle assertion
    runs before each main rep, and the model file is hashed before any PR
    code runs and re-verified before the verdict. Any drift aborts the
    eval as TAMPER SUSPECTED.
  * Residual risk, stated honestly: PR code runs as root on the box, so a
    determined attacker can still compromise the interpreter, the CUDA
    toolchain, or rc files, or leave a daemon that races the checks
    (TOCTOU). Closing that requires a fresh container per arm — on the
    roadmap. The checks above turn a silent one-line cheat into overt
    sabotage code that has to survive human review of the diff.

Durable evidence — verdicts outlive editable comments:

  * Every verdict lands as a `quench/eval` commit status on the PR head sha
    (statuses are what branch protection can require).
  * The full measured record (the canonical evidence bundle, the verdict
    document, and the attested receipt) is appended as a git note on the
    head under refs/notes/quench-eval and pushed — an append-only audit
    trail anyone can fetch — and mirrored to a local ledger at
    ~/quench-pr-eval-ledger.jsonl.
  * The verdict itself is attested: the policy (scripts/eval_scorer.py) is
    shipped byte-identical into a polaris.computer Intel TDX machine with
    the evidence bundle (egress sealed), re-derives the verdict there, and
    the returned DCAP quote binds scorer + bundle + verdict in its
    report_data. scripts/verify_receipt.py recomputes every binding
    offline. The local and TEE verdicts must agree byte-for-byte or the
    eval aborts. Scope honesty: the receipt proves the SCORING; the 5090
    measurement itself still happens outside any TEE (consumer GPUs have
    no confidential-compute mode) — same ceiling as every attested-eval
    pipeline on consumer hardware.

Box ownership — the part that keeps this from costing money or corrupting a
measurement:

  * If the box is already running, the bot SKIPS the whole cycle: a running
    box is somebody's session, and rsyncing or benching under it would break
    the never-touch-a-box-mid-benchmark rule. Evals wait for the next poll.
  * If the bot starts the box, it stops it in a finally block, with retries,
    and screams into the log if the stop fails. Never destroys.
  * The box needs a one-time toolchain provision
    (scripts/provision_eval_box.sh) — CUDA 13.3 nvcc, CMake >= 3.25, GCC 12+,
    Ninja; it persists on the instance disk across stop/start.

Never merges — but the verdict now decides merges. The bot only ever writes a
label, a comment, a commit status and a git note; it calls no merge API. What
changed is downstream: `quench/eval` is a required status check on main, and
.github/workflows/auto-merge.yml arms GitHub's squash auto-merge when the bot
applies eval:pass or eval:noise. GitHub performs the merge, and only once every
required check is green. Consequences to keep in mind when editing this file:

  * eval:pass and eval:noise are unattended-merge verdicts. eval:tainted,
    eval:reject and eval:error are `failure`/`error` and hold the PR.
  * A non-runtime PR gets a `success` status with no eval:* label ("eval not
    required"). That is deliberately NOT enough to arm auto-merge — the
    workflow keys on the label, so a green-because-not-applicable status
    cannot merge a stranger's docs or packaging PR unreviewed.

Usage:
  python3 scripts/pr_eval_bot.py              # poll: eval all eligible heads
  python3 scripts/pr_eval_bot.py --pr 7       # eval one PR (still eligibility-checked)
  python3 scripts/pr_eval_bot.py --pr 7 --force   # bypass eligibility+idempotency (smoke tests)
  python3 scripts/pr_eval_bot.py --dry-run    # print the plan, touch nothing
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

# The verdict policy lives in eval_scorer.py — one file, imported here and
# shipped byte-identical into the TEE for the attested receipt. The bot must
# never grow its own copy of these functions.
_SCORER_PATH = pathlib.Path(__file__).resolve().parent / "eval_scorer.py"
_spec = importlib.util.spec_from_file_location("eval_scorer", _SCORER_PATH)
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)
classify_taint = scorer.classify_taint
verdict = scorer.verdict
median = scorer.median
NOISE_PCT = scorer.NOISE_PCT
GATED = scorer.GATED
PINNED_DIRS = scorer.PINNED_DIRS
INFRA_FILES = scorer.INFRA_FILES
INFRA_PREFIXES = scorer.INFRA_PREFIXES

INSTANCE = os.environ.get("VAST_INSTANCE", "47055458")
SSH_KEY = os.path.expanduser(os.environ.get("QUENCH_SSH_KEY", "~/.ssh/rtx5090"))
SSH_HOST = os.environ.get("QUENCH_SSH_HOST", "root@ssh5.vast.ai")
SSH_PORT = os.environ.get("QUENCH_SSH_PORT", "15458")
BENCH_MODEL = os.environ.get("QUENCH_EVAL_MODEL",
                             "/root/models/Mistral-Nemo-Instruct-2407-Q8_0.gguf")
CLI_EXTRA = os.environ.get("QUENCH_EVAL_CLI_ARGS", "")   # e.g. "--vram-budget 28000"
BUILD_JOBS = os.environ.get("QUENCH_EVAL_BUILD_JOBS", "$(nproc)")
REMOTE_BASE = "/root/quench_eval"         # per-eval arm trees; never a dev tree
WORKLOADS = ("pp", "tg")                  # measured; the verdict gates on scorer.GATED
RUNTIME_PREFIXES = ("src/", "include/", "tools/", "tests/", "bench/",
                    "cmake/", "CMakeLists.txt", "Dockerfile")
NON_RUNTIME_REASON = "no engine, bench, or tests paths"
STATUS_CONTEXT = "quench/eval"
POLARIS_ATTEST_URL = os.environ.get("QUENCH_POLARIS_ATTEST_URL",
                                    "https://polaris.computer/v1/attest")
POLARIS_KEY = os.environ.get("QUENCH_POLARIS_API_KEY", "")
SCORER_WORKLOAD = "python3 /in/eval_scorer.py /in/bundle.json"
NOTES_REF = "quench-eval"                 # refs/notes/quench-eval
LEDGER = os.path.expanduser("~/quench-pr-eval-ledger.jsonl")
EVAL_LABELS = {
    "eval:pass": ("0E8A16", f"measured decode speedup beyond the {NOISE_PCT}% bar"),
    "eval:noise": ("FBCA04", f"measured delta within the {NOISE_PCT}% noise bar"),
    "eval:tainted": ("5319E7",
                     "no regression, but the PR touches harness or supply-chain files"),
    "eval:reject": ("B60205",
                    f"suite failed or measured regression beyond {NOISE_PCT}%"),
    "eval:error": ("D93F0B", "evaluation could not complete"),
}
# The two verdicts that arm auto-merge (see .github/workflows/auto-merge.yml).
# Kept next to STATUS_STATE so the two can never drift: a label listed here
# MUST map to a `success` status, or the workflow would arm a merge that the
# required check then blocks forever.
AUTO_MERGE_LABELS = ("eval:pass", "eval:noise")
STATUS_STATE = {                          # eval label -> commit-status state
    "eval:pass": "success",
    "eval:noise": "success",
    "eval:tainted": "failure",
    "eval:reject": "failure",
    "eval:error": "error",
}
# build-eval lives inside each arm tree on the box; excluding it keeps
# verify_tree comparing sources only (and keeps --delete from nuking builds).
SYNC_EXCLUDES = [".git", "__pycache__", "*.pyc", "build-eval"]


# ---------------------------------------------------------------------------
# Pure policy — everything here is unit-tested locally
# (tests/eval/test_pr_eval_bot.py) and none of it shells out.

def ticked_5090(body: str | None) -> bool:
    """Opt-in signal: a ticked box on a 5090 line in the PR body."""
    import re
    if not body:
        return False
    return any("5090" in ln and re.search(r"-\s*\[\s*[xX]\s*\]", ln)
               for ln in body.split("\n"))


def touches_runtime(paths: list[str]) -> bool:
    return any(p.startswith(RUNTIME_PREFIXES) for p in paths)


def overlay_harness(pr_dir: str, base_dir: str, dirs: tuple[str, ...] = PINNED_DIRS) -> None:
    """Replace the PR tree's harness dirs with the base tree's, wholesale.

    Wholesale (delete then copy), not a merge: a merge would keep PR-added
    files, and a PR-added conftest.py under tests/api/ could monkeypatch the
    trusted API tests from inside their own pytest process.
    """
    for d in dirs:
        dst, src = os.path.join(pr_dir, d), os.path.join(base_dir, d)
        shutil.rmtree(dst, ignore_errors=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst)


def marker(sha: str) -> str:
    return f"<!-- quench-eval sha={sha} -->"


def error_marker(sha: str) -> str:
    return f"<!-- quench-eval-error sha={sha} -->"


def already_evaluated(comment_bodies: list[str], sha: str) -> bool:
    """A verdict parks the head for good; errors get one automatic retry.

    Error comments carry a different marker so a transient infra failure is
    retried on the next poll — but only once, or a persistent failure would
    start the box (and bill) every cycle forever. After two errors the head is
    parked until a new commit arrives.
    """
    bodies = [b or "" for b in comment_bodies]
    if any(marker(sha) in b for b in bodies):
        return True
    return sum(error_marker(sha) in b for b in bodies) >= 2


def pick_tok_s(bench_stdout: str, workload: str) -> float:
    """tok/s for a workload from bench/decode_speed.py --json output.

    The harness prints health/progress lines around the JSON, so take the
    last line that parses as an object with "workloads". A missing workload
    is an error, not a zero — a tree that stops reporting the gated figure
    must fail loudly."""
    doc = None
    for ln in bench_stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                cand = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and "workloads" in cand:
                doc = cand
    if doc is None:
        raise ValueError("no decode_speed JSON found in bench output")
    for w in doc["workloads"]:
        if w.get("name") == workload:
            return float(w["tok_per_s"])
    raise ValueError(f"workload {workload!r} missing from bench output")


def missing_tests(base_names: list[str], pr_names: list[str]) -> list[str]:
    """Test names present on the base arm but absent from the PR arm.

    The suite is compiled by the PR's own CMakeLists, so a CMake edit could
    silently drop a pinned test from the gate; a non-empty return is treated
    as tampering by the caller. Additions are fine (they join the gate)."""
    return sorted(set(base_names) - set(pr_names))


def build_bundle(pr: int, head: str, eval_sha: str, base_sha: str, mode: str,
                 reps: int, tests_ok: bool, samples: dict, name_status: list,
                 suite_tail: str, model_sha: str, main_build_hash: str | None,
                 pp: int, tg: int) -> dict:
    """The canonical evidence bundle: everything the verdict is a function
    of, in one JSON document. The scorer (locally AND inside the TEE)
    derives the verdict from this and nothing else; its hash is bound into
    the attested receipt's report_data."""
    return {
        "schema": scorer.SCHEMA_IN,
        "pr": pr, "head": head, "eval_sha": eval_sha, "base_sha": base_sha,
        "mode": mode, "workloads": list(WORKLOADS),
        "bench": {"pp_tokens": pp, "tg_tokens": tg, "batch": 1},
        "reps": reps, "noise_pct": NOISE_PCT, "box": INSTANCE,
        "model": os.path.basename(BENCH_MODEL),
        "tests_ok": tests_ok,
        "suite_tail_sha256": hashlib.sha256(suite_tail.encode()).hexdigest(),
        "samples": {a: {w: list(v) for w, v in per.items()}
                    for a, per in samples.items()},
        "name_status": [list(p) for p in name_status],
        "integrity": {"model_sha256": model_sha,
                      "main_build_hash": main_build_hash or ""},
    }


def request_receipt(bundle: dict) -> dict | None:
    """Re-score the bundle inside a polaris.computer Intel TDX machine and
    return the attested receipt (raw DCAP quote + bindings), or None.

    Best-effort by design: the receipt is evidence, not the gate — a
    polaris.computer outage must not block evaluations. But a receipt that
    RETURNS a different verdict than the local scorer is handled by the
    caller as a hard error, because the scorer is deterministic and a
    divergence means a bug or tampering."""
    if not POLARIS_KEY:
        print(">> QUENCH_POLARIS_API_KEY unset — verdict will not carry a TDX receipt")
        return None
    payload = json.dumps({
        "nonce": bundle["head"],                    # 40 hex chars: the PR head sha
        "workload": SCORER_WORKLOAD,
        "egress": "none",
        "files": {
            "/in/eval_scorer.py":
                base64.b64encode(_SCORER_PATH.read_bytes()).decode(),
            "/in/bundle.json":
                base64.b64encode(scorer.canonical(bundle).encode()).decode(),
        },
    }).encode()
    req = urllib.request.Request(
        POLARIS_ATTEST_URL, data=payload,
        headers={"Authorization": f"Bearer {POLARIS_KEY}",
                 "Content-Type": "application/json",
                 # Cloudflare fronts the API and 403s the default urllib UA
                 "User-Agent": "quench-pr-eval-bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            doc = json.load(r)
    except (OSError, json.JSONDecodeError) as e:
        print(f"!! attested receipt unavailable: {e}", file=sys.stderr)
        return None
    if doc.get("exit_code") != 0:
        print(f"!! scorer failed inside the TEE (exit {doc.get('exit_code')}): "
              f"{str(doc.get('stdout', ''))[:400]}", file=sys.stderr)
        return None
    return doc


# ---------------------------------------------------------------------------
# Shell plumbing

def run(cmd: list[str], timeout: int = 120, check: bool = True,
        input_: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=False, input=input_,
                          timeout=timeout, check=check)


def gh(args: list[str], timeout: int = 60) -> str:
    return run(["gh", *args], timeout=timeout).stdout.decode()


def gh_json(args: list[str]) -> object:
    return json.loads(gh(args))


class Box:
    """The rented 5090, held politely: skip if busy, stop if we started it."""

    def __init__(self) -> None:
        self.we_started = False

    def status(self) -> str:
        out = run(["vastai", "show", "instance", INSTANCE, "--raw"]).stdout
        return json.loads(out).get("actual_status", "unknown")

    def ssh(self, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return run(["ssh", "-i", SSH_KEY, "-p", SSH_PORT,
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=25", SSH_HOST, cmd],
                   timeout=timeout, check=False)

    def start(self) -> None:
        run(["vastai", "start", "instance", INSTANCE])
        self.we_started = True
        for _ in range(40):                       # ~7 min of patience
            time.sleep(10)
            if self.ssh("true", timeout=40).returncode == 0:
                return
        raise RuntimeError("box started but ssh never came up")

    def stop(self) -> None:
        if not self.we_started:
            return
        for attempt in range(4):
            try:
                run(["vastai", "stop", "instance", INSTANCE])
                self.we_started = False
                print(f">> box {INSTANCE} stopped")
                return
            except subprocess.SubprocessError as e:
                print(f"!! stop attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(15)
        print(f"!! BOX {INSTANCE} MAY STILL BE RUNNING AND BILLING — stop it "
              f"by hand: vastai stop instance {INSTANCE}", file=sys.stderr)

    def gpu_busy(self) -> bool:
        r = self.ssh("nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
        return r.returncode != 0 or int(r.stdout.decode().strip() or "1") != 0

    def _rsync_ssh(self) -> str:
        return f"ssh -i {SSH_KEY} -p {SSH_PORT} -o StrictHostKeyChecking=accept-new"

    def push_tree(self, local_dir: str, remote_dir: str) -> None:
        ex = [f"--exclude={e}" for e in SYNC_EXCLUDES]
        run(["rsync", "-az", "--delete", *ex, "-e", self._rsync_ssh(),
             f"{local_dir}/", f"{SSH_HOST}:{remote_dir}/"], timeout=1800)

    def verify_tree(self, local_dir: str, remote_dir: str) -> list[str]:
        """Checksum-compare the remote tree against the local truth and
        repair it. Returns the itemized drift — non-empty means something on
        the box rewrote the tree since the last push, which after PR code
        has run is tamper evidence, not noise."""
        ex = [f"--exclude={e}" for e in SYNC_EXCLUDES]
        r = run(["rsync", "-azc", "--delete", "--itemize-changes", *ex,
                 "-e", self._rsync_ssh(),
                 f"{local_dir}/", f"{SSH_HOST}:{remote_dir}/"], timeout=1800)
        return [ln for ln in r.stdout.decode().splitlines() if ln.strip()]

    def wipe(self, *paths: str) -> None:
        joined = " ".join(paths)
        self.ssh(f"rm -rf {joined} && mkdir -p {joined}")

    def tree_hash(self, path: str) -> str:
        """One sha256 over every file under path (symlinks followed, order
        fixed). Used to lock the main arm's build dir."""
        r = self.ssh(
            f"cd {path} 2>/dev/null && find -L . -type f -print0 | LC_ALL=C sort -z"
            f" | xargs -0 sha256sum | sha256sum | cut -d' ' -f1 || echo MISSING",
            timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"tree_hash failed for {path}: {r.stderr.decode()[-500:]}")
        return r.stdout.decode().strip().splitlines()[-1]

    def file_hash(self, path: str) -> str:
        """sha256 of one file (the staged GGUF model)."""
        r = self.ssh(f"sha256sum {path} | cut -d' ' -f1", timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"file_hash failed for {path}: {r.stderr.decode()[-500:]}")
        return r.stdout.decode().strip()


# ---------------------------------------------------------------------------
# Getting the code without running it

def checkout_tree(sha: str, dest: str) -> None:
    """Materialise a commit into dest via git archive — nothing is executed."""
    ar = run(["git", "archive", sha], timeout=120)
    os.makedirs(dest, exist_ok=True)
    run(["tar", "-x", "-C", dest], input_=ar.stdout, timeout=120)


def fetch_arms(pr: int) -> tuple[str, str, str]:
    """(eval_sha, base_sha, mode). Prefer the merge ref: it measures what
    merging actually does. Fall back to head-vs-merge-base when GitHub has no
    merge ref (conflicts)."""
    run(["git", "fetch", "origin", "main"], timeout=120)
    base = run(["git", "rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
    r = run(["git", "fetch", "origin", f"refs/pull/{pr}/merge"],
            timeout=120, check=False)
    if r.returncode == 0:
        merged = run(["git", "rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
        return merged, base, "merge-vs-main"
    run(["git", "fetch", "origin", f"refs/pull/{pr}/head"], timeout=120)
    head = run(["git", "rev-parse", "FETCH_HEAD"]).stdout.decode().strip()
    mb = run(["git", "merge-base", base, head]).stdout.decode().strip()
    return head, mb, "head-vs-merge-base"


def diff_name_status(base_sha: str, eval_sha: str) -> list[tuple[str, str]]:
    """[(status, path)] for the PR's effective change, renames split so a
    rename of a pinned file shows up as a taint-carrying D."""
    out = run(["git", "diff", "--name-status", "--no-renames",
               base_sha, eval_sha], timeout=120).stdout.decode()
    pairs = []
    for ln in out.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[-1].strip()))
    return pairs


# ---------------------------------------------------------------------------
# GitHub layer

def pr_info(n: int) -> dict:
    return gh_json(["pr", "view", str(n), "--json",
                    "number,state,isDraft,labels,headRefOid,body,files,comments,title"])


def eligible(info: dict) -> tuple[bool, str]:
    labels = [lb["name"] for lb in info.get("labels", [])]
    if info.get("state") != "OPEN":
        return False, "not open"
    if info.get("isDraft"):
        return False, "draft"
    if "hold" in labels:
        return False, "hold label"
    paths = [f["path"] for f in info.get("files", [])]
    if not touches_runtime(paths):
        return False, NON_RUNTIME_REASON
    if not (ticked_5090(info.get("body")) or "eval" in labels):
        return False, "no 5090 attestation (tick the box, or label `eval` to force)"
    bodies = [c.get("body", "") for c in info.get("comments", [])]
    if already_evaluated(bodies, info["headRefOid"]):
        return False, f"head {info['headRefOid'][:9]} already evaluated"
    return True, "eligible"


def ensure_labels() -> None:
    for name, (color, desc) in EVAL_LABELS.items():
        run(["gh", "label", "create", name, "--color", color,
             "--description", desc, "--force"], check=False)


def post_status(sha: str, state: str, description: str) -> None:
    """A `quench/eval` commit status on the head sha — the thing branch
    protection can require, and the thing a later comment edit cannot
    retroactively change for that sha."""
    run(["gh", "api", f"repos/{{owner}}/{{repo}}/statuses/{sha}",
         "-f", f"state={state}", "-f", f"context={STATUS_CONTEXT}",
         "-f", f"description={description[:140]}"], check=False)


def current_status(sha: str) -> str | None:
    try:
        doc = gh_json(["api", f"repos/{{owner}}/{{repo}}/commits/{sha}/status"])
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None
    for st in doc.get("statuses", []):
        if st.get("context") == STATUS_CONTEXT:
            return st.get("state")
    return None


def ensure_status(sha: str, state: str, description: str) -> None:
    """Post only on change, so the */30 poll does not spam the statuses API."""
    if current_status(sha) != state:
        post_status(sha, state, description)


def stamp_queue_status(info: dict, ok: bool, why: str) -> None:
    """Give every open head a `quench/eval` status so required-status branch
    protection is livable: runtime PRs show pending until measured, and
    docs-only PRs are not held hostage by a check that will never run."""
    sha = info["headRefOid"]
    if ok:
        ensure_status(sha, "pending", "queued for measured eval on the RTX 5090")
    elif why == NON_RUNTIME_REASON:
        ensure_status(sha, "success", "eval not required — no runtime paths")
    elif why.startswith("no 5090 attestation"):
        ensure_status(sha, "pending", "awaiting the RTX 5090 attestation tick")


def record_evidence(head: str, record: dict) -> None:
    """Append the measured record where a comment edit cannot reach it: a
    git note on the head sha pushed to refs/notes/quench-eval (append-only —
    every edit is a commit on the notes ref), plus a local ledger line."""
    line = json.dumps(record, sort_keys=True)
    try:
        with open(LEDGER, "a") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"!! ledger append failed: {e}", file=sys.stderr)
    notes_spec = f"+refs/notes/{NOTES_REF}:refs/notes/{NOTES_REF}"
    run(["git", "fetch", "origin", notes_spec], timeout=60, check=False)
    for _ in range(2):
        r = run(["git", "notes", "--ref", NOTES_REF, "append", "-m", line, head],
                check=False)
        if r.returncode != 0:
            break
        if run(["git", "push", "origin", f"refs/notes/{NOTES_REF}"],
               timeout=60, check=False).returncode == 0:
            return
        run(["git", "fetch", "origin", notes_spec], timeout=60, check=False)
    print("!! could not push the eval note — evidence is in the ledger only",
          file=sys.stderr)


def apply_verdict(pr: int, label: str, body: str) -> None:
    ensure_labels()
    for other in EVAL_LABELS:
        if other != label:
            run(["gh", "pr", "edit", str(pr), "--remove-label", other], check=False)
    run(["gh", "pr", "edit", str(pr), "--add-label", label], check=False)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        gh(["pr", "comment", str(pr), "--body-file", path])
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# The evaluation itself

def build_arm(box: Box, arm_dir: str) -> None:
    """From-scratch Release build on the box. The build dir lives inside the
    arm tree (excluded from rsync verification) and is wiped with it per
    eval — no build artifact survives across PRs."""
    cmd = (f"cd {arm_dir} && "
           f"cmake -S . -B build-eval -G Ninja -DCMAKE_BUILD_TYPE=Release "
           f"-DQUENCH_BUILD_SERVER=OFF > build-eval-configure.log 2>&1 && "
           f"cmake --build build-eval -j{BUILD_JOBS} > build-eval-build.log 2>&1")
    r = box.ssh(cmd, timeout=5400)
    if r.returncode != 0:
        tail = box.ssh(f"tail -c 2000 {arm_dir}/build-eval-build.log "
                       f"{arm_dir}/build-eval-configure.log 2>/dev/null")
        raise RuntimeError(f"build failed in {arm_dir}:\n{tail.stdout.decode()[-2500:]}")


def list_tests(box: Box, arm_dir: str) -> list[str]:
    """`ctest -N` test names for the arm's build tree."""
    r = box.ssh(f"cd {arm_dir}/build-eval && ctest -N", timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ctest -N failed in {arm_dir}: {r.stderr.decode()[-800:]}")
    names = []
    for ln in r.stdout.decode().splitlines():
        ln = ln.strip()                       # "  Test #1: unit_core"
        if ln.startswith("Test #") and ": " in ln:
            names.append(ln.split(": ", 1)[1].strip())
    return names


def run_suite(box: Box, arm_dir: str, lanes: list[str]) -> tuple[bool, str]:
    ctest = " && ".join(
        f"ctest --test-dir build-eval -L {ln} --output-on-failure" for ln in lanes)
    r = box.ssh(f"cd {arm_dir} && timeout 3600 bash -c '{ctest}'", timeout=3720)
    tail = (r.stdout.decode() + r.stderr.decode())[-1500:]
    return r.returncode == 0, tail


def run_bench(box: Box, arm_dir: str, pp: int, tg: int) -> dict[str, float]:
    extra = f" -- {CLI_EXTRA}" if CLI_EXTRA else ""
    r = box.ssh(f"cd {arm_dir} && python3 -B bench/decode_speed.py "
                f"--cli build-eval/quench-cli --model {BENCH_MODEL} "
                f"--pp {pp} --tg {tg} --reps 1{extra}", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"bench failed in {arm_dir}:\n{r.stderr.decode()[-2000:]}")
    out = r.stdout.decode()
    return {w: pick_tok_s(out, w) for w in WORKLOADS}


def check_main_arm(box: Box, main_dir: str, main_build_hash: str | None) -> None:
    """Everything that must hold before a main rep is trusted. PR code has
    already run in this session; any drift is tampering, not noise."""
    drift = box.verify_tree(main_dir, f"{REMOTE_BASE}/main")
    if drift:
        raise RuntimeError(
            "TAMPER SUSPECTED — the main tree drifted after PR code ran "
            "(repaired, eval aborted):\n" + "\n".join(drift[:20]))
    if main_build_hash is not None:
        now = box.tree_hash(f"{REMOTE_BASE}/main/build-eval")
        if now != main_build_hash:
            raise RuntimeError(
                "TAMPER SUSPECTED — main's build tree changed between reps "
                f"({main_build_hash[:12]} -> {now[:12]})")
    if box.gpu_busy():
        raise RuntimeError(
            "TAMPER SUSPECTED — a GPU process is alive before a main rep; "
            "PR code may have left a daemon behind")


def evaluate(box: Box, pr: int, info: dict, args, model_sha: str) -> None:
    head = info["headRefOid"]
    eval_sha, base_sha, mode = fetch_arms(pr)
    name_status = diff_name_status(base_sha, eval_sha)
    tainted, unexercised = classify_taint(name_status)
    print(f">> PR #{pr} head={head[:9]}: {mode} — eval {eval_sha[:9]} vs {base_sha[:9]}"
          + (f"; tainted: {tainted}" if tainted else ""))

    with tempfile.TemporaryDirectory(prefix="quench-eval-") as tmp:
        pr_dir, main_dir = os.path.join(tmp, "pr"), os.path.join(tmp, "main")
        checkout_tree(eval_sha, pr_dir)
        checkout_tree(base_sha, main_dir)
        overlay_harness(pr_dir, main_dir)     # bench/ and tests/ come from base
        box.wipe(f"{REMOTE_BASE}/pr", f"{REMOTE_BASE}/main")  # nothing survives across PRs
        box.push_tree(main_dir, f"{REMOTE_BASE}/main")
        box.push_tree(pr_dir, f"{REMOTE_BASE}/pr")

        # Main is built FIRST: at this point no PR code has executed on the
        # box, so PR configure/build hooks cannot have poisoned the
        # toolchain or main's build tree. Lock main's build dir before any
        # PR code runs.
        print(">> building main (trusted baseline)")
        build_arm(box, f"{REMOTE_BASE}/main")
        main_build_hash = box.tree_hash(f"{REMOTE_BASE}/main/build-eval")
        base_tests = list_tests(box, f"{REMOTE_BASE}/main")

        print(">> building PR (first PR code execution on the box)")
        build_arm(box, f"{REMOTE_BASE}/pr")
        dropped = missing_tests(base_tests, list_tests(box, f"{REMOTE_BASE}/pr"))
        if dropped:
            raise RuntimeError(
                "TAMPER SUSPECTED — the PR's build drops pinned tests from "
                f"the suite (CMake edit?): {dropped}")

        print(f">> suite: ctest lanes {args.lanes} (tests pinned to base) on the PR tree")
        tests_ok, tail = run_suite(box, f"{REMOTE_BASE}/pr", args.lanes)
        print(f">> suite: {'ok' if tests_ok else 'FAILED'}")

        samples: dict[str, dict[str, list[float]]] = {
            "pr": {w: [] for w in WORKLOADS}, "main": {w: [] for w in WORKLOADS}}
        if tests_ok:
            for rep in range(args.reps):
                order = ["pr", "main"] if rep % 2 == 0 else ["main", "pr"]
                for arm_name in order:
                    if arm_name == "main":
                        check_main_arm(box, main_dir, main_build_hash)
                    got = run_bench(box, f"{REMOTE_BASE}/{arm_name}", args.pp, args.tg)
                    for w, v in got.items():
                        samples[arm_name][w].append(v)
                    print(f">> rep {rep + 1} {arm_name}: "
                          + " ".join(f"{w} {v:.1f}" for w, v in sorted(got.items())))

        model_now = box.file_hash(BENCH_MODEL)
        if model_now != model_sha:
            raise RuntimeError(
                "TAMPER SUSPECTED — the model file changed during the eval "
                f"({model_sha[:12]} -> {model_now[:12]})")

    # One canonical bundle; one scorer; two executions. The local call decides
    # the verdict, the TEE call produces the receipt — and they must agree.
    bundle = build_bundle(pr, head, eval_sha, base_sha, mode, args.reps,
                          tests_ok, samples, name_status, tail,
                          model_sha, main_build_hash, args.pp, args.tg)
    vdoc = scorer.score_bundle(bundle)
    expected_stdout = scorer.canonical(vdoc) + "\n"
    label, reason = vdoc["label"], vdoc["reason"]
    med, deltas = vdoc["medians"], vdoc["deltas_pct"]

    receipt = request_receipt(bundle)
    if receipt is not None and receipt.get("stdout") != expected_stdout:
        raise RuntimeError(
            "TEE-scored verdict differs from the local scorer on the same "
            "bundle — deterministic policy diverged; refusing to post. "
            f"local={expected_stdout!r} tee={receipt.get('stdout')!r}")

    rows = "\n".join(
        f"| {w}{'' if w in GATED else ' (informational)'} "
        f"| {med['main'][w]:.1f} | {med['pr'][w]:.1f} | {deltas[w]:+.1f}% |"
        for w in WORKLOADS) if deltas else ""
    suite_note = "suite passed" if tests_ok else f"suite FAILED — tail:\n```\n{tail}\n```"
    taint_note = (
        "\n> **Harness or supply-chain files modified by this PR** (harness files "
        "were measured from the pinned base versions; supply-chain files are not "
        "measured at all). Auto-merge is held until these are reviewed by hand:\n"
        + "".join(f"> - `{p}`\n" for p in tainted) if tainted else "")
    unex_note = (
        "\n> New files under pinned paths — cannot weaken the gate, but they did "
        "**not** run in this eval; they join the gate once merged:\n"
        + "".join(f"> - `{p}`\n" for p in unexercised) if unexercised else "")
    body = (
        f"{marker(head)}\n"
        f"## Measured on the RTX 5090 — `{label}`\n\n"
        f"Same-session A/B on box {INSTANCE}: `{mode}` "
        f"(`{eval_sha[:9]}` vs `{base_sha[:9]}`), both arms built from source on "
        f"the box, median of {args.reps} reps with arm order alternated. "
        f"`quench-cli --bench` pp={args.pp} tg={args.tg} batch 1, model "
        f"`{os.path.basename(BENCH_MODEL)}`. Every number below is **measured**; "
        f"the verdict gates on `tg` only (prefill varies with cuBLAS algorithm "
        f"selection) and the bar is the same-session ±{NOISE_PCT:.1f}% rule.\n\n"
        f"Harness pinned to base `{base_sha[:9]}`: `bench/` and `tests/` were "
        f"overlaid from main before the PR tree reached the box — the PR's engine "
        f"is measured by main's bench and gated by main's tests against main's "
        f"oracles. Integrity: arm trees wiped per eval, main built before any PR "
        f"code ran and its build dir hash-locked, main tree checksum-verified "
        f"before every main rep, suite shape checked (`ctest -N` superset), model "
        f"file hash verified.\n\n"
        + (f"**Attested:** this verdict was independently recomputed inside an Intel "
           f"TDX machine ({receipt['tee_attestation'].get('kind', 'tdx')}, "
           f"polaris.computer). The DCAP quote binds the scorer "
           f"(`{receipt['tee_attestation']['bound_digest'][:19]}…`), the evidence "
           f"bundle and the verdict; intel_verified="
           f"{receipt.get('verification', {}).get('intel_verified')}. Receipt + "
           f"bundle in this PR's git note — check with `scripts/verify_receipt.py`.\n"
           if receipt is not None else
           "**Attested:** no TDX receipt for this run (attestation service "
           "unavailable); the verdict is the local scorer's alone.\n")
        + taint_note + unex_note + "\n"
        + ("| workload | main tok/s | PR tok/s | delta |\n|--:|--:|--:|--:|\n"
           + rows + "\n\n" if rows else "")
        + f"**Verdict:** {reason}. {suite_note}\n\n"
        + (f"This verdict arms squash auto-merge; GitHub merges once every required "
           f"check is green.\n\n" if label in AUTO_MERGE_LABELS else
           f"This verdict holds the PR: `{label}` is not an auto-merge verdict.\n\n")
        + f"<sub>Automated eval (scripts/pr_eval_bot.py). It calls no merge API; a new "
        f"push re-queues evaluation. Verdict also lands as the `{STATUS_CONTEXT}` commit "
        f"status on `{head[:9]}` and as a git note under `refs/notes/{NOTES_REF}`. "
        f"Box stopped after the run.</sub>"
    )
    apply_verdict(pr, label, body)
    post_status(head, STATUS_STATE[label], f"{label}: {reason}")
    record_evidence(head, {
        "ts": int(time.time()), "label": label, "reason": reason,
        "bundle": bundle, "verdict_doc": vdoc, "receipt": receipt,
    })
    print(f">> PR #{pr}: {label} — {reason}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pr", type=int, help="evaluate one PR instead of polling")
    p.add_argument("--force", action="store_true",
                   help="with --pr: skip eligibility and idempotency checks")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--pp", type=int, default=512)
    p.add_argument("--tg", type=int, default=128)
    p.add_argument("--lanes", nargs="+", default=["unit", "gpu"],
                   help="ctest label lanes that form the suite gate")
    p.add_argument("--max-evals", type=int, default=2,
                   help="cost cap: at most this many PRs per cycle (each eval "
                        "carries two from-scratch CUDA builds)")
    args = p.parse_args()

    if args.pr is not None:
        todo = [args.pr]
    else:
        todo = [pr["number"] for pr in gh_json(["pr", "list", "--json", "number"])]
    plan: list[tuple[int, dict]] = []
    for n in todo:
        info = pr_info(n)
        ok, why = eligible(info)
        if args.force and args.pr is not None:
            ok, why = True, "forced"
        print(f"-- PR #{n}: {why}")
        if not args.dry_run:
            stamp_queue_status(info, ok, why)
        if ok:
            plan.append((n, info))
    plan = plan[: args.max_evals]
    if not plan:
        print(">> nothing to evaluate")
        return 0
    if args.dry_run:
        print(f">> dry-run: would evaluate {[n for n, _ in plan]} — stopping here")
        return 0

    box = Box()
    status = box.status()
    if status == "running":
        print(">> box is already running — that is somebody's session, and "
              "touching it mid-work is forbidden. Deferring to the next poll.")
        return 0
    print(f">> box {INSTANCE} is {status}; starting")
    try:
        box.start()
        if box.gpu_busy():
            raise RuntimeError("GPU busy right after start — refusing to eval")
        # The trusted baseline for the whole session, taken before ANY PR
        # code has had a chance to run on the box.
        model_sha = box.file_hash(BENCH_MODEL)
        print(f">> model sha {model_sha[:12]} ({BENCH_MODEL})")
        for n, info in plan:
            try:
                evaluate(box, n, info, args, model_sha)
            except Exception as e:                       # noqa: BLE001 — verdict must land
                print(f"!! PR #{n} eval error: {e}", file=sys.stderr)
                apply_verdict(n, "eval:error",
                              f"{error_marker(info['headRefOid'])}\n"
                              f"## Evaluation error — `eval:error`\n\n"
                              f"```\n{str(e)[-1800:]}\n```\n\n"
                              f"<sub>Automated eval; retried once on the next poll. "
                              f"After two errors this head is parked until a new "
                              f"commit is pushed.</sub>")
                post_status(info["headRefOid"], "error",
                            f"eval:error: {str(e)[:120]}")
    finally:
        box.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
