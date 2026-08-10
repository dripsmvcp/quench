#!/usr/bin/env python3
"""Pinned bench harness: one quench-cli --bench run, canonical JSON out.

This is the score producer the PR eval pipeline pins: during an evaluation
this directory is overlaid from the BASE commit onto the PR tree before the
tree reaches the runner, so a PR cannot bring its own copy of the file that
produces its number (scripts/pr_eval_bot.py, "overlay_harness").

It wraps `quench-cli --bench` (synthetic benchmark, greedy, ignore-EOS,
batch 1) and reprints the two throughput figures as one machine-readable
line so the caller never scrapes human-formatted output:

    {"schema": "quench-bench/1",
     "workloads": [{"name": "pp", "tok_per_s": 700.1},
                   {"name": "tg", "tok_per_s": 189.4}], "health": "ok"}

The verdict gates on "tg" only — prefill varies up to 2.6x across container
restarts from cuBLAS algorithm selection (CONTRIBUTING.md, "Benchmark");
decode is the reliable A/B signal. "pp" is reported for the record.

Stdlib only, no repo imports: the file must run on a bare runner box with
nothing but python3 and the built binary.

Usage:
  python3 bench/decode_speed.py --cli build/quench-cli --model m.gguf \
      [--pp 512] [--tg 128] [--reps 1] [-- extra quench-cli args]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# quench-cli prints (stderr):  tg   128 tokens  avg   677.00 ms  ( 189.00 tok/s)  [1 reps]
LINE = re.compile(r"^(pp|tg)\s+\d+\s+tokens\s+avg\s+[\d.]+\s+ms\s+"
                  r"\(\s*([\d.]+)\s+tok/s\)", re.MULTILINE)


def parse_bench(output: str) -> dict[str, float]:
    """{'pp': tok_per_s, 'tg': tok_per_s} from quench-cli --bench output.
    A missing workload is an error, not a zero — a tree whose bench mode
    stops printing a figure must fail loudly."""
    found = {name: float(v) for name, v in LINE.findall(output)}
    missing = [w for w in ("pp", "tg") if w not in found]
    if missing:
        raise ValueError(f"quench-cli --bench output missing {missing}; "
                         f"tail:\n{output[-1500:]}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", required=True, help="path to the built quench-cli")
    ap.add_argument("--model", required=True, help="path to the GGUF model")
    ap.add_argument("--pp", type=int, default=512, help="synthetic prompt tokens")
    ap.add_argument("--tg", type=int, default=128, help="decode tokens per rep")
    ap.add_argument("--reps", type=int, default=1,
                    help="quench-cli internal reps (the eval bot does its own "
                         "rep loop with arm alternation, so it calls with 1)")
    ap.add_argument("extra", nargs="*",
                    help="extra quench-cli args after `--` (e.g. --vram-budget)")
    args = ap.parse_args()

    cmd = [args.cli, "--model", args.model, "--bench",
           "--bench-pp", str(args.pp), "--max-tokens", str(args.tg),
           "--bench-reps", str(args.reps), *args.extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print(f"quench-cli --bench failed (exit {r.returncode})", file=sys.stderr)
        return 1
    tok = parse_bench(r.stdout + r.stderr)
    print(json.dumps({
        "schema": "quench-bench/1",
        "workloads": [{"name": w, "tok_per_s": tok[w]} for w in ("pp", "tg")],
        "health": "ok",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
