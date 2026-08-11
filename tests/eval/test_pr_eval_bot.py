"""Policy tests for the PR eval bot — GPU-free, stdlib-only.

The bot spends real money and posts public verdicts, so every decision it
makes (who is eligible, what counts as evaluated, how a verdict is derived
from measurements) is a pure function tested here without a GPU, a box, or a
GitHub token. The shell/ssh plumbing is deliberately not mocked-and-tested:
its correctness is established by the live smoke run, and a mock of rsync
would only test the mock.

Runs as `python3 tests/eval/test_pr_eval_bot.py` and via
`ctest -L unit` (unit_eval_policy).
"""
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bot = _load("pr_eval_bot", "scripts/pr_eval_bot.py")
bench = _load("decode_speed", "bench/decode_speed.py")


class TickedCheckbox(unittest.TestCase):
    def test_ticked(self):
        self.assertTrue(bot.ticked_5090("x\n- [x] Tested on **RTX 5090** — done\ny"))

    def test_capital_x_and_spaces(self):
        self.assertTrue(bot.ticked_5090("- [ X ] Tested on RTX 5090"))

    def test_unticked(self):
        self.assertFalse(bot.ticked_5090("- [ ] Tested on **RTX 5090**"))

    def test_tick_on_a_non_5090_line_does_not_count(self):
        self.assertFalse(bot.ticked_5090("- [x] Parity test added\n- [ ] RTX 5090"))

    def test_no_body(self):
        self.assertFalse(bot.ticked_5090(None))
        self.assertFalse(bot.ticked_5090(""))


class RuntimePaths(unittest.TestCase):
    def test_engine_tests_and_build_files_count(self):
        self.assertTrue(bot.touches_runtime(["src/runtime/engine.cpp"]))
        self.assertTrue(bot.touches_runtime(["README.md", "tests/test_quant.cu"]))
        self.assertTrue(bot.touches_runtime(["CMakeLists.txt"]))
        self.assertTrue(bot.touches_runtime(["bench/decode_speed.py"]))

    def test_docs_and_scripts_do_not(self):
        self.assertFalse(bot.touches_runtime(["docs/architecture.md", "scripts/x.sh"]))

    def test_prefix_is_a_directory_not_a_substring(self):
        self.assertFalse(bot.touches_runtime(["srcs/other.cpp"]))


class Idempotency(unittest.TestCase):
    SHA = "abc123def456"

    def test_verdict_parks_the_head(self):
        bodies = ["hi", bot.marker(self.SHA) + "\nverdict"]
        self.assertTrue(bot.already_evaluated(bodies, self.SHA))

    def test_other_heads_are_not_parked(self):
        bodies = [bot.marker("othersha") + "\nverdict"]
        self.assertFalse(bot.already_evaluated(bodies, self.SHA))

    def test_one_error_gets_a_retry_two_do_not(self):
        one = [bot.error_marker(self.SHA)]
        self.assertFalse(bot.already_evaluated(one, self.SHA))
        two = one + ["x", bot.error_marker(self.SHA)]
        self.assertTrue(bot.already_evaluated(two, self.SHA))

    def test_none_bodies_are_tolerated(self):
        self.assertFalse(bot.already_evaluated([None, ""], self.SHA))


class BenchParsing(unittest.TestCase):
    OUT = (
        "Benchmark: pp=512, tg=128, reps=1\n"
        '{"unrelated": 1}\n'
        '{"health":"ok","schema":"quench-bench/1","workloads":'
        '[{"name":"pp","tok_per_s":700.5},{"name":"tg","tok_per_s":189.4}]}\n'
    )

    def test_picks_workload(self):
        self.assertEqual(bot.pick_tok_s(self.OUT, "pp"), 700.5)
        self.assertEqual(bot.pick_tok_s(self.OUT, "tg"), 189.4)

    def test_missing_workload_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            bot.pick_tok_s(self.OUT, "tg-graphs-off")

    def test_no_json_is_an_error(self):
        with self.assertRaises(ValueError):
            bot.pick_tok_s("OOM during prefill: ...\n", "tg")


class CliOutputParsing(unittest.TestCase):
    """bench/decode_speed.py's scrape of quench-cli --bench stderr."""
    CLI = (
        "Benchmark: pp=512, tg=128, reps=1\n"
        "Warmup...\n"
        "pp   512 tokens  avg   731.43 ms  ( 700.00 tok/s)  [1 reps]\n"
        "tg   128 tokens  avg   677.25 ms  ( 189.00 tok/s)  [1 reps]\n"
    )

    def test_parses_both_figures(self):
        self.assertEqual(bench.parse_bench(self.CLI), {"pp": 700.0, "tg": 189.0})

    def test_missing_figure_is_an_error(self):
        with self.assertRaises(ValueError):
            bench.parse_bench("pp   512 tokens  avg 1.0 ms  (  1.00 tok/s)  [1 reps]\n")


class SuiteShape(unittest.TestCase):
    BASE = ["unit_core", "unit_text", "gpu_quant", "gpu_e2e"]

    def test_identical_suite_is_clean(self):
        self.assertEqual(bot.missing_tests(self.BASE, list(self.BASE)), [])

    def test_added_tests_are_fine(self):
        self.assertEqual(bot.missing_tests(self.BASE, self.BASE + ["gpu_new"]), [])

    def test_dropped_or_renamed_test_is_flagged(self):
        pr = ["unit_core", "unit_text", "gpu_quant_v2", "gpu_e2e"]
        self.assertEqual(bot.missing_tests(self.BASE, pr), ["gpu_quant"])


class Verdict(unittest.TestCase):
    def test_suite_failure_rejects_regardless_of_numbers(self):
        label, _ = bot.verdict(False, {"tg": 50.0})
        self.assertEqual(label, "eval:reject")

    def test_regression_outranks_improvement(self):
        label, reason = bot.verdict(True, {"tg": -3.0, "tg2": 8.0})
        self.assertEqual(label, "eval:reject")
        self.assertIn("tg", reason)

    def test_gain_beyond_the_bar_passes(self):
        label, reason = bot.verdict(True, {"tg": 5.2})
        self.assertEqual(label, "eval:pass")
        self.assertIn("tg", reason)

    def test_within_the_bar_is_noise_not_pass(self):
        label, _ = bot.verdict(True, {"tg": 1.4})
        self.assertEqual(label, "eval:noise")

    def test_the_bar_itself_is_noise(self):
        label, _ = bot.verdict(True, {"tg": 1.5})
        self.assertEqual(label, "eval:noise")
        label, _ = bot.verdict(True, {"tg": -1.5})
        self.assertEqual(label, "eval:noise")

    def test_just_beyond_the_bar_resolves(self):
        # the deltas that only the 1.5% bar (not the old 2.0%) decides —
        # these are the cases the threshold change exists to move
        self.assertEqual(bot.verdict(True, {"tg": 1.6})[0], "eval:pass")
        self.assertEqual(bot.verdict(True, {"tg": -1.6})[0], "eval:reject")

    def test_the_bar_is_the_single_sourced_constant(self):
        # guards against a future edit that moves NOISE_PCT but leaves a
        # hardcoded figure behind in verdict()
        self.assertEqual(bot.NOISE_PCT, 1.5)
        eps = 0.01
        self.assertEqual(bot.verdict(True, {"tg": bot.NOISE_PCT + eps})[0],
                         "eval:pass")
        self.assertEqual(bot.verdict(True, {"tg": bot.NOISE_PCT - eps})[0],
                         "eval:noise")

    def test_the_reason_string_renders_the_bar_faithfully(self):
        # `:.0f` renders 1.5 as "2" — the reason must not claim a bar the
        # code does not gate on
        _, reason = bot.verdict(True, {"tg": 0.1})
        self.assertIn("1.5%", reason)
        self.assertNotIn("±2%", reason)


class Taint(unittest.TestCase):
    def test_engine_changes_do_not_taint(self):
        t, u = bot.classify_taint([("M", "src/runtime/engine.cpp"),
                                   ("A", "src/compute/new_kernel.cu"),
                                   ("M", "CMakeLists.txt")])
        self.assertEqual((t, u), ([], []))

    def test_modified_bench_taints(self):
        t, _ = bot.classify_taint([("M", "bench/decode_speed.py")])
        self.assertEqual(t, ["bench/decode_speed.py"])

    def test_modified_or_deleted_test_taints(self):
        t, _ = bot.classify_taint([("M", "tests/test_quant.cu")])
        self.assertTrue(t)
        t, _ = bot.classify_taint([("D", "tests/test_e2e.cpp")])
        self.assertTrue(t)

    def test_weakened_reference_oracle_taints(self):
        t, _ = bot.classify_taint([("M", "tests/refs/e2e_greedy_locks.h")])
        self.assertEqual(t, ["tests/refs/e2e_greedy_locks.h"])

    def test_added_test_is_unexercised_not_tainted(self):
        t, u = bot.classify_taint([("A", "tests/test_new_kernel.cu")])
        self.assertEqual(t, [])
        self.assertEqual(u, ["tests/test_new_kernel.cu"])

    def test_touching_the_bot_or_workflows_taints(self):
        t, _ = bot.classify_taint([("M", "scripts/pr_eval_bot.py")])
        self.assertTrue(t)
        t, _ = bot.classify_taint([("A", ".github/workflows/evil.yml")])
        self.assertTrue(t)

    def test_prefix_is_a_directory_not_a_substring(self):
        t, u = bot.classify_taint([("M", "benchmarks_notes.md"),
                                   ("M", "tests_helper.py")])
        self.assertEqual((t, u), ([], []))

    def test_other_scripts_do_not_taint(self):
        t, _ = bot.classify_taint([("M", "scripts/smoke_test.sh")])
        self.assertEqual(t, [])

    def test_supply_chain_paths_taint(self):
        # touches_runtime() does not cover these, so without a taint they
        # would ride a "eval not required" success status into an
        # unattended merge. Dependency pins and the container entrypoint
        # are the highest-value target in the repo.
        for path in ("cmake/quench-deps.cmake", "Dockerfile",
                     "docker-entrypoint.sh", "docker-compose.yml"):
            with self.subTest(path=path):
                t, _ = bot.classify_taint([("M", path)])
                self.assertEqual(t, [path])

    def test_added_supply_chain_file_taints_too(self):
        # unlike tests/, an ADDED cmake module is not harmless: it can be
        # include()d by an existing one
        t, u = bot.classify_taint([("A", "cmake/evil.cmake")])
        self.assertEqual(t, ["cmake/evil.cmake"])
        self.assertEqual(u, [])

    def test_supply_chain_prefix_is_a_directory_not_a_substring(self):
        t, u = bot.classify_taint([("M", "cmakelists_notes.md"),
                                   ("M", "docs/Dockerfile.md")])
        self.assertEqual((t, u), ([], []))

    def test_root_cmakelists_still_does_not_taint(self):
        # build files are engine-side (they produce the artifact being
        # measured); the ctest -N superset check is what guards them
        t, _ = bot.classify_taint([("M", "CMakeLists.txt")])
        self.assertEqual(t, [])


class OverlayHarness(unittest.TestCase):
    @staticmethod
    def _write(root, rel, text):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)

    def test_pinned_dirs_come_from_base_wholesale(self):
        with tempfile.TemporaryDirectory() as tmp:
            pr, base = os.path.join(tmp, "pr"), os.path.join(tmp, "base")
            self._write(pr, "src/runtime/engine.cpp", "PR ENGINE")
            self._write(pr, "bench/decode_speed.py", "CHEAT")
            self._write(pr, "tests/test_e2e.cpp", "WEAKENED")
            self._write(pr, "tests/api/conftest.py", "MONKEYPATCH")
            self._write(base, "bench/decode_speed.py", "TRUSTED")
            self._write(base, "tests/test_e2e.cpp", "TRUSTED")
            bot.overlay_harness(pr, base)
            read = lambda rel: open(os.path.join(pr, rel)).read()  # noqa: E731
            self.assertEqual(read("src/runtime/engine.cpp"), "PR ENGINE")
            self.assertEqual(read("bench/decode_speed.py"), "TRUSTED")
            self.assertEqual(read("tests/test_e2e.cpp"), "TRUSTED")
            # wholesale replace: a PR-added conftest must NOT survive into
            # the pinned suite's process
            self.assertFalse(os.path.exists(os.path.join(pr, "tests/api/conftest.py")))

    def test_dir_missing_in_base_is_removed_from_pr(self):
        with tempfile.TemporaryDirectory() as tmp:
            pr, base = os.path.join(tmp, "pr"), os.path.join(tmp, "base")
            self._write(pr, "bench/new_harness.py", "PR HARNESS")
            os.makedirs(base, exist_ok=True)
            bot.overlay_harness(pr, base)
            self.assertFalse(os.path.exists(os.path.join(pr, "bench")))


class TaintedVerdict(unittest.TestCase):
    def test_taint_caps_a_pass_at_tainted(self):
        label, reason = bot.verdict(True, {"tg": 5.2},
                                    tainted=["bench/decode_speed.py"])
        self.assertEqual(label, "eval:tainted")
        self.assertIn("harness", reason)

    def test_taint_caps_noise_too(self):
        # a tainted no-delta PR used to read eval:noise, which was safe while
        # merging was manual. eval:noise now arms auto-merge, and a Dockerfile
        # or cmake/ edit has no reason to move decode throughput — so the
        # noise path is exactly the one such a diff would take.
        label, reason = bot.verdict(True, {"tg": 1.0}, tainted=["Dockerfile"])
        self.assertEqual(label, "eval:tainted")
        self.assertIn("supply-chain", reason)
        self.assertNotIn(label, bot.AUTO_MERGE_LABELS)

    def test_taint_does_not_upgrade_a_reject(self):
        self.assertEqual(bot.verdict(True, {"tg": -9.0}, tainted=["tests/t.cu"])[0],
                         "eval:reject")
        self.assertEqual(bot.verdict(False, {}, tainted=["tests/t.cu"])[0],
                         "eval:reject")

    def test_untainted_pass_is_still_a_pass(self):
        self.assertEqual(bot.verdict(True, {"tg": 5.0}, tainted=[])[0], "eval:pass")


def synthetic_bundle(**kw):
    base = {
        "schema": bot.scorer.SCHEMA_IN,
        "pr": 7, "head": "a" * 40, "eval_sha": "b" * 40, "base_sha": "c" * 40,
        "mode": "merge-vs-main", "workloads": ["pp", "tg"],
        "bench": {"pp_tokens": 512, "tg_tokens": 128, "batch": 1},
        "reps": 3, "noise_pct": bot.NOISE_PCT, "box": "47055458",
        "model": "Mistral-Nemo-Instruct-2407-Q8_0.gguf", "tests_ok": True,
        "suite_tail_sha256": "0" * 64,
        "samples": {"pr": {"pp": [720.0, 715.0, 718.0], "tg": [199.0, 198.5, 199.4]},
                    "main": {"pp": [700.0, 702.0, 699.0], "tg": [189.0, 189.2, 188.8]}},
        "name_status": [["M", "src/runtime/engine.cpp"]],
        "integrity": {"model_sha256": "m" * 64, "main_build_hash": "e" * 64},
    }
    base.update(kw)
    return base


class Scorer(unittest.TestCase):
    def test_policy_is_single_sourced(self):
        # the bot must not grow its own copies — same function objects
        self.assertIs(bot.verdict, bot.scorer.verdict)
        self.assertIs(bot.classify_taint, bot.scorer.classify_taint)
        self.assertEqual(bot.NOISE_PCT, bot.scorer.NOISE_PCT)
        self.assertEqual(bot.GATED, bot.scorer.GATED)

    def test_score_bundle_pass(self):
        v = bot.scorer.score_bundle(synthetic_bundle())
        self.assertEqual(v["label"], "eval:pass")
        self.assertEqual(v["deltas_pct"].keys(), {"pp", "tg"})
        self.assertAlmostEqual(v["deltas_pct"]["tg"], 5.291, places=3)

    def test_prefill_is_informational_not_gated(self):
        # a pp regression must not reject while tg is clean — prefill varies
        # up to 2.6x with cuBLAS algorithm selection (CONTRIBUTING.md)
        b = synthetic_bundle()
        b["samples"]["pr"]["pp"] = [500.0, 501.0, 499.0]     # pp -28%
        v = bot.scorer.score_bundle(b)
        self.assertEqual(v["label"], "eval:pass")
        # and a pp gain alone must not pass while tg is noise
        b = synthetic_bundle()
        b["samples"]["pr"]["pp"] = [900.0, 901.0, 899.0]
        b["samples"]["pr"]["tg"] = [189.1, 189.3, 188.9]
        self.assertEqual(bot.scorer.score_bundle(b)["label"], "eval:noise")

    def test_tg_regression_rejects(self):
        b = synthetic_bundle()
        b["samples"]["pr"]["tg"] = [180.0, 180.2, 179.8]
        self.assertEqual(bot.scorer.score_bundle(b)["label"], "eval:reject")

    def test_score_bundle_taint_caps_pass(self):
        b = synthetic_bundle(name_status=[["M", "bench/decode_speed.py"]])
        self.assertEqual(bot.scorer.score_bundle(b)["label"], "eval:tainted")

    def test_score_bundle_suite_failure(self):
        v = bot.scorer.score_bundle(synthetic_bundle(tests_ok=False, samples={}))
        self.assertEqual(v["label"], "eval:reject")
        self.assertEqual(v["medians"], {})

    def test_gated_workload_missing_is_an_error(self):
        b = synthetic_bundle(workloads=["pp"])
        with self.assertRaises(ValueError):
            bot.scorer.score_bundle(b)

    def test_scorer_cli_matches_in_process(self):
        # what the TEE prints must equal what the bot computes locally
        import subprocess
        import sys
        bundle = synthetic_bundle()
        expected = bot.scorer.canonical(bot.scorer.score_bundle(bundle)) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bundle.json")
            with open(p, "w") as f:
                json.dump(bundle, f)
            out = subprocess.run([sys.executable, "-B", str(bot._SCORER_PATH), p],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout, expected)

    def test_canonical_is_stable_under_key_order(self):
        a = bot.scorer.canonical({"b": 1, "a": [1, 2]})
        b = bot.scorer.canonical({"a": [1, 2], "b": 1})
        self.assertEqual(a, b)
        self.assertNotIn(" ", a)

    def test_build_bundle_round_trips_through_scorer(self):
        samples = {"pr": {"pp": [720.0], "tg": [199.0]},
                   "main": {"pp": [700.0], "tg": [189.0]}}
        bundle = bot.build_bundle(9, "d" * 40, "e" * 40, "f" * 40, "merge-vs-main",
                                  1, True, samples,
                                  [("A", "tests/test_new.cu")], "suite tail text",
                                  "m" * 64, "e" * 64, 512, 128)
        v = bot.scorer.score_bundle(bundle)
        self.assertEqual(v["label"], "eval:pass")
        self.assertEqual(v["unexercised"], ["tests/test_new.cu"])
        # canonicalizable (json round-trip identical)
        s = bot.scorer.canonical(bundle)
        self.assertEqual(bot.scorer.canonical(json.loads(s)), s)

    def test_touching_the_scorer_itself_taints(self):
        t, _ = bot.classify_taint([("M", "scripts/eval_scorer.py")])
        self.assertTrue(t)


class DurableEvidence(unittest.TestCase):
    def test_every_label_maps_to_a_commit_status_state(self):
        self.assertEqual(set(bot.STATUS_STATE), set(bot.EVAL_LABELS))
        self.assertTrue(set(bot.STATUS_STATE.values())
                        <= {"success", "failure", "error", "pending"})

    def test_tainted_and_reject_block_a_required_status(self):
        self.assertEqual(bot.STATUS_STATE["eval:tainted"], "failure")
        self.assertEqual(bot.STATUS_STATE["eval:reject"], "failure")
        self.assertEqual(bot.STATUS_STATE["eval:pass"], "success")


class AutoMerge(unittest.TestCase):
    """The label is what .github/workflows/auto-merge.yml keys on, and the
    status is what main's branch protection requires. If those two disagree
    the workflow arms a merge the required check then blocks forever."""

    def test_auto_merge_labels_are_real_labels(self):
        self.assertTrue(set(bot.AUTO_MERGE_LABELS) <= set(bot.EVAL_LABELS))

    def test_every_auto_merge_label_is_a_success_status(self):
        for label in bot.AUTO_MERGE_LABELS:
            with self.subTest(label=label):
                self.assertEqual(bot.STATUS_STATE[label], "success")

    def test_no_other_label_is_a_success_status(self):
        # the converse: a success status that is NOT an auto-merge label would
        # satisfy branch protection while the workflow declines to arm — a PR
        # that is mergeable but never merges
        success = {lb for lb, st in bot.STATUS_STATE.items() if st == "success"}
        self.assertEqual(success, set(bot.AUTO_MERGE_LABELS))

    def test_holding_verdicts_are_excluded(self):
        for label in ("eval:tainted", "eval:reject", "eval:error"):
            with self.subTest(label=label):
                self.assertNotIn(label, bot.AUTO_MERGE_LABELS)


class ScenarioMatrix(unittest.TestCase):
    """Every PR shape the bot can meet, end to end: measurements in, then the
    three things that actually decide the PR's fate — the label the maintainer
    sees, the commit status branch protection reads, and whether
    .github/workflows/auto-merge.yml arms an unattended merge.

    The per-rule tests above check each hop in isolation. This one exists
    because the hops are wired together by hand (verdict -> STATUS_STATE ->
    AUTO_MERGE_LABELS -> a string literal in a YAML file), and a merge gate
    that is right in three places and wrong in the join is still wrong.
    """

    MAIN_TG = 200.0

    def bundle(self, *, delta_pct=0.0, name_status=(("M", "src/runtime/engine.cpp"),),
               tests_ok=True):
        pr_tg = self.MAIN_TG * (1 + delta_pct / 100.0)
        if not tests_ok:
            return synthetic_bundle(tests_ok=False, samples={},
                                    name_status=[list(p) for p in name_status])
        return synthetic_bundle(
            tests_ok=True,
            name_status=[list(p) for p in name_status],
            samples={"pr": {"pp": [700.0], "tg": [pr_tg]},
                     "main": {"pp": [700.0], "tg": [self.MAIN_TG]}})

    # scenario, kwargs, expected label, expected commit status, merges unattended
    CASES = (
        ("clean speedup well beyond the bar",
         dict(delta_pct=6.0), "eval:pass", "success", True),
        ("clean speedup just beyond the bar",
         dict(delta_pct=1.6), "eval:pass", "success", True),
        ("exactly on the bar is noise, not a pass",
         dict(delta_pct=1.5), "eval:noise", "success", True),
        ("perf-neutral bugfix",
         dict(delta_pct=0.0), "eval:noise", "success", True),
        ("small regression inside the bar",
         dict(delta_pct=-1.4), "eval:noise", "success", True),
        ("regression just beyond the bar",
         dict(delta_pct=-1.6), "eval:reject", "failure", False),
        ("regression well beyond the bar",
         dict(delta_pct=-20.0), "eval:reject", "failure", False),
        ("suite failure outranks a huge speedup",
         dict(delta_pct=50.0, tests_ok=False), "eval:reject", "failure", False),
        ("speedup while editing the pinned bench",
         dict(delta_pct=6.0, name_status=(("M", "bench/decode_speed.py"),)),
         "eval:tainted", "failure", False),
        ("speedup while weakening a pinned oracle",
         dict(delta_pct=6.0, name_status=(("M", "tests/refs/e2e_greedy_locks.h"),)),
         "eval:tainted", "failure", False),
        ("speedup while editing the scorer itself",
         dict(delta_pct=6.0, name_status=(("M", "scripts/eval_scorer.py"),)),
         "eval:tainted", "failure", False),
        ("speedup while editing a workflow",
         dict(delta_pct=6.0, name_status=(("M", ".github/workflows/ci.yml"),)),
         "eval:tainted", "failure", False),
        # the cases the supply-chain taint was added for: no plausible effect
        # on decode throughput, so without the taint they ride the noise path
        ("engine change plus a dependency-pin edit, no delta",
         dict(delta_pct=0.2, name_status=(("M", "src/runtime/engine.cpp"),
                                          ("M", "cmake/quench-deps.cmake"))),
         "eval:tainted", "failure", False),
        ("engine change plus a Dockerfile edit, no delta",
         dict(delta_pct=0.0, name_status=(("M", "src/compute/gemm.cu"),
                                          ("M", "Dockerfile"))),
         "eval:tainted", "failure", False),
        ("engine change plus a container-entrypoint edit, no delta",
         dict(delta_pct=-0.3, name_status=(("M", "src/compute/gemm.cu"),
                                           ("M", "docker-entrypoint.sh"))),
         "eval:tainted", "failure", False),
        ("regression outranks the taint",
         dict(delta_pct=-9.0, name_status=(("M", "Dockerfile"),)),
         "eval:reject", "failure", False),
        # additions under pinned paths are reported, not punished
        ("speedup plus a newly added test",
         dict(delta_pct=6.0, name_status=(("M", "src/compute/gemm.cu"),
                                          ("A", "tests/test_new_kernel.cu"))),
         "eval:pass", "success", True),
        ("root CMakeLists edit is engine-side, not a taint",
         dict(delta_pct=6.0, name_status=(("M", "CMakeLists.txt"),)),
         "eval:pass", "success", True),
    )

    def test_matrix(self):
        for scenario, kw, label, status, merges in self.CASES:
            with self.subTest(scenario=scenario):
                v = bot.scorer.score_bundle(self.bundle(**kw))
                self.assertEqual(v["label"], label, v["reason"])
                self.assertEqual(bot.STATUS_STATE[v["label"]], status)
                self.assertEqual(v["label"] in bot.AUTO_MERGE_LABELS, merges,
                                 f"{scenario}: wrong merge decision")

    def test_matrix_covers_every_label_the_bot_can_apply(self):
        # eval:error is not reachable through the scorer — it is the
        # exception path in evaluate() — so it is excluded by name here
        # rather than silently missing from the matrix.
        seen = {bot.scorer.score_bundle(self.bundle(**kw))["label"]
                for _, kw, _, _, _ in self.CASES}
        self.assertEqual(seen, set(bot.EVAL_LABELS) - {"eval:error"})

    def test_eval_error_never_merges(self):
        self.assertEqual(bot.STATUS_STATE["eval:error"], "error")
        self.assertNotIn("eval:error", bot.AUTO_MERGE_LABELS)

    def test_no_tainted_scenario_is_ever_mergeable(self):
        # the property behind the individual rows: taint and merge are
        # mutually exclusive whatever the measurement says
        for delta in (-50.0, -1.6, -1.5, 0.0, 1.5, 1.6, 50.0):
            with self.subTest(delta=delta):
                v = bot.scorer.score_bundle(
                    self.bundle(delta_pct=delta,
                                name_status=(("M", "cmake/quench-deps.cmake"),)))
                self.assertNotIn(v["label"], bot.AUTO_MERGE_LABELS)


class WorkflowWiring(unittest.TestCase):
    """The last hop is a string literal in a YAML file that no Python test
    would otherwise touch. Checked as raw text on purpose: what matters is
    exactly what is written in the file, not what a YAML loader normalizes it
    to, and keeping this stdlib-only keeps the suite runnable as a bare
    `python3` script under ctest."""

    @classmethod
    def setUpClass(cls):
        cls.text = (_ROOT / ".github/workflows/auto-merge.yml").read_text()

    def test_the_workflow_gates_on_exactly_the_bots_merge_labels(self):
        import re
        found = set(re.findall(r"github\.event\.label\.name\s*==\s*'([^']+)'",
                               self.text))
        self.assertEqual(found, set(bot.AUTO_MERGE_LABELS),
                         "auto-merge.yml and AUTO_MERGE_LABELS have drifted")

    def test_every_gated_label_is_one_the_bot_actually_applies(self):
        import re
        found = set(re.findall(r"github\.event\.label\.name\s*==\s*'([^']+)'",
                               self.text))
        self.assertTrue(found <= set(bot.EVAL_LABELS),
                        "workflow waits for a label the bot never applies")

    def test_privileged_trigger_never_checks_out_pr_code(self):
        # pull_request_target runs with a write token in the base repo's
        # context; a checkout here would execute fork code with that token
        self.assertIn("pull_request_target", self.text)
        self.assertNotIn("actions/checkout", self.text)

    def test_privileged_trigger_is_limited_to_the_labeled_event(self):
        import re
        block = self.text.split("pull_request_target:", 1)[1]
        types = re.search(r"types:\s*\[([^\]]*)\]", block).group(1)
        self.assertEqual([t.strip() for t in types.split(",")], ["labeled"])

    def test_draft_prs_are_excluded_from_both_paths(self):
        self.assertEqual(self.text.count("draft == false"), 2)


class Eligibility(unittest.TestCase):
    def info(self, **kw):
        base = {
            "state": "OPEN", "isDraft": False, "labels": [],
            "headRefOid": "feedbeef1234",
            "body": "- [x] Tested on **RTX 5090**",
            "files": [{"path": "src/runtime/engine.cpp"}], "comments": [],
        }
        base.update(kw)
        return base

    def test_happy_path(self):
        ok, why = bot.eligible(self.info())
        self.assertTrue(ok, why)

    def test_draft_and_hold_are_skipped(self):
        self.assertFalse(bot.eligible(self.info(isDraft=True))[0])
        self.assertFalse(bot.eligible(self.info(labels=[{"name": "hold"}]))[0])

    def test_docs_only_is_skipped(self):
        ok, why = bot.eligible(self.info(files=[{"path": "README.md"}]))
        self.assertFalse(ok)
        self.assertEqual(why, bot.NON_RUNTIME_REASON)

    def test_unticked_needs_the_eval_label(self):
        no_tick = self.info(body="- [ ] Tested on **RTX 5090**")
        self.assertFalse(bot.eligible(no_tick)[0])
        forced = self.info(body="- [ ] Tested on **RTX 5090**",
                           labels=[{"name": "eval"}])
        self.assertTrue(bot.eligible(forced)[0])

    def test_evaluated_head_is_parked(self):
        done = self.info(comments=[{"body": bot.marker("feedbeef1234")}])
        self.assertFalse(bot.eligible(done)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
