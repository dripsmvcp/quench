#!/bin/bash
# verify.sh — pre-commit/pre-push verification for quench.
#
# Four-step gate:
#   1. Build (incremental)
#   2. Tests (gtest filter or full suite)
#   3. CUDA-graphs ON vs OFF decode gate (catches broken graph capture)
#   4. Smoke prompt on a real model (degeneration detector)
#
# Modes:
#   verify.sh fast    Unit tests + graphs gate + smoke prompt   (~90s)
#   verify.sh full    Full ctest + graphs gate + smoke prompt   (~5min)
#
# Env overrides:
#   QUENCH_VERIFY_BIN=build/quench-cli
#   QUENCH_VERIFY_TESTS=build/quench-tests
#   QUENCH_VERIFY_MODELS=models
#   QUENCH_VERIFY_SKIP_BUILD=1   skip cmake build step
#   QUENCH_VERIFY_SKIP_GRAPHS=1  skip the graphs ON/OFF decode gate
#   QUENCH_VERIFY_IN_DOCKER=1    sentinel set by the auto-re-exec block; do not set manually
#
# Auto-Docker fallback: if cmake is not on PATH (Clean-Host workflow), the
# script re-execs itself inside the quench:test container, mounting the repo at
# /src and using the prebuilt /usr/local/bin/quench-cli + quench-tests. Requires
# 'make build' to have produced the quench:test image first.
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Auto re-exec in the quench:test container when cmake is unavailable on the host
# (Clean-Host workflow: no language toolchains installed). The runtime image
# has prebuilt binaries at /usr/local/bin/, so we skip the build step and point
# QUENCH_VERIFY_BIN/TESTS at them. QUENCH_VERIFY_IN_DOCKER guards against infinite
# re-exec if the runtime image somehow also lacks cmake.
if ! command -v cmake >/dev/null 2>&1 && [ "${QUENCH_VERIFY_IN_DOCKER:-0}" != "1" ]; then
    if ! docker image inspect quench:test >/dev/null 2>&1; then
        echo "verify: cmake not found on host and quench:test image not built." >&2
        echo "        Run 'make build' first, then re-run." >&2
        exit 1
    fi
    echo "verify: host cmake unavailable — re-executing in quench:test container"
    exec docker run --rm --gpus all \
        -v "$ROOT":/src -w /src \
        -v "$HOME/models":"$HOME/models":ro \
        -e QUENCH_VERIFY_IN_DOCKER=1 \
        -e QUENCH_VERIFY_BIN=/usr/local/bin/quench-cli \
        -e QUENCH_VERIFY_TESTS=/usr/local/bin/quench-tests \
        -e QUENCH_VERIFY_SKIP_BUILD=1 \
        -e QUENCH_VERIFY_SKIP_GRAPHS="${QUENCH_VERIFY_SKIP_GRAPHS:-0}" \
        --entrypoint bash quench:test scripts/verify.sh "$@"
fi

MODE="${1:-fast}"

BIN="${QUENCH_VERIFY_BIN:-build/quench-cli}"
TESTS_BIN="${QUENCH_VERIFY_TESTS:-build/quench-tests}"
MODELS="${QUENCH_VERIFY_MODELS:-models}"

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[0;33m'; RST=$'\033[0m'
FAIL=0
section() { echo; echo "${YLW}== $* ==${RST}"; }
pass()    { echo "${GRN}PASS${RST} $*"; }
fail()    { echo "${RED}FAIL${RST} $*"; FAIL=$((FAIL+1)); }
skip()    { echo "${YLW}SKIP${RST} $*"; }
warn()    { echo "${YLW}WARN${RST} $*"; }

# --- Host-drift guard -------------------------------------------------
# This WSL2 box has day-level "depressed host" states where decode reads 8-15%
# low DESPITE full methodology (a known host artifact). The
# culprit is host/driver state, not code — but a bench-based gate cannot tell the
# two apart from a single bench number. So we sample GPU clocks/power DURING the
# decode bench and, if a regression coincides with the depressed-host signature,
# we degrade FAIL→WARN (clearly labeled) instead of crying false-positive.
#
# Healthy under-load signature: ~2850 MHz SM / 13801 MHz mem / ~500 W.
# Depressed         => mem-clock median < 13801 MHz, OR power max < 400 W,
#                       OR SM-clock median < 2000 MHz (healthy decode load is
#                       ~2850 MHz sustained; a depressed host can pin SM
#                       in the hundreds of MHz with mem partly downclocked).
#   - mem-clock is the cleanest tell (GDDR7 either P0-clocks or it doesn't).
#   - power uses MAX over the run (a robust upper bound) with a conservative
#     400 W floor so a healthy ~500 W run never trips it.
#   - The SM clock ramps ~1s at bench start (cold-start artifact, audit §5), so
#     we drop the first 2 samples before aggregating to avoid a false depressed
#     classification from the ramp. A run with only 1-2 samples can't drop the
#     ramp, so it's treated as no-data (gate fails open: plain FAIL).
# Fail-open: if nvidia-smi is missing or errors, the sampler no-ops and the gate
# behaves exactly as before (no degradation logic kicks in).
GPU_DRIFT_MEM_FLOOR=13801   # mem clock (MHz) at/above which the host is healthy
GPU_DRIFT_POWER_FLOOR=400   # power (W) max below which the host is depressed
GPU_DRIFT_SM_FLOOR=2000     # SM clock (MHz) median below which the host is depressed
_SAMPLE_FILE=""
_SAMPLE_PID=""

# Start a background sampler writing "sm,mem,power" CSV rows every 1s.
# No-op (leaves _SAMPLE_PID empty) if nvidia-smi can't be queried.
gpu_sample_start() {
    _SAMPLE_FILE=""; _SAMPLE_PID=""
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw \
        --format=csv,noheader,nounits >/dev/null 2>&1 || return 0
    _SAMPLE_FILE=$(mktemp)
    ( while true; do
        nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw \
            --format=csv,noheader,nounits 2>/dev/null | head -1
        sleep 1
      done ) >>"$_SAMPLE_FILE" 2>/dev/null &
    _SAMPLE_PID=$!
}

# Stop the sampler and classify the run. Sets globals:
#   GPU_DRIFT_DEPRESSED = 0|1
#   GPU_DRIFT_DESC      = human-readable "mem=… power=…" summary ("" if no data)
gpu_sample_stop() {
    GPU_DRIFT_DEPRESSED=0
    GPU_DRIFT_DESC=""
    [ -n "$_SAMPLE_PID" ] && kill "$_SAMPLE_PID" >/dev/null 2>&1
    [ -n "$_SAMPLE_PID" ] && wait "$_SAMPLE_PID" 2>/dev/null
    [ -n "$_SAMPLE_FILE" ] && [ -s "$_SAMPLE_FILE" ] || { _cleanup_sample; return 0; }
    # Drop the first 2 samples (clock-ramp cold-start), then aggregate:
    # median mem clock + max power over the steady portion of the run.
    read -r MEM_MED PWR_MAX SM_MED N < <(awk -F, '
        { gsub(/ /,""); n++; sm[n]=$1+0; mem[n]=$2+0; pwr[n]=$3+0 }
        END {
            # n<=2 is too short to drop the ~1s cold-ramp samples: classify as
            # no-data (N=0) so the gate fails open (plain FAIL on regression),
            # never letting cold-ramp low-mem samples force a depressed WARN.
            if (n <= 2) { print 0, 0, 0, 0; exit }
            start = 3; cnt = 0; pmax = 0;
            for (i = start; i <= n; i++) {
                m[++cnt] = mem[i]; sm2[cnt] = sm[i];
                if (pwr[i] > pmax) pmax = pwr[i];
            }
            if (cnt == 0) { print 0, 0, 0, 0; exit }
            # medians of m[1..cnt] and sm2[1..cnt]
            for (a = 1; a <= cnt; a++) for (b = a+1; b <= cnt; b++) {
                if (m[b] < m[a]) { t=m[a]; m[a]=m[b]; m[b]=t }
                if (sm2[b] < sm2[a]) { t=sm2[a]; sm2[a]=sm2[b]; sm2[b]=t }
            }
            med = (cnt % 2) ? m[(cnt+1)/2] : (m[cnt/2] + m[cnt/2+1]) / 2;
            smmed = (cnt % 2) ? sm2[(cnt+1)/2] : (sm2[cnt/2] + sm2[cnt/2+1]) / 2;
            printf "%.0f %.0f %.0f %d\n", med, pmax, smmed, cnt;
        }' "$_SAMPLE_FILE")
    _cleanup_sample
    [ "${N:-0}" -gt 0 ] || return 0
    GPU_DRIFT_DESC="sm=${SM_MED}MHz(med) mem=${MEM_MED}MHz(med) power=${PWR_MAX}W(max) n=${N}"
    if [ "$MEM_MED" -lt "$GPU_DRIFT_MEM_FLOOR" ] || [ "$PWR_MAX" -lt "$GPU_DRIFT_POWER_FLOOR" ] ||
       [ "$SM_MED" -lt "$GPU_DRIFT_SM_FLOOR" ]; then
        GPU_DRIFT_DEPRESSED=1
    fi
}

_cleanup_sample() {
    [ -n "$_SAMPLE_FILE" ] && rm -f "$_SAMPLE_FILE"
    _SAMPLE_FILE=""; _SAMPLE_PID=""
}

# Report a decode regression, degrading FAIL→WARN under the depressed-host
# signature. $1 = context label for the message.
decode_regression() {
    local ctx="$1"
    if [ "${GPU_DRIFT_DEPRESSED:-0}" = "1" ]; then
        warn "$ctx: depressed-host signature ($GPU_DRIFT_DESC) — decode delta not attributable to code, FAIL degraded to WARN"
    else
        fail "$ctx"
    fi
}

# Docker-only host (host has neither cmake nor a build/ directory): run
# the canonical Docker build, then exit early — the test / perf / smoke
# gates below need the host build artefacts and there's no clean way to
# reach them from here. Override with QUENCH_VERIFY_SKIP_BUILD=1 if
# cmake-on-host is preferred.
if [ "${QUENCH_VERIFY_SKIP_BUILD:-0}" != "1" ] && ! command -v cmake >/dev/null 2>&1; then
    section "build (docker — host has no cmake)"
    if make build >/tmp/quench_verify_build.log 2>&1; then
        pass "docker build"
    else
        fail "docker build (see /tmp/quench_verify_build.log)"
        tail -20 /tmp/quench_verify_build.log
        exit 1
    fi
    section "remaining gates (skipped — Docker-only host)"
    skip "tests / graphs / smoke require host build artefacts; run 'make test-gpu'"
    skip "and 'make verify' inside Docker manually for the full pre-merge gate"
    exit 0
fi

# -------------------------------------------------------------------- 1. build
section "build"
if [ "${QUENCH_VERIFY_SKIP_BUILD:-0}" = "1" ]; then
    skip "build (QUENCH_VERIFY_SKIP_BUILD=1)"
elif [ ! -d build ]; then
    fail "no build/ directory — run 'cmake -B build -DCMAKE_BUILD_TYPE=Release' first"
else
    if cmake --build build -j"$(nproc)" >/tmp/quench_verify_build.log 2>&1; then
        pass "incremental build"
    else
        fail "build (see /tmp/quench_verify_build.log)"
        tail -20 /tmp/quench_verify_build.log
        exit 1
    fi
fi

# -------------------------------------------------------------------- 2. tests
section "tests"
if [ ! -x "$TESTS_BIN" ]; then
    fail "$TESTS_BIN not found"
else
    if [ "$MODE" = "fast" ]; then
        # VramBudgetReserve is in here because CI structurally cannot run it:
        # compute_vram_budget queries the device for total VRAM, so the suite is
        # SKIP_IF_NO_CUDA and the GPU-less CI lane skips it. Three of its tests
        # can sit red unnoticed for a long time. The pre-push
        # gate is the only place that can catch that class.
        FILTER="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:KVCache*:GemmTest.*:FP8GemmTest.*:SamplingTest.*:SoftmaxTest.*:AttentionTest.*:VramBudget*"
        if "$TESTS_BIN" --gtest_filter="$FILTER" >/tmp/quench_verify_tests.log 2>&1; then
            pass "fast gtest filter"
        else
            fail "gtest (see /tmp/quench_verify_tests.log)"
            tail -30 /tmp/quench_verify_tests.log
        fi
    else
        if "$TESTS_BIN" >/tmp/quench_verify_tests.log 2>&1; then
            pass "full gtest suite"
        else
            fail "gtest (see /tmp/quench_verify_tests.log)"
            grep -E "FAIL|fatal" /tmp/quench_verify_tests.log | head -20
        fi

        # test-e2e unit/gpu lane-split guard (R5/): the unit lane is a
        # gtest_filter, so a rename could silently move a CPU test into the GPU
        # lane. This asserts the filter still resolves to the frozen CPU set.
        # Fail-open: skips cleanly if the binary or script isn't locatable.
        _E2E_BIN="$(dirname "$TESTS_BIN")/test-e2e"
        # DUPLICATE of _unit_e2e_filter in CMakeLists.txt (~line 607) — keep in
        # sync. Can't source it here: the container path runs without a
        # configured build/ dir, so ctest/CMake variables are unreachable.
        # Going stale is caught by the primary ctest guard (guard_e2e_lane_split)
        # plus the frozen EXPECTED list inside check_e2e_lane_split.sh itself.
        _LANE_FILTER="BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub"
        if [ -x "$_E2E_BIN" ] && [ -x scripts/check_e2e_lane_split.sh ]; then
            if scripts/check_e2e_lane_split.sh "$_E2E_BIN" "$_LANE_FILTER" >/tmp/quench_verify_lane.log 2>&1; then
                pass "e2e unit/gpu lane split"
            else
                fail "e2e lane split (see /tmp/quench_verify_lane.log)"
                cat /tmp/quench_verify_lane.log
            fi
        else
            skip "e2e lane-split guard (test-e2e or guard script not found)"
        fi
    fi
fi

# --------------------------- 3. graphs-ON vs graphs-OFF decode regression gate
# Catches future PRs that silently break CUDA Graph capture in the decode loop.
# Without graphs, decode is launch-overhead-bound (875-1170 launches/step on
# dense Q8). Graphs-ON wins by integer factors.
# A graphs-ON improvement that drops below kMinSpeedupX = 1.5× signals a path
# that fell out of capture (host sync inside captured region, malloc on hot
# path, etc.). Skipped by QUENCH_VERIFY_SKIP_GRAPHS=1.
section "graphs ON vs OFF decode gate"
if [ "${QUENCH_VERIFY_SKIP_GRAPHS:-0}" = "1" ]; then
    skip "graphs gate (QUENCH_VERIFY_SKIP_GRAPHS=1)"
elif [ ! -f "$BIN" ]; then
    skip "graphs gate requires host build artefacts"
else
    GRAPHS_MODEL="${QUENCH_VERIFY_GRAPHS_MODEL:-Qwen3-4B-Instruct-2507-Q8_0.gguf}"
    GRAPHS_MODEL_PATH="$MODELS/$GRAPHS_MODEL"
    if [ ! -f "$GRAPHS_MODEL_PATH" ]; then
        skip "graphs gate model $GRAPHS_MODEL_PATH not present"
    else
        # Threshold 1.3: catches catastrophic graph failures (≈ 1.0x = full
        # fallback to per-step decode) without rejecting healthy big-model
        # decodes — bigger models have larger kernel time, hence less
        # launch-overhead share and a lower ON/OFF ratio.
        MIN_SPEEDUP_X="${QUENCH_VERIFY_MIN_GRAPH_SPEEDUP:-1.3}"
        ERR_NG=$(mktemp); ERR_G=$(mktemp)
        # Sample clocks/power across BOTH runs: a depressed host ( — SM
        # pinned low / mem at half / power capped) makes kernels dominate and
        # compresses the ON/OFF ratio toward 1.0x on ANY build (measured
        # 0.85-1.0x on main and branches alike, 2 consecutive nights) — the
        # ratio stops being attributable to code.
        # NOTE: the power floor (400 W) is calibrated on 8B-class decode; the
        # 4B gate model can draw less even when healthy, so this gate leans
        # WARN. That is deliberate: a false WARN costs a re-run on a healthy
        # host, a false FAIL blocks a push on host noise. The hard FAIL still
        # fires whenever the signature reads healthy.
        gpu_sample_start
        "$BIN" --model "$GRAPHS_MODEL_PATH" --bench --bench-pp 256 --bench-reps 2 \
              --max-tokens 256 --temperature 0 --no-cuda-graphs \
              --set speculative.ngram=false >/dev/null 2>"$ERR_NG"
        "$BIN" --model "$GRAPHS_MODEL_PATH" --bench --bench-pp 256 --bench-reps 2 \
              --max-tokens 256 --temperature 0 \
              --set speculative.ngram=false >/dev/null 2>"$ERR_G"
        gpu_sample_stop
        TG_NG=$(grep -oP '^tg\s+256\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR_NG" | head -1)
        TG_G=$(grep -oP '^tg\s+256\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR_G" | head -1)
        if [ -z "$TG_NG" ] || [ -z "$TG_G" ]; then
            fail "could not parse graphs bench output"
            echo "  no-graphs stderr tail:"; tail -8 "$ERR_NG"
            echo "  graphs stderr tail:";    tail -8 "$ERR_G"
        else
            SPEEDUP=$(awk -v g="$TG_G" -v ng="$TG_NG" 'BEGIN{printf "%.3f", g/ng}')
            BELOW=$(awk -v s="$SPEEDUP" -v t="$MIN_SPEEDUP_X" 'BEGIN{print (s < t) ? 1 : 0}')
            printf "  graphs OFF tg256 = %7.2f tok/s\n" "$TG_NG"
            printf "  graphs ON  tg256 = %7.2f tok/s   (%.2fx, threshold %sx)\n" \
                   "$TG_G" "$SPEEDUP" "$MIN_SPEEDUP_X"
            if [ "$BELOW" = "1" ]; then
                if [ "${GPU_DRIFT_DEPRESSED:-0}" = "1" ]; then
                    warn "graphs speedup ${SPEEDUP}x < ${MIN_SPEEDUP_X}x under depressed-host signature ($GPU_DRIFT_DESC) — ratio not attributable to code, FAIL degraded to WARN. Re-run on a healthy host; if it still fails there, look for new host syncs / cudaMalloc on the decode hot path"
                else
                    fail "graph capture broken: speedup ${SPEEDUP}x < ${MIN_SPEEDUP_X}x — \
look for new host syncs / cudaMalloc on the decode hot path"
                fi
            else
                pass "graphs ON delivers ≥${MIN_SPEEDUP_X}x decode speedup"
            fi
        fi
        rm -f "$ERR_NG" "$ERR_G"
    fi
fi

# ------------------------------------------------------------- 4. smoke prompts
section "smoke prompts (degeneration check)"

# Greedy decode on a known-deterministic prompt.
# Quality gate: output must contain expected substring AND last 32 tokens must
# have at least 8 distinct tokens (catches "own own own" stuck-token failures).
smoke_prompt() {
    local label="$1" model="$2" prompt="$3" expect="$4"
    if [ ! -f "$MODELS/$model" ]; then
        skip "$label ($model not present)"
        return
    fi
    local ERR; ERR=$(mktemp)
    OUT=$("$BIN" --model "$MODELS/$model" --prompt "$prompt" \
          --max-tokens 64 --temperature 0 --chat-template none 2>"$ERR")
    # Token markers '[tok=NNNN ' word']' land on stderr.
    TOKS=$(grep -oP '\[tok=\K[0-9]+' "$ERR" | tail -32)
    DISTINCT=$(echo "$TOKS" | sort -u | wc -l)
    # Generated word stream (stripped of token id prefix) for NaN/Inf scan
    WORDS=$(grep -oP "\[tok=[0-9]+ '\K[^']*" "$ERR" | tr '\n' ' ')
    rm -f "$ERR"

    if echo " $WORDS " | grep -qiE ' (nan|inf|-inf|-nan) '; then
        fail "$label — NaN/Inf token in output"
        echo "  words: $WORDS"
        return
    fi
    if [ "$DISTINCT" -lt 8 ]; then
        fail "$label — degenerate (only $DISTINCT distinct tokens in last 32)"
        echo "  tokens: $(echo "$TOKS" | tr '\n' ' ')"
        return
    fi
    # Generated text appears interleaved with logs on stdout — substring match works
    if ! echo "$OUT$WORDS" | grep -q "$expect"; then
        fail "$label — expected '$expect' in output"
        echo "  words: $WORDS"
        return
    fi
    pass "$label (distinct=$DISTINCT, contains '$expect')"
}

smoke_prompt "Qwen3-4B Q8_0 (dense)" \
    "Qwen3-4B-Instruct-2507-Q8_0.gguf" \
    "The capital of France is" \
    "Paris"

# ------------------------------------------------------------------- summary
echo
if [ "$FAIL" -eq 0 ]; then
    echo "${GRN}=== verify $MODE: OK ===${RST}"
    exit 0
else
    echo "${RED}=== verify $MODE: $FAIL failure(s) ===${RST}"
    exit 1
fi
