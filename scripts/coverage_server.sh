#!/usr/bin/env bash
# Measure real gcov LINE coverage of tools/quench-server/ over an end-to-end run.
#
# The CI suite only unit-tests the server's pure helpers (anthropic/SSE/stream
# utils) and runs a Python *mock* contract — it never executes the real
# handlers.cpp/main.cpp/batching_engine.cpp. This harness builds quench-server with
# gcov instrumentation (only the server TUs; the CUDA quench lib is left alone),
# drives every endpoint + the manual server batteries against a real model on
# the GPU, and reports measured line coverage.
#
# Usage:   scripts/coverage_server.sh
# Env:     QUENCH_COV_MODEL    (default Qwen3-8B-Q8_0.gguf) — must support tools
#          QUENCH_MODELS_DIR   (default $HOME/models)      — host dir mounted at /models
#          QUENCH_COV_PORT     (default 8080)
#          QUENCH_COV_KEEP     (set to 1 to keep the quench:cov image + container)
# Output:  prints the gcovr table; writes build/coverage/ (txt + html) if -DQUENCH_COVERAGE produced data.
set -euo pipefail

MODEL="${QUENCH_COV_MODEL:-Qwen3-8B-Q8_0.gguf}"
MODELS_DIR="${QUENCH_MODELS_DIR:-$HOME/models}"
PORT="${QUENCH_COV_PORT:-8080}"
IMG=quench:cov
CTR=quench_cov
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cleanup() { [ "${QUENCH_COV_KEEP:-0}" = "1" ] || docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== 1/5 build instrumented quench-server (gcov, server TUs only) =="
docker build --target builder --build-arg QUENCH_BUILD_TESTS=OFF \
    --build-arg QUENCH_EXTRA_CMAKE="-DQUENCH_COVERAGE=ON" -t "$IMG" .

echo "== 2/5 launch container + server =="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT":"$PORT" -w /src "$IMG" sleep infinity >/dev/null
docker exec "$CTR" bash -c "apt-get update -qq && apt-get install -y -qq gcovr >/dev/null 2>&1"
# Rich flag set so args.cpp / main.cpp pre-routing (rate-limit/concurrency) are exercised too.
docker exec -d "$CTR" bash -c "cd /src && ./build/quench-server --model /models/$MODEL \
    --host 0.0.0.0 --port $PORT --max-concurrent 8 --rate-limit 100000 --max-input-tokens 100000 \
    > /tmp/srv.log 2>&1"
for i in $(seq 1 90); do
    curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true' && break
    sleep 2
done

echo "== 3/5 exercise endpoints + manual batteries =="
QUENCH_BASE="http://localhost:$PORT" QUENCH_MODEL="$MODEL" python3 tests/exercise_all_endpoints.py || true
QUENCH_HOST=localhost QUENCH_PORT="$PORT" QUENCH_MODEL="$MODEL" python3 tests/test_server_robustness.py || true
QUENCH_MODEL="$MODEL" N=4 LOAD=40 FAIL_THRESHOLD=0.5 python3 tests/test_server_0token_battery.py || true

echo "== 4/5 stop server (flush gcov) =="
docker exec "$CTR" bash -c "kill -TERM \$(pgrep quench-server); for i in \$(seq 1 30); do pgrep quench-server >/dev/null || break; sleep 1; done"

echo "== 5/5 gcovr report (tools/quench-server) =="
docker exec "$CTR" bash -c "cd /src && mkdir -p build/coverage && \
    gcovr --root . --filter 'tools/quench-server/' --txt --html-details build/coverage/index.html 2>/dev/null"
docker cp "$CTR":/src/build/coverage "$ROOT/build/coverage" 2>/dev/null || true
echo "HTML report (if produced): build/coverage/index.html"
