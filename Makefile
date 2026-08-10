# quench test runner targets
# Usage: make test-unit, make test-gpu, make test-all

# Fail loudly: a failure anywhere in a piped recipe (e.g. `cmd | tee`) must
# propagate, never get masked by the exit code of the last pipe stage.
SHELL := bash
.SHELLFLAGS := -o pipefail -c

DOCKER_IMG ?= quench:test
DOCKER_RUN = docker run --rm --gpus all -v $(PWD)/models:/models $(DOCKER_IMG)
BUILD_ARGS = --build-arg QUENCH_BUILD_TESTS=ON
# Dependency pins live once in cmake/quench-deps.cmake; inject them into the Docker
# build so the tags are not duplicated (bump that file only). Extraction is in a
# script — inlining the sed breaks make's $(shell ...) paren matching.
DEP_ARGS = $(shell scripts/dep_build_args.sh)

.PHONY: build test-unit test-gpu test-fast test-all test-e2e test-server test-golden check-gpu verify verify-fast install-hooks format format-check tidy sanitize asan coverage

# Check that no other process is using the GPU (games, other inference, etc.)
check-gpu:
	@GPU_PROCS=$$(nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader 2>/dev/null | grep -v "^$$"); \
	if [ -n "$$GPU_PROCS" ]; then \
		echo "ERROR: GPU is in use — benchmarks will be unreliable:"; \
		echo "$$GPU_PROCS"; \
		echo "Close other GPU processes first (games, other inference, etc.)"; \
		exit 1; \
	fi; \
	GPU_UTIL=$$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); \
	if [ "$$GPU_UTIL" -gt 5 ] 2>/dev/null; then \
		echo "WARNING: GPU utilization at $${GPU_UTIL}% — results may be noisy"; \
	fi; \
	echo "GPU is free (utilization: $${GPU_UTIL:-0}%)"

build:
	docker build $(BUILD_ARGS) $(DEP_ARGS) -t $(DOCKER_IMG) .

# ---------------------------------------------------------------------------
# Fast inner loop (`make dev`) — incremental compile, seconds not minutes.
#
# `make build` copies the tree into an image and compiles from scratch every
# time: correct, reproducible, and ~3.5 min even for a one-line edit. That is
# the right gate before a PR and the wrong tool for iterating.
#
# `make dev` mounts the working tree into the toolchain image and runs ninja
# against a PERSISTENT build dir, so only what changed recompiles. Codegen is
# identical to the image build (both -march=x86-64-v3, same toolchain layers),
# so a dev binary is a valid thing to run tests against.
#
# NOT a replacement for `make build`:
#   - `make verify-fast` / CI build the image. Green here is not green there.
# Use it to compile, run unit tests and iterate; then `make build` once.
#
# build-dev/ is root-owned (container writes it) — remove via the dev-clean
# target, never `sudo` on the host.
DEV_IMG ?= quench:toolchain
DEV_DIR ?= build-dev
DEV_RUN = docker run --rm -v $(PWD):/src -w /src $(DEV_IMG)
DEV_CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Release -DQUENCH_BUILD_TESTS=ON -DQUENCH_BUILD_TOOLS=ON \
                 -DQUENCH_BUILD_SERVER=ON \
                 -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/deps/googletest \
                 -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/deps/cutlass \
                 -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=/deps/httplib \
                 -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/json

.PHONY: dev dev-image dev-test dev-clean

# Toolchain-only image (compiler + pinned deps, no source). Always re-runs
# rather than guarding on `docker image inspect`: fully cached this costs ~1 s,
# and the guard would silently keep a stale toolchain after a dependency-pin
# bump — the exact class of "green build, wrong inputs" this repo keeps paying
# for elsewhere.
dev-image:
	@docker build $(DEP_ARGS) --target toolchain -t $(DEV_IMG) . >/dev/null

# Incremental build. `cmake -B` on an existing dir is a fast reconfigure, so
# this is safe to run every time.
dev: dev-image
	$(DEV_RUN) bash -c 'cmake -B $(DEV_DIR) -G Ninja $(DEV_CMAKE_ARGS) >/dev/null \
	  && cmake --build $(DEV_DIR) -j$$(nproc)'

# CPU unit lane against the dev build. Mirrors what CI's `ctest -L unit` runs,
# so a failure here is a real failure there — but the reverse does not hold
# (CI builds the image from a clean tree).
dev-test: dev
	$(DEV_RUN) ctest --test-dir $(DEV_DIR) -L unit --output-on-failure

dev-clean:
	docker run --rm -v $(PWD):/src -w /src $(DEV_IMG) rm -rf $(DEV_DIR)

# Unit tests: CPU-only, no GPU, no model, < 5s
# Mirrors `ctest -L unit`. Filter is sourced from CMakeLists.txt (_unit_e2e_filter).
test-unit: build
	$(DOCKER_RUN) quench-tests-unit

# GPU tests: everything including CUDA kernels. ~4-5 min without models —
# 7 of 8 binaries finish in <11s, but test-attention alone is ~241s (the
# paged-/crosspath-oracle sweeps).
test-gpu: build
	$(DOCKER_RUN) quench-tests

# Stage 3 — the SERVER stage (local, GPU-only). Boots a real quench-server against
# a live model and GATES on the OpenAI+Anthropic wire batteries (endpoints,
# robustness, logprobs, /v1/messages stream, embed/chat interleave,
# 0-token). CI has no GPU runner, so this is the only place handlers.cpp /
# batching_engine run end-to-end. See the script header for env knobs.
test-server: build
	bash scripts/test_server.sh

# Measured gcov line coverage of tools/quench-server/ over an end-to-end GPU run
# (builds an instrumented quench-server, drives every endpoint + the manual server
# batteries, reports coverage). Needs a GPU + a local model. See the script header.
coverage:
	bash scripts/coverage_server.sh

# Fast: unit tests only (no Docker GPU needed if built already)
test-fast: test-unit

# All tests (full GPU suite)
test-all: build
	$(DOCKER_RUN) quench-tests

# E2E model tests: load a real model, generate, verify output
test-e2e: build
	docker run --rm --gpus all -v $(PWD)/models:/models \
		-e QUENCH_TEST_MODEL=/models/Qwen3-4B-Instruct-2507-Q8_0.gguf \
		$(DOCKER_IMG) quench-tests --gtest_filter="PrimaryModelTest.*:EndToEndModelTest.*"

# Golden output comparison (greedy, temp=0)
test-golden: build
	@echo "Golden output tests require running server — use pytest tests/api/ instead"

# verify: full pre-merge gate (host build, ~5 min). build + ctest + graphs gate + smoke.
verify:
	@scripts/verify.sh full

# verify-fast: pre-push gate (host build, ~90s). build + filtered tests + graphs gate + smoke.
verify-fast:
	@scripts/verify.sh fast

# Install the local git hooks. Two-stage test gate:
#   Stage 1 — pre-commit (GPU): runs the full GPU suite (make test-gpu) when
#             staged sources change. CI has no GPU runner, so GPU correctness is
#             gated here, locally, before the commit lands.
#   pre-push: keeps the verify-fast regression gate.
#   Stage 2 — CI (CPU): ctest -L unit, in .github/workflows/ci.yml (no hook).
install-hooks:
	@cp scripts/pre-commit.hook .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@cp scripts/pre-push.hook .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "hooks installed:"
	@echo "  pre-commit → Stage 1 'make test-gpu' (full GPU suite) on staged src/tests changes"
	@echo "  pre-push   → 'make verify-fast' (graphs + smoke regression) on source changes"
	@echo "  CI (Stage 2) runs 'ctest -L unit' — the CPU lane — automatically"

# clang-format settings live in .clang-format. Host has no clang-format
# installed (clean-host policy), so we run it in a throwaway container.
CLANG_FORMAT_IMG ?= silkeh/clang:18
CLANG_FORMAT_RUN = docker run --rm -v $(PWD):/work -w /work $(CLANG_FORMAT_IMG) clang-format
CLANG_FORMAT_FILES = $$(find src include tools tests -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh')

# compute-sanitizer (memcheck) over the GPU-numeric test binaries
#. Runs inside the BUILDER stage (the runtime image has
# no CUDA toolkit, hence no compute-sanitizer; the builder keeps build/).
#
# DOES NOT WORK ON WSL2: the WDDM driver model exposes no debugger interface,
# compute-sanitizer reports "Error: Failed to initialize". Run this target on a native-Linux GPU
# host (e.g. a future self-hosted CI runner). Listed here so the invocation
# is documented and ready, not because it runs on the dev box.
# Host-code ASan+UBSan over the CPU test binaries (test-core, test-text).
# Works on WSL2 (host-side sanitizers only — nvcc-compiled device code is NOT
# sanitized, see QUENCH_SANITIZERS in CMakeLists.txt). Suppressions live in
# tools/sanitizers/: vendored-stb unaligned stores (UBSan,) and NVIDIA
# driver one-time allocations (LSan). Build tree persists in a named docker
# volume so re-runs are incremental.
asan:
	docker build --target builder $(BUILD_ARGS) $(DEP_ARGS) -t quench:builder .
	docker run --rm --gpus all -v $(PWD):/src -v quench-asan-build:/basan -w /src quench:builder bash -c '\
	  cmake -B /basan -S /src -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DQUENCH_SANITIZERS=ON \
	        -DQUENCH_BUILD_TOOLS=OFF -DQUENCH_BUILD_BENCH=OFF -DQUENCH_BUILD_SERVER=OFF > /basan/configure.log && \
	  cmake --build /basan --target test-core test-text -j$$(nproc) && \
	  for b in test-core test-text; do \
	    echo "== ASan+UBSan: $$b =="; \
	    UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:suppressions=/src/tools/sanitizers/ubsan.supp \
	    ASAN_OPTIONS=detect_leaks=1 \
	    LSAN_OPTIONS=suppressions=/src/tools/sanitizers/lsan.supp \
	    /basan/$$b || exit 1; \
	  done'

sanitize:
	docker build --target builder $(BUILD_ARGS) -t quench:sanitize .
	@for b in test-attention test-quant test-kv; do \
		echo "== compute-sanitizer memcheck: $$b =="; \
		docker run --rm --gpus all -v $(PWD)/models:/models quench:sanitize \
			/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
			/src/build/$$b || exit 1; \
	done

# Apply clang-format in place across src/, include/, tools/, tests/.
format:
	@$(CLANG_FORMAT_RUN) -i --style=file $(CLANG_FORMAT_FILES)
	@echo "clang-format applied"

# Check formatting without modifying files. Exits non-zero on violation.
format-check:
	@$(CLANG_FORMAT_RUN) --dry-run -Werror --style=file $(CLANG_FORMAT_FILES)

# clang-tidy over host C++ TUs (advisory — findings surface, do not fail). Runs in
# the CUDA builder image so the CUDA headers our .cpp files include are present;
# clang-tidy is apt-installed on the fly. .cu files are out of scope (need full
# nvcc flags). Configures first so build/compile_commands.json exists.
CLANG_TIDY_FILES = $$(find src tools -name '*.cpp')
tidy:
	@docker run --rm -v $(PWD):/work -w /work quench:builder bash -c '\
	  apt-get update -qq && apt-get install -y -qq clang-tidy >/dev/null 2>&1; \
	  test -f build/compile_commands.json || cmake --preset ci \
	      -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/deps/googletest \
	      -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/deps/cutlass \
	      -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=/deps/httplib \
	      -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/json >/dev/null; \
	  clang-tidy -p build --warnings-as-errors= $(CLANG_TIDY_FILES) || true'
