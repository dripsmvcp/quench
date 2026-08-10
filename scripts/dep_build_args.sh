#!/usr/bin/env bash
# Emit --build-arg lines for the dependency pins defined once in
# cmake/quench-deps.cmake. Used by `make build` so the tags are not duplicated
# (inlining the sed in the Makefile breaks make's $(shell ...) paren matching).
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sed -n 's/^set(QUENCH_DEP_\([A-Z_]*\)_TAG[ \t]*\([^ \t]*\).*/--build-arg QUENCH_DEP_\1_TAG=\2/p' \
    "$dir/cmake/quench-deps.cmake"
