# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Build quench from source
# =============================================================================
# Native CUDA 13.3 devel image: nvcc 13.3 (V13.3.73) is on PATH at
# /usr/local/cuda (-> /usr/local/cuda-13.3) out of the box — no apt toolkit
# install needed. The host driver (UMD 13.3) supports it. sm_120 gains the 13.3
# ptxas/PTX-ISA-9.3 codegen; no new tensor-core HW (still mma.sync, no
# tcgen05/wgmma).
# Ubuntu 26.04 LTS host userland → GCC 15.2 / libstdc++ 15. GCC 15 no longer
# pulls <algorithm>/<numeric> in transitively through other headers, so it
# catches the missing-include class of bugs at build time that the
# older GCC 13 toolchain silently accepted.
# Split into `toolchain` (compiler + pinned deps, no source) and `builder`
# (toolchain + this checkout). The split is what makes `make dev` possible:
# that target mounts the working tree into the toolchain image and runs an
# INCREMENTAL ninja build against a persistent build dir, so a one-file edit
# costs seconds instead of the full-image rebuild's minutes. The final image is
# byte-for-byte what it was — `builder` still starts from exactly these layers.
FROM nvidia/cuda:13.3.1-devel-ubuntu26.04 AS toolchain

ARG CMAKE_BUILD_TYPE=Release

RUN { sed -i 's|archive.ubuntu.com|de.archive.ubuntu.com|g; s|security.ubuntu.com|de.archive.ubuntu.com|g' \
          /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; } \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ git ninja-build ca-certificates python3 wget \
    && wget -qO /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v4.3.1/cmake-4.3.1-linux-x86_64.sh \
    && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake.sh \
    && rm -rf /var/lib/apt/lists/*

# Hint CMake's find_package(CUDAToolkit) at the toolkit (nvcc is already on PATH).
ENV CUDA_HOME=/usr/local/cuda

# Pre-clone third-party deps into their own layer. Only invalidated when the
# pinned tags below change — code-only edits keep this layer cached, saving
# the FetchContent git-clone step (~30-60s) on every Docker rebuild.
# Tags are authoritative in cmake/quench-deps.cmake; `make build` injects them as
# --build-arg from that file. The defaults below keep a bare `docker build .`
# working and MUST match cmake/quench-deps.cmake.
ARG QUENCH_DEP_GOOGLETEST_TAG=v1.17.0
ARG QUENCH_DEP_CUTLASS_TAG=v4.6.0
ARG QUENCH_DEP_HTTPLIB_TAG=v0.50.1
ARG QUENCH_DEP_NLOHMANN_JSON_TAG=v3.12.0
RUN git clone --depth=1 --branch ${QUENCH_DEP_GOOGLETEST_TAG} https://github.com/google/googletest.git /deps/googletest \
 && git clone --depth=1 --branch ${QUENCH_DEP_CUTLASS_TAG}    https://github.com/NVIDIA/cutlass.git    /deps/cutlass    \
 && git clone --depth=1 --branch ${QUENCH_DEP_HTTPLIB_TAG}    https://github.com/yhirose/cpp-httplib.git /deps/httplib  \
 && git clone --depth=1 --branch ${QUENCH_DEP_NLOHMANN_JSON_TAG} https://github.com/nlohmann/json.git   /deps/json

FROM toolchain AS builder

WORKDIR /src
COPY . .

# Historical no-op kept as a guard: cmake/CompilerFlags.cmake pins
# -march=x86-64-v3 directly now. If a `-march=native` ever comes back, this
# rewrites it so the shipped image stays portable — and, just as importantly,
# so `make dev` (which does NOT run this) keeps producing identical codegen to
# the image. Two build paths that disagree on -march would silently confound
# every A/B measured on one and compared against the other.
RUN sed -i 's/-march=native/-march=x86-64-v3/g' cmake/CompilerFlags.cmake

ARG QUENCH_BUILD_TESTS=OFF
ARG QUENCH_BUILD_BENCH=OFF
ARG QUENCH_EXTRA_CMAKE=

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DQUENCH_BUILD_TESTS=${QUENCH_BUILD_TESTS} \
        -DQUENCH_BUILD_BENCH=${QUENCH_BUILD_BENCH} \
        -DQUENCH_BUILD_TOOLS=ON \
        -DQUENCH_BUILD_SERVER=ON \
        ${QUENCH_EXTRA_CMAKE} \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/deps/googletest \
        -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/deps/cutlass \
        -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=/deps/httplib \
        -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/json \
    && cmake --build build -j$(nproc) \
    && cp build/quench-server build/quench-cli /tmp/ \
    && ([ -f build/quench-quantize ] && cp build/quench-quantize /tmp/ || true) \
    && if [ -f build/quench-tests ]; then \
           cp build/quench-tests build/quench-tests-unit /tmp/ \
           && for b in test-core test-text test-compute test-attention \
                       test-quant test-kv test-moe-gdn test-e2e; do \
                  [ -f "build/$b" ] && cp "build/$b" /tmp/; \
              done; \
       fi \
    && ([ -f build/quench-bench ] && cp build/quench-bench /tmp/ || true) \
    && ([ -f build/test-gdn ] && cp build/test-gdn /tmp/ || true)

# =============================================================================
# Stage 2: Minimal runtime image
# =============================================================================
# Native CUDA 13.3 runtime image already ships the matching cudart + cuBLAS
# (and transitive deps like libnvjitlink) at /usr/local/cuda; only the small
# entrypoint/healthcheck helpers need adding.
FROM nvidia/cuda:13.3.1-runtime-ubuntu26.04

# OCI image metadata — GHCR renders org.opencontainers.image.description on the
# package page (https://github.com/dripsmvcp/quench/pkgs/container/quench). Hardcoded here
# (not only via docker/metadata-action) so local builds carry it too.
LABEL org.opencontainers.image.title="quench" \
      org.opencontainers.image.description="LLM inference engine in C++/CUDA for NVIDIA Blackwell sm_120 (RTX 5090/5080/5070 Ti, RTX PRO 6000). Native NVFP4 + GGUF, OpenAI/Anthropic-compatible server." \
      org.opencontainers.image.source="https://github.com/dripsmvcp/quench" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries
COPY --from=builder /tmp/quench-server /usr/local/bin/quench-server
COPY --from=builder /tmp/quench-cli /usr/local/bin/quench-cli
COPY --from=builder /tmp/quench-quantiz[e] /usr/local/bin/
COPY --from=builder /tmp/quench-test[s] /usr/local/bin/
COPY --from=builder /tmp/quench-tests-uni[t] /usr/local/bin/
COPY --from=builder /tmp/test-cor[e] /usr/local/bin/
COPY --from=builder /tmp/test-tex[t] /usr/local/bin/
COPY --from=builder /tmp/test-comput[e] /usr/local/bin/
COPY --from=builder /tmp/test-attentio[n] /usr/local/bin/
COPY --from=builder /tmp/test-quan[t] /usr/local/bin/
COPY --from=builder /tmp/test-k[v] /usr/local/bin/
COPY --from=builder /tmp/test-moe-gd[n] /usr/local/bin/
COPY --from=builder /tmp/test-e2[e] /usr/local/bin/
COPY --from=builder /tmp/quench-benc[h] /usr/local/bin/
COPY --from=builder /tmp/test-gd[n] /usr/local/bin/

# Copy entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Non-root user with write access to /models and to the cache directory.
#
# The cache dir must EXIST in the image and be owned by quench, even though nothing
# is shipped in it: Docker only copies ownership into a fresh named volume from
# a directory that is already there. Mounting a volume over a path the image
# does not create yields a root-owned mount that `quench` cannot write, which
# silently disables both caches that live here — the library-reserve
# measurement (A1.5) and the warm weight cache. Measured: with the volume
# and without this line, "library reserve: could not write" and "Warm cache: not
# writable — skipping", i.e. mounting the volume was WORSE than not mounting it.
RUN useradd -m -s /bin/bash quench \
    && mkdir -p /models /home/quench/.cache/quench \
    && chown quench:quench /models \
    && chown -R quench:quench /home/quench/.cache

USER quench
WORKDIR /home/quench

EXPOSE 8080
VOLUME /models

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${QUENCH_PORT:-8080}/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["quench-server"]
