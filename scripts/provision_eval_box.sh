#!/usr/bin/env bash
# One-time toolchain provision for the eval runner box (rented 5090).
#
# The box is an unprivileged container (no Docker), so the eval bot builds
# both arms directly on it — that needs the same toolchain floor the repo's
# canonical Docker image has: CUDA >= 13.2 nvcc, CMake >= 3.25, Ninja,
# python3, and a host compiler nvcc will accept `-std=c++23` with.
#
# Run it once per box:
#   scp -i ~/.ssh/rtx5090 -P <port> scripts/provision_eval_box.sh root@<host>:/root/
#   ssh -i ~/.ssh/rtx5090 -p <port> root@<host> 'bash /root/provision_eval_box.sh'
#
# History worth keeping: this script used to install g++-12 and report
# versions as its acceptance check. That is not enough. CMAKE_CXX_STANDARD is
# 23, and nvcc silently DOWNGRADES rather than failing —
#
#   nvcc warning : The -std=c++23 flag is not supported with the configured
#   host compiler. Flag will be ignored.
#
# — so provisioning "succeeded", and the first real eval died 20 minutes in on
# src/compute/sampling_topk_topp.cu. The version report could not catch it
# because every version printed looked fine. The acceptance check below now
# configures and builds a miniature CUDA + C++23 CMake project using the SAME
# flags the bot uses, so a box that cannot build quench fails HERE, in
# seconds, instead of inside a billed eval.
#
# Model staging is opt-in: re-run with QUENCH_STAGE_MODEL=1 to fetch the
# benchmark GGUF. Left off by default because it is ~13 GB and the file
# usually already exists on the instance disk.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
MODEL_PATH="${QUENCH_EVAL_MODEL:-/root/models/Mistral-Nemo-Instruct-2407-Q8_0.gguf}"
MODEL_URL="${QUENCH_EVAL_MODEL_URL:-https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q8_0.gguf}"

apt-get update
apt-get install -y --no-install-recommends \
    git ninja-build python3 python3-pip rsync ca-certificates \
    software-properties-common curl

# Newer GCC than the distro ships. jammy tops out at 12, which nvcc 13.x will
# not accept -std=c++23 with; the toolchain PPA carries 13 and 14. Failure to
# add the PPA is not fatal — the probe below decides what is actually usable.
add-apt-repository -y ppa:ubuntu-toolchain-r/test || true
apt-get update || true
for v in 14 13 12; do
    apt-get install -y --no-install-recommends "g++-$v" "gcc-$v" || true
done

# CUDA 13.3 toolkit from the NVIDIA apt repo the CUDA base images ship
# pre-configured. Skipped if a new enough nvcc is already present.
if ! command -v nvcc >/dev/null || ! nvcc --version | grep -qE "release 1[3-9]\.[2-9]"; then
    apt-get install -y --no-install-recommends cuda-toolkit-13-3
fi
export PATH="/usr/local/cuda/bin:$PATH"
grep -q "cuda/bin" /root/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /root/.bashrc

# CMake >= 3.25: distro packages are often too old; pip's wheel is current.
if ! command -v cmake >/dev/null || \
   ! python3 -c 'import subprocess,sys;v=subprocess.run(["cmake","--version"],capture_output=True,text=True).stdout.split()[2];sys.exit(0 if tuple(map(int,v.split(".")[:2]))>=(3,25) else 1)'; then
    pip3 install --break-system-packages cmake 2>/dev/null || pip3 install cmake
fi

# ---------------------------------------------------------------------------
# Pick a host compiler nvcc will genuinely accept -std=c++23 with. Probed, not
# assumed: nvcc's supported-host-compiler matrix moves per CUDA release, and
# the failure mode is a warning plus a downgrade, not an error.
probe=/tmp/quench_probe
mkdir -p $probe
cat > $probe/t.cu <<'EOF'
#include <version>
#include <string_view>
// if -std=c++23 were dropped, __cplusplus stays at the older value and this
// fails at compile time rather than silently building the wrong dialect
static_assert(__cplusplus >= 202302L, "host compiler did not accept c++23");
__global__ void k() {}
int main() { k<<<1,1>>>(); return 0; }
EOF

HOSTCC=""
for v in 14 13 12; do
    command -v "g++-$v" >/dev/null || continue
    if nvcc -std=c++23 -ccbin "g++-$v" -arch=sm_120a -c $probe/t.cu -o $probe/t.o 2>$probe/err-$v.log; then
        HOSTCC="$v"; break
    fi
done
if [ -z "$HOSTCC" ]; then
    echo "=== FAILED: no installed g++ accepts -std=c++23 under this nvcc"
    for f in $probe/err-*.log; do echo "--- $f"; tail -5 "$f"; done
    exit 1
fi
echo "=== host compiler: g++-$HOSTCC accepted by nvcc for -std=c++23"

# CMake resolves the host compiler through cc/c++, NOT gcc/g++ — repointing
# only the latter (as this script used to) leaves the build on the distro
# default. All four link groups move together.
for pair in "gcc:gcc-$HOSTCC" "g++:g++-$HOSTCC" "cc:gcc-$HOSTCC" "c++:g++-$HOSTCC"; do
    name="${pair%%:*}"; target="/usr/bin/${pair##*:}"
    update-alternatives --install "/usr/bin/$name" "$name" "$target" 90
    update-alternatives --set "$name" "$target"
done

# ---------------------------------------------------------------------------
# Acceptance check: the bot's exact configure+build path, in miniature. This
# is the gate — a green version report means nothing on its own.
accept=/tmp/quench_accept
rm -rf $accept && mkdir -p $accept
cp $probe/t.cu $accept/main.cu
cat > $accept/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.25)
project(quench_accept LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120a)
add_executable(accept main.cu)
EOF
if ! ( cd $accept && cmake -S . -B b -G Ninja -DCMAKE_BUILD_TYPE=Release >cfg.log 2>&1 \
        && cmake --build b >bld.log 2>&1 ); then
    echo "=== FAILED: the box cannot build a CUDA + C++23 target"
    tail -25 $accept/cfg.log $accept/bld.log 2>/dev/null
    exit 1
fi
echo "=== acceptance: CUDA + C++23 CMake target configured and built"

# ---------------------------------------------------------------------------
# Benchmark model. Reported always, fetched only on request — it is ~13 GB and
# lives on the instance disk across stop/start, so it is normally already here.
mkdir -p "$(dirname "$MODEL_PATH")"
if [ -f "$MODEL_PATH" ]; then
    echo "=== model present: $(du -h "$MODEL_PATH" | cut -f1)  $MODEL_PATH"
elif [ "${QUENCH_STAGE_MODEL:-0}" = "1" ]; then
    echo "=== staging model (~13 GB) -> $MODEL_PATH"
    curl -fL --retry 3 -o "$MODEL_PATH.part" "$MODEL_URL"
    mv "$MODEL_PATH.part" "$MODEL_PATH"
    echo "=== model staged: $(du -h "$MODEL_PATH" | cut -f1)"
else
    echo "=== MODEL MISSING: $MODEL_PATH"
    echo "    other GGUFs on this box:"
    find /root -maxdepth 3 -name '*.gguf' -printf '      %s  %p\n' 2>/dev/null | head
    echo "    re-run with QUENCH_STAGE_MODEL=1 to fetch it"
fi

echo "=== provision report ==="
nvcc --version | grep -E "release" || nvcc --version | tail -1
cmake --version | head -1
c++ --version | head -1          # what CMake will actually use
ninja --version
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /root | tail -1
echo "=== ok ==="
