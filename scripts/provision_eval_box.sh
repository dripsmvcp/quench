#!/usr/bin/env bash
# One-time toolchain provision for the eval runner box (rented 5090).
#
# The box is an unprivileged container (no Docker), so the eval bot builds
# both arms directly on it — that needs the same toolchain floor the repo's
# canonical Docker image has: CUDA 13.3 nvcc, CMake >= 3.25, GCC 12+, Ninja,
# python3. Everything lands on the instance disk and survives stop/start;
# re-running is idempotent and cheap.
#
# Run it once per box:
#   scp -i ~/.ssh/rtx5090 -P <port> scripts/provision_eval_box.sh root@<host>:/root/
#   ssh -i ~/.ssh/rtx5090 -p <port> root@<host> 'bash /root/provision_eval_box.sh'
#
# The trailing version report is the acceptance check — read it. If the
# image's distro ships a too-old cmake, the pip fallback below covers it.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    g++-12 gcc-12 git ninja-build python3 python3-pip rsync ca-certificates

# CUDA 13.3 toolkit from the NVIDIA apt repo the CUDA base images ship
# pre-configured. Skipped if a new enough nvcc is already present.
if ! command -v nvcc >/dev/null || ! nvcc --version | grep -qE "release 13\.[3-9]"; then
    apt-get install -y --no-install-recommends cuda-toolkit-13-3
fi
export PATH="/usr/local/cuda/bin:$PATH"
grep -q "cuda/bin" /root/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /root/.bashrc

# CMake >= 3.25: distro packages are often too old; pip's wheel is current.
if ! command -v cmake >/dev/null || \
   ! python3 -c 'import subprocess,sys;v=subprocess.run(["cmake","--version"],capture_output=True,text=True).stdout.split()[2];sys.exit(0 if tuple(map(int,v.split(".")[:2]))>=(3,25) else 1)'; then
    pip3 install --break-system-packages cmake 2>/dev/null || pip3 install cmake
fi

update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60 || true
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 || true

echo "=== provision report ==="
nvcc --version | tail -1
cmake --version | head -1
g++ --version | head -1
ninja --version
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== ok ==="
