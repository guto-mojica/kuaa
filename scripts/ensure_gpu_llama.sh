#!/usr/bin/env bash
# Ensure the project venv has a GPU-capable llama-cpp-python build.
#
# Idempotent: instant no-op if already GPU-capable; otherwise rebuilds from
# source with CUDA per docs/GPU_LLAMA_CPP_CUDA_BUILD.md. Run this before GPU
# describer jobs, or any time after `uv sync`/`uv run` (which silently
# replace the CUDA build with the cached CPU wheel).
#
# Exit codes: 0 = GPU-capable (or no GPU, nothing to do); 1 = manual step
# required (root patch / missing toolchain) or build failed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

PY=".venv/bin/python"
CUDA_HOME="/usr/local/cuda-13.0"
CUDA_HEADER="$CUDA_HOME/targets/x86_64-linux/include/crt/math_functions.h"
DOC="docs/GPU_LLAMA_CPP_CUDA_BUILD.md"

say() { printf '  %s\n' "$*"; }

[ -x "$PY" ] || { echo "✗ $PY not found — run 'uv sync --extra full --group dev' first."; exit 1; }

gpu_capable() {
  "$PY" - <<'EOF' 2>/dev/null
import sys
try:
    import llama_cpp.llama_cpp as L
    sys.exit(0 if L.llama_supports_gpu_offload() else 1)
except Exception:
    sys.exit(1)
EOF
}

if gpu_capable; then
  echo "✓ llama-cpp-python is already GPU-capable. Nothing to do."
  exit 0
fi

echo "• llama-cpp-python is CPU-only. Checking whether a CUDA build is possible…"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "✓ No NVIDIA GPU detected — a CPU-only build is correct here. Nothing to do."
  exit 0
fi

# --- Preconditions for the from-source CUDA build -------------------------
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "✗ $CUDA_HOME/bin/nvcc missing. Install: sudo dnf install cuda-toolkit-13-0"
  echo "  Then re-run this script. See $DOC §2."
  exit 1
fi

if [ ! -x /usr/bin/g++-15 ]; then
  echo "✗ g++-15 missing (CUDA 13 needs host gcc <=15; this distro ships gcc 16)."
  echo "  Run: sudo dnf install -y gcc15 gcc15-c++   (parallel-installable; see $DOC §2)"
  exit 1
fi

if ! grep -q 'rsqrt(double x) noexcept' "$CUDA_HEADER" 2>/dev/null; then
  echo "✗ CUDA header not patched (glibc 2.43 ↔ CUDA 13.0.2 rsqrt clash)."
  echo "  This needs sudo — run exactly this, then re-run the script:"
  echo
  echo "    F=$CUDA_HEADER"
  echo '    sudo test -e "$F.bak-kuaa" || sudo cp "$F" "$F.bak-kuaa"'
  echo "    sudo sed -i -e 's/rsqrt(double x);\$/rsqrt(double x) noexcept (true);/' \\"
  echo "                -e 's/rsqrtf(float x);\$/rsqrtf(float x) noexcept (true);/' \"\$F\""
  echo
  echo "  Full rationale: $DOC §2–§3."
  exit 1
fi

# --- Build from source with CUDA -----------------------------------------
echo "• Building llama-cpp-python from source with CUDA (sm_120, ~10-15 min)…"
CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 \
CUDACXX="$CUDA_HOME/bin/nvcc" \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15" \
FORCE_CMAKE=1 \
PATH="$CUDA_HOME/bin:$PATH" \
UV_LINK_MODE=copy \
uv pip install --no-cache --reinstall --no-binary llama-cpp-python "llama-cpp-python>=0.3,<0.4"

if gpu_capable; then
  echo "✓ Done — llama-cpp-python is now GPU-capable (llama_supports_gpu_offload=True)."
  exit 0
fi

echo "✗ Build completed but gpu_offload is still False. Investigate: see $DOC §4/§7."
exit 1
