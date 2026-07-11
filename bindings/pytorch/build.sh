#!/usr/bin/env bash
# Build the QuixiCore XPU PyTorch extension (tk_xpu) and its shared ops lib.
#
# Two non-obvious requirements (both learned the hard way — see README.md):
#   1. The ops must be a *shared* library built with `icpx -fsycl`. A static .a
#      linked into the extension does NOT get its SYCL device-image registration
#      wired into the runtime ProgramManager -> kernel submit segfaults.
#   2. The binding source must end in `.sycl` so torch's SyclExtension compiles it
#      with -fsycl and performs the SYCL device link at the end.
#
# Prereqs: `source /opt/intel/oneapi/setvars.sh` and ninja installed in the
# repository `.venv` (SyclExtension requires ninja).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$REPO/build-sycl-shared"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create $REPO/.venv" >&2
  exit 1
fi
export PATH="$(dirname "$PYTHON"):$PATH"

echo ">> Building shared ops lib (icpx -fsycl) in $BUILD"
cmake -S "$REPO" -B "$BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  -DQUIXICORE_XPU_ENABLE_SYCL=ON -DBUILD_SHARED_LIBS=ON \
  -DQUIXICORE_XPU_BUILD_TESTS=OFF -DQUIXICORE_XPU_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" -j"$(nproc)" --target quixicore_xpu_ops

echo ">> Building the tk_xpu extension"
cd "$HERE"
"$PYTHON" setup.py build_ext --inplace

echo ">> Done. Run:  $PYTHON $HERE/test_parity.py"
