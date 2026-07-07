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
# Prereqs: `source /opt/intel/oneapi/setvars.sh`, the .venv with torch+xpu active,
# and `pip install ninja` (SyclExtension requires ninja).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$REPO/build-sycl-shared"

echo ">> Building shared ops lib (icpx -fsycl) in $BUILD"
cmake -S "$REPO" -B "$BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  -DQUIXICORE_XPU_ENABLE_SYCL=ON -DBUILD_SHARED_LIBS=ON \
  -DQUIXICORE_XPU_BUILD_TESTS=OFF -DQUIXICORE_XPU_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" -j"$(nproc)" --target quixicore_xpu_ops

echo ">> Building the tk_xpu extension"
cd "$HERE"
python setup.py build_ext --inplace

echo ">> Done. Run:  python test_parity.py"
