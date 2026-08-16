#!/usr/bin/env bash
#
# QuixiCore XPU bench entrypoint. Thin wrapper around the shared core
# (run_bench_core.sh, synced from the umbrella); this file is hand-written
# and backend-owned.
#
#   perf/harness/run_bench.sh --preset sycl --label rms-norm-ab --kernel rms_norm
#   perf/harness/run_bench.sh --dry-run
#
# Wraps perf/bench_kernels.py (build-health phases plus the on-device
# xpu_bench kernel matrix under the sycl preset). Set QUIXICORE_XPU_DEVICE to
# record the GPU name in run.json (e.g. "Intel Arc Pro B60").
# SCAFFOLDING NOTE: this entrypoint has not yet been executed on a B60 host;
# verify it there before treating its output as evidence.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QC_BACKEND="xpu"

qc_out_dir() {
    # bench_kernels.py owns the run directory: perf/results/<date>/<run-id>
    echo "$REPO_ROOT/perf/results/$2/$1"
}

qc_bench_cmd() {
    qc_exec python3 "$REPO_ROOT/perf/bench_kernels.py" \
        --phase all --preset "${QC_PRESET:-sycl}" \
        --run-id "$(basename "$OUT_DIR")" \
        ${QC_PASSTHROUGH[@]+"${QC_PASSTHROUGH[@]}"}
}

qc_device_info() {
    command -v sycl-ls >/dev/null 2>&1 && echo "sycl_ls=$(sycl-ls 2>/dev/null | head -1)"
    command -v icpx >/dev/null 2>&1 && echo "icpx=$(icpx --version 2>/dev/null | head -1)"
    echo "uname=$(uname -srm)"
}

source "$(dirname "${BASH_SOURCE[0]}")/run_bench_core.sh"
