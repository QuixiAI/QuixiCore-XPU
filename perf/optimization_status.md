# QuixiCore XPU Optimization Status

This is the running notebook for XPU kernel implementation and optimization.
Raw output belongs under `perf/results/`; stable conclusions belong here.

## Entry Template

Use this structure for every kernel family or optimization pass:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Record enough context to reproduce the run: Intel GPU target, oneAPI compiler,
SYCL backend/runtime, Level Zero driver when available, command, git commit or
working-tree label, dtype, shape, quant format, warmups, iterations, median,
variance, correctness tolerance, and observed error.

## 2026-07-05: Initial Scaffold

Status: landed scaffold, no kernel families claimed complete.

Added:

- Backend metadata in `.quixicore/backend.yaml`.
- CMake build with `dev` and `sycl` presets.
- Public backend status API.
- Smoke test and backend-info example.
- SYCL device probe gated by `QUIXICORE_XPU_ENABLE_SYCL`.
- Performance handbook and CMake-based baseline harness.

References available under `.reference/`:

- `intel-xpu-backend-for-triton`
- `xetla`
- `oneDNN`
- `tiny-dpcpp-nn`

Verification:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Local result: passed on macOS with AppleClang. The `sycl` preset was not run
locally because `icpx` was not on `PATH`.

## First Kernel Plan

Status: not started.

Priority order:

1. RMSNorm / LayerNorm.
2. Softmax.
3. GELU / GLU.
4. Quant format decode helpers.
5. Quant GEMV.

Open questions:

- Which Intel GPU target should be the first performance baseline: Arc,
  Data Center GPU, or both?
- Should the first public API target be a C++ library API only, or should a
  Python extension be introduced before the first kernels?
- Which framework baseline should be primary for XPU comparisons: oneDNN,
  PyTorch XPU, IPEX, Triton XPU, or a mix?
