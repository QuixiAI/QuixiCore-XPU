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

## 2026-07-06: GELU reference kernel (activations/gelu) — SYCL + oneDNN

Status: landed (both variants shipped and correctness-checked; one SYCL
optimization pass kept).

Environment: Intel Arc Pro B60 Graphics (Battlemage G21, 160 XVEs, subgroup
sizes 16/32), 4x in host but single-device runs. oneAPI DPC++/C++ 2026.0.0,
Unified Runtime over Level-Zero V2, driver 1.14.37020. Compiler `icpx`, JIT
(no AoT). Working-tree label: gelu-reference-kernel.

Current implementation: `kernels/activations/gelu/variants/xpu_sycl/gelu.sycl.cpp`
(native) and `kernels/activations/gelu/variants/xpu_onednn/gelu.onednn.cpp`
(oneDNN eltwise via SYCL interop). Dispatch: `src/dispatch/activations.cpp`.

Current public route: `quixicore::xpu::ops::gelu(queue, in, out, n, dt, approx,
variant, blocking)`. Both `Variant::sycl` and `Variant::vendor` are shipped and
selectable; `Variant::best` currently resolves to `sycl` (no per-shape perf
model yet — see decision).

References inspected: oneDNN eltwise_gelu_erf/tanh as the vendor variant and the
baseline to beat.

Correctness: on-device C++ smoke (`tests/xpu_ops_smoke.cpp`) + pytest
(`tests/correctness/activations/gelu`) vs an fp64 host reference. f32 max_abs
4.4e-7; combined tolerance abs_err <= atol + rtol*|ref| (atol 1e-5, rtol 1e-4),
which is the correct check for a zero-crossing activation (a pure relative
metric spuriously fails near GELU's root). Both variants pass.

Baseline (n=4,194,304, iters=100, warmup=20, median device time from SYCL
profiling events; GB/s = 2*n*bytes/median):

- SYCL naive (one element per work-item): f32 271, bf16 178, f16 178 GB/s.
- oneDNN vendor: f32 411 GB/s (~90% of the ~456 GB/s B60 roofline).

Experiments:

1. erf vs tanh on the SYCL f32 path: 271 vs 273 GB/s — identical. REJECTS the
   "ALU-bound on the transcendental" hypothesis; the naive path is bound by
   memory access granularity, not the special function.
2. VEC=4, each work-item processes a contiguous strip (base = tid*4): f32 271 ->
   196 GB/s. REJECTED — a contiguous strip per work-item breaks subgroup
   coalescing (adjacent work-items no longer touch adjacent addresses).
3. VEC=4 coalesced grid stride (i = tid + k*threads): each subgroup issues
   contiguous loads/stores while amortizing index/launch overhead. KEPT:
   f32 271 -> 329 (+21%), bf16 178 -> 229 (+29%), f16 178 -> 227 (+27%).

Decision: keep experiment 3 as the shipped SYCL variant (well past the >=3%
low-risk threshold, correctness preserved). For f32 GELU the oneDNN vendor
variant is still measured ~1.25x faster than the tuned SYCL path (411 vs 329);
callers wanting peak f32 throughput should select `Variant::vendor`. This is the
concrete justification for shipping both co-equal variants. Wiring
`Variant::best` to pick vendor-for-f32 / sycl-elsewhere from recorded data is
deferred (needs a small per-op/dtype perf table).

Open questions: kVec sweep (8/16) and an explicit work-group size / nd_range
were not yet tried and may narrow the remaining gap to oneDNN on f32; a
`sycl::vec`-typed load path may help the 16-bit dtypes further.

Raw results: `perf/results/<date>/<run-id>/` via
`python3 perf/bench_kernels.py --phase kernels --preset sycl`.

## First Kernel Plan

Status: in progress — GELU landed (both variants). Next: RMSNorm/LayerNorm,
Softmax, then GLU modes + GELU backward.

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
