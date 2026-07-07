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

## 2026-07-06: norms (rms_norm, layernorm) — SYCL + oneDNN, three truisms smashed

Status: landed (rms_norm SYCL; layernorm SYCL + oneDNN vendor, best-routed).

Environment: same as the GELU entry (Arc Pro B60, oneAPI 2026.0, Level-Zero V2,
oneDNN 3.11).

Implementation: one work-group per row, fp32-accumulated `reduce_over_group`
reductions. `kernels/norms/rms_norm/variants/xpu_sycl`, `kernels/norms/layernorm/
variants/{xpu_sycl,xpu_onednn}`. Dispatch `src/dispatch/norms.cpp`.

Correctness: on-device gate, rms_norm + layernorm x {f32,f16,bf16} x
{sycl,vendor}, rows=64 dim=1024, vs an fp64 host reference rounded to storage
dtype. All 12 pass (f32 max_abs ~1e-7, bf16 ~1e-8..0).

This entry exists to record THREE assumptions that were tested and reversed on
the actual silicon rather than inherited (measure, don't assume):

1. TRUISM "oneDNN layer_normalization requires f32 scale/shift." FALSE on this
   stack. The basic `primitive_desc` overloads merely DEFAULT scale_shift_data_type
   to f32; oneDNN >= 3.x exposes an explicit `scale_shift_data_type` overload.
   Passing the input dtype, oneDNN's GPU layernorm accepted bf16 AND f16
   scale/shift with correct results and no fallback (verified with the
   QUIXICORE_XPU_TRACE_FALLBACK trace: the fallback path never fired). Removed
   the f32-only restriction.

2. TRUISM "the vendor library is the fast path to beat." FALSE for f32 layernorm.
   Native SYCL 387 GB/s vs oneDNN 244 GB/s at 8192x4096 — the hand-written
   kernel wins by 1.58x.

3. TRUISM "then the native kernel is the fast path." Also FALSE — for bf16
   layernorm, oneDNN 333 GB/s beats native SYCL 196 GB/s (1.70x). There is no
   universal winner; the answer is per (op, dtype).

Baseline (median device time, profiling events; 8192x4096; GB/s = (2*rows*dim +
affine*dim)*bytes / median):

- rms_norm SYCL f32: dim1024 312, dim4096 368, dim8192 386 GB/s.
- layernorm f32: SYCL 387 / oneDNN 244 GB/s.
- layernorm bf16: SYCL 196 / oneDNN 333 GB/s.

Decision: ship both variants; wire `Variant::best` for layernorm to route f32 ->
sycl, 16-bit -> vendor (encoded in src/dispatch/norms.cpp from this data). rms_norm
is SYCL-only (no oneDNN RMSNorm primitive); its `vendor`/`best` resolve to sycl.

Open questions / next experiments: bf16 native norms (196 GB/s) trail f32 (387)
— same scalar-16-bit-access bottleneck the GELU vectorization pass exposed;
apply a vectorized/sub-group load path to the bf16 norm reduction. A single
sub-group per row (no work-group barrier) may beat the 256-thread group for
small dim.

Raw results: `perf/results/<date>/<run-id>/`.

## 2026-07-06: softmax (activations/softmax) — SYCL + oneDNN

Status: landed (SYCL + oneDNN vendor, best-routed).

Environment: same as the GELU entry (Arc Pro B60, oneAPI 2026.0, oneDNN 3.11).

Implementation: one work-group per row, two fp32 reductions (row max for
stability, then sum of exp). `kernels/activations/softmax/variants/{xpu_sycl,
xpu_onednn}` (oneDNN `softmax_accurate`, axis=1). Dispatch in
`src/dispatch/activations.cpp`.

Correctness: softmax x {f32,f16,bf16} x {sycl,vendor}, rows=64 dim=1024, vs an
fp64 max-subtract reference rounded to storage dtype; also checks each row sums
to 1. All 6 pass (row-sum error <= 5e-4).

Baseline (8192x4096, median device time): softmax f32 SYCL 316 / oneDNN 223
GB/s; softmax bf16 SYCL 195 / oneDNN 346 GB/s.

Decision: this REPRODUCES the layernorm per-dtype split on a second, independent
reduction kernel — native SYCL wins f32 (1.42x), oneDNN wins bf16 (1.78x). Not a
fluke; a robust pattern on B60: hand-written SYCL is competitive-to-winning at
f32, while oneDNN's 16-bit path is better tuned. `Variant::best` routes softmax
f32 -> sycl, 16-bit -> vendor. The bf16 native gap (195 vs f32 316) is again the
scalar-16-bit-access bottleneck flagged for a vectorization pass.

Raw results: `perf/results/<date>/<run-id>/`.

## 2026-07-06: 16-byte vector-load pass — bf16/f16 row kernels ~2x, one truism smashed and one own-conclusion reversed

Status: landed across gelu, rms_norm, layernorm, softmax (all dtypes).

Hypothesis (from the earlier entries): the bf16/f16 native row kernels ran ~2x
slower than f32 despite moving HALF the bytes, at the SAME wall time as f32 —
i.e. NOT bandwidth-bound. Diagnosis: the kernels read one 2-byte element per
work-item per step. Even though adjacent items were coalesced, the per-thread
transaction was too narrow to use the memory pipeline. (This is the exact
bottleneck the Metal notebook found for gelu_bwd: "scalar bf16 access, not the
tanh.")

Experiments (8192x4096, median device time; gelu at n=4194304):

1. Unrolled scalar V-block (thread handles V contiguous elements via a scalar
   loop): REGRESSED — f32 368 -> 209, bf16 184 -> 162. The compiler emitted V
   STRIDED scalar loads, breaking coalescing. Rejected.
2. Explicit `sycl::vec<T, V>` with V*sizeof(T) = 16 bytes, reinterpret the row as
   vectors, thread reads vector vi/vi+threads (adjacent threads -> adjacent
   vectors = coalesced AND wide): KEPT. `sycl::vec<bfloat16, 8>` compiles and is
   bit-exact (smashes the "can't vectorize bf16 in SYCL" assumption).

Results (GB/s, before -> after; roofline ~456):
- rms_norm:  f32 368->393, bf16 184->392 (2.1x), f16 160->393 (2.4x)
- layernorm: f32 387->393, bf16 196->388 (2.0x)
- softmax:   f32 316->389, bf16 195->302 (1.5x)
- gelu:      f32 329->409, bf16 230->406 (1.8x), f16 227->416

Correctness re-verified after each change (all dtypes x variants pass).

Routing consequences (data reversed a prior decision):
- layernorm bf16: native SYCL 196->388 now BEATS oneDNN 333. This OVERTURNS the
  2026-07-06 "route bf16 layernorm -> vendor" call. `Variant::best` for layernorm
  is now sycl at every dtype. (Do not treat even your own measured conclusion as
  permanent — improve the kernel and re-measure.)
- gelu f32: native 329->409 now TIES oneDNN 411 (was a 1.25x vendor win).
- softmax bf16: native 195->302 but oneDNN 348 STILL wins (exp()-bound, not
  load-bound) — kept `best` bf16 -> vendor. Honest nuance retained, not overclaimed.

Open questions: softmax bf16 vs oneDNN (exp throughput — try a faster exp
approximation within tolerance, or a single-pass online-softmax to cut one x
read); a `dim`/occupancy sweep (multiple rows per work-group for small `dim`).

Raw results: `perf/results/<date>/<run-id>/`.

## First Kernel Plan

Status: in progress — GELU + RMSNorm + LayerNorm + Softmax landed, both variants,
all dtypes vectorized (~90% of roofline). Next: GLU modes + GELU backward, then
Phase 2 quantization surface (qgemv/qgemm on XMX/DPAS int8).

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
