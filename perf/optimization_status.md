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

## 2026-07-06: activations breadth — silu, gelu_backward, glu (feature-matrix pass)

Status: landed (native SYCL, vectorized-by-default). Optimization intentionally
deferred (breadth pass); baselines recorded to satisfy the perf gate.

Implementation: silu and gelu_backward via the shared `kernels/common/vec_map.hpp`
(vec_unary / vec_binary) so they get the 16-byte vector-load treatment for free;
glu is a custom row kernel (gate/value halves) vectorized when d % V == 0. Modes:
swiglu/geglu/reglu/glu.

Correctness: silu, gelu_backward x {f32,f16,bf16} and glu {swiglu,geglu,reglu} x
{f32,bf16}, incl. a d=1000 (non-multiple-of-V) scalar-path case, vs fp64
references rounded to storage dtype. All pass. The test tolerance now adds one
storage ULP on top of the contract rtol: the kernel computes in fp32 and the
oracle in fp64, and a single bf16 ULP (~0.4%) exceeds the 2e-3 bf16 rtol, so
transcendental bf16 ops would otherwise fail by construction (silu bf16 hit
exactly 1 ULP = 0.03125 at silu(~6)).

Baselines (B60, GB/s): silu f32 408 / bf16 485; glu f32 392 / bf16 394.
Caveat: gelu_backward's bench aliases one buffer for both inputs, so its reported
~565/414 GB/s is cache-inflated and NOT a real bandwidth figure — recorded only
as a smoke/baseline, to be re-measured with distinct buffers if optimized.

Decision: keep as the shipped native implementations; revisit perf later per the
breadth-first directive.

## 2026-07-06: matmul/dense_gemm — native SYCL tile + oneDNN (XMX), family opened

Status: landed (both variants). Native SYCL is an untuned SLM-tiled baseline;
oneDNN is the XMX/DPAS-backed fast path.

Correctness: C=A*B for f32 + bf16, both variants, vs an fp64 reference (rtol
scaled by sqrt(K) for the accumulation). All pass.

Baseline (4096x4096x4096, GFLOP/s = 2*M*N*K/median):
- native SYCL tiled: f32 1014, bf16 1115.
- oneDNN vendor:     f32 12083 (12x), bf16 89891 (~90 TFLOP/s, 80x).

Decision: `Variant::best` -> vendor for matmul (the naive tile has no matrix
engine). This is the honest, expected gap: the native path needs XMX/DPAS
`joint_matrix` + register blocking to compete, which is the flagged native-GEMM
optimization (deferred under breadth-first). The ~90 TFLOP/s bf16 oneDNN number
confirms the B60 XMX is real and is exactly why the quantization/XMX surface is
the high-value frontier.

## 2026-07-06: three more families — rope (attention), adamw (optimizers), argmax (sampling)

Status: landed (native SYCL). Breadth pass; baselines recorded, optimization
deferred.

Correctness: rope (f32/bf16) vs an fp64 NeoX-rotation reference; adamw (f32
params) vs an fp64 fused-update reference; argmax (f32) exact-match vs a host
scan (0 mismatches). All pass.

Baselines (B60): adamw f32 339 GB/s (memory-bound, 6 buffer touches);
argmax f32 390 GB/s (near roofline); rope bf16 62 GB/s. rope is
transcendental-bound (sin/cos/pow per element) — the clear optimization is
precomputed per-position cos/sin (or a freq table), deferred.

## 2026-07-06: quantization/qgemv int4 — Marlin/Metal-guided dequant GEMV opens the family

Status: landed (native SYCL). Opens the quantization family with the flagship
batch-1 dequant-on-the-fly GEMV; three-step optimization following the Marlin
(wide loads + in-register decode + fused scale) and Metal (one-simdgroup-per-row,
branchless nibble decode) playbooks.

Format: int4 symmetric signed, packed 2/byte along K, per-group fp16 scales
(group 128). Decode: y[n] = sum_k int4(W[n,k]) * scale[n,k/group] * x[k], fp32
accum. Correctness (act f32 + bf16, N=128 K=4096) vs an fp64 reference: pass.

The truism under test: "4-bit weights are 4x smaller so a dequant GEMV beats an
fp16 GEMV." FALSE for a naive decoder, TRUE once optimized. Measured
(8192x8192, bf16 act, weight-bytes bandwidth; fp16 GEMV = oneDNN M=1 baseline,
0.303 ms, ~443 GB/s near roofline):

1. naive 1-byte-per-thread, work-group-per-row: 0.549 ms, 61 GB/s -> LOSES to
   fp16 (narrow 1-byte transactions; not weight-memory-bound).
2. + 16-byte wide loads (sycl::vec<uint32,4> = 32 int4/thread), decode in
   registers, one scale per 32-k chunk: 0.314 ms, 107 GB/s -> ties fp16.
3. + one 32-wide subgroup per row, 8 rows/work-group, subgroup reduce (no SLM
   barrier): 0.257 ms, 130 GB/s -> BEATS fp16 1.18x. KEPT.

Decision: ship step 3. Honest limit: 130 GB/s is well under the ~456 roofline,
so this is not yet weight-memory-bound -- remaining headroom is SLM activation
staging and cutting decode ALU / int->float conversions (Metal reached 2-3x over
fp16; we are at 1.18x). Flagged as the next qgemv optimization. This is also the
first proof point for the "all quant formats work on XPU" thesis: int4 group
decode runs natively and competitively, no NVIDIA hardware required.

## 2026-07-06: quantization/qgemm int8 w8a8 — oneDNN XMX hits 182 TOPS

Status: landed (native int8 tile + oneDNN vendor). Prefill counterpart to qgemv.

Format: A int8 [M,K] with per-row (per-token) fp32 scale; B int8 [K,N] with
per-col (per-channel) fp32 scale; C = (A@B) * a_scale[m] * b_scale[n], int32
accum, out f32/f16/bf16. Correctness (out f32+bf16, both variants) vs an fp64
reference: pass.

Obstacle smashed: oneDNN GPU int8 matmul rejects a per-M (per-token) SRC scale
attribute ("unsupported scales configuration", matmul.cpp:130) — it only takes
per-tensor src / per-channel weights scales. Rather than drop to per-tensor
activation scaling (wrong for dynamic per-token quant), the per-row scale is
applied as a binary-mul POST-OP with an [M,1] tensor that broadcasts over N;
the per-col weight scale stays a scale attribute. A try/catch falls back to the
native tile if a config is still unsupported. This is the honest way around a
vendor constraint — work with the API's real capabilities, verified on device.

Baseline (4096x4096x4096, GOPS = 2*M*N*K/median):
- native SYCL int8 tile: 826 GOPS (untuned, no DPAS joint_matrix).
- oneDNN vendor (XMX int8 DPAS): 182189 GOPS = ~182 TOPS (0.75 ms) — 2x the bf16
  dense GEMM (90 TFLOP/s) and 220x the naive tile.

Decision: best -> vendor for qgemm. The ~182 TOPS int8 confirms B60's DPAS int8
peak and is the real payoff of the quant surface. Native DPAS joint_matrix int8
is the flagged follow-up. Another proof for "quant works natively on XPU":
w8a8 int8 GEMM runs at full XMX throughput, no NVIDIA hardware. int8 format ->
experimental in quant-formats.yaml.

## 2026-07-06: quantization/fp8_gemm (e4m3, e5m2) — works natively, not yet fast on B60

Status: landed (oneDNN vendor-only). Also exposes fp8_encode/fp8_decode codecs
(oneDNN reorder) as public quant utilities.

The truism: "fp8 is a Hopper/Blackwell (NVIDIA) thing." FALSE for correctness —
both e4m3 and e5m2 matmul run natively on the B60 via oneDNN and are numerically
correct (max_abs 0 / 4.8e-7 vs an fp8-round-tripped fp64 reference; fp8 inputs
built with the oneDNN reorder codec so the reference sees exactly the values the
GEMM consumes). But ALSO measure the performance, don't assume it:

Baseline (4096^3, GFLOP/s): e4m3 15556, e5m2 18912 -- i.e. ~16-19 TFLOP/s.
That is FAR below this backend's int8 (182 TOPS) and even bf16 (90 TFLOP/s) GEMM.
Conclusion: Battlemage/Xe2 supports fp8 functionally but has no fast native fp8
XMX/DPAS path this generation, so oneDNN takes a slow/emulated route. So the
honest, nuanced claim is: fp8 WORKS on Intel (portable, correct — no NVIDIA
hardware needed), but is NOT accelerated on B60 today. Don't ship an fp8-fast
claim; do ship fp8 correctness/portability.

Decision: keep fp8_gemm (vendor) + codecs for correctness/portability; flag "not
accelerated on B60" in metadata. Revisit on a future Intel GPU with native fp8.

## 2026-07-07: quantization/mxfp4_gemv — OCP microscaling FP4 decodes natively on Intel

Status: landed (native SYCL, hand-written decoder — no vendor path).

Format: OCP mxfp4 — e2m1 (fp4) elements packed 2/byte, e8m0 (power-of-two) block
scale per 32 elements. Decoder: 8-entry e2m1 magnitude LUT {0,.5,1,1.5,2,3,4,6}
+ sign bit, block scale = 2^(e8m0 - 127). Reuses the tuned qgemv structure (one
32-wide subgroup per row, 16-byte wide loads); a 16-byte chunk is exactly one
32-element mx block, so one scale per chunk.

Correctness (act f32 + bf16, N=128 K=4096) vs an fp64 decode reference: pass
(f32 max_abs 1e-3, bf16 0). The point: mxfp4 is a data encoding (fixed LUT +
power-of-two scale), NOT Blackwell silicon — it decodes natively on the B60 via
a hand-written kernel.

Baseline (8192x8192, bf16): 0.386 ms, 87 GB/s weight bandwidth. About on par
with the fp16 GEMV (0.303 ms) and a bit behind the int4 qgemv (0.257 ms) -- the
extra e2m1 LUT + per-block exp2 decode ALU is the difference. Correct and
functional; optimization (hoist the block scale to a float pre-pass, cut decode
ALU) deferred. mxfp4 -> experimental in quant-formats.yaml.

## 2026-07-07: quantization/nvfp4_gemv — NVIDIA FP4 decodes natively on Intel too

Status: landed (native SYCL, hand-written decoder). Completes the FP4 story
alongside mxfp4.

Format: nvfp4 — e2m1 (fp4) elements + e4m3 (fp8) block scale per 16 + a
per-tensor fp32 global scale. Decoder: e2m1 LUT + a hand-written e4m3->float
(1-4-3, bias 7, subnormals). A 16-byte chunk (32 fp4) spans two 16-element
blocks, so two e4m3 scales per chunk.

Correctness (act f32 + bf16, N=128 K=4096): pass. The block-scale bytes are built
with the oneDNN fp8 codec (fp8_encode) and decoded back for the reference, so
passing ALSO confirms the hand-written e4m3 decode matches oneDNN's exactly --
a nice cross-check.

Baseline (8192x8192, bf16): 0.343 ms, 98 GB/s -- between mxfp4 (0.386) and int4
(0.257); the ldexp-based e4m3 decode is a touch cheaper than mxfp4's per-block
exp2. Correct + functional; same decode-ALU optimization headroom. nvfp4 ->
experimental.

Both "FP4 is Blackwell-only" (mxfp4 = OCP, nvfp4 = NVIDIA) claims are now
empirically false on Intel: both decode natively via hand-written kernels, no
special silicon.

## 2026-07-07: quantization/gguf_gemv — llama.cpp q8_0/q4_0 decode natively on Intel

Status: landed (native SYCL, hand-written decoder from the on-disk block layout).

Format: authentic GGUF blocks laid consecutively per row — q8_0 = {fp16 d; 32
int8} (34 B), q4_0 = {fp16 d; 16 packed bytes} (18 B), q4_0 dequant (nibble-8)*d
with low nibbles = elems 0..15, high = 16..31. One subgroup per row; each lane
decodes whole blocks; unaligned fp16 scale read via a 2-byte bit_cast.

Correctness (q8_0 + q4_0, act f32 + bf16) vs an fp64 reference using the exact
fp16-rounded scales and int quants: pass (bf16 max_abs 0).

Baseline (8192x8192, bf16, weight bytes incl. scales): q8_0 0.86 ms / 83 GB/s,
q4_0 0.61 ms / 62 GB/s. Slower than the packed formats (int4 130 GB/s) because
the interleaved 34/18-byte block layout is not GPU-coalescing-friendly (odd
strides, unaligned scale reads, scalar per-element decode). Correctness-first;
the optimization is a one-time repack from GGUF layout to a GPU-friendly
scale-planar + aligned-quant layout. GGUF k-quants decode natively on Intel.

## 2026-07-07: FULL-MATRIX push begins (breadth-first). serving family opened

Executing the approved full-matrix plan: breadth-first (one op per remaining
family to `partial`) then depth to full Metal parity (~230 ops). torch-xpu now
in .venv (torch 2.14 dev, 4 B60s) for the Python parity harness (Track C).

### serving — embedding_lookup + kv_cache_scatter + kv_cache_gather (native)
Indexed row copies, dtype-agnostic by element width, 2D coalesced launch.
Correctness: exact match (embedding gather; scatter->gather round-trip),
f32 + bf16, 0 mismatches. Baseline: embedding bf16 8192x4096 = 258 GB/s
(scattered table gather, below streaming roofline as expected). Native-only.

## 2026-07-07: Track C — PyTorch-XPU binding works end-to-end (validated vs torch.xpu)

`bindings/pytorch/tk_xpu`: torch.xpu tensors -> SYCL queue off the tensor's XPU
stream -> our ops ABI on the USM data_ptr, zero-copy. Parity harness
(test_parity.py) checks tk_xpu.<op> vs PyTorch's own XPU kernels across
f32/bf16/f16: gelu/silu/softmax/rms_norm/layernorm/attention/argmax/dense_gemm
ALL PASS (max abs err at storage-dtype epsilon).

Non-obvious blocker solved (looked like a SYCL version skew; was not — torch's
libsycl and system icpx are the identical 20260331 build). Two real causes:
(1) the ops must be a SHARED lib built with `icpx -fsycl` so device images
self-register — a static .a linked via plain c++ never registers -> submit
segfaults in ProgramManager::getDeviceKernelInfo; (2) the binding source must end
in `.sycl` so SyclExtension applies -fsycl + does the device link. Proven by a
minimal named-kernel .sycl running correctly on a torch.xpu tensor. Build via
bindings/pytorch/build.sh (BUILD_SHARED_LIBS=ON + .sycl source).

## 2026-07-07: quantization — act_quant (w8a8 activation quant) + all formats done

### quantization/quantize_int4_group — weight quantization (round-trips qgemv)
Symmetric group-wise int4 weight quant: per group scale=|max|/7, round+clamp
[-8,7], pack 2/byte in exactly the layout qgemv_int4 decodes. Correctness:
quantize -> qgemv_int4 round-trip matches a host decode exactly (excess 0) and
recon error within half a quant step, f32+bf16. The int4 weight path is now
end-to-end (quantize weights -> decode GEMV).

### quantization/act_quant — per-token int8 activation quantization
Work-group per row: reduce |max|, scale = |max|/127, round each element to int8.
Produces the int8 activations + per-row scales consumed by qgemm_int8 -> the w8a8
pipeline is now end-to-end (quantize acts -> XMX int8 GEMM at 182 TOPS ->
dequant). Correctness: exact per-row scale + reconstruction within half a quant
step, f32+bf16.

## 2026-07-07: Track B quant depth — GGUF q6_K k-quant

### quantization/gguf_gemv — q6_K (native k-quant decode)
First GGUF k-quant (256-element super-block, 210 bytes: ql[128]+qh[64]+int8
scales[16]+fp16 d; 6-bit quant = 4 low bits from ql + 2 high bits from qh,
recentred -32; per-16 sub-block int8 scales). Follows ggml dequantize_row_q6_K
exactly, one 32-wide subgroup per row. Correctness vs an INDEPENDENT host replica
of the ggml reference over random-byte blocks (no shared packer): worst_excess 0,
f32+bf16. Baseline 8192x8192 bf16 = 82 GB/s weight bandwidth. Proves the k-quant
super-block layout decodes natively on Intel.

### quantization/gguf_gemv — iq1_s: ALL GGUF FORMATS COMPLETE
iq1_s (50B: fp16 d + qs[32] + qh[u16*8]); 11-bit grid index (qs + 3 qh bits),
per-group scale + /-0.125 delta; iq1s_grid[2048] u64 (16KB, ported). Correctness
vs host replica: worst_excess 0, f32+bf16. With this, ALL 16 GGUF weight formats
decode natively on Intel (5 legacy + 5 k-quant + 6 i-quant), each vs a ggml
replica. q8_1/q8_K are activation intermediates (N/A on the weight path).

### quantization/gguf_gemv — grid i-quants iq2_xxs/iq2_xs/iq3_xxs
Codebook/grid i-quants. Ported the ggml grid tables (iq2xxs_grid[256] u64,
iq2xs_grid[512] u64, iq3xxs_grid[256] u32, ksigns_iq2xs[128], kmask_iq2xs[8]) to
a header (kernels/.../gguf_iq_tables.hpp), lazily uploaded to a device buffer and
cached. Decoders follow ggml dequantize_row_iq2_xxs/iq2_xs/iq3_xxs (grid lookup +
sign mask + block scale). Correctness vs independent host replicas using the same
tables: worst_excess 0, f32+bf16. Grid decode works natively on Intel. GGUF now
15 formats; only iq1_s (needs the u64 iq1s_grid) remains.

### quantization/gguf_gemv — q4_1/q5_0/q5_1 (legacy linear) + iq4_xs
Legacy 32-elem blocks: q4_1 (20B affine d/m), q5_0 (22B, 5-bit signed -16, 5th
bit from qh), q5_1 (24B, 5-bit affine). iq4_xs (136B super-block: 6-bit split
scales_l/scales_h + iq4_nl codebook). All follow ggml exactly; correctness vs
independent host replicas: worst_excess 0, f32+bf16. GGUF now 12 formats.
Grid-based i-quants (iq2_xxs/iq2_xs/iq3_xxs/iq1_s) next.

### quantization/gguf_gemv — iq4_nl (first i-quant, codebook)
iq4_nl = q4_0 footprint (18B: fp16 d + qs[16]) but the 4-bit index maps through
the fixed 16-entry non-linear kvalues_iq4nl codebook (no linear scale). ggml
dequantize_row_iq4_nl exactly. Correctness vs host replica: worst_excess 0,
f32+bf16. Baseline 8192x8192 bf16 = 59 GB/s. Proves the i-quant codebook approach;
iq4_xs + grid-based iq2/iq3/iq1 (need ported grid tables) next.

### quantization/gguf_gemv — q3_K (native k-quant decode, the fiddliest)
q3_K = 110-byte super-block (hmask[32], qs[64], scales[12], fp16 d). 3-bit quant
= 2 low bits (qs) + 1 INVERTED high bit (hmask); 6-bit scales via a custom
kmask1/kmask2 bit-shuffle. Ported ggml dequantize_row_q3_K + the scale unpack
exactly. Correctness vs independent host replica: worst_excess 0, f32+bf16.
Baseline 8192x8192 bf16 = 32 GB/s. ALL primary GGUF k-quants now native on Intel
(q8_0/q4_0/q6_K/q4_K/q5_K/q2_K/q3_K). i-quants (grid/codebook) next.

### quantization/gguf_gemv — q2_K (native k-quant decode)
q2_K = 84-byte super-block (scales[16] as 4-bit scale+min, qs[64] 2-bit, fp16
d+dmin), 16 sub-blocks of 16. ggml dequantize_row_q2_K exactly. Correctness vs
independent host replica: worst_excess 0, f32+bf16. Baseline 8192x8192 bf16 =
51 GB/s (fewer weight bytes at 2-bit).

### quantization/gguf_gemv — q5_K (native k-quant decode)
q5_K = q4_K + a 5th bit per quant from qh[32] (bit shifts by 2 per 64-block),
176-byte super-block. ggml dequantize_row_q5_K exactly. Correctness vs independent
host replica: worst_excess 0, f32+bf16. Baseline 8192x8192 bf16 = 81 GB/s.

### quantization/gguf_gemv — q4_K (native k-quant decode, the common one)
q4_K (144-byte super-block: fp16 d + dmin, 12-byte packed 6-bit scales+mins,
qs[128]; 8 sub-blocks of 32; w = d*sc*q - dmin*m). Ported ggml get_scale_min_k4
sub-scale unpacker + dequantize_row_q4_K. Correctness vs an independent host ggml
replica over random bytes: worst_excess 0, f32+bf16. Baseline 8192x8192 bf16 =
88 GB/s. q4_K is the format most real quantized LLMs ship in — now native on B60.
q5_K/q2_K/q3_K + i-quants next.

## 2026-07-07: Track B depth begins — attention + sampling suite

### sampling — sample_categorical + top_k_sample (native)
Reuses common/rng.hpp (stateless per-row uniform). Categorical: max/normalizer/
inverse-CDF scan over temperature-scaled logits. top_k: k-pass argmax selection
then softmax-sample within the top-k set. No oneDNN sampling primitive -> native.
Correctness by tie/precision-robust invariants: top_k sample's logit >= k-th
largest value; near-greedy (temp 1e-3) within margin of max. f32+bf16 pass.
Completes the inference sampling path (was argmax-only). Baseline categorical
4096x4096 = 34 GB/s / 1.96 ms. HONEST LIMIT: one-work-item-per-row with 3
exp-heavy vocab scans is slow at real inference shapes (few rows x 128k vocab);
the fix is work-group-per-row (parallel max/sum reductions like softmax + a
prefix-scan for the inverse CDF), flagged for the sampling optimization pass.

### attention — flash-style scaled dot-product attention (native)

### attention — flash-style scaled dot-product attention (native)
Online-softmax attention (running max/denom/weighted-acc per query), so the
seq x seq score matrix is never materialized — the flash property. One work-item
per (head, query) streams keys; supports MHA + GQA (q head -> kv head), causal
(end-aligned) and cross-attention (seq_q != seq_k), d<=128. Correctness vs fp64:
worst_excess 0 across f32 MHA-causal, bf16 GQA d=128, f32 cross-attn. Baseline
32h x 2048 x 64 causal = 191 GFLOP/s. This is the correctness-first per-query
shape; the SLM-tiled joint_matrix flash variant (XMX QK/PV, subgroup dot) is the
throughput optimization, deferred. oneDNN-Graph SDPA is the vendor variant path.

### collectives — all_reduce_sum (native multi-GPU) — BREADTH PASS COMPLETE
Real sum all-reduce across all 4 Arc Pro B60s. The B60s appear on 2 backends
(Level Zero + OpenCL), so a single context can't span platforms — select the
single platform exposing the most GPUs (L0, 4 devices), one queue per device,
shared context, reduce onto GPU0 via cross-device USM copies, broadcast.
Correctness: ng=4, sum(1..4)=10, 0 mismatches — verified on the real 4-GPU box.
Native; oneCCL ring/tree all-reduce is the vendor variant (deferred).

With this, EVERY kernel family has at least one implemented op (breadth-first
Track A done): activations, norms, matmul, attention, optimizers, sampling,
quantization, serving, utils, moe, linear_attention, ssm, collectives. Next:
Track B depth waves to full Metal parity + Track C PyTorch-XPU binding.

### ssm — selective_scan (Mamba S6 forward, native)
One work-item per channel; state vector h[state] in registers; the scan runs
sequentially over seq (inherently serial recurrence), parallelism from the many
independent channels. fp32 recurrence. Correctness vs fp64 (worst_excess 0,
sqrt(seq) tol), f32+bf16. Baseline 4096 chan x 2048 x 16 state = 1.5 Gelem/s
(sequential-bound). Native-only; the chunked parallel scan (ssd_chunk_*) is the
optimization, deferred to the ssm depth wave.

### linear_attention — linear_attn (non-causal, native)
Exploits linear-attention associativity O = Q(K^T V): one work-group per head
builds the (dim x dim) KV state + (dim) normalizer z in SLM (O(seq*dim^2)), then
applies O[t] = (Q[t] @ KV)/(Q[t].z). dim<=64 SLM path. Correctness vs fp64
(worst_excess 0, sqrt(seq)-scaled tol), f32+bf16. Baseline 32h x 2048 x 64 =
136 GFLOP/s (naive triple loops; register-tiling deferred). Native-only.
Causal/decay/chunked + gdn/based/hedgehog land in the depth wave.

### moe — moe_route_topk (native)
Top-k expert routing: one work-item per token, iterative argmax selection over
expert logits + softmax over the k selected. Correctness vs fp64 (exact top-k
ids, weights max_abs 5e-8), f32 + bf16. Baseline: 32768 tok x 128 experts, k=4 =
70 Mtok/s. Native-only (no oneDNN routing primitive). moe_grouped_gemm + the
gather/scatter/finalize ops land in the moe depth wave.

### utils — dropout + cross_entropy + hadamard (native)
New `kernels/common/rng.hpp` (stateless PCG counter-based uniform). dropout
(inverted, elementwise + RNG), cross_entropy (per-row logsumexp - logit[target],
softmax-shaped reduction), hadamard (FWHT, SLM butterfly, log2(n) passes).
Correctness: dropout zero-frac 0.302≈p + deterministic + kept=1/(1-p);
cross_entropy vs fp64 (max_abs 1.2e-6); hadamard vs fp64 FWHT with sqrt(n)-scaled
tolerance (accumulation error grows ~sqrt(n)). Baselines bf16: dropout 150,
cross_entropy 93, hadamard(n=1024) 69 GB/s (RNG/exp/butterfly compute-bound).
Native-only.

## First Kernel Plan

Status: in progress — 7 families now have implementations: activations (gelu,
gelu_backward, silu, glu, softmax), norms (rms_norm, layernorm), matmul
(dense_gemm), attention (rope), optimizers (adamw), sampling (argmax),
quantization (qgemv int4). Next: quant GEMM (int8 w8a8 via oneDNN + native),
more formats (fp8/mxfp4/GGUF decode), attention depth.

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

## 2026-07-07: Optimization pass — profiled all kernels, fixed the biggest gaps

Ran a full bench sweep vs the ~456 GB/s B60 roofline. Most elementwise/row
kernels were already 82-87% (silu/glu/rms_norm/layernorm/gelu, vectorized); the
GEMM vendor path is at 90 TFLOP/s. Attacked the kernels far below roofline that
had known fixes. All correctness gates still green; every number is a measured
median (50 iters, 15 warmup) on Arc Pro B60.

| kernel | before | after | speedup | fix |
|---|---|---|---|---|
| sample_categorical | 45.6 ms | 0.87 ms | **52x** | one-item-per-row -> work-group-per-row; parallel max/Z reductions; exclusive_scan_over_group for the inverse-CDF selection over contiguous chunks |
| argmax | 80 GB/s | 395/447 GB/s (bf16/f32) | **4.9x** | serial 256-iter final reduction -> SLM tree reduction; 16-byte vectorized scan |
| dropout | 144 GB/s | 400 GB/s | **2.8x** | scalar per-element -> 16-byte vectorized load/store (RNG index preserved, mask bit-identical) |
| quantize_int4 | 43 GB/s | 121 GB/s | **2.8x** | scalar group scan -> 16-byte vectorized double read pass |
| rope | 151 GB/s | 284/400 GB/s (bf16/f32) | **1.9x** | flat-id div/mod -> 3D range; pow(exp+log) -> exp2; cos+sin -> sincos |

Truisms confirmed/smashed: (1) argmax "reduction-bound" was really "serial-tail
bound" — the parallel scan was fine, one thread's 256-iter final loop dominated.
(2) rope "transcendental-bound" was PARTLY integer-div/mod-bound (flat-id
decomposition); the 3D range gave more than exp2/sincos did. (3) cross_entropy
reads the row twice so its "34% roofline" is a bench single-pass undercount (~300
GB/s effective) — left as-is.

Left as-is (already near roofline or inherently sequential): silu/glu/rms_norm/
layernorm/gelu/gelu_backward (82-87%), adamw (74%), dense_gemm vendor (90
TFLOP/s), softmax (69%). Deferred as bigger projects: native dense_gemm XMX
joint_matrix (1.1 vs 90 TFLOP/s vendor, task #9); quant-GEMV decode coalescing
(qgemv_int4/gguf/mxfp4/nvfp4 at 13-25% weight-bw); selective_scan/linear_attn
(sequential recurrence).

## 2026-07-07: Optimization pass #2 — 4-bit quant-GEMV decode (truism busted)

The quant-GEMVs sat at 13-25% of the 456 GB/s roofline and their comments *claimed*
"weight-memory-bound". Measured: they are **decode-ALU/latency-bound**. Three
fixes, applied uniformly (correctness exact, bf16 max_abs=0 vs host replica):
(1) hoist the per-block/per-group scale OUT of the nibble loop — one mul per
8-nibble word or per block instead of two muls per nibble; (2) vectorize the
activation read — one `sycl::vec<T,8>` per word vs 8 scalar loads; (3) accumulate
each of the 4 words into its own register so they're independent (breaks the
serial `acc` dependency chain -> ILP).

| kernel (8192x8192 bf16) | before | after | speedup |
|---|---|---|---|
| qgemv_int4 | 116 GB/s (0.288 ms) | 167 GB/s (0.201 ms) | **1.44x** |
| mxfp4_gemv | 80.7 GB/s (0.416 ms) | 119 GB/s (0.282 ms) | **1.48x** |
| nvfp4_gemv | 78.5 GB/s (0.427 ms) | 114 GB/s (0.294 ms) | **1.45x** |

Truism busted: "int4/fp4 GEMV is weight-memory-bound" — false on B60 at this
decode cost. Still ~37% of roofline, so decode ALU (nibble unpack + sign +
fp-convert) remains the ceiling; further wins would need SIMD unpack or a LUT, or
a repack to a coalescing-friendly layout. GGUF GEMVs (18-byte block stride fights
16-byte coalescing) are the remaining decode target.

## 2026-07-08: quantization/fp8_gemm — "fp8 is not accelerated on B60" mostly busted

Status: landed (native SYCL M=1 GEMV, NEW + vendor oneDNN fixed). Supersedes the
2026-07-06 verdict, which measured only the compute regime and only the shipped
path (which was paying per-call setup).

Correctness: `ctest --preset sycl` 3/3; fp8_gemm vendor + sycl checks all
max_abs=0 (the native bit-cast decode is exact — e5m2 IS truncated f16, e4m3
relocates into the f16 grid with a 2^8 factor, subnormals included) except
vendor e5m2 at 4.8e-7. Ragged N=101 / K=203 and bf16-out paths exercised.

Three findings, one per bottleneck actually measured:

1. **M=1 (decode) was never memory-bound on the vendor route — it was the wrong
   tool entirely.** oneDNN fp8 matmul at M=1 8192x8192: 4.09 ms = 15.7 GB/s =
   3.4% of roofline. New native SYCL GEMV (thread-per-8-columns over B[K,N],
   32 K-slabs + f32 partials, bit-cast pair-decode): **0.22-0.26 ms, 260-310
   GB/s, 16-19x faster**. LLM shapes hold up: 14336x4096 224 GB/s, 4096x11008
   221 GB/s. `Variant::best` now routes M=1 -> native GEMV.
   - The decode-ALU lesson from pass #2 recurred and was fixed the same day it
     was reintroduced: a scalar per-byte decode ran 116 GB/s with e5m2 36%
     faster than e4m3 (= ALU-bound). Pair-decode in u32 registers (two f16
     patterns per masked-shift pass, one vector convert) + folding the e4m3
     2^8-per-operand factor into ONE reduce-kernel multiply (exact) closed the
     kind gap (262 vs 278 GB/s) and got 2.3x over the scalar decoder.
   - Rejected: kSlabs=64 (211 GB/s vs 262 at 32 — partial traffic + shorter
     slabs lose to the occupancy gain).

2. **Half the shipped vendor cost was per-call engine/primitive re-creation.**
   Standalone A/B at 4096^3 e4m3: cached primitive 3.83 ms vs shipped 6.87 ms.
   Fix: engine cache per sycl::context + primitive cache per (shape, kind,
   out_dt, scale) in the oneDNN variant.

3. **`fpmath_mode::f16` reroutes fp8 onto the XMX f16 path** (up-convert is
   lossless for both kinds; accumulation stays f32): 44.8 vs 35.8 TFLOP/s at
   4096^3 e4m3. bf16/tf32/any modes measured identical to f16.

| shape / route | before | after | speedup |
|---|---|---|---|
| M=1 8192x8192 e4m3 (best) | 4.09 ms, 15.7 GB/s | 0.22 ms, 310 GB/s (68% roofline) | **18.9x** |
| 4096^3 e4m3 (vendor) | 6.87 ms, 20.0 TFLOP/s | 3.10 ms, 44.4 TFLOP/s | **2.2x** |
| 4096^3 e5m2 (vendor) | 18.9 TFLOP/s | **85.3 TFLOP/s** (95% of bf16 XMX peak) | **4.5x** |

The honest updated claim: e5m2 GEMM now runs at effectively full XMX speed on
B60; e4m3 GEMM at ~half (oneDNN's e4m3->f16 up-convert is the gap — theirs, not
ours); fp8 decode-GEMV is a first-class native path at 50-68% of bandwidth
roofline. Remaining headroom: the e4m3 vendor up-convert, and the GEMV's last
~30% (per-call scratch malloc/free + partial traffic are candidates).

## 2026-07-08: quantization/nvfp4_gemv pass #3 — bit-relocation decode, 1.92x

Status: landed. The fp8_gemv lesson transferred wholesale: e2m1 ALSO embeds
exactly in the f16 grid. A nibble s.e1e0.m relocates as (s<<15)|(e<<10)|(m<<9)
= value * 2^-14 (exact incl. the 0.5 subnormal), and the e4m3 block scale
relocates as ((b&0x80)<<8)|((b&0x7f)<<7) = value * 2^-8 (replacing a branchy
ldexp decode). The hot loop now has NO LUT, NO sign select, NO ldexp: per u32
word, 4 masked-shift pair passes + 4 vec2 converts decode all 8 nibbles, and
the uniform 2^-22 per product is compensated by one host-side multiply folded
into the global scale (zero device cost).

Correctness: ctest 3/3; nvfp4 checks identical to the LUT decoder (bf16
max_abs=0, f32 4.9e-4 vs fp64 reference).

| act dtype (8192x8192) | before | after | speedup |
|---|---|---|---|
| bf16 | 0.293 ms, 114.6 GB/s | **0.153 ms, 219.6 GB/s (48% roofline)** | **1.92x** |
| f16 | 0.302 ms, 111.0 GB/s | 0.161 ms, 207.8 GB/s | 1.87x |
| f32 | 0.373 ms, 90.0 GB/s | 0.328 ms, 102.3 GB/s | 1.14x (f32 x-loads dominate) |

Truism update: pass #2 said further decode wins "would need SIMD unpack or a
LUT" — the actual answer was neither: bit-relocation into f16 IS the SIMD
unpack, and it deletes the LUT. Follow-ups this unlocks: the same trick applies
verbatim to mxfp4_gemv's e2m1 nibbles (its e8m0 scale already folds via exp2),
and plausibly to GGUF q4_0-style scale*int decodes via a fixup constant.

## 2026-07-08: vLLM integration — native NVFP4 MoE + fp8 W8A16 GEMV (Qwen3.6-35B-A3B-NVFP4, 1x B60)

Status: landed (serving-level). These are QuixiCore-derived kernels vendored into
the vLLM tree (`~/vllm/csrc_xpu_quixi/`, self-contained torch SyclExtension, no
oneDNN link) and wired into vLLM's kernel selection. **Evidence caveat:** numbers
here are end-to-end serving decode (greedy tok/s) + torch parity, NOT the C++
profiling-event GB/s harness — the per-kernel `quixicore_xpu_bench` micro-baseline
for the two NEW kernels (nvfp4 fused MoE, fp8 W8A16 GEMV) is the remaining
paperwork. Model: 40 layers (30 GDN linear-attn + 10 full-attn), 256-expert NVFP4
W4A16 MoE (top-8), NVFP4 lm_head/shared, fp8 attn/GDN projections. enforce-eager,
max-model-len 4096, max-num-seqs 4 (single-GPU; TP disabled — PCIe oneCCL hangs).

Method: profiled a 48-token decode with the torch/XPU profiler (SYCL kernel
events). Baseline device compute 39 ms/tok, of which the Triton NVFP4 MoE
emulation was 29.5 ms (75%); pipeline is host-launch-bound (2393 kernel
launches/tok, 65 ms enqueue) with 81 device->host syncs/tok. Ran one pass per
bottleneck; A/B via env toggles.

Correctness: greedy "The capital of France is" -> "Paris, a city renowned..."
unchanged across every kept change. Per-kernel parity vs the path replaced:
nvfp4 MoE exact vs dequant-ref; fp8 W8A16 rel 2.5-3.2e-3 vs torch dequant.

| pass | change | tok/s | verdict |
|---|---|---|---|
| baseline | native NVFP4 linear only (emulation MoE + oneDNN fp8) | 14.1 | — |
| host-syncs | cache global-scale as host float (kill 81 syncs/tok) | 14.5 | keep (+2.5%) |
| native MoE | fused NVFP4 MoE SYCL kernel | 17.7 | **keep (+22%)** |
| native fp8 | bf16 x fp8-weight decode GEMV vs oneDNN fp8_gemm_w8a16 | 18.9 | keep (+7%) |
| M-tiled nvfp4 | decode-once/reuse-across-M GEMM | 18.9 | **reject** |

Cumulative 14.1 -> 18.9 tok/s (+34%); vs the pre-native Triton-emulation serving
baseline (6.0 tok/s) this is 3.15x. Findings:

1. **Fused NVFP4 MoE (new kernel, biggest win).** One work-group per routed
   (token,expert) pair: w13 decode-GEMV -> SwiGLU -> w2 decode-GEMV, atomic
   weighted sum, reusing the pass-#3 e2m1/e4m3 bit-relocation decode. Replaces
   the emulation's align + 2 dequant-GEMMs + activation + sum (~5 launches/layer,
   BLOCK_M-padded to 16-32 rows for 1-8 real tokens). Profiled MoE device time
   29.5 -> 16.1 ms/tok. Honest limit: at decode M=4 the grid is only M*T=32
   work-groups -> ~6.5% of the 456 GB/s roofline; occupancy is the next lever
   (a two-kernel g/a-then-o split, or M*T*rowtiles, to fill 160 XVEs).

2. **fp8 W8A16 as a native decode-GEMV, not oneDNN.** The QuixiCore fp8 result
   (this file, 2026-07-08) is W8A8 [K,N]; the vLLM path is W8A16 (bf16 act x fp8
   weight) through oneDNN `fp8_gemm_w8a16` firing ~190 gemm_kernel/tok. Wrote a
   bf16xfp8 [N,K] subgroup-per-row decode-GEMV (fp8 checkpoint weight is already
   [out,in]=[N,K] -> zero transpose, memory-neutral) and routed the GDN/attn
   projections through it. +7% (fewer + leaner launches; device fp8 3.9 -> ~1 ms).

3. **host-syncs were self-inflicted.** The 81 `aten::item`/tok traced (via the
   profiler's with_stack frames) to a `float(weight_global_scale)` on a 0-dim XPU
   tensor every forward in the native NVFP4 *linear* kernel. Caching it host-side
   at load removed all 81. Small (+2.5%) because the pipeline is launch-bound, not
   sync-bound (wall ~= enqueue, not compute) — matched the prediction.

4. **M-tiled nvfp4 GEMM rejected.** Decoding each weight row once and accumulating
   M partials (to avoid the row-loop's M-fold weight re-read) is slower than the
   row-loop for M<=6 — the acc[8]+wv[32] register pressure loses the fused
   decode-dot; it only wins at M>=8. Decode is M=1 (single stream) / M<=4
   (max-num-seqs). Reverted; kernel kept unused in `nvfp4_kernel.hpp` for future
   batched/prefill work. Parity was fine (rel 2.2-2.5e-3, M=1..8).

Remaining levers (measured, not yet done): MoE-kernel occupancy (the 6.5%-roofline
gap above); the pipeline is host-launch-bound (2273 launches/tok after the MoE
fusion) so XPU-graph capture is the largest single lever but out of scope for a
kernels-first pass. All changes are behind env toggles
(`VLLM_QUIXI_NVFP4_MOE_DISABLE`, `VLLM_QUIXI_FP8_DISABLE`,
`VLLM_QUIXI_NVFP4_DISABLE`) for clean A/B.

### Follow-up run #2: MoE occupancy 2-kernel split — device win, serving wash (launch-bound wall)

Measured the MoE kernel in isolation (weight-BW GB/s, real decode shape
E=256/2I=1024/K=2048/I=512, Tk=8): it is occupancy-starved at low batch because
the fused design uses only M*Tk work-groups.

| M | WGs (M*Tk) | fused GB/s | split GB/s |
|---|---|---|---|
| 4 (decode) | 32 | 77 (17% roofline) | **103 (+34%)** |
| 8 | 64 | 195 | 177 |
| 16 | 128 | 238 (52%) | 196 |
| 32-128 | 256-1024 | 219-250 | ~199 |

Split = two kernels (G: g[2I] per (pair,row-tile) -> f32 scratch; O: SwiGLU + o[K]
per (pair,row-tile), atomic weighted-sum). Parity exact (7e-8 vs fused). It wins at
M<=8 (decode) by spreading rows over thousands of WGs, and loses at M>=16 (extra
scratch + atomic traffic once occupancy is already met).

**End-to-end serving A/B was flat: 18.85 vs 19.0 tok/s.** The split adds +1
launch/MoE-layer (+40/tok); the vLLM decode pipeline is host-launch-bound
(~2273 launches/tok), so the device speedup is exactly cancelled by the extra
enqueue. Rejected for serving (kept behind `VLLM_QUIXI_MOE_SPLIT=1`). Lesson,
consistent with the whole pass: at this operating point end-to-end tok/s tracks
kernel-launch COUNT, not device time — the wins that landed (fused MoE 5->1
launches, native fp8 replacing ~190 oneDNN kernels/tok) all reduced launches; the
ones that didn't (host-sync cache, this occupancy split) did not. Next lever is
launch reduction (XPU-graph capture / torch.compile / elementwise+norm+rope
fusion), not faster kernels.

## 2026-07-09: Deep dive — path to 60 tok/s (launch-reduction levers, Qwen3.6-35B-A3B-NVFP4, 1x B60)

Status: analysis + 3 decisive on-box experiments. Question posed: "how do we get
from ~19 to 60 tok/s — switch to int4, or push the nvfp4 kernels harder?" Answer:
neither. The wall is host launch count (~2273 launches/tok), not device compute,
so device-side dtype/kernel work cannot move it. The two levers that DO cut launch
count are both currently BLOCKED on this torch/driver stack; unblocking them is the
whole job.

Baseline (this session, clean): **18.75 tok/s** greedy decode, 128 tok, dead-stable
(6.83 s/128 tok = 53.3 ms/tok). enforce-eager, native kernels engaged.

Device-compute budget (from the 2026-07-08 trace, after the MoE+fp8 fusions):
~22-25 ms/tok. => a *perfect* launch-elimination ceiling is only ~40-45 tok/s.
**60 tok/s therefore requires BOTH: (1) remove the host-launch overhead to become
device-bound (~40-45 tok/s), THEN (2) cut device compute ~25-30% (MoE occupancy +
GDN fusion) to close 45->60.** No single lever reaches 60.

### Experiment A — Full-graph XPU capture (the 3x lever). BLOCKED (runtime segfault).
XPU graph capture IS wired in this tree: `torch.xpu.XPUGraph` exists,
`supports_xpu_graph()`=True, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, runner aliases
`torch.cuda.graph->torch.xpu.graph` (xpu_model_runner.py:59). Launched with
`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`, no
enforce-eager. Result: **SIGSEGV during capture entry** —
`c10::xpu::device_synchronize -> sycl::device_impl::wait -> queue_impl::~queue_impl
-> ur_loader::urQueueRelease` (use-after-free in Intel's UR loader). Reproduced
**identically with all our custom kernels disabled** (VLLM_QUIXI_*_DISABLE=1) => it
is a torch-2.13.0.dev20260603+xpu / Battlemage(Level-Zero V2) XPUGraph bug, NOT our
code. This crash is at capture setup (the wrapper's own synchronize), before any
decode request. Blocks the single largest lever (capture collapses ~2273 launches
into one replay).

### Experiment B — Inductor compile, cudagraph OFF (fuse elementwise, no capture). BLOCKED (our ops untraceable).
`--compilation-config '{"cudagraph_mode":"NONE"}'`, compile on. Dynamo errors: it
cannot trace our pybind kernels (`quixi_nvfp4.fp8_gemm_w8a16`, `nvfp4_moe`, ...)
because they are raw pybind functions, not registered torch custom ops (gb0007
graph-break -> hard error). Also flagged the `functools.lru_cache` on
`quixi_fp8_xpu.py:57` `_load_quixi`. => inductor fusion (which would coalesce the
norm/rope/residual/cast chains and cut launches WITHOUT needing capture) is blocked
until we register every quixi kernel via `direct_register_custom_op` with meta/fake
impls (vllm/utils/torch_utils.py:935). This is fully in our control, no driver dep.

### Confirmed: int4 and "faster nvfp4 kernels" are the wrong axis.
Device compute is not the wall (proof: 2026-07-08 occupancy split made the MoE
kernel +34% on-device and moved end-to-end tok/s by ~0). nvfp4 decode already runs
at 48% of the 456 GB/s roofline (bit-relocation, no LUT). int4 would give a
marginally cheaper decode, cost accuracy + a requant, and touch launch count by
zero. Rejected as a direction.

### Roadmap to 60 (ordered by leverage / our-control):
1. **Register quixi kernels as torch custom ops** (`direct_register_custom_op`,
   meta impls; drop/relocate the lru_cache off the hot call). Unblocks Experiment B
   => inductor fuses elementwise and cuts a real fraction of the 2273 launches with
   no driver dependency. Highest-leverage next step; prerequisite for piecewise
   cudagraph too.
2. **Get XPUGraph capture working** — either (a) piecewise (FULL_AND_PIECEWISE) once
   step 1 lets inductor split the graph (captures only compiled regions, runs
   attention + our ops eager — may dodge the full-capture-entry segfault), or (b) a
   torch-xpu / Level-Zero driver build where XPUGraph doesn't UAF. Capture is the
   only lever that gets the full ~3x (launches -> ~1 replay -> device-bound ~40-45
   tok/s).
3. **Device compute, only after 1-2** (once launch-bound is gone the earlier
   device wins stop being cancelled): land the MoE 2-kernel occupancy split
   (VLLM_QUIXI_MOE_SPLIT=1, +34% device MoE at decode), then a fused native GDN
   SYCL kernel (the 30 GDN layers are the biggest remaining launch+compute mass).
   These carry 45 -> 60.

Raw: baseline bench 18.69/18.74/18.75 tok/s. Crash logs in
scratchpad/vllm_graph.log, vllm_graph_stock.log, vllm_inductor.log.

## 2026-07-09: Path-to-60-tok/s deep dive — the wall is launch overhead, not kernels

Status: analysis + 3 decisive on-B60 experiments. Target 60 tok/s (from 18.75).
Model/config identical to the 2026-07-08 entry (Qwen3.6-35B-A3B-NVFP4, 1x B60,
enforce-eager, max-num-seqs 4).

Question posed: "int4 or better nvfp4 kernels to reach 60 tok/s?" Answer from the
data: **neither** — both optimize device compute, and device compute is not the
wall. We are launch-bound (~2273 kernel launches/tok; wall ~= host enqueue).

Numbers that frame it:
- 18.75 tok/s = 53.3 ms/tok wall (re-measured, 128-tok greedy, +-0.1 tok/s).
- Device compute ~23 ms/tok (post-fusion: MoE 16.1 + fp8 ~1 + nvfp4 GEMV 1.25 +
  GDN 0.65 + routing 0.6 + elementwise ~3 + attn/other).
- => ~30 ms/tok is pure host launch/dispatch overhead. 60 tok/s = 16.7 ms/tok,
  which is BELOW current device compute. So reaching 60 needs BOTH: kill the
  launch overhead (graph capture, ~53->~23ms => ~43 tok/s) AND then cut device
  compute ~23->16ms (kernel work) to close 43->60. Neither lever alone gets there.

Experiment A — full-graph XPU capture (FULL_DECODE_ONLY, VLLM_XPU_ENABLE_XPU_GRAPH=1,
compilation mode=NONE). This is the ~3x lever (collapses 2273 launches -> 1 replay).
Result: **SIGSEGV during capture entry**, in Intel's UR loader:
`c10::xpu::device_synchronize -> sycl queue_impl::~queue_impl -> urQueueRelease`
(use-after-free on a queue during the capture-entry synchronize). torch
2.13.0.dev20260603+xpu; torch.xpu.XPUGraph exists, supports_xpu_graph()=True.

Experiment B — same capture with ALL our custom kernels disabled (stock emulation
MoE + oneDNN fp8 + native-nvfp4 off). Result: **identical segfault**. => the crash
is a torch-xpu/Level-Zero XPUGraph bug on Battlemage, NOT our kernels. Full-graph
capture is currently non-viable on this stack.

Experiment C — inductor compile, cudagraph OFF (mode=inductor, cudagraph_mode=NONE)
to fuse elementwise/norm/rope without capture. Result: **engine-core init fails** —
`torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped`:
Dynamo cannot trace our raw-pybind ops (`quixi_nvfp4.fp8_gemm_w8a16`, nvfp4_moe,
nvfp4_gemm). They must be registered as PyTorch custom operators
(`direct_register_custom_op` + meta/fake impls, preallocated outputs, no in-op
device sync) before torch.compile OR capture can handle them.

Conclusions / roadmap to 60:
1. **int4 does not help.** At decode M<=4 the MoE is a memory-bound GEMV; int4 and
   nvfp4 both read 4 bits/weight => identical weight-byte traffic => identical
   bandwidth. int4 would only pay via an INT8-XMX (w4a8) matmul path, which needs
   compute-bound (large-M) operation — not our decode regime. nvfp4 stays; its
   bit-relocation decode is already 48% of the 456 GB/s roofline.
2. **Register the 4 custom kernels as torch custom ops** (we own them; low risk).
   Unblocks BOTH inductor fusion and graph capture. Do first.
3. **Graph capture is the only realistic route to ~43 tok/s.** After (2), retry
   FULL_DECODE_ONLY; the UR-loader segfault (Exp A/B) must be resolved separately
   — torch-xpu nightly bump, or narrower capture, or upstream/driver fix. This is
   the make-or-break gate for 60.
4. **Capture inverts the optimization calculus.** Device-side wins we REJECTED
   under launch-bound serving (the MoE occupancy 2-kernel split: device +34% at
   M=4, serving wash because +1 launch/layer) become KEEPS once capture removes
   per-launch cost. Phase 2 (43->60) = MoE occupancy split + faster fp8, re-run
   under capture. Only here is an int4 A/B even worth the paperwork (and per (1)
   it should lose).

Fallback if XPUGraph stays broken: manual launch reduction via native fusion of
the tiny elementwise (RMSNorm/residual/rope/SwiGLU/cast) and GDN small-op chains.
Estimated 2273 -> ~1200 launches => ~1.5-1.8x (~30 tok/s). Real but a grind and
cannot reach 60 alone. Capture remains the primary lever.

### 2026-07-09 Phase 0 de-risk: SYCL graph works AND native submit is ~10x cheaper than torch

Standalone SYCL probes on B60 (Battlemage, LZ V2, oneAPI 2026.0), server stopped,
QuixiCore::XPUOps shared lib. Two programs: graph_probe (record rms_norm->nvfp4_gemv
->silu, replay) and graph_probe2 (R tiny silu kernels, eager-submit vs graph-replay).

Gate results:
- **Gate B (no crash): PASS.** device.has(ext_oneapi_graph)=1, ext_oneapi_limited_graph=1.
  command_graph record -> finalize -> q.ext_oneapi_graph(exec) replay -> teardown runs
  clean, NO urQueueRelease UAF. The SYCL graph path DODGES the torch XPUGraph segfault
  (single persistent in-order queue, no transient-stream churn).
- **Gate A (correctness): PASS.** replay vs eager max_abs = 0.0 (bitwise).
- **Gate C (launch overhead), R silu kernels:**
  | R | eager us/it | graph us/it | graph speedup |
  |---|---|---|---|
  | 8 | 21.0 | 30.0 | 0.70x |
  | 32 | 82.8 | 90.6 | 0.91x |
  | 128 | 308.5 | 172.9 | 1.78x |
  | 512 | 1222.9 | 667.9 | 1.83x |

Interpretation — the decisive finding:
- Native SYCL eager submit is **~2.4 us/kernel, flat in R**. Torch dispatch in vLLM is
  **~23 us/op** (2273 launches/tok x 23us ~ 52ms host = the 18.75 tok/s wall). So going
  NATIVE (no Python/aten/dispatcher) is a ~10x host-overhead cut by itself.
- Graph replay is ~1.3 us/kernel at R>=128 (1.85x over eager) but has fixed per-replay
  overhead that loses below R~64. It is a SECONDARY optimization.
- Projection for real decode (~2273 kernels/tok, device ~23ms/tok):
  - native eager: host ~2273x2.4us = 5.5ms, overlaps device 23ms -> **device-bound ~43 tok/s**.
  - native + graph: host ~3ms -> still device-bound 23ms -> 43 tok/s (graph moot once device-bound).
- **Conclusion: the lever is a native SYCL decode engine, NOT graph capture.** Native
  in-order non-blocking submit alone reaches device-bound ~43 tok/s; the SYCL graph is
  gravy (and confirmed working as a fallback/refinement). Phase 3 device cuts (MoE
  occupancy split, fp8, GDN) then take 23->~16ms -> 60 tok/s. The torch-XPUGraph UAF
  is now irrelevant to the plan. Phase 0 GREEN -> commit to Phase 2 (native engine).

### 2026-07-09 Phase 1: quixi ops registered as torch custom ops (+ inductor negative result)

New module vllm/.../kernels/linear/nvfp4/quixi_ops.py: moved _load_quixi() there;
registered quixi_nvfp4_gemm / quixi_fp8_gemm_w8a16 / quixi_nvfp4_moe / _split via
direct_register_custom_op (dispatch_key XPU, fake/meta impls returning correctly-shaped
empty tensors). Schemas inferred OK. Call sites (quixi_xpu.py, quixi_fp8_xpu.py,
nvfp4_quixi_moe.py) route through is_compiling() dispatch wrappers: under torch.compile
-> torch.ops.vllm.quixi_* (Dynamo-traceable); eager -> direct pybind (no op-dispatch tax).

Results (bench.py 8333, greedy 128 tok, enforce-eager):
- eager, always-through-custom-op: 18.19 tok/s (-3% vs 18.75 baseline: op-dispatch tax in
  the launch-bound regime).
- eager, is_compiling() wrappers (direct pybind): **18.56 tok/s** (== baseline within noise).
  Correctness: "Paris..." (minor wording variance = known atomic-accum nondeterminism in
  the fused MoE kernel, not a regression; eager path is byte-identical to baseline).
- **inductor compile, cudagraph OFF: STARTS + serves (previously FATAL Dynamo
  "Unsupported: call to skipped function")** -> the custom-op registration unblocked
  tracing. BUT **4.8 tok/s = ~4x SLOWER than eager.** With cudagraph off, compile
  graph-breaks around every opaque quixi op + GDN splitting_ops and adds Dynamo/guard +
  Triton-XPU overhead with no offsetting fusion. REJECT inductor-without-cudagraph.

Conclusion: Phase 1 is INFRASTRUCTURE (ops now traceable/compilable, needed for the Phase 2
native bridge and any future capture path) with NO current serving win — inductor-compile
is a 4x loss here, confirming that only native execution (Phase 2) or working graph capture
cuts the launch wall. Serving stays enforce-eager at ~18.6 tok/s. Custom ops kept behind the
is_compiling() wrappers (zero eager cost). Proceed to Phase 2 (native SYCL decode engine).

### 2026-07-09 Phase 2 vertical slice start: native GDN decode core prototype

Status: first correctness slice landed in `~/vllm/csrc_xpu_quixi/` (prototype,
not routed into serving yet). Added `gdn_decode_kernel.hpp` plus pybind op
`quixi_nvfp4.qwen_gdn_decode(projected_qkvz, projected_ba, conv_state, ssm_state,
conv_weight, conv_bias, A_log, dt_bias, state_indices) -> (core_attn_out, z)`.

Scope intentionally narrow: Qwen3.5/Qwen3.6 non-interleaved decode path only
(`gqa_interleaved_layout=False`, no prefill/spec decode, TP=1), matching
`nvidia/Qwen3.6-35B-A3B-NVFP4` through `qwen3_5.py`. The op performs:
1. depthwise causal conv update over `[q,k,v]` channels (`8192`, kernel width 4),
   mutating `conv_state` and emitting convolved `mixed_qkv`;
2. packed recurrent GDN update over `[B,32,128,128]` SSM state:
   L2-normalize q/k, apply `q *= 128^-0.5`, compute
   `g=-exp(A_log)*softplus(a+dt_bias)`, `beta=sigmoid(b)`, decay state,
   delta-correct `v - S @ k`, update `S += beta*delta*k`, emit `S @ q`;
3. return `z` from the projected `[q,k,v,z]` tail.

Synthetic parity (`csrc_xpu_quixi/gdn_parity.py`, B=4, slots=8, random small
inputs, server stopped, B60 clean):

| dtype / state | core max_abs | core max_rel | z | conv_state | ssm_state max_abs |
|---|---:|---:|---:|---:|---:|
| f32 / f32 | 2.79e-09 | 3.93e-04 | 0 | 0 | 1.49e-08 |
| bf16 / f32 | 3.81e-06 | 7.09e-03 | 0 | 0 | 1.12e-08 |

Interpretation: numerical spec is correct for the deployed decode math. This is
not yet a performance kernel: recurrent update currently uses one work-item per
value lane and redundantly recomputes q/k norms. It is the parity anchor for the
next optimization step (one work-group per `(token,value-head)` with local/shared
q/k reductions and state-row vectorization), then layer-level parity against the
existing `_xpu_C.gdn_attention` wrapper.

Direct production-kernel check also passed. Calling `torch.ops._xpu_C.gdn_attention`
with synthetic non-spec decode metadata (same random tensors, bf16 activation,
f32 SSM state) and comparing against `quixi_nvfp4.qwen_gdn_decode`:

| comparison | max_abs | max_rel |
|---|---:|---:|
| core_attn_out | 1.53e-05 | 7.75e-03 |
| z | 0 | 0 |

This validates the new prototype against the actual vLLM XPU GDN kernel, not just
the Python reference.

Follow-up optimization in the same vertical slice:

1. Shared q/k norm per `(token,value-head)` work-group instead of recomputing q/k
   L2 norms independently for all 128 value lanes.
2. Removed the intermediate decayed-state store/reload; only final updated SSM
   state is written.
3. Cached normalized q/k in local memory once per work-group.

Synthetic parity remains inside tolerance:
- f32/f32: core max_abs 2.56e-09, z exact, conv_state exact, ssm_state max_abs
  1.49e-08.
- bf16/f32: core max_abs 3.81e-06, z exact, conv_state exact, ssm_state max_abs
  1.49e-08.
- Direct `_xpu_C.gdn_attention` comparison remains: core max_abs 1.53e-05,
  max_rel 7.75e-03, z exact.

Microbench (`csrc_xpu_quixi/gdn_bench.py`, 200 iters, B60 clean, bf16 activation,
f32 SSM state):

| B | quixi prototype before opt | after norm+state opt | `_xpu_C.gdn_attention` |
|---|---:|---:|---:|
| 1 | 0.115 ms | **0.019 ms** | 0.031 ms |
| 4 | 0.205 ms | **0.045 ms** | 0.037 ms |

Verdict: keep. The GDN core prototype is now faster than production for the B=1
single-stream decode case and close for B=4. This is still two launches (conv +
recurrent) and not wired into serving; next step is layer-level integration/parity
and then folding into the native decode engine so launch overhead is paid by the
engine, not torch.

Serving A/B of the layer hook confirms the launch-bound rule again:
- `VLLM_QUIXI_GDN_ENABLE=1` (quixi GDN through torch forward): best 18.71 tok/s.
- default production `_xpu_C.gdn_attention`: best 18.83 tok/s.

Verdict for serving hook: reject by default. The quixi core is faster in isolation
for B=1, but routing it through torch uses two native launches plus Python/op
dispatch allocation, replacing one production custom op. It belongs inside the
native decode engine where the launches are submitted directly/recorded, not as a
standalone torch-layer replacement. The hook is left opt-in only via
`VLLM_QUIXI_GDN_ENABLE=1` for future A/B.

### 2026-07-09 Phase 2 vertical slice: native SYCL GDN decode core (decode-only)

Status: landed as opt-in vertical slice behind `VLLM_QUIXI_GDN_ENABLE=1`.
Implementation already present in the local tree and validated this pass:
`csrc_xpu_quixi/gdn_decode_kernel.hpp` + `qwen_gdn_decode` binding in
`quixi_nvfp4_ext.sycl`; vLLM hook in `qwen_gdn_linear_attn.py::_try_quixi_gdn_decode`.
Scope is intentionally narrow: non-interleaved qkvz layout, TP=1, no prefill, no
spec decode, real Qwen3.6 GDN shapes `[q,k,v,z]=[2048,2048,4096,4096]`, BA `[64]`,
conv state `[slots,3,8192]` or `[slots,8192,3]`, SSM state `[slots,32,128,128]`.

Correctness:
- Rebuilt `quixi_nvfp4` extension cleanly (`icpx -fsycl`, torch SyclExtension).
- `csrc_xpu_quixi/gdn_parity.py` vs a PyTorch reference of the exact FLA decode math:
  - fp32/state fp32: core max_abs 2.56e-09, z exact, conv_state exact, ssm_state max_abs 1.49e-08.
  - bf16/state fp32: core max_abs 3.81e-06, z exact, conv_state exact, ssm_state max_abs 1.49e-08.

Microbench (`csrc_xpu_quixi/gdn_bench.py`, wall timing, 200 iters):
| B | quixi qwen_gdn_decode | production `_xpu_C.gdn_attention` | verdict |
|---|---:|---:|---|
| 1 | 0.024 ms | 0.031 ms | quixi +29% |
| 4 | 0.046 ms | 0.037 ms | quixi -24% |

Serving A/B, enforce-eager, port 8333, greedy 128-token single request:
- baseline after custom-op wrappers: ~18.56 tok/s (previous stable baseline 18.75 tok/s)
- `VLLM_QUIXI_GDN_ENABLE=1`: **19.19 tok/s best**

Decision: KEEP as opt-in for the single-request decode path. It is a modest but real
serving win (+2-3%) and proves the native-GDN vertical slice. Do not enable by default
yet because the B=4 microbench regresses vs `_xpu_C.gdn_attention`; the next GDN pass
should target either (a) a one-launch fused conv+recurrent kernel, or (b) a batch-aware
path that preserves the B=1 win while falling back to `_xpu_C` for B>1.

Correction / repeat on the same Phase 2 GDN slice:
- A later serving repeat with the same direct-pybind GDN path was 18.56 tok/s best,
  i.e. serving-neutral vs baseline. The earlier 19.19 sample should be treated as noise.
- Attempting to route GDN through a `torch.ops.vllm.quixi_qwen_gdn_decode` wrapper in eager
  also measured ~18.36 tok/s, so the GDN call site remains direct-pybind in eager. The custom
  op registration is kept for future compile/capture infrastructure only.
- Updated verdict: the native GDN decode core is CORRECT and useful as the Phase-2 vertical
  slice, but not a standalone serving win under torch eager. It reinforces the main plan:
  native per-op kernels are insufficient; the win comes from moving the full decode loop into
  native SYCL submit/engine so the ~10x host-dispatch reduction applies globally.
## 2026-07-09: vLLM full XPU graph + split NVFP4 MoE reaches 68 tok/s

Status: keep in the vLLM integration worktree. Hardware: one Intel Arc Pro B60;
model: `nvidia/Qwen3.6-35B-A3B-NVFP4`; bf16 activations, NVFP4 W4A16 routed MoE;
single-request greedy decode; 8 warmup tokens followed by 128 measured tokens.

The torch 2.13.0.dev20260603+xpu Python graph lifecycle crashed in device-wide
synchronization (`device_impl::wait -> queue_impl::~queue_impl ->
urQueueRelease`). Direct `at::xpu::XPUGraph` recording on torch's current
in-order queue succeeded, including the full 40-layer model, mutable GDN/KV
state, routing, attention, and PyTorch graph allocator pool. A native
current-queue wait replaced the broken all-queue synchronization; the
post-capture cache purge was skipped to retain graph-pool allocations.

Command: vLLM serve with `VLLM_XPU_ENABLE_XPU_GRAPH=1`,
`VLLM_QUIXI_MOE_SPLIT=1`, `FULL_DECODE_ONLY`, capture size 1, TP=1,
max-num-seqs=4. Benchmark: `csrc_xpu_quixi/perf_analysis/bench.py 8333 128`.

| mode | 128-token runs | decision |
|---|---|---|
| eager, fused MoE | 18.41, 18.44, 18.50 tok/s | baseline |
| full graph, fused MoE | 36.84, 36.88, 36.94 tok/s | keep graph |
| full graph, split MoE | 67.30, 68.07, 68.07 tok/s | keep split under graph |

Correctness: graph-split and eager-split produced byte-identical full 128-token
greedy output. The split kernel changes one token versus fused MoE in this sample
because atomic accumulation order changes floating-point rounding; graph replay
adds no further difference. Graph capture took 0.02 GiB and completed in under
one second for batch size 1.

## 2026-07-09: dual-B60 TP2, FP8 KV cache, 128k context

Status: working eager long-context configuration. Model:
`nvidia/Qwen3.6-35B-A3B-NVFP4`; two Arc Pro B60 devices; TP=2; FP8 E4M3 KV;
XPU FlashAttention; 8,192-token chunked prefill; exact 128,000-token synthetic
prompt followed by 128 greedy output tokens.

Result: 217.94 s TTFT (587 average prompt tok/s), then 14.29 decode tok/s at
128k context. The allocated KV cache capacity was 2,007,784 tokens, or 15.32
concurrent 131,072-token sequences by vLLM's accounting.

Two integration issues were isolated:

- Attention `_q/_k/_v/_prob_scale` buffers were explicitly created on CPU. FP8
  XPU FlashAttention lost the Level Zero device, while Triton's cache-write
  kernel rejected `k_scale` as an inaccessible pointer. Moving these registered
  scalar buffers to the query device on first forward fixed both backends.
- The native Quixi FP8 decode GEMV is unsuitable for large-M prefill because
  every M row rereads the weights. `VLLM_QUIXI_FP8_DISABLE=1` selects oneDNN's
  W8A16 FP8 GEMM for this configuration and materially improves prefill.

## 2026-07-09: segmented XPU Graph enables TP2 decode

Status: working. The monolithic TP2 graph hung because XCCL collectives cannot
execute while their submission queue is being recorded. PIECEWISE capture that
also broke attention replayed incorrect state. The kept design uses full decode
capture and breaks only tensor-returning collectives into eager segments.

Implementation:

- Breakable graph capture now supports out-of-place tensor results. Capture-time
  output addresses remain static; replay copies each fresh collective result into
  the original output before launching the next graph segment.
- XPU graph segments retain their capture stream and perform explicit queue
  handoffs around eager XCCL work.
- `all_reduce`, `reduce_scatter`, and `all_gather` use lazy breakable wrappers to
  avoid an import cycle in distributed initialization.
- FP8 linear dispatches M > 4 to oneDNN GEMM and M <= 4 to Quixi GEMV, preserving
  practical prefill and fast decode without duplicate weight storage.

Results, TP2, batch 1, 128 greedy output tokens:

| configuration | decode tok/s |
|---|---:|
| eager, split MoE, short prompt | 14.16 |
| segmented full graph, fused MoE, short prompt | 40.85 |
| segmented full graph, split MoE, short prompt | 64.81-65.22 |
| eager, FP8 KV, 128k prompt | 14.29 |
| segmented full graph, FP8 KV, 128k prompt | 57.67-58.53 |

Correctness: the 128-token TP2 graph-split completion was byte-identical to the
TP2 eager-split completion. The exact 128k synthetic request also completed
without XCCL, graph, or device errors. Only capture size 1 is validated.

## 2026-07-10: Port Qwen serving kernels back into QuixiCore-XPU

Status: ported and correctness-gated in the working tree. Source provenance was
reviewed before the port: the integration prototypes in
`~/vllm/csrc_xpu_quixi/` originated from this repository's NVFP4/FP8 kernels
and were developed locally for the Qwen optimization work. The backend port was
adapted to QuixiCore's MIT-licensed raw-pointer ABI; no external reference
implementation source was imported.

Scope:

- batched NVFP4 W4A16 row-loop plus the measured/rejected M-tiled research path;
- native and oneDNN FP8 W8A16 in checkpoint-native `[N,K]` layout;
- fused and high-occupancy split NVFP4 routed MoE;
- Qwen3.5/Qwen3.6 non-interleaved GDN decode state update;
- fused residual-add RMSNorm;
- framework-neutral oneAPI command graphs and a current-stream-only PyTorch
  `XPUGraph` wrapper.

Correctness:

- Pre-port baseline: `ctest --preset sycl --output-on-failure` passed 3/3.
- Post-port C++ oracle: `tests/xpu_ops_smoke.cpp` passes FP32/FP16/BF16 NVFP4
  GEMM, fused/split MoE, FP8 W8A16, GDN state transitions, and fused
  add-RMSNorm. The complete op gate reports `PASS`.
- Native graph replay: `tests/xpu_graph_smoke.cpp`, two replays with mutated
  input, max abs `4.77603e-08`.
- PyTorch parity: `bindings/pytorch/test_parity.py` reports `PASS`, including
  W8A16 BF16 parity within tolerance (max abs `7.812e-03`), exact fused
  residual mutation, fused/split NVFP4 MoE equality, split-MoE graph replay
  with persistent scratch, and two mutated-input XPUGraph replays. Torch and
  `tk_xpu` both enumerate the two physical B60 devices after filtering duplicate
  OpenCL aliases in favor of the Level Zero platform.
- Final repository gate after the port: `ctest --preset sycl
  --output-on-failure` passes 4/4 (backend, device, full ops, command graph).

Focused performance run: one Arc Pro B60, driver 1.14.37020, Unified Runtime
over Level-Zero V2, oneAPI DPC++ 2026.1.0. Each table entry is the median of
three independent process runs; min/max are across those runs. Each process
used the listed command with 8-15 warmups and 20-50 timed iterations. Raw data:
`perf/results/2026-07-10/qwen-serving-port/`.

```bash
./build-sycl/quixicore_xpu_bench --kernel <kernel> --variant <variant> \
  --dtype bf16 <shape arguments> --iters <20|30|50> --warmup <8|10|15>
```

| kernel / shape | baseline median (min-max) | candidate median (min-max) | decision |
|---|---:|---:|---|
| FP8 W8A16 M1 N4096 K4096 | oneDNN 0.05066 ms (0.04418-0.05284) | SYCL 0.04565 ms (0.04515-0.04864) | keep native at M=1 (9.9%) |
| NVFP4 GEMM M4 N4096 K4096 | row loop 0.17097 ms (0.17092-0.17133) | M-tiled 0.29380 ms (0.27223-0.29543) | reject M-tiled (1.72x slower) |
| NVFP4 MoE M1 E256 top8 K2048 I512 | fused 0.62034 ms (0.53299-0.62918) | split 0.15930 ms (0.15910-0.15935) | keep split (3.89x) |
| NVFP4 MoE M4 E256 top8 K2048 I512 | fused 0.69596 ms (0.69189-0.69874) | split 0.56130 ms (0.51841-0.56152) | keep split (19.4%) |
| Qwen GDN B1 | production `_xpu_C` 0.031 ms (prior same-host run) | SYCL 0.02467 ms (0.02458-0.02483) | keep experimental (20.4%) |
| fused add-RMSNorm rows1 dim4096 | exact Torch composite 85.801 us (81.492-89.537) | `tk_xpu` 7.847 us (7.739-7.953) | keep (10.93x API wall time) |

The W8A16 crossover was re-measured rather than copied from the older vLLM
threshold: oneDNN is already faster at M=2 (0.05186 vs 0.08639 ms) and M=4
(0.04931 vs 0.16433 ms). `Variant::best` therefore selects native only for M=1
and oneDNN for M>1. The final single-sequence graph configuration still takes
the native path.

Final harness validation corrected the multi-launch JSON statistics from one
batch average to the median/min/max of five profiled batches. Three independent
processes per route, 15 warmups and 50 calls per batch, preserved the crossover:

| shape | oneDNN median (process range) | native median (process range) |
|---|---:|---:|
| M1 N4096 K4096 | 0.05118 ms (0.04623-0.05217) | 0.03090 ms (0.02869-0.04803) |
| M2 N4096 K4096 | 0.05200 ms (0.05096-0.05228) | 0.07822 ms (0.07543-0.08505) |
| M4 N4096 K4096 | 0.04336 ms (0.03416-0.05000) | 0.14316 ms (0.13176-0.14482) |

The M1 native result is clock-state sensitive, which is reflected in the wider
range, but its process median and the earlier three-process run both beat
oneDNN. Raw output is in
`perf/results/2026-07-10/qwen-serving-port-harness-v2/`.

Graph decision: keep the reusable wrappers. Both native oneAPI command-graph
replay and direct `at::xpu::XPUGraph` replay are stable. Collectives remain
outside capture; TP2's eager-XCCL segment orchestration and stable-address copy
are vLLM integration policy, not duplicated in the framework-neutral kernel
library.

## 2026-07-10: Qwen GDN state-slot contract synchronization

Status: candidate correctness fix in the working tree; no speedup claim.

Contract change: state slot 0 is now a valid active slot. Negative and
out-of-range indices leave convolution and recurrent state untouched, return a
zero recurrent core, and still pass through the projected `z`. This matches the
vLLM scheduler-facing contract and removes the former first-slot correctness
hole. The benchmark now includes slot 0 instead of reserving it.

Environment: one Intel Arc Pro B60, oneAPI DPC++/C++ 2026.1.0 (20260617),
Unified Runtime over Level-Zero V2, driver 1.14.37020. Commands used bf16,
8 state slots, 25 warmups, and 100 timed calls; each process result is the
median of five profiled batches with batch-level min/max.

Correctness:

- `ctest --preset sycl --output-on-failure`: 4/4 passed. The C++ oracle covers
  slot 0 plus negative and upper-bound-invalid indices across f32/f16/bf16.
- `bindings/pytorch/test_parity.py`: PASS, including slot-zero mutation,
  invalid-index state preservation, exact `z` passthrough, manual GELU
  backward, and graph lifecycle checks.

| shape | pre-change process median (range) | post-change process median (range) | decision |
|---|---:|---:|---|
| GDN B1, slots8 | 0.02137 ms (0.02047-0.02382) | 0.02363 ms (0.02351-0.02370; 6 runs) | keep correctness fix; no performance claim |
| GDN B4, slots8 | 0.06800 ms (0.06622-0.06832) | 0.06580 ms (0.06573-0.06807) | keep correctness fix |

The B1 measurements remain clock-state sensitive: one pre-change process ran
at 0.02382 ms and overlaps the post-change process range. The valid path still
performs the same state update and `z` store; the change only broadens the
index predicate and moves the existing `z` store before it. No speedup or
regression claim is made from this run.

The corresponding `vllm-xpu-kernels` wrapper routing was checked on the same
device. At M1, automatic MoE selection matched explicit split (0.07806 vs
0.07817 ms wall median) and beat fused (0.37646 ms); at M4, automatic matched
split (0.28060 vs 0.28083 ms) and beat fused (0.35924 ms). Eager FP8 W8A16 M1
selected the native-eligible checkpoint view at 0.01418 ms versus 0.01628 ms
for a forced oneDNN layout at N=4096, K=2048. These are integration-level wall
timings, not QuixiCore device-event speedup claims.

Raw results: `perf/results/2026-07-10/gdn-contract-sync/raw.md`.

## 2026-07-10: NVFP4 split-MoE output-row tiling

Status: **candidate — KEEP** in the working tree. Public route:
`ops::nvfp4_moe_split` / `kernels::nvfp4_moe_split_sycl`, also exercised through
`tk_xpu.nvfp4_moe(..., split=True)`. Format is ModelOpt NVFP4 (e2m1 packed
weights, e4m3 block scales), bf16 activations, and f32 accumulation/output.

Environment: one Intel Arc Pro B60 (`0000:6c:00.0`, VTune target
`0:108:0.0`), 160 XVEs, oneAPI DPC++/C++ 2026.1.0
(`2026.1.0.20260617`), Unified Runtime over Level-Zero V2, driver
`1.14.37020`, VTune 2026.3, Ubuntu 26.04 / kernel 7.0.0-27. Two idle vLLM
workers remained resident across the A/B, so every final comparison alternated
preserved baseline and candidate executables after a long on-device warmup. The
reported runs held the GPU at 2.35-2.40 GHz and about 45-46 C.

### Bottleneck evidence

The untouched split route was profiled at the production decode shape
`M=1,E=256,top_k=8,K=2048,I=512`. VTune XPU Offload reported no spills and
split the device time almost evenly:

| task | baseline average per invocation | work size |
|---|---:|---:|
| `Nvfp4MoeGateUpKernel<bf16>` | 78 us | `32768 x 8` |
| `Nvfp4MoeOutputKernel` | 75 us | `65536 x 8` |

The output grid had 256 work-groups per routed `(token,expert)` pair. Every
work-group rebuilt the same 512-element SwiGLU activation in local memory and
paid a group barrier before its eight subgroups computed only one output row
each. Hypothesis: let every subgroup consume several output rows so activation
construction and the barrier are amortized, while retaining enough independent
work-groups to occupy the B60.

GPU hardware-counter collection was attempted but is unavailable on this host:
VTune reports that neither `libigdmd.so` nor `libmd.so` (Intel Metrics Discovery)
is installed. The optimization decision therefore uses SYCL profiling-event
timing plus the successful XPU Offload task timeline; no unsupported hardware
counter claim is made.

### Controlled row-tile sweep

Only output rows processed per subgroup changed. `M=1,top_k=8` bf16 results
used 500 warmups and 200 calls in each of five profiling-event batches:

| rows per subgroup | median ms | weight GB/s | decision |
|---:|---:|---:|---|
| 1 (baseline) | 0.07982 | 157.6 | reference |
| 2 | 0.07257 | 173.4 | better, continue |
| 4 | 0.06950 | 181.0 | better, continue |
| 8 | 0.06672 | 188.6 | better, continue |
| 16 | **0.06485** | **194.0** | best priority shape |
| 32 | 0.07698 | 163.5 | reject: only 64 output WGs, under-occupied |

The kept dispatch specializes `{1,2,4,8,16}` rows at compile time and chooses
the largest tile that leaves at least 128 output work-groups. This preserves the
original one-row mapping for small `K`/pair counts and selects 2/4/8/16 rows as
routing density rises. It changes neither launch count nor scratch/output
layout.

### Final A/B

Exact command shape (with `M` varied):

```bash
./build-sycl/quixicore_xpu_bench --kernel nvfp4_moe --variant sycl \
  --dtype bf16 --approx split --M <1|4|8|16> --N 256 --K 2048 \
  --rows 8 --dim 512 --iters 200 --warmup 500 --device 0
```

Each process result is the median of five device-event batches. M=1 and M=4
are medians of three independent, alternating baseline/candidate processes;
the parenthesized range is the range of those process medians. M=8/M=16 are
focused regression-shape repeats with the batch-level min/max shown.

| shape | baseline ms | candidate ms | weight GB/s before -> after | decision |
|---|---:|---:|---:|---|
| M1, top8 | 0.079858 (0.079841-0.079965) | **0.064962** (0.064932-0.065021) | 157.6 -> **193.7** | keep, **18.65%** lower latency |
| M4, top8 | 0.281078 (0.281006-0.293872) | **0.233018** (0.232967-0.233157) | 179.1 -> **216.0** | keep, **17.10%** lower latency |
| M8, top8 | 0.549860 (0.549707-0.550076) | **0.449256** (0.448996-0.475012) | 183.1 -> **224.1** | keep, **18.30%** lower latency |
| M16, top8 | 1.087130 (1.087040-1.087150) | **0.868502** (0.868380-0.868682) | 185.2 -> **231.8** | keep, **20.11%** lower latency |

Low-routing-density checks did not regress: M1/top-k 1 improved from 0.02820
to 0.02250 ms with the 2-row specialization, top-k 2 from 0.02817 to
0.02676 ms, and top-k 4 from 0.04587 to 0.03770 ms. The untouched fused route
was neutral: M1 0.31627 -> 0.31657 ms and M4 0.35020 -> 0.35033 ms.

Post-change XPU Offload kept gate/up at 77 us and reduced the output task from
75 to **46 us** (38.7%), shrinking its work size from `65536 x 8` to
`4096 x 8` at M1/top-k 8. Instrumented whole-op time moved from 0.1736 to
0.1406 ms; the uninstrumented event results above are the source of the speedup
claim.

Correctness:

- `ctest --preset sycl --output-on-failure`: **4/4 passed**. The independent
  C++ oracle now uses `M=8,top_k=8,K=256,I=32`, which selects the 16-row
  specialization; fused/split max abs is `2.33e-10` across f32/f16/bf16.
- Rebuilt the shared SYCL ops library and PyTorch extension with
  `bindings/pytorch/build.sh`; `bindings/pytorch/test_parity.py`: **PASS**.
  Direct split MoE and persistent-scratch XPU Graph replay both have max abs 0.

Rejected GDN experiments from the same pass:

1. Broadcasting uniform recurrence scalars (`decay`, `beta`) from one lane was
   neutral at B1 and up to ~2% slower at B4; the compiler already handles the
   uniform work well enough and the added SLM traffic did not pay back. Reverted.
2. A one-launch 256-lane fused convolution/recurrent kernel passed all dtype and
   invalid-state correctness checks, but improved sustained B1 only 1.8% and B4
   0.25%. Rejected and reverted because the complexity exceeded the measured win.

Raw benchmark record:
`perf/results/2026-07-10/nvfp4-moe-output-row-tiling/raw.md`. VTune results:
`perf/results/2026-07-10/nvfp4-moe-split-baseline-offload/` and
`perf/results/2026-07-10/nvfp4-moe-split-row-tile-offload/`.

## 2026-07-11: NVFP4 MoE paired gate/up decode

Status: **candidate — KEEP** in the working tree. This pass starts from the
kept pair-aware output-row tiling above and optimizes the remaining gate/up
bottleneck in both `nvfp4_moe_split_sycl` and the co-equal
`nvfp4_moe_fused_sycl` route. Format and public contract are unchanged:
ModelOpt NVFP4 e2m1 packed weights, e4m3 block scales, f32 accumulation/output,
and f32/f16/bf16 activations.

Environment: one Intel Arc Pro B60 (`0000:6c:00.0`, VTune
`0:108:0.0`), 160 XVEs, oneAPI DPC++/C++ 2026.1.0
(`2026.1.0.20260617`), Unified Runtime over Level-Zero V2, driver
`1.14.37020`, VTune 2026.3, Ubuntu 26.04 / kernel 7.0.0-27. The same idle vLLM
workers remained resident for baseline and candidate; final runs alternated a
preserved baseline executable with the candidate after 500 warmups. Longer
low-top-k runs used 2500-5000 warmups to hold the GPU at 2.35 GHz.

### Bottleneck and change

After output-row tiling, XPU Offload showed gate/up at 77 us and output at
46 us for `M=1,E=256,top_k=8,K=2048,I=512`. Gate row `i` and up row `i`
consume the same hidden vector but were evaluated by separate subgroups, so
each loaded and converted the same 2048 activation elements independently.

The kept `nvfp4_row_dot_pair` decodes the gate and up weight rows together. A
lane loads each hidden value once, applies it to both decoded weights, retains
two accumulators, and performs two subgroup reductions. Split gate/up now uses
one subgroup per logical gate/up pair; the fused route uses the same primitive
while filling its local gate/up buffer. The layout, accumulation precision,
scratch contents, output atomics, and launch count are unchanged.

Post-change XPU Offload reports SIMD32, zero spill bytes, gate/up **49 us** and
output 46 us. Gate/up work size falls from `32768 x 8` to `16384 x 8`, and its
time falls 36.4%. Instrumented whole-op time moves from the prior 0.1406 ms to
0.1144 ms. GPU hardware counters remain unavailable because Metrics Discovery
(`libigdmd.so`/`libmd.so`) is not installed; speedup claims use uninstrumented
SYCL profiling events.

### Final split-route A/B

```bash
./build-sycl/quixicore_xpu_bench --kernel nvfp4_moe --variant sycl \
  --dtype bf16 --approx split --M <1|4|8|16> --N 256 --K 2048 \
  --rows 8 --dim 512 --iters 200 --warmup 500 --device 0
```

M1/M4 are medians of three alternating independent processes; parentheses are
the process-median range. Each process result is itself the median of five
200-call event batches. M8/M16 are focused repeats with batch min/max.

| shape | output-tiled baseline ms | paired gate/up ms | weight GB/s before -> after | decision |
|---|---:|---:|---:|---|
| M1, top8 | 0.064926 (0.064888-0.065002) | **0.050748** (0.050686-0.050879) | 193.8 -> **247.9** | keep, **21.84%** lower latency |
| M4, top8 | 0.233093 (0.233074-0.233102) | **0.178114** (0.178043-0.178205) | 215.9 -> **282.6** | keep, **23.59%** lower latency |
| M8, top8 | 0.449206 (0.449035-0.449274) | **0.339395** (0.339355-0.339650) | 224.1 -> **296.6** | keep, **24.45%** lower latency |
| M16, top8 | 0.868634 (0.868274-0.868773) | **0.649967** (0.649870-0.668995) | 231.8 -> **309.7** | keep, **25.17%** lower latency |

The improvement is not bf16-specific at M1/top-k 8: f16 moves 0.06514 ->
0.05114 ms (21.49%) and f32 moves 0.07008 -> 0.05502 ms (21.49%). Sustained
M1 routing-density checks also improve: top-k 1 by 11.73%, top-k 2 by 20.26%,
top-k 4 by 20.72%, and top-k 8 by 21.78%.

The same factor improves the fused route without changing its occupancy model:

| shape | fused baseline ms | paired fused ms | decision |
|---|---:|---:|---|
| M1, top8 | 0.310482 | **0.280802** | keep, 9.56% |
| M4, top8 | 0.348085 | **0.296017** | keep, 14.96% |
| M8, top8 | 0.542395 | **0.406080** | keep, 25.13% |
| M16, top8 | 0.916574 | **0.644884** | keep, 29.64% |

Cumulatively versus the pre-output-tiling split kernel, M1 moves 0.07986 ->
0.05075 ms and M4 moves 0.28108 -> 0.17811 ms: about 36% lower latency in
two independently measured passes.

### Rejected factors

1. **SLM hidden staging:** copying the hidden vector once per gate/up
   work-group plus a barrier regressed M1 by 4.46% (0.06496 -> 0.06785 ms) and
   M4 by 6.93% (0.23301 -> 0.24916 ms). The existing cache path is already
   effective. Reverted.
2. **Generic two-row gate tiling:** processing two unrelated rows per subgroup
   regressed M1 by 1.52% (0.06487 -> 0.06585 ms) and was neutral at M4
   (0.23306 -> 0.23303 ms). Scheduling was not the bottleneck. Reverted.

### Correctness

- `ctest --preset sycl --output-on-failure`: **4/4 passed**. The independent
  host oracle exercises `M=8,top_k=8,K=256,I=32`; fused/split disagreement is
  at most `4.66e-10` across f32/f16/bf16.
- Rebuilt `build-sycl-shared` and the PyTorch extension with
  `bindings/pytorch/build.sh`; `bindings/pytorch/test_parity.py`: **PASS**.
  Direct split MoE and persistent-scratch XPU Graph replay both have max abs 0.

Raw benchmark record:
`perf/results/2026-07-11/nvfp4-moe-paired-gate-up/raw.md`. VTune result:
`perf/results/2026-07-11/nvfp4-moe-paired-gate-up-offload/`.

## 2026-07-11: NVFP4 MoE packed-dot vector loads

Status: **candidate — KEEP** in the working tree. This pass starts from the
paired gate/up and pair-aware output-row tiling above. It changes the shared
packed-dot implementation used by both `ops::nvfp4_moe_split` and
`ops::nvfp4_moe_fused`; format, public routing, launch shapes, fp32 accumulation
order, scratch layout, and output layout are unchanged.

Environment: one Intel Arc Pro B60 (`0000:6c:00.0`, VTune target
`0:108:0.0`), 160 XVEs, oneAPI DPC++/C++ 2026.1.0
(`2026.1.0.20260617`), Unified Runtime over Level-Zero V2, driver
`1.14.37020`, VTune 2026.3, Ubuntu 26.04 / kernel 7.0.0-27. The same two idle
vLLM workers remained resident throughout the alternating baseline/candidate
runs.

### Hypothesis and change

Both `nvfp4_row_dot` and `nvfp4_row_dot_pair` consumed each adjacent group of
eight activations through scalar indexed expressions. The standalone NVFP4 and
INT4 GEMV implementations already expose this access as `sycl::vec<T,8>`.
Hypothesis: an explicit vector load will reduce load/address work in the gate/up
global-hidden path and the output local-activation path without increasing
register pressure.

The kept factor loads one `sycl::vec<T,8>` per packed 32-bit weight word and
feeds its low/high halves to the existing decode and fp32 accumulators. XPU
Offload confirms unchanged SIMD32 code, zero spill bytes, and unchanged work
sizes. The instrumented gate/up average moves 48.98 -> 48.11 us; the output
average is profiler-noise neutral at 45.89 -> 46.47 us. Hardware counters remain
blocked by missing Metrics Discovery (`libigdmd.so`/`libmd.so`), so all speedup
claims below use uninstrumented SYCL profiling events.

### Final split-route A/B

```bash
./build-sycl/quixicore_xpu_bench --kernel nvfp4_moe --variant sycl \
  --dtype bf16 --approx split --M <1|4|8|16> --N 256 --K 2048 \
  --rows 8 --dim 512 --iters 200 --warmup 500 --device 0
```

M1/M4 are medians of three alternating independent processes; parentheses are
the process-median range. Each process is the median of five 200-call batches.
M8/M16 are focused repeats with the batch min/max.

| shape | paired-dot baseline ms | vector-load candidate ms | weight GB/s before -> after | decision |
|---|---:|---:|---:|---|
| M1, top8 | 0.050753 (0.050733-0.050834) | **0.049084** (0.049030-0.049216) | 247.9 -> **256.4** | keep, **3.29%** lower latency |
| M4, top8 | 0.178185 (0.177999-0.178189) | **0.174661** (0.174503-0.174690) | 282.5 -> **288.2** | keep, **1.98%** lower latency |
| M8, top8 | 0.339487 (0.339439-0.339680) | **0.335119** (0.335048-0.335284) | 296.5 -> **300.4** | keep, **1.29%** lower latency |
| M16, top8 | 0.649958 (0.649933-0.649988) | **0.642690** (0.642528-0.714032) | 309.8 -> **313.3** | keep, **1.12%** lower latency |

At M1/top-k 8, f16 improves 3.71% (0.051073 -> 0.049177 ms) and f32 is
neutral at 0.34% (0.055008 -> 0.054821 ms). Sustained bf16 routing checks do
not regress: top-k 1 improves 4.77%, top-k 2 by 3.64%, and top-k 4 by 4.66%.

The co-equal fused route also remains safe: M1 improves 5.84%, M8 by 1.27%,
and M16 by 4.02%. The noisier M4 result used three independent processes and
improves 1.99% by process median (0.296146 -> 0.290252 ms).

### Rejected factors

1. **Global activated scratch:** computing SwiGLU once in gate/up and removing
   output SLM staging/barriers regressed M1 by 11.8% and M4 by 11.9%. Reverted.
2. **Precomputed activation plus SLM staging:** retaining local output staging
   after writing activated values to scratch was neutral at both M1 and M4.
   Reverted because the extra contract/traffic had no measured payoff.
3. **Forced two-way chunk unroll:** `#pragma unroll 2` regressed M1 by 3.6% and
   M4 by 1.1%, consistent with extra live state. Reverted.

### Correctness

- Focused f32/f16/bf16 C++ oracle: fused and split pass with max abs
  `8.15e-10`.
- `ctest --preset sycl --output-on-failure`: **4/4 passed**.
- Rebuilt the shared SYCL ops library and PyTorch extension;
  `bindings/pytorch/test_parity.py`: **PASS**. Direct split MoE and
  persistent-scratch XPU Graph replay both have max abs 0.

Raw benchmark record:
`perf/results/2026-07-11/nvfp4-moe-packed-dot-vector-load/raw.md`. VTune result:
`perf/results/2026-07-11/nvfp4-moe-vector-load-offload/`.

## 2026-07-22: attention_f16ctx (fused f16 context store)

Status: landed.

Current implementation: `kernels/attention/attention/variants/xpu_sycl/attention_f16ctx.sycl.cpp`
(`attention_f16ctx_sycl`), a shape-named sibling of `attention_sycl`. Identical
flash / online-softmax math and per-query register layout, but the epilogue
writes the output twice from one `acc[j]*inv`: `O` in dtype dt and `O_f16` in
f16. Shape: online-attention + fused f16 context store.

Current public route: `ops::attention_f16ctx(...)` (dispatch/attention.cpp),
new operation in the attention family (co-exists with `attention`, does not
replace it). New op because the extra `O_f16` output is a signature change --
same precedent as `fused_add_rms_norm` vs `rms_norm`.

References inspected: embeddinggemma.c `src/engine_xpu.cpp`
`EI_XPU_FUSE_ATTN_OUTPUT_HALF` / `fuse_attn_output_half` -- its online-attention
epilogue emits a half copy of the context so the attention-output GEMM reads
`half_input` directly, skipping a standalone `float_to_half(ctx)` submission.
This kernel is the decoupled-library form of that fusion.

Environment: Arc Pro B60 (Battlemage G21), Level Zero V2 driver 1.14.37020,
oneAPI icpx, `--preset sycl`, working tree at HEAD 4eb01b1, device 0.
gnome-shell shares the GPUs (noisy) -> best-of-N with interleaved fused/unfused.

Correctness: fp64 host oracle in `tests/xpu_ops_smoke.cpp`
(`check_attention_f16ctx`), umbrella per-dtype tolerances. Checks BOTH outputs:
the dtype-dt O against the SDPA oracle (unchanged compute path) and the fused
O_f16 against the f16 rounding of the same result. f32/f16/bf16, MHA + GQA
(16/4) + cross-attn (sq=96, sk=160). All cases `worst_excess=0 worst16=0`. Base
`attention` op unchanged (still `worst_excess=0`). Full smoke suite: PASS.

Baseline / A/B: harness `--kernel attention_f16ctx --approx {fused,unfused}`.
`unfused` = `attention_sycl` (writes O) + a standalone O->O_f16 convert kernel
(the pass the fusion folds away), timed as the sum of both device events;
`fused` = the single `attention_f16ctx_sycl` event. Interleaved best-of-7
medians (warmup 12, iters 50), f16, device 0:

| shape (heads x seq, d=64) | unfused ms | fused ms | fused faster |
|---|---:|---:|---:|
| 32 x 128 (decode-ish) | 1.70823 | **1.69198** | **0.95%** |
| 16 x 512 | 6.39812 | **6.36437** | **0.53%** |
| 8 x 1024 | 12.5369 | **12.4283** | **0.87%** |

Decision: **keep**. Correctness-equivalent and never slower than the two-kernel
baseline (consistently ~0.5-1% lower device time across shapes), while
structurally removing a full O-sized (2*|O|) read+write convert pass and its
launch. The absolute gain is small here only because the current attention
variant is the correctness-first one-work-item-per-query shape (compute-bound),
so the eliminated convert is a small fraction of total; the relative payoff
grows once the tiled/subgroup throughput attention variant lands (attention
cheaper -> convert a larger share), which is the regime the fusion targets in
embeddinggemma (fast subgroup online-attention + free f16 store).

Open questions: fold into the deferred tiled/joint_matrix attention variant when
it lands, and expose an `attention` `Variant::best` route that picks the fused
store when the caller passes an O_f16 sink.

Raw results: interleaved best-of-7 A/B above; machine-specific JSON not archived
(git-ignored perf/results/).

## 2026-07-22: pool_mean_rms_l2 (sentence-embedding pooling head)

Status: landed.

Current implementation: `kernels/serving/variants/xpu_sycl/serving.sycl.cpp`
(`pool_mean_rms_l2_sycl`), a new op in the serving family alongside
embedding_lookup / kv_cache. One subgroup owns each sequence's dim-vector; each
lane holds DIM/16 pooled accumulators in registers (DIM is a compile-time shape
key, dispatched over {256,512,768,1024}). The kernel fuses three passes: a
per-token RMSNorm with the learned weight, a masked mean over the sequence's
token range (CSR `offsets[batch+1]`), and a final L2 normalize -- reading the
token embeddings once and writing one vector per sequence. Two subgroup
reductions (one per token for the RMS scale, one at the end for the L2 sum);
fp32 accumulation; empty sequence -> zero vector. Shape: masked-mean +
per-token RMSNorm + L2 pooling head.

Current public route: `ops::pool_mean_rms_l2(...)` (dispatch/serving.cpp), new
operation in the serving family. New op (not a variant) -- distinct signature
(token embeddings + CSR offsets + weight -> one vector per sequence), the
pooling family did not exist in QuixiCore-XPU before this.

References inspected: embeddinggemma.c `src/engine_xpu.cpp` `launch_pool`
(the plain subgroup masked-mean -> per-token RMS -> L2 variant; the
cooperative/workgroup variant was not ported) and the CPU reference
`src/kernels.c` `ei_mean_pool_rms_l2` -> `ei_rms_norm_inplace` /
`ei_norm_scale` / `ei_l2_normalize`. Math matched exactly: per-token
`inv = rsqrt(mean(x^2) + eps)`, accumulate `x*weight*inv`, `*= 1/count`, then
`rsqrt(sum(m^2))` (with the reference's `sum==0 -> 1` L2 guard). The learned
RMSNorm weight is applied directly (`x*inv*w`) exactly as both references do;
the Gemma `(1+w)` convention is folded into the weight upstream at load, not in
the kernel.

Environment: Arc Pro B60 (Battlemage), Level Zero V2 driver 1.14.37020, oneAPI
DPC++/C++ 2026.1.0, `--preset sycl`, working tree at HEAD 3253bac, device 0.
gnome-shell shares the GPUs (noisy) -> best-of-7 with interleaved fused/unfused.

Correctness: fp64 host oracle in `tests/xpu_ops_smoke.cpp`
(`check_pool_mean_rms_l2`), umbrella per-dtype tolerances with rtol widened by
sqrt(dim) for the two-level reduction (same convention as the attention /
linear-attn oracles). Ragged token counts {1,5,16,37,64,80,3} (incl. a
single-token sequence) exercise the masked mean; swept over f32/f16/bf16 x
dim {256,512,768,1024} = 12 cases. All `worst_excess=0`; max_abs f32 <=4.5e-8,
f16 <=1.5e-5, bf16 <=3.8e-6. Full smoke suite: PASS.

Baseline / A/B: harness `--kernel pool_mean_rms_l2 --approx {fused,unfused}`.
`unfused` = the naive two-pass decomposition -- `rms_norm_sycl` over all
[total_tokens, dim] token rows into a scratch buffer, then a masked-mean+L2 pass
reading scratch -- timed as the sum of both device events. `fused` = the single
`pool_mean_rms_l2_sycl` event. Same shapes; the delta is the eliminated
[total_tokens, dim] scratch write+read (2x the token-embedding traffic) and a
kernel launch. Interleaved best-of-7 medians (warmup 12, iters 50), device 0:

| dtype | batch x tokens x dim | unfused ms | fused ms | fused faster |
|---|---|---:|---:|---:|
| f32  | 64 x 64 x 768  | 0.6673 | **0.3218** | **2.07x** |
| f32  | 32 x 256 x 768 | 1.0064 | **0.6810** | **1.48x** |
| f32  | 64 x 64 x 1024 | 0.7617 | **0.3500** | **2.18x** |
| f16  | 64 x 64 x 768  | 0.6003 | **0.3401** | **1.76x** |
| bf16 | 64 x 64 x 768  | 0.6776 | **0.3324** | **2.04x** |

Decision: **keep**. Correctness-equivalent and 1.48-2.18x faster than the naive
two-pass decomposition across dtypes/shapes. Short sequences (tokens=64) pay the
scratch round-trip as a larger share of the work, so the fusion wins ~2x;
longer sequences (tokens=256) amortize it into the growing mean reduction, so
the win narrows to ~1.5x -- consistent with the memory-traffic model (unfused
moves ~3x total*dim vs the fused ~1x total*dim + batch*dim).

Open questions: a cooperative/workgroup variant (as in engine_xpu.cpp) could
help very large `dim` or tiny `batch` where one-subgroup-per-sequence
under-fills the device; not needed at the measured serving shapes. A
`Variant::best` heuristic is trivial today (single variant).

Raw results: interleaved best-of-7 medians above; machine-specific JSON not
archived (git-ignored perf/results/).

## 2026-07-22: rms_residual_next (fused residual-add + double RMSNorm -> f16)

Status: landed.

Current implementation: `kernels/norms/rms_norm/variants/xpu_sycl/rms_norm.sycl.cpp`
(`rms_residual_next_sycl`), a new op in the norms family alongside rms_norm /
fused_add_rms_norm. One work-group per row; two group reductions -- RMS of the
sublayer output `projection` (for its post-norm scale), then RMS of the updated
residual (for the next layer's pre-norm). The kernel fuses four passes: the
sublayer post-norm (`projection * post_weight * pinv`), the residual add (writes
the residual stream in place), the next layer's pre-norm
(`residual * next_weight * rinv`), and the f16 convert of that pre-norm into
`next_out`. The residual store uses the fp32 sum while its square feeds the
second reduction unrounded; the pre-norm re-reads the storage-dtype residual and
converts to f16, exactly as the reference does. Shape: residual-add + double
RMSNorm -> f16.

Current public route: `ops::rms_residual_next(...)` (dispatch/norms.cpp). New
operation in the norms family -- distinct signature (two weights + an in-place
residual + an f16 sink) extending fused_add_rms_norm (single add+norm) to the
transformer layer boundary's post-norm/pre-norm pair, same precedent as
fused_add_rms_norm vs rms_norm.

References inspected: embeddinggemma.c `src/engine_xpu.cpp`
`launch_rms_residual_next_half` (the clean subgroup-per-row form; the
register-cache and cooperative-workgroup variants were NOT ported -- their
self-parity ceiling ~0.999997 is documented in the source, and the clean form
that re-reads the residual from global is more accurate) and the CPU reference
`src/kernels.c` `ei_rms_norm_residual_inplace` / `ei_rms_norm` / `ei_norm_scale`.
Math matched exactly: `pinv = rsqrt(mean(projection^2) + eps)`, then
`residual += projection * post_weight * pinv`, then
`rinv = rsqrt(mean(residual^2) + eps)`, then `next_out = f16(residual *
next_weight * rinv)`. The learned RMSNorm weights are applied directly (Gemma's
`(1+w)` convention is folded upstream at load, not in the kernel).

Environment: Arc Pro B60 (Battlemage), Level Zero V2 driver 1.14.37020, oneAPI
DPC++/C++ 2026.1.0, `--preset sycl`, base HEAD b4b177e, device 0. gnome-shell
shares the GPUs (noisy) -> interleaved best-of-7 with fused/unfused alternated.

Correctness: fp64 host oracle in `tests/xpu_ops_smoke.cpp`
(`check_rms_residual_next`), checking BOTH the in-place residual (dtype dt) and
the f16 `next_out`; the two-level reduction widens rtol by sqrt(dim) (same
convention as the pool / attention oracles). Swept f32/f16/bf16 x dim
{256,768,1024} = 9 cases. All `worst_res=0` and `worst_out=0`; max_abs f32/f16
<=4.9e-4, bf16 <=6.1e-5. Full smoke suite: PASS.

Baseline / A/B: harness `--kernel rms_residual_next --approx {fused,unfused}`.
`unfused` = the composed separate-ops decomposition --
`rms_norm(projection,post_weight)`->scratch, `residual += scratch`,
`rms_norm(residual,next_weight)`->scratch2, `convert scratch2->f16` -- timed as
the sum of the four device events. `fused` = the single `rms_residual_next_sycl`
event. Same shapes; the delta is ~2 eliminated scratch round-trips and 3 kernel
launches -- the launch cut being the point on this submission-bound backend.
Interleaved best-of-7 medians (warmup 12, iters 50), device 0:

| dtype | rows x dim | unfused us | fused us | fused faster |
|---|---|---:|---:|---:|
| f32  | 64 x 768   | 7.290  | **4.063** | **1.79x** |
| f32  | 512 x 768  | 24.062 | **13.958**| **1.72x** |
| f32  | 64 x 1024  | 7.915  | **4.688** | **1.69x** |
| f16  | 64 x 768   | 7.603  | **4.167** | **1.83x** |
| bf16 | 64 x 768   | 7.604  | **4.063** | **1.87x** |

Decision: **keep**. Correctness-equivalent and 1.69-1.87x faster than the
four-launch decomposition across dtypes/shapes. The win is dominated by cutting
3 launches/instance (submission-bound) plus eliminating two [rows,dim] scratch
round-trips; it holds at both small (64) and larger (512) row counts.

Open questions: the register-cache / cooperative-workgroup variants from the
engine_xpu.cpp reference could help very large `dim` or tiny `rows`, at the cost
of the documented ~0.999997 self-parity ceiling; not needed at the measured
serving shapes. `next_out` is always f16 (the fused-convert purpose); a
dtype-generic sink is a trivial future extension.

Raw results: interleaved best-of-7 medians above; machine-specific JSON not
archived (git-ignored perf/results/).

## 2026-07-22: qk_norm_rope (fused per-head QK-norm + query-scale + RoPE)

Status: landed.

Current implementation:
`kernels/attention/qk_norm_rope/variants/xpu_sycl/qk_norm_rope.sycl.cpp`
(`qk_norm_rope_sycl`), a new op in the attention family alongside attention /
rope. One subgroup owns each (token, head); the subgroup reduces sum(x^2) over
head_dim (fp32) for the RMS scale, then each lane rotates its strided NeoX
half-split pairs (i, i+head_dim/2). For every (token, head) of Q
([tokens,n_head,head_dim]) and K ([tokens,n_head_kv,head_dim], GQA via
n_head_kv<n_head): RMS-normalize by the learned weight, fold the query scale
into the inverse-RMS for query heads (keys use 1), apply weight*inv before the
rotation, and optionally also write an f16 copy (Q_f16/K_f16, null to skip). The
RoPE angle is computed on the fly from `base` and a contiguous position
(pos0+token), matching rope.sycl.cpp exactly so the unfused baseline composes
the shipped rms_norm + rope. Shape: per-head RMSNorm + query-scale + RoPE.

Current public route: `ops::qk_norm_rope(...)` (dispatch/attention.cpp). New
operation in the attention family -- signature couples in-place Q/K, two learned
weights, a query scale, and optional f16 sinks; it did not exist before.

References inspected: embeddinggemma.c `src/engine_xpu.cpp`
`launch_qk_norm_rope` (the per-head QK-norm + query-scale + RoPE shape; the
richer `launch_qkv_epilogue`, which additionally encodes the model's fused-QKV
memory layout, was intentionally NOT ported -- it is more model-specific) and
the CPU reference `src/kernels.c` `ei_qk_norm_rope_qk_inplace`. Math matched
exactly: `inv = rsqrt(mean(x^2) + eps)`, query heads `*= query_scale`, then
`x0 = x[i]*w[i]*inv`, `x1 = x[i+half]*w[i+half]*inv`,
`out[i] = x0*cos - x1*sin`, `out[i+half] = x0*sin + x1*cos` with
`angle = pos * base^(-2i/head_dim)`. The reference applies the query scale after
the rotation; folding it into inv before rotation is identical (rotation is
linear).

Environment: Arc Pro B60 (Battlemage), Level Zero V2 driver 1.14.37020, oneAPI
DPC++/C++ 2026.1.0, `--preset sycl`, base HEAD 459fbe0, device 0. gnome-shell
shares the GPUs (noisy) -> interleaved best-of-7 fused/unfused alternated.

Correctness: fp64 host oracle in `tests/xpu_ops_smoke.cpp`
(`check_qk_norm_rope`), checking the in-place dt output of BOTH Q and K and, when
enabled, the f16 sinks; the head_dim reduction widens rtol by sqrt(head_dim)
(pool/attention oracle convention). Cases: MHA (8/8, hd 64), GQA (16/4 hd 128;
32/8 hd 128), with and without the f16 sink, across f32/f16/bf16. All `worst=0`
and `worst16=0`; max_abs f32 <=9.1e-7, f16 <=9.8e-4, bf16 <=3.9e-3. Full smoke
suite: PASS.

Baseline / A/B: harness `--kernel qk_norm_rope --approx {fused,unfused}`.
`unfused` = the composed separate-ops decomposition -- `rms_norm(Q,q_weight)`,
scale Q by query_scale, `rope(Q)`, `rms_norm(K,k_weight)`, `rope(K)` -- timed as
the sum of the five device events. `fused` = the single `qk_norm_rope_sycl`
event. Shapes: tokens=rows, n_head 32, n_head_kv 8 (GQA), head_dim 128. The
delta is 4 eliminated kernel launches -- the whole point on this
submission-bound backend. Interleaved best-of-7 medians (warmup 12, iters 50),
device 0:

| dtype | tokens | unfused us | fused us | fused faster |
|---|---|---:|---:|---:|
| f32  | 32  | 19.062 | **5.104**  | **3.74x** |
| f32  | 128 | 58.750 | **17.187** | **3.42x** |
| f16  | 32  | 21.145 | **5.104**  | **4.14x** |
| bf16 | 32  | 21.563 | **5.000**  | **4.31x** |

Decision: **keep**. Correctness-equivalent and 3.4-4.3x faster than the
five-launch decomposition -- the largest fusion win of the three ported kernels,
consistent with the submission-bound cost model (5 launches -> 1). The advantage
narrows slightly as tokens grow (more per-launch compute to amortize) but stays
>3.4x at 128 tokens.

Open questions: the f16 sink was benchmarked off (the launch-cut is measured on
the core dt path); with the sink on, the fused kernel additionally subsumes two
convert passes the decomposition would need. A cooperative multi-subgroup form
(as in the engine_xpu.cpp reference) could raise occupancy at very small token
counts; not needed at the measured shapes.

Raw results: interleaved best-of-7 medians above; machine-specific JSON not
archived (git-ignored perf/results/).

## 2026-07-22: glu_gelu_f16 (GEGLU with fused f16 output)

Status: landed.

Current implementation:
`kernels/activations/glu/variants/xpu_sycl/glu.sycl.cpp` (`glu_gelu_f16_sycl`),
a new op in the activations family alongside glu. One work-item per output
element (a [rows, d] range, no flat-id div/mod); reads the two dt input halves
(gate half then value half, glu's layout), computes `gelu_tanh(gate) * value` in
fp32, and stores f16. The gate uses the tanh GELU approximation, matching the
embeddinggemma.c source -- deliberately NOT `detail::geluf` (the erf GELU that
glu mode=geglu uses). Shape: GEGLU -> f16.

Current public route: `ops::glu_gelu_f16(...)` (dispatch/activations.cpp). New
operation in the activations family -- the f16-context variant of glu (the f16
output is a signature change, same precedent as attention_f16ctx vs attention);
the base glu op is unchanged.

References inspected: embeddinggemma.c `src/engine_xpu.cpp`
`launch_up_gate_gelu_half` (GELU-tanh gate x up with a fused f32->f16 convert;
the plain launch_gelu_mul is the dt-out sibling) and the CPU reference
`src/kernels.c` `ei_gelu_tanh` / `ei_gelu_mul_inplace`. Math matched exactly:
`0.5*x*(1 + tanh(0.7978845608 * x * (1 + 0.044715*x*x)))` times the value half.
The source packs the combined buffer as [up, gate]; this op follows glu's [gate,
value] half order instead -- the product gelu(gate)*value is identical, and it
keeps glu_gelu_f16 a true sibling of glu (same input layout, f16 out).

Environment: Arc Pro B60 (Battlemage), Level Zero V2 driver 1.14.37020, oneAPI
DPC++/C++ 2026.1.0, `--preset sycl`, base HEAD 703fed4, device 0. gnome-shell
shares the GPUs (noisy) -> interleaved best-of-7 fused/unfused alternated.

Correctness: fp64 host oracle in `tests/xpu_ops_smoke.cpp`
(`check_glu_gelu_f16`); pure elementwise (no reduction), so `report<half_t>`
with the f16 tolerance suffices -- the oracle computes `gelu_tanh(gate)*value`
in fp64 and report rounds to f16. Cases: dt-in {f32,f16,bf16} at d=1024 plus a
d=1000 (scalar-tail) f32 case. All ok; max_abs f16/f32-in <=3.1e-5, bf16-in
<=4.9e-4. Full smoke suite: PASS.

Baseline / A/B: harness `--kernel glu_gelu_f16 --approx {fused,unfused}`.
`unfused` = the composed baseline -- a tanh-gelu GLU writing dt into scratch
(same math as the fused kernel, so the A/B is correctness-equivalent; the
library glu mode=geglu could NOT serve as the baseline because it is erf-gelu),
then a dt->f16 convert -- timed as the sum of both device events. `fused` = the
single `glu_gelu_f16_sycl` event. The delta is the eliminated [rows,d] scratch
round-trip and a launch. Interleaved best-of-7 medians (warmup 12, iters 50),
device 0:

| dtype | rows x d | unfused us | fused us | fused faster |
|---|---|---:|---:|---:|
| f32  | 64 x 1152  | 4.583  | **2.916**  | **1.57x** |
| f32  | 512 x 1152 | 24.896 | **16.875** | **1.48x** |
| f16  | 64 x 1152  | 4.791  | **2.916**  | **1.64x** |
| bf16 | 64 x 1152  | 4.687  | **2.812**  | **1.67x** |

Decision: **keep**. Correctness-equivalent and 1.48-1.67x faster than the
glu + standalone convert two-pass across dtypes/shapes -- the smallest but still
solid of the three fusions (only one launch and one scratch round-trip are
folded away). The win narrows slightly as rows grow (the scratch traffic is a
smaller share of a larger activation), consistent with the memory-traffic model.

Open questions: the elementwise form leaves a vectorized 16-byte-load variant
(as in glu_typed / vec_map) on the table; not pursued because glu_gelu_f16 is
memory-bound and already saturates at the measured shapes, and the launch cut is
the point on this submission-bound backend.

Raw results: interleaved best-of-7 medians above; machine-specific JSON not
archived (git-ignored perf/results/).
