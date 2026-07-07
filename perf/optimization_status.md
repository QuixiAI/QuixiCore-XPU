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

## 2026-07-07: Track B quant depth — GGUF q6_K k-quant

### quantization/gguf_gemv — q6_K (native k-quant decode)
First GGUF k-quant (256-element super-block, 210 bytes: ql[128]+qh[64]+int8
scales[16]+fp16 d; 6-bit quant = 4 low bits from ql + 2 high bits from qh,
recentred -32; per-16 sub-block int8 scales). Follows ggml dequantize_row_q6_K
exactly, one 32-wide subgroup per row. Correctness vs an INDEPENDENT host replica
of the ggml reference over random-byte blocks (no shared packer): worst_excess 0,
f32+bf16. Baseline 8192x8192 bf16 = 82 GB/s weight bandwidth. Proves the k-quant
super-block layout decodes natively on Intel.

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
