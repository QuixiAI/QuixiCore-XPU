# QuixiCore XPU Performance Handbook

The operating guide for baselining and optimizing every kernel under `kernels/`
on Intel GPUs (oneAPI / SYCL / Level Zero, Arc Pro B60 / Battlemage). It is not a
bag of tricks. For each kernel we run one disciplined loop: name the bottleneck,
measure a clean baseline on the real device, run controlled A/Bs, keep only
verified wins, and record enough that the next pass starts from evidence.

Two companion files:

- `perf/optimization_status.md` — the running log. One dated entry per kernel /
  pass with baseline → after numbers and an honest keep/reject verdict.
- `perf/baseline_status.md` — the environment snapshot and the current
  per-family roofline table.

Raw per-run JSON lands under `perf/results/` (git-ignored; machine-specific).

## The prime directive: smash truisms with evidence

This is the rule that overrides habit. **Never accept a hardware or library
"truism" without an on-B60 A/B.** In this backend the same assumption flips by
hardware, by dtype, and by shape. Things that turned out false when measured:

- *"oneDNN scale/shift must be f32."* False — the explicit
  `scale_shift_data_type` overload accepts bf16/f16.
- *"the vendor library is always faster."* False per-dtype: native SYCL beats
  oneDNN 1.58x on f32 layernorm; oneDNN beats native 1.70x on bf16 layernorm.
  There is no universal winner — so we ship **both** variants and route
  `Variant::best` per `(op, dtype)` from measured data.
- *"argmax is reduction-bound."* It was **serial-tail-bound** — the parallel
  scan was fine; one thread's 256-iteration final reduction dominated (fix: SLM
  tree reduction, 80 → 447 GB/s).
- *"rope is transcendental-bound."* Substantially **integer div/mod-bound** from
  decomposing a flat id; a 3D `nd_range` beat `pow→exp2` + `sincos` combined.
- *"these quant formats are NVIDIA/CPU-only"* (mxfp4/nvfp4/fp8/GGUF k+i-quants).
  False — they are data encodings; all decode natively on Intel.
- *"fp8 is not accelerated on B60"* — our own 2026-07-06 verdict, mostly false
  once each regime was measured with the right tool: e5m2 GEMM hits 85 TFLOP/s
  (95% of XMX peak) via `fpmath_mode::f16` up-convert + primitive caching, and
  the M=1 decode GEMV runs 310 GB/s natively (was 15.7 on the vendor route).

Name the assumed bottleneck, then confirm it with numbers/profiling before
optimizing — it is often not what you assumed. Fix the ruler first (profiling
events, warmup, adaptive batching), then re-verify a gap on the corrected
harness. Copying NVIDIA/Apple machinery blindly is a known trap: there is no
`cp.async`/TMA analogue on Xe2, but the B60 *has* XMX/DPAS — use `joint_matrix`
where Metal had to emulate.

## The B60 roofline (what "good" means)

Single Arc Pro B60 (Battlemage G21, 160 XVEs, subgroup sizes 16/32):

- **~456 GB/s** memory bandwidth — the ceiling for memory-bound kernels
  (elementwise, norms, softmax, sampling, quant-GEMV weight streaming).
- **~90 TFLOP/s** bf16 GEMM (via oneDNN XMX).
- **182 TOPS** int8 XMX (w8a8).

A memory-bound kernel is "done" near 85–90% of 456 GB/s. Our vectorized
elementwise/row kernels sit at 82–87%; that is the bar. Report every kernel as a
percentage of the relevant ceiling, not as a raw number.

## Measurement harness

Device timing is done in C++ with **SYCL queue-profiling events** (command
start→end timestamps), which measure device-side execution spans and exclude the
host wall-clock submit/sync floor that produced false regressions in the Metal
harness's early timing. A single-launch event isolates that kernel. A batched
multi-launch span can also contain device idle gaps between its boundary events;
use the timeline workflow below to determine whether those gaps come from host
submission, synchronization, or the kernels themselves.

One-off A/B (the common case during an optimization pass):

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build --preset sycl --target quixicore_xpu_bench
./build-sycl/quixicore_xpu_bench --kernel argmax --dtype bf16 \
    --rows 8192 --dim 8192 --iters 50 --warmup 15
# -> one JSON line: {"kernel":...,"variant":...,"median_ms":...,"gbps":...,"device":...}
```

`--variant {sycl,vendor,best}` selects the implementation; elementwise kernels
take `--n`, row/quant kernels `--rows/--dim`, GEMM `--M/--N/--K`, GGUF
`--approx <format>`. Add a new op by extending the early-return block in
`perf/harness/xpu_bench.cpp` with its buffers + `emit(median, metric)`.

Tracked sweeps (dated, logged) go through the orchestrator, which runs the C++
bench once per entry in a config and writes `perf/results/<date>/<run-id>/`:

```bash
python3 perf/bench_kernels.py --phase kernels --preset sycl   # reads perf/configs/*.yaml
```

The orchestrator sweeps the **full shipped kernel matrix** by default
(`DEFAULT_KERNEL_BENCH` in `bench_kernels.py`, one broad run per kernel, no
third-party dep). A `perf/configs/<family>_<op>.yaml` lists a richer `runs:`
sweep (variant × dtype × shape) and *overrides* the default entry for the kernels
it covers — so a run always covers everything, with detail where configured. Both
co-equal variants are benched on the same hardware so the routing data is honest.
Add a new op's buffers + `emit()` to `perf/harness/xpu_bench.cpp`, then a line to
the default matrix (and optionally a config for a deeper sweep).

## Bottleneck profiling workflow

Profiling is a funnel, not a substitute for the benchmark. First establish an
uninstrumented event-timed baseline, then use an execution timeline to isolate
the expensive kernel or launch pattern, and only then collect focused hardware
counters. Profiler instrumentation perturbs execution, so never use a VTune
duration as the baseline or as evidence for a performance claim. Validate every
candidate with the normal event-timed harness.

### Tooling on the B60 host

As checked on 2026-07-10, the host has VTune 2026.3 and `xpu-smi`. VTune exposes
`xpu-offload` and `gpu-hotspots`; the older `gpu-offload` collection is
deprecated. `unitrace` is not currently installed, so profiling instructions
must not depend on it. Re-check the environment before a run:

```bash
source /opt/intel/oneapi/setvars.sh
vtune -collect-list
vtune -help collect xpu-offload
vtune -help collect gpu-hotspots
sycl-ls
xpu-smi discovery -j
```

This machine currently exposes two Arc Pro B60 GPUs. The benchmark selects one
with `--device`; VTune selects it with `-knob target-gpu=<domain:bus:device.function>`.
For example, benchmark device 0 was PCI BDF `0000:6c:00.0`, represented by VTune
as `0:108:0.0`. Always verify that mapping rather than assuming PCI enumeration
is stable.

### Stage 1: uninstrumented baseline and roofline classification

Run the exact public route, dtype/format, and priority shapes with the normal
benchmark. Record median and min/max, effective bandwidth or FLOP/s, and the
percentage of the appropriate B60 ceiling. Compare all relevant native, vendor,
fused, and split variants before selecting a profile target.

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build --preset sycl --target quixicore_xpu_bench

./build-sycl/quixicore_xpu_bench \
    --kernel nvfp4_moe --variant sycl --dtype bf16 --approx split \
    --M 4 --N 256 --K 2048 --rows 8 --dim 512 \
    --device 0 --warmup 15 --iters 50
```

Use bytes moved, FLOPs, achieved throughput, and variance to form a one-sentence
bottleneck hypothesis. A memory-bound kernel near 388–410 GB/s is already at
85–90% of the measured 456 GB/s B60 ceiling; further gains require removing
traffic rather than rearranging arithmetic.

### Stage 2: XPU Offload timeline

Use `xpu-offload` to correlate host work, SYCL/Level Zero submissions, GPU
queues, transfers, synchronization, and device execution. This answers whether
the workload is host/launch-bound or device-bound and identifies the computing
task to inspect next. Intel's current description and knobs are in the
[VTune XPU Offload documentation](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-1/xpu-offload-analysis.html).

Use fewer iterations than the baseline to control trace size, but retain enough
repetition to distinguish steady state from initialization:

```bash
TRACE="perf/results/$(date +%F_%H%M%S)-nvfp4-moe-offload"

vtune -collect xpu-offload \
    -knob target-gpu=0:108:0.0 \
    -knob collect-programming-api=true \
    -knob enable-tasks-stack-collection=true \
    -result-dir "$TRACE" -- \
    ./build-sycl/quixicore_xpu_bench \
        --kernel nvfp4_moe --variant sycl --dtype bf16 --approx split \
        --M 4 --N 256 --K 2048 --rows 8 --dim 512 \
        --device 0 --warmup 3 --iters 10

vtune-gui "$TRACE"
```

Inspect the timeline for:

- an empty GPU queue or gaps between small kernels, indicating host submission
  or launch overhead;
- repeated allocation, primitive construction, queue waits, or synchronization;
- unexpected host/device copies or materialized intermediate tensors;
- serialized work that could overlap; and
- one subkernel dominating a fused or multi-launch operation.

### Stage 3: focused GPU hardware counters

After the timeline names the expensive task, use `gpu-hotspots` to inspect that
kernel's occupancy, XVE active/stalled/idle behavior, memory and cache activity,
SIMD utilization, and instruction mix. Intel recommends this as the focused
follow-up to offload analysis; see the
[GPU Compute/Media Hotspots documentation](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-1/gpu-compute-media-hotspots-analysis.html).

Start with the overview metric set:

```bash
HOT="perf/results/$(date +%F_%H%M%S)-nvfp4-moe-hotspots"

vtune -collect gpu-hotspots \
    -knob target-gpu=0:108:0.0 \
    -knob gpu-profiling-mode=characterization \
    -knob characterization-mode=overview \
    -result-dir "$HOT" -- \
    ./build-sycl/quixicore_xpu_bench \
        --kernel nvfp4_moe --variant sycl --dtype bf16 --approx split \
        --M 4 --N 256 --K 2048 --rows 8 --dim 512 \
        --device 0 --warmup 3 --iters 10

vtune -report hotspots -group-by=computing-task -r "$HOT"
vtune-gui "$HOT"
```

Then rerun only the counter group needed to test the hypothesis:

- `characterization-mode=global-memory-accesses` for streaming, cache,
  coalescing, staging, or quant-decode questions;
- `characterization-mode=compute-extended` for ALU/XMX utilization and
  occupancy;
- `characterization-mode=instruction-count` for integer indexing, format
  decode, branching, or transcendental work; and
- `characterization-mode=full-compute` only when smaller focused passes do not
  classify the issue.

Source-analysis and stall-sampling support varies by GPU. Do not assume a mode
is supported on B60 merely because it appears in `vtune -help`; start with
characterization and follow the collector's supported-mode guidance.

### Reading the evidence

| Observation | Likely bottleneck | First controlled A/B |
| --- | --- | --- |
| GPU queue repeatedly empty | Launch/runtime overhead | Fuse kernels, cache primitives, remove waits |
| Bandwidth near 388–410 GB/s | At the memory roofline | Reduce bytes moved or eliminate staging |
| Low bandwidth and low occupancy | Grid, work-group, subgroup, or register pressure | Sweep one launch/tile factor at a time |
| High memory stalls and poor cache behavior | Irregular or noncoalesced access | Change layout or vectorize adjacent accesses |
| High instruction count with low bandwidth | Decode, indexing, or scalar ALU work | Remove div/mod, hoist branches, simplify decode |
| Low SIMD utilization | Divergent control flow | Specialize dtype, format, or edge paths |
| Several intermediate kernels or copies | Decomposition overhead | Fuse or dequantize directly into compute |

Register pressure deserves explicit attention in fused MoE, attention, and
decode kernels. It can reduce SIMD width/occupancy or spill private state into
memory. See Intel's
[register-pressure guidance](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/registers-and-performance.html),
then test smaller tiles, less unrolling, shorter live ranges, or SLM staging one
factor at a time.

### Source and assembly attribution

The normal `sycl` preset is the source of truth for benchmark numbers. Its
current RelWithDebInfo compile line uses `-O2 -g`. For reliable VTune mapping
from GPU instructions back to source, create a separate profiling build/preset
that preserves optimization and adds `-gline-tables-only` and
`-fdebug-info-for-profiling`, as required by Intel's
[current DPC++ profiling guidance](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html).
Do not replace the production timing preset with profiler-specific flags without
first confirming identical generated-code performance.

### Profiling record

Store result directories under `perf/results/`; do not commit large VTune
results or profiler traces. In `perf/optimization_status.md`, record the kernel,
route, dtype/format, shapes, BDF/device, compiler/runtime/driver, exact command,
trace path, observed bottleneck, and the controlled A/B it motivated. After the
change, pass correctness and rerun the uninstrumented event benchmark. The final
entry must include baseline/current/candidate timing and a keep/reject decision.

## Correctness gates (a win is not a win until these pass)

- **fp64 host oracle** — `ctest --preset sycl` runs `tests/xpu_ops_smoke.cpp`:
  each op vs an fp64 reference rounded to storage dtype (+1 storage ULP for
  transcendentals), across `{f32,f16,bf16}` × `{sycl,vendor}`. Quant/GGUF ops
  check against an independent host replica of the reference decode.
- **torch.xpu parity** — `bindings/pytorch/test_parity.py` compares `tk_xpu.<op>`
  against PyTorch's own XPU kernels (a second, independent oracle).

Exactness is part of the contract for packing/quantization kernels; numeric
tolerance is per-dtype for the rest. A change that regresses either gate is not a
candidate, no matter the speedup.

## The optimization loop (per kernel)

1. **Inventory** — entry points, dtypes, shape constraints, existing tests/bench.
2. **Reference** — read the real sources (below), record exact paths in
   `optimization_status.md`. Translate ideas; never import source.
3. **Baseline** — measure on realistic shapes with the profiling-event harness;
   compute % of the relevant ceiling. Compare native SYCL vs the vendor path.
4. **Classify the bottleneck** — bytes moved vs FLOPs vs achieved throughput vs
   variance. State the hypothesis in one sentence.
5. **One factor at a time** — define the A/B before editing.
6. **Test then bench** — `ctest --preset sycl` first, then the bench matrix.
7. **Decide with numbers** — keep, reject, or defer; record the rejected
   alternative too.
8. **Log it** — append to `optimization_status.md`.

## Reference sources (read, don't import)

We keep no `.reference/` mirror; the sources live outside the repo and are
consulted, not vendored:

- **`~/llama.cpp`** (`ggml/src/ggml-common.h`, `ggml-quants.c`) — the
  authoritative GGUF block layouts and dequant algorithms (q*_K, iq*_ grids,
  `get_scale_min_k4`, `ksigns_iq2xs`). Ported to SYCL, validated vs an
  independent host replica.
- **`~/vllm/csrc/quantization/marlin`** — int4 group-GEMV structure and the
  reminder that its `cp.async` pipeline has *no* Intel analogue.
- **`../QuixiCore-Metal`** — the op contract (which ops exist, byte layouts) and
  its anti-staging reversals. Approach only, never source.
- **oneDNN / oneMKL examples + docs** — the vendor primitive APIs (matmul with
  s4/u4/fp8/fp4 weights + grouped scales, `rms_norm` flag, SDPA via Graph API).
- **Intel oneAPI DPC++ & Level Zero docs** — subgroups, `joint_matrix`, USM,
  profiling.

Classify each idea as portable-algorithm / Intel-specific-mechanism /
compiler-runtime-lesson / benchmark-shape.

## The proven technique catalogue (measured on B60)

These are the moves that have actually paid off here. Reach for them by
bottleneck; each has a live example in `optimization_status.md`.

### Memory-bound (elementwise, norms, sampling, quant streaming)

- **16-byte coalesced `sycl::vec` loads** (`kernels/common/vec_map.hpp`,
  `vec_width<T>()` = 8 for 2-byte, 4 for 4-byte). This is *the* recurring win —
  ~2x on bf16/f16 memory-bound kernels. Apply by default; `sycl::vec<bfloat16,8>`
  works. Verify adjacent lanes hit adjacent addresses.
- **Subgroup-per-row / work-group-per-row** for reductions (norms, softmax,
  argmax, cross-entropy, categorical sampling): strided coalesced reads +
  `reduce_over_group`, `[[sycl::reqd_sub_group_size(32)]]`. Never one work-item
  per row when rows ≪ device width (categorical sampling: 52x from this alone).
- **SLM tree reduction** over a serial final loop by one thread (argmax: 4.9x).
- **`exclusive_scan_over_group`** for ordered prefix/CDF work (inverse-CDF
  categorical selection over contiguous per-thread chunks).

### Compute / scalar-side-work

- **Kill integer div/mod in hot loops** — use a 2D/3D `nd_range` so indices come
  from `get_global_id` dimensions instead of decomposing a flat id (rope: the
  dominant factor).
- **Cheaper transcendentals** — `pow(base, x)` → `exp2(x·kc)` with `kc` folded
  host-side (one often-native op vs exp+log); `cos`+`sin` → one `sincos`.
- **fp32 accumulation** for norms/softmax/attention/long-K reductions; cast once
  at the end (casting an intermediate scale to bf16 over-quantizes).
- Hoist dtype/causal/format/dimension branches out of inner loops via templates.

### Matrix / quant

- **XMX `joint_matrix`** for GEMM/attention where the shape and compiler expose
  it (the native path's current gap vs the 90 TFLOP/s oneDNN route — task #9).
- **oneDNN SYCL interop** —
  `dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())`, USM memory,
  try/catch → native fallback. Ship it as the co-equal `vendor` variant.
- **Dequant-direct-to-compute** — decode a quant block straight into the dot
  product / matrix op; don't materialize a dense tensor when the value is used
  once. All quant formats are in scope (see `CLAUDE.md` corollary).

## Shape strategy

Do not optimize only square toy shapes. Per family, cover small/non-pow-2 edges,
tile-aligned fast paths, tile-ragged shapes, real LLM projections
(`K=4096,N=11008/14336`), and stress shapes (long context, large K/N, batch
sweeps, many experts). Starter sets:

- Row/elementwise: rows `{4096,16384,65536}`, hidden `{64…8192}`.
- GEMM: square `{1024,2048,4096,8192}` + LLM rectangles.
- Quant GEMV/GEMM: `M∈{1,2,4,8,16,32,64,128}`, `N/K∈{4096,8192,16384}`.
- Attention: `D∈{64,128}`, context `{512,2048,8192}`.
- MoE: tokens `{128,1024,4096}`, experts `{8,16,64}`, top-k `{1,2,4}`.

Record skipped shapes with the reason. Re-run surprising results on an idle
machine before trusting an A/B.

## Decision rules

A change is a candidate winner when: focused correctness passes; median improves
≥3% for low-risk local changes or ≥8–10% for changes that add real complexity;
required shapes don't regress; and there is a bytes/FLOPs/profiling explanation.

Reject or defer when: the win is inside noise; it appears only on toy shapes;
complexity rises without a durable real-shape win; it depends on unavailable
hardware/driver features; or it changes the numeric contract.

## Recording format (`optimization_status.md`)

Each entry: status (baselining / experimenting / candidate / **landed** /
deferred); the current implementation + public route; references inspected (exact
paths); the correctness command + last result; a baseline→after table with % of
ceiling; the decision log (including rejected alternatives); open questions. Do
not commit large profiler traces — record their local path, device, and summary.

## The performance gate (hard rule)

No backend kernel implementation, routing change, benchmark change, or
performance claim is committed without at least one focused on-B60 optimization
run recorded in `optimization_status.md`. If the XPU runtime is unavailable, do
not commit a claimed win — report the blocker, or limit the change to
docs/scaffolding with no performance claim. (See `AGENTS.md` and both `CLAUDE.md`
files.)
