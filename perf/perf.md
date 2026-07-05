# QuixiCore XPU Performance Handbook

This is the operating guide for baselining and optimizing QuixiCore XPU kernels.
It follows the discipline used in QuixiCore Metal and QuixiCore CUDA, but the
hardware assumptions are Intel GPU, oneAPI, SYCL, and Level Zero specific.

The goal is not to collect tricks. For each kernel, find references, state a
bottleneck hypothesis, measure a clean baseline, run controlled experiments,
keep only verified wins, and record enough detail that the next pass starts from
evidence instead of memory.

The running notebook is `perf/optimization_status.md`. Baseline snapshots live
in `perf/baseline_status.md`. Raw results live under `perf/results/`.

## Principles

Correctness comes before performance. A change is not a win until it passes the
kernel's correctness tests, improves the target metric on realistic shapes, and
does not regress supported edge shapes or numeric tolerances.

Prefer experiments that attack a named bottleneck:

- Memory-bound: reduce bytes moved, improve coalescing, improve cache reuse,
  avoid extra global-memory passes, or use narrower formats.
- Compute-bound: raise arithmetic intensity, feed XMX/DPAS paths effectively,
  reduce scalar side work, and fuse epilogues.
- Latency-bound: grow resident work, reduce serial loops, batch tiny dispatches,
  and avoid unnecessary host/device synchronization.
- Occupancy-bound: tune work-group size, subgroup count, register pressure,
  local-memory use, and grid size so the GPU has enough resident work.
- Synchronization-bound: reduce barriers, local-memory traffic, and cross-work-
  group coordination; prefer subgroup reductions when they fit the operation.
- Launch-bound: fuse small operations or route them through a framework path if
  dispatch overhead dominates.

## XPU Baseline Assumptions

Do not blindly port CUDA or Metal machinery.

CUDA concepts such as `cp.async`, TMA, WGMMA, warp-specialized producer/consumer
pipelines, CUDA events, and Nsight counters have no one-to-one meaning in SYCL.
Metal concepts such as simdgroup matrix operations, threadgroup memory behavior,
and command-buffer timing are useful comparisons but not XPU design rules.

XPU kernels should be written in terms of Intel-native mechanisms:

- SYCL queues, events, USM or buffers, subgroups, and local memory.
- oneAPI DPC++ compiler behavior and generated device code.
- Level Zero runtime/profiling where direct timing, command-list control, or
  device inspection is needed.
- Intel XMX/DPAS or joint-matrix paths where the target hardware and compiler
  expose them.
- XeTLA and oneDNN examples as implementation references, not source to vendor.
- Triton XPU backend as a compiler/runtime reference for mapping high-level
  tensor kernels to Intel GPUs.

When a backend-local layout is needed for performance, preserve QuixiCore
contract names and byte layouts at public API boundaries.

## Repo Facts To Preserve

Current scaffold:

- Public headers: `include/quixicore/xpu/`.
- Native source: `src/`.
- Examples: `examples/`.
- Tests: `tests/`.
- Performance docs and harnesses: `perf/`.
- Local reference mirrors: `.reference/` (git-ignored).
- Backend metadata: `.quixicore/backend.yaml`.

Build and smoke-test commands:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

SYCL/oneAPI probe:

```bash
cmake --preset sycl
cmake --build --preset sycl
ctest --preset sycl
```

The `sycl` preset expects `icpx` on `PATH`.

## Reference Search Protocol

For each kernel, inspect references before designing the XPU implementation.
Record exact files in `perf/optimization_status.md`.

Reference roots currently present:

- `.reference/intel-xpu-backend-for-triton`
- `.reference/xetla`
- `.reference/oneDNN`
- `.reference/tiny-dpcpp-nn`

Useful search patterns:

```bash
rg -n "sub_group|joint_matrix|dpas|local_accessor|nd_range|event" .reference
rg -n "layernorm|rms_norm|softmax|gemm|attention|quant" .reference
rg -n "level_zero|zeEvent|profiling|queue" .reference
```

Classify reference ideas into:

- Portable algorithm idea: worth considering.
- Intel-specific mechanism: translate if it matches the target XPU generation.
- Compiler/runtime lesson: useful for build flags, profiling, and scheduling.
- Benchmark shape or oracle idea: usually worth adopting.

Do not import implementation code from references into this repository unless a
future license and provenance review explicitly allows it.

## Measurement Harness Requirements

Every benchmark result should include:

- Git commit or working-tree label.
- QuixiCore contract version.
- oneAPI compiler version and flags.
- SYCL backend/runtime and Level Zero driver version where available.
- Intel GPU model, tile/slice information where available, memory size, and
  driver/runtime versions.
- Kernel family, operation, public entry point, dtype, quant format, and shape.
- Warmup count, measured iteration count, median, p20/p80 or min/max, and
  coefficient of variation.
- Correctness tolerance and observed max absolute/relative error.
- Derived throughput: GB/s, GFLOP/s, TOP/s, tokens/s, or elements/s.
- Raw output path under `perf/results/`.

The scaffold harness is:

```bash
python3 perf/bench_kernels.py --phase all --preset dev
python3 perf/bench_kernels.py --phase all --preset sycl
```

As kernels are added, extend the harness with a registry of runnable cases. Each
case should self-skip with a reason when a compiler, device, format, or runtime
feature is unavailable.

## Timing Rules

Use native device timing for device work.

- For SYCL, enable queue profiling and read event start/end timestamps.
- For Level Zero, use events or metric tools appropriate to the experiment.
- Synchronize outside the measured region unless the cost being studied includes
  host synchronization.
- Warm up first to avoid JIT, module load, cache, and power-state artifacts.
- Do not allocate, initialize, or copy inputs in the measured region unless that
  is the metric under study.
- For tiny kernels, batch repeated launches per sample and divide, then record
  the batch size.
- Re-run surprising results on an idle machine before trusting an A/B.

Derived metrics:

```text
GEMM FLOPs          = 2 * M * N * K
attention FLOPs     ~= 4 * B * H * N * N * D   (halve for causal)
quant decode GB/s   = packed_weight_bytes_read / seconds / 1e9
row-kernel GB/s     = conservative required reads+writes / seconds / 1e9
```

State when an estimate ignores cache reuse, repeated passes, metadata reads, or
write allocation.

## Shape Strategy

Do not optimize only square toy shapes. For each family, cover:

- Small edge shapes and non-power-of-two dimensions.
- Tile-aligned fast-path shapes.
- Tile-ragged shapes.
- Real model shapes from Llama/Qwen/DeepSeek-style projections and attention.
- Stress shapes: long context, large K/N, batch sweeps, and many experts.

Starter shapes:

- Norm/softmax/GELU: rows in `{4096, 16384, 65536}`, hidden in
  `{64, 128, 256, 512, 768, 1024, 2048, 4096, 8192}`.
- GEMM: square `{1024, 2048, 4096, 8192}` and LLM rectangles such as
  `K=4096`, `N=11008`, `N=14336`.
- Quant GEMV/GEMM: `M in {1, 2, 4, 8, 16, 32, 64, 128}`,
  `N/K in {4096, 8192, 16384}`.
- Attention: `D in {64, 128}`, context in `{512, 2048, 8192}`.
- MoE: tokens in `{128, 1024, 4096}`, experts in `{8, 16, 64}`,
  top-k in `{1, 2, 4}`.

Record skipped shapes with the reason.

## Per-Kernel Optimization Loop

1. Inventory the kernel: entry points, dtypes, shape constraints, tests, and
   benchmark coverage.
2. Find references in `.reference/` and sibling backend docs.
3. Establish correctness against a deterministic host reference.
4. Measure the baseline against framework and naive decomposed baselines where
   available.
5. Classify the bottleneck with bytes, FLOPs, achieved throughput, variance, and
   profiling counters.
6. Define experiments before editing code. Change one meaningful factor at a
   time.
7. Execute focused tests first, then the same benchmark matrix.
8. Decide with recorded numbers and rejected alternatives.
9. Update `perf/optimization_status.md`.

## Experiment Catalogue

Use these as templates. Not every kernel needs every experiment.

### Launch Geometry

- Sweep work-group size and subgroup count.
- Change rows/items per work-group for row kernels.
- For GEMM/attention, sweep output tile sizes and K/sequence block sizes.
- Split large reductions across more work-groups only when merge overhead is
  measured.
- Watch tail effects when grid size does not fill the device evenly.

### Memory And Layout

- Verify adjacent lanes read adjacent addresses on hot paths.
- Compare USM and buffer paths only when the API contract allows either.
- Test local-memory staging against direct global loads.
- Use vectorized loads/stores where alignment and layout allow.
- Keep scales, metadata, and lookup tables in layouts that favor subgroup access
  patterns.

### XMX / Matrix Paths

- Use oneDNN, XeTLA, and Triton XPU as references for matrix instruction use.
- Compare joint-matrix or XeTLA paths against scalar/vector SYCL kernels.
- Separate alignment-specialized fast paths from generic edge paths.
- Record compiler flags and target architecture whenever a matrix path is used.

### Reductions And Numerics

- Prefer subgroup reductions before work-group reductions.
- Keep fp32 accumulation for norms, softmax, attention, and long K reductions
  unless a lower-precision variant passes tolerance.
- Use deterministic reduction orders where the contract needs determinism.
- For exact integer or packing kernels, exactness is part of the contract.

### Fusion And Routing

- Fuse epilogues such as bias, residual, scale, activation, or normalization
  when an intermediate would otherwise round-trip through device memory.
- Fuse dequantization with matmul or attention when the dequantized value is
  used once.
- Split a fused kernel when register pressure, branching, or lower occupancy
  dominates saved memory traffic.
- Find crossovers between custom kernels, oneDNN calls, Triton-generated code,
  and framework primitives.

## First Kernel Targets

Start with deterministic row kernels before serving kernels:

- `rms_norm` / `layernorm`: reduction plus pointwise write, good for subgroup
  and memory-bandwidth calibration.
- `softmax`: numerically sensitive row reduction, good for profiling
  synchronization and exponent cost.
- `gelu` / `glu`: pointwise/fused epilogue calibration.

Only claim a kernel family complete after native implementation, correctness
coverage, benchmark coverage, and contract status updates.

## Decision Rules

A change is a candidate winner when:

- Focused correctness tests pass.
- Median performance improves by at least 3% for low-risk local changes, or
  8-10% for changes that add meaningful complexity.
- Required correctness shapes do not regress.
- Secondary performance shapes do not regress beyond an agreed tolerance.
- There is a plausible explanation backed by bytes, FLOPs, profiling, or a clean
  A/B.

Reject or defer when:

- The win is inside measurement noise.
- The win appears only on toy shapes.
- Complexity rises without a durable real-shape win.
- The optimization depends on unavailable hardware or driver features.
- The numeric contract changes.

## Recording Format

Each section in `perf/optimization_status.md` should contain:

- Status: not started, baselining, experimenting, candidate, landed, deferred.
- Current implementation and public route.
- References inspected, with exact paths.
- Correctness command and last result.
- Baseline table.
- Experiment table.
- Decision log.
- Open questions.

Do not commit large profiler traces. Record their local path, device, and summary.

## External References

- Intel oneAPI DPC++ documentation: compiler, SYCL, queues, profiling, USM,
  subgroups, and matrix extensions.
- Intel Level Zero documentation: device discovery, command queues/lists,
  events, metrics, and memory.
- Intel XeTLA: XMX-oriented tiling and GEMM examples.
- oneDNN: production Intel GPU primitive implementations and benchmark shapes.
- Intel XPU backend for Triton: lowering and runtime behavior for Triton kernels
  on Intel GPUs.
- tiny-dpcpp-nn: compact DPC++ examples and build conventions.
