# QuixiCore XPU Baseline Status

Method and measurement policy are described in `perf/perf.md`. Raw results live
under `perf/results/` and are ignored by git.

## Environment

Date: 2026-07-06.

Current dev host (the standing XPU baseline environment):

- 4x Intel Arc Pro B60 Graphics (Battlemage G21, 160 XVEs each, subgroup sizes
  16/32, ~456 GB/s per-GPU memory bandwidth). Kernel runs are single-device
  unless noted.
- oneAPI DPC++/C++ Compiler 2026.0.0 (`icpx`), sourced via
  `/opt/intel/oneapi/setvars.sh`.
- SYCL runtime: Unified Runtime over Level-Zero V2 (also OpenCL). Driver
  1.14.37020. All 4 GPUs enumerate via `sycl-ls`.
- Vendor libraries present: oneDNN and oneCCL.

The earlier macOS/no-`icpx` baseline is superseded: the performance gate is now
satisfiable on real Intel GPU hardware.

## Baseline

The default (non-SYCL) `dev` build still configures/builds/tests on any C++20
host (scaffold invariant). The `sycl` preset builds the native ops + vendor
variants and runs on the B60s.

Verified on the B60 host (after `source /opt/intel/oneapi/setvars.sh`):

```bash
cmake --preset dev   && cmake --build --preset dev   && ctest --preset dev    # non-SYCL, 1/1
cmake --preset sycl  && cmake --build --preset sycl  && ctest --preset sycl   # SYCL, 3/3
python3 perf/bench_kernels.py --phase all --preset sycl
```

Result:

- `dev`: configure/build/test pass (backend status library only).
- `sycl`: 3/3 tests pass — backend smoke, SYCL device probe (lists 4x B60), and
  the on-device op correctness gate (GELU across {f32,f16,bf16} x {sycl,vendor}
  vs an fp64 host oracle, all within contract tolerance).
- Kernel benchmark (GELU, n=4,194,304, median device time from profiling
  events): SYCL f32 329 GB/s, vendor f32 410 GB/s; SYCL bf16 230 / f16 227 GB/s.

## Per-Kernel Status

| Kernel Family | XPU Status | Runtime Blocker |
|---|---|---|
| RMSNorm / LayerNorm | planned | no implementation |
| Softmax | planned | no implementation |
| GELU / GLU | partial | GELU landed (sycl + vendor); GLU + backward pending |
| Causal Attention | planned | no implementation |
| Paged Attention | planned | no implementation |
| MLA Decode | planned | no implementation |
| Quant GEMV | planned | no implementation |
| Quant GEMM | planned | no implementation |
| Quantized LM Head | planned | no implementation |
| Sampling | planned | no implementation |
| Beam Search | planned | no implementation |
| Speculative Decode | planned | no implementation |
| Mamba / SSD | planned | no implementation |
| MoE Routing | planned | no implementation |
| Grouped MoE GEMM | planned | no implementation |
| Optimizers | planned | no implementation |

## Decision Log

- 2026-07-05: Adopted the sibling-backend pattern of `perf/perf.md`,
  `perf/optimization_status.md`, `perf/baseline_status.md`, and
  `perf/results/`.
- 2026-07-05: Kept SYCL support opt-in so the project configures on non-oneAPI
  hosts while still exposing a oneAPI validation path.
- 2026-07-05: Chose deterministic row kernels as the first implementation
  target family.
- 2026-07-06: Validated the full SYCL pipeline on 4x Arc Pro B60 (oneAPI
  2026.0, Level-Zero V2). Refreshed the baseline environment off the stale
  macOS host.
- 2026-07-06: Shipped two co-equal implementation variants per op — native
  `Variant::sycl` and vendor `Variant::vendor` (oneDNN) — mirroring the Metal
  backend's co-equal framework bindings. GELU landed in both; native SYCL tuned
  via a coalesced grid stride (+21% f32 vs naive), vendor still fastest on f32.
