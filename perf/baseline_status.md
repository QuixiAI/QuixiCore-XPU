# QuixiCore XPU Baseline Status

Method and measurement policy are described in `perf/perf.md`. Raw results live
under `perf/results/` and are ignored by git.

## Environment

Date: 2026-07-05.

Current local host:

- macOS development machine.
- C++ compiler: AppleClang through the default CMake `dev` preset.
- oneAPI compiler: not found locally (`icpx` not on `PATH`).
- Intel GPU runtime: not validated on this host.

## Scaffold Baseline

The current baseline is build/test health only. No XPU kernels exist yet.

Verified locally:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/quixicore_xpu_info
python3 perf/bench_kernels.py --phase all --preset dev --run-id scaffold-dev-check
```

Result:

- Configure: passed.
- Build: passed.
- Tests: 1/1 passed.
- Backend info executable: printed backend `xpu`, contract `v0.1`, status
  `planned`, targets `intel_arc`, `intel_data_center_gpu`, `future_xpu`.
- Scaffold harness: configure/build/test/info passed.

Not run locally:

```bash
cmake --preset sycl
cmake --build --preset sycl
ctest --preset sycl
```

Blocker: `icpx` is not available on this host.

## Per-Kernel Status

| Kernel Family | XPU Status | Runtime Blocker |
|---|---|---|
| RMSNorm / LayerNorm | planned | no implementation |
| Softmax | planned | no implementation |
| GELU / GLU | planned | no implementation |
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
