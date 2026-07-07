# Development

QuixiCore XPU is the native Intel GPU backend for the QuixiCore contract.

The backend should use Intel-native tooling and runtime APIs:

- oneAPI DPC++ / `icpx`
- SYCL kernels
- Level Zero runtime integration where direct runtime control is needed
- Intel GPU profiling and timing tools

It should not import implementation code from QuixiCore CUDA, QuixiCore Metal, or
the umbrella QuixiCore repository.

## Build

The default build compiles the metadata/status library and smoke tests with any
C++20 compiler:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

SYCL targets are opt-in so the repository can be configured on hosts without
oneAPI installed. On an Intel GPU host, first put `icpx` and the runtime on the
environment, then use the `sycl` preset:

```bash
source /opt/intel/oneapi/setvars.sh      # puts icpx + Level Zero on PATH
sycl-ls                                   # confirm the Intel GPU(s) enumerate
cmake --preset sycl
cmake --build --preset sycl
ctest --preset sycl                       # includes the on-device correctness gate
```

The `sycl` preset builds the native op library (`quixicore_xpu_ops`) plus the
on-device correctness test and benchmark. If oneDNN is found, the vendor
variants are compiled too (`-DQUIXICORE_XPU_ENABLE_ONEDNN=ON`, default); if
absent, `Variant::vendor` transparently falls back to the SYCL variant.

Ahead-of-time compilation for a specific Intel GPU is optional. For Battlemage:

```bash
cmake --preset sycl -DQUIXICORE_XPU_SYCL_TARGETS=bmg
```

### Implementation variants

Each op ships two co-equal implementations, selectable at the ABI via
`quixicore::xpu::Variant`:

- `Variant::sycl` — native hand-written SYCL (the QuixiCore "native kernel").
- `Variant::vendor` — Intel vendor library path (oneDNN today).

Both are correctness-checked and benchmarked; `Variant::best` will route to the
empirically faster path per op/dtype once the perf table is populated. This
mirrors how the Metal backend ships co-equal PyTorch and MLX bindings.

## First Kernel Workflow

Follow the pattern established by the reference op `activations/gelu`:

1. Add the native SYCL variant under
   `kernels/<family>/<operation>/variants/xpu_sycl/<op>.sycl.cpp`, exposing an
   event-returning `kernels::<op>_sycl(...)` entry declared in a per-op internal
   header (`kernels/<family>/<operation>/<op>_kernel.hpp`).
2. Add the vendor variant under `.../variants/xpu_onednn/<op>.onednn.cpp`
   (guarded by `QUIXICORE_XPU_HAS_ONEDNN`), exposing `kernels::<op>_onednn(...)`.
3. Add the public op factory to `include/quixicore/xpu/ops.hpp` and route it in
   `src/dispatch/<family>.cpp` (switch on `Variant`, apply blocking).
4. Add correctness coverage to the on-device gate (`tests/xpu_ops_smoke.cpp`),
   comparing against an fp64 host reference rounded to the storage dtype.
5. Add benchmark coverage: a `perf/configs/<family>_<operation>.yaml` and a case
   in the harness; run `python3 perf/bench_kernels.py --phase kernels --preset sycl`.
6. Run one focused optimization pass and record it in
   `perf/optimization_status.md` (required by the performance gate).
7. Only then update `.quixicore/kernels.yaml` and the umbrella matrices to claim
   support.

New SYCL translation units are picked up automatically by naming them
`*.sycl.cpp` / `*.onednn.cpp` under `kernels/` (the CMake glob uses
`CONFIGURE_DEPENDS`).

The first implementation target should be a deterministic, low-surface kernel
family such as RMSNorm/LayerNorm or Softmax.
