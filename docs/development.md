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
oneAPI installed:

```bash
cmake -S . -B build-sycl \
  -DCMAKE_CXX_COMPILER=icpx \
  -DQUIXICORE_XPU_ENABLE_SYCL=ON
cmake --build build-sycl
ctest --test-dir build-sycl --output-on-failure
```

## First Kernel Workflow

1. Add the kernel contract notes to the relevant local docs.
2. Add a native SYCL implementation under `src/`.
3. Add public declarations under `include/quixicore/xpu/`.
4. Add correctness tests under `tests/` using deterministic host references.
5. Add benchmark coverage under `perf/`.
6. Only then update status metadata or umbrella matrices to claim support.

The first implementation target should be a deterministic, low-surface kernel
family such as RMSNorm/LayerNorm or Softmax.
