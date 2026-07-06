# Contributing To QuixiCore XPU

QuixiCore XPU is one native backend in the QuixiCore family. Contributions
should preserve the shared QuixiCore contract while using Intel GPU-native
implementation techniques.

## Backend Boundary

- Implementation code belongs in this repository, not in the QuixiCore umbrella
  repository and not in another backend.
- Shared semantics belong in QuixiAI/QuixiCore.
- XPU-specific tuning, SYCL, Level Zero, XeTLA, oneDNN, Triton XPU experiments,
  and oneCCL workflows belong in this repository.

## Adding Or Changing A Kernel

1. Put source under `kernels/<family>/<operation>/`.
2. Put implementation-specific choices under `variants/xpu_sycl/`,
   `variants/xpu_xetla/`, `variants/xpu_onednn/`, or the appropriate variant.
3. Add correctness coverage under `tests/correctness/<family>/<operation>/`.
4. Add benchmark coverage under `perf/`.
5. Update `.quixicore/kernels.yaml`.
6. Update `.quixicore/quant-formats.yaml` when quant layouts, packing, or
   supported formats change.
7. Document backend-specific behavior in the operation README or relevant docs.

## Required Checks

Use the common entrypoints when possible:

```bash
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
```

The default scaffold path must keep working on non-XPU hosts. Hardware-specific
checks should be gated behind the SYCL preset or an explicit script option.

## Pull Request Checklist

- Kernel semantics match the QuixiCore contract or document an XPU-only
  extension.
- Correctness tests cover the changed behavior.
- Benchmarks cover the relevant shapes or explain why benchmark coverage is not
  applicable yet.
- `.quixicore/kernels.yaml` reflects implementation status.
- `.quixicore/quant-formats.yaml` reflects quant format support.
- New source follows `docs/repository-structure.md`.
