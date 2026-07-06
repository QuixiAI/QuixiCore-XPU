# Repository Structure

QuixiCore-XPU should share the same contract-facing structure as the other
QuixiCore backends while keeping Intel GPU implementation choices native to
SYCL, Level Zero, XeTLA, oneDNN, Triton XPU backend experiments, and XMX tuning.

The rule is: public taxonomy is common; implementation details are native XPU.

## Target Layout

```text
QuixiCore-XPU/
  .quixicore/
    backend.yaml
    kernels.yaml
    quant-formats.yaml

  .reference/

  docs/
    repository-structure.md
    development.md
    kernel-roadmap.md
    performance.md
    backend-notes.md

  include/quixicore/xpu/
    backend.hpp
    runtime.hpp
    ops.hpp

  src/
    runtime/
    dispatch/
    errors/

  kernels/
    common/
    norms/
    activations/
    attention/
    linear_attention/
    ssm/
    matmul/
    quantization/
    moe/
    sampling/
    serving/
    optimizers/
    collectives/
    utils/

  bindings/
    c/
    python/
    pytorch/

  tests/
    correctness/
    integration/
    smoke/
    testdata/

  perf/
    harness/
    configs/
    results/
    baselines/

  examples/
  scripts/
  tools/
  assets/
```

XPU-specific additions:

```text
CMakeLists.txt
CMakePresets.json
cmake/
.reference/
```

`.reference/` is for local upstream research clones and must not be treated as
backend source. New XPU kernels should not copy another backend's directory
layout; they should start in the target structure above.

## Manifests

`.quixicore/backend.yaml` identifies this repository as the XPU backend and
declares supported Intel GPU targets and contract compatibility.

`.quixicore/kernels.yaml` should be the machine-readable parity source for
implemented operations:

```yaml
operations:
  dense_gemm:
    family: matmul
    status: planned
    path: kernels/matmul/dense_gemm
    bindings:
      pytorch: bindings/pytorch/dense_gemm.cpp
    tests:
      correctness: tests/correctness/matmul/dense_gemm
    benchmarks:
      default: perf/configs/matmul_dense_gemm.yaml
    variants:
      - name: xpu_sycl
        status: planned
      - name: xpu_xetla
        status: planned
      - name: xpu_onednn
        status: planned
```

`.quixicore/quant-formats.yaml` should list supported quant formats, packing
layouts, and XPU-only layout constraints.

## Kernel Families

The top-level directories under `kernels/` are semantic families, not compiler
or reference-project buckets:

- `norms/`: RMSNorm, LayerNorm, add-norm, norm-to-quant, QK norm.
- `activations/`: GELU, GLU, SiLU/SwiGLU helpers, standalone softmax.
- `attention/`: flash attention, causal/non-causal/varlen attention, backward,
  paged attention, MLA, rotary, quantized-KV attention, state merging.
- `linear_attention/`: Based, Hedgehog, linear attention, causal/decay linear
  attention, GDN, complex linear attention primitives.
- `ssm/`: Mamba, SSD, selective scan, FFT convolution.
- `matmul/`: dense GEMM, staged GEMM, complex matmul, Flux, XeTLA/oneDNN-backed
  or custom SYCL GEMM.
- `quantization/`: act quant, runtime quant, qgemm, qgemv, quantized LM head,
  fp8/int8/fp4 packing, TurboQuant.
- `moe/`: routing, expert alignment, gather/scatter, grouped GEMM, quantized
  MoE GEMM, LoRA alignment, finalize.
- `sampling/`: sampling, logit transforms, penalties, rejection sampling, beam
  search, speculative decode and EAGLE helpers.
- `serving/`: KV cache mutation, block/page tables, indexers, MInference, cache
  copy/gather helpers.
- `optimizers/`: AdamW and other training optimizer kernels.
- `collectives/`: oneCCL/Level Zero collective and fused collective kernels
  when meaningful on Intel GPU platforms.
- `utils/`: bit packing, column permutation, Hadamard/FWHT, small reusable
  user-visible utilities.

## Operation Layout

Prefer one directory per operation:

```text
kernels/<family>/<operation>/
  README.md
  include/
  src/
  variants/
    xpu_sycl/
    xpu_xetla/
    xpu_onednn/
    xpu_triton/
    xpu_level_zero/
  tests/
  bench/
```

For small operations, direct `.cpp`, `.hpp`, or `.sycl.cpp` files under the
family are acceptable until there is more than one implementation.

## Tests And Benchmarks

Correctness and performance assets should mirror the kernel taxonomy:

```text
tests/correctness/<family>/<operation>/
perf/configs/<family>_<operation>.yaml
perf/baselines/<family>/<operation>/
```

Common developer entrypoints should exist:

```text
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
scripts/clean
```

For XPU these scripts should wrap CMake presets, SYCL compiler setup, Level Zero
device checks, and benchmark harnesses.

## Rules For New Work

- Add new kernels under semantic family directories.
- Keep PyTorch, Python, and extension glue in `bindings/`.
- Keep SYCL, XeTLA, oneDNN, Triton XPU, Level Zero, and architecture-specific
  choices under operation variants.
- Use `collectives/` for oneCCL or Level Zero collective extensions; mark them
  capability gated in `.quixicore/kernels.yaml`.
- If an operation has no meaningful XPU implementation yet, mark it planned or
  unsupported in metadata rather than adding a stub kernel.
