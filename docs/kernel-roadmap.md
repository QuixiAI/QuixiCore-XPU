# Kernel Roadmap

This file tracks the local XPU implementation plan. The canonical contract lives
in `QuixiAI/QuixiCore`; this repository should only claim support after native
implementation, correctness coverage, and benchmark coverage exist.

## Phase 0: Scaffold

- Backend compatibility metadata.
- CMake project and C++20 smoke tests.
- SYCL/oneAPI device probe behind an explicit build flag.
- Benchmark result location and reporting notes.

## Phase 1: Deterministic Row Kernels

- RMSNorm / LayerNorm.
- Softmax.
- GELU / GLU.

These are the best first XPU targets because their semantics are simple enough
to validate locally while still exercising Intel GPU memory bandwidth,
subgroup reductions, and dtype handling.

## Phase 2: Quantization Surface

- Quant format decode helpers.
- Quant GEMV.
- Quant GEMM.
- Quantized LM head.

Quant formats must match the umbrella format names and byte layouts at API
boundaries. Backend-local layouts are acceptable only as internal variants.

## Phase 3: Serving Kernels

- Causal attention.
- Paged attention.
- MLA decode.
- Sampling, beam search, and speculative decode.
- MoE routing and grouped MoE GEMM.
- Mamba / SSD.

## Current Contract Status

All kernel families are planned for XPU. None are currently claimed complete.
