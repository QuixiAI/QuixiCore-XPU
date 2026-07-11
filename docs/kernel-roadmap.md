# Kernel Roadmap

This file tracks the local XPU implementation plan. The canonical contract lives
in `QuixiAI/QuixiCore`; this repository should only claim support after native
implementation, correctness coverage, and benchmark coverage exist.

## Phase 0: Backend Foundation

- Backend compatibility metadata.
- CMake project and C++20 smoke tests.
- SYCL/oneAPI device probe behind an explicit build flag.
- Benchmark result location and reporting notes.

## Phase 1: Deterministic Row Kernels

- RMSNorm / LayerNorm.
- Softmax.
- GELU / GLU.

Native RMSNorm, fused residual-add RMSNorm, LayerNorm, softmax, GELU, and GLU
variants are present. Additional shape and dtype tuning remains ongoing.

## Phase 2: Quantization Surface

- Quant format decode helpers.
- Quant GEMV.
- Quant GEMM.
- Quantized LM head.

INT4, INT8, GGUF, MXFP4, NVFP4, and FP8 paths are present at different maturity
levels. Quant formats must match the umbrella format names and byte layouts at
API boundaries. Backend-local layouts are acceptable only as internal variants.

## Phase 3: Serving Kernels

- Causal attention.
- Paged attention.
- MLA decode.
- Sampling, beam search, and speculative decode.
- MoE routing and grouped MoE GEMM.
- Mamba / SSD.

Attention, sampling, Mamba/SSD, quantized MoE, Qwen GDN decode, and command
graph capture now have native implementations. Remaining work is model breadth,
shape tuning, and framework integration rather than initial bring-up.

## Current Contract Status

The backend is active. Family and operation status is authoritative in
`.quixicore/kernels.yaml`; families remain partial while contract gaps exist.
