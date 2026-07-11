# Qwen Serving Kernel Port

This port moves the reusable Intel XPU kernels developed during Qwen3.6 serving
optimization into QuixiCore-XPU. The implementation is native to this repository
and exposes framework-neutral pointer ABIs in `include/quixicore/xpu/ops.hpp`.

## Ported Surface

- NVFP4 GEMM for `[M,K] x [N,K/2]`, including the decode-optimized GEMV path.
- Fused and split NVFP4 top-k MoE for Qwen expert layouts.
- FP8 E4M3/E5M2 W8A16 GEMM with per-tensor or per-channel scales.
- Qwen GDN decode with convolution and recurrent-state updates.
- Fused residual-add RMSNorm with exact in-place residual semantics.
- Current-stream SYCL command graph capture and replay.
- PyTorch XPU zero-copy wrappers and parity coverage for the serving surface.
- Level Zero-preferred Intel GPU enumeration, avoiding duplicate OpenCL aliases
  so dual-B60 device indices match tensor-parallel ranks.

The public header documents input layouts, scale layouts, dtype contracts, and
state mutation. `.quixicore/kernels.yaml` records operation maturity and source,
test, and benchmark locations.

GDN state indices in `[0, slots)` are active, including slot zero, and must be
unique within a decode batch. Negative or out-of-range indices skip state
mutation and return a zero core result while preserving the projected `z`
output.

## Integration Boundary

The vLLM adapter remains responsible for scheduler metadata, tensor-parallel
rank orchestration, FP8 KV-cache scale placement, and framework tensor lifetime.
Command graphs capture device-local compute only; XCCL collectives execute
between captured segments so tensor-parallel ordering remains explicit.

## Validation

Correctness lives in `tests/xpu_ops_smoke.cpp`, `tests/xpu_graph_smoke.cpp`, and
`bindings/pytorch/test_parity.py`. The focused B60 benchmark matrix is
`perf/configs/serving_qwen_decode.yaml`; measurements and routing decisions are
recorded in `perf/optimization_status.md`.
