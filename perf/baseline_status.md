# QuixiCore XPU Baseline Status

Method and measurement policy are in `perf/perf.md`. The running experiment log
is `perf/optimization_status.md`. Raw per-run JSON lives under `perf/results/`
(git-ignored).

## Environment

Date: 2026-07-07.

Standing baseline host:

- **4x Intel Arc Pro B60 Graphics** (Battlemage G21, 160 XVEs each, subgroup
  sizes 16/32, **~456 GB/s** per-GPU memory bandwidth). Kernel runs are
  single-device unless noted; collectives use all 4.
- Roofline ceilings: ~456 GB/s memory, **~90 TFLOP/s** bf16 GEMM (oneDNN XMX),
  **182 TOPS** int8 XMX.
- **oneAPI DPC++/C++ 2026.0.0** (`icpx`), via `source /opt/intel/oneapi/setvars.sh`.
- SYCL runtime: Unified Runtime over Level-Zero V2 (OpenCL also present), driver
  1.14.37020; all 4 GPUs enumerate via `sycl-ls`.
- Vendor libraries: oneDNN 3.11, oneCCL.
- **`.venv`**: `torch==2.14.0.dev+xpu`, numpy, triton-xpu, oneMKL. torch bundles
  its own SYCL runtime (identical `20260331` build to system `icpx`). The
  PyTorch-XPU binding (`bindings/pytorch/tk_xpu`) is built and parity-validated.

## Build + gate (the standing invariant)

```bash
# non-SYCL scaffold builds on any C++20 host:
cmake --preset dev   && cmake --build --preset dev   && ctest --preset dev    # 1/1
# on the B60 host, after sourcing setvars:
cmake --preset sycl  && cmake --build --preset sycl  && ctest --preset sycl   # 3/3
```

`sycl` ctest = backend smoke + SYCL device probe (lists 4x B60) + the on-device
op correctness gate (`tests/xpu_ops_smoke.cpp`: every op × {f32,f16,bf16} ×
{sycl,vendor} vs an fp64 host oracle; quant/GGUF vs independent host replicas).

## Kernel roofline snapshot (2026-07-07, single B60)

Measured device-time medians (`quixicore_xpu_bench`, 50 iters / 15 warmup) at
8192×8192 (row/quant) or 16M elems (elementwise), bf16 unless noted. Elementwise
and row kernels are vectorized (16-byte `sycl::vec`); the roofline is 456 GB/s
for memory-bound ops, 90 TFLOP/s for bf16 GEMM. See `optimization_status.md` for
the per-kernel history and the 2026-07-07 optimization pass (before → after).

| kernel | metric | % of ceiling | notes |
|---|---|---|---|
| silu / glu / rms_norm / layernorm | 390–396 GB/s | 85–87% | vectorized row/elementwise |
| gelu / gelu_backward | 371–375 GB/s | 81–82% | |
| dropout | 400 GB/s | 88% | vectorized (was 144 / 32%) |
| argmax | 447 GB/s (f32) | 98% | SLM tree reduction (was 80 / 18%) |
| rope | 400 GB/s (f32) | 88% | 3D range + exp2/sincos (was 151 / 33%) |
| adamw | 339 GB/s | 74% | |
| softmax | 315 GB/s | 69% | 2-pass |
| dense_gemm (vendor) | ~90 TFLOP/s | ~roofline | oneDNN XMX; native path is task #9 |
| qgemm_int8 (vendor) | 182 TOPS | — | w8a8 XMX |
| sample_categorical | 0.87 ms | — | work-group-per-row (was 45.6 ms, 52x) |
| quantize_int4 | 121 GB/s | 27% | vectorized (was 43 / 9.5%) |
| qgemv_int4 / gguf / mxfp4 / nvfp4 | 60–116 GB/s weight-bw | 13–25% | decode coalescing = next pass |

## Per-family status

All 13 families are implemented natively (co-equal `sycl` + `vendor` where a
oneDNN/oneMKL primitive exists), each with fp64-oracle correctness and a bench.

| Family | Status | Notes |
|---|---|---|
| activations (gelu/silu/glu/softmax/+bwd) | implemented | sycl + oneDNN; vectorized |
| norms (rms_norm/layernorm) | implemented | sycl + oneDNN; best-routed per dtype |
| matmul (dense_gemm) | implemented | oneDNN XMX at 90 TFLOP/s; native XMX = task #9 |
| attention (flash SDPA, GQA, causal) | implemented | native online-softmax |
| rope | implemented | optimized (3D range) |
| optimizers (adamw) | implemented | |
| sampling (argmax/categorical/top_k) | implemented | argmax+categorical optimized |
| serving (kv_cache scatter/gather, embedding) | implemented | |
| moe (route_topk, grouped_gemm) | implemented | |
| ssm (selective_scan) | implemented | sequential recurrence |
| linear_attention | implemented | |
| utils (hadamard/cross_entropy/dropout) | implemented | dropout optimized |
| collectives (all_reduce) | implemented | native multi-GPU, capability-gated |
| quantization | implemented | int4/int8(182 TOPS)/fp8(e5m2 85 TFLOP/s + native M=1 GEMV)/mxfp4/nvfp4 + **16 GGUF formats**; act_quant + quantize_int4 |
| bindings (torch.xpu) | implemented | tk_xpu, parity-validated |

## Deferred (bigger projects, flagged not faked)

- **Native `dense_gemm` XMX `joint_matrix`** — 1.1 vs 90 TFLOP/s vendor (task #9).
  Vendor is routed as `best`; native is the co-equal loser until this lands.
- **Quant-GEMV decode coalescing** — qgemv_int4/gguf/mxfp4/nvfp4 at 13–25%
  weight-BW; the interleaved on-disk layouts fight coalescing.
- **Sequential scans** — selective_scan / linear_attn recurrences.

## Decision log

- 2026-07-05: Adopted the `perf.md` / `optimization_status.md` /
  `baseline_status.md` / `results/` layout; kept SYCL opt-in so the scaffold
  configures on non-oneAPI hosts.
- 2026-07-06: Validated the full SYCL pipeline on 4x Arc Pro B60 (oneAPI 2026.0,
  Level-Zero V2); refreshed off the stale macOS host. Shipped co-equal
  `Variant::{sycl,vendor}` with `best`-routing per (op, dtype) — no universal
  winner (native beats oneDNN on f32 layernorm, oneDNN wins on bf16).
- 2026-07-07: Built out all 13 families / ~40 ops; complete quant surface (all
  formats + 16 GGUF weight formats + act/weight quant); working torch.xpu binding
  (shared-lib + `.sycl` source, not a version skew). Optimization pass: 52x
  categorical, 4.9x argmax, 2.8x dropout, 2.8x quantize_int4, 1.9x rope — all
  measured, correctness-gated, recorded.
