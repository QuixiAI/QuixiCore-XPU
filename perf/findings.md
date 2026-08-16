# XPU Established Findings — Do Not Re-Derive

Distilled from `perf/optimization_status.md` through 2026-07-22. Treat as
current truth until re-measured; every entry names its date and notebook
entry so it can be challenged with new data.

## Environment anchor

From `perf/baseline_status.md` (dated 2026-07-07): 4x Intel Arc Pro B60
Graphics (Battlemage G21, 160 XVEs each, subgroup sizes 16/32, ~456 GB/s
per-GPU memory bandwidth); roofline ceilings ~456 GB/s memory, ~90 TFLOP/s
bf16 GEMM (oneDNN XMX), 182 TOPS int8 XMX. oneAPI DPC++/C++ 2026.0.0 (`icpx`),
SYCL Unified Runtime over Level-Zero V2, driver 1.14.37020; oneDNN 3.11,
oneCCL; `.venv` torch 2.14.0.dev+xpu with parity-validated `tk_xpu` binding.
Kernel runs are single-device unless noted. Note: notebook entries from
2026-07-10 onward record oneAPI DPC++/C++ 2026.1.0 (20260617) on the same
driver — see Open contradictions.

## Wins

| finding | effect | date | notebook entry |
|---|---|---|---|
| VEC=4 coalesced grid-stride GELU (adjacent lanes, adjacent addresses) | f32 271→329, bf16 178→229, f16 178→227 GB/s | 2026-07-06 | GELU reference kernel (activations/gelu) — SYCL + oneDNN |
| 16-byte `sycl::vec` loads on 16-bit row kernels (`sycl::vec<bfloat16,8>` works) | rms_norm bf16 184→392 (2.1x), layernorm bf16 196→388 (2.0x), gelu bf16 230→406 (1.8x), softmax bf16 195→302 (1.5x) GB/s | 2026-07-06 | 16-byte vector-load pass |
| int4 qgemv: wide loads + in-register decode + one-subgroup-per-row | 0.549→0.257 ms, 61→130 GB/s; beats fp16 GEMV 1.18x | 2026-07-06 | quantization/qgemv int4 — Marlin/Metal-guided dequant GEMV opens the family |
| oneDNN int8 w8a8 XMX, per-token scale as binary post-op ([M,1] broadcast) | ~182 TOPS at 4096^3 (0.75 ms), 2x bf16 GEMM peak | 2026-07-06 | quantization/qgemm int8 w8a8 — oneDNN XMX hits 182 TOPS |
| sample_categorical: work-group-per-row + `exclusive_scan_over_group` CDF | 45.6→0.87 ms (52x) | 2026-07-07 | Optimization pass — profiled all kernels, fixed the biggest gaps |
| argmax: SLM tree reduction replacing one thread's serial 256-iter tail | 80→447 GB/s f32 (4.9x) | 2026-07-07 | Optimization pass — profiled all kernels, fixed the biggest gaps |
| dropout / quantize_int4 16-byte vectorization | 144→400 GB/s and 43→121 GB/s (2.8x each) | 2026-07-07 | Optimization pass — profiled all kernels, fixed the biggest gaps |
| rope: 3D nd_range (kills flat-id div/mod) + exp2 + sincos | 151→400 GB/s f32, 284 bf16 (1.9x) | 2026-07-07 | Optimization pass — profiled all kernels, fixed the biggest gaps |
| quant-GEMV decode-ALU fixes: hoist scale, vectorize activation read, independent accumulators | qgemv_int4 116→167, mxfp4 80.7→119, nvfp4 78.5→114 GB/s (1.44–1.48x) | 2026-07-07 | Optimization pass #2 — 4-bit quant-GEMV decode (truism busted) |
| fp8 native M=1 decode GEMV (bit-cast pair-decode, 32 K-slabs) | 4.09→0.22 ms, 15.7→310 GB/s (18.9x); LLM shapes 221–224 GB/s | 2026-07-08 | quantization/fp8_gemm — "fp8 is not accelerated on B60" mostly busted |
| oneDNN engine+primitive caching and `fpmath_mode::f16` for fp8 GEMM | e4m3 20.0→44.4 TFLOP/s (2.2x); e5m2 18.9→85.3 TFLOP/s (4.5x, 95% of bf16 XMX peak) | 2026-07-08 | quantization/fp8_gemm — "fp8 is not accelerated on B60" mostly busted |
| nvfp4 bit-relocation decode into the f16 grid (no LUT, no ldexp) | bf16 0.293→0.153 ms, 114.6→219.6 GB/s (1.92x, 48% roofline) | 2026-07-08 | quantization/nvfp4_gemv pass #3 — bit-relocation decode, 1.92x |
| fused NVFP4 MoE kernel + native fp8 W8A16 GEMV in vLLM serving | 14.1→18.9 tok/s (+34%; MoE fusion +22%, fp8 +7%) | 2026-07-08 | vLLM integration — native NVFP4 MoE + fp8 W8A16 GEMV |
| native SYCL submit vs torch dispatch | ~2.4 us/kernel flat vs ~23 us/op (~10x); graph replay ~1.3 us/kernel at R>=128 | 2026-07-09 | Path-to-60-tok/s deep dive (Phase 0 de-risk) |
| full XPU graph capture + split NVFP4 MoE | eager 18.4 → graph 36.9 → graph+split 68.07 tok/s | 2026-07-09 | vLLM full XPU graph + split NVFP4 MoE reaches 68 tok/s |
| segmented graph (eager-XCCL breaks) for TP2 and long context | TP2 64.81–65.22 tok/s; 128k-context 57.67–58.53 tok/s (eager 14.16/14.29) | 2026-07-09 | segmented XPU Graph enables TP2 decode |
| Qwen serving-kernel port: split MoE, fused add-RMSNorm, native fp8 M=1 | split MoE M1 3.89x (0.620→0.159 ms); add-RMSNorm 10.93x API wall; fp8 W8A16 M1 native +9.9% over oneDNN | 2026-07-10 | Port Qwen serving kernels back into QuixiCore-XPU |
| NVFP4 split-MoE output-row tiling (2/4/8/16 rows per subgroup) | 17.1–20.1% lower latency M1–M16; M1 157.6→193.7 GB/s | 2026-07-10 | NVFP4 split-MoE output-row tiling |
| NVFP4 MoE paired gate/up decode (one activation load feeds both rows) | 21.8–25.2% lower latency M1–M16; M1 →247.9 GB/s | 2026-07-11 | NVFP4 MoE paired gate/up decode |
| NVFP4 MoE packed-dot `sycl::vec<T,8>` activation loads | 1.1–3.3% lower latency M1–M16; M1 →256.4 GB/s | 2026-07-11 | NVFP4 MoE packed-dot vector loads |
| qk_norm_rope fusion (5 launches → 1) | 3.42–4.31x vs composed rms_norm+scale+rope | 2026-07-22 | qk_norm_rope (fused per-head QK-norm + query-scale + RoPE) |
| rms_residual_next fusion (4 launches → 1) | 1.69–1.87x vs the four-launch decomposition | 2026-07-22 | rms_residual_next (fused residual-add + double RMSNorm -> f16) |
| pool_mean_rms_l2 fused pooling head | 1.48–2.18x vs two-pass rms_norm+pool | 2026-07-22 | pool_mean_rms_l2 (sentence-embedding pooling head) |
| glu_gelu_f16 fused GEGLU→f16 (2 launches → 1) | 1.48–1.67x vs glu + convert | 2026-07-22 | glu_gelu_f16 (GEGLU with fused f16 output) |
| attention_f16ctx fused f16 context store | 0.53–0.95% lower device time; removes a full O-sized convert pass | 2026-07-22 | attention_f16ctx (fused f16 context store) |
| attn_swa symmetric banded attention (O(window) per query) | 2.94–11.27x vs dense at seq 1024–4096, window 256 | 2026-07-22 | attn_swa (symmetric sliding-window attention) |
| w4a16 int4-weight DPAS GEMM (native `joint_matrix`, no cutlass) | bit-exact; 1.53x at M=16 / 2.12x at M=32 vs per-row int4 GEMV (still 0.37–0.53x vs int8 SLM tile) | 2026-07-22 | w4a16_gemm (int4-weight x f16/bf16-activation DPAS GEMM) |

## Rejected — with the reason, so they are not retried

**Contiguous per-thread strips — REJECTED (2026-07-06).** VEC=4 with base =
tid*4 dropped GELU f32 271→196 GB/s: a contiguous strip per work-item breaks
subgroup coalescing. Generalized: adjacent work-items must touch adjacent
addresses; amortize via grid-stride, not per-thread runs. (GELU reference
kernel entry.)

**Unrolled scalar V-blocks — REJECTED (2026-07-06).** A scalar loop over V
contiguous elements regressed rms_norm f32 368→209, bf16 184→162 GB/s: the
compiler emits V strided scalar loads. Generalized: the compiler will not
vectorize a scalar unroll; use explicit `sycl::vec` with 16-byte width.
(16-byte vector-load pass entry.)

**kSlabs=64 in the fp8 M=1 GEMV — REJECTED (2026-07-08).** 211 vs 262 GB/s at
32 slabs: extra partial traffic + shorter slabs lose to the occupancy gain.
(quantization/fp8_gemm entry.)

**M-tiled nvfp4 GEMM at decode M — REJECTED (2026-07-08, re-confirmed
2026-07-10).** Decoding each weight row once and accumulating M partials loses
to the row loop for M<=6 (acc[8]+wv[32] register pressure kills the fused
decode-dot); 2026-07-10 harness: 0.294 vs 0.171 ms at M4 (1.72x slower). Only
wins at M>=8; kept unused for future prefill work. (vLLM integration entry;
Port Qwen serving kernels entry.)

**Extra launches for device-side wins under torch-eager serving — REJECTED
(2026-07-08).** The MoE occupancy 2-kernel split gained +34% device GB/s at
decode M=4 yet was serving-flat (18.85 vs 19.0 tok/s): the pipeline is
launch-bound, so +1 launch/layer cancels the device win. Generalized: under
eager serving, tok/s tracks launch count, not device time. NOTE the reversal:
under graph capture the same split is a KEEP (36.9→68 tok/s, 2026-07-09).
(vLLM integration follow-up run #2; vLLM full XPU graph entry.)

**int4 as the route to faster decode — REJECTED (2026-07-09).** At decode
M<=4 the MoE is a memory-bound GEMV; int4 and nvfp4 both read 4 bits/weight,
so identical weight-byte traffic and identical bandwidth. int4 only pays via a
w4a8 INT8-XMX path in compute-bound large-M regimes. (Path-to-60-tok/s deep
dive entry.)

**Inductor compile without cudagraph — REJECTED (2026-07-09).** 4.8 tok/s vs
18.56 eager (~4x slower): compile graph-breaks around every opaque custom op
and adds Dynamo/guard + Triton-XPU overhead with no offsetting fusion.
(Phase 1, Path-to-60-tok/s deep dive entry.)

**Routing eager calls through torch custom-op dispatch — REJECTED
(2026-07-09).** Always-through-custom-op eager serving measured 18.19 vs 18.75
tok/s (-3% dispatch tax). Keep `is_compiling()` wrappers: custom-op path only
under compile/capture. (Phase 1, Path-to-60-tok/s deep dive entry.)

**GDN as a torch-layer serving hook — REJECTED (2026-07-09).** The native GDN
core wins B=1 in isolation but routed through torch it replaces one production
op with two launches + Python dispatch: serving 18.71 vs 18.83 tok/s, and the
one 19.19 sample did not reproduce (noise). Belongs inside a native decode
engine, not as a torch-layer swap. (Phase 2, Path-to-60-tok/s deep dive
entry.)

**Broadcasting uniform GDN recurrence scalars via SLM — REJECTED
(2026-07-10).** Neutral at B1, up to ~2% slower at B4; the compiler already
handles uniform work and the SLM traffic does not pay. (NVFP4 split-MoE
output-row tiling entry, rejected GDN experiments.)

**One-launch fused GDN conv+recurrent kernel — REJECTED (2026-07-10).**
Correct, but only +1.8% B1 / +0.25% B4 — complexity exceeded the measured win.
(NVFP4 split-MoE output-row tiling entry, rejected GDN experiments.)

**32 output rows per subgroup in the split MoE — REJECTED (2026-07-10).**
0.07698 vs 0.06485 ms at 16 rows: only 64 output work-groups leaves the B60
under-occupied. Generalized: keep >=128 work-groups when picking tile sizes.
(NVFP4 split-MoE output-row tiling entry.)

**SLM hidden-vector staging for MoE gate/up — REJECTED (2026-07-11).**
Regressed M1 by 4.46% and M4 by 6.93%; the cache path already covers the
reuse. (NVFP4 MoE paired gate/up decode entry, Rejected factors.)

**Generic two-row gate tiling (unrelated rows) — REJECTED (2026-07-11).**
M1 -1.52%, M4 neutral: scheduling was not the bottleneck; pairing only pays
when the rows share an activation read (gate/up). (NVFP4 MoE paired gate/up
decode entry, Rejected factors.)

**Global activated scratch for the split MoE — REJECTED (2026-07-11).**
Computing SwiGLU once in gate/up and dropping output SLM staging regressed M1
by 11.8% and M4 by 11.9%. (NVFP4 MoE packed-dot vector loads entry, Rejected
factors.)

**Precomputed activation plus SLM staging — REJECTED (2026-07-11).** Neutral
at M1 and M4; extra contract/traffic with no measured payoff. (NVFP4 MoE
packed-dot vector loads entry, Rejected factors.)

**Forced two-way chunk unroll (`#pragma unroll 2`) — REJECTED (2026-07-11).**
M1 -3.6%, M4 -1.1%, consistent with extra live state. (NVFP4 MoE packed-dot
vector loads entry, Rejected factors.)

## Patterns and generalized rules

- There is no universal vendor-vs-native winner; ship both variants and route
  `Variant::best` per (op, dtype) from measured data (2026-07-06, norms entry;
  reproduced same day on softmax).
- Vendor-vs-native reversals are per-dtype AND unstable under kernel
  improvement: f32 layernorm native beat oneDNN 1.58x while bf16 layernorm
  oneDNN beat native 1.70x (2026-07-06, norms entry), then the vector-load
  pass overturned the bf16 call — final direction: layernorm best = sycl at
  every dtype; softmax f32 = sycl, bf16 = vendor (exp-bound); matmul/qgemm =
  vendor (2026-07-06, 16-byte vector-load pass). Do not treat even your own
  measured conclusion as permanent.
- fp8 verdict reversal, recorded once: 2026-07-06 said "works, NOT accelerated
  on B60" (16–19 TFLOP/s); overturned 2026-07-08 — e5m2 GEMM 85.3 TFLOP/s
  (95% of XMX peak) via primitive caching + `fpmath_mode::f16`, e4m3 at ~half
  (oneDNN's up-convert), and M=1 decode belongs on the native GEMV (310 GB/s).
  Final direction: fp8 IS fast on B60 with the right tool per regime
  (2026-07-08, quantization/fp8_gemm entry).
- "oneDNN layernorm scale/shift must be f32" is false: the explicit
  `scale_shift_data_type` overload accepts bf16/f16 with no fallback
  (2026-07-06, norms entry).
- The assumed bottleneck is usually wrong — name it, then profile: argmax was
  serial-tail-bound not reduction-bound; rope was int-div/mod-bound not
  transcendental-bound; "weight-memory-bound" quant GEMVs were
  decode-ALU-bound; "tanh-bound" GELU was access-width-bound (2026-07-06/07,
  GELU entry, Optimization pass, Optimization pass #2).
- 16-byte coalesced `sycl::vec` loads are the default memory-bound move (~2x
  on 16-bit dtypes); verify adjacent lanes hit adjacent addresses
  (2026-07-06, 16-byte vector-load pass).
- Formats that embed in the f16 grid (e5m2, e4m3, e2m1) decode fastest by
  bit-relocation with the constant factor folded host-side — no LUT, no
  ldexp, no sign select (2026-07-08, fp8_gemm and nvfp4_gemv pass #3 entries).
- Half the vendor-path cost can be per-call engine/primitive re-creation:
  cache per context and per (shape, kind, dtype, scale) (2026-07-08,
  quantization/fp8_gemm entry).
- Decode-GEMVs reread all weights once per row: route them only at M=1 (or
  smallest M) and use a GEMM for prefill/M>1 — the measured W8A16 crossover is
  native at M=1 only, oneDNN from M=2 up (2026-07-09, dual-B60 TP2 entry;
  2026-07-10, Port Qwen serving kernels entry).
- Under torch-eager serving, end-to-end tok/s tracks kernel-launch count, not
  device time (~23 us/op torch dispatch vs ~2.4 us native submit); graph
  capture inverts the calculus and flips launch-costly device wins from
  reject to keep (2026-07-08 follow-up run #2; 2026-07-09 Phase 0 and full
  XPU graph entries).
- Fusion wins on this submission-bound backend scale with launches
  eliminated: 5→1 = 3.4–4.3x, 4→1 = 1.7–1.9x, 2→1 = 1.5–1.7x (2026-07-22,
  qk_norm_rope / rms_residual_next / glu_gelu_f16 entries).
- All quant formats are data encodings that decode natively on Intel — all 16
  GGUF weight formats, mxfp4, nvfp4, fp8, int4/int8 — each verified against
  an independent host replica; "NVIDIA-only format" claims are empirically
  false here (2026-07-06/07 quantization entries).
- Interleaved on-disk quant layouts (GGUF 18–210-byte blocks) fight GPU
  coalescing; the fix is a one-time repack to scale-planar + aligned quants,
  not a smarter in-place decoder (2026-07-07, gguf_gemv entries).
- SYCL `joint_matrix` (XMX/DPAS) compiles and runs bit-exact on B60 under
  both AoT and JIT — the native tensor-engine path is open (2026-07-22,
  w4a16_gemm entry, feasibility probe).

## Open contradictions

- The two 2026-07-11 entries ("NVFP4 MoE paired gate/up decode",
  "NVFP4 MoE packed-dot vector loads") carry heading verdict REJECTED while
  their bodies record "Status: candidate — KEEP" with kept, measured wins
  (21.8–25.2% and 1.1–3.3%); only their "Rejected factors" subsections were
  rejected. Resolution: one confirming A/B at M1/M4 against the pre-pass
  baseline, then relabel the headings by re-read evidence (never by guessing).
- Device count: the baseline env block (2026-07-07) says all 4 B60s enumerate
  via sycl-ls, but the 2026-07-10 "Port Qwen serving kernels" entry reports
  torch and tk_xpu enumerating "the two physical B60 devices" after filtering
  OpenCL aliases. Resolution: a fresh `sycl-ls`/device-probe run on the
  standing host, then re-anchor the environment block.
- Toolchain drift: the baseline anchors oneAPI DPC++/C++ 2026.0.0, but
  entries from 2026-07-10 onward record 2026.1.0 (20260617) on the same
  driver 1.14.37020, and the roofline snapshot predates the bump. Resolution:
  re-run the kernel roofline snapshot under 2026.1.0 and refresh
  `baseline_status.md`.
- Split-MoE M1 medians measured the same day differ ~2x at the same shape
  (M1/E256/top8/K2048/I512): 0.15930 ms in the port harness (8–15 warmups) vs
  0.079858 ms in the row-tiling baseline (500 warmups, clocks held at
  2.35–2.40 GHz) — almost certainly clock state (2026-07-10, Port Qwen
  serving kernels vs NVFP4 split-MoE output-row tiling). Resolution: one
  interleaved run with a fixed warmup discipline, and record which discipline
  standing baselines use.
