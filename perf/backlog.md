# XPU Optimization Backlog

The beam: 3-5 active idea families, best first. Pick from the top. Update
after every concluded experiment. Kill criteria are binding — when one fires,
record the kill in `perf/findings.md` and remove the family.

Where measurable, each family carries a quantitative target derived from
recorded data — a percentage of the measured roofline, or beating a named
baseline by a stated margin — set from `perf/findings.md` or
`perf/baseline_status.md`, never invented. The backend's aggregate score
lives in `perf/scoreboard.md`.

## Beam

### 1. Native XMX `joint_matrix` GEMM (w4a16 throughput pass, then dense task #9)
- Parent result: w4a16_gemm DPAS baseline is bit-exact and beats the per-row
  int4 GEMV 1.53x at M=16 / 2.12x at M=32, but trails the int8 SLM tile at
  0.37-0.53x; native dense_gemm sits at 1.1 vs 90 TFLOP/s vendor
  (2026-07-22, "w4a16_gemm"; 2026-07-06, "matmul/dense_gemm").
- Hypothesis: register-blocking several DPAS tiles per subgroup, loading the
  A tile straight from global (skip the SLM round-trip), double-buffering the
  dequantized weight tile, and a one-time VNNI-ready int4 repack close the
  gap to the int8 tile and pull native GEMM toward the XMX roofline.
- Evidence so far: `joint_matrix` runs bit-exact on B60 under both AoT and
  JIT (2026-07-22 feasibility probe); the DPAS advantage grows with M
  (0.72x at M=8 -> 2.12x at M=32), i.e. the loss is blocking/occupancy, not
  the tensor engine.
- Next action: add BM/BN register blocking (multiple accumulators per
  subgroup) to `w4a16_gemm_sycl` and A/B vs the gemv and int8 baselines at
  bf16, N=K=4096, M in {8,16,32}, group 128.
- Kill criteria: if register blocking + direct A loads + double buffering
  still lose to the int8 w8a8 SLM tile at every M<=32, stop the w4a16
  throughput line (keep the kernel only where it beats the GEMV); if native
  dense_gemm stays under 50% of oneDNN after the same treatment, mark task #9
  vendor-permanent.

### 2. Quant-GEMV decode bandwidth (bit-relocation transfer + GGUF repack)
- Parent result: nvfp4 bit-relocation decode reached 219.6 GB/s (48% of
  roofline), 1.92x (2026-07-08, "nvfp4_gemv pass #3"); GGUF GEMVs sit at
  32-88 GB/s because the interleaved 18-210-byte on-disk block layouts fight
  coalescing (2026-07-07, gguf_gemv entries; baseline snapshot).
- Hypothesis: (i) the e2m1 relocation applies verbatim to mxfp4 (its e8m0
  scale already folds via exp2) and plausibly to GGUF q4_0-style scale*int
  decodes via a fixup constant; (ii) a one-time repack from GGUF layout to
  scale-planar + aligned quants makes the GGUF GEMVs weight-BW-bound like the
  packed formats.
- Evidence so far: the decode-ALU diagnosis held uniformly across
  int4/mxfp4/nvfp4 (1.44-1.48x, 2026-07-07 pass #2), and relocation added a
  further 1.92x on nvfp4; mxfp4 is still on the LUT decoder at 119 GB/s.
- Next action: port the bit-relocation decode into mxfp4_gemv and A/B at
  8192x8192 bf16 against the 119 GB/s baseline.
- Kill criteria: if relocated mxfp4 cannot approach nvfp4 parity (~200+
  GB/s), or a GGUF repack cannot beat the in-place decoder by >=1.5x on
  q4_K/q6_K, record and drop; any repack that changes the caller-visible
  weight-format contract is out of bounds.

### 3. Attention throughput (cooperative subgroup, then tiled `joint_matrix` flash)
- Parent result: the per-query flash kernel measures 191 GFLOP/s at
  32h x 2048 x 64 (2026-07-07, "Track B depth begins"); attn_swa at d=256
  carries ~2 KB of registers per work-item and may spill (2026-07-22,
  "attn_swa"); the attention_f16ctx fusion payoff grows once attention gets
  faster (2026-07-22, "attention_f16ctx").
- Hypothesis: a cooperative-subgroup form (each lane holds d/16 of q/acc),
  then an SLM-tiled `joint_matrix` QK/PV path, multiplies throughput at
  seq>=1024 the way DPAS does for GEMM, and unlocks the deferred banded-GEMM
  SWA route.
- Evidence so far: XMX is real on this part (90 TFLOP/s bf16, 182 TOPS int8;
  `joint_matrix` probe bit-exact, 2026-07-22); no tiled attention variant has
  been measured yet.
- Next action: prototype the cooperative-subgroup attention and A/B vs
  `attention_sycl` at 32h x 2048 x 64 causal, f16.
- Kill criteria: if neither the cooperative nor the tiled form beats the
  per-query kernel by >=2x at seq>=2048 without a correctness regression, or
  register spills dominate at d=256, keep per-query + banded and remove the
  family until a new design exists.

### 4. Native decode-engine / graph-replay integration
- Parent result: native SYCL eager submit is ~2.4 us/kernel (~10x under
  torch's ~23 us/op) and graph replay ~1.3 us/kernel at R>=128; full-graph
  serving reached 68.07 tok/s vs 18.4 eager (2026-07-09, "Path-to-60-tok/s
  deep dive" Phase 0; "vLLM full XPU graph"); the framework-neutral command
  graph + XPUGraph wrappers are ported and replay-stable (2026-07-10, "Port
  Qwen serving kernels").
- Hypothesis: driving a realistic decode-layer chain through QuixiCore's own
  native submit / command-graph wrappers converts the "serving-wash" device
  wins (occupancy split, GDN core) into end-to-end wins independent of vLLM.
- Evidence so far: graph replay loses below R~64 (0.70x at R=8, fixed
  per-replay overhead) and wins 1.78-1.83x at R>=128 (2026-07-09, Phase 0
  Gate C); the GDN core belongs inside the engine, not behind torch dispatch
  (2026-07-09, Phase 2 correction).
- Next action: build a decode-layer chain benchmark in `quixicore_xpu_bench`
  (qk_norm_rope -> attention -> rms_residual_next -> glu_gelu_f16 -> quant
  GEMV) comparing native eager submit vs command-graph replay.
- Kill criteria: if graph replay gains <10% over native eager submit at
  realistic chain lengths, record eager-submit-only as the engine design and
  drop the graph arm; if the chain benchmark shows submission overhead is
  already <10% of device time, the whole family dies.

### 5. Sequential recurrences (chunked selective_scan / linear_attn tiling)
- Parent result: selective_scan runs 1.5 Gelem/s, sequential-bound, and
  linear_attn 136 GFLOP/s with naive triple loops (2026-07-07, "Track B depth
  begins"); both are flagged Deferred in `baseline_status.md`.
- Hypothesis: a chunked scan (intra-chunk parallel scan + inter-chunk state
  carry, ssd_chunk-style) converts the serial recurrence into bandwidth-bound
  passes; register tiling lifts linear_attn's naive loops.
- Evidence so far: nothing measured beyond the naive baselines; the GDN work
  showed recurrence state fits registers/SLM at decode shapes (2026-07-09
  Phase 2; 2026-07-10 port).
- Next action: implement a chunked selective_scan (sweep chunk 64/128) and
  A/B vs the sequential kernel at 4096 chan x 2048 seq x 16 state.
- Kill criteria: if chunking cannot beat the sequential kernel by >=2x at
  seq>=2048 (carry/materialization overhead eats the parallelism), record and
  re-defer.

## Parked (not on the beam)
- GDN batch-aware routing: native core wins B=1 (0.02467 vs 0.031 ms) but
  loses B=4 (0.046 vs 0.037 ms); serving-neutral under torch eager
  (2026-07-09 Phase 2; 2026-07-10 port entry). Revisit inside the native
  engine (Beam 4).
- fp8 M=1 GEMV's last ~30% of roofline: per-call scratch malloc/free +
  partial traffic are the named candidates (2026-07-08, fp8_gemm entry).
- e4m3 vendor GEMM at ~half of e5m2: oneDNN's e4m3->f16 up-convert, not ours;
  revisit on an oneDNN version bump (2026-07-08, fp8_gemm entry).
- softmax bf16 native 302 vs oneDNN 348 GB/s: exp-bound; try a faster exp
  approximation within tolerance or a one-pass online softmax (2026-07-06,
  16-byte vector-load pass, open questions).
- oneCCL vendor all_reduce variant vs the native 4-GPU USM reduce
  (2026-07-07, collectives breadth entry).
- Cooperative/workgroup variants of pool_mean_rms_l2 / rms_residual_next /
  qk_norm_rope for tiny batch x large dim (2026-07-22 entries, open
  questions).
- `attention` Variant::best route that picks the fused f16 store when the
  caller passes an O_f16 sink (2026-07-22, attention_f16ctx open questions).
- int4-vs-nvfp4 serving A/B under graph capture — predicted to lose
  (identical weight bytes at decode); only worth the paperwork once
  device-bound (2026-07-09, deep dive).
- gelu kVec 8/16 + explicit nd_range sweep to close the f32 gap to oneDNN's
  411 GB/s (2026-07-06, GELU entry, open questions).

## Migrated sources
- (a) "Roadmap to 60 (ordered by leverage / our-control)" (2026-07-09, "Deep
  dive — path to 60 tok/s" entry): item 1 (torch custom-op registration)
  completed 2026-07-09 Phase 1; item 2 (graph capture) completed 2026-07-09
  ("vLLM full XPU graph" 68 tok/s; "segmented XPU Graph enables TP2 decode");
  item 3 (device compute under capture) landed via the split MoE + output-row
  tiling + paired gate/up passes (2026-07-10/11) — the residual fused-GDN
  idea is Parked/Beam 4. The roadmap is marked superseded in the notebook
  (2026-08-15).
- (b) `perf/baseline_status.md` "Deferred (bigger projects, flagged not
  faked)": native dense_gemm XMX joint_matrix -> Beam 1; quant-GEMV decode
  coalescing -> Beam 2; sequential scans -> Beam 5.
- (c) Entry open-questions/deferred lines: register-blocking DPAS tiles
  deferred to a w4a16 throughput pass (2026-07-22, w4a16_gemm) -> Beam 1;
  tiled joint_matrix / banded-GEMM attention (2026-07-07 attention breadth;
  2026-07-22 attn_swa) -> Beam 3; mxfp4 bit-relocation + GGUF repack
  (2026-07-07 pass #2; 2026-07-08 nvfp4 pass #3) -> Beam 2; softmax bf16,
  oneCCL all_reduce, cooperative variants, f16ctx best-routing, gelu kVec
  sweep -> Parked.
- (d) "First Kernel Plan" (H3 inside the 2026-07-07 "Track B depth begins"
  entry): all five priority items completed — norms and softmax and GELU/GLU
  (2026-07-06 entries), quant format decode helpers and quant GEMV
  (2026-07-06/07 quantization entries); its three open questions were
  resolved by events (baseline target = Arc Pro B60; API = C++ library +
  `tk_xpu` torch binding; comparison baseline = oneDNN + torch-xpu parity
  mix). Nothing carried forward; the plan is marked superseded in the
  notebook (2026-08-15).
