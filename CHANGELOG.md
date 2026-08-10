# Changelog

All notable QuixiCore XPU changes should be recorded here.

## Unreleased

### Serving-port wave 2 (from the vLLM XPU optimization work)

- ssd_decode: Mamba-2 SSD selective-state-update decode (scalar-A-per-head)
  with independent act/state dtypes, strided state views, and copy-on-write
  slot indices. Graph-capture-safe.
- causal_conv1d_decode: depthwise causal conv decode update (width <= 8)
  with in-place shift-append state, both serving layouts via strides, and
  null-slot semantics. Graph-capture-safe.
- ssd_prefill: varlen Mamba-2 SSD prefill selective scan (sequential
  xpu_sycl_seq variant, SLM-resident state rows, softplus + dt_limit clamp,
  caller-owned state gather/scatter). Graph-capture-safe.
- causal_conv1d_prefill: varlen depthwise causal conv prefill (dim-major
  layout, event-chained output + state write-back passes, has_init and
  null-slot semantics). Graph-capture-safe.
- nvfp4_moe_relu2_fused / nvfp4_moe_relu2_split: ReLU²-ungated NVFP4 MoE
  experts (NemotronH) — single up-projection w1 [E,I,K/2], relu(g)² with no
  gate multiply, W4A16, EP-safe expert-id skip.
- all_reduce: capturable P2P sum all-reduce (world 2..8, one-shot +
  two-shot) with bitwise-identical fixed-rank-order results, replay-safe
  generation-counter rendezvous, and no peer-atomic requirement;
  all_reduce_sum now orchestrates it in-process across all visible GPUs.
- group_rms_norm_gated: gated group-RMSNorm (Mamba-2 mixer norm, fp32
  gating, per-group variance, torch rounding order) — replaces the last
  eager-torch op in the XPU Mamba decode path.
- glu_quant: fused SwiGLU + activation quant (per-group fp8 e4m3,
  mxfp4), scale rules shared with norm_quant.
- norm_quant: fused RMSNorm + activation quant (static/dynamic fp8 e4m3,
  mxfp4 with power-of-two group scales); in-place residual form covers the
  residual_rms_norm_quant contract.
- turboquant: KV-cache codec encode/decode implementing the format v2 ABI
  (rotated Lloyd-Max keys or e4m3 byte keys, per-group uniform values,
  LSB-first packing), byte-identical to the host-shared codec oracle.
- moe_route_topk gating modes (sigmoid with routed scaling, softplus_sqrt)
  + moe_permute / moe_unpermute_weighted_reduce: the routing + permutation
  pipeline feeding moe_grouped_qgemm, EP-safe and allocation-free.
- merge_attn_states + kv_cache_gather_paged: the split-KV combiner (with
  the empty-partition NaN guard) and the paged gather/dequant — completing
  the paged-attention utility set.
- mrope / rotary_positioned: multimodal (sectioned) and positioned rotary
  embedding, NeoX + GPT-J, in-place with untouched tails.
- kv_cache_scatter_paged: paged KV-cache write (flat slot mapping, fp8
  e4m3 mode) — the write side completing the paged-attention story.
- gated_delta_rule_varlen + gdn_l2norm_qk: general varlen Gated DeltaNet
  (exact recurrence, GQA, slot-indirected state) — GDN prefill/decode beyond
  the shape-locked qwen_gdn_decode.
- moe_grouped_qgemm / moe_grouped_qswiglu: segmented per-expert GEMM with
  fused w16/int4/nvfp4 dequant on the native DPAS block (graph-safe
  on-device segmentation; composite swiglu); experimental — GEMV split
  stays the production route until the recorded throughput pass.
- paged_attention_decode / paged_attention_prefill: native paged KV-cache
  attention (runtime page size, split-KV decode with caller workspaces and
  LSE merge, varlen prefill with causal/window/sinks/LSE/mixed-batch mask,
  fp8 KV) — the backend's biggest contract gap closed, cutlass-free.
- mqa_logits: fp8 MQA indexer logits on the new native joint_matrix
  building block (kernels/common/xmx_tile.hpp + quant_codecs.hpp) — the
  first cutlass-free DPAS rewrite consumer.
- Build: deterministic fp32 device arithmetic (-fp-model=precise +
  correctly-rounded offload divide/sqrt) — required for bit-exact codec
  contracts; manual integer fp16 conversions in codecs (the optimizer may
  elide compiler half round-trips).

### Kernel families — breadth

- Opened all thirteen families with native SYCL implementations: activations
  (gelu, silu, glu swiglu/geglu/reglu, softmax, gelu_backward), norms
  (rms_norm, layernorm), matmul (dense_gemm), attention (rope, flash-style
  online-softmax attention with MHA/GQA/causal), sampling (argmax,
  sample_categorical, top_k_sample), quantization, moe (moe_route_topk),
  linear_attention (linear_attn), ssm (selective_scan), serving
  (embedding_lookup, kv_cache scatter/gather), utils (dropout, cross_entropy,
  hadamard), optimizers (adamw), collectives (native multi-GPU
  all_reduce_sum, capability-gated).
- Co-equal oneDNN vendor variants for gelu, softmax, layernorm, dense_gemm,
  qgemm_int8, fp8_gemm, selectable at the ABI with data-driven
  `Variant::best` routing.

### Quantization — depth

- All 16 GGUF weight formats decode natively on Intel (q8_0, q4_0, q4_1,
  q5_0, q5_1, q6_K, q4_K, q5_K, q2_K, q3_K, iq4_nl, iq4_xs, iq2_xxs, iq2_xs,
  iq3_xxs, iq1_s).
- int4 group quant end-to-end: qgemv_int4, quantize_int4_group, act_quant
  (per-token int8) completing the w8a8 pipeline, qgemm_int8.
- FP4/FP8: mxfp4 GEMV (OCP microscaling), nvfp4 GEMV, fp8 GEMM (e4m3/e5m2)
  with fp8 codecs.
- w4a16_gemm: int4-weight x f16/bf16-activation DPAS GEMM written natively
  against SYCL joint_matrix (no external GEMM library).
- Declared turboquant codec ABI (format version 2): rotated Lloyd-Max keys
  (sign vector + FWHT + per-group FP16 RMS + centroid tables), per-group
  uniform values; SYCL implementation to follow.

### Performance campaign

- Three optimization passes recorded in perf/optimization_status.md,
  including 4-bit quant-GEMV decode 1.44-1.48x, fp8 M=1 decode GEMV 18.9x,
  nvfp4_gemv bit-relocation decode 1.92x (219.6 GB/s bf16), and vectorized
  row kernels (~2x bf16/f16).
- Benchmark harness and perf docs made backend-owned
  (perf/bench_kernels.py, perf/harness/xpu_bench.cpp, perf/configs/).

### Qwen serving port (from the vLLM XPU optimization work)

- nvfp4_gemm (W4A16 [M,K]x[N,K/2]) with row-loop and M-tiled variants;
  nvfp4_moe fused + split top-k routed MoE with paired gate/up decode.
- fp8_gemm_w8a16 (e4m3/e5m2, per-tensor or per-channel scales) with
  measured native/vendor M-crossover routing.
- qwen_gdn_decode: Qwen3.5/3.6 non-interleaved Gated DeltaNet decode
  (conv + recurrent state update).
- fused_add_rms_norm (exact in-place residual variant).
- SYCL command-graph capture/replay on the current stream
  (src/runtime/graph.cpp) and Level-Zero-preferred GPU enumeration.
- PyTorch XPU zero-copy binding (bindings/pytorch tk_xpu) validated against
  torch.xpu eager kernels.

### EmbeddingGemma port

- attention_f16ctx (fused f16 context store), attn_swa (symmetric
  sliding-window attention), qk_norm_rope (fused per-head QK-norm +
  query-scale + RoPE), rms_residual_next (fused residual-add + double
  RMSNorm -> f16), glu_gelu_f16 (GEGLU with fused f16 output),
  pool_mean_rms_l2 (masked-mean + per-token RMSNorm + L2 pooling head).

### Contract and metadata

- Canonical kernel contract stubs generated from the umbrella registry
  (.quixicore/kernel-stubs.yaml, include/quixicore/xpu/contract_stubs.hpp).
- QuixiCore-standard repository structure documentation, kernel/quant
  coverage manifests, and standardized contributor, security, changelog,
  formatting, and script entrypoint files.
