# Changelog

All notable QuixiCore XPU changes should be recorded here.

## Unreleased

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
