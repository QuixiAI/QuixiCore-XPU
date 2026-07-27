# XPU Capability Gaps

Inventory date: 2026-07-27.

This file compares the XPU backend with the **263-operation semantic
union** and **17 exact quant-format IDs** recorded in the
[QuixiCore umbrella capability map](https://github.com/QuixiAI/QuixiCore/blob/main/matrices/capability-map.md).

Backend snapshot: `main` @ `67c70fe4dc0c`.

Normalized adapter stubs, including planned practical inference and fused
operations, are indexed in `.quixicore/kernel-stubs.yaml` and declared in
`include/quixicore/xpu/contract_stubs.hpp`. Their counts differ from this
exact-ID evidence comparison because aliases are collapsed and planned
contracts are included.

Union source revisions: CUDA d959679b0163; Metal a6d984377288; ROCm 636ae5ae983f; XPU 67c70fe4dc0c; CPU 0159223979db.

## How gaps are classified

- **family-only metadata**: this backend marks the family implemented but
  does not publish the exact operation ID. This is an enumeration/evidence
  gap, not proof that the semantic kernel is missing.
- **partial-family coverage**: the exact ID is absent and the backend marks
  the family partial.
- **capability-gated**: the family or operation depends on hardware/runtime
  conditions and is not an unconditional capability.
- **planned family**, **no family claim**, **partial operation**, and
  **experimental operation** are implementation or maturity gaps relative
  to a fully evidenced union capability.

Exact accelerator stage/layout aliases remain separate because the umbrella
map preserves published operation IDs. A backend may close a metadata gap
by documenting a proven semantic collapse instead of adding duplicate code.

## Summary

| Measure | Count |
|---|---:|
| Union operation capabilities | 263 |
| Fully implemented or semantically mapped | 30 |
| Operation gaps or enumeration gaps | 233 |
| experimental operation | 1 |
| capability-gated | 4 |
| partial-family coverage | 191 |
| no family claim | 37 |
| Union quant-format IDs | 17 |
| Fully declared quant-format IDs | 1 |
| Quant-format gaps or missing declarations | 16 |
| quant: experimental operation | 5 |
| quant: planned operation | 5 |
| quant: no exact format declaration | 6 |

## Operation gap list

| Union family | Capability | Gap class |
|---|---|---|
| Norms | `norm_quant` | partial-family coverage |
| Norms | `qk_norm_rope` | partial-family coverage |
| Norms | `qk_norm_rope_kv_f16` | partial-family coverage |
| Norms | `qk_norm_rope_positioned` | partial-family coverage |
| Norms | `rms_norm_residual_next` | partial-family coverage |
| Norms | `rms_residual_next` | partial-family coverage |
| Norms | `rmsnorm` | partial-family coverage |
| Activations | `elementwise` | partial-family coverage |
| Activations | `leaky_relu` | partial-family coverage |
| Activations | `sigmoid_mul` | partial-family coverage |
| Activations | `sigmoid_mul_backward` | partial-family coverage |
| Activations | `silu_backward` | partial-family coverage |
| Activations | `softmax_backward` | partial-family coverage |
| Activations | `unary` | partial-family coverage |
| Activations | `value_clip` | partial-family coverage |
| Attention | `attn_composites` | partial-family coverage |
| Attention | `attn_decode_bh` | partial-family coverage |
| Attention | `attn_fwd_sg_d256` | partial-family coverage |
| Attention | `biased_attention` | partial-family coverage |
| Attention | `cross_attention` | partial-family coverage |
| Attention | `decode_cache_attention` | partial-family coverage |
| Attention | `gqa` | partial-family coverage |
| Attention | `gqa_backward` | partial-family coverage |
| Attention | `gqa_causal` | partial-family coverage |
| Attention | `gqa_causal_backward` | partial-family coverage |
| Attention | `gqa_swa` | partial-family coverage |
| Attention | `mrope` | partial-family coverage |
| Attention | `paged_attention_q8_0` | partial-family coverage |
| Attention | `rope_variants` | partial-family coverage |
| Attention | `rotary` | partial-family coverage |
| Attention | `rotary_positioned` | partial-family coverage |
| Linear attention | `based` | partial-family coverage |
| Linear attention | `gated_linear_attention` | partial-family coverage |
| Linear attention | `gdn_gate_beta` | partial-family coverage |
| Linear attention | `gdn_gated_rmsnorm` | partial-family coverage |
| Linear attention | `gdn_qkv_prepare` | partial-family coverage |
| Linear attention | `gdn_recur` | partial-family coverage |
| Linear attention | `gdn_recurrence` | partial-family coverage |
| Linear attention | `gdn_short_conv` | partial-family coverage |
| Linear attention | `hedgehog` | partial-family coverage |
| Linear attention | `linear_attention_unnormalized` | partial-family coverage |
| Linear attention | `rwkv_wkv6` | partial-family coverage |
| Linear attention | `rwkv_wkv7` | partial-family coverage |
| State-space models | `dsv4_hc_comb` | partial-family coverage |
| State-space models | `dsv4_hc_post` | partial-family coverage |
| State-space models | `dsv4_hc_pre` | partial-family coverage |
| State-space models | `fftconv` | partial-family coverage |
| State-space models | `mamba2_backward` | partial-family coverage |
| State-space models | `ssd_chunked_backward` | partial-family coverage |
| State-space models | `ssd_decode` | partial-family coverage |
| Dense matmul and projections | `bf16fp32_matmul` | partial-family coverage |
| Dense matmul and projections | `complex_gemm` | partial-family coverage |
| Dense matmul and projections | `decode_linear` | partial-family coverage |
| Dense matmul and projections | `decode_linear_epilogue` | partial-family coverage |
| Dense matmul and projections | `decode_linear_epilogue_dense` | partial-family coverage |
| Dense matmul and projections | `decode_linear_epilogue_packed` | partial-family coverage |
| Dense matmul and projections | `decode_linear_q8` | partial-family coverage |
| Dense matmul and projections | `decode_linear_residual` | partial-family coverage |
| Dense matmul and projections | `decode_swiglu` | partial-family coverage |
| Dense matmul and projections | `decode_swiglu_dense` | partial-family coverage |
| Dense matmul and projections | `decode_swiglu_packed` | partial-family coverage |
| Dense matmul and projections | `flux` | partial-family coverage |
| Dense matmul and projections | `fp8fp32_matmul` | partial-family coverage |
| Dense matmul and projections | `gemm_gate_residual` | partial-family coverage |
| Dense matmul and projections | `gemm_staged` | partial-family coverage |
| Dense matmul and projections | `grouped_gemm` | partial-family coverage |
| Dense matmul and projections | `int8_matmul` | partial-family coverage |
| Dense matmul and projections | `lora_apply` | partial-family coverage |
| Dense matmul and projections | `lora_apply_direct_f16` | partial-family coverage |
| Dense matmul and projections | `matmul_custom` | partial-family coverage |
| Dense matmul and projections | `mxfp8_matmul` | partial-family coverage |
| Dense matmul and projections | `nvfp4_matmul` | partial-family coverage |
| Dense matmul and projections | `scaled_matmul` | partial-family coverage |
| Quantization | `base_q_dequant` | partial-family coverage |
| Quantization | `base_q_embedding` | partial-family coverage |
| Quantization | `base_q_fused_consumers` | partial-family coverage |
| Quantization | `base_q_gemm` | partial-family coverage |
| Quantization | `base_q_gemv` | partial-family coverage |
| Quantization | `base_q_gemv_qkv` | partial-family coverage |
| Quantization | `base_q_gemv_swiglu` | partial-family coverage |
| Quantization | `base_q_lm_head_argmax` | partial-family coverage |
| Quantization | `base_q_moe_gemm` | partial-family coverage |
| Quantization | `base_q_moe_swiglu` | partial-family coverage |
| Quantization | `base_q_qkv` | partial-family coverage |
| Quantization | `base_q_swiglu` | partial-family coverage |
| Quantization | `calibration_absmax` | partial-family coverage |
| Quantization | `dequant_gather` | partial-family coverage |
| Quantization | `fake_quant_float8` | partial-family coverage |
| Quantization | `fake_quant_int8` | partial-family coverage |
| Quantization | `fp8_gemm` | experimental operation |
| Quantization | `lm_head` | partial-family coverage |
| Quantization | `lm_head_beam_advance` | partial-family coverage |
| Quantization | `lm_head_candidates` | partial-family coverage |
| Quantization | `lm_head_masked` | partial-family coverage |
| Quantization | `qgeglu` | partial-family coverage |
| Quantization | `qgemm` | partial-family coverage |
| Quantization | `qgemm_backward_input` | partial-family coverage |
| Quantization | `qgemm_int` | partial-family coverage |
| Quantization | `qgemm_q4q8` | partial-family coverage |
| Quantization | `qgemv` | partial-family coverage |
| Quantization | `qgemv_q4_0_f32_qkv` | partial-family coverage |
| Quantization | `qgemv_q4_0_f32_up_gate` | partial-family coverage |
| Quantization | `qgemv_q4_0_f32_up_gate_gelu` | partial-family coverage |
| Quantization | `qkv_proj_fused` | partial-family coverage |
| Quantization | `quant_rt` | partial-family coverage |
| Quantization | `quantized_embedding` | partial-family coverage |
| Quantization | `quantized_embedding_bag` | partial-family coverage |
| Quantization | `ternary_code_flip_count` | partial-family coverage |
| Quantization | `ternary_pack` | partial-family coverage |
| Quantization | `ternary_stats` | partial-family coverage |
| Quantization | `ternary_unpack` | partial-family coverage |
| Quantization | `tq2_0_pack` | partial-family coverage |
| Quantization | `tq2_0_unpack` | partial-family coverage |
| Quantization | `turboquant` | partial-family coverage |
| Mixture of experts | `moe` | partial-family coverage |
| Mixture of experts | `moe_finalize_backward` | partial-family coverage |
| Mixture of experts | `moe_gather_backward` | partial-family coverage |
| Mixture of experts | `moe_grouped_gemm_backward_input` | partial-family coverage |
| Mixture of experts | `moe_grouped_gemm_backward_weight` | partial-family coverage |
| Mixture of experts | `moe_grouped_qgemm` | partial-family coverage |
| Mixture of experts | `moe_grouped_qswiglu` | partial-family coverage |
| Mixture of experts | `moe_quant` | partial-family coverage |
| Mixture of experts | `moe_route_grouped` | partial-family coverage |
| Sampling | `logits_softcap` | partial-family coverage |
| Sampling | `top_k_renorm` | partial-family coverage |
| Sampling | `top_p_renorm` | partial-family coverage |
| Serving and caches | `embedding_lookup_types` | partial-family coverage |
| Serving and caches | `kv_cache_copy_blocks_q8_0` | partial-family coverage |
| Serving and caches | `kv_cache_gather_bitnet_kv3` | partial-family coverage |
| Serving and caches | `kv_cache_gather_q8_0` | partial-family coverage |
| Serving and caches | `kv_cache_q8_0` | partial-family coverage |
| Serving and caches | `kv_cache_scatter_bitnet_kv3` | partial-family coverage |
| Serving and caches | `kv_cache_scatter_q8_0` | partial-family coverage |
| Serving and caches | `masked_mean_pool_rms_l2` | partial-family coverage |
| Serving and caches | `mean_pool_rms_l2` | partial-family coverage |
| Serving and caches | `paged_attention_advanced` | partial-family coverage |
| Serving and caches | `paged_attention_bitnet_kv3` | partial-family coverage |
| Serving and caches | `paged_attention_turboquant` | partial-family coverage |
| Serving and caches | `quantized_attention` | partial-family coverage |
| Serving and caches | `serving` | partial-family coverage |
| Optimizers | `adamw_masked` | partial-family coverage |
| Optimizers | `sgd` | partial-family coverage |
| Collectives | `broadcast` | capability-gated |
| Collectives | `fp8_gemm_collectives` | capability-gated |
| Collectives | `reduce_sum` | capability-gated |
| Collectives | `standalone_collectives` | capability-gated |
| Vision | `add_relative_position_2d` | no family claim |
| Vision | `avg_pool2d_tokens` | no family claim |
| Vision | `edge_mlp_256x7` | no family claim |
| Vision | `extract_patches_2d` | no family claim |
| Vision | `extract_patches_3d` | no family claim |
| Vision | `factorized_position_2d` | no family claim |
| Vision | `get_relative_position` | no family claim |
| Vision | `interpolate_position_2d` | no family claim |
| Vision | `patch_merge_layer_norm` | no family claim |
| Vision | `pool_tokens_by_position` | no family claim |
| Vision | `qwen_vision_rope_2d` | no family claim |
| Vision | `space_to_depth_norm_linear` | no family claim |
| Vision | `timestep_embedding` | no family claim |
| Vision | `upscale_nearest_2d` | no family claim |
| Vision | `vision_patch_projection` | no family claim |
| Vision | `vision_patch_projection_3d` | no family claim |
| Vision | `vision_rope_2d` | no family claim |
| Vision | `window_partition` | no family claim |
| Vision | `window_unpartition` | no family claim |
| Audio | `audio_causal_depthwise_conv1d` | no family claim |
| Audio | `audio_conv1d` | no family claim |
| Audio | `audio_conv1d_direct` | no family claim |
| Audio | `audio_depthwise_conv1d` | no family claim |
| Audio | `audio_relative_attention` | no family claim |
| Convolution | `col2im_1d` | no family claim |
| Convolution | `col2im_2d` | no family claim |
| Convolution | `conv2d` | no family claim |
| Convolution | `conv3d` | no family claim |
| Convolution | `conv_transpose_1d` | no family claim |
| Convolution | `conv_transpose_2d` | no family claim |
| Convolution | `depthwise_conv2d` | no family claim |
| Convolution | `im2col_2d` | no family claim |
| Convolution | `im2col_3d` | no family claim |
| Convolution | `pool1d` | no family claim |
| Convolution | `pool2d` | no family claim |
| Convolution | `pool2d_backward` | no family claim |
| Pooling | `pool_mean_rms_l2` | no family claim |
| Utilities and training | `accumulate` | partial-family coverage |
| Utilities and training | `add_id` | partial-family coverage |
| Utilities and training | `add_scalar` | partial-family coverage |
| Utilities and training | `arange` | partial-family coverage |
| Utilities and training | `argsort` | partial-family coverage |
| Utilities and training | `clamp` | partial-family coverage |
| Utilities and training | `concat` | partial-family coverage |
| Utilities and training | `cosine` | partial-family coverage |
| Utilities and training | `count_equal` | partial-family coverage |
| Utilities and training | `cumulative_sum` | partial-family coverage |
| Utilities and training | `diag_embed` | partial-family coverage |
| Utilities and training | `diag_mask` | partial-family coverage |
| Utilities and training | `divide` | partial-family coverage |
| Utilities and training | `fill` | partial-family coverage |
| Utilities and training | `group_norm` | partial-family coverage |
| Utilities and training | `kd_ce_fused_bwd` | partial-family coverage |
| Utilities and training | `kd_ce_fused_fwd` | partial-family coverage |
| Utilities and training | `kd_kl_dense_bwd` | partial-family coverage |
| Utilities and training | `kd_kl_dense_fwd` | partial-family coverage |
| Utilities and training | `kd_kl_topk_bwd` | partial-family coverage |
| Utilities and training | `kd_kl_topk_fwd` | partial-family coverage |
| Utilities and training | `l2_normalize` | partial-family coverage |
| Utilities and training | `logarithm` | partial-family coverage |
| Utilities and training | `marginal` | partial-family coverage |
| Utilities and training | `multiply` | partial-family coverage |
| Utilities and training | `outer_product` | partial-family coverage |
| Utilities and training | `pad_2d` | partial-family coverage |
| Utilities and training | `pad_reflect_1d` | partial-family coverage |
| Utilities and training | `reduce_mean` | partial-family coverage |
| Utilities and training | `reduce_sum_all` | partial-family coverage |
| Utilities and training | `repeat_2d` | partial-family coverage |
| Utilities and training | `repeat_backward_2d` | partial-family coverage |
| Utilities and training | `roll_2d` | partial-family coverage |
| Utilities and training | `scale` | partial-family coverage |
| Utilities and training | `set_rows` | partial-family coverage |
| Utilities and training | `sine` | partial-family coverage |
| Utilities and training | `solve_lower_triangular` | partial-family coverage |
| Utilities and training | `square` | partial-family coverage |
| Utilities and training | `square_root` | partial-family coverage |
| Utilities and training | `subtract` | partial-family coverage |
| Utilities and training | `tensor_copy` | partial-family coverage |
| Utilities and training | `tensor_set_4d` | partial-family coverage |
| Utilities and training | `threshold_topk_indices` | partial-family coverage |
| Utilities and training | `triangular_fill` | partial-family coverage |
| Attention | `attention_with_lse` | partial-family coverage |
| Utilities and training | `cross_entropy_backward` | partial-family coverage |
| Serving and caches | `embedding_backward` | partial-family coverage |
| Serving and caches | `indexer_k_gather` | partial-family coverage |
| Norms | `rms_norm_backward` | partial-family coverage |
| Activations | `swiglu_oai` | partial-family coverage |

## Quant-format gap list

| Format ID | Gap class |
|---|---|
| `awq` | planned operation |
| `base_qn` | no exact format declaration |
| `bitnet` | planned operation |
| `fp4` | planned operation |
| `fp8` | experimental operation |
| `int4_group` | experimental operation |
| `int8` | experimental operation |
| `marlin_awq_gptq_hqq` | no exact format declaration |
| `mx` | no exact format declaration |
| `mxfp4` | experimental operation |
| `mxfp6` | planned operation |
| `mxfp8` | planned operation |
| `nvfp4` | experimental operation |
| `q8_0_kv` | no exact format declaration |
| `tq2_0` | no exact format declaration |
| `turboquant` | no exact format declaration |

## Evidence rule

Removing an implementation or maturity gap requires the backend's native
path, correctness coverage, focused performance evidence, and an updated
manifest/status entry. Removing a family-only metadata gap requires an exact
operation entry or a documented semantic alias backed by the existing tests
and performance notebook. Directory presence alone is not sufficient.

Evidence remains backend-owned in `perf/optimization_status.md`,
`perf/baseline_status.md`, `perf/results/`, and the
backend correctness tests.
