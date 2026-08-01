// Dispatch layer for the moe family (native only).

#include "quixicore/xpu/ops.hpp"

#include "moe/moe_route/moe_route_kernel.hpp"
#include "moe/nvfp4_moe/nvfp4_moe_kernel.hpp"

namespace quixicore::xpu::ops {

void moe_route_topk(sycl::queue& q, const void* router_logits, int* expert_ids,
                    float* expert_weights, std::size_t n_tokens,
                    std::size_t n_experts, int k, DType dt, Variant variant,
                    bool blocking) {
  (void)variant;
  sycl::event ev = kernels::moe_route_topk_sycl(q, router_logits, expert_ids,
                                                expert_weights, n_tokens,
                                                n_experts, k, dt);
  if (blocking) ev.wait();
}

void nvfp4_moe_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *out_f32, std::size_t M, std::size_t E,
                     std::size_t top_k, std::size_t K, std::size_t I, DType act_dt,
                     bool multiply_router_weight, Variant variant, bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_fused_sycl(
      q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales, w2, w2_scales,
      w2_global_scales, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt, zeroed);
  if (blocking)
    event.wait();
}

void nvfp4_moe_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *scratch_f32, float *out_f32,
                     std::size_t M, std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                     DType act_dt, bool multiply_router_weight, Variant variant, bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_split_sycl(
      q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales, w2, w2_scales,
      w2_global_scales, scratch_f32, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt,
      zeroed);
  if (blocking)
    event.wait();
}

}  // namespace quixicore::xpu::ops
