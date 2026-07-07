// Dispatch layer for the moe family (native only).

#include "quixicore/xpu/ops.hpp"

#include "moe/moe_route/moe_route_kernel.hpp"

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

}  // namespace quixicore::xpu::ops
