#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event moe_route_topk_sycl(sycl::queue& q, const void* router_logits,
                                int* expert_ids, float* expert_weights,
                                std::size_t n_tokens, std::size_t n_experts,
                                int k, DType dt);

}  // namespace quixicore::xpu::kernels
