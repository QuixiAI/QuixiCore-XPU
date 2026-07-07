// MoE top-k routing, native SYCL. One work-item per token (n_experts is small,
// tens–hundreds). Iterative argmax selection over the expert logits (k passes,
// masking chosen experts by lowest-index tie-break), then a softmax over the k
// selected logits gives the routing weights. kMaxK bounds the register arrays.

#include "moe/moe_route/moe_route_kernel.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxK = 16;

template <typename T>
sycl::event route_typed(sycl::queue& q, const T* logits, int* ids,
                        float* weights, std::size_t n_tokens,
                        std::size_t n_experts, int k) {
  return q.parallel_for(sycl::range<1>(n_tokens), [=](sycl::id<1> idx) {
    const std::size_t t = idx[0];
    const T* row = logits + t * n_experts;
    int sel[kMaxK];
    float val[kMaxK];

    for (int i = 0; i < k; ++i) {
      float best = -std::numeric_limits<float>::infinity();
      int best_e = 0;
      for (std::size_t e = 0; e < n_experts; ++e) {
        bool taken = false;
        for (int j = 0; j < i; ++j) taken |= (sel[j] == static_cast<int>(e));
        if (taken) continue;
        const float v = static_cast<float>(row[e]);
        if (v > best) { best = v; best_e = static_cast<int>(e); }
      }
      sel[i] = best_e;
      val[i] = best;
    }

    // softmax over the k selected logits
    float m = val[0];
    for (int i = 1; i < k; ++i) m = sycl::max(m, val[i]);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) { val[i] = sycl::exp(val[i] - m); sum += val[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < k; ++i) {
      ids[t * k + i] = sel[i];
      weights[t * k + i] = val[i] * inv;
    }
  });
}

}  // namespace

sycl::event moe_route_topk_sycl(sycl::queue& q, const void* router_logits,
                                int* expert_ids, float* expert_weights,
                                std::size_t n_tokens, std::size_t n_experts,
                                int k, DType dt) {
  switch (dt) {
    case DType::f32:
      return route_typed(q, static_cast<const float*>(router_logits), expert_ids, expert_weights, n_tokens, n_experts, k);
    case DType::f16:
      return route_typed(q, static_cast<const half_t*>(router_logits), expert_ids, expert_weights, n_tokens, n_experts, k);
    case DType::bf16:
      return route_typed(q, static_cast<const bf16_t*>(router_logits), expert_ids, expert_weights, n_tokens, n_experts, k);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
