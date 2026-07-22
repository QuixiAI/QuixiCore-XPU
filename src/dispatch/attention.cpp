// Dispatch layer for the attention family.

#include "quixicore/xpu/ops.hpp"

#include "attention/attention/attention_kernel.hpp"
#include "attention/rope/rope_kernel.hpp"

#include "attention/qk_norm_rope/qk_norm_rope_kernel.hpp"

namespace quixicore::xpu::ops {

void attention(sycl::queue& q, const void* Q, const void* K, const void* V,
               void* O, std::size_t n_heads, std::size_t n_kv_heads,
               std::size_t seq_q, std::size_t seq_k, std::size_t d, bool causal,
               DType dt, Variant variant, bool blocking) {
  (void)variant;  // native flash; oneDNN-Graph SDPA vendor variant deferred
  sycl::event ev = kernels::attention_sycl(q, Q, K, V, O, n_heads, n_kv_heads,
                                           seq_q, seq_k, d, causal, dt);
  if (blocking) ev.wait();
}

void attention_f16ctx(sycl::queue& q, const void* Q, const void* K,
                      const void* V, void* O, void* O_f16, std::size_t n_heads,
                      std::size_t n_kv_heads, std::size_t seq_q,
                      std::size_t seq_k, std::size_t d, bool causal, DType dt,
                      Variant variant, bool blocking) {
  (void)variant;  // native flash + fused f16 store
  sycl::event ev =
      kernels::attention_f16ctx_sycl(q, Q, K, V, O, O_f16, n_heads, n_kv_heads,
                                     seq_q, seq_k, d, causal, dt);
  if (blocking) ev.wait();
}

void attn_swa(sycl::queue& q, const void* Q, const void* K, const void* V,
              void* O, std::size_t n_heads, std::size_t n_kv_heads,
              std::size_t seq_q, std::size_t seq_k, std::size_t d,
              std::size_t window, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native flash + symmetric sliding-window band mask
  sycl::event ev = kernels::attn_swa_sycl(q, Q, K, V, O, n_heads, n_kv_heads,
                                          seq_q, seq_k, d, window, dt);
  if (blocking) ev.wait();
}

void rope(sycl::queue& q, const void* x, void* out, std::size_t tokens,
          std::size_t n_heads, std::size_t head_dim, float base,
          std::size_t pos0, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev =
      kernels::rope_sycl(q, x, out, tokens, n_heads, head_dim, base, pos0, dt);
  if (blocking) ev.wait();
}


void qk_norm_rope(sycl::queue& q, void* Q, void* K, const void* q_weight,
                  const void* k_weight, void* Q_f16, void* K_f16,
                  std::size_t tokens, std::size_t n_head, std::size_t n_head_kv,
                  std::size_t head_dim, float base, std::size_t pos0,
                  float query_scale, float eps, DType dt, Variant variant,
                  bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::qk_norm_rope_sycl(
      q, Q, K, q_weight, k_weight, Q_f16, K_f16, tokens, n_head, n_head_kv,
      head_dim, base, pos0, query_scale, eps, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
