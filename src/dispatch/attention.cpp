// Dispatch layer for the attention family.

#include "quixicore/xpu/ops.hpp"

#include "attention/attention/attention_kernel.hpp"
#include "attention/rope/rope_kernel.hpp"

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

void rope(sycl::queue& q, const void* x, void* out, std::size_t tokens,
          std::size_t n_heads, std::size_t head_dim, float base,
          std::size_t pos0, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev =
      kernels::rope_sycl(q, x, out, tokens, n_heads, head_dim, base, pos0, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
