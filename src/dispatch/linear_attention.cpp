// Dispatch layer for the linear_attention family (native only).

#include "quixicore/xpu/ops.hpp"

#include "linear_attention/linear_attn/linear_attn_kernel.hpp"

namespace quixicore::xpu::ops {

void linear_attn(sycl::queue& q, const void* Q, const void* K, const void* V,
                 void* O, std::size_t n_heads, std::size_t seq, std::size_t dim,
                 DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::linear_attn_sycl(q, Q, K, V, O, n_heads, seq, dim, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
