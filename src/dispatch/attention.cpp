// Dispatch layer for the attention family.

#include "quixicore/xpu/ops.hpp"

#include "attention/rope/rope_kernel.hpp"

namespace quixicore::xpu::ops {

void rope(sycl::queue& q, const void* x, void* out, std::size_t tokens,
          std::size_t n_heads, std::size_t head_dim, float base,
          std::size_t pos0, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev =
      kernels::rope_sycl(q, x, out, tokens, n_heads, head_dim, base, pos0, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
