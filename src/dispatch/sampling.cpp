// Dispatch layer for the sampling family.

#include "quixicore/xpu/ops.hpp"

#include "sampling/argmax/argmax_kernel.hpp"

namespace quixicore::xpu::ops {

void argmax(sycl::queue& q, const void* logits, int* out, std::size_t rows,
            std::size_t vocab, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::argmax_sycl(q, logits, out, rows, vocab, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
