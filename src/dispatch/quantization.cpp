// Dispatch layer for the quantization family.

#include "quixicore/xpu/ops.hpp"

#include "quantization/qgemv/qgemv_kernel.hpp"

namespace quixicore::xpu::ops {

void qgemv_int4(sycl::queue& q, const void* w_packed, const void* scales,
                const void* x, void* y, std::size_t N, std::size_t K,
                std::size_t group, DType act_dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev =
      kernels::qgemv_int4_sycl(q, w_packed, scales, x, y, N, K, group, act_dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
