// Dispatch layer for the ssm family (native only — sequential scan).

#include "quixicore/xpu/ops.hpp"

#include "ssm/selective_scan/selective_scan_kernel.hpp"

namespace quixicore::xpu::ops {

void selective_scan(sycl::queue& q, const void* u, const void* delta,
                    const void* A, const void* B, const void* C, const void* D,
                    void* y, std::size_t n_chan, std::size_t seq,
                    std::size_t state, DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::selective_scan_sycl(q, u, delta, A, B, C, D, y,
                                                n_chan, seq, state, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
