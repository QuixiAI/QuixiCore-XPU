// Dispatch layer for the ssm family (native only — sequential scan).

#include "quixicore/xpu/ops.hpp"

#include "ssm/selective_scan/selective_scan_kernel.hpp"
#include "ssm/ssd/ssd_kernel.hpp"

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

void ssd_decode(sycl::queue& q, void* state, const void* x, const float* dt_raw,
                const float* A, const void* B, const void* C, const float* D,
                const float* dt_bias, const std::int32_t* src_indices,
                const std::int32_t* dst_indices, void* out, bool dt_softplus,
                std::size_t batch, std::size_t nheads, std::size_t headdim,
                std::size_t dstate, std::size_t ngroups, std::size_t nslots,
                std::int64_t s0, std::int64_t s1, std::int64_t s2,
                std::int64_t s3, DType act_dt, DType state_dt, Variant variant,
                bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::ssd_decode_sycl(
      q, state, x, dt_raw, A, B, C, D, dt_bias, src_indices, dst_indices, out,
      dt_softplus, batch, nheads, headdim, dstate, ngroups, nslots, s0, s1, s2,
      s3, act_dt, state_dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
