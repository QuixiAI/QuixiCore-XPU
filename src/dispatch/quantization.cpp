// Dispatch layer for the quantization family.

#include "quixicore/xpu/ops.hpp"

#include "quantization/qgemm/qgemm_kernel.hpp"
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

void qgemm_int8(sycl::queue& q, const void* a_int8, const void* b_int8,
                const void* a_scale, const void* b_scale, void* c, std::size_t M,
                std::size_t N, std::size_t K, DType out_dt, Variant variant,
                bool blocking) {
  // int8 GEMM is compute-bound; oneDNN's XMX int8 path far outpaces the untuned
  // native tile, so best -> vendor when available.
  if (variant == Variant::best)
    variant = variant_available(Variant::vendor) ? Variant::vendor : Variant::sycl;
  const Variant v = resolve_variant(variant);

  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::qgemm_int8_onednn(q, a_int8, b_int8, a_scale, b_scale, c, M, N, K, out_dt);
      break;
#endif
    default:
      ev = kernels::qgemm_int8_sycl(q, a_int8, b_int8, a_scale, b_scale, c, M, N, K, out_dt);
      break;
  }
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
