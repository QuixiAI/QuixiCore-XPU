// Dispatch layer for the matmul family. Default routing prefers the vendor
// (oneDNN, XMX-backed) path when available, since the native SYCL variant is an
// untuned SLM-tiled baseline; callers can force Variant::sycl.

#include "quixicore/xpu/ops.hpp"

#include "matmul/dense_gemm/dense_gemm_kernel.hpp"

namespace quixicore::xpu::ops {

void dense_gemm(sycl::queue& q, const void* a, const void* b, void* c,
                std::size_t M, std::size_t N, std::size_t K, DType dt,
                Variant variant, bool blocking) {
  // GEMM is compute-bound; the oneDNN XMX path far outperforms the untuned
  // native tile kernel, so `best` -> vendor when oneDNN is present.
  if (variant == Variant::best) {
    variant = variant_available(Variant::vendor) ? Variant::vendor : Variant::sycl;
  }
  const Variant v = resolve_variant(variant);

  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::dense_gemm_onednn(q, a, b, c, M, N, K, dt);
      break;
#endif
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::dense_gemm_sycl(q, a, b, c, M, N, K, dt);
      break;
  }
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
