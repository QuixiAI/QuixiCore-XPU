// Dispatch layer for the norms family: routes the public ops ABI to the selected
// implementation variant (native SYCL or vendor oneDNN) and applies blocking.

#include "quixicore/xpu/ops.hpp"

#include "norms/norms_kernel.hpp"

namespace quixicore::xpu::ops {

void rms_norm(sycl::queue& q, const void* x, const void* weight, void* out,
              std::size_t rows, std::size_t dim, float eps, DType dt,
              Variant variant, bool blocking) {
  // No oneDNN RMSNorm primitive; every variant resolves to the native path.
  (void)variant;
  sycl::event ev = kernels::rms_norm_sycl(q, x, weight, out, rows, dim, eps, dt);
  if (blocking) ev.wait();
}

void layernorm(sycl::queue& q, const void* x, const void* weight,
               const void* bias, void* out, std::size_t rows, std::size_t dim,
               float eps, DType dt, Variant variant, bool blocking) {
  // Data-driven best routing (perf/optimization_status.md 2026-07-06, B60):
  // native SYCL wins f32 layernorm (387 vs 244 GB/s); oneDNN wins the 16-bit
  // dtypes (bf16 333 vs 196). No universal winner -- route per dtype.
  if (variant == Variant::best) {
    variant = (dt == DType::f32) ? Variant::sycl : Variant::vendor;
  }
  const Variant v = resolve_variant(variant);
  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::layernorm_onednn(q, x, weight, bias, out, rows, dim, eps, dt);
      break;
#endif
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::layernorm_sycl(q, x, weight, bias, out, rows, dim, eps, dt);
      break;
  }
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
