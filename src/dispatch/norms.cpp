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

void fused_add_rms_norm(sycl::queue &q, const void *x, void *residual, const void *weight,
                        void *out, std::size_t rows, std::size_t dim, float eps, DType dt,
                        Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev =
      kernels::fused_add_rms_norm_sycl(q, x, residual, weight, out, rows, dim, eps, dt);
  if (blocking)
    ev.wait();
}


void rms_residual_next(sycl::queue &q, const void *projection, const void *post_weight,
                       void *residual, const void *next_weight, void *next_out,
                       std::size_t rows, std::size_t dim, float eps, DType dt,
                       Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::rms_residual_next_sycl(
      q, projection, post_weight, residual, next_weight, next_out, rows, dim, eps, dt);
  if (blocking)
    ev.wait();
}

void layernorm(sycl::queue& q, const void* x, const void* weight,
               const void* bias, void* out, std::size_t rows, std::size_t dim,
               float eps, DType dt, Variant variant, bool blocking) {
  // Data-driven best routing (perf/optimization_status.md 2026-07-06, B60).
  // After the 16-byte vector-load pass, native SYCL wins layernorm at ALL dtypes
  // (f32 393 vs 244, bf16 388 vs 333) -- this overturned the pre-vectorization
  // "route bf16 -> vendor" call. best == sycl for every dtype now.
  if (variant == Variant::best) {
    variant = Variant::sycl;
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
