// Dispatch layer for activation ops: routes the public ops ABI to the selected
// implementation variant (native SYCL or vendor oneDNN) and applies blocking
// semantics. The XPU analogue of the Metal backend's `tk::launch_*` layer.

#include "quixicore/xpu/ops.hpp"

#include "activations/gelu/gelu_kernel.hpp"
#include "activations/softmax/softmax_kernel.hpp"

namespace quixicore::xpu::ops {

void gelu(sycl::queue& q, const void* in, void* out, std::size_t n, DType dt,
          GeluApprox approx, Variant variant, bool blocking) {
  const bool tanh_approx = (approx == GeluApprox::tanh);
  const Variant v = resolve_variant(variant);

  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::gelu_onednn(q, in, out, n, dt, tanh_approx);
      break;
#endif
      // falls through to sycl when vendor is not compiled in
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::gelu_sycl(q, in, out, n, dt, tanh_approx);
      break;
  }

  if (blocking) {
    ev.wait();
  }
}

void softmax(sycl::queue& q, const void* x, void* out, std::size_t rows,
             std::size_t dim, DType dt, Variant variant, bool blocking) {
  // Data-driven best routing (perf/optimization_status.md 2026-07-06, B60).
  // Native SYCL wins f32 softmax (389 vs 223 GB/s). For bf16, the vector-load
  // pass narrowed the gap (195 -> 302) but oneDNN (348) still wins -- softmax is
  // exp()-bound, unlike layernorm where native now wins bf16 too. Route bf16/f16
  // to vendor.
  if (variant == Variant::best) {
    variant = (dt == DType::f32) ? Variant::sycl : Variant::vendor;
  }
  const Variant v = resolve_variant(variant);
  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::softmax_onednn(q, x, out, rows, dim, dt);
      break;
#endif
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::softmax_sycl(q, x, out, rows, dim, dt);
      break;
  }
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
