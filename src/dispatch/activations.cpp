// Dispatch layer for activation ops: routes the public ops ABI to the selected
// implementation variant (native SYCL or vendor oneDNN) and applies blocking
// semantics. The XPU analogue of the Metal backend's `tk::launch_*` layer.

#include "quixicore/xpu/ops.hpp"

#include "activations/gelu/gelu_kernel.hpp"

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

}  // namespace quixicore::xpu::ops
