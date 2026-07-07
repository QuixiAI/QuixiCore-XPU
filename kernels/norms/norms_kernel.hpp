#pragma once

// Internal variant entry points for the norms family (rms_norm, layernorm).
// Included by the dispatch layer and the benchmark harness via the ops target's
// private kernels/ include path. Each entry submits and returns its sycl::event
// without waiting.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Native SYCL. Always available.
sycl::event rms_norm_sycl(sycl::queue& q, const void* x, const void* weight,
                          void* out, std::size_t rows, std::size_t dim,
                          float eps, DType dt);

sycl::event layernorm_sycl(sycl::queue& q, const void* x, const void* weight,
                           const void* bias, void* out, std::size_t rows,
                           std::size_t dim, float eps, DType dt);

#if defined(QUIXICORE_XPU_HAS_ONEDNN)
// Vendor LayerNorm via oneDNN layer_normalization. (oneDNN has no dedicated
// RMSNorm primitive, so rms_norm ships SYCL-only and Variant::vendor falls back
// to SYCL for it.)
sycl::event layernorm_onednn(sycl::queue& q, const void* x, const void* weight,
                             const void* bias, void* out, std::size_t rows,
                             std::size_t dim, float eps, DType dt);
#endif

}  // namespace quixicore::xpu::kernels
