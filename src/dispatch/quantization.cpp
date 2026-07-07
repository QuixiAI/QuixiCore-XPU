// Dispatch layer for the quantization family.

#include <stdexcept>

#include "quixicore/xpu/ops.hpp"

#include "quantization/fp8_gemm/fp8_kernel.hpp"
#include "quantization/mxfp4_gemv/mxfp4_kernel.hpp"
#include "quantization/qgemm/qgemm_kernel.hpp"
#include "quantization/qgemv/qgemv_kernel.hpp"

namespace quixicore::xpu::ops {

void mxfp4_gemv(sycl::queue& q, const void* w_packed, const void* block_scales,
                const void* x, void* y, std::size_t N, std::size_t K,
                DType act_dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev =
      kernels::mxfp4_gemv_sycl(q, w_packed, block_scales, x, y, N, K, act_dt);
  if (blocking) ev.wait();
}

void fp8_gemm(sycl::queue& q, const void* a_fp8, const void* b_fp8, void* c,
              std::size_t M, std::size_t N, std::size_t K, Fp8Kind kind,
              float scale, DType out_dt, Variant variant, bool blocking) {
  (void)variant;  // vendor-only
  (void)blocking;  // oneDNN path waits internally
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
  const bool ok = kernels::fp8_gemm_onednn(q, a_fp8, b_fp8, c, M, N, K,
                                           static_cast<int>(kind), scale, out_dt);
  if (!ok)
    throw std::runtime_error(
        "QuixiCore XPU: fp8 matmul is not supported by this oneDNN/device build.");
#else
  (void)a_fp8; (void)b_fp8; (void)c; (void)M; (void)N; (void)K; (void)kind;
  (void)scale; (void)out_dt;
  throw std::runtime_error("QuixiCore XPU: fp8_gemm requires the oneDNN vendor build.");
#endif
}

void fp8_encode(sycl::queue& q, const float* in, void* out_fp8, std::size_t n,
                Fp8Kind kind, bool blocking) {
  (void)blocking;
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
  kernels::fp8_from_f32(q, in, out_fp8, n, static_cast<int>(kind));
#else
  (void)q; (void)in; (void)out_fp8; (void)n; (void)kind;
  throw std::runtime_error("QuixiCore XPU: fp8_encode requires the oneDNN vendor build.");
#endif
}

void fp8_decode(sycl::queue& q, const void* in_fp8, float* out, std::size_t n,
                Fp8Kind kind, bool blocking) {
  (void)blocking;
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
  kernels::fp8_to_f32(q, in_fp8, out, n, static_cast<int>(kind));
#else
  (void)q; (void)in_fp8; (void)out; (void)n; (void)kind;
  throw std::runtime_error("QuixiCore XPU: fp8_decode requires the oneDNN vendor build.");
#endif
}

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
