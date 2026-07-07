// GELU activation, native SYCL variant (Battlemage / Xe2 and portable).
//
// Elementwise, memory-bound. This is the reference kernel for the XPU backend:
// it establishes the variant-entry -> dispatch -> ops-ABI pattern that every
// later operation copies. Compute is done in fp32 regardless of storage dtype.
//
// Not blindly ported from Metal: no threadgroup staging, no simdgroup tricks --
// a flat parallel_for is the right shape for an elementwise op on Xe. Launch
// geometry / vectorization tuning is a later perf lever, recorded in
// perf/optimization_status.md before any speedup is claimed.

#include "activations/gelu/gelu_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

// sqrt(2/pi), used by the tanh approximation.
constexpr float kSqrt2OverPi = 0.7978845608028654f;
// 1/sqrt(2), used by the exact erf form.
constexpr float kInvSqrt2 = 0.7071067811865476f;

inline float gelu_scalar(float x, bool tanh_approx) {
  if (tanh_approx) {
    const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + sycl::tanh(inner));
  }
  return 0.5f * x * (1.0f + sycl::erf(x * kInvSqrt2));
}

// Each work-item processes kVec elements with a COALESCED grid stride: for a
// fixed k, adjacent work-items touch adjacent addresses (i = tid + k*threads),
// so each subgroup issues contiguous loads/stores while amortizing launch and
// index overhead over kVec elements. See perf/optimization_status.md 2026-07-06
// for the measured comparison against the naive one-element-per-item mapping and
// the oneDNN baseline.
constexpr int kVec = 4;

template <typename T>
sycl::event gelu_typed(sycl::queue& q, const T* in, T* out, std::size_t n,
                       bool tanh_approx) {
  const std::size_t threads = (n + kVec - 1) / kVec;
  return q.parallel_for(sycl::range<1>(threads), [=](sycl::id<1> idx) {
    const std::size_t tid = idx[0];
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const std::size_t i = tid + static_cast<std::size_t>(k) * threads;
      if (i < n) {
        out[i] = static_cast<T>(gelu_scalar(static_cast<float>(in[i]), tanh_approx));
      }
    }
  });
}

}  // namespace

sycl::event gelu_sycl(sycl::queue& q, const void* in, void* out, std::size_t n,
                      DType dt, bool tanh_approx) {
  switch (dt) {
    case DType::f32:
      return gelu_typed(q, static_cast<const float*>(in),
                        static_cast<float*>(out), n, tanh_approx);
    case DType::f16:
      return gelu_typed(q, static_cast<const half_t*>(in),
                        static_cast<half_t*>(out), n, tanh_approx);
    case DType::bf16:
      return gelu_typed(q, static_cast<const bf16_t*>(in),
                        static_cast<bf16_t*>(out), n, tanh_approx);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
