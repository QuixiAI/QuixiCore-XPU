// RMSNorm over the last axis, native SYCL variant.
//
// One work-group per row; the group cooperatively reduces sum(x^2) via
// sycl::reduce_over_group (fp32 accumulation), then rescales. This is the
// reduction-pattern reference for the backend (contrast the elementwise GELU
// reference). Deterministic within a fixed launch geometry.
//
// Launch geometry (work-group size, rows-per-group, SLM staging) is a perf lever
// recorded in perf/optimization_status.md before any speedup is claimed.

#include "norms/norms_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

// Work-items per row. Each item strides over `dim`, so any dim is handled; 256
// balances occupancy against reduction cost for typical hidden sizes.
constexpr std::size_t kRowThreads = 256;

// Elements per vector load. V*sizeof(T) = 16 bytes (8x bf16/f16, 4x f32) so each
// thread issues one wide load and ADJACENT threads read ADJACENT vectors
// (coalesced). Reinterpreting the row as sycl::vec<T,V> forces the compiler to
// emit the wide load; an unrolled scalar loop instead compiled to V strided
// scalar loads and regressed (perf/optimization_status.md 2026-07-06).
template <typename T>
constexpr int vec_width() {
  return sizeof(T) == 2 ? 8 : 4;
}

template <typename T>
sycl::event rms_norm_typed(sycl::queue& q, const T* x, const T* w, T* out,
                           std::size_t rows, std::size_t dim, float eps) {
  constexpr int V = vec_width<T>();
  using Vec = sycl::vec<T, V>;
  const std::size_t nvec = dim / V;      // whole vectors per row
  const std::size_t tail0 = nvec * V;    // first scalar-tail index
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t row = it.get_group(0);
    const std::size_t lid = it.get_local_id(0);
    const T* xr = x + row * dim;
    const T* wr = w;
    T* outr = out + row * dim;
    const Vec* xv = reinterpret_cast<const Vec*>(xr);

    float partial = 0.0f;
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec v = xv[vi];
#pragma unroll
      for (int k = 0; k < V; ++k) {
        const float f = static_cast<float>(v[k]);
        partial += f * f;
      }
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      const float f = static_cast<float>(xr[i]);
      partial += f * f;
    }
    const float sumsq =
        sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
    const float inv = sycl::rsqrt(sumsq / static_cast<float>(dim) + eps);

    const Vec* wv = reinterpret_cast<const Vec*>(wr);
    Vec* ov = reinterpret_cast<Vec*>(outr);
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec xvec = xv[vi];
      const Vec wvec = wv[vi];
      Vec ovec;
#pragma unroll
      for (int k = 0; k < V; ++k) {
        ovec[k] = static_cast<T>(static_cast<float>(xvec[k]) * inv *
                                 static_cast<float>(wvec[k]));
      }
      ov[vi] = ovec;
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      outr[i] = static_cast<T>(static_cast<float>(xr[i]) * inv *
                               static_cast<float>(wr[i]));
    }
  });
}

}  // namespace

sycl::event rms_norm_sycl(sycl::queue& q, const void* x, const void* weight,
                          void* out, std::size_t rows, std::size_t dim,
                          float eps, DType dt) {
  switch (dt) {
    case DType::f32:
      return rms_norm_typed(q, static_cast<const float*>(x),
                            static_cast<const float*>(weight),
                            static_cast<float*>(out), rows, dim, eps);
    case DType::f16:
      return rms_norm_typed(q, static_cast<const half_t*>(x),
                            static_cast<const half_t*>(weight),
                            static_cast<half_t*>(out), rows, dim, eps);
    case DType::bf16:
      return rms_norm_typed(q, static_cast<const bf16_t*>(x),
                            static_cast<const bf16_t*>(weight),
                            static_cast<bf16_t*>(out), rows, dim, eps);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
