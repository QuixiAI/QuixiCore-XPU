// Numerically stable softmax over the last axis, native SYCL variant.
//
// One work-group per row, two fp32 reductions: row max (for stability), then the
// sum of exp(x - max). A final pass writes exp(x - max) / sum. Deterministic
// within a fixed launch geometry.

#include <limits>

#include "activations/softmax/softmax_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

template <typename T>
sycl::event softmax_typed(sycl::queue& q, const T* x, T* out, std::size_t rows,
                          std::size_t dim) {
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t row = it.get_group(0);
    const std::size_t lid = it.get_local_id(0);
    const T* xr = x + row * dim;
    T* outr = out + row * dim;
    const auto grp = it.get_group();

    float pmax = -std::numeric_limits<float>::infinity();
    for (std::size_t i = lid; i < dim; i += kRowThreads) {
      pmax = sycl::max(pmax, static_cast<float>(xr[i]));
    }
    const float row_max =
        sycl::reduce_over_group(grp, pmax, sycl::maximum<float>());

    float psum = 0.0f;
    for (std::size_t i = lid; i < dim; i += kRowThreads) {
      psum += sycl::exp(static_cast<float>(xr[i]) - row_max);
    }
    const float row_sum =
        sycl::reduce_over_group(grp, psum, sycl::plus<float>());
    const float inv = 1.0f / row_sum;

    for (std::size_t i = lid; i < dim; i += kRowThreads) {
      outr[i] = static_cast<T>(sycl::exp(static_cast<float>(xr[i]) - row_max) * inv);
    }
  });
}

}  // namespace

sycl::event softmax_sycl(sycl::queue& q, const void* x, void* out,
                         std::size_t rows, std::size_t dim, DType dt) {
  switch (dt) {
    case DType::f32:
      return softmax_typed(q, static_cast<const float*>(x),
                           static_cast<float*>(out), rows, dim);
    case DType::f16:
      return softmax_typed(q, static_cast<const half_t*>(x),
                           static_cast<half_t*>(out), rows, dim);
    case DType::bf16:
      return softmax_typed(q, static_cast<const bf16_t*>(x),
                           static_cast<bf16_t*>(out), rows, dim);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
