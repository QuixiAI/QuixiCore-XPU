// Greedy argmax over the last axis, native SYCL. One work-group per row; each
// thread scans a strided slice for its local (max, idx), then an SLM reduction
// picks the row winner (lowest index on ties).

#include "sampling/argmax/argmax_kernel.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

template <typename T>
sycl::event argmax_typed(sycl::queue& q, const T* logits, int* out,
                         std::size_t rows, std::size_t vocab) {
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> svals(sycl::range<1>(kRowThreads), h);
    sycl::local_accessor<int, 1> sidx(sycl::range<1>(kRowThreads), h);
    h.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      const std::size_t row = it.get_group(0);
      const std::size_t lid = it.get_local_id(0);
      const T* lr = logits + row * vocab;

      float best = -std::numeric_limits<float>::infinity();
      int best_i = 0;
      for (std::size_t j = lid; j < vocab; j += kRowThreads) {
        const float val = static_cast<float>(lr[j]);
        if (val > best) {
          best = val;
          best_i = static_cast<int>(j);
        }
      }
      svals[lid] = best;
      sidx[lid] = best_i;
      it.barrier(sycl::access::fence_space::local_space);

      if (lid == 0) {
        float m = svals[0];
        int mi = sidx[0];
        for (std::size_t k = 1; k < kRowThreads; ++k) {
          if (svals[k] > m || (svals[k] == m && sidx[k] < mi)) {
            m = svals[k];
            mi = sidx[k];
          }
        }
        out[row] = mi;
      }
    });
  });
}

}  // namespace

sycl::event argmax_sycl(sycl::queue& q, const void* logits, int* out,
                        std::size_t rows, std::size_t vocab, DType dt) {
  switch (dt) {
    case DType::f32:
      return argmax_typed(q, static_cast<const float*>(logits), out, rows, vocab);
    case DType::f16:
      return argmax_typed(q, static_cast<const half_t*>(logits), out, rows, vocab);
    case DType::bf16:
      return argmax_typed(q, static_cast<const bf16_t*>(logits), out, rows, vocab);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
