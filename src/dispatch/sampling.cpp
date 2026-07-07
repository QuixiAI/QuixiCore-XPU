// Dispatch layer for the sampling family.

#include "quixicore/xpu/ops.hpp"

#include "sampling/argmax/argmax_kernel.hpp"
#include "sampling/sample/sample_kernel.hpp"

namespace quixicore::xpu::ops {

void argmax(sycl::queue& q, const void* logits, int* out, std::size_t rows,
            std::size_t vocab, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::argmax_sycl(q, logits, out, rows, vocab, dt);
  if (blocking) ev.wait();
}

void sample_categorical(sycl::queue& q, const void* logits, int* out,
                        std::size_t rows, std::size_t vocab, float temperature,
                        std::uint32_t seed, DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::sample_categorical_sycl(q, logits, out, rows, vocab, temperature, seed, dt);
  if (blocking) ev.wait();
}

void top_k_sample(sycl::queue& q, const void* logits, int* out, std::size_t rows,
                  std::size_t vocab, int k, float temperature, std::uint32_t seed,
                  DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::top_k_sample_sycl(q, logits, out, rows, vocab, k, temperature, seed, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
