// Rotary position embedding (RoPE), NeoX half-split, native SYCL. One work-item
// per rotated pair (token, head, i<head_dim/2); compute in fp32.

#include "attention/rope/rope_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

template <typename T>
sycl::event rope_typed(sycl::queue& q, const T* x, T* out, std::size_t tokens,
                       std::size_t n_heads, std::size_t head_dim, float base,
                       std::size_t pos0) {
  const std::size_t half = head_dim / 2;
  const std::size_t total = tokens * n_heads * half;
  const float inv_hd = 1.0f / static_cast<float>(head_dim);
  return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    const std::size_t id = idx[0];
    const std::size_t i = id % half;
    const std::size_t hh = (id / half) % n_heads;
    const std::size_t t = id / (half * n_heads);
    const std::size_t bufi = (t * n_heads + hh) * head_dim;

    const float pos = static_cast<float>(pos0 + t);
    const float freq = sycl::pow(base, -2.0f * static_cast<float>(i) * inv_hd);
    const float angle = pos * freq;
    const float c = sycl::cos(angle);
    const float s = sycl::sin(angle);

    const float x1 = static_cast<float>(x[bufi + i]);
    const float x2 = static_cast<float>(x[bufi + i + half]);
    out[bufi + i] = static_cast<T>(x1 * c - x2 * s);
    out[bufi + i + half] = static_cast<T>(x1 * s + x2 * c);
  });
}

}  // namespace

sycl::event rope_sycl(sycl::queue& q, const void* x, void* out,
                      std::size_t tokens, std::size_t n_heads,
                      std::size_t head_dim, float base, std::size_t pos0,
                      DType dt) {
  switch (dt) {
    case DType::f32:
      return rope_typed(q, static_cast<const float*>(x), static_cast<float*>(out),
                        tokens, n_heads, head_dim, base, pos0);
    case DType::f16:
      return rope_typed(q, static_cast<const half_t*>(x), static_cast<half_t*>(out),
                        tokens, n_heads, head_dim, base, pos0);
    case DType::bf16:
      return rope_typed(q, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(out),
                        tokens, n_heads, head_dim, base, pos0);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
