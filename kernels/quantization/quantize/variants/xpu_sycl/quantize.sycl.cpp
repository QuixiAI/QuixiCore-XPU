// Weight quantization to symmetric group-wise int4, native SYCL. One work-item
// per (row, group): reduce the group |max|, scale = |max|/7, round+clamp each
// element to [-8,7], pack two nibbles per byte. Produces exactly the layout
// qgemv_int4 decodes, so quantize -> qgemv is a lossy round-trip.

#include "quantization/quantize/quantize_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

template <typename T>
sycl::event quant_typed(sycl::queue& q, const T* w, std::uint8_t* wp,
                        half_t* scales, std::size_t N, std::size_t K,
                        std::size_t group) {
  const std::size_t gpr = K / group;         // groups per row
  const std::size_t bpr = K / 2;             // packed bytes per row
  return q.parallel_for(sycl::range<1>(N * gpr), [=](sycl::id<1> idx) {
    const std::size_t t = idx[0];
    const std::size_t n = t / gpr;
    const std::size_t g = t % gpr;
    const T* wr = w + n * K + g * group;

    float amax = 0.0f;
    for (std::size_t j = 0; j < group; ++j)
      amax = sycl::max(amax, sycl::fabs(static_cast<float>(wr[j])));
    const float scale = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
    const float inv = 1.0f / scale;
    scales[n * gpr + g] = static_cast<half_t>(scale);

    // pack the `group` quantized nibbles (group is even) into the row's bytes
    const std::size_t byte0 = n * bpr + (g * group) / 2;
    for (std::size_t j = 0; j < group; j += 2) {
      int q0 = static_cast<int>(sycl::round(static_cast<float>(wr[j]) * inv));
      int q1 = static_cast<int>(sycl::round(static_cast<float>(wr[j + 1]) * inv));
      q0 = sycl::max(-8, sycl::min(7, q0));
      q1 = sycl::max(-8, sycl::min(7, q1));
      wp[byte0 + j / 2] = static_cast<std::uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  });
}

}  // namespace

sycl::event quantize_int4_group_sycl(sycl::queue& q, const void* w,
                                     void* w_packed, void* scales, std::size_t N,
                                     std::size_t K, std::size_t group, DType dt) {
  auto* wp = static_cast<std::uint8_t*>(w_packed);
  auto* sc = static_cast<half_t*>(scales);
  switch (dt) {
    case DType::f32:
      return quant_typed(q, static_cast<const float*>(w), wp, sc, N, K, group);
    case DType::f16:
      return quant_typed(q, static_cast<const half_t*>(w), wp, sc, N, K, group);
    case DType::bf16:
      return quant_typed(q, static_cast<const bf16_t*>(w), wp, sc, N, K, group);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
