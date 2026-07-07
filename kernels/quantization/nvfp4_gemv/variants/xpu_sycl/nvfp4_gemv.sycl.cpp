// nvfp4 GEMV, native SYCL decoder. e2m1 (fp4) elements + e4m3 (fp8) block scale
// per 16 + a per-tensor fp32 global scale. Same subgroup-per-row / 16-byte-load
// structure as mxfp4; a 16-byte chunk (32 fp4) spans TWO 16-element blocks, so
// two e4m3 scales per chunk. Both the fp4 magnitudes and the e4m3 scale decode
// are fixed tables/bit-math -- nvfp4 is a data encoding, decoded natively here.

#include "quantization/nvfp4_gemv/nvfp4_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kRowsPerWG = 8;
constexpr int kWG = kSG * kRowsPerWG;

constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

using WVec = sycl::vec<std::uint32_t, 4>;  // 16 bytes = 32 fp4 = two nvfp4 blocks

// Decode an e4m3 (fp8) byte to float (bias 7, 1-4-3, subnormals supported).
inline float e4m3_to_float(std::uint8_t b) {
  const int sign = (b >> 7) & 1;
  const int exp = (b >> 3) & 0xF;
  const int mant = b & 0x7;
  float v;
  if (exp == 0)
    v = static_cast<float>(mant) * 0.001953125f;  // 2^-9 subnormal step
  else
    v = sycl::ldexp(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
  return sign ? -v : v;
}

template <typename T>
sycl::event nvfp4_typed(sycl::queue& q, const std::uint8_t* w,
                        const std::uint8_t* bscale, float gscale, const T* x,
                        T* y, std::size_t N, std::size_t K) {
  const std::size_t bytes_per_row = K / 2;
  const std::size_t blocks_per_row = K / 16;     // e4m3 scales per row
  const std::size_t nchunks = bytes_per_row / 16;  // 16-byte chunks (2 blocks each)
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;

    const WVec* wvrow = reinterpret_cast<const WVec*>(w + n * bytes_per_row);
    const std::uint8_t* srow = bscale + n * blocks_per_row;

    // Decode is ALU/latency-bound (was 17% of BW roofline). Hoist the two block
    // scales out of the nibble loop (one mul per 16-elem block, not per nibble);
    // vectorize the activation read; accumulate each word independently for ILP.
    float acc = 0.0f;
    for (std::size_t c = lane; c < nchunks; c += kSG) {
      const WVec chunk = wvrow[c];
      // Two 16-element blocks in this chunk -> two e4m3 scales.
      const float s0 = e4m3_to_float(srow[2 * c]) * gscale;
      const float s1 = e4m3_to_float(srow[2 * c + 1]) * gscale;
      const T* xc = x + c * 32;
      float aw[4];
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        const std::uint32_t word = chunk[wi];
        const sycl::vec<T, 8> xv = *reinterpret_cast<const sycl::vec<T, 8>*>(xc + wi * 8);
        float a = 0.0f;
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          const int n4 = (word >> (nib * 4)) & 0xF;
          float val = kE2M1[n4 & 7];
          if (n4 & 8) val = -val;
          a += val * static_cast<float>(xv[nib]);
        }
        aw[wi] = a;
      }
      acc += (aw[0] + aw[1]) * s0 + (aw[2] + aw[3]) * s1;
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

}  // namespace

sycl::event nvfp4_gemv_sycl(sycl::queue& q, const void* w_packed,
                            const void* block_scales, float global_scale,
                            const void* x, void* y, std::size_t N, std::size_t K,
                            DType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_packed);
  const auto* s = static_cast<const std::uint8_t*>(block_scales);
  switch (act_dt) {
    case DType::f32:
      return nvfp4_typed(q, w, s, global_scale, static_cast<const float*>(x), static_cast<float*>(y), N, K);
    case DType::f16:
      return nvfp4_typed(q, w, s, global_scale, static_cast<const half_t*>(x), static_cast<half_t*>(y), N, K);
    case DType::bf16:
      return nvfp4_typed(q, w, s, global_scale, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(y), N, K);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
