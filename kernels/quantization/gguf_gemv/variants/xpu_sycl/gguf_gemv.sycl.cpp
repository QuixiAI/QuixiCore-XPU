// GGUF q8_0 / q4_0 GEMV, native SYCL decoder from the authentic llama.cpp block
// layout: each 32-element block is { fp16 d; quants } stored consecutively along
// a row. q8_0: 32 int8 (w = int8*d). q4_0: 16 packed bytes, low nibbles are
// elements 0..15 and high nibbles 16..31, w = (nibble-8)*d.
//
// One 32-wide subgroup per row; each lane decodes whole blocks (strided). The
// interleaved on-disk layout is not GPU-coalescing-friendly (odd 34/18-byte
// strides), so this is correctness-first; a repack to a GPU layout is the
// optimization. GGUF k-quants are data encodings -> decode natively on Intel.

#include "quantization/gguf_gemv/gguf_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kRowsPerWG = 8;
constexpr int kWG = kSG * kRowsPerWG;

inline float load_half(const std::uint8_t* p) {
  const std::uint16_t bits = static_cast<std::uint16_t>(p[0]) |
                             (static_cast<std::uint16_t>(p[1]) << 8);
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

template <typename T, int TYPE>
sycl::event gguf_typed(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                       std::size_t N, std::size_t K) {
  constexpr int BLOCK_BYTES = (TYPE == 0) ? 34 : 18;  // q8_0 : q4_0
  const std::size_t blocks_per_row = K / 32;
  const std::size_t row_bytes = blocks_per_row * BLOCK_BYTES;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < blocks_per_row; b += kSG) {
      const std::uint8_t* blk = wrow + b * BLOCK_BYTES;
      const float d = load_half(blk);
      const std::uint8_t* qs = blk + 2;
      const std::size_t kbase = b * 32;
      if (TYPE == 0) {  // q8_0
#pragma unroll
        for (int i = 0; i < 32; ++i)
          acc += static_cast<float>(static_cast<std::int8_t>(qs[i])) * d *
                 static_cast<float>(x[kbase + i]);
      } else {  // q4_0
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int lo = (qs[i] & 0xF) - 8;
          const int hi = (qs[i] >> 4) - 8;
          acc += static_cast<float>(lo) * d * static_cast<float>(x[kbase + i]);
          acc += static_cast<float>(hi) * d * static_cast<float>(x[kbase + 16 + i]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q6_K: 256-element super-block (210 bytes): ql[128], qh[64], int8
// scales[16], fp16 d. 6-bit quant = 4 low bits (ql) + 2 high bits (qh),
// recentred by -32. Follows ggml dequantize_row_q6_K exactly. One 32-wide
// subgroup per row; each lane decodes whole super-blocks.
template <typename T>
sycl::event gguf_q6k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 210;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 210;
      const std::uint8_t* ql = blk;
      const std::uint8_t* qh = blk + 128;
      const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
      const float d = load_half(blk + 208);
      const std::size_t kbase = b * 256;
      for (int half = 0; half < 2; ++half) {
        const std::uint8_t* qlh = ql + half * 64;
        const std::uint8_t* qhh = qh + half * 32;
        const std::int8_t* sch = sc + half * 8;
        const std::size_t yoff = kbase + static_cast<std::size_t>(half) * 128;
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int q1 = static_cast<int>((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
          const int q2 = static_cast<int>((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
          const int q3 = static_cast<int>((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
          const int q4 = static_cast<int>((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
          acc += d * static_cast<float>(sch[is + 0]) * q1 * static_cast<float>(x[yoff + l + 0]);
          acc += d * static_cast<float>(sch[is + 2]) * q2 * static_cast<float>(x[yoff + l + 32]);
          acc += d * static_cast<float>(sch[is + 4]) * q3 * static_cast<float>(x[yoff + l + 64]);
          acc += d * static_cast<float>(sch[is + 6]) * q4 * static_cast<float>(x[yoff + l + 96]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// ggml get_scale_min_k4: unpack the j-th 6-bit sub-scale (sc) and sub-min (m)
// from the 12-byte packed scales array of q4_K/q5_K.
inline void scale_min_k4(int j, const std::uint8_t* s, int& sc, int& m) {
  if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
  else {
    sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

// GGUF q4_K: 256-element super-block (144 bytes): fp16 d, fp16 dmin,
// scales[12] (8 packed 6-bit scale+min pairs), qs[128]. 8 sub-blocks of 32,
// w = d*sc*(q&0xF or q>>4) - dmin*m. Follows ggml dequantize_row_q4_K.
template <typename T>
sycl::event gguf_q4k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 144;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 144;
      const float d = load_half(blk);
      const float dmin = load_half(blk + 2);
      const std::uint8_t* sc = blk + 4;
      const std::uint8_t* qs = blk + 16;
      const std::size_t kbase = b * 256;
      for (int iter = 0; iter < 4; ++iter) {
        int s1, m1i, s2, m2i;
        scale_min_k4(iter * 2 + 0, sc, s1, m1i);
        scale_min_k4(iter * 2 + 1, sc, s2, m2i);
        const float d1 = d * s1, mm1 = dmin * m1i;
        const float d2 = d * s2, mm2 = dmin * m2i;
        const std::uint8_t* qq = qs + iter * 32;
        const std::size_t yb = kbase + iter * 64;
        for (int l = 0; l < 32; ++l) {
          acc += (d1 * (qq[l] & 0xF) - mm1) * static_cast<float>(x[yb + l]);
          acc += (d2 * (qq[l] >> 4) - mm2) * static_cast<float>(x[yb + 32 + l]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q5_K: 176-byte super-block (fp16 d + dmin, scales[12], qh[32], qs[128]).
// Like q4_K but each 4-bit quant gains a 5th bit from qh (bit shifts by 2 per
// 64-block). Follows ggml dequantize_row_q5_K.
template <typename T>
sycl::event gguf_q5k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 176;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 176;
      const float d = load_half(blk);
      const float dmin = load_half(blk + 2);
      const std::uint8_t* sc = blk + 4;
      const std::uint8_t* qh = blk + 16;
      const std::uint8_t* qs = blk + 48;
      const std::size_t kbase = b * 256;
      for (int iter = 0; iter < 4; ++iter) {
        int s1, m1i, s2, m2i;
        scale_min_k4(iter * 2 + 0, sc, s1, m1i);
        scale_min_k4(iter * 2 + 1, sc, s2, m2i);
        const float d1 = d * s1, mm1 = dmin * m1i, d2 = d * s2, mm2 = dmin * m2i;
        const std::uint8_t* ql = qs + iter * 32;
        const int u1 = 1 << (2 * iter), u2 = 2 << (2 * iter);
        const std::size_t yb = kbase + iter * 64;
        for (int l = 0; l < 32; ++l) {
          const float q1 = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
          const float q2 = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
          acc += (d1 * q1 - mm1) * static_cast<float>(x[yb + l]);
          acc += (d2 * q2 - mm2) * static_cast<float>(x[yb + 32 + l]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q2_K: 84-byte super-block (scales[16], qs[64], fp16 d, fp16 dmin). 16
// sub-blocks of 16, 2-bit quants; each scales byte = 4-bit scale (low) + 4-bit
// min (high). w = d*sc*q - dmin*m. Follows ggml dequantize_row_q2_K.
template <typename T>
sycl::event gguf_q2k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 84;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 84;
      const std::uint8_t* scales = blk;
      const std::uint8_t* qs = blk + 16;
      const float d = load_half(blk + 80);
      const float dmin = load_half(blk + 82);
      const std::size_t kbase = b * 256;
      for (int h = 0; h < 2; ++h) {
        const std::uint8_t* qb = qs + h * 32;
        for (int j = 0; j < 4; ++j) {
          const int shift = 2 * j;
          const std::uint8_t sc1 = scales[h * 8 + j * 2];
          const std::uint8_t sc2 = scales[h * 8 + j * 2 + 1];
          const float dl1 = d * (sc1 & 0xF), ml1 = dmin * (sc1 >> 4);
          const float dl2 = d * (sc2 & 0xF), ml2 = dmin * (sc2 >> 4);
          const std::size_t yb = kbase + h * 128 + j * 32;
          for (int l = 0; l < 16; ++l) {
            acc += (dl1 * ((qb[l] >> shift) & 3) - ml1) * static_cast<float>(x[yb + l]);
            acc += (dl2 * ((qb[l + 16] >> shift) & 3) - ml2) * static_cast<float>(x[yb + 16 + l]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q3_K: 110-byte super-block (hmask[32], qs[64], scales[12], fp16 d). 16
// sub-blocks of 16, 3-bit quant = 2 low bits (qs) + 1 inverted high bit (hmask);
// 6-bit scales packed via a custom kmask bit-shuffle. Follows ggml
// dequantize_row_q3_K exactly (the fiddliest k-quant).
template <typename T>
sycl::event gguf_q3k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 110;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    constexpr std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 110;
      const std::uint8_t* hm = blk;
      const std::uint8_t* qs = blk + 32;
      const std::uint8_t* sc = blk + 96;
      const float d = load_half(blk + 108);
      // unpack the 16 int8 scales (ggml kmask shuffle)
      auto rd32 = [&](int o) {
        return static_cast<std::uint32_t>(sc[o]) | (static_cast<std::uint32_t>(sc[o + 1]) << 8) |
               (static_cast<std::uint32_t>(sc[o + 2]) << 16) | (static_cast<std::uint32_t>(sc[o + 3]) << 24);
      };
      const std::uint32_t a0 = rd32(0), a1 = rd32(4), tmp = rd32(8);
      std::uint32_t aux[4];
      aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
      aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
      aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
      auto scl = [&](int i) { return static_cast<int>(static_cast<std::int8_t>((aux[i >> 2] >> (8 * (i & 3))) & 0xFF)); };

      const std::size_t kbase = b * 256;
      for (int h = 0; h < 2; ++h) {
        const std::uint8_t* qb = qs + h * 32;
        for (int j = 0; j < 4; ++j) {
          const int shift = 2 * j;
          const int is = h * 8 + j * 2;
          const std::uint8_t mbit = static_cast<std::uint8_t>(1u << (h * 4 + j));
          const float dl1 = d * (scl(is) - 32), dl2 = d * (scl(is + 1) - 32);
          const std::size_t yb = kbase + h * 128 + j * 32;
          for (int l = 0; l < 16; ++l) {
            const int v1 = static_cast<int>((qb[l] >> shift) & 3) - ((hm[l] & mbit) ? 0 : 4);
            const int v2 = static_cast<int>((qb[l + 16] >> shift) & 3) - ((hm[l + 16] & mbit) ? 0 : 4);
            acc += dl1 * v1 * static_cast<float>(x[yb + l]);
            acc += dl2 * v2 * static_cast<float>(x[yb + 16 + l]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF IQ4_NL: q4_0 footprint (18-byte block: fp16 d + qs[16]) but the 4-bit
// index maps through a fixed 16-entry non-linear codebook instead of a linear
// scale. QK=32, w = d * kvalues_iq4nl[nibble]. Follows ggml dequantize_row_iq4_nl.
constexpr int kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                            1, 13, 25, 38, 53, 69, 89, 113};

template <typename T>
sycl::event gguf_iq4nl_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                             T* y, std::size_t N, std::size_t K) {
  const std::size_t blocks_per_row = K / 32;
  const std::size_t row_bytes = blocks_per_row * 18;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < blocks_per_row; b += kSG) {
      const std::uint8_t* blk = wrow + b * 18;
      const float d = load_half(blk);
      const std::uint8_t* qs = blk + 2;
      const std::size_t kbase = b * 32;
      for (int j = 0; j < 16; ++j) {
        acc += d * kIQ4NL[qs[j] & 0xF] * static_cast<float>(x[kbase + j]);
        acc += d * kIQ4NL[qs[j] >> 4] * static_cast<float>(x[kbase + 16 + j]);
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

template <typename T>
sycl::event dispatch_type(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                          std::size_t N, std::size_t K, int type) {
  if (type == 7) return gguf_iq4nl_typed<T>(q, w, x, y, N, K);
  if (type == 2) return gguf_q6k_typed<T>(q, w, x, y, N, K);
  if (type == 3) return gguf_q4k_typed<T>(q, w, x, y, N, K);
  if (type == 4) return gguf_q5k_typed<T>(q, w, x, y, N, K);
  if (type == 5) return gguf_q2k_typed<T>(q, w, x, y, N, K);
  if (type == 6) return gguf_q3k_typed<T>(q, w, x, y, N, K);
  return type == 0 ? gguf_typed<T, 0>(q, w, x, y, N, K)
                   : gguf_typed<T, 1>(q, w, x, y, N, K);
}

}  // namespace

sycl::event gguf_gemv_sycl(sycl::queue& q, const void* w_blocks, const void* x,
                           void* y, std::size_t N, std::size_t K, int type,
                           DType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_blocks);
  switch (act_dt) {
    case DType::f32:
      return dispatch_type(q, w, static_cast<const float*>(x), static_cast<float*>(y), N, K, type);
    case DType::f16:
      return dispatch_type(q, w, static_cast<const half_t*>(x), static_cast<half_t*>(y), N, K, type);
    case DType::bf16:
      return dispatch_type(q, w, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(y), N, K, type);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
