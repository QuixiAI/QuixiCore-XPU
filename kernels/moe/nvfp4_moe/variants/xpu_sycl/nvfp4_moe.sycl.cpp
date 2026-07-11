// Decode-oriented fused and split routed MoE for ModelOpt NVFP4 weights.

#include "moe/nvfp4_moe/nvfp4_moe_kernel.hpp"

#include <cstdint>
#include <vector>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kSubgroups = 8;
constexpr int kWG = kSG * kSubgroups;
using WVec = sycl::vec<std::uint32_t, 4>;

inline sycl::vec<float, 2> decode_e2m1_pair(std::uint32_t word) {
  const std::uint32_t nibbles = word & 0x000f000fu;
  const std::uint32_t halves = ((nibbles & 0x00080008u) << 12) | ((nibbles & 0x00070007u) << 9);
  return sycl::bit_cast<sycl::vec<sycl::half, 2>>(halves).template convert<float>();
}

inline float decode_e4m3_raw(std::uint8_t value) {
  const auto bits = static_cast<std::uint16_t>(((value & 0x80u) << 8) | ((value & 0x7fu) << 7));
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

template <typename T>
inline float nvfp4_row_dot(const sycl::sub_group &subgroup, const std::uint8_t *weight,
                           const std::uint8_t *block_scales, float global_scale, const T *input,
                           std::size_t K) {
  const WVec *packed = reinterpret_cast<const WVec *>(weight);
  const int lane = static_cast<int>(subgroup.get_local_linear_id());
  float partial = 0.0f;
  for (std::size_t chunk = lane; chunk < K / 32; chunk += kSG) {
    const WVec values = packed[chunk];
    const float scale0 = decode_e4m3_raw(block_scales[2 * chunk]) * global_scale;
    const float scale1 = decode_e4m3_raw(block_scales[2 * chunk + 1]) * global_scale;
    const T *x = input + chunk * 32;
    float words[4];
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      float low = 0.0f;
      float high = 0.0f;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const sycl::vec<float, 2> decoded = decode_e2m1_pair(values[wi] >> (4 * pair));
        low += decoded[0] * static_cast<float>(x[wi * 8 + pair]);
        high += decoded[1] * static_cast<float>(x[wi * 8 + pair + 4]);
      }
      words[wi] = low + high;
    }
    partial += (words[0] + words[1]) * scale0 + (words[2] + words[3]) * scale1;
  }
  return sycl::reduce_over_group(subgroup, partial, sycl::plus<float>());
}

inline float silu(float value) {
  return value / (1.0f + sycl::exp(-value));
}

template <typename T> class Nvfp4MoeFusedKernel;

template <typename T>
sycl::event
fused_typed(sycl::queue &q, const T *hidden, const int *topk_ids, const float *topk_weights,
            const std::uint8_t *w13, const std::uint8_t *w13_scales, const float *w13_global_scales,
            const std::uint8_t *w2, const std::uint8_t *w2_scales, const float *w2_global_scales,
            float *output, std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
            std::size_t I, bool multiply_router_weight, const sycl::event &output_ready) {
  const std::size_t two_i = 2 * I;
  const std::size_t w13_expert_stride = two_i * (K / 2);
  const std::size_t s13_expert_stride = two_i * (K / 16);
  const std::size_t w2_expert_stride = K * (I / 2);
  const std::size_t s2_expert_stride = K * (I / 16);
  const sycl::nd_range<1> range(sycl::range<1>(M * top_k * kWG), sycl::range<1>(kWG));

  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(output_ready);
    sycl::local_accessor<float, 1> gate_up(sycl::range<1>(two_i), handler);
    sycl::local_accessor<float, 1> activated(sycl::range<1>(I), handler);
    handler.parallel_for<Nvfp4MoeFusedKernel<T>>(
        range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_group(0);
          const std::size_t token = pair / top_k;
          const int expert = topk_ids[pair];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;

          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const int thread = static_cast<int>(item.get_local_linear_id());
          const std::size_t expert_index = static_cast<std::size_t>(expert);
          const float w13_global = w13_global_scales[expert_index] * 4194304.0f;
          const std::uint8_t *expert_w13 = w13 + expert_index * w13_expert_stride;
          const std::uint8_t *expert_s13 = w13_scales + expert_index * s13_expert_stride;

          for (std::size_t row = subgroup_id; row < two_i; row += kSubgroups) {
            const float value =
                nvfp4_row_dot(subgroup, expert_w13 + row * (K / 2), expert_s13 + row * (K / 16),
                              w13_global, hidden + token * K, K);
            if (lane == 0)
              gate_up[row] = value;
          }
          sycl::group_barrier(item.get_group());

          for (std::size_t i = thread; i < I; i += kWG) {
            activated[i] = silu(gate_up[i]) * gate_up[i + I];
          }
          sycl::group_barrier(item.get_group());

          const float router_weight = multiply_router_weight ? topk_weights[pair] : 1.0f;
          const float w2_global = w2_global_scales[expert_index] * 4194304.0f;
          const std::uint8_t *expert_w2 = w2 + expert_index * w2_expert_stride;
          const std::uint8_t *expert_s2 = w2_scales + expert_index * s2_expert_stride;
          const float *activation = &activated[0];
          for (std::size_t row = subgroup_id; row < K; row += kSubgroups) {
            const float value = nvfp4_row_dot(subgroup, expert_w2 + row * (I / 2),
                                              expert_s2 + row * (I / 16), w2_global, activation, I);
            if (lane == 0) {
              sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  accumulator(output[token * K + row]);
              accumulator.fetch_add(router_weight * value);
            }
          }
        });
  });
}

template <typename T> class Nvfp4MoeGateUpKernel;

class Nvfp4MoeOutputKernel;

template <typename T>
sycl::event gate_up_typed(sycl::queue &q, const T *hidden, const int *topk_ids,
                          const std::uint8_t *w13, const std::uint8_t *w13_scales,
                          const float *w13_global_scales, float *scratch, std::size_t pairs,
                          std::size_t E, std::size_t top_k, std::size_t K, std::size_t I) {
  const std::size_t two_i = 2 * I;
  const std::size_t w13_expert_stride = two_i * (K / 2);
  const std::size_t s13_expert_stride = two_i * (K / 16);
  const std::size_t row_tiles = (two_i + kSubgroups - 1) / kSubgroups;
  const sycl::nd_range<2> range(sycl::range<2>(pairs, row_tiles * kWG), sycl::range<2>(1, kWG));
  return q.parallel_for<Nvfp4MoeGateUpKernel<T>>(
      range, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
        const std::size_t pair = item.get_global_id(0);
        const int expert = topk_ids[pair];
        if (expert < 0 || static_cast<std::size_t>(expert) >= E)
          return;
        const std::size_t token = pair / top_k;
        const sycl::sub_group subgroup = item.get_sub_group();
        const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
        const int lane = static_cast<int>(subgroup.get_local_linear_id());
        const std::size_t row = item.get_group(1) * kSubgroups + subgroup_id;
        if (row >= two_i)
          return;
        const std::size_t expert_index = static_cast<std::size_t>(expert);
        const float global = w13_global_scales[expert_index] * 4194304.0f;
        const float value =
            nvfp4_row_dot(subgroup, w13 + expert_index * w13_expert_stride + row * (K / 2),
                          w13_scales + expert_index * s13_expert_stride + row * (K / 16), global,
                          hidden + token * K, K);
        if (lane == 0)
          scratch[pair * two_i + row] = value;
      });
}

sycl::event output_typed(sycl::queue &q, const int *topk_ids, const float *topk_weights,
                         const std::uint8_t *w2, const std::uint8_t *w2_scales,
                         const float *w2_global_scales, const float *scratch, float *output,
                         std::size_t pairs, std::size_t E, std::size_t top_k, std::size_t K,
                         std::size_t I, bool multiply_router_weight,
                         const sycl::event &gate_up_ready, const sycl::event &output_ready) {
  const std::size_t two_i = 2 * I;
  const std::size_t w2_expert_stride = K * (I / 2);
  const std::size_t s2_expert_stride = K * (I / 16);
  const std::size_t row_tiles = (K + kSubgroups - 1) / kSubgroups;
  const sycl::nd_range<2> range(sycl::range<2>(pairs, row_tiles * kWG), sycl::range<2>(1, kWG));
  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(std::vector<sycl::event>{gate_up_ready, output_ready});
    sycl::local_accessor<float, 1> activated(sycl::range<1>(I), handler);
    handler.parallel_for<Nvfp4MoeOutputKernel>(
        range, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_global_id(0);
          const int expert = topk_ids[pair];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;
          const std::size_t token = pair / top_k;
          const int thread = static_cast<int>(item.get_local_linear_id());
          const float *gate_up = scratch + pair * two_i;
          for (std::size_t i = thread; i < I; i += kWG) {
            activated[i] = silu(gate_up[i]) * gate_up[i + I];
          }
          sycl::group_barrier(item.get_group());

          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const std::size_t row = item.get_group(1) * kSubgroups + subgroup_id;
          if (row >= K)
            return;
          const std::size_t expert_index = static_cast<std::size_t>(expert);
          const float global = w2_global_scales[expert_index] * 4194304.0f;
          const float value =
              nvfp4_row_dot(subgroup, w2 + expert_index * w2_expert_stride + row * (I / 2),
                            w2_scales + expert_index * s2_expert_stride + row * (I / 16), global,
                            &activated[0], I);
          if (lane == 0) {
            const float router_weight = multiply_router_weight ? topk_weights[pair] : 1.0f;
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                accumulator(output[token * K + row]);
            accumulator.fetch_add(router_weight * value);
          }
        });
  });
}

template <typename T>
sycl::event dispatch_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w13, const void *w13_scales,
                           const float *w13_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *out_f32, std::size_t M,
                           std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                           bool multiply_router_weight, const sycl::event &output_ready) {
  return fused_typed(q, static_cast<const T *>(hidden), topk_ids, topk_weights,
                     static_cast<const std::uint8_t *>(w13),
                     static_cast<const std::uint8_t *>(w13_scales), w13_global_scales,
                     static_cast<const std::uint8_t *>(w2),
                     static_cast<const std::uint8_t *>(w2_scales), w2_global_scales, out_f32, M, E,
                     top_k, K, I, multiply_router_weight, output_ready);
}

template <typename T>
sycl::event dispatch_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w13, const void *w13_scales,
                           const float *w13_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *scratch_f32, float *out_f32,
                           std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
                           std::size_t I, bool multiply_router_weight,
                           const sycl::event &output_ready) {
  const std::size_t pairs = M * top_k;
  const sycl::event gate_up_ready = gate_up_typed(
      q, static_cast<const T *>(hidden), topk_ids, static_cast<const std::uint8_t *>(w13),
      static_cast<const std::uint8_t *>(w13_scales), w13_global_scales, scratch_f32, pairs, E,
      top_k, K, I);
  return output_typed(q, topk_ids, topk_weights, static_cast<const std::uint8_t *>(w2),
                      static_cast<const std::uint8_t *>(w2_scales), w2_global_scales, scratch_f32,
                      out_f32, pairs, E, top_k, K, I, multiply_router_weight, gate_up_ready,
                      output_ready);
}

} // namespace

sycl::event nvfp4_moe_fused_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *out_f32, std::size_t M, std::size_t E, std::size_t top_k,
                                 std::size_t K, std::size_t I, bool multiply_router_weight,
                                 DType act_dt, const sycl::event &output_ready) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_fused<float>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                 w13_global_scales, w2, w2_scales, w2_global_scales, out_f32, M, E,
                                 top_k, K, I, multiply_router_weight, output_ready);
  case DType::f16:
    return dispatch_fused<half_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, out_f32, M, E,
                                  top_k, K, I, multiply_router_weight, output_ready);
  case DType::bf16:
    return dispatch_fused<bf16_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, out_f32, M, E,
                                  top_k, K, I, multiply_router_weight, output_ready);
  }
  return {};
}

sycl::event nvfp4_moe_split_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *scratch_f32, float *out_f32, std::size_t M, std::size_t E,
                                 std::size_t top_k, std::size_t K, std::size_t I,
                                 bool multiply_router_weight, DType act_dt,
                                 const sycl::event &output_ready) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_split<float>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                 w13_global_scales, w2, w2_scales, w2_global_scales, scratch_f32,
                                 out_f32, M, E, top_k, K, I, multiply_router_weight, output_ready);
  case DType::f16:
    return dispatch_split<half_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, scratch_f32,
                                  out_f32, M, E, top_k, K, I, multiply_router_weight, output_ready);
  case DType::bf16:
    return dispatch_split<bf16_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, scratch_f32,
                                  out_f32, M, E, top_k, K, I, multiply_router_weight, output_ready);
  }
  return {};
}

} // namespace quixicore::xpu::kernels
