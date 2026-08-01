#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event nvfp4_moe_fused_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *out_f32, std::size_t M, std::size_t E, std::size_t top_k,
                                 std::size_t K, std::size_t I, bool multiply_router_weight,
                                 DType act_dt, const sycl::event &output_ready);

sycl::event nvfp4_moe_split_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *scratch_f32, float *out_f32, std::size_t M, std::size_t E,
                                 std::size_t top_k, std::size_t K, std::size_t I,
                                 bool multiply_router_weight, DType act_dt,
                                 const sycl::event &output_ready);

} // namespace quixicore::xpu::kernels
