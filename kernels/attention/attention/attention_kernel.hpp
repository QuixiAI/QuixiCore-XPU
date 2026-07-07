#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event attention_sycl(sycl::queue& q, const void* Q, const void* K,
                           const void* V, void* O, std::size_t n_heads,
                           std::size_t n_kv_heads, std::size_t seq_q,
                           std::size_t seq_k, std::size_t d, bool causal,
                           DType dt);

}  // namespace quixicore::xpu::kernels
