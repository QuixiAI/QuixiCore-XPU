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

// Same flash SDPA, but the epilogue additionally writes a rounded half copy of
// the output to O_f16 (fused ctx->f16 context store). O is dtype dt; O_f16 is
// always f16. Shape: online-attention + fused f16 context store.
sycl::event attention_f16ctx_sycl(sycl::queue& q, const void* Q, const void* K,
                                  const void* V, void* O, void* O_f16,
                                  std::size_t n_heads, std::size_t n_kv_heads,
                                  std::size_t seq_q, std::size_t seq_k,
                                  std::size_t d, bool causal, DType dt);

}  // namespace quixicore::xpu::kernels
