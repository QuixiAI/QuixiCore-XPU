#pragma once

// Internal entry points for the serving family (embedding lookup, KV-cache
// scatter/gather). All are indexed row copies; dtype-agnostic (copied by byte
// width) so one kernel covers f32/f16/bf16.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event embedding_lookup_sycl(sycl::queue& q, const void* table,
                                  const int* ids, void* out, std::size_t n,
                                  std::size_t dim, DType dt);

sycl::event kv_cache_scatter_sycl(sycl::queue& q, void* cache, const void* src,
                                  const int* slots, std::size_t n,
                                  std::size_t row, DType dt);

sycl::event kv_cache_gather_sycl(sycl::queue& q, const void* cache,
                                 const int* idx, void* out, std::size_t n,
                                 std::size_t row, DType dt);

}  // namespace quixicore::xpu::kernels
