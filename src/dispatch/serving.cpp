// Dispatch layer for the serving family (native only — index copies).

#include "quixicore/xpu/ops.hpp"

#include "serving/serving_kernel.hpp"

namespace quixicore::xpu::ops {

void embedding_lookup(sycl::queue& q, const void* table, const int* ids,
                      void* out, std::size_t n, std::size_t dim, DType dt,
                      Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::embedding_lookup_sycl(q, table, ids, out, n, dim, dt);
  if (blocking) ev.wait();
}

void kv_cache_scatter(sycl::queue& q, void* cache, const void* src,
                      const int* slots, std::size_t n, std::size_t row, DType dt,
                      Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::kv_cache_scatter_sycl(q, cache, src, slots, n, row, dt);
  if (blocking) ev.wait();
}

void kv_cache_gather(sycl::queue& q, const void* cache, const int* idx,
                     void* out, std::size_t n, std::size_t row, DType dt,
                     Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::kv_cache_gather_sycl(q, cache, idx, out, n, row, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
