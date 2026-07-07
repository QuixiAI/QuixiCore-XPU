// Serving family: embedding lookup + KV-cache scatter/gather. All are indexed
// row copies; the payload is copied by element width (uint16 for f16/bf16,
// uint32 for f32), so one templated kernel covers every dtype. 2D launch
// [rows, row_width] keeps the inner copy coalesced.

#include "serving/serving_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

// index_src=true: gather, dst[i,j] = src[index[i], j].
// index_src=false: scatter, dst[index[i], j] = src[i, j] (negative index skips).
template <typename W, bool INDEX_SRC>
sycl::event indexed_copy(sycl::queue& q, const W* src, W* dst, const int* index,
                         std::size_t n, std::size_t row) {
  return q.parallel_for(sycl::range<2>(n, row), [=](sycl::id<2> id) {
    const std::size_t i = id[0];
    const std::size_t j = id[1];
    const int idx = index[i];
    if (idx < 0) return;  // only meaningful for scatter; harmless for gather
    if (INDEX_SRC)
      dst[i * row + j] = src[static_cast<std::size_t>(idx) * row + j];
    else
      dst[static_cast<std::size_t>(idx) * row + j] = src[i * row + j];
  });
}

template <bool INDEX_SRC>
sycl::event by_width(sycl::queue& q, const void* src, void* dst, const int* index,
                     std::size_t n, std::size_t row, DType dt) {
  if (dtype_size(dt) == 4)
    return indexed_copy<std::uint32_t, INDEX_SRC>(
        q, static_cast<const std::uint32_t*>(src), static_cast<std::uint32_t*>(dst),
        index, n, row);
  return indexed_copy<std::uint16_t, INDEX_SRC>(
      q, static_cast<const std::uint16_t*>(src), static_cast<std::uint16_t*>(dst),
      index, n, row);
}

}  // namespace

sycl::event embedding_lookup_sycl(sycl::queue& q, const void* table,
                                  const int* ids, void* out, std::size_t n,
                                  std::size_t dim, DType dt) {
  return by_width</*INDEX_SRC=*/true>(q, table, out, ids, n, dim, dt);
}

sycl::event kv_cache_scatter_sycl(sycl::queue& q, void* cache, const void* src,
                                  const int* slots, std::size_t n,
                                  std::size_t row, DType dt) {
  return by_width</*INDEX_SRC=*/false>(q, src, cache, slots, n, row, dt);
}

sycl::event kv_cache_gather_sycl(sycl::queue& q, const void* cache,
                                 const int* idx, void* out, std::size_t n,
                                 std::size_t row, DType dt) {
  return by_width</*INDEX_SRC=*/true>(q, cache, out, idx, n, row, dt);
}

}  // namespace quixicore::xpu::kernels
