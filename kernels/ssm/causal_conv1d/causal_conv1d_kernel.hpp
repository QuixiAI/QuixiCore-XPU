#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Depthwise causal conv1d decode update. Shapes, the kernel<=8 envelope, and
// slot semantics are documented on ops::causal_conv1d_decode.
sycl::event causal_conv1d_decode_sycl(
    sycl::queue& q, void* conv_state, const void* x, const void* weight,
    const void* bias, const std::int32_t* indices, void* out, bool silu,
    std::size_t batch, std::size_t dim, std::size_t state_len,
    std::size_t kernel, std::size_t nslots, std::int64_t cs0, std::int64_t cs1,
    std::int64_t cs2, DType act_dt, DType state_dt);

}  // namespace quixicore::xpu::kernels
