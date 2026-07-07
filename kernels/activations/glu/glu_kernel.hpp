#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// mode: 0 swiglu (silu gate), 1 geglu (gelu gate), 2 reglu (relu gate),
//       3 glu (sigmoid gate). Input [rows, 2*d], output [rows, d].
sycl::event glu_sycl(sycl::queue& q, const void* x, void* out, std::size_t rows,
                     std::size_t d, DType dt, int mode);

}  // namespace quixicore::xpu::kernels
