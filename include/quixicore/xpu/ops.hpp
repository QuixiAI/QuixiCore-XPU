#pragma once

// QuixiCore XPU op ABI.
//
// This is the framework-agnostic launch surface for the backend: every op is a
// free function that takes a sycl::queue plus raw device pointers and shape
// metadata. It is the XPU analogue of the Metal backend's `tk::launch_*` layer.
//
// Both callers use the SAME entry points:
//   * the native C++ helpers (which allocate USM and manage a queue), and
//   * the PyTorch-XPU binding (which passes a Torch tensor's data pointer and
//     the tensor's own sycl::queue).
//
// Contract names and semantics are shared with the other QuixiCore backends;
// Intel-specific layout / subgroup / XMX choices stay inside the kernel
// variants under kernels/<family>/<operation>/variants/.
//
// Requires a SYCL toolchain; only compiled under QUIXICORE_XPU_ENABLE_SYCL.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"
#include "quixicore/xpu/variants.hpp"

namespace quixicore::xpu::ops {

// ----------------------------------------------------------------------------
// activations
// ----------------------------------------------------------------------------

// GELU approximation selector. `erf` is the exact Gaussian error function form
// 0.5*x*(1+erf(x/sqrt2)); `tanh` is the tanh approximation used by many LLMs.
enum class GeluApprox {
  erf,
  tanh,
};

// Elementwise GELU over `n` contiguous elements. `in` and `out` are device
// pointers of dtype `dt` (may alias for in-place). Computes in fp32.
//
// `variant` selects the native SYCL or vendor (oneDNN) implementation; both are
// shipped and produce results within the same contract tolerance. When
// `blocking` is true the call waits for completion; otherwise the caller owns
// synchronization.
void gelu(sycl::queue& q, const void* in, void* out, std::size_t n, DType dt,
          GeluApprox approx = GeluApprox::erf, Variant variant = Variant::sycl,
          bool blocking = true);

// Numerically stable softmax over the last axis of a [rows, dim] row-major
// tensor: subtract the row max, exponentiate, normalize by the row sum. `x`,
// `out` are device pointers of dtype `dt` ([rows*dim]); exp/sum accumulate in
// fp32.
void softmax(sycl::queue& q, const void* x, void* out, std::size_t rows,
             std::size_t dim, DType dt, Variant variant = Variant::sycl,
             bool blocking = true);

// SiLU / swish: out[i] = x[i] * sigmoid(x[i]). Elementwise over `n`.
void silu(sycl::queue& q, const void* in, void* out, std::size_t n, DType dt,
          Variant variant = Variant::sycl, bool blocking = true);

// GELU backward: grad_in[i] = grad_out[i] * gelu'(x[i]). `approx` selects the
// erf or tanh derivative to match the forward. Elementwise over `n`.
void gelu_backward(sycl::queue& q, const void* grad_out, const void* x,
                   void* grad_in, std::size_t n, DType dt,
                   GeluApprox approx = GeluApprox::erf,
                   Variant variant = Variant::sycl, bool blocking = true);

// Gated linear unit variants. Input `x` is [rows, 2*d] row-major (gate half then
// value half); output is [rows, d]: out[r,i] = act(x[r,i]) * x[r,d+i], where act
// is silu (swiglu), gelu (geglu), relu (reglu), or sigmoid (glu).
enum class GluMode {
  swiglu,
  geglu,
  reglu,
  glu,
};

void glu(sycl::queue& q, const void* x, void* out, std::size_t rows,
         std::size_t d, DType dt, GluMode mode = GluMode::swiglu,
         Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// norms
// ----------------------------------------------------------------------------

// RMSNorm over the last axis of a [rows, dim] row-major tensor:
//   out[r, i] = x[r, i] * rsqrt(mean_i(x[r, :]^2) + eps) * weight[i]
// `x`, `out` are device pointers of dtype `dt` ([rows*dim]); `weight` is [dim]
// of dtype `dt`. Reduction accumulates in fp32. Deterministic family.
void rms_norm(sycl::queue& q, const void* x, const void* weight, void* out,
              std::size_t rows, std::size_t dim, float eps, DType dt,
              Variant variant = Variant::sycl, bool blocking = true);

// LayerNorm over the last axis of a [rows, dim] row-major tensor:
//   out[r, i] = (x[r, i] - mean) * rsqrt(var + eps) * weight[i] + bias[i]
// with mean/var over x[r, :]. `bias` may be null to skip the shift. `weight`,
// `bias` are [dim] of dtype `dt`. Reduction accumulates in fp32.
void layernorm(sycl::queue& q, const void* x, const void* weight,
               const void* bias, void* out, std::size_t rows, std::size_t dim,
               float eps, DType dt, Variant variant = Variant::sycl,
               bool blocking = true);

}  // namespace quixicore::xpu::ops
