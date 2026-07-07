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
// attention
// ----------------------------------------------------------------------------

// Rotary position embedding (RoPE), NeoX half-split form. `x`, `out` are
// [tokens, n_heads, head_dim] row-major of dtype `dt`; token t uses position
// (pos0 + t). Rotates pairs (i, i + head_dim/2). head_dim must be even.
void rope(sycl::queue& q, const void* x, void* out, std::size_t tokens,
          std::size_t n_heads, std::size_t head_dim, float base,
          std::size_t pos0, DType dt, Variant variant = Variant::sycl,
          bool blocking = true);

// ----------------------------------------------------------------------------
// optimizers
// ----------------------------------------------------------------------------

// Fused AdamW, in-place. `p` (params), `m`, `v` (moments) are updated; `g` is
// the gradient. All are [n] of dtype `dt`. `step` is 1-based (bias correction).
void adamw(sycl::queue& q, void* p, const void* g, void* m, void* v,
           std::size_t n, float lr, float beta1, float beta2, float eps,
           float weight_decay, int step, DType dt,
           Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// sampling
// ----------------------------------------------------------------------------

// Greedy argmax over the last axis. `logits` is [rows, vocab] of dtype `dt`;
// `out` is [rows] int32 (lowest index on ties).
void argmax(sycl::queue& q, const void* logits, int* out, std::size_t rows,
            std::size_t vocab, DType dt, Variant variant = Variant::sycl,
            bool blocking = true);

// ----------------------------------------------------------------------------
// matmul
// ----------------------------------------------------------------------------

// Dense GEMM: C[M,N] = A[M,K] * B[K,N], all row-major, fp32 accumulation. `a`,
// `b`, `c` are device pointers of dtype `dt`. The vendor variant is oneDNN
// matmul (XMX/DPAS-backed); the native SYCL variant is an SLM-tiled baseline.
void dense_gemm(sycl::queue& q, const void* a, const void* b, void* c,
                std::size_t M, std::size_t N, std::size_t K, DType dt,
                Variant variant = Variant::best, bool blocking = true);

// ----------------------------------------------------------------------------
// quantization
// ----------------------------------------------------------------------------

// int4 group-quantized GEMV (batch-1 decode), Marlin/Metal-style dequant on the
// fly. Weight W is [N, K] symmetric signed int4 packed 2-per-byte along K (low
// nibble = even k); `scales` is [N, K/group] fp16 (half); activation `x` is [K]
// and output `y` is [N], both of dtype `act_dt`. Accumulates in fp32:
//   y[n] = sum_k (int4(W[n,k]) * scales[n, k/group]) * x[k]
// K must be even and a multiple of `group`.
void qgemv_int4(sycl::queue& q, const void* w_packed, const void* scales,
                const void* x, void* y, std::size_t N, std::size_t K,
                std::size_t group, DType act_dt, Variant variant = Variant::sycl,
                bool blocking = true);

// fp8 format selector (OCP / NVIDIA fp8).
enum class Fp8Kind {
  e4m3,
  e5m2,
};

// fp8 GEMM: C[M,N] = A_fp8[M,K] @ B_fp8[K,N], scaled by a single global `scale`.
// A, B are opaque fp8 bytes (1 byte/elem) of kind `kind`; C is `out_dt`
// (f32/f16/bf16). Vendor-only (oneDNN XMX fp8); if fp8 matmul is unsupported on
// the device the call reports it via the runtime (no silent wrong result).
void fp8_gemm(sycl::queue& q, const void* a_fp8, const void* b_fp8, void* c,
              std::size_t M, std::size_t N, std::size_t K, Fp8Kind kind,
              float scale, DType out_dt, Variant variant = Variant::vendor,
              bool blocking = true);

// fp8 codecs: f32 -> fp8 (out is 1 byte/elem) and fp8 -> f32, both over `n`
// contiguous elements. Vendor-backed (oneDNN reorder). Useful for quantizing
// activations/weights and for exact round-trip references.
void fp8_encode(sycl::queue& q, const float* in, void* out_fp8, std::size_t n,
                Fp8Kind kind, bool blocking = true);
void fp8_decode(sycl::queue& q, const void* in_fp8, float* out, std::size_t n,
                Fp8Kind kind, bool blocking = true);

// mxfp4 GEMV (OCP microscaling FP4), native decode. Weight W is [N, K] of e2m1
// (fp4) elements packed 2/byte along K, with one e8m0 (power-of-two) block scale
// per 32 elements: `block_scales` is [N, K/32] uint8. Activation `x` is [K] and
// output `y` is [N], dtype `act_dt`. Dequant in fp32:
//   w = e2m1(nibble) * 2^(e8m0 - 127);  y[n] = sum_k w * x[k]
// K must be a multiple of 32. Proves mxfp4 (not a hardware feature) runs
// natively on Intel via a hand-written decoder.
void mxfp4_gemv(sycl::queue& q, const void* w_packed, const void* block_scales,
                const void* x, void* y, std::size_t N, std::size_t K,
                DType act_dt, Variant variant = Variant::sycl,
                bool blocking = true);

// nvfp4 GEMV (NVIDIA FP4), native decode. Weight W is [N, K] of e2m1 (fp4)
// packed 2/byte, with one e4m3 (fp8) block scale per 16 elements
// (`block_scales` is [N, K/16] uint8) and a per-tensor fp32 `global_scale`.
// Activation `x` is [K], output `y` is [N], dtype `act_dt`. Dequant in fp32:
//   w = e2m1(nibble) * e4m3(block_scale) * global_scale;  y[n] = sum_k w * x[k]
// K must be a multiple of 32. Proves nvfp4 decodes natively on Intel.
void nvfp4_gemv(sycl::queue& q, const void* w_packed, const void* block_scales,
                float global_scale, const void* x, void* y, std::size_t N,
                std::size_t K, DType act_dt, Variant variant = Variant::sycl,
                bool blocking = true);

// int8 w8a8 GEMM: C[M,N] = (A_int8[M,K] @ B_int8[K,N]) * a_scale[M] * b_scale[N].
// A, B are int8 device pointers; a_scale (per-row/token) and b_scale (per-col/
// channel) are fp32 [M] and [N]; C is `out_dt` (f32/f16/bf16). Accumulates int32.
// The vendor variant is oneDNN int8 matmul (XMX/DPAS); the native variant is an
// SLM-tiled int8 baseline.
void qgemm_int8(sycl::queue& q, const void* a_int8, const void* b_int8,
                const void* a_scale, const void* b_scale, void* c, std::size_t M,
                std::size_t N, std::size_t K, DType out_dt,
                Variant variant = Variant::best, bool blocking = true);

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
