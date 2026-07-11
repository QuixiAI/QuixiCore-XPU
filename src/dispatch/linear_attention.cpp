// Dispatch layer for the linear_attention family (native only).

#include "quixicore/xpu/ops.hpp"

#include "linear_attention/linear_attn/linear_attn_kernel.hpp"
#include "linear_attention/qwen_gdn_decode/qwen_gdn_kernel.hpp"

namespace quixicore::xpu::ops {

void linear_attn(sycl::queue& q, const void* Q, const void* K, const void* V,
                 void* O, std::size_t n_heads, std::size_t seq, std::size_t dim,
                 DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::linear_attn_sycl(q, Q, K, V, O, n_heads, seq, dim, dt);
  if (blocking) ev.wait();
}

void qwen_gdn_decode(sycl::queue &q, const void *projected_qkvz, const void *projected_ba,
                     void *conv_state, void *ssm_state, const void *conv_weight,
                     const void *conv_bias, const float *A_log, const void *dt_bias,
                     const int *state_indices, void *mixed_qkv, void *core_out, void *z_out,
                     std::size_t batch, std::size_t state_slots, bool conv_dim_first, DType act_dt,
                     DType state_dt, DType dt_bias_dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event event = kernels::qwen_gdn_decode_sycl(
      q, projected_qkvz, projected_ba, conv_state, ssm_state, conv_weight, conv_bias, A_log,
      dt_bias, state_indices, mixed_qkv, core_out, z_out, batch, state_slots, conv_dim_first,
      act_dt, state_dt, dt_bias_dt);
  if (blocking)
    event.wait();
}

}  // namespace quixicore::xpu::ops
