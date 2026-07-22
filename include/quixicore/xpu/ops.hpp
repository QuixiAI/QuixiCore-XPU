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

// Flash-style scaled dot-product attention (online softmax; no materialized
// score matrix). Q is [n_heads, seq_q, d]; K, V are [n_kv_heads, seq_k, d]
// (GQA: q head h uses kv head h / (n_heads/n_kv_heads)); O is [n_heads, seq_q,
// d], dtype dt. scale = 1/sqrt(d). `causal` masks key positions k > q (aligned
// at the sequence end when seq_q==seq_k). fp32 accumulation. head_dim d <= 128.
void attention(sycl::queue& q, const void* Q, const void* K, const void* V,
               void* O, std::size_t n_heads, std::size_t n_kv_heads,
               std::size_t seq_q, std::size_t seq_k, std::size_t d, bool causal,
               DType dt, Variant variant = Variant::sycl, bool blocking = true);

// Flash-style SDPA with a fused f16 context store. Same contract and math as
// attention() above, but writes the output twice in one epilogue: O in dtype dt
// and O_f16 in f16 (same [n_heads, seq_q, d] layout). This folds the ctx->f16
// convert a downstream attention-output GEMM needs into the attention pass,
// removing a standalone O-sized convert kernel. O_f16 equals the f16 rounding of
// O. Shape: online-attention + fused f16 context store.
void attention_f16ctx(sycl::queue& q, const void* Q, const void* K,
                      const void* V, void* O, void* O_f16, std::size_t n_heads,
                      std::size_t n_kv_heads, std::size_t seq_q,
                      std::size_t seq_k, std::size_t d, bool causal, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

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

// Categorical sampling from temperature-scaled softmax(logits). `logits`
// [rows, vocab] dt; `out` [rows] int32. `seed` drives the stateless RNG (one
// uniform per row). temperature -> 0 reduces to argmax.
void sample_categorical(sycl::queue& q, const void* logits, int* out,
                        std::size_t rows, std::size_t vocab, float temperature,
                        std::uint32_t seed, DType dt,
                        Variant variant = Variant::sycl, bool blocking = true);

// Top-k sampling: restrict to the k highest logits, softmax over them
// (temperature), then sample. `out` [rows] int32 is always one of the row's
// top-k tokens.
void top_k_sample(sycl::queue& q, const void* logits, int* out, std::size_t rows,
                  std::size_t vocab, int k, float temperature,
                  std::uint32_t seed, DType dt, Variant variant = Variant::sycl,
                  bool blocking = true);

// ----------------------------------------------------------------------------
// serving (kv cache, embedding)
// ----------------------------------------------------------------------------

// Embedding lookup: out[t, :] = table[ids[t], :]. `table` is [vocab, dim] dtype
// `dt`, `ids` is [n] int32, `out` is [n, dim] dtype `dt`.
void embedding_lookup(sycl::queue& q, const void* table, const int* ids,
                      void* out, std::size_t n, std::size_t dim, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

// KV-cache scatter: cache[slots[t], :] = src[t, :]. `cache` is [max_slots, row],
// `src` is [n, row], `slots` is [n] int32 (a negative slot skips the row).
// `row` = n_heads * head_dim (or any contiguous row width), dtype `dt`.
void kv_cache_scatter(sycl::queue& q, void* cache, const void* src,
                      const int* slots, std::size_t n, std::size_t row, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

// KV-cache gather: out[i, :] = cache[idx[i], :]. Inverse of scatter.
void kv_cache_gather(sycl::queue& q, const void* cache, const int* idx,
                     void* out, std::size_t n, std::size_t row, DType dt,
                     Variant variant = Variant::sycl, bool blocking = true);

// Sentence-embedding pooling head: masked mean-pool over each sequence's tokens
// with a per-token RMSNorm (learned weight) folded in, then L2-normalize. `x` is
// [total_tokens, dim] row-major; sequence s owns rows [offsets[s], offsets[s+1])
// (`offsets` is [batch+1] int32, monotonic). `weight` is [dim], `out` is
// [batch, dim], all dtype dt. For sequence s over token range [a, b):
//   r_t   = x[t] * rsqrt(mean_d(x[t,d]^2) + eps) * weight   (RMSNorm, per token)
//   m     = (1/(b-a)) * sum_t r_t                           (masked mean)
//   out[s]= m * rsqrt(sum_d m[d]^2)                         (L2; 0-vector passes)
// `dim` is the shape key {256,512,768,1024}; fp32 accumulation. An empty
// sequence (b==a) yields a zero vector. Shape: masked-mean + per-token RMSNorm +
// L2 pooling head. Native-only.
void pool_mean_rms_l2(sycl::queue& q, const void* x, const void* weight,
                      const int* offsets, void* out, std::size_t batch,
                      std::size_t dim, float eps, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// utils
// ----------------------------------------------------------------------------

// Inverted dropout: out[i] = uniform(seed,i) < p ? 0 : in[i]/(1-p), over `n`.
void dropout(sycl::queue& q, const void* in, void* out, std::size_t n, float p,
             std::uint32_t seed, DType dt, Variant variant = Variant::sycl,
             bool blocking = true);

// Per-row cross-entropy loss from logits: loss[r] = logsumexp(logits[r,:]) -
// logits[r, target[r]]. `logits` [rows, vocab] dt, `target` [rows] int32,
// `loss` [rows] fp32. fp32 accumulation.
void cross_entropy(sycl::queue& q, const void* logits, const int* target,
                   float* loss, std::size_t rows, std::size_t vocab, DType dt,
                   Variant variant = Variant::sycl, bool blocking = true);

// Fast Walsh-Hadamard transform (unnormalized) over each row: out[r,:] =
// H_n * in[r,:]. `in`/`out` [rows, n] dt, n a power of two (<= 2048).
void hadamard(sycl::queue& q, const void* in, void* out, std::size_t rows,
              std::size_t n, DType dt, Variant variant = Variant::sycl,
              bool blocking = true);

// ----------------------------------------------------------------------------
// moe
// ----------------------------------------------------------------------------

// MoE top-k routing. `router_logits` [n_tokens, n_experts] dtype dt. Selects the
// top-k experts per token and softmax-normalizes over the selected k. Outputs
// `expert_ids` [n_tokens, k] int32 and `expert_weights` [n_tokens, k] fp32.
void moe_route_topk(sycl::queue& q, const void* router_logits, int* expert_ids,
                    float* expert_weights, std::size_t n_tokens,
                    std::size_t n_experts, int k, DType dt,
                    Variant variant = Variant::sycl, bool blocking = true);

// Decode-oriented routed MoE with ModelOpt NVFP4 weights. `hidden` is [M,K]
// dtype `act_dt`; ids/weights are [M,top_k] int32/f32. Weight layouts are
// w13 [E,2I,K/2], w13_scales [E,2I,K/16], w2 [E,K,I/2], and w2_scales
// [E,K,I/16], with one fp32 global scale per expert/projection. The output is
// fp32 [M,K]. Invalid expert ids (<0 or >=E) are skipped. The call initializes
// `out_f32` to zero before accumulating routed outputs.
void nvfp4_moe_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *out_f32, std::size_t M, std::size_t E,
                     std::size_t top_k, std::size_t K, std::size_t I, DType act_dt,
                     bool multiply_router_weight = true, Variant variant = Variant::sycl,
                     bool blocking = true);

// Higher-occupancy two-stage form of `nvfp4_moe_fused`. The caller supplies
// fp32 scratch [M*top_k,2I]. This is the preferred batch-1 graph-replay path;
// the fused form can be faster once M*top_k already fills the device.
void nvfp4_moe_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *scratch_f32, float *out_f32,
                     std::size_t M, std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                     DType act_dt, bool multiply_router_weight = true,
                     Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// linear_attention
// ----------------------------------------------------------------------------

// Non-causal linear attention. Q, K, V are [n_heads, seq, dim] dtype dt; O is
// [n_heads, seq, dim]. Per head: KV = sum_t K[t]^T V[t] (dim x dim), z = sum_t
// K[t] (dim), O[t] = (Q[t] @ KV) / (Q[t] . z + eps). fp32 accumulation. dim must
// be <= 64 for the SLM path. Native-only.
void linear_attn(sycl::queue& q, const void* Q, const void* K, const void* V,
                 void* O, std::size_t n_heads, std::size_t seq, std::size_t dim,
                 DType dt, Variant variant = Variant::sycl, bool blocking = true);

// Qwen3.5/Qwen3.6 Gated DeltaNet decode core for the non-interleaved layout.
// Fixed model dimensions are q/k=16x128, v/z=32x128, conv width=4. Inputs are
// projected_qkvz [B,12288] and projected_ba [B,64]. The call mutates conv_state
// ([slots,3,8192] or [slots,8192,3]) and ssm_state [slots,32,128,128], writes
// scratch `mixed_qkv` [B,8192], and returns core/z [B,32,128]. State indices in
// [0,slots) are valid and must be unique within a decode batch. Negative or
// out-of-range indices leave state untouched and produce zero core output;
// `z` always passes through from the projected input.
void qwen_gdn_decode(sycl::queue &q, const void *projected_qkvz, const void *projected_ba,
                     void *conv_state, void *ssm_state, const void *conv_weight,
                     const void *conv_bias, const float *A_log, const void *dt_bias,
                     const int *state_indices, void *mixed_qkv, void *core_out, void *z_out,
                     std::size_t batch, std::size_t state_slots, bool conv_dim_first, DType act_dt,
                     DType state_dt, DType dt_bias_dt, Variant variant = Variant::sycl,
                     bool blocking = true);

// ----------------------------------------------------------------------------
// ssm (state-space / Mamba)
// ----------------------------------------------------------------------------

// Mamba selective scan (S6), forward. Per channel c and state s the recurrence
// is h_s = exp(delta*A[c,s]) * h_s + delta*B[t,s]*u[c,t]; y[c,t] = sum_s C[t,s]*h_s
// + D[c]*u[c,t]. Shapes: u,delta [n_chan, seq]; A [n_chan, state]; B,C [seq,
// state] (shared across channels); D [n_chan]; y [n_chan, seq]. All dtype dt
// except the recurrence runs in fp32. `state` <= 16. Native-only (sequential).
void selective_scan(sycl::queue& q, const void* u, const void* delta,
                    const void* A, const void* B, const void* C, const void* D,
                    void* y, std::size_t n_chan, std::size_t seq,
                    std::size_t state, DType dt, Variant variant = Variant::sycl,
                    bool blocking = true);

// ----------------------------------------------------------------------------
// collectives (multi-GPU)
// ----------------------------------------------------------------------------

// Sum all-reduce across ALL visible Intel GPUs (the 4x B60). `in_per_gpu` is a
// host buffer [n_gpus * count] where GPU g's contribution is at offset g*count;
// `out` is host [count] = the elementwise sum across GPUs. Internally scatters
// to each GPU (shared SYCL context), reduces via cross-device USM copies, and
// broadcasts. Returns the number of GPUs used (capability-gated: 0 if none).
// Native path; a oneCCL vendor variant is the production route (deferred).
std::size_t all_reduce_sum(const float* in_per_gpu, float* out,
                           std::size_t count);

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

// Quantize a weight matrix W [N, K] (dtype dt) to symmetric int4, group-wise:
// per group of `group` columns, scale = groupmax(|W|)/7, q = round(W/scale)
// clamped to [-8,7], packed 2-per-byte along K (low nibble = even k). Outputs
// `w_packed` [N*K/2] uint8 and `scales` [N, K/group] fp16 — the exact layout
// qgemv_int4 consumes (so quantize -> qgemv round-trips). Native.
void quantize_int4_group(sycl::queue& q, const void* w, void* w_packed,
                         void* scales, std::size_t N, std::size_t K,
                         std::size_t group, DType dt,
                         Variant variant = Variant::sycl, bool blocking = true);

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
// (f32/f16/bf16). best-routing (measured on B60): M=1 -> native SYCL decode
// GEMV (weight-memory-bound fast path); M>1 -> oneDNN matmul. If the vendor
// fp8 matmul is unsupported the call reports it (no silent wrong result).
void fp8_gemm(sycl::queue& q, const void* a_fp8, const void* b_fp8, void* c,
              std::size_t M, std::size_t N, std::size_t K, Fp8Kind kind,
              float scale, DType out_dt, Variant variant = Variant::best,
              bool blocking = true);

// FP8 weight-only GEMM: C[M,N] = A[M,K] @ dequant(W[N,K])^T. Activations and
// output use `act_dt`; W stores raw e4m3/e5m2 bytes in checkpoint-native [N,K]
// layout. `weight_scale` is an fp32 device pointer to [N] (per-channel) or [1].
// `best` routes M=1 to native decode GEMV and M>1 to oneDNN when present.
void fp8_gemm_w8a16(sycl::queue &q, const void *activations, const void *weight_fp8,
                    const float *weight_scale, void *out, std::size_t M, std::size_t N,
                    std::size_t K, Fp8Kind kind, bool per_channel, DType act_dt,
                    Variant variant = Variant::best, bool blocking = true);

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

// Batched NVFP4 W4A16 GEMM with checkpoint-native packed W [N,K/2]. The
// decode path submits one optimized GEMV per activation row; this beat the
// decode-once M-tiled alternative at every measured serving shape M=1,4,8.
void nvfp4_gemm(sycl::queue &q, const void *w_packed, const void *block_scales, float global_scale,
                const void *x, void *y, std::size_t M, std::size_t N, std::size_t K, DType act_dt,
                Variant variant = Variant::sycl, bool blocking = true);

// GGUF (llama.cpp) block-quant GEMV, native decode from the authentic on-disk
// block layout. `w_blocks` is row-major [N rows], each row = K/32 blocks laid
// consecutively; a block is { fp16 scale d; quants } — q8_0: 34 bytes (32 int8),
// q4_0: 18 bytes (32 int4 packed, dequant (nibble-8)*d). Activation `x` is [K],
// output `y` is [N], dtype `act_dt`. K must be a multiple of 32.
enum class GgufType {
  q8_0,
  q4_0,
  q6_K,
  q4_K,
  q5_K,
  q2_K,
  q3_K,
  iq4_nl,
  q4_1,
  q5_0,
  q5_1,
  iq4_xs,
  iq2_xxs,
  iq2_xs,
  iq3_xxs,
  iq1_s,
};

void gguf_gemv(sycl::queue& q, const void* w_blocks, const void* x, void* y,
               std::size_t N, std::size_t K, GgufType type, DType act_dt,
               Variant variant = Variant::sycl, bool blocking = true);

// Per-token symmetric int8 activation quantization. `x` [rows, dim] dtype dt ->
// `q` [rows, dim] int8 + `scale` [rows] fp32, where scale = rowmax(|x|)/127 and
// q = round(x/scale). Feeds qgemm_int8 (the w8a8 path). Native.
void act_quant_int8(sycl::queue& q, const void* x, signed char* q_out,
                    float* scale, std::size_t rows, std::size_t dim, DType dt,
                    Variant variant = Variant::sycl, bool blocking = true);

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

// Fused residual add + RMSNorm. For each row, normalize the unrounded fp32 sum
// of x and residual, update residual in place with the storage-dtype sum, and
// write the normalized/weighted result to out. `out` must not alias `residual`.
void fused_add_rms_norm(sycl::queue &q, const void *x, void *residual, const void *weight,
                        void *out, std::size_t rows, std::size_t dim, float eps, DType dt,
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
