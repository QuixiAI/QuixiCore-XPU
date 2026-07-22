// QuixiCore XPU on-device benchmark harness.
//
// Times backend ops on a real Intel GPU using SYCL queue-profiling events
// (device command_start -> command_end timestamps), which measure pure device
// execution and are immune to the host submit/sync latency floor that produced
// false regressions in the Metal harness's early wall-clock timing.
//
// It calls the same variant entry points the dispatch layer uses, so --variant
// sycl and --variant vendor benchmark exactly the shipped implementations.
//
// Methodology:
//   * warm up (JIT + caches) for `--warmup` launches,
//   * time single-launch kernels by event and multi-launch ops in five
//     profiled batches of `--iters` calls,
//   * report median / min / max device time and effective bandwidth.
//
// Output is one JSON object per line on stdout (schema_version 2), suitable for
// appending to perf/results/<date>/<run-id>/results.jsonl. Correctness is NOT
// asserted here (that is the pytest / ops-smoke job); this only measures time.
//
// Gated behind the SYCL build. Requires an Intel GPU.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"
#include "quixicore/xpu/variants.hpp"

#include "activations/gelu/gelu_kernel.hpp"
#include "activations/gelu_backward/gelu_backward_kernel.hpp"
#include "activations/glu/glu_kernel.hpp"
#include "activations/silu/silu_kernel.hpp"
#include "activations/softmax/softmax_kernel.hpp"
#include "attention/attention/attention_kernel.hpp"
#include "attention/rope/rope_kernel.hpp"
#include "matmul/dense_gemm/dense_gemm_kernel.hpp"
#include "norms/norms_kernel.hpp"
#include "optimizers/adamw/adamw_kernel.hpp"
#include "quantization/act_quant/act_quant_kernel.hpp"
#include "quantization/quantize/quantize_kernel.hpp"
#include "quantization/fp8_gemm/fp8_kernel.hpp"
#include "quantization/gguf_gemv/gguf_kernel.hpp"
#include "quantization/mxfp4_gemv/mxfp4_kernel.hpp"
#include "quantization/nvfp4_gemv/nvfp4_kernel.hpp"
#include "quantization/qgemm/qgemm_kernel.hpp"
#include "quantization/qgemv/qgemv_kernel.hpp"
#include "sampling/argmax/argmax_kernel.hpp"
#include "sampling/sample/sample_kernel.hpp"
#include "linear_attention/linear_attn/linear_attn_kernel.hpp"
#include "linear_attention/qwen_gdn_decode/qwen_gdn_kernel.hpp"
#include "moe/moe_route/moe_route_kernel.hpp"
#include "moe/nvfp4_moe/nvfp4_moe_kernel.hpp"
#include "ssm/selective_scan/selective_scan_kernel.hpp"
#include "serving/serving_kernel.hpp"
#include "utils/utils_kernel.hpp"

namespace {

using quixicore::xpu::DType;
using quixicore::xpu::Variant;
using quixicore::xpu::half_t;
using quixicore::xpu::bf16_t;

DType parse_dtype(const std::string& s) {
  if (s == "f32") return DType::f32;
  if (s == "f16") return DType::f16;
  if (s == "bf16") return DType::bf16;
  throw std::invalid_argument("unknown dtype: " + s);
}

Variant parse_variant(const std::string& s) {
  if (s == "sycl") return Variant::sycl;
  if (s == "vendor") return Variant::vendor;
  if (s == "best") return Variant::best;
  throw std::invalid_argument("unknown variant: " + s);
}

double event_ms(const sycl::event& ev) {
  const auto start =
      ev.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end =
      ev.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1e-6;  // ns -> ms
}

struct DeviceTiming {
  double median_ms;
  double min_ms;
  double max_ms;
};

// Baseline second pass for the pool_mean_rms_l2 A/B: masked mean over each
// sequence's tokens + L2, reading rows that a prior rms_norm pass already
// normalized. Together with kernels::rms_norm_sycl this is the naive two-pass
// decomposition the fused kernel collapses (the delta is the [total,dim] scratch
// round-trip). Same subgroup-per-sequence layout as the shipped kernel.
template <typename T, int DIM>
sycl::event pool_meanl2_from_normed(sycl::queue& q, const T* normed,
                                    const int* off, T* out, std::size_t batch) {
  constexpr int SG = 16, SLOTS = DIM / SG;
  return q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(batch * SG), sycl::range<1>(SG)),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const std::size_t seq = it.get_group(0);
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int a = off[seq], b = off[seq + 1];
        float p[SLOTS];
#pragma unroll
        for (int s = 0; s < SLOTS; ++s) p[s] = 0.0f;
        for (int t = a; t < b; ++t) {
          const std::size_t base = static_cast<std::size_t>(t) * DIM;
#pragma unroll
          for (int s = 0; s < SLOTS; ++s)
            p[s] += static_cast<float>(normed[base + lane + s * SG]);
        }
        const int c = b - a;
        const float invt = c > 0 ? 1.0f / static_cast<float>(c) : 0.0f;
        float ss = 0.0f;
#pragma unroll
        for (int s = 0; s < SLOTS; ++s) { p[s] *= invt; ss = sycl::fma(p[s], p[s], ss); }
        ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
        const float invl = ss == 0.0f ? 1.0f : sycl::rsqrt(ss);
#pragma unroll
        for (int s = 0; s < SLOTS; ++s)
          out[seq * DIM + lane + s * SG] = static_cast<T>(p[s] * invl);
      });
}

template <typename T>
sycl::event pool_meanl2_dispatch(sycl::queue& q, const T* normed, const int* off,
                                 T* out, std::size_t batch, std::size_t dim) {
  switch (dim) {
    case 256:  return pool_meanl2_from_normed<T, 256>(q, normed, off, out, batch);
    case 512:  return pool_meanl2_from_normed<T, 512>(q, normed, off, out, batch);
    case 768:  return pool_meanl2_from_normed<T, 768>(q, normed, off, out, batch);
    default:   return pool_meanl2_from_normed<T, 1024>(q, normed, off, out, batch);
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace quixicore::xpu;

  std::string kernel = "gelu";
  std::string dtype_s = "f32";
  std::string variant_s = "sycl";
  std::string approx_s = "erf";
  std::size_t n = 1u << 20;  // 1,048,576 elements (elementwise kernels)
  std::size_t rows = 4096;   // row kernels: [rows, dim]
  std::size_t dim = 4096;
  std::size_t M = 1024, N = 1024, K = 1024;  // gemm dims
  int iters = 50;
  int warmup = 10;
  std::size_t device_index = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
      return argv[++i];
    };
    if (a == "--kernel") kernel = next();
    else if (a == "--dtype") dtype_s = next();
    else if (a == "--variant") variant_s = next();
    else if (a == "--approx") approx_s = next();
    else if (a == "--n") n = std::stoull(next());
    else if (a == "--rows") rows = std::stoull(next());
    else if (a == "--dim") dim = std::stoull(next());
    else if (a == "--M") M = std::stoull(next());
    else if (a == "--N") N = std::stoull(next());
    else if (a == "--K") K = std::stoull(next());
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--warmup") warmup = std::stoi(next());
    else if (a == "--device") device_index = std::stoull(next());
    else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
  }

  const DType dt = parse_dtype(dtype_s);
  const Variant requested = parse_variant(variant_s);
  const Variant variant = resolve_variant(requested);
  const bool tanh_approx = (approx_s == "tanh");

  const auto devices = gpu_devices();
  if (devices.empty()) {
    std::cerr << "no SYCL GPU device; cannot benchmark\n";
    return 0;  // skip, not fail
  }
  sycl::queue q = make_gpu_queue(device_index, /*enable_profiling=*/true);

  if (iters <= 0 || warmup < 0) {
    throw std::invalid_argument("iters must be positive and warmup nonnegative");
  }

  auto time_device_batches = [&](auto &&submit_once) {
    for (int i = 0; i < warmup; ++i)
      submit_once();
    q.wait();
    std::vector<double> samples;
    constexpr int kSamples = 5;
    samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
      sycl::event begin = q.single_task([] {});
      for (int i = 0; i < iters; ++i)
        submit_once();
      sycl::event end = q.single_task([] {});
      end.wait();
      const auto start = begin.get_profiling_info<sycl::info::event_profiling::command_end>();
      const auto stop = end.get_profiling_info<sycl::info::event_profiling::command_start>();
      samples.push_back(static_cast<double>(stop - start) * 1e-6 / static_cast<double>(iters));
    }
    std::sort(samples.begin(), samples.end());
    return DeviceTiming{samples[samples.size() / 2], samples.front(), samples.back()};
  };

  const std::size_t elem = dtype_size(dt);
  const bool is_gemm = (kernel == "dense_gemm");
  const bool is_softmax = (kernel == "softmax");
  const bool is_norm = (kernel == "rms_norm" || kernel == "layernorm");
  const bool is_glu = (kernel == "glu");
  const bool is_row = is_norm || is_softmax;

  // GEMM has its own buffer set and metric; handle it and return early.
  if (is_gemm) {
    void* ga = sycl::malloc_device(M * K * elem, q);
    void* gb = sycl::malloc_device(K * N * elem, q);
    void* gc = sycl::malloc_device(M * N * elem, q);
    q.memset(ga, 0, M * K * elem).wait();
    q.memset(gb, 0, K * N * elem).wait();

    auto gemm_once = [&]() -> sycl::event {
      if (variant == Variant::vendor) {
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
        return kernels::dense_gemm_onednn(q, ga, gb, gc, M, N, K, dt);
#endif
      }
      return kernels::dense_gemm_sycl(q, ga, gb, gc, M, N, K, dt);
    };
    for (int i = 0; i < warmup; ++i) gemm_once().wait();
    std::vector<double> s;
    s.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      sycl::event ev = gemm_once();
      ev.wait();
      s.push_back(event_ms(ev));
    }
    std::sort(s.begin(), s.end());
    const double median = s[s.size() / 2];
    const double gflops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                          static_cast<double>(K) / (median * 1e-3) / 1e9;
    sycl::free(ga, q); sycl::free(gb, q); sycl::free(gc, q);
    std::cout << "{\"schema_version\":2,\"kernel\":\"dense_gemm\",\"variant\":\""
              << variant_name(variant) << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
              << ",\"iters\":" << iters << ",\"median_ms\":" << median
              << ",\"gflops\":" << gflops << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    return 0;
  }

  if (kernel == "fp8_gemm") {
    void* A = sycl::malloc_device(M * K, q);   // 1 byte/elem
    void* B = sycl::malloc_device(K * N, q);
    void* C = sycl::malloc_device(M * N * elem, q);
    q.memset(A, 0, M * K).wait(); q.memset(B, 0, K * N).wait();
    const int fk = (approx_s == "e5m2") ? 1 : 0;  // reuse --approx to pick fp8 kind
    const bool native = (variant != Variant::vendor && M == 1);  // native path is the M=1 GEMV
    auto once = [&]() {
      if (native) { kernels::fp8_gemv_sycl(q, A, B, C, N, K, fk, 1.0f, dt).wait(); return true; }
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      return kernels::fp8_gemm_onednn(q, A, B, C, M, N, K, fk, 1.0f, dt);
#else
      return false;
#endif
    };
    for (int i = 0; i < warmup; ++i) once();
    // Both routes synchronize internally (oneDNN waits; the GEMV is a 3-event
    // chain), so time a batch between two profiling markers and divide.
    const int batch = iters;
    sycl::event a = q.single_task([] {});
    a.wait();
    const auto b0 = a.get_profiling_info<sycl::info::event_profiling::command_end>();
    for (int i = 0; i < batch; ++i) once();
    sycl::event z = q.single_task([] {});
    z.wait();
    const auto b1 = z.get_profiling_info<sycl::info::event_profiling::command_start>();
    const double median = (static_cast<double>(b1 - b0) * 1e-6) / batch;  // ms/call
    const double gflops = 2.0 * (double)M * (double)N * (double)K / (median * 1e-3) / 1e9;
    const double gbps = ((double)M * K + (double)K * N + (double)M * N * elem) /
                        (median * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"fp8_gemm\",\"variant\":\""
              << (native ? "sycl" : "vendor") << "\",\"fp8\":\""
              << (fk ? "e5m2" : "e4m3") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
              << ",\"iters\":" << batch << ",\"median_ms\":" << median
              << ",\"gflops\":" << gflops << ",\"gbps\":" << gbps << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
    return 0;
  }
  if (kernel == "fp8_w8a16") {
    void *activations = sycl::malloc_device(M * K * elem, q);
    void *weight = sycl::malloc_device(N * K, q);
    float *scales = sycl::malloc_device<float>(N, q);
    void *output = sycl::malloc_device(M * N * elem, q);
    q.memset(activations, 0, M * K * elem).wait();
    q.memset(weight, 0, N * K).wait();
    q.fill(scales, 1.0f, N).wait();
    const int fp8_kind = approx_s == "e5m2" ? 1 : 0;
    bool vendor_supported = true;
    auto once = [&] {
      if (variant == Variant::vendor) {
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
        vendor_supported = kernels::fp8_gemm_w8a16_onednn(q, activations, weight, scales, true,
                                                          output, M, N, K, fp8_kind, dt);
        return;
#endif
      }
      kernels::fp8_gemm_w8a16_sycl(q, activations, weight, scales, true, output, M, N, K, fp8_kind,
                                   dt);
    };
    const DeviceTiming timing = time_device_batches(once);
    const double median = timing.median_ms;
    if (!vendor_supported) {
      throw std::runtime_error("oneDNN does not support the requested W8A16 shape");
    }
    const double weight_gbps = static_cast<double>(M) * N * K / (median * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"fp8_w8a16\","
              << "\"variant\":\"" << variant_name(variant) << "\",\"fp8\":\""
              << (fp8_kind ? "e5m2" : "e4m3") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K << ",\"iters\":" << iters
              << ",\"median_ms\":" << median << ",\"min_ms\":" << timing.min_ms
              << ",\"max_ms\":" << timing.max_ms << ",\"weight_gbps\":" << weight_gbps
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(activations, q);
    sycl::free(weight, q);
    sycl::free(scales, q);
    sycl::free(output, q);
    return 0;
  }
  if (kernel == "nvfp4_gemm") {
    void *weight = sycl::malloc_device(N * K / 2, q);
    void *scales = sycl::malloc_device(N * K / 16, q);
    void *activations = sycl::malloc_device(M * K * elem, q);
    void *output = sycl::malloc_device(M * N * elem, q);
    q.memset(weight, 0, N * K / 2).wait();
    q.memset(scales, 0x38, N * K / 16).wait();
    q.memset(activations, 0, M * K * elem).wait();
    const bool mtiled = approx_s == "mtiled";
    auto once = [&] {
      if (mtiled) {
        kernels::nvfp4_gemm_mtiled_sycl(q, weight, scales, 1.0f, activations, output, M, N, K, dt);
      } else {
        kernels::nvfp4_gemm_sycl(q, weight, scales, 1.0f, activations, output, M, N, K, dt);
      }
    };
    const DeviceTiming timing = time_device_batches(once);
    const double median = timing.median_ms;
    const double weight_gbps = static_cast<double>(M) * N * K / 2.0 / (median * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"nvfp4_gemm\","
              << "\"variant\":\"" << (mtiled ? "mtiled" : "row_loop") << "\",\"dtype\":\""
              << dtype_name(dt) << "\",\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
              << ",\"iters\":" << iters << ",\"median_ms\":" << median
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"weight_gbps\":" << weight_gbps << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(weight, q);
    sycl::free(scales, q);
    sycl::free(activations, q);
    sycl::free(output, q);
    return 0;
  }
  if (kernel == "nvfp4_moe") {
    const std::size_t experts = N;
    const std::size_t top_k = rows;
    const std::size_t intermediate = dim;
    const std::size_t pairs = M * top_k;
    void *hidden = sycl::malloc_device(M * K * elem, q);
    int *expert_ids = sycl::malloc_device<int>(pairs, q);
    float *router_weights = sycl::malloc_device<float>(pairs, q);
    void *w13 = sycl::malloc_device(experts * 2 * intermediate * K / 2, q);
    void *w13_scales = sycl::malloc_device(experts * 2 * intermediate * K / 16, q);
    float *w13_global = sycl::malloc_device<float>(experts, q);
    void *w2 = sycl::malloc_device(experts * K * intermediate / 2, q);
    void *w2_scales = sycl::malloc_device(experts * K * intermediate / 16, q);
    float *w2_global = sycl::malloc_device<float>(experts, q);
    float *scratch = sycl::malloc_device<float>(pairs * 2 * intermediate, q);
    float *output = sycl::malloc_device<float>(M * K, q);
    q.memset(hidden, 0, M * K * elem).wait();
    q.memset(expert_ids, 0, pairs * sizeof(int)).wait();
    q.fill(router_weights, 1.0f / static_cast<float>(top_k), pairs).wait();
    q.memset(w13, 0, experts * 2 * intermediate * K / 2).wait();
    q.memset(w13_scales, 0x38, experts * 2 * intermediate * K / 16).wait();
    q.fill(w13_global, 1.0f, experts).wait();
    q.memset(w2, 0, experts * K * intermediate / 2).wait();
    q.memset(w2_scales, 0x38, experts * K * intermediate / 16).wait();
    q.fill(w2_global, 1.0f, experts).wait();
    const bool split = approx_s == "split";
    auto once = [&] {
      const sycl::event zeroed = q.memset(output, 0, M * K * sizeof(float));
      if (split) {
        kernels::nvfp4_moe_split_sycl(q, hidden, expert_ids, router_weights, w13, w13_scales,
                                      w13_global, w2, w2_scales, w2_global, scratch, output, M,
                                      experts, top_k, K, intermediate, true, dt, zeroed);
      } else {
        kernels::nvfp4_moe_fused_sycl(q, hidden, expert_ids, router_weights, w13, w13_scales,
                                      w13_global, w2, w2_scales, w2_global, output, M, experts,
                                      top_k, K, intermediate, true, dt, zeroed);
      }
    };
    const DeviceTiming timing = time_device_batches(once);
    const double median = timing.median_ms;
    const double fp4_weight_bytes =
        static_cast<double>(pairs) * (2.0 * intermediate * K / 2.0 + K * intermediate / 2.0);
    const double weight_gbps = fp4_weight_bytes / (median * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"nvfp4_moe\","
              << "\"variant\":\"" << (split ? "split" : "fused") << "\",\"dtype\":\""
              << dtype_name(dt) << "\",\"M\":" << M << ",\"experts\":" << experts
              << ",\"top_k\":" << top_k << ",\"K\":" << K << ",\"I\":" << intermediate
              << ",\"iters\":" << iters << ",\"median_ms\":" << median
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"weight_gbps\":" << weight_gbps << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(hidden, q);
    sycl::free(expert_ids, q);
    sycl::free(router_weights, q);
    sycl::free(w13, q);
    sycl::free(w13_scales, q);
    sycl::free(w13_global, q);
    sycl::free(w2, q);
    sycl::free(w2_scales, q);
    sycl::free(w2_global, q);
    sycl::free(scratch, q);
    sycl::free(output, q);
    return 0;
  }
  if (kernel == "qwen_gdn_decode") {
    if (M == 0 || N < 2) {
      throw std::invalid_argument("qwen_gdn_decode requires M > 0 and at least two state slots");
    }
    constexpr std::size_t conv_dim = 8192;
    constexpr std::size_t qkvz_dim = 12288;
    constexpr std::size_t value_dim = 4096;
    const std::size_t batch = M;
    const std::size_t slots = N;
    void *projected_qkvz = sycl::malloc_device(batch * qkvz_dim * elem, q);
    void *projected_ba = sycl::malloc_device(batch * 64 * elem, q);
    void *conv_state = sycl::malloc_device(slots * 3 * conv_dim * elem, q);
    float *ssm_state = sycl::malloc_device<float>(slots * 32 * 128 * 128, q);
    void *conv_weight = sycl::malloc_device(conv_dim * 4 * elem, q);
    void *conv_bias = sycl::malloc_device(conv_dim * elem, q);
    float *A_log = sycl::malloc_device<float>(32, q);
    void *dt_bias = sycl::malloc_device(32 * elem, q);
    int *state_indices = sycl::malloc_shared<int>(batch, q);
    void *mixed_qkv = sycl::malloc_device(batch * conv_dim * elem, q);
    void *core = sycl::malloc_device(batch * value_dim * elem, q);
    void *z = sycl::malloc_device(batch * value_dim * elem, q);
    q.memset(projected_qkvz, 0, batch * qkvz_dim * elem).wait();
    q.memset(projected_ba, 0, batch * 64 * elem).wait();
    q.memset(conv_state, 0, slots * 3 * conv_dim * elem).wait();
    q.memset(ssm_state, 0, slots * 32 * 128 * 128 * sizeof(float)).wait();
    q.memset(conv_weight, 0, conv_dim * 4 * elem).wait();
    q.memset(conv_bias, 0, conv_dim * elem).wait();
    q.memset(A_log, 0, 32 * sizeof(float)).wait();
    q.memset(dt_bias, 0, 32 * elem).wait();
    for (std::size_t i = 0; i < batch; ++i)
      state_indices[i] = static_cast<int>(i % slots);
    auto once = [&] {
      kernels::qwen_gdn_decode_sycl(q, projected_qkvz, projected_ba, conv_state, ssm_state,
                                    conv_weight, conv_bias, A_log, dt_bias, state_indices,
                                    mixed_qkv, core, z, batch, slots, false, dt, DType::f32, dt);
    };
    const DeviceTiming timing = time_device_batches(once);
    const double median = timing.median_ms;
    std::cout << "{\"schema_version\":2,\"kernel\":\"qwen_gdn_decode\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt) << "\",\"batch\":" << batch
              << ",\"slots\":" << slots << ",\"iters\":" << iters << ",\"median_ms\":" << median
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(projected_qkvz, q);
    sycl::free(projected_ba, q);
    sycl::free(conv_state, q);
    sycl::free(ssm_state, q);
    sycl::free(conv_weight, q);
    sycl::free(conv_bias, q);
    sycl::free(A_log, q);
    sycl::free(dt_bias, q);
    sycl::free(state_indices, q);
    sycl::free(mixed_qkv, q);
    sycl::free(core, q);
    sycl::free(z, q);
    return 0;
  }
  if (kernel == "fused_add_rms_norm") {
    void *input = sycl::malloc_device(rows * dim * elem, q);
    void *residual = sycl::malloc_device(rows * dim * elem, q);
    void *weight = sycl::malloc_device(dim * elem, q);
    void *output = sycl::malloc_device(rows * dim * elem, q);
    q.memset(input, 0, rows * dim * elem).wait();
    q.memset(residual, 0, rows * dim * elem).wait();
    q.memset(weight, 0, dim * elem).wait();
    auto once = [&] {
      kernels::fused_add_rms_norm_sycl(q, input, residual, weight, output, rows, dim, 1e-6f, dt);
    };
    const DeviceTiming timing = time_device_batches(once);
    const double median = timing.median_ms;
    const double bytes = (4.0 * static_cast<double>(rows * dim) + dim) * elem;
    const double gbps = bytes / (median * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,"
              << "\"kernel\":\"fused_add_rms_norm\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt) << "\",\"rows\":" << rows
              << ",\"dim\":" << dim << ",\"iters\":" << iters << ",\"median_ms\":" << median
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"gbps\":" << gbps << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(input, q);
    sycl::free(residual, q);
    sycl::free(weight, q);
    sycl::free(output, q);
    return 0;
  }
  if (kernel == "qgemm_int8") {
    std::int8_t* A = sycl::malloc_device<std::int8_t>(M * K, q);
    std::int8_t* B = sycl::malloc_device<std::int8_t>(K * N, q);
    float* as = sycl::malloc_device<float>(M, q);
    float* bs = sycl::malloc_device<float>(N, q);
    void* C = sycl::malloc_device(M * N * elem, q);
    q.memset(A, 0, M * K).wait(); q.memset(B, 0, K * N).wait();
    q.memset(as, 0, M * 4).wait(); q.memset(bs, 0, N * 4).wait();
    auto once = [&]() -> sycl::event {
      if (variant == Variant::vendor) {
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
        return kernels::qgemm_int8_onednn(q, A, B, as, bs, C, M, N, K, dt);
#endif
      }
      return kernels::qgemm_int8_sycl(q, A, B, as, bs, C, M, N, K, dt);
    };
    for (int i = 0; i < warmup; ++i) once().wait();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) { sycl::event e = once(); e.wait(); s.push_back(event_ms(e)); }
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    const double gops = 2.0 * (double)M * (double)N * (double)K / (med * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"qgemm_int8\",\"variant\":\""
              << variant_name(variant) << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"gops\":" << gops << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(A, q); sycl::free(B, q); sycl::free(as, q); sycl::free(bs, q); sycl::free(C, q);
    return 0;
  }

  if (kernel == "mxfp4_gemv") {
    const std::size_t Nn = rows, Kk = dim;
    void* w = sycl::malloc_device(Nn * (Kk / 2), q);       // fp4 packed
    void* bs = sycl::malloc_device(Nn * (Kk / 32), q);     // e8m0 block scales
    void* xx = sycl::malloc_device(Kk * elem, q);
    void* yy = sycl::malloc_device(Nn * elem, q);
    q.memset(w, 0, Nn * (Kk / 2)).wait();
    q.memset(bs, 127, Nn * (Kk / 32)).wait();
    q.memset(xx, 0, Kk * elem).wait();
    auto once = [&] { return kernels::mxfp4_gemv_sycl(q, w, bs, xx, yy, Nn, Kk, dt); };
    for (int i = 0; i < warmup; ++i) once().wait();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) { sycl::event e = once(); e.wait(); s.push_back(event_ms(e)); }
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    const double wbytes = static_cast<double>(Nn) * static_cast<double>(Kk) / 2.0;  // fp4 = 0.5 byte
    std::cout << "{\"schema_version\":2,\"kernel\":\"mxfp4_gemv\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"N\":" << Nn << ",\"K\":" << Kk
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"weight_gbps\":" << (wbytes / (med * 1e-3) / 1e9)
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
              << "\"}" << std::endl;
    sycl::free(w, q); sycl::free(bs, q); sycl::free(xx, q); sycl::free(yy, q);
    return 0;
  }

  if (kernel == "gguf_gemv") {
    const std::size_t Nn = rows, Kk = dim;
    const int gt = (approx_s == "q4_0") ? 1 : (approx_s == "q6_K" ? 2 : (approx_s == "q4_K" ? 3 : (approx_s == "q5_K" ? 4 : (approx_s == "q2_K" ? 5 : (approx_s == "q3_K" ? 6 : (approx_s == "iq4_nl" ? 7 : 0))))));
    const std::size_t row_bytes = (gt == 2) ? (Kk / 256) * 210
                                  : (gt == 3) ? (Kk / 256) * 144
                                  : (gt == 4) ? (Kk / 256) * 176
                                  : (gt == 5) ? (Kk / 256) * 84
                                  : (gt == 6) ? (Kk / 256) * 110
                                  : (gt == 7) ? (Kk / 32) * 18
                                              : (Kk / 32) * (gt == 0 ? 34 : 18);
    void* w = sycl::malloc_device(Nn * row_bytes, q);
    void* xx = sycl::malloc_device(Kk * elem, q);
    void* yy = sycl::malloc_device(Nn * elem, q);
    q.memset(w, 0, Nn * row_bytes).wait();
    q.memset(xx, 0, Kk * elem).wait();
    auto once = [&] { return kernels::gguf_gemv_sycl(q, w, xx, yy, Nn, Kk, gt, dt); };
    for (int i = 0; i < warmup; ++i) once().wait();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) { sycl::event e = once(); e.wait(); s.push_back(event_ms(e)); }
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    const double wbytes = static_cast<double>(Nn) * static_cast<double>(row_bytes);
    std::cout << "{\"schema_version\":2,\"kernel\":\"gguf_gemv\",\"variant\":\"sycl\",\"gguf\":\""
              << (gt == 7 ? "iq4_nl" : gt == 6 ? "q3_K" : gt == 5 ? "q2_K" : gt == 4 ? "q5_K" : gt == 3 ? "q4_K" : gt == 2 ? "q6_K" : gt == 1 ? "q4_0" : "q8_0") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"N\":" << Nn << ",\"K\":" << Kk << ",\"iters\":" << iters
              << ",\"median_ms\":" << med << ",\"weight_gbps\":" << (wbytes / (med * 1e-3) / 1e9)
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
              << "\"}" << std::endl;
    sycl::free(w, q); sycl::free(xx, q); sycl::free(yy, q);
    return 0;
  }
  if (kernel == "nvfp4_gemv") {
    const std::size_t Nn = rows, Kk = dim;
    void* w = sycl::malloc_device(Nn * (Kk / 2), q);
    void* bs = sycl::malloc_device(Nn * (Kk / 16), q);  // e4m3 block scales, block 16
    void* xx = sycl::malloc_device(Kk * elem, q);
    void* yy = sycl::malloc_device(Nn * elem, q);
    q.memset(w, 0, Nn * (Kk / 2)).wait();
    q.memset(bs, 0x38, Nn * (Kk / 16)).wait();  // e4m3 ~1.0
    q.memset(xx, 0, Kk * elem).wait();
    auto once = [&] { return kernels::nvfp4_gemv_sycl(q, w, bs, 1.0f, xx, yy, Nn, Kk, dt); };
    for (int i = 0; i < warmup; ++i) once().wait();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) { sycl::event e = once(); e.wait(); s.push_back(event_ms(e)); }
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    const double wbytes = static_cast<double>(Nn) * static_cast<double>(Kk) / 2.0;
    std::cout << "{\"schema_version\":2,\"kernel\":\"nvfp4_gemv\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"N\":" << Nn << ",\"K\":" << Kk
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"weight_gbps\":" << (wbytes / (med * 1e-3) / 1e9)
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
              << "\"}" << std::endl;
    sycl::free(w, q); sycl::free(bs, q); sycl::free(xx, q); sycl::free(yy, q);
    return 0;
  }

  if (kernel == "embedding") {
    const std::size_t vocab = 128000, ntok = rows;
    void* table = sycl::malloc_device(vocab * dim * elem, q);
    int* ids = sycl::malloc_device<int>(ntok, q);
    void* out = sycl::malloc_device(ntok * dim * elem, q);
    q.memset(table, 0, vocab * dim * elem).wait();
    q.memset(ids, 0, ntok * sizeof(int)).wait();
    auto once = [&] { return kernels::embedding_lookup_sycl(q, table, ids, out, ntok, dim, dt); };
    for (int i = 0; i < warmup; ++i) once().wait();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) { sycl::event e = once(); e.wait(); s.push_back(event_ms(e)); }
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    const double bytes = 2.0 * static_cast<double>(ntok) * static_cast<double>(dim) * elem;
    std::cout << "{\"schema_version\":2,\"kernel\":\"embedding\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"rows\":" << ntok << ",\"dim\":" << dim
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"gbps\":" << (bytes / (med * 1e-3) / 1e9)
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
              << "\"}" << std::endl;
    sycl::free(table, q); sycl::free(ids, q); sycl::free(out, q);
    return 0;
  }

  // rope / adamw / argmax: self-contained buffer sets + timing.
  auto time_median = [&](auto thunk) {
    for (int i = 0; i < warmup; ++i) thunk().wait();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) { sycl::event e = thunk(); e.wait(); s.push_back(event_ms(e)); }
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
  };
  auto emit = [&](double median, double gbps) {
    std::cout << "{\"schema_version\":2,\"kernel\":\"" << kernel << "\",\"variant\":\"sycl\""
              << ",\"dtype\":\"" << dtype_name(dt) << "\",\"rows\":" << rows
              << ",\"dim\":" << dim << ",\"iters\":" << iters
              << ",\"median_ms\":" << median << ",\"gbps\":" << gbps
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
              << "\"}" << std::endl;
  };
  if (kernel == "attention") {
    const std::size_t nh = rows, seq = dim, d = 64;
    void* Q = sycl::malloc_device(nh * seq * d * elem, q);
    void* K = sycl::malloc_device(nh * seq * d * elem, q);
    void* V = sycl::malloc_device(nh * seq * d * elem, q);
    void* O = sycl::malloc_device(nh * seq * d * elem, q);
    q.memset(Q, 0, nh * seq * d * elem).wait(); q.memset(K, 0, nh * seq * d * elem).wait(); q.memset(V, 0, nh * seq * d * elem).wait();
    const double med = time_median([&] { return kernels::attention_sycl(q, Q, K, V, O, nh, nh, seq, seq, d, true, dt); });
    const double gflop = 2.0 * nh * seq * seq * d /* causal ~half */ / (med * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"attention\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"heads\":" << nh << ",\"seq\":" << seq << ",\"d\":" << d
              << ",\"causal\":true,\"iters\":" << iters << ",\"median_ms\":" << med << ",\"gflops\":" << gflop
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q);
    return 0;
  }
  if (kernel == "attention_f16ctx") {
    // A/B for the fused ctx->f16 store. --approx fused (default): time the single
    // attention_f16ctx kernel (writes O + O_f16). --approx unfused: baseline of
    // attention_sycl (writes O) followed by a standalone O->O_f16 convert kernel
    // -- the exact pass the fused epilogue folds away -- timed as the sum of both
    // device events. Same shapes; the delta is the eliminated convert traffic.
    const std::size_t nh = rows, seq = dim, d = 64;
    const std::size_t ne = nh * seq * d;
    void* Q = sycl::malloc_device(ne * elem, q);
    void* K = sycl::malloc_device(ne * elem, q);
    void* V = sycl::malloc_device(ne * elem, q);
    void* O = sycl::malloc_device(ne * elem, q);
    half_t* O16 = sycl::malloc_device<half_t>(ne, q);
    q.memset(Q, 0, ne * elem).wait(); q.memset(K, 0, ne * elem).wait(); q.memset(V, 0, ne * elem).wait();
    const bool unfused = (approx_s == "unfused");
    auto convert = [&]() -> sycl::event {
      switch (dt) {
        case DType::f16: { const half_t* o = static_cast<const half_t*>(O);
          return q.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> i) { O16[i[0]] = o[i[0]]; }); }
        case DType::bf16: { const bf16_t* o = static_cast<const bf16_t*>(O);
          return q.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> i) { O16[i[0]] = static_cast<half_t>(static_cast<float>(o[i[0]])); }); }
        default: { const float* o = static_cast<const float*>(O);
          return q.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> i) { O16[i[0]] = static_cast<half_t>(o[i[0]]); }); }
      }
    };
    auto once = [&]() -> double {
      if (unfused) {
        sycl::event ea = kernels::attention_sycl(q, Q, K, V, O, nh, nh, seq, seq, d, true, dt);
        ea.wait();
        sycl::event ec = convert();
        ec.wait();
        return event_ms(ea) + event_ms(ec);
      }
      sycl::event ef = kernels::attention_f16ctx_sycl(q, Q, K, V, O, O16, nh, nh, seq, seq, d, true, dt);
      ef.wait();
      return event_ms(ef);
    };
    for (int i = 0; i < warmup; ++i) once();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) s.push_back(once());
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    std::cout << "{\"schema_version\":2,\"kernel\":\"attention_f16ctx\",\"variant\":\"sycl\",\"approx\":\""
              << (unfused ? "unfused" : "fused") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"heads\":" << nh << ",\"seq\":" << seq << ",\"d\":" << d
              << ",\"causal\":true,\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"min_ms\":" << s.front() << ",\"max_ms\":" << s.back()
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q); sycl::free(O16, q);
    return 0;
  }
  if (kernel == "pool_mean_rms_l2") {
    // Sentence-embedding pooling head. A/B for the RMSNorm -> masked-mean -> L2
    // fusion. --approx fused (default): the single pool_mean_rms_l2_sycl kernel
    // (reads x once, writes one vector per sequence). --approx unfused: the naive
    // two-pass decomposition -- rms_norm over all [total,dim] token rows into
    // scratch, then a masked-mean+L2 pass reading scratch -- timed as the sum of
    // both device events. The delta is the eliminated [total,dim] scratch
    // round-trip. --rows = batch (sequences), --M = tokens/sequence, --dim = the
    // shape key (256/512/768/1024; other values fall back to 768).
    const std::size_t D =
        (dim == 256 || dim == 512 || dim == 768 || dim == 1024) ? dim : 768;
    const std::size_t batch = rows;
    const std::size_t tok = M;  // tokens per sequence (uniform for the benchmark)
    const std::size_t total = batch * tok;
    void* x = sycl::malloc_device(total * D * elem, q);
    void* w = sycl::malloc_device(D * elem, q);
    void* out = sycl::malloc_device(batch * D * elem, q);
    void* scratch = sycl::malloc_device(total * D * elem, q);
    int* off = sycl::malloc_shared<int>(batch + 1, q);
    q.memset(x, 0, total * D * elem).wait();
    q.memset(w, 0, D * elem).wait();
    for (std::size_t s = 0; s <= batch; ++s) off[s] = static_cast<int>(s * tok);
    const float eps = 1e-6f;
    const bool unfused = (approx_s == "unfused");
    auto once = [&]() -> double {
      if (unfused) {
        sycl::event er = kernels::rms_norm_sycl(q, x, w, scratch, total, D, eps, dt);
        er.wait();
        sycl::event em;
        switch (dt) {
          case DType::f16:
            em = pool_meanl2_dispatch<half_t>(q, static_cast<const half_t*>(scratch), off, static_cast<half_t*>(out), batch, D);
            break;
          case DType::bf16:
            em = pool_meanl2_dispatch<bf16_t>(q, static_cast<const bf16_t*>(scratch), off, static_cast<bf16_t*>(out), batch, D);
            break;
          default:
            em = pool_meanl2_dispatch<float>(q, static_cast<const float*>(scratch), off, static_cast<float*>(out), batch, D);
            break;
        }
        em.wait();
        return event_ms(er) + event_ms(em);
      }
      sycl::event ef =
          kernels::pool_mean_rms_l2_sycl(q, x, w, off, out, batch, D, eps, dt);
      ef.wait();
      return event_ms(ef);
    };
    for (int i = 0; i < warmup; ++i) once();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) s.push_back(once());
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    std::cout << "{\"schema_version\":2,\"kernel\":\"pool_mean_rms_l2\",\"variant\":\"sycl\",\"approx\":\""
              << (unfused ? "unfused" : "fused") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"batch\":" << batch << ",\"tokens\":" << tok << ",\"dim\":" << D
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"min_ms\":" << s.front() << ",\"max_ms\":" << s.back()
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(x, q); sycl::free(w, q); sycl::free(out, q); sycl::free(scratch, q); sycl::free(off, q);
    return 0;
  }
  if (kernel == "selective_scan") {
    const std::size_t nc = rows, seq = dim, st = 16;
    void* u = sycl::malloc_device(nc * seq * elem, q);
    void* dl = sycl::malloc_device(nc * seq * elem, q);
    void* A = sycl::malloc_device(nc * st * elem, q);
    void* B = sycl::malloc_device(seq * st * elem, q);
    void* C = sycl::malloc_device(seq * st * elem, q);
    void* D = sycl::malloc_device(nc * elem, q);
    void* y = sycl::malloc_device(nc * seq * elem, q);
    q.memset(u, 0, nc * seq * elem).wait(); q.memset(dl, 0, nc * seq * elem).wait();
    q.memset(A, 0, nc * st * elem).wait(); q.memset(B, 0, seq * st * elem).wait();
    q.memset(C, 0, seq * st * elem).wait(); q.memset(D, 0, nc * elem).wait();
    const double med = time_median([&] { return kernels::selective_scan_sycl(q, u, dl, A, B, C, D, y, nc, seq, st, dt); });
    const double gtok = static_cast<double>(nc) * seq / (med * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"selective_scan\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"chan\":" << nc << ",\"seq\":" << seq << ",\"state\":" << st
              << ",\"iters\":" << iters << ",\"median_ms\":" << med << ",\"Gelem_s\":" << gtok
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(u, q); sycl::free(dl, q); sycl::free(A, q); sycl::free(B, q); sycl::free(C, q); sycl::free(D, q); sycl::free(y, q);
    return 0;
  }
  if (kernel == "linear_attn") {
    const std::size_t nh = rows, seq = dim, d = 64;
    const std::size_t sz = nh * seq * d * elem;
    void* Q = sycl::malloc_device(sz, q); void* K = sycl::malloc_device(sz, q);
    void* V = sycl::malloc_device(sz, q); void* O = sycl::malloc_device(sz, q);
    q.memset(Q, 0, sz).wait(); q.memset(K, 0, sz).wait(); q.memset(V, 0, sz).wait();
    const double med = time_median([&] { return kernels::linear_attn_sycl(q, Q, K, V, O, nh, seq, d, dt); });
    const double gflop = 2.0 * nh * (seq * d * d + seq * d * d) / (med * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"linear_attn\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"heads\":" << nh << ",\"seq\":" << seq << ",\"dim\":" << d
              << ",\"iters\":" << iters << ",\"median_ms\":" << med << ",\"gflops\":" << gflop
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q);
    return 0;
  }
  if (kernel == "sample_categorical") {
    void* lg = sycl::malloc_device(rows * dim * elem, q);
    int* o = sycl::malloc_device<int>(rows, q);
    q.memset(lg, 0, rows * dim * elem).wait();
    const double med = time_median([&] { return kernels::sample_categorical_sycl(q, lg, o, rows, dim, 1.0f, 5u, dt); });
    emit(med, static_cast<double>(rows) * dim * elem / (med * 1e-3) / 1e9);
    sycl::free(lg, q); sycl::free(o, q);
    return 0;
  }
  if (kernel == "quantize_int4") {
    const std::size_t Nn = rows, Kk = dim, group = 128;
    void* wi = sycl::malloc_device(Nn * Kk * elem, q);
    void* wp = sycl::malloc_device(Nn * (Kk / 2), q);
    void* sc = sycl::malloc_device(Nn * (Kk / group) * 2, q);
    q.memset(wi, 0, Nn * Kk * elem).wait();
    const double med = time_median([&] { return kernels::quantize_int4_group_sycl(q, wi, wp, sc, Nn, Kk, group, dt); });
    emit(med, static_cast<double>(Nn) * Kk * elem / (med * 1e-3) / 1e9);
    sycl::free(wi, q); sycl::free(wp, q); sycl::free(sc, q);
    return 0;
  }
  if (kernel == "act_quant") {
    void* xx = sycl::malloc_device(rows * dim * elem, q);
    signed char* qo = sycl::malloc_device<signed char>(rows * dim, q);
    float* sc = sycl::malloc_device<float>(rows, q);
    q.memset(xx, 0, rows * dim * elem).wait();
    const double med = time_median([&] { return kernels::act_quant_int8_sycl(q, xx, qo, sc, rows, dim, dt); });
    emit(med, (static_cast<double>(rows) * dim * elem + static_cast<double>(rows) * dim) / (med * 1e-3) / 1e9);
    sycl::free(xx, q); sycl::free(qo, q); sycl::free(sc, q);
    return 0;
  }
  if (kernel == "moe_route") {
    const std::size_t nt = rows, ne = dim; const int kk = 4;
    void* lg = sycl::malloc_device(nt * ne * elem, q);
    int* ids = sycl::malloc_device<int>(nt * kk, q);
    float* w = sycl::malloc_device<float>(nt * kk, q);
    q.memset(lg, 0, nt * ne * elem).wait();
    const double med = time_median([&] { return kernels::moe_route_topk_sycl(q, lg, ids, w, nt, ne, kk, dt); });
    std::cout << "{\"schema_version\":2,\"kernel\":\"moe_route\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"n_tokens\":" << nt << ",\"n_experts\":" << ne
              << ",\"k\":" << kk << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"Mtok_s\":" << (nt / (med * 1e-3) / 1e6) << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(lg, q); sycl::free(ids, q); sycl::free(w, q);
    return 0;
  }
  if (kernel == "dropout") {
    const std::size_t ne = rows * dim;
    void* in = sycl::malloc_device(ne * elem, q);
    void* o = sycl::malloc_device(ne * elem, q);
    q.memset(in, 0, ne * elem).wait();
    const double med = time_median([&] { return kernels::dropout_sycl(q, in, o, ne, 0.1f, 7u, dt); });
    emit(med, 2.0 * ne * elem / (med * 1e-3) / 1e9);
    sycl::free(in, q); sycl::free(o, q);
    return 0;
  }
  if (kernel == "cross_entropy") {
    void* lg = sycl::malloc_device(rows * dim * elem, q);
    int* tg = sycl::malloc_device<int>(rows, q);
    float* ls = sycl::malloc_device<float>(rows, q);
    q.memset(lg, 0, rows * dim * elem).wait(); q.memset(tg, 0, rows * sizeof(int)).wait();
    const double med = time_median([&] { return kernels::cross_entropy_sycl(q, lg, tg, ls, rows, dim, dt); });
    emit(med, static_cast<double>(rows) * dim * elem / (med * 1e-3) / 1e9);
    sycl::free(lg, q); sycl::free(tg, q); sycl::free(ls, q);
    return 0;
  }
  if (kernel == "hadamard") {
    void* in = sycl::malloc_device(rows * dim * elem, q);
    void* o = sycl::malloc_device(rows * dim * elem, q);
    q.memset(in, 0, rows * dim * elem).wait();
    const double med = time_median([&] { return kernels::hadamard_sycl(q, in, o, rows, dim, dt); });
    emit(med, 2.0 * static_cast<double>(rows) * dim * elem / (med * 1e-3) / 1e9);
    sycl::free(in, q); sycl::free(o, q);
    return 0;
  }
  if (kernel == "rope") {
    const std::size_t ne = rows * dim;  // tokens=rows, heads=1, head_dim=dim
    void* x = sycl::malloc_device(ne * elem, q);
    void* o = sycl::malloc_device(ne * elem, q);
    q.memset(x, 0, ne * elem).wait();
    const double med = time_median([&] { return kernels::rope_sycl(q, x, o, rows, 1, dim, 10000.0f, 0, dt); });
    emit(med, 2.0 * ne * elem / (med * 1e-3) / 1e9);
    sycl::free(x, q); sycl::free(o, q);
    return 0;
  }
  if (kernel == "adamw") {
    void* p = sycl::malloc_device(n * elem, q);
    void* g = sycl::malloc_device(n * elem, q);
    void* m = sycl::malloc_device(n * elem, q);
    void* vv = sycl::malloc_device(n * elem, q);
    q.memset(p, 0, n * elem).wait(); q.memset(g, 0, n * elem).wait();
    q.memset(m, 0, n * elem).wait(); q.memset(vv, 0, n * elem).wait();
    const double med = time_median([&] { return kernels::adamw_sycl(q, p, g, m, vv, n, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.5f, 0.5f, dt); });
    std::cout << "{\"schema_version\":2,\"kernel\":\"adamw\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"n\":" << n << ",\"iters\":" << iters
              << ",\"median_ms\":" << med << ",\"gbps\":" << (6.0 * n * elem / (med * 1e-3) / 1e9)
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
              << "\"}" << std::endl;
    sycl::free(p, q); sycl::free(g, q); sycl::free(m, q); sycl::free(vv, q);
    return 0;
  }
  if (kernel == "argmax") {
    void* lg = sycl::malloc_device(rows * dim * elem, q);
    int* o = sycl::malloc_device<int>(rows, q);
    q.memset(lg, 0, rows * dim * elem).wait();
    const double med = time_median([&] { return kernels::argmax_sycl(q, lg, o, rows, dim, dt); });
    emit(med, static_cast<double>(rows * dim) * elem / (med * 1e-3) / 1e9);
    sycl::free(lg, q); sycl::free(o, q);
    return 0;
  }
  if (kernel == "qgemv_int4") {
    // N = rows (output dim), K = dim (contraction), group fixed 128.
    const std::size_t Nn = rows, Kk = dim, group = 128;
    void* w = sycl::malloc_device(Nn * (Kk / 2), q);        // int4 packed
    void* sc = sycl::malloc_device(Nn * (Kk / group) * 2, q);  // fp16 scales
    void* xx = sycl::malloc_device(Kk * elem, q);
    void* yy = sycl::malloc_device(Nn * elem, q);
    q.memset(w, 0, Nn * (Kk / 2)).wait();
    q.memset(sc, 0, Nn * (Kk / group) * 2).wait();
    q.memset(xx, 0, Kk * elem).wait();
    const double med = time_median([&] { return kernels::qgemv_int4_sycl(q, w, sc, xx, yy, Nn, Kk, group, dt); });
    // Weight-bytes bandwidth (the dominant term at batch 1): int4 => N*K/2 bytes.
    const double wbytes = static_cast<double>(Nn) * static_cast<double>(Kk) / 2.0;
    std::cout << "{\"schema_version\":2,\"kernel\":\"qgemv_int4\",\"variant\":\"sycl\",\"dtype\":\""
              << dtype_name(dt) << "\",\"N\":" << Nn << ",\"K\":" << Kk
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"weight_gbps\":" << (wbytes / (med * 1e-3) / 1e9)
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
              << "\"}" << std::endl;
    sycl::free(w, q); sycl::free(sc, q); sycl::free(xx, q); sycl::free(yy, q);
    return 0;
  }

  // Input/output element counts differ for GLU (input [rows,2*dim], out [rows,dim]).
  const std::size_t in_elems = is_glu ? rows * 2 * dim : (is_row ? rows * dim : n);
  const std::size_t out_elems = is_glu ? rows * dim : (is_row ? rows * dim : n);
  const std::size_t n_elems = out_elems;

  void* d_in = sycl::malloc_device(in_elems * elem, q);
  void* d_out = sycl::malloc_device(out_elems * elem, q);
  void* d_w = is_norm ? sycl::malloc_device(dim * elem, q) : nullptr;
  void* d_b = is_norm ? sycl::malloc_device(dim * elem, q) : nullptr;
  q.memset(d_in, 0, in_elems * elem).wait();
  if (is_norm) {
    q.memset(d_w, 0, dim * elem).wait();
    q.memset(d_b, 0, dim * elem).wait();
  }
  const float eps = 1e-6f;

  // Event-returning submit of the selected op + variant.
  auto run_once = [&]() -> sycl::event {
    if (kernel == "gelu") {
      if (variant == Variant::vendor) {
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
        return kernels::gelu_onednn(q, d_in, d_out, n, dt, tanh_approx);
#endif
      }
      return kernels::gelu_sycl(q, d_in, d_out, n, dt, tanh_approx);
    }
    if (kernel == "silu") {
      return kernels::silu_sycl(q, d_in, d_out, n, dt);
    }
    if (kernel == "gelu_backward") {
      return kernels::gelu_backward_sycl(q, d_in, d_in, d_out, n, dt, tanh_approx);
    }
    if (kernel == "glu") {
      return kernels::glu_sycl(q, d_in, d_out, rows, dim, dt, /*mode=*/0);
    }
    if (kernel == "rms_norm") {
      return kernels::rms_norm_sycl(q, d_in, d_w, d_out, rows, dim, eps, dt);
    }
    if (kernel == "layernorm") {
      if (variant == Variant::vendor) {
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
        return kernels::layernorm_onednn(q, d_in, d_w, d_b, d_out, rows, dim, eps, dt);
#endif
      }
      return kernels::layernorm_sycl(q, d_in, d_w, d_b, d_out, rows, dim, eps, dt);
    }
    if (kernel == "softmax") {
      if (variant == Variant::vendor) {
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
        return kernels::softmax_onednn(q, d_in, d_out, rows, dim, dt);
#endif
      }
      return kernels::softmax_sycl(q, d_in, d_out, rows, dim, dt);
    }
    throw std::invalid_argument("unknown kernel: " + kernel);
  };

  for (int i = 0; i < warmup; ++i) run_once().wait();

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    sycl::event ev = run_once();
    ev.wait();
    samples.push_back(event_ms(ev));
  }
  std::sort(samples.begin(), samples.end());
  const double median = samples[samples.size() / 2];
  const double min_ms = samples.front();
  const double max_ms = samples.back();

  // Effective bandwidth. Elementwise: read + write of n. Norms: read x + write
  // out (2*rows*dim) plus the small weight/bias reads (dim, or 2*dim w/ bias).
  double bytes;
  if (is_norm) {
    const double affine = (kernel == "layernorm") ? 2.0 : 1.0;
    bytes = (2.0 * static_cast<double>(rows * dim) + affine * static_cast<double>(dim)) *
            static_cast<double>(elem);
  } else if (is_softmax) {
    bytes = 2.0 * static_cast<double>(rows * dim) * static_cast<double>(elem);
  } else if (is_glu) {
    bytes = 3.0 * static_cast<double>(rows * dim) * static_cast<double>(elem);
  } else if (kernel == "gelu_backward") {
    bytes = 3.0 * static_cast<double>(n) * static_cast<double>(elem);
  } else {
    bytes = static_cast<double>(n) * static_cast<double>(elem) * 2.0;
  }
  const double gbps = bytes / (median * 1e-3) / 1e9;

  sycl::free(d_in, q);
  sycl::free(d_out, q);
  if (d_w) sycl::free(d_w, q);
  if (d_b) sycl::free(d_b, q);

  std::cout << "{\"schema_version\":2"
            << ",\"kernel\":\"" << kernel << "\""
            << ",\"variant\":\"" << variant_name(variant) << "\""
            << ",\"requested_variant\":\"" << variant_name(requested) << "\""
            << ",\"approx\":\"" << (tanh_approx ? "tanh" : "erf") << "\""
            << ",\"dtype\":\"" << dtype_name(dt) << "\""
            << ",\"n\":" << n_elems
            << ",\"rows\":" << rows
            << ",\"dim\":" << dim
            << ",\"iters\":" << iters
            << ",\"warmup\":" << warmup
            << ",\"median_ms\":" << median
            << ",\"min_ms\":" << min_ms
            << ",\"max_ms\":" << max_ms
            << ",\"gbps\":" << gbps
            << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>()
            << "\"}" << std::endl;
  return 0;
}
