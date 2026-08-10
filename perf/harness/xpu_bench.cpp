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

#include "norms/qk_norm_rope/qk_norm_rope_kernel.hpp"
#include "activations/glu_quant/glu_quant_kernel.hpp"
#include "attention/paged_attention/paged_attention_kernel.hpp"
#include "serving/mqa_logits/mqa_logits_kernel.hpp"
#include "norms/norm_quant/norm_quant_kernel.hpp"
#include "quantization/turboquant/turboquant_kernel.hpp"
#include "quantization/turboquant/turboquant_tables.hpp"
#include "ssm/causal_conv1d/causal_conv1d_kernel.hpp"
#include "ssm/ssd/ssd_kernel.hpp"
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
#include "quantization/w4a16_gemm/w4a16_gemm_kernel.hpp"
#include "sampling/argmax/argmax_kernel.hpp"
#include "sampling/sample/sample_kernel.hpp"
#include "linear_attention/linear_attn/linear_attn_kernel.hpp"
#include "linear_attention/qwen_gdn_decode/qwen_gdn_kernel.hpp"
#include "moe/moe_route/moe_route_kernel.hpp"
#include "linear_attention/gated_delta_rule/gated_delta_rule_kernel.hpp"
#include "moe/grouped_qgemm/grouped_qgemm_kernel.hpp"
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

// Baseline for the glu_gelu_f16 A/B: GEGLU (tanh-gelu gate x value) writing the
// storage dtype, matching the fused kernels math exactly so the only delta is
// the [rows,d] scratch round-trip + f16 convert that the fusion folds away.
inline float bench_gelu_tanh(float x) {
  constexpr float a = 0.044715f, s = 0.79788456080286535587989211986876f;
  return 0.5f * x * (1.0f + sycl::tanh(s * x * (1.0f + a * x * x)));
}
template <typename T>
sycl::event glu_gelu_tanh_dt(sycl::queue& q, const T* x, T* out, std::size_t rows,
                             std::size_t d) {
  return q.parallel_for(sycl::range<2>(rows, d), [=](sycl::id<2> idx) {
    const std::size_t row = idx[0], col = idx[1];
    const T* gate = x + row * 2 * d;
    const T* val = gate + d;
    out[row * d + col] = static_cast<T>(
        bench_gelu_tanh(static_cast<float>(gate[col])) * static_cast<float>(val[col]));
  });
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
  std::size_t window = 256;  // attn_swa symmetric sliding-window size
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
    else if (a == "--window") window = std::stoull(next());
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
  if (kernel == "nvfp4_moe_relu2") {
    // NemotronH-style ungated ReLU2 experts: w1 is a single up-projection
    // [E, I, K/2]. --M tokens, --N experts, --rows top_k, --dim intermediate,
    // --K hidden. --approx {fused,split}.
    const std::size_t experts = N;
    const std::size_t top_k = rows;
    const std::size_t intermediate = dim;
    const std::size_t pairs = M * top_k;
    void *hidden = sycl::malloc_device(M * K * elem, q);
    int *expert_ids = sycl::malloc_device<int>(pairs, q);
    float *router_weights = sycl::malloc_device<float>(pairs, q);
    void *w1 = sycl::malloc_device(experts * intermediate * K / 2, q);
    void *w1_scales = sycl::malloc_device(experts * intermediate * K / 16, q);
    float *w1_global = sycl::malloc_device<float>(experts, q);
    void *w2 = sycl::malloc_device(experts * K * intermediate / 2, q);
    void *w2_scales = sycl::malloc_device(experts * K * intermediate / 16, q);
    float *w2_global = sycl::malloc_device<float>(experts, q);
    float *scratch = sycl::malloc_device<float>(pairs * intermediate, q);
    float *output = sycl::malloc_device<float>(M * K, q);
    q.memset(hidden, 0, M * K * elem).wait();
    q.memset(expert_ids, 0, pairs * sizeof(int)).wait();
    q.fill(router_weights, 1.0f / static_cast<float>(top_k), pairs).wait();
    q.memset(w1, 0, experts * intermediate * K / 2).wait();
    q.memset(w1_scales, 0x38, experts * intermediate * K / 16).wait();
    q.fill(w1_global, 1.0f, experts).wait();
    q.memset(w2, 0, experts * K * intermediate / 2).wait();
    q.memset(w2_scales, 0x38, experts * K * intermediate / 16).wait();
    q.fill(w2_global, 1.0f, experts).wait();
    const bool split = approx_s == "split";
    auto once = [&] {
      const sycl::event zeroed = q.memset(output, 0, M * K * sizeof(float));
      if (split) {
        kernels::nvfp4_moe_relu2_split_sycl(q, hidden, expert_ids, router_weights, w1, w1_scales,
                                            w1_global, w2, w2_scales, w2_global, scratch, output,
                                            M, experts, top_k, K, intermediate, true, dt, zeroed);
      } else {
        kernels::nvfp4_moe_relu2_fused_sycl(q, hidden, expert_ids, router_weights, w1, w1_scales,
                                            w1_global, w2, w2_scales, w2_global, output, M,
                                            experts, top_k, K, intermediate, true, dt, zeroed);
      }
    };
    const DeviceTiming timing = time_device_batches(once);
    const double median = timing.median_ms;
    const double fp4_weight_bytes =
        static_cast<double>(pairs) * (intermediate * K / 2.0 + K * intermediate / 2.0);
    const double weight_gbps = fp4_weight_bytes / (median * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"nvfp4_moe_relu2\","
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
    sycl::free(w1, q);
    sycl::free(w1_scales, q);
    sycl::free(w1_global, q);
    sycl::free(w2, q);
    sycl::free(w2_scales, q);
    sycl::free(w2_global, q);
    sycl::free(scratch, q);
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
  if (kernel == "ssd_decode") {
    // NemotronH-shaped Mamba-2 SSD decode step (TP2 slice by default):
    // batch = --M sequences, --N state slots, nheads 64, headdim 64,
    // dstate 128, ngroups 4, f32 state, act dtype = --dtype.
    const std::size_t batch = M;
    const std::size_t slots = std::max<std::size_t>(N, batch);
    const std::size_t nheads = 64, headdim = 64, dstate = 128, ngroups = 4;
    const std::int64_t ss3 = 1, ss2 = dstate,
                       ss1 = static_cast<std::int64_t>(headdim) * dstate,
                       ss0 = static_cast<std::int64_t>(nheads) * headdim * dstate;
    float *state = sycl::malloc_device<float>(slots * nheads * headdim * dstate, q);
    void *x = sycl::malloc_device(batch * nheads * headdim * elem, q);
    void *B = sycl::malloc_device(batch * ngroups * dstate * elem, q);
    void *C = sycl::malloc_device(batch * ngroups * dstate * elem, q);
    void *out = sycl::malloc_device(batch * nheads * headdim * elem, q);
    float *dt_raw = sycl::malloc_device<float>(batch * nheads, q);
    float *A = sycl::malloc_shared<float>(nheads, q);
    float *dt_bias = sycl::malloc_device<float>(nheads, q);
    float *D = sycl::malloc_device<float>(nheads * headdim, q);
    int *idx = sycl::malloc_shared<int>(batch, q);
    q.memset(state, 0, slots * nheads * headdim * dstate * sizeof(float)).wait();
    q.memset(x, 0, batch * nheads * headdim * elem).wait();
    q.memset(B, 0, batch * ngroups * dstate * elem).wait();
    q.memset(C, 0, batch * ngroups * dstate * elem).wait();
    q.memset(dt_raw, 0, batch * nheads * sizeof(float)).wait();
    q.memset(dt_bias, 0, nheads * sizeof(float)).wait();
    q.memset(D, 0, nheads * headdim * sizeof(float)).wait();
    for (std::size_t h = 0; h < nheads; ++h) A[h] = -1.0f;
    for (std::size_t i = 0; i < batch; ++i) idx[i] = static_cast<int>(i % slots);
    auto once = [&] {
      kernels::ssd_decode_sycl(q, state, x, dt_raw, A, B, C, D, dt_bias, idx,
                               idx, out, true, batch, nheads, headdim, dstate,
                               ngroups, slots, ss0, ss1, ss2, ss3, dt,
                               DType::f32);
    };
    const DeviceTiming timing = time_device_batches(once);
    std::cout << "{\"schema_version\":2,\"kernel\":\"ssd_decode\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"batch\":" << batch << ",\"slots\":" << slots
              << ",\"iters\":" << iters << ",\"median_ms\":" << timing.median_ms
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(state, q); sycl::free(x, q); sycl::free(B, q); sycl::free(C, q);
    sycl::free(out, q); sycl::free(dt_raw, q); sycl::free(A, q);
    sycl::free(dt_bias, q); sycl::free(D, q); sycl::free(idx, q);
    return 0;
  }
  if (kernel == "gated_delta_rule") {
    // Qwen3.6 shape: Hk16/dk128, Hv32/dv128; --M tokens, one sequence.
    const std::size_t T2 = M, Hk = 16, dk2 = 128, Hv = 32, dv = 128, slots = 4;
    void *Q = sycl::malloc_device(T2 * Hk * dk2 * elem, q);
    void *Kb = sycl::malloc_device(T2 * Hk * dk2 * elem, q);
    void *V = sycl::malloc_device(T2 * Hv * dv * elem, q);
    void *out = sycl::malloc_device(T2 * Hv * dv * elem, q);
    float *b = sycl::malloc_device<float>(T2 * Hv, q);
    float *a = sycl::malloc_device<float>(T2 * Hv, q);
    float *Al = sycl::malloc_shared<float>(Hv, q);
    float *db = sycl::malloc_device<float>(Hv, q);
    float *st = sycl::malloc_device<float>(slots * Hv * dv * dk2, q);
    auto *cs = sycl::malloc_shared<std::int32_t>(2, q);
    auto *cs1 = sycl::malloc_shared<std::int32_t>(2, q);
    auto *si = sycl::malloc_shared<std::int32_t>(1, q);
    bool *hi2 = sycl::malloc_shared<bool>(1, q);
    q.memset(Q, 0, T2 * Hk * dk2 * elem).wait();
    q.memset(Kb, 0, T2 * Hk * dk2 * elem).wait();
    q.memset(V, 0, T2 * Hv * dv * elem).wait();
    q.memset(b, 0, T2 * Hv * sizeof(float)).wait();
    q.memset(a, 0, T2 * Hv * sizeof(float)).wait();
    q.memset(db, 0, Hv * sizeof(float)).wait();
    q.memset(st, 0, slots * Hv * dv * dk2 * sizeof(float)).wait();
    for (std::size_t h2 = 0; h2 < Hv; ++h2) Al[h2] = -0.5f;
    cs[0] = 0; cs[1] = static_cast<std::int32_t>(T2);
    cs1[0] = 0; cs1[1] = 1;
    si[0] = 0; hi2[0] = true;
    auto once = [&] {
      kernels::gated_delta_rule_varlen_sycl(q, Q, Kb, V, b, a, Al, db, st, out,
                                            cs, si, hi2, 1, Hk, dk2, Hv, dv,
                                            slots, dt, DType::f32);
    };
    const DeviceTiming tp = time_device_batches(once);
    // Sequential-decode yardstick: T calls of one token each.
    auto seq_once = [&] {
      for (std::size_t t = 0; t < T2; ++t)
        kernels::gated_delta_rule_varlen_sycl(q, Q, Kb, V, b, a, Al, db, st,
                                              out, cs1, si, hi2, 1, Hk, dk2,
                                              Hv, dv, slots, dt, DType::f32);
    };
    for (int i = 0; i < 2; ++i) seq_once();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) seq_once();
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    const double seq_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / 3;
    std::cout << "{\"schema_version\":2,\"kernel\":\"gated_delta_rule\","
              << "\"dtype\":\"" << dtype_name(dt) << "\",\"tokens\":" << T2
              << ",\"iters\":" << iters << ",\"varlen_median_ms\":" << tp.median_ms
              << ",\"seq_decode_ms\":" << seq_ms
              << ",\"speedup\":" << seq_ms / tp.median_ms << "}" << std::endl;
    sycl::free(Q, q); sycl::free(Kb, q); sycl::free(V, q); sycl::free(out, q);
    sycl::free(b, q); sycl::free(a, q); sycl::free(Al, q); sycl::free(db, q);
    sycl::free(st, q); sycl::free(cs, q); sycl::free(cs1, q); sycl::free(si, q);
    sycl::free(hi2, q);
    return 0;
  }
  if (kernel == "moe_grouped_qgemm") {
    // --M rows total (spread evenly over --n experts... uses --N experts),
    // --dim N, --K K, nvfp4 weights.
    const std::size_t E2 = N && N <= 256 ? N : 64;
    const std::size_t Mt = M, Ncols = dim, Kd = K;
    auto *A = sycl::malloc_device(Mt * Kd * elem, q);
    auto *W = sycl::malloc_device<std::uint8_t>(E2 * Ncols * Kd / 2, q);
    auto *S = sycl::malloc_device<std::uint8_t>(E2 * Ncols * (Kd / 16), q);
    float *G = sycl::malloc_device<float>(E2, q);
    auto *C = sycl::malloc_device(Mt * Ncols * elem, q);
    auto *rpe = sycl::malloc_shared<std::int32_t>(E2, q);
    q.memset(A, 0, Mt * Kd * elem).wait();
    q.memset(W, 0x22, E2 * Ncols * Kd / 2).wait();
    q.memset(S, 0x38, E2 * Ncols * (Kd / 16)).wait();
    q.fill(G, 0.02f, E2).wait();
    for (std::size_t e = 0; e < E2; ++e)
      rpe[e] = static_cast<std::int32_t>(Mt / E2 + (e < Mt % E2 ? 1 : 0));
    auto once = [&] {
      kernels::moe_grouped_qgemm_sycl(q, A, W, S, G, C, rpe, Mt, Ncols, Kd,
                                      E2, 16, 2, dt);
    };
    const DeviceTiming timing = time_device_batches(once);
    const double tflops = 2.0 * Mt * Ncols * Kd / (timing.median_ms * 1e-3) / 1e12;
    const double wgb = (E2 * Ncols * Kd / 2.0) / 1e9 / (timing.median_ms * 1e-3);
    std::cout << "{\"schema_version\":2,\"kernel\":\"moe_grouped_qgemm\",\"fmt\":\"nvfp4\","
              << "\"dtype\":\"" << dtype_name(dt) << "\",\"M\":" << Mt << ",\"N\":" << Ncols
              << ",\"K\":" << Kd << ",\"E\":" << E2 << ",\"iters\":" << iters
              << ",\"median_ms\":" << timing.median_ms << ",\"tflops\":" << tflops
              << ",\"weight_gbps\":" << wgb << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(A, q); sycl::free(W, q); sycl::free(S, q); sycl::free(G, q);
    sycl::free(C, q); sycl::free(rpe, q);
    return 0;
  }
  if (kernel == "paged_attention_decode") {
    // --M batch, --N context, --dim head_dim, --rows heads (kv heads = /4),
    // --K splits, page 64.
    const std::size_t B = M, ctx = N, hd = dim, H = rows ? rows : 32;
    const std::size_t Hkv = H >= 4 ? H / 4 : 1;
    const int splits = K >= 1 && K <= 64 ? static_cast<int>(K) : 4;
    const std::size_t page = 64;
    const std::size_t mp = (ctx + page - 1) / page;
    const std::size_t pe = page * Hkv * hd;
    const std::size_t np = B * mp;
    void *Q = sycl::malloc_device(B * H * hd * elem, q);
    void *O = sycl::malloc_device(B * H * hd * elem, q);
    void *kc = sycl::malloc_device(np * pe * elem, q);
    void *vc = sycl::malloc_device(np * pe * elem, q);
    float *tmp = sycl::malloc_device<float>(B * H * splits * hd, q);
    float *es = sycl::malloc_device<float>(B * H * splits, q);
    float *ml = sycl::malloc_device<float>(B * H * splits, q);
    auto *bt = sycl::malloc_shared<std::int32_t>(B * mp, q);
    auto *sl = sycl::malloc_shared<std::int32_t>(B, q);
    q.memset(Q, 0, B * H * hd * elem).wait();
    q.memset(kc, 0, np * pe * elem).wait();
    q.memset(vc, 0, np * pe * elem).wait();
    for (std::size_t b = 0; b < B; ++b) {
      sl[b] = static_cast<std::int32_t>(ctx);
      for (std::size_t p2 = 0; p2 < mp; ++p2)
        bt[b * mp + p2] = static_cast<std::int32_t>(b * mp + p2);
    }
    auto once = [&] {
      kernels::paged_attention_decode_sycl(q, Q, kc, vc, O, tmp, es, ml, bt,
                                           sl, B, H, Hkv, hd, page, mp, pe,
                                           splits, 0.088f, -1, nullptr,
                                           nullptr, nullptr, dt, 0);
    };
    const DeviceTiming timing = time_device_batches(once);
    const double kv_gb = 2.0 * B * ctx * Hkv * hd * elem / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"paged_attention_decode\","
              << "\"dtype\":\"" << dtype_name(dt) << "\",\"batch\":" << B
              << ",\"ctx\":" << ctx << ",\"heads\":" << H << ",\"d\":" << hd
              << ",\"splits\":" << splits << ",\"iters\":" << iters
              << ",\"median_ms\":" << timing.median_ms
              << ",\"kv_gbps\":" << kv_gb / (timing.median_ms * 1e-3)
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(Q, q); sycl::free(O, q); sycl::free(kc, q); sycl::free(vc, q);
    sycl::free(tmp, q); sycl::free(es, q); sycl::free(ml, q); sycl::free(bt, q);
    sycl::free(sl, q);
    return 0;
  }
  if (kernel == "mqa_logits") {
    // --M q tokens, --rows heads, --dim D, --N kv positions.
    const std::size_t S = M, H = rows, D2 = dim, Skv = N;
    auto *qf = sycl::malloc_device<std::uint8_t>(S * H * D2, q);
    auto *kf = sycl::malloc_device<std::uint8_t>(Skv * D2, q);
    float *kscale = sycl::malloc_device<float>(Skv, q);
    float *w = sycl::malloc_device<float>(S * H, q);
    auto *ks = sycl::malloc_shared<std::int32_t>(S, q);
    auto *ke = sycl::malloc_shared<std::int32_t>(S, q);
    float *logits = sycl::malloc_device<float>(S * Skv, q);
    q.memset(qf, 0x30, S * H * D2).wait();
    q.memset(kf, 0x30, Skv * D2).wait();
    q.memset(kscale, 0, Skv * sizeof(float)).wait();
    q.memset(w, 0, S * H * sizeof(float)).wait();
    for (std::size_t s2 = 0; s2 < S; ++s2) { ks[s2] = 0; ke[s2] = static_cast<int>(Skv); }
    auto once = [&] {
      kernels::mqa_logits_sycl(q, qf, kf, kscale, w, ks, ke, logits, S, H, D2, Skv);
    };
    const DeviceTiming timing = time_device_batches(once);
    const double tflops = 2.0 * S * H * D2 * Skv / (timing.median_ms * 1e-3) / 1e12;
    std::cout << "{\"schema_version\":2,\"kernel\":\"mqa_logits\",\"variant\":\"sycl\","
              << "\"S\":" << S << ",\"H\":" << H << ",\"D\":" << D2 << ",\"Skv\":" << Skv
              << ",\"iters\":" << iters << ",\"median_ms\":" << timing.median_ms
              << ",\"tflops\":" << tflops << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(qf, q); sycl::free(kf, q); sycl::free(kscale, q); sycl::free(w, q);
    sycl::free(ks, q); sycl::free(ke, q); sycl::free(logits, q);
    return 0;
  }
  if (kernel == "glu_quant") {
    // --rows x [rows, 2*--dim] input, --approx {fp8,mxfp4}, --window group.
    const int mode = approx_s == "mxfp4" ? 1 : 0;
    const std::size_t group = mode == 1 ? 32 : (window ? window : 128);
    void *x = sycl::malloc_device(rows * 2 * dim * elem, q);
    auto *outq = sycl::malloc_device<std::uint8_t>(rows * dim, q);
    float *scales = sycl::malloc_device<float>(rows * (dim / group), q);
    q.memset(x, 0, rows * 2 * dim * elem).wait();
    auto once = [&] {
      kernels::glu_quant_sycl(q, x, outq, scales, rows, dim, group, mode, dt);
    };
    const DeviceTiming timing = time_device_batches(once);
    std::cout << "{\"schema_version\":2,\"kernel\":\"glu_quant\",\"variant\":\"sycl\","
              << "\"dtype\":\"" << dtype_name(dt) << "\",\"mode\":\"" << approx_s
              << "\",\"rows\":" << rows << ",\"d\":" << dim << ",\"group\":" << group
              << ",\"iters\":" << iters << ",\"median_ms\":" << timing.median_ms
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(x, q); sycl::free(outq, q); sycl::free(scales, q);
    return 0;
  }
  if (kernel == "norm_quant") {
    // --rows x --dim, --approx {static,dynamic,mxfp4}, optional residual via
    // --window 1.
    const int mode = approx_s == "dynamic" ? 1 : (approx_s == "mxfp4" ? 2 : 0);
    const bool with_res = window == 1;
    void *x = sycl::malloc_device(rows * dim * elem, q);
    void *res = sycl::malloc_device(rows * dim * elem, q);
    void *w = sycl::malloc_device(dim * elem, q);
    auto *outq = sycl::malloc_device<std::uint8_t>(rows * dim, q);
    float *ss = sycl::malloc_shared<float>(1, q);
    float *os = sycl::malloc_device<float>(rows * (dim / 32 + 1), q);
    q.memset(x, 0, rows * dim * elem).wait();
    q.memset(res, 0, rows * dim * elem).wait();
    q.memset(w, 0, dim * elem).wait();
    ss[0] = 0.01f;
    auto once = [&] {
      kernels::norm_quant_sycl(q, x, with_res ? res : nullptr, w, outq, ss, os,
                               rows, dim, 1e-6f, mode, dt);
    };
    const DeviceTiming timing = time_device_batches(once);
    std::cout << "{\"schema_version\":2,\"kernel\":\"norm_quant\",\"variant\":\"sycl\","
              << "\"dtype\":\"" << dtype_name(dt) << "\",\"mode\":\"" << approx_s
              << "\",\"residual\":" << (with_res ? 1 : 0) << ",\"rows\":" << rows
              << ",\"dim\":" << dim << ",\"iters\":" << iters
              << ",\"median_ms\":" << timing.median_ms << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(x, q); sycl::free(res, q); sycl::free(w, q); sycl::free(outq, q);
    sycl::free(ss, q); sycl::free(os, q);
    return 0;
  }
  if (kernel == "group_rms_norm_gated") {
    // --rows x --dim hidden, --N groups.
    const std::size_t groups = std::max<std::size_t>(1, N <= dim ? N : 1);
    void *x = sycl::malloc_device(rows * dim * elem, q);
    void *gate = sycl::malloc_device(rows * dim * elem, q);
    void *w = sycl::malloc_device(dim * elem, q);
    void *out = sycl::malloc_device(rows * dim * elem, q);
    q.memset(x, 0, rows * dim * elem).wait();
    q.memset(gate, 0, rows * dim * elem).wait();
    q.memset(w, 0, dim * elem).wait();
    auto once = [&] {
      kernels::group_rms_norm_gated_sycl(q, x, gate, w, out, rows, dim, groups,
                                         1e-6f, true, dt);
    };
    const DeviceTiming timing = time_device_batches(once);
    const double gbps = 3.0 * static_cast<double>(rows * dim * elem) /
                        (timing.median_ms * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"group_rms_norm_gated\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"rows\":" << rows << ",\"dim\":" << dim
              << ",\"groups\":" << groups << ",\"iters\":" << iters
              << ",\"median_ms\":" << timing.median_ms << ",\"gbps\":" << gbps
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(x, q); sycl::free(gate, q); sycl::free(w, q); sycl::free(out, q);
    return 0;
  }
  if (kernel == "turboquant") {
    // KV codec: --M tokens, 8 kv heads, --dim head_size, key/value bits from
    // --K (packed as kb*10+vb, default 44 => kb=4 vb=4). Encode + decode
    // timings per call.
    const std::size_t tokens = M, heads = 8, hs = dim;
    const int kb = K >= 22 && K <= 88 ? static_cast<int>(K / 10) : 4;
    const int vb = K >= 22 && K <= 88 ? static_cast<int>(K % 10) : 4;
    const std::size_t groups = hs / 32;
    const std::size_t kbytes = (hs * kb + 7) / 8, vbytes = (hs * vb + 7) / 8;
    const std::size_t rows = tokens * heads;
    void *key = sycl::malloc_device(tokens * heads * hs * elem, q);
    void *value = sycl::malloc_device(tokens * heads * hs * elem, q);
    auto *kc = sycl::malloc_device<std::uint8_t>(rows * kbytes, q);
    auto *vc = sycl::malloc_device<std::uint8_t>(rows * vbytes, q);
    auto *ks = sycl::malloc_device<sycl::half>(rows * groups, q);
    auto *vs = sycl::malloc_device<sycl::half>(rows * groups, q);
    auto *vz = sycl::malloc_device<sycl::half>(rows * groups, q);
    auto *slots = sycl::malloc_shared<std::int64_t>(tokens, q);
    float *cent = sycl::malloc_shared<float>(1u << kb, q);
    float *signs = sycl::malloc_shared<float>(hs, q);
    float *kout = sycl::malloc_device<float>(tokens * heads * hs, q);
    float *vout = sycl::malloc_device<float>(tokens * heads * hs, q);
    q.memset(key, 0, tokens * heads * hs * elem).wait();
    q.memset(value, 0, tokens * heads * hs * elem).wait();
    for (std::size_t i = 0; i < tokens; ++i) slots[i] = static_cast<std::int64_t>(i);
    const auto cv = quixicore::xpu::turboquant::lloyd_max_centroids(hs, kb);
    for (std::size_t i = 0; i < cv.size(); ++i) cent[i] = cv[i];
    for (std::size_t i = 0; i < hs; ++i) signs[i] = (i & 1) ? -1.0f : 1.0f;
    auto enc = [&] {
      kernels::turboquant_encode_sycl(q, key, value, kc, vc, ks, vs, vz, slots,
                                      cent, signs, tokens, heads, hs, kb, vb,
                                      0, dt);
    };
    auto dec = [&] {
      kernels::turboquant_decode_sycl(q, kc, vc, ks, vs, vz, slots, cent,
                                      signs, kout, vout, tokens, heads, hs, kb,
                                      vb, 0);
    };
    const DeviceTiming te = time_device_batches(enc);
    const DeviceTiming td = time_device_batches(dec);
    std::cout << "{\"schema_version\":2,\"kernel\":\"turboquant\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"tokens\":" << tokens << ",\"heads\":" << heads
              << ",\"head_size\":" << hs << ",\"key_bits\":" << kb
              << ",\"value_bits\":" << vb << ",\"iters\":" << iters
              << ",\"encode_median_ms\":" << te.median_ms
              << ",\"decode_median_ms\":" << td.median_ms << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(key, q); sycl::free(value, q); sycl::free(kc, q);
    sycl::free(vc, q); sycl::free(ks, q); sycl::free(vs, q); sycl::free(vz, q);
    sycl::free(slots, q); sycl::free(cent, q); sycl::free(signs, q);
    sycl::free(kout, q); sycl::free(vout, q);
    return 0;
  }
  if (kernel == "all_reduce") {
    // In-process capturable P2P all-reduce across all visible GPUs (cap 4):
    // --n elements of --dtype per rank; wall-clock per collective.
    std::vector<sycl::device> ar_devices;
    for (const auto &p : sycl::platform::get_platforms()) {
      std::vector<sycl::device> gpus;
      for (auto &d : p.get_devices())
        if (d.is_gpu()) gpus.push_back(d);
      if (gpus.size() > ar_devices.size()) ar_devices = std::move(gpus);
    }
    if (ar_devices.size() < 2) {
      std::cerr << "all_reduce bench needs >1 GPU\n";
      return 0;
    }
    const int world = static_cast<int>(std::min<std::size_t>(ar_devices.size(), 4));
    ar_devices.resize(static_cast<std::size_t>(world));
    for (int a = 0; a < world; ++a)
      for (int b = 0; b < world; ++b) {
        if (a == b) continue;
        try { ar_devices[a].ext_oneapi_enable_peer_access(ar_devices[b]); }
        catch (const sycl::exception &) {}
      }
    sycl::context ar_ctx(ar_devices);
    std::vector<sycl::queue> ar_qs;
    for (int g = 0; g < world; ++g) ar_qs.emplace_back(ar_ctx, ar_devices[g]);
    const std::size_t bytes = n * elem;
    const std::size_t region_bytes = ops::all_reduce_region_bytes(bytes);
    std::vector<void *> regions(static_cast<std::size_t>(world));
    std::vector<void *> in_dev(world), out_dev(world);
    for (int g = 0; g < world; ++g) {
      regions[g] = sycl::malloc_device(region_bytes, ar_qs[g]);
      in_dev[g] = sycl::malloc_device(bytes, ar_qs[g]);
      out_dev[g] = sycl::malloc_device(bytes, ar_qs[g]);
      ar_qs[g].memset(regions[g], 0, region_bytes);
      ar_qs[g].memset(in_dev[g], 0, bytes);
    }
    for (auto &qq : ar_qs) qq.wait();
    auto run_once = [&] {
      for (int g = 0; g < world; ++g)
        ops::all_reduce(ar_qs[g], in_dev[g], out_dev[g], regions.data(), g,
                        world, n, dt, 65536, Variant::sycl, /*blocking=*/false);
      for (auto &qq : ar_qs) qq.wait();
    };
    for (int i = 0; i < warmup; ++i) run_once();
    std::vector<double> samples;
    for (int s = 0; s < 5; ++s) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) run_once();
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    std::sort(samples.begin(), samples.end());
    std::cout << "{\"schema_version\":2,\"kernel\":\"all_reduce\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"world\":" << world << ",\"numel\":" << n
              << ",\"bytes\":" << bytes << ",\"iters\":" << iters
              << ",\"median_ms\":" << samples[samples.size() / 2]
              << ",\"min_ms\":" << samples.front() << ",\"max_ms\":" << samples.back()
              << ",\"algo\":\"" << (bytes < 65536 ? "one-shot" : "two-shot")
              << "\"}" << std::endl;
    for (int g = 0; g < world; ++g) {
      sycl::free(regions[g], ar_qs[g]);
      sycl::free(in_dev[g], ar_qs[g]);
      sycl::free(out_dev[g], ar_qs[g]);
    }
    return 0;
  }
  if (kernel == "ssd_prefill") {
    // Varlen SSD prefill: --M packed tokens split across --rows equal
    // sequences, NemotronH TP2 slice (nheads 64, headdim 64, dstate 128,
    // ngroups 4), f32 states, act dtype = --dtype.
    const std::size_t tokens = M;
    const std::size_t nseq = std::max<std::size_t>(1, std::min(rows, tokens));
    const std::size_t nheads = 64, headdim = 64, dstate = 128, ngroups = 4;
    void *x = sycl::malloc_device(tokens * nheads * headdim * elem, q);
    void *B = sycl::malloc_device(tokens * ngroups * dstate * elem, q);
    void *C = sycl::malloc_device(tokens * ngroups * dstate * elem, q);
    void *out = sycl::malloc_device(tokens * nheads * headdim * elem, q);
    float *dt_raw = sycl::malloc_device<float>(tokens * nheads, q);
    float *A = sycl::malloc_shared<float>(nheads, q);
    float *dt_bias = sycl::malloc_device<float>(nheads, q);
    float *D = sycl::malloc_device<float>(nheads * headdim, q);
    int *qsl = sycl::malloc_shared<int>(nseq + 1, q);
    float *vstates = sycl::malloc_device<float>(nseq * nheads * headdim * dstate, q);
    q.memset(x, 0, tokens * nheads * headdim * elem).wait();
    q.memset(B, 0, tokens * ngroups * dstate * elem).wait();
    q.memset(C, 0, tokens * ngroups * dstate * elem).wait();
    q.memset(dt_raw, 0, tokens * nheads * sizeof(float)).wait();
    q.memset(dt_bias, 0, nheads * sizeof(float)).wait();
    q.memset(D, 0, nheads * headdim * sizeof(float)).wait();
    for (std::size_t h = 0; h < nheads; ++h) A[h] = -1.0f;
    for (std::size_t s = 0; s <= nseq; ++s)
      qsl[s] = static_cast<int>(s * tokens / nseq);
    auto once = [&] {
      kernels::ssd_prefill_sycl(q, x, dt_raw, A, B, C, D, dt_bias, qsl,
                                nullptr, out, vstates, true, 0.0f, 1e9f, nseq,
                                nheads, headdim, dstate, ngroups, dt,
                                DType::f32);
    };
    const DeviceTiming timing = time_device_batches(once);
    std::cout << "{\"schema_version\":2,\"kernel\":\"ssd_prefill\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"tokens\":" << tokens << ",\"seqs\":" << nseq
              << ",\"iters\":" << iters << ",\"median_ms\":" << timing.median_ms
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(x, q); sycl::free(B, q); sycl::free(C, q); sycl::free(out, q);
    sycl::free(dt_raw, q); sycl::free(A, q); sycl::free(dt_bias, q);
    sycl::free(D, q); sycl::free(qsl, q); sycl::free(vstates, q);
    return 0;
  }
  if (kernel == "causal_conv1d_prefill") {
    // Varlen conv prefill: --M packed tokens in one sequence, --dim channels,
    // width-4 taps, dim-major layout, f32 state.
    const std::size_t tokens = M;
    const std::size_t taps = 4, state_len = taps - 1, slots = 4;
    const std::int64_t xs0 = static_cast<std::int64_t>(tokens), xs1 = 1;
    const std::int64_t cs0 = static_cast<std::int64_t>(dim) * state_len,
                       cs1 = state_len, cs2 = 1;
    float *state = sycl::malloc_device<float>(slots * dim * state_len, q);
    void *x = sycl::malloc_device(dim * tokens * elem, q);
    void *out = sycl::malloc_device(dim * tokens * elem, q);
    void *w = sycl::malloc_device(dim * taps * elem, q);
    void *bias = sycl::malloc_device(dim * elem, q);
    int *qsl = sycl::malloc_shared<int>(2, q);
    int *idx = sycl::malloc_shared<int>(1, q);
    bool *hinit = sycl::malloc_shared<bool>(1, q);
    q.memset(state, 0, slots * dim * state_len * sizeof(float)).wait();
    q.memset(x, 0, dim * tokens * elem).wait();
    q.memset(w, 0, dim * taps * elem).wait();
    q.memset(bias, 0, dim * elem).wait();
    qsl[0] = 0; qsl[1] = static_cast<int>(tokens); idx[0] = 0; hinit[0] = true;
    auto once = [&] {
      kernels::causal_conv1d_prefill_sycl(q, state, x, w, bias, qsl, idx,
                                          hinit, out, true, tokens, 1, dim,
                                          state_len, taps, slots, xs0, xs1,
                                          xs0, xs1, cs0, cs1, cs2, dt,
                                          DType::f32);
    };
    const DeviceTiming timing = time_device_batches(once);
    std::cout << "{\"schema_version\":2,\"kernel\":\"causal_conv1d_prefill\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"tokens\":" << tokens << ",\"dim\":" << dim
              << ",\"iters\":" << iters << ",\"median_ms\":" << timing.median_ms
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(state, q); sycl::free(x, q); sycl::free(out, q);
    sycl::free(w, q); sycl::free(bias, q); sycl::free(qsl, q);
    sycl::free(idx, q); sycl::free(hinit, q);
    return 0;
  }
  if (kernel == "causal_conv1d_decode") {
    // Mamba-2 conv decode step: batch = --M rows, --dim channels, width-4
    // taps, --N state slots, f32 state, act dtype = --dtype.
    const std::size_t batch = M;
    const std::size_t slots = std::max<std::size_t>(N, batch);
    const std::size_t taps = 4, state_len = taps - 1;
    const std::int64_t cs0 = static_cast<std::int64_t>(dim) * state_len,
                       cs1 = state_len, cs2 = 1;
    float *state = sycl::malloc_device<float>(slots * dim * state_len, q);
    void *x = sycl::malloc_device(batch * dim * elem, q);
    void *w = sycl::malloc_device(dim * taps * elem, q);
    void *bias = sycl::malloc_device(dim * elem, q);
    void *out = sycl::malloc_device(batch * dim * elem, q);
    int *idx = sycl::malloc_shared<int>(batch, q);
    q.memset(state, 0, slots * dim * state_len * sizeof(float)).wait();
    q.memset(x, 0, batch * dim * elem).wait();
    q.memset(w, 0, dim * taps * elem).wait();
    q.memset(bias, 0, dim * elem).wait();
    for (std::size_t i = 0; i < batch; ++i) idx[i] = static_cast<int>(i % slots);
    auto once = [&] {
      kernels::causal_conv1d_decode_sycl(q, state, x, w, bias, idx, out, true,
                                         batch, dim, state_len, taps, slots,
                                         cs0, cs1, cs2, dt, DType::f32);
    };
    const DeviceTiming timing = time_device_batches(once);
    std::cout << "{\"schema_version\":2,\"kernel\":\"causal_conv1d_decode\","
              << "\"variant\":\"sycl\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"batch\":" << batch << ",\"dim\":" << dim
              << ",\"iters\":" << iters << ",\"median_ms\":" << timing.median_ms
              << ",\"min_ms\":" << timing.min_ms << ",\"max_ms\":" << timing.max_ms
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}"
              << std::endl;
    sycl::free(state, q); sycl::free(x, q); sycl::free(w, q);
    sycl::free(bias, q); sycl::free(out, q); sycl::free(idx, q);
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

  if (kernel == "w4a16_gemm") {
    // int4-weight x 16-bit-activation DPAS GEMM vs the existing int4/int8 paths.
    // --approx dpas (default): the new w4a16_gemm_sycl (joint_matrix). --approx
    // gemv: the current int4 path applied per-row -- M sequential qgemv_int4
    // launches (summed device time), i.e. how int4-weight GEMM is done today.
    // --approx int8: qgemm_int8_sycl (the existing int8 w8a8 GEMM native tile) at
    // the same M/N/K. Same M,N,K; group=128 (or K). Activation dtype must be
    // 16-bit; f32 is remapped to bf16 for this kernel.
    DType adt = (dt == DType::f32) ? DType::bf16 : dt;
    const std::size_t elem16 = dtype_size(adt);
    const std::size_t group = (K % 128 == 0) ? 128 : K;
    void* w = sycl::malloc_device(N * (K / 2), q);            // int4 packed [N,K/2]
    void* sc = sycl::malloc_device(N * (K / group) * 2, q);   // f16 scales [N,K/group]
    void* A = sycl::malloc_device(M * K * elem16, q);         // [M,K] act
    void* C = sycl::malloc_device(M * N * elem16, q);         // [M,N] out
    q.memset(w, 1, N * (K / 2)).wait();
    q.memset(sc, 0, N * (K / group) * 2).wait();
    q.memset(A, 0, M * K * elem16).wait();
    // int8 comparison operands
    std::int8_t* iA = sycl::malloc_device<std::int8_t>(M * K, q);
    std::int8_t* iB = sycl::malloc_device<std::int8_t>(K * N, q);
    float* ias = sycl::malloc_device<float>(M, q);
    float* ibs = sycl::malloc_device<float>(N, q);
    void* iC = sycl::malloc_device(M * N * elem16, q);
    q.memset(iA, 0, M * K).wait(); q.memset(iB, 0, K * N).wait();
    q.memset(ias, 0, M * 4).wait(); q.memset(ibs, 0, N * 4).wait();

    const std::string mode = approx_s;  // dpas | gemv | int8
    auto once = [&]() -> double {
      if (mode == "gemv") {
        double t = 0.0;
        for (std::size_t m = 0; m < M; ++m) {
          const char* xrow = static_cast<const char*>(A) + m * K * elem16;
          char* yrow = static_cast<char*>(C) + m * N * elem16;
          sycl::event e = kernels::qgemv_int4_sycl(q, w, sc, xrow, yrow, N, K, group, adt);
          e.wait(); t += event_ms(e);
        }
        return t;
      }
      if (mode == "int8") {
        sycl::event e = kernels::qgemm_int8_sycl(q, iA, iB, ias, ibs, iC, M, N, K, adt);
        e.wait(); return event_ms(e);
      }
      sycl::event e = kernels::w4a16_gemm_sycl(q, A, w, sc, C, M, N, K, group, adt);
      e.wait(); return event_ms(e);
    };
    for (int i = 0; i < warmup; ++i) once();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) s.push_back(once());
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    const double gops = 2.0 * (double)M * (double)N * (double)K / (med * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"w4a16_gemm\",\"variant\":\"sycl\",\"approx\":\""
              << (mode == "gemv" ? "gemv" : (mode == "int8" ? "int8" : "dpas"))
              << "\",\"dtype\":\"" << dtype_name(adt) << "\",\"M\":" << M << ",\"N\":" << N
              << ",\"K\":" << K << ",\"group\":" << group << ",\"iters\":" << iters
              << ",\"median_ms\":" << med << ",\"min_ms\":" << s.front() << ",\"gops\":" << gops
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(w, q); sycl::free(sc, q); sycl::free(A, q); sycl::free(C, q);
    sycl::free(iA, q); sycl::free(iB, q); sycl::free(ias, q); sycl::free(ibs, q); sycl::free(iC, q);
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
  if (kernel == "attn_swa") {
    // A/B for the symmetric sliding-window band. --approx banded (default):
    // attn_swa with the requested --window W, so each query streams only the
    // ~W keys in its symmetric band. --approx dense: the SAME kernel with
    // window=0, which streams all `seq` keys -- i.e. the dense flash SDPA
    // baseline (identical math to attention_sycl non-causal). The only
    // difference between the two is the banded key range, so at long seq with
    // W << seq the banded pass does O(W) work per query instead of O(seq).
    // d = 64 (matches the attention harness); nh kv heads == nh (MHA).
    const std::size_t nh = rows, seq = dim, d = 64;
    const std::size_t ne = nh * seq * d;
    void* Q = sycl::malloc_device(ne * elem, q);
    void* K = sycl::malloc_device(ne * elem, q);
    void* V = sycl::malloc_device(ne * elem, q);
    void* O = sycl::malloc_device(ne * elem, q);
    q.memset(Q, 0, ne * elem).wait(); q.memset(K, 0, ne * elem).wait(); q.memset(V, 0, ne * elem).wait();
    const bool dense = (approx_s == "dense");
    const std::size_t win = dense ? 0 : window;
    const double med = time_median([&] { return kernels::attn_swa_sycl(q, Q, K, V, O, nh, nh, seq, seq, d, win, dt); });
    std::cout << "{\"schema_version\":2,\"kernel\":\"attn_swa\",\"variant\":\"sycl\",\"approx\":\""
              << (dense ? "dense" : "banded") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"heads\":" << nh << ",\"seq\":" << seq << ",\"d\":" << d
              << ",\"window\":" << win << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q);
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
  if (kernel == "rms_residual_next") {
    // A/B for the fused residual-add + double RMSNorm -> f16. --approx fused
    // (default): the single rms_residual_next_sycl kernel (sublayer post-norm +
    // residual add + next pre-norm + f16 convert in one launch). --approx
    // unfused: the composed separate-ops baseline --
    // rms_norm(projection,post_weight)->scratch, residual += scratch,
    // rms_norm(residual,next_weight)->scratch2, convert scratch2->f16 -- timed as
    // the sum of the four device events. The delta is ~2 eliminated scratch
    // round-trips and 3 launches. --rows, --dim.
    const std::size_t R = rows, D = dim;
    const std::size_t total = R * D;
    void* proj = sycl::malloc_device(total * elem, q);
    void* pw = sycl::malloc_device(D * elem, q);
    void* res = sycl::malloc_device(total * elem, q);
    void* nw = sycl::malloc_device(D * elem, q);
    half_t* out = sycl::malloc_device<half_t>(total, q);
    void* scratch = sycl::malloc_device(total * elem, q);
    void* scratch2 = sycl::malloc_device(total * elem, q);
    q.memset(proj, 0, total * elem).wait(); q.memset(pw, 0, D * elem).wait();
    q.memset(res, 0, total * elem).wait(); q.memset(nw, 0, D * elem).wait();
    const float eps = 1e-6f;
    auto add_into = [&]() -> sycl::event {
      switch (dt) {
        case DType::f16: { half_t* r = static_cast<half_t*>(res); const half_t* s = static_cast<const half_t*>(scratch);
          return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i){ r[i[0]] = static_cast<half_t>(static_cast<float>(r[i[0]]) + static_cast<float>(s[i[0]])); }); }
        case DType::bf16: { bf16_t* r = static_cast<bf16_t*>(res); const bf16_t* s = static_cast<const bf16_t*>(scratch);
          return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i){ r[i[0]] = static_cast<bf16_t>(static_cast<float>(r[i[0]]) + static_cast<float>(s[i[0]])); }); }
        default: { float* r = static_cast<float*>(res); const float* s = static_cast<const float*>(scratch);
          return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i){ r[i[0]] += s[i[0]]; }); }
      }
    };
    auto convert16 = [&]() -> sycl::event {
      switch (dt) {
        case DType::f16: { const half_t* s = static_cast<const half_t*>(scratch2);
          return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i){ out[i[0]] = s[i[0]]; }); }
        case DType::bf16: { const bf16_t* s = static_cast<const bf16_t*>(scratch2);
          return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i){ out[i[0]] = static_cast<half_t>(static_cast<float>(s[i[0]])); }); }
        default: { const float* s = static_cast<const float*>(scratch2);
          return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i){ out[i[0]] = static_cast<half_t>(s[i[0]]); }); }
      }
    };
    const bool unfused = (approx_s == "unfused");
    auto once = [&]() -> double {
      if (unfused) {
        sycl::event e1 = kernels::rms_norm_sycl(q, proj, pw, scratch, R, D, eps, dt); e1.wait();
        sycl::event e2 = add_into(); e2.wait();
        sycl::event e3 = kernels::rms_norm_sycl(q, res, nw, scratch2, R, D, eps, dt); e3.wait();
        sycl::event e4 = convert16(); e4.wait();
        return event_ms(e1) + event_ms(e2) + event_ms(e3) + event_ms(e4);
      }
      sycl::event ef = kernels::rms_residual_next_sycl(q, proj, pw, res, nw, out, R, D, eps, dt); ef.wait();
      return event_ms(ef);
    };
    for (int i = 0; i < warmup; ++i) once();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) s.push_back(once());
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    std::cout << "{\"schema_version\":2,\"kernel\":\"rms_residual_next\",\"variant\":\"sycl\",\"approx\":\""
              << (unfused ? "unfused" : "fused") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"rows\":" << R << ",\"dim\":" << D
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"min_ms\":" << s.front() << ",\"max_ms\":" << s.back()
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(proj, q); sycl::free(pw, q); sycl::free(res, q); sycl::free(nw, q); sycl::free(out, q); sycl::free(scratch, q); sycl::free(scratch2, q);
    return 0;
  }
  if (kernel == "qk_norm_rope") {
    // A/B for the fused per-head QK-norm + query-scale + RoPE. --approx fused
    // (default): the single qk_norm_rope_sycl kernel. --approx unfused: the
    // composed separate-ops baseline -- rms_norm(Q,qw)->Q, scale Q by
    // query_scale, rope(Q), rms_norm(K,kw)->K, rope(K) -- timed as the sum of
    // the five device events. Same shapes; the delta is 4 eliminated launches
    // (submission-bound). --rows = tokens; n_head/n_head_kv/head_dim fixed below
    // (GQA: 32/8, head_dim 128).
    const std::size_t tok = rows, NH = 32, NKV = 8, HD = 128;
    const std::size_t nq = tok * NH * HD, nk = tok * NKV * HD;
    void* Q = sycl::malloc_device(nq * elem, q);
    void* K = sycl::malloc_device(nk * elem, q);
    void* qw = sycl::malloc_device(HD * elem, q);
    void* kw = sycl::malloc_device(HD * elem, q);
    q.memset(Q, 0, nq * elem).wait(); q.memset(K, 0, nk * elem).wait();
    q.memset(qw, 0, HD * elem).wait(); q.memset(kw, 0, HD * elem).wait();
    const float eps = 1e-6f, base = 10000.0f, qscale = 0.0625f;
    auto scaleQ = [&]() -> sycl::event {
      switch (dt) {
        case DType::f16: { half_t* p = static_cast<half_t*>(Q);
          return q.parallel_for(sycl::range<1>(nq), [=](sycl::id<1> i){ p[i[0]] = static_cast<half_t>(static_cast<float>(p[i[0]]) * qscale); }); }
        case DType::bf16: { bf16_t* p = static_cast<bf16_t*>(Q);
          return q.parallel_for(sycl::range<1>(nq), [=](sycl::id<1> i){ p[i[0]] = static_cast<bf16_t>(static_cast<float>(p[i[0]]) * qscale); }); }
        default: { float* p = static_cast<float*>(Q);
          return q.parallel_for(sycl::range<1>(nq), [=](sycl::id<1> i){ p[i[0]] *= qscale; }); }
      }
    };
    const bool unfused = (approx_s == "unfused");
    auto once = [&]() -> double {
      if (unfused) {
        sycl::event e1 = kernels::rms_norm_sycl(q, Q, qw, Q, tok * NH, HD, eps, dt); e1.wait();
        sycl::event e2 = scaleQ(); e2.wait();
        sycl::event e3 = kernels::rope_sycl(q, Q, Q, tok, NH, HD, base, 0, dt); e3.wait();
        sycl::event e4 = kernels::rms_norm_sycl(q, K, kw, K, tok * NKV, HD, eps, dt); e4.wait();
        sycl::event e5 = kernels::rope_sycl(q, K, K, tok, NKV, HD, base, 0, dt); e5.wait();
        return event_ms(e1) + event_ms(e2) + event_ms(e3) + event_ms(e4) + event_ms(e5);
      }
      sycl::event ef = kernels::qk_norm_rope_sycl(q, Q, K, qw, kw, nullptr, nullptr, tok, NH, NKV, HD, base, 0, qscale, eps, dt); ef.wait();
      return event_ms(ef);
    };
    for (int i = 0; i < warmup; ++i) once();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) s.push_back(once());
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    std::cout << "{\"schema_version\":2,\"kernel\":\"qk_norm_rope\",\"variant\":\"sycl\",\"approx\":\""
              << (unfused ? "unfused" : "fused") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"tokens\":" << tok << ",\"n_head\":" << NH << ",\"n_head_kv\":" << NKV
              << ",\"head_dim\":" << HD << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"min_ms\":" << s.front() << ",\"max_ms\":" << s.back()
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(Q, q); sycl::free(K, q); sycl::free(qw, q); sycl::free(kw, q);
    return 0;
  }
  if (kernel == "glu_gelu_f16") {
    // A/B for the fused GEGLU -> f16. --approx fused (default): the single
    // glu_gelu_f16_sycl kernel (dt in -> f16 out). --approx unfused: the composed
    // baseline -- a tanh-gelu GLU writing dt into scratch, then a dt->f16 convert
    // -- timed as the sum of both device events. Same math; the delta is the
    // eliminated [rows,d] scratch round-trip and a launch. --rows, --dim = d.
    const std::size_t R = rows, D = dim;
    void* x = sycl::malloc_device(R * 2 * D * elem, q);
    void* scratch = sycl::malloc_device(R * D * elem, q);
    half_t* out = sycl::malloc_device<half_t>(R * D, q);
    q.memset(x, 0, R * 2 * D * elem).wait();
    auto convert16 = [&]() -> sycl::event {
      switch (dt) {
        case DType::f16: { const half_t* s = static_cast<const half_t*>(scratch);
          return q.parallel_for(sycl::range<1>(R * D), [=](sycl::id<1> i){ out[i[0]] = s[i[0]]; }); }
        case DType::bf16: { const bf16_t* s = static_cast<const bf16_t*>(scratch);
          return q.parallel_for(sycl::range<1>(R * D), [=](sycl::id<1> i){ out[i[0]] = static_cast<half_t>(static_cast<float>(s[i[0]])); }); }
        default: { const float* s = static_cast<const float*>(scratch);
          return q.parallel_for(sycl::range<1>(R * D), [=](sycl::id<1> i){ out[i[0]] = static_cast<half_t>(s[i[0]]); }); }
      }
    };
    const bool unfused = (approx_s == "unfused");
    auto once = [&]() -> double {
      if (unfused) {
        sycl::event e1;
        switch (dt) {
          case DType::f16: e1 = glu_gelu_tanh_dt<half_t>(q, static_cast<const half_t*>(x), static_cast<half_t*>(scratch), R, D); break;
          case DType::bf16: e1 = glu_gelu_tanh_dt<bf16_t>(q, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(scratch), R, D); break;
          default: e1 = glu_gelu_tanh_dt<float>(q, static_cast<const float*>(x), static_cast<float*>(scratch), R, D); break;
        }
        e1.wait();
        sycl::event e2 = convert16(); e2.wait();
        return event_ms(e1) + event_ms(e2);
      }
      sycl::event ef = kernels::glu_gelu_f16_sycl(q, x, out, R, D, dt); ef.wait();
      return event_ms(ef);
    };
    for (int i = 0; i < warmup; ++i) once();
    std::vector<double> s;
    for (int i = 0; i < iters; ++i) s.push_back(once());
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    std::cout << "{\"schema_version\":2,\"kernel\":\"glu_gelu_f16\",\"variant\":\"sycl\",\"approx\":\""
              << (unfused ? "unfused" : "fused") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"rows\":" << R << ",\"d\":" << D
              << ",\"iters\":" << iters << ",\"median_ms\":" << med
              << ",\"min_ms\":" << s.front() << ",\"max_ms\":" << s.back()
              << ",\"device\":\"" << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(x, q); sycl::free(scratch, q); sycl::free(out, q);
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
    const double med = time_median([&] { return kernels::moe_route_topk_sycl(q, lg, ids, w, nt, ne, kk, 0, 1, 1.0f, dt); });
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
