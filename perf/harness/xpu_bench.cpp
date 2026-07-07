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
//   * time `--iters` launches, each measured by its own profiling event,
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
#include "matmul/dense_gemm/dense_gemm_kernel.hpp"
#include "norms/norms_kernel.hpp"

namespace {

using quixicore::xpu::DType;
using quixicore::xpu::Variant;

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
