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
#include "attention/rope/rope_kernel.hpp"
#include "matmul/dense_gemm/dense_gemm_kernel.hpp"
#include "norms/norms_kernel.hpp"
#include "optimizers/adamw/adamw_kernel.hpp"
#include "quantization/fp8_gemm/fp8_kernel.hpp"
#include "quantization/gguf_gemv/gguf_kernel.hpp"
#include "quantization/mxfp4_gemv/mxfp4_kernel.hpp"
#include "quantization/nvfp4_gemv/nvfp4_kernel.hpp"
#include "quantization/qgemm/qgemm_kernel.hpp"
#include "quantization/qgemv/qgemv_kernel.hpp"
#include "sampling/argmax/argmax_kernel.hpp"

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

#if defined(QUIXICORE_XPU_HAS_ONEDNN)
  if (kernel == "fp8_gemm") {
    void* A = sycl::malloc_device(M * K, q);   // 1 byte/elem
    void* B = sycl::malloc_device(K * N, q);
    void* C = sycl::malloc_device(M * N * elem, q);
    q.memset(A, 0, M * K).wait(); q.memset(B, 0, K * N).wait();
    const int fk = (approx_s == "e5m2") ? 1 : 0;  // reuse --approx to pick fp8 kind
    for (int i = 0; i < warmup; ++i) kernels::fp8_gemm_onednn(q, A, B, C, M, N, K, fk, 1.0f, dt);
    // fp8_gemm_onednn is synchronous (waits internally), so time a batch between
    // two profiling markers and divide -> ms/call.
    const int batch = iters;
    sycl::event a = q.single_task([] {});
    a.wait();
    const auto b0 = a.get_profiling_info<sycl::info::event_profiling::command_end>();
    for (int i = 0; i < batch; ++i) kernels::fp8_gemm_onednn(q, A, B, C, M, N, K, fk, 1.0f, dt);
    sycl::event z = q.single_task([] {});
    z.wait();
    const auto b1 = z.get_profiling_info<sycl::info::event_profiling::command_start>();
    const double median = (static_cast<double>(b1 - b0) * 1e-6) / batch;  // ms/call
    const double gflops = 2.0 * (double)M * (double)N * (double)K / (median * 1e-3) / 1e9;
    std::cout << "{\"schema_version\":2,\"kernel\":\"fp8_gemm\",\"variant\":\"vendor\",\"fp8\":\""
              << (fk ? "e5m2" : "e4m3") << "\",\"dtype\":\"" << dtype_name(dt)
              << "\",\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
              << ",\"iters\":" << batch << ",\"median_ms\":" << median
              << ",\"gflops\":" << gflops << ",\"device\":\""
              << q.get_device().get_info<sycl::info::device::name>() << "\"}" << std::endl;
    sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
    return 0;
  }
#endif
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
    const int gt = (approx_s == "q4_0") ? 1 : 0;  // reuse --approx to pick type
    const int bb = gt == 0 ? 34 : 18;
    const std::size_t row_bytes = (Kk / 32) * bb;
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
              << (gt ? "q4_0" : "q8_0") << "\",\"dtype\":\"" << dtype_name(dt)
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
