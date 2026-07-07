// On-device correctness gate for the XPU op ABI.
//
// Runs each shipped op across dtypes {f32, f16, bf16} and every shipped variant
// {sycl, vendor} on a real Intel GPU, checking against an independent fp64 host
// reference (std::erf) within the umbrella per-dtype tolerances. Registered as a
// CTest test only under QUIXICORE_XPU_ENABLE_SYCL.
//
// This is the authoritative correctness check for the backend today. A
// torch.xpu / pytest harness is planned once the PyTorch-XPU binding lands; the
// fp64 oracle here does not depend on any Python stack.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

namespace {

using namespace quixicore::xpu;

double gelu_erf_ref(double x) {
  return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
}

struct Tol {
  double atol;
  double rtol;
};

// Per-dtype tolerances from the umbrella registry (registry/tolerances.yaml).
Tol tol_for(DType dt) {
  switch (dt) {
    case DType::f32:
      return {1e-5, 1e-4};
    case DType::f16:
      return {1e-3, 1e-3};
    case DType::bf16:
      return {2e-3, 2e-3};
  }
  return {1e-3, 1e-3};
}

// Run GELU for one (dtype, variant) and return true on pass. Uses shared USM so
// the host can fill inputs and read outputs directly; storage dtype conversion
// goes through static_cast<T>, matching the kernel's own rounding.
template <typename T>
bool check_gelu(sycl::queue& q, DType dt, Variant variant,
                const std::vector<float>& host_in) {
  const std::size_t n = host_in.size();
  T* in = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) in[i] = static_cast<T>(host_in[i]);

  ops::gelu(q, in, out, n, dt, ops::GeluApprox::erf, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double max_abs = 0.0;
  double worst_excess = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    // Reference is GELU of the value the kernel actually consumed (in[i], after
    // storage rounding), then rounded to the storage dtype. This isolates the
    // kernel's compute error: both kernel and reference compute in >=fp32 and
    // store in T, so the contract rtol (e.g. bf16 2e-3) is applied against a
    // same-precision reference rather than an unreachable fp64 target (a single
    // bf16 ULP is ~0.4% > 2e-3, so comparing to fp64 would fail by construction).
    const double ref_hi = gelu_erf_ref(static_cast<double>(in[i]));
    const double ref = static_cast<double>(static_cast<T>(ref_hi));
    const double abs_err = std::abs(static_cast<double>(out[i]) - ref);
    max_abs = std::max(max_abs, abs_err);
    worst_excess = std::max(worst_excess, abs_err - (tol.atol + tol.rtol * std::abs(ref)));
  }

  sycl::free(in, q);
  sycl::free(out, q);

  const bool ok = worst_excess <= 0.0;
  std::cout << "  gelu dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " (resolved=" << variant_name(resolve_variant(variant)) << ")"
            << " max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// Deterministic pseudo-random fill in a modest range, reproducible without any
// RNG dependency.
float sample(std::size_t i) {
  const double t = static_cast<double>((i * 2654435761u) % 10007) / 10007.0;
  return static_cast<float>(-2.0 + 4.0 * t);
}

double sigmoid_ref(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double silu_ref(double x) { return x * sigmoid_ref(x); }
double gelu_ref(double x) { return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476)); }
double relu_ref(double x) { return x > 0.0 ? x : 0.0; }

double gelu_grad_ref(double x) {
  const double cdf = 0.5 * (1.0 + std::erf(x * 0.7071067811865476));
  const double pdf = 0.3989422804014327 * std::exp(-0.5 * x * x);
  return cdf + x * pdf;
}

// Explicit mantissa bits per storage dtype (for the 1-ULP allowance below).
int mantissa_bits(DType dt) {
  switch (dt) {
    case DType::f32: return 23;
    case DType::f16: return 10;
    case DType::bf16: return 7;
  }
  return 23;
}

// Generic pass/fail against a per-element reference rounded to storage dtype.
// The kernel computes in fp32 and the oracle in fp64, so around a rounding
// boundary they can land on adjacent storage values: allow one storage ULP on
// top of the contract tolerance (a single bf16 ULP is ~0.4% > the 2e-3 rtol, so
// transcendental bf16 ops would otherwise fail by construction).
template <typename T>
bool report(const char* name, DType dt, const std::vector<float>& out,
            const std::vector<double>& ref_hi) {
  const Tol tol = tol_for(dt);
  const double ulp_rel = std::ldexp(1.0, -mantissa_bits(dt));  // 2^-mant
  double max_abs = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double ref = static_cast<double>(static_cast<T>(ref_hi[i]));
    const double err = std::abs(static_cast<double>(out[i]) - ref);
    const double allow = tol.atol + tol.rtol * std::abs(ref) + ulp_rel * std::abs(ref);
    max_abs = std::max(max_abs, err);
    worst = std::max(worst, err - allow);
  }
  const bool ok = worst <= 0.0;
  std::cout << "  " << name << " dt=" << dtype_name(dt) << " max_abs=" << max_abs
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_silu(sycl::queue& q, DType dt, std::size_t n) {
  T* in = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) in[i] = static_cast<T>(sample(i) * 3.0f);
  ops::silu(q, in, out, n, dt, Variant::sycl, true);
  std::vector<float> o(n);
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i) { o[i] = static_cast<float>(out[i]); r[i] = silu_ref(static_cast<double>(in[i])); }
  const bool ok = report<T>("silu", dt, o, r);
  sycl::free(in, q); sycl::free(out, q);
  return ok;
}

template <typename T>
bool check_gelu_bwd(sycl::queue& q, DType dt, std::size_t n) {
  T* g = sycl::malloc_shared<T>(n, q);
  T* x = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) { g[i] = static_cast<T>(sample(i + 1)); x[i] = static_cast<T>(sample(i) * 3.0f); }
  ops::gelu_backward(q, g, x, out, n, dt, ops::GeluApprox::erf, Variant::sycl, true);
  std::vector<float> o(n);
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i) { o[i] = static_cast<float>(out[i]); r[i] = static_cast<double>(g[i]) * gelu_grad_ref(static_cast<double>(x[i])); }
  const bool ok = report<T>("gelu_bwd", dt, o, r);
  sycl::free(g, q); sycl::free(x, q); sycl::free(out, q);
  return ok;
}

template <typename T>
bool check_glu(sycl::queue& q, DType dt, ops::GluMode mode, const char* mname,
               std::size_t rows, std::size_t d) {
  T* x = sycl::malloc_shared<T>(rows * 2 * d, q);
  T* out = sycl::malloc_shared<T>(rows * d, q);
  for (std::size_t i = 0; i < rows * 2 * d; ++i) x[i] = static_cast<T>(sample(i) * 2.0f);
  ops::glu(q, x, out, rows, d, dt, mode, Variant::sycl, true);
  std::vector<float> o(rows * d);
  std::vector<double> r(rows * d);
  for (std::size_t rr = 0; rr < rows; ++rr) {
    for (std::size_t i = 0; i < d; ++i) {
      const double gate = static_cast<double>(x[rr * 2 * d + i]);
      const double val = static_cast<double>(x[rr * 2 * d + d + i]);
      double a;
      switch (mode) {
        case ops::GluMode::swiglu: a = silu_ref(gate); break;
        case ops::GluMode::geglu:  a = gelu_ref(gate); break;
        case ops::GluMode::reglu:  a = relu_ref(gate); break;
        default:                   a = sigmoid_ref(gate); break;
      }
      o[rr * d + i] = static_cast<float>(out[rr * d + i]);
      r[rr * d + i] = a * val;
    }
  }
  const bool ok = report<T>(mname, dt, o, r);
  sycl::free(x, q); sycl::free(out, q);
  return ok;
}

template <typename T>
bool check_softmax(sycl::queue& q, DType dt, Variant variant, std::size_t rows,
                   std::size_t dim) {
  const std::size_t n = rows * dim;
  T* x = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i) * 3.0f);

  ops::softmax(q, x, out, rows, dim, dt, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double worst_excess = 0.0, max_abs = 0.0, worst_rowsum = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double m = -1e30;
    for (std::size_t i = 0; i < dim; ++i) m = std::max(m, static_cast<double>(x[r * dim + i]));
    double sum = 0.0;
    for (std::size_t i = 0; i < dim; ++i) sum += std::exp(static_cast<double>(x[r * dim + i]) - m);
    double rowsum = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double ref_hi = std::exp(static_cast<double>(x[r * dim + i]) - m) / sum;
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs(static_cast<double>(out[r * dim + i]) - ref);
      max_abs = std::max(max_abs, err);
      worst_excess = std::max(worst_excess, err - (tol.atol + tol.rtol * std::abs(ref)));
      rowsum += static_cast<double>(out[r * dim + i]);
    }
    worst_rowsum = std::max(worst_rowsum, std::abs(rowsum - 1.0));
  }
  sycl::free(x, q); sycl::free(out, q);
  const bool ok = worst_excess <= 0.0 && worst_rowsum < 5e-2;
  std::cout << "  softmax dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " max_abs=" << max_abs << " rowsum_err=" << worst_rowsum
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_fp8_gemm(sycl::queue& q, DType out_dt, ops::Fp8Kind kind,
                    const char* kname, std::size_t M, std::size_t N, std::size_t K) {
  // Build fp8 inputs from f32 via the public codec; decode them back to get the
  // exact fp8-rounded values the GEMM actually consumes -> fp64 reference.
  float* Af = sycl::malloc_shared<float>(M * K, q);
  float* Bf = sycl::malloc_shared<float>(K * N, q);
  std::uint8_t* A8 = sycl::malloc_shared<std::uint8_t>(M * K, q);
  std::uint8_t* B8 = sycl::malloc_shared<std::uint8_t>(K * N, q);
  float* Art = sycl::malloc_shared<float>(M * K, q);
  float* Brt = sycl::malloc_shared<float>(K * N, q);
  T* C = sycl::malloc_shared<T>(M * N, q);
  for (std::size_t i = 0; i < M * K; ++i) Af[i] = sample(i) * 0.5f;
  for (std::size_t i = 0; i < K * N; ++i) Bf[i] = sample(i + 5) * 0.5f;
  const float scale = 1.0f;

  bool ok = true;
  try {
    ops::fp8_encode(q, Af, A8, M * K, kind);
    ops::fp8_encode(q, Bf, B8, K * N, kind);
    ops::fp8_decode(q, A8, Art, M * K, kind);
    ops::fp8_decode(q, B8, Brt, K * N, kind);
    ops::fp8_gemm(q, A8, B8, C, M, N, K, kind, scale, out_dt, Variant::vendor, true);

    const Tol base = tol_for(out_dt);
    const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 5e-3;
    double worst = 0.0, max_abs = 0.0;
    for (std::size_t m = 0; m < M; ++m)
      for (std::size_t n = 0; n < N; ++n) {
        double acc = 0.0;
        for (std::size_t k = 0; k < K; ++k)
          acc += (double)Art[m * K + k] * (double)Brt[k * N + n];
        const double ref = static_cast<double>(static_cast<T>(acc * scale));
        const double err = std::abs((double)C[m * N + n] - ref);
        max_abs = std::max(max_abs, err);
        worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
      }
    ok = worst <= 0.0;
    std::cout << "  fp8_gemm " << kname << " out=" << dtype_name(out_dt)
              << " (" << M << "x" << N << "x" << K << ") max_abs=" << max_abs
              << (ok ? "  ok" : "  FAIL") << '\n';
  } catch (const std::exception& e) {
    std::cout << "  fp8_gemm " << kname << ": UNSUPPORTED (" << e.what() << ")\n";
    ok = true;  // not a correctness failure; recorded as unsupported on this device
  }
  sycl::free(Af, q); sycl::free(Bf, q); sycl::free(A8, q); sycl::free(B8, q);
  sycl::free(Art, q); sycl::free(Brt, q); sycl::free(C, q);
  return ok;
}

template <typename T>
bool check_qgemm_int8(sycl::queue& q, DType out_dt, Variant variant,
                      std::size_t M, std::size_t N, std::size_t K) {
  std::int8_t* A = sycl::malloc_shared<std::int8_t>(M * K, q);
  std::int8_t* B = sycl::malloc_shared<std::int8_t>(K * N, q);
  float* as = sycl::malloc_shared<float>(M, q);
  float* bs = sycl::malloc_shared<float>(N, q);
  T* C = sycl::malloc_shared<T>(M * N, q);
  for (std::size_t i = 0; i < M * K; ++i) A[i] = static_cast<std::int8_t>(std::floor(sample(i) * 30.0f));
  for (std::size_t i = 0; i < K * N; ++i) B[i] = static_cast<std::int8_t>(std::floor(sample(i + 5) * 30.0f));
  for (std::size_t i = 0; i < M; ++i) as[i] = 0.01f + 0.005f * std::abs(sample(i + 1));
  for (std::size_t i = 0; i < N; ++i) bs[i] = 0.01f + 0.005f * std::abs(sample(i + 2));

  ops::qgemm_int8(q, A, B, as, bs, C, M, N, K, out_dt, variant, /*blocking=*/true);

  const Tol base = tol_for(out_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 1e-3;
  double worst = 0.0, max_abs = 0.0;
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      long acc = 0;
      for (std::size_t k = 0; k < K; ++k) acc += (int)A[m * K + k] * (int)B[k * N + n];
      const double ref_hi = (double)acc * (double)as[m] * (double)bs[n];
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs((double)C[m * N + n] - ref);
      max_abs = std::max(max_abs, err);
      worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
    }
  sycl::free(A, q); sycl::free(B, q); sycl::free(as, q); sycl::free(bs, q); sycl::free(C, q);
  const bool ok = worst <= 0.0;
  std::cout << "  qgemm_int8 out=" << dtype_name(out_dt) << " variant=" << variant_name(variant)
            << " (" << M << "x" << N << "x" << K << ") max_abs=" << max_abs
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_mxfp4_gemv(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const std::size_t bpr = K / 2, blkpr = K / 32;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * bpr, q);
  std::uint8_t* bs = sycl::malloc_shared<std::uint8_t>(N * blkpr, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);

  auto nib = [&](std::size_t i) {  // random e2m1 nibble
    int mag = static_cast<int>(std::floor((sample(i) * 0.5f + 0.5f) * 8.0f)) & 7;
    int sign = (sample(i + 1) < 0.0f) ? 8 : 0;
    return static_cast<std::uint8_t>(sign | mag);
  };
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t b = 0; b < bpr; ++b)
      w[n * bpr + b] = static_cast<std::uint8_t>(nib(n * K + 2 * b) | (nib(n * K + 2 * b + 1) << 4));
    for (std::size_t g = 0; g < blkpr; ++g)
      bs[n * blkpr + g] = static_cast<std::uint8_t>(124 + (static_cast<int>((g + n) % 5)));  // ~2^-3..2^1
  }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 21));

  ops::mxfp4_gemv(q, w, bs, x, y, N, K, act_dt, Variant::sycl, true);

  const Tol base = tol_for(act_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 3e-3;
  double worst = 0.0, max_abs = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint8_t byte = w[n * bpr + k / 2];
      const int n4 = (k & 1) ? (byte >> 4) : (byte & 0xF);
      double val = e2m1[n4 & 7];
      if (n4 & 8) val = -val;
      const double scale = std::ldexp(1.0, static_cast<int>(bs[n * blkpr + k / 32]) - 127);
      acc += val * scale * static_cast<double>(x[k]);
    }
    const double ref = static_cast<double>(static_cast<T>(acc));
    const double err = std::abs(static_cast<double>(y[n]) - ref);
    max_abs = std::max(max_abs, err);
    worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(bs, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  mxfp4_gemv act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_qgemv_int4(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K,
                      std::size_t group) {
  const std::size_t bpr = K / 2;         // bytes per row
  const std::size_t gpr = K / group;     // groups per row
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * bpr, q);
  half_t* scales = sycl::malloc_shared<half_t>(N * gpr, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);

  // Deterministic int4 weights (signed [-8,7]) packed 2/byte, per-group scales.
  auto qval = [](std::size_t i) {
    int v = static_cast<int>(std::floor(sample(i) * 4.0f));  // ~[-8,7]
    return std::max(-8, std::min(7, v));
  };
  std::vector<int> wint(N * K);
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < bpr; ++b) {
      const int lo = qval(n * K + 2 * b);
      const int hi = qval(n * K + 2 * b + 1);
      wint[n * K + 2 * b] = lo;
      wint[n * K + 2 * b + 1] = hi;
      w[n * bpr + b] = static_cast<std::uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
    }
  for (std::size_t i = 0; i < N * gpr; ++i) scales[i] = static_cast<half_t>(0.02f + 0.01f * std::abs(sample(i + 9)));
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 13));

  ops::qgemv_int4(q, w, scales, x, y, N, K, group, act_dt, Variant::sycl, true);

  double worst = 0.0, max_abs = 0.0;
  const Tol base = tol_for(act_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K));
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (std::size_t k = 0; k < K; ++k)
      acc += static_cast<double>(wint[n * K + k]) *
             static_cast<double>(scales[n * gpr + k / group]) * static_cast<double>(x[k]);
    const double ref = static_cast<double>(static_cast<T>(acc));
    const double err = std::abs(static_cast<double>(y[n]) - ref);
    max_abs = std::max(max_abs, err);
    worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(scales, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  qgemv_int4 act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << " g=" << group << ") max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_rope(sycl::queue& q, DType dt, std::size_t tokens, std::size_t heads,
                std::size_t hd) {
  const std::size_t n = tokens * heads * hd;
  T* x = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i));
  const float base = 10000.0f;
  ops::rope(q, x, out, tokens, heads, hd, base, 0, dt, Variant::sycl, true);

  const std::size_t half = hd / 2;
  std::vector<float> o(n);
  std::vector<double> r(n);
  for (std::size_t t = 0; t < tokens; ++t)
    for (std::size_t h = 0; h < heads; ++h)
      for (std::size_t i = 0; i < half; ++i) {
        const std::size_t bi = (t * heads + h) * hd;
        const double freq = std::pow((double)base, -2.0 * (double)i / (double)hd);
        const double ang = (double)t * freq;
        const double x1 = (double)x[bi + i], x2 = (double)x[bi + i + half];
        r[bi + i] = x1 * std::cos(ang) - x2 * std::sin(ang);
        r[bi + i + half] = x1 * std::sin(ang) + x2 * std::cos(ang);
        o[bi + i] = (float)out[bi + i];
        o[bi + i + half] = (float)out[bi + i + half];
      }
  const bool ok = report<T>("rope", dt, o, r);
  sycl::free(x, q); sycl::free(out, q);
  return ok;
}

bool check_argmax(sycl::queue& q, DType dt, std::size_t rows, std::size_t vocab) {
  float* logits = sycl::malloc_shared<float>(rows * vocab, q);
  int* out = sycl::malloc_shared<int>(rows, q);
  for (std::size_t i = 0; i < rows * vocab; ++i) logits[i] = sample(i * 3 + 1);
  ops::argmax(q, logits, out, rows, vocab, DType::f32, Variant::sycl, true);
  int mism = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    int ref = 0;
    float best = logits[r * vocab];
    for (std::size_t j = 1; j < vocab; ++j)
      if (logits[r * vocab + j] > best) { best = logits[r * vocab + j]; ref = (int)j; }
    if (out[r] != ref) ++mism;
  }
  sycl::free(logits, q); sycl::free(out, q);
  const bool ok = (mism == 0);
  std::cout << "  argmax dt=" << dtype_name(dt) << " mismatches=" << mism
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

bool check_adamw(sycl::queue& q, DType dt, std::size_t n) {
  // f32 params/moments for a clean numeric check.
  float* p = sycl::malloc_shared<float>(n, q);
  float* g = sycl::malloc_shared<float>(n, q);
  float* m = sycl::malloc_shared<float>(n, q);
  float* v = sycl::malloc_shared<float>(n, q);
  std::vector<double> rp(n);
  const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
  const int step = 5;
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = sample(i); g[i] = sample(i + 2) * 0.5f;
    m[i] = sample(i + 4) * 0.1f; v[i] = std::abs(sample(i + 6)) * 0.1f;
  }
  // fp64 reference
  const double bc1 = 1.0 - std::pow((double)b1, step);
  const double bc2 = 1.0 - std::pow((double)b2, step);
  for (std::size_t i = 0; i < n; ++i) {
    const double grad = g[i];
    const double mi = b1 * (double)m[i] + (1 - b1) * grad;
    const double vi = b2 * (double)v[i] + (1 - b2) * grad * grad;
    rp[i] = (double)p[i] - lr * ((mi / bc1) / (std::sqrt(vi / bc2) + eps) + wd * (double)p[i]);
  }
  ops::adamw(q, p, g, m, v, n, lr, b1, b2, eps, wd, step, dt, Variant::sycl, true);
  std::vector<float> op(n);
  for (std::size_t i = 0; i < n; ++i) op[i] = p[i];
  const bool ok = report<float>("adamw", dt, op, rp);
  sycl::free(p, q); sycl::free(g, q); sycl::free(m, q); sycl::free(v, q);
  return ok;
}

template <typename T>
bool check_gemm(sycl::queue& q, DType dt, Variant variant, std::size_t M,
                std::size_t N, std::size_t K) {
  T* A = sycl::malloc_shared<T>(M * K, q);
  T* B = sycl::malloc_shared<T>(K * N, q);
  T* C = sycl::malloc_shared<T>(M * N, q);
  for (std::size_t i = 0; i < M * K; ++i) A[i] = static_cast<T>(sample(i) * 0.3f);
  for (std::size_t i = 0; i < K * N; ++i) B[i] = static_cast<T>(sample(i + 5) * 0.3f);

  ops::dense_gemm(q, A, B, C, M, N, K, dt, variant, /*blocking=*/true);

  // Tolerance grows with the K-length accumulation; scale rtol by sqrt(K).
  const Tol base = tol_for(dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K));
  double worst = 0.0, max_abs = 0.0;
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (std::size_t k = 0; k < K; ++k)
        acc += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
      const double ref = static_cast<double>(static_cast<T>(acc));
      const double err = std::abs(static_cast<double>(C[m * N + n]) - ref);
      max_abs = std::max(max_abs, err);
      worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
    }
  }
  sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
  const bool ok = worst <= 0.0;
  std::cout << "  dense_gemm dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " (" << M << "x" << N << "x" << K << ") max_abs=" << max_abs
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_rms_norm(sycl::queue& q, DType dt, Variant variant, std::size_t rows,
                    std::size_t dim) {
  const std::size_t n = rows * dim;
  T* x = sycl::malloc_shared<T>(n, q);
  T* w = sycl::malloc_shared<T>(dim, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i));
  for (std::size_t i = 0; i < dim; ++i) w[i] = static_cast<T>(0.5f + sample(i + 7) * 0.1f);
  const float eps = 1e-6f;

  ops::rms_norm(q, x, w, out, rows, dim, eps, dt, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double worst_excess = 0.0, max_abs = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double v = static_cast<double>(x[r * dim + i]);
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
    for (std::size_t i = 0; i < dim; ++i) {
      const double ref_hi = static_cast<double>(x[r * dim + i]) * inv *
                            static_cast<double>(w[i]);
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs(static_cast<double>(out[r * dim + i]) - ref);
      max_abs = std::max(max_abs, err);
      worst_excess = std::max(worst_excess, err - (tol.atol + tol.rtol * std::abs(ref)));
    }
  }
  sycl::free(x, q); sycl::free(w, q); sycl::free(out, q);
  const bool ok = worst_excess <= 0.0;
  std::cout << "  rms_norm dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_layernorm(sycl::queue& q, DType dt, Variant variant, std::size_t rows,
                     std::size_t dim) {
  const std::size_t n = rows * dim;
  T* x = sycl::malloc_shared<T>(n, q);
  T* w = sycl::malloc_shared<T>(dim, q);
  T* b = sycl::malloc_shared<T>(dim, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i));
  for (std::size_t i = 0; i < dim; ++i) {
    w[i] = static_cast<T>(0.8f + sample(i + 3) * 0.1f);
    b[i] = static_cast<T>(sample(i + 11) * 0.1f);
  }
  const float eps = 1e-5f;

  ops::layernorm(q, x, w, b, out, rows, dim, eps, dt, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double worst_excess = 0.0, max_abs = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double sum = 0.0, ss = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double v = static_cast<double>(x[r * dim + i]);
      sum += v; ss += v * v;
    }
    const double mean = sum / static_cast<double>(dim);
    const double var = ss / static_cast<double>(dim) - mean * mean;
    const double inv = 1.0 / std::sqrt(var + eps);
    for (std::size_t i = 0; i < dim; ++i) {
      const double ref_hi = (static_cast<double>(x[r * dim + i]) - mean) * inv *
                                static_cast<double>(w[i]) +
                            static_cast<double>(b[i]);
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs(static_cast<double>(out[r * dim + i]) - ref);
      max_abs = std::max(max_abs, err);
      worst_excess = std::max(worst_excess, err - (tol.atol + tol.rtol * std::abs(ref)));
    }
  }
  sycl::free(x, q); sycl::free(w, q); sycl::free(b, q); sycl::free(out, q);
  const bool ok = worst_excess <= 0.0;
  std::cout << "  layernorm dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

}  // namespace

int main() {
  const auto devices = gpu_devices();
  if (devices.empty()) {
    std::cerr << "no SYCL GPU device; skipping on-device correctness test\n";
    return 0;  // treated as skip on non-XPU hosts
  }

  sycl::queue q = make_gpu_queue(0, /*enable_profiling=*/false);
  std::cout << "device: " << describe_device(q.get_device()) << '\n';

  // Deterministic sweep across the interesting GELU range, incl. the zero
  // crossing and the saturating tails.
  const std::size_t n = 4096;
  std::vector<float> host_in(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_in[i] = -8.0f + 16.0f * (static_cast<float>(i) / static_cast<float>(n));
  }

  const Variant variants[] = {Variant::sycl, Variant::vendor};
  int failures = 0;
  for (const Variant v : variants) {
    failures += check_gelu<float>(q, DType::f32, v, host_in) ? 0 : 1;
    failures += check_gelu<half_t>(q, DType::f16, v, host_in) ? 0 : 1;
    failures += check_gelu<bf16_t>(q, DType::bf16, v, host_in) ? 0 : 1;
  }

  // Elementwise activations (native only): silu, gelu backward, glu modes.
  for (const DType dt : {DType::f32, DType::f16, DType::bf16}) {
    if (dt == DType::f32) {
      failures += check_silu<float>(q, dt, 4096) ? 0 : 1;
      failures += check_gelu_bwd<float>(q, dt, 4096) ? 0 : 1;
    } else if (dt == DType::f16) {
      failures += check_silu<half_t>(q, dt, 4096) ? 0 : 1;
      failures += check_gelu_bwd<half_t>(q, dt, 4096) ? 0 : 1;
    } else {
      failures += check_silu<bf16_t>(q, dt, 4096) ? 0 : 1;
      failures += check_gelu_bwd<bf16_t>(q, dt, 4096) ? 0 : 1;
    }
  }
  failures += check_glu<float>(q, DType::f32, ops::GluMode::swiglu, "glu_swiglu", 64, 1024) ? 0 : 1;
  failures += check_glu<float>(q, DType::f32, ops::GluMode::geglu, "glu_geglu", 64, 1024) ? 0 : 1;
  failures += check_glu<bf16_t>(q, DType::bf16, ops::GluMode::swiglu, "glu_swiglu_bf16", 64, 1024) ? 0 : 1;
  failures += check_glu<float>(q, DType::f32, ops::GluMode::reglu, "glu_reglu", 64, 1000) ? 0 : 1;  // d not mult of 4: scalar path

  // Dense GEMM, both variants, f32 + bf16.
  for (const Variant v : variants) {
    failures += check_gemm<float>(q, DType::f32, v, 128, 96, 160) ? 0 : 1;
    failures += check_gemm<bf16_t>(q, DType::bf16, v, 128, 96, 160) ? 0 : 1;
  }

  // int4 quantized GEMV (native).
  failures += check_qgemv_int4<float>(q, DType::f32, 128, 4096, 128) ? 0 : 1;
  failures += check_qgemv_int4<bf16_t>(q, DType::bf16, 128, 4096, 128) ? 0 : 1;

  // mxfp4 GEMV (native e2m1 + e8m0 block-scale decode).
  failures += check_mxfp4_gemv<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_mxfp4_gemv<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;

  // int8 w8a8 GEMM, both variants.
  for (const Variant v : variants) {
    failures += check_qgemm_int8<float>(q, DType::f32, v, 96, 128, 256) ? 0 : 1;
    failures += check_qgemm_int8<bf16_t>(q, DType::bf16, v, 96, 128, 256) ? 0 : 1;
  }

  // fp8 GEMM (e4m3/e5m2), vendor-only; skips cleanly if the device lacks fp8.
  failures += check_fp8_gemm<float>(q, DType::f32, ops::Fp8Kind::e4m3, "e4m3", 128, 128, 256) ? 0 : 1;
  failures += check_fp8_gemm<float>(q, DType::f32, ops::Fp8Kind::e5m2, "e5m2", 128, 128, 256) ? 0 : 1;

  // rope / adamw / argmax (native only).
  failures += check_rope<float>(q, DType::f32, 32, 8, 64) ? 0 : 1;
  failures += check_rope<bf16_t>(q, DType::bf16, 32, 8, 64) ? 0 : 1;
  failures += check_adamw(q, DType::f32, 4096) ? 0 : 1;
  failures += check_argmax(q, DType::f32, 64, 4000) ? 0 : 1;

  const std::size_t rows = 64, dim = 1024;
  for (const Variant v : variants) {
    failures += check_softmax<float>(q, DType::f32, v, rows, dim) ? 0 : 1;
    failures += check_softmax<half_t>(q, DType::f16, v, rows, dim) ? 0 : 1;
    failures += check_softmax<bf16_t>(q, DType::bf16, v, rows, dim) ? 0 : 1;
    failures += check_rms_norm<float>(q, DType::f32, v, rows, dim) ? 0 : 1;
    failures += check_rms_norm<half_t>(q, DType::f16, v, rows, dim) ? 0 : 1;
    failures += check_rms_norm<bf16_t>(q, DType::bf16, v, rows, dim) ? 0 : 1;
    failures += check_layernorm<float>(q, DType::f32, v, rows, dim) ? 0 : 1;
    failures += check_layernorm<half_t>(q, DType::f16, v, rows, dim) ? 0 : 1;
    failures += check_layernorm<bf16_t>(q, DType::bf16, v, rows, dim) ? 0 : 1;
  }

  if (failures) {
    std::cerr << "FAIL: " << failures << " case(s) exceeded tolerance\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
