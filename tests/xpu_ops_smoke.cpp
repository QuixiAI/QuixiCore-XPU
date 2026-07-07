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

  if (failures) {
    std::cerr << "FAIL: " << failures << " case(s) exceeded tolerance\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
