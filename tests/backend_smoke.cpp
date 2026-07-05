#include <cassert>
#include <iostream>

#include "quixicore/xpu/backend.hpp"

int main() {
  const auto& info = quixicore::xpu::runtime_backend_info();

  assert(info.backend == "xpu");
  assert(info.name == "QuixiCore XPU");
  assert(info.repo == "QuixiAI/QuixiCore-XPU");
  assert(info.umbrella == "QuixiAI/QuixiCore");
  assert(info.contract == "v0.1");
  assert(info.status == "planned");

  assert(quixicore::xpu::is_contract_kernel_family("norms"));
  assert(quixicore::xpu::is_contract_kernel_family("quant_gemm"));
  assert(!quixicore::xpu::is_contract_kernel_family("not_a_kernel"));

  assert(quixicore::xpu::status_for_kernel_family("norms") == "planned");
  assert(quixicore::xpu::status_for_kernel_family("not_a_kernel") == "unknown");

  std::cout << "QuixiCore XPU backend smoke test passed\n";
  return 0;
}
