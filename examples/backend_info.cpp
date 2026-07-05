#include <iostream>

#include "quixicore/xpu/backend.hpp"

int main() {
  const auto& info = quixicore::xpu::runtime_backend_info();

  std::cout << info.name << '\n';
  std::cout << "backend: " << info.backend << '\n';
  std::cout << "repo: " << info.repo << '\n';
  std::cout << "umbrella: " << info.umbrella << '\n';
  std::cout << "contract: " << info.contract << '\n';
  std::cout << "status: " << info.status << '\n';

  std::cout << "targets:";
  for (const auto target : info.targets) {
    std::cout << ' ' << target;
  }
  std::cout << '\n';

  std::cout << "integrations:";
  for (const auto integration : info.integrations) {
    std::cout << ' ' << integration;
  }
  std::cout << '\n';

  return 0;
}
