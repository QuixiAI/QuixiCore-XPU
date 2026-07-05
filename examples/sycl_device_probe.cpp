#include <iostream>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/backend.hpp"

int main() {
  const auto& info = quixicore::xpu::runtime_backend_info();
  std::cout << info.name << " SYCL device probe\n";

  const auto platforms = sycl::platform::get_platforms();
  if (platforms.empty()) {
    std::cout << "No SYCL platforms found.\n";
    return 0;
  }

  for (const auto& platform : platforms) {
    std::cout << "platform: "
              << platform.get_info<sycl::info::platform::name>() << '\n';
    for (const auto& device : platform.get_devices()) {
      std::cout << "  device: "
                << device.get_info<sycl::info::device::name>()
                << " | vendor: "
                << device.get_info<sycl::info::device::vendor>()
                << " | type: ";

      if (device.is_gpu()) {
        std::cout << "gpu";
      } else if (device.is_cpu()) {
        std::cout << "cpu";
      } else if (device.is_accelerator()) {
        std::cout << "accelerator";
      } else {
        std::cout << "other";
      }

      std::cout << '\n';
    }
  }

  return 0;
}
