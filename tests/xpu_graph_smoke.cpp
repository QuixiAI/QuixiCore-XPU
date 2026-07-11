#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/graph.hpp"
#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

int main() {
  using namespace quixicore::xpu;

  const auto devices = gpu_devices();
  if (devices.empty()) {
    std::cout << "no SYCL GPU device; skipping command-graph test\n";
    return 0;
  }
  sycl::queue queue = make_gpu_queue();
  if (!command_graph_supported(queue.get_device())) {
    std::cout << "device lacks oneAPI command graphs; skipping\n";
    return 0;
  }

  constexpr std::size_t count = 4096;
  float *input = sycl::malloc_shared<float>(count, queue);
  float *output = sycl::malloc_shared<float>(count, queue);
  for (std::size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(i % 97) * 0.01f - 0.4f;
    output[i] = 0.0f;
  }

  CommandGraph graph(queue);
  graph.capture_begin();
  ops::silu(graph.queue(), input, output, count, DType::f32, Variant::sycl, false);
  graph.capture_end();
  if (!graph.ready() || graph.recording())
    return 1;

  graph.replay().wait();
  double max_error = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = input[i];
    const double reference = value / (1.0 + std::exp(-value));
    max_error = std::max(max_error, std::abs(static_cast<double>(output[i]) - reference));
  }

  for (std::size_t i = 0; i < count; ++i)
    input[i] *= 0.5f;
  graph.replay().wait();
  for (std::size_t i = 0; i < count; ++i) {
    const double value = input[i];
    const double reference = value / (1.0 + std::exp(-value));
    max_error = std::max(max_error, std::abs(static_cast<double>(output[i]) - reference));
  }

  graph.reset();
  sycl::free(input, queue);
  sycl::free(output, queue);
  std::cout << "command_graph replay max_abs=" << max_error << '\n';
  return max_error <= 1e-6 ? 0 : 1;
}
