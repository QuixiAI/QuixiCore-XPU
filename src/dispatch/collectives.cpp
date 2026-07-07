// Collectives family: native multi-GPU sum all-reduce across the visible Intel
// GPUs. Uses a single SYCL context shared over all devices so device USM
// pointers are peer-copyable; reduces onto GPU 0 with cross-device copies, then
// broadcasts. This exercises the 4x B60 multi-GPU path natively; a production
// oneCCL vendor variant (ring/tree all-reduce) is deferred to the collectives
// depth wave. Capability-gated: returns 0 GPUs when none are present.

#include <vector>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::ops {

std::size_t all_reduce_sum(const float* in_per_gpu, float* out,
                           std::size_t count) {
  // The B60s are exposed on multiple backends (Level Zero + OpenCL); a single
  // context cannot span platforms, so pick the single platform exposing the
  // most GPUs (the Level Zero platform with all 4 B60s).
  std::vector<sycl::device> devices;
  for (const auto& p : sycl::platform::get_platforms()) {
    std::vector<sycl::device> gpus;
    for (auto& d : p.get_devices())
      if (d.is_gpu()) gpus.push_back(d);
    if (gpus.size() > devices.size()) devices = std::move(gpus);
  }
  const std::size_t ng = devices.size();
  if (ng == 0) return 0;

  sycl::context ctx(devices);
  std::vector<sycl::queue> qs;
  std::vector<float*> buf(ng);
  qs.reserve(ng);
  for (std::size_t g = 0; g < ng; ++g) {
    qs.emplace_back(ctx, devices[g]);
    buf[g] = sycl::malloc_device<float>(count, qs[g]);
    qs[g].memcpy(buf[g], in_per_gpu + g * count, count * sizeof(float));
  }
  for (auto& q : qs) q.wait();

  // Reduce onto GPU 0 via cross-device copies (peer USM in the shared context).
  float* acc = buf[0];
  float* tmp = sycl::malloc_device<float>(count, qs[0]);
  for (std::size_t g = 1; g < ng; ++g) {
    qs[0].memcpy(tmp, buf[g], count * sizeof(float)).wait();
    qs[0].parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
             acc[i] += tmp[i];
           }).wait();
  }

  // Broadcast the reduced result back to every GPU and out to the host.
  for (std::size_t g = 1; g < ng; ++g) qs[0].memcpy(buf[g], acc, count * sizeof(float));
  qs[0].memcpy(out, acc, count * sizeof(float)).wait();

  sycl::free(tmp, qs[0]);
  for (std::size_t g = 0; g < ng; ++g) sycl::free(buf[g], qs[g]);
  return ng;
}

}  // namespace quixicore::xpu::ops
