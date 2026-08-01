# Backend Notes

QuixiCore XPU is the Intel GPU implementation of the QuixiCore contract. XPU
source may use SYCL, Level Zero, XeTLA, oneDNN, Triton XPU backend experiments,
and oneCCL when those choices remain behind the shared operation semantics.

Native SYCL and oneDNN variants share the public ABI in
`include/quixicore/xpu/ops.hpp`. New kernel work should start directly in the
semantic family layout documented in `docs/repository-structure.md`, with the
operation and its maturity recorded in `.quixicore/kernels.yaml`.

`quixicore::xpu::CommandGraph` captures only the caller's current SYCL queue.
Tensor-parallel integrations should capture one device-local compute segment per
rank and execute XCCL/oneCCL collectives outside the graph. This avoids global
device synchronization and keeps collective ordering explicit.

The FP8 KV-cache scale values and their placement remain integration-owned:
attention callers must pass the scale representation required by their runtime.
QuixiCore does not silently reinterpret framework-specific KV-cache metadata.
