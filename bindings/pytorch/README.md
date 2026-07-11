# QuixiCore XPU - PyTorch binding (`tk_xpu`)

A co-equal PyTorch binding (mirroring the Metal backend's PyTorch+MLX bindings):
each op takes `torch.xpu` tensors, pulls the SYCL queue straight off the tensor's
current XPU stream (`c10::xpu::getCurrentXPUStream().queue()`), and calls the same
`quixicore::xpu::ops` ABI on the tensor's USM `data_ptr()` — **no data copies**.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
source ../../.venv/bin/activate       # torch==2.14.dev+xpu
uv pip install --python ../../.venv/bin/python ninja
./build.sh                            # shared ops lib + extension
../../.venv/bin/python test_parity.py # vs torch.xpu eager kernels
```

`test_parity.py` compares `tk_xpu.<op>` against PyTorch's own XPU kernels
(`F.gelu`, `F.softmax`, `F.layer_norm`, `F.scaled_dot_product_attention`, `@`, …)
across f32/bf16/f16 — a second correctness oracle alongside the C++ fp64 gate.
All ops pass (max abs error at storage-dtype epsilon).

## Two non-obvious requirements (learned the hard way)

The binding initially **segfaulted at kernel submit**. The backtrace pointed at
`ProgramManager::getDeviceKernelInfo`, which *looks* like a SYCL runtime version
skew — but torch's bundled `libsycl.so.9` and the system `icpx` are the **exact
same build** (`20260331`), with identical headers. The real causes were mundane
and are both handled by `build.sh` / `setup.py`:

1. **The ops must be a SHARED library built with `icpx -fsycl`.** Our first attempt
   linked the prebuilt **static** `libquixicore_xpu_ops.a` into the extension. The
   final extension link is done by plain `c++`, which does **not** perform the SYCL
   device link, and a static archive's device-image registration constructors don't
   get wired into the runtime `ProgramManager`. So `getDeviceKernelInfo` looks up a
   kernel that was never registered → segfault. Building the ops as a **shared
   object** with `icpx -fsycl` (so it self-registers its device images at load)
   fixes it. `build.sh` configures `-DBUILD_SHARED_LIBS=ON`.

2. **The binding source must end in `.sycl`.** `torch.utils.cpp_extension`'s
   `SyclExtension` only applies `-fsycl` to files whose extension is exactly
   `.sycl` (`_is_sycl_file`). A `.cpp` binding is compiled *without* `-fsycl`, so no
   device link happens at the end and the shared lib's device images aren't pulled
   in. Hence `tk_xpu_ext.sycl`. (It defines no kernels itself — the `.sycl`
   extension is what forces the final `icpx -fsycl` device link.)

Diagnosis proof: a minimal `.sycl` extension with an inline named kernel runs
correctly on a `torch.xpu` tensor, confirming the SYCL↔torch interop path itself
is sound; only the static-archive linkage was the problem.

## Ops exposed

`gelu` (erf/tanh), manual `gelu_backward`, `silu`, `softmax`, `rms_norm`,
`fused_add_rms_norm`, `layernorm`, `dense_gemm`, `fp8_gemm_w8a16`,
`nvfp4_gemm`, `nvfp4_moe`, `qwen_gdn_decode`, `attention` (flash, GQA,
causal), `argmax`, plus `device_count`. Each routes `Variant::best` where both
a native SYCL and a oneDNN variant exist.

This binding is inference-first and does not register PyTorch autograd formulas
for its forward operators. Passing a tensor with `requires_grad=True` therefore
does not create a supported QuixiCore gradient path. Call `gelu_backward`
explicitly when that primitive is needed; full training support would require
registered autograd formulas and backward kernels for the remaining operations.

`XPUGraph` captures and replays the current PyTorch XPU stream without a
device-wide synchronization. Inputs and outputs retain their addresses across
replay, so callers must mutate existing tensor storage rather than replace the
tensors. Multi-device collectives remain outside each rank's captured graph.
The wrapper exposes explicit `instantiate`, reset-safe stream lifetime, pool
validation, and the underlying graph debug controls.
Graph-captured split MoE also requires caller-owned persistent fp32 scratch with
shape `[M * top_k, 2 * I]`; pass it as `split_scratch` and retain it for the
graph's lifetime. Eager split MoE allocates scratch automatically when omitted.
