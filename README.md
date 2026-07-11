# QuixiCore XPU

QuixiCore XPU is the Intel GPU implementation of the QuixiCore kernel library.

It is a standalone native implementation for Intel XPU-class accelerators using the oneAPI, SYCL, and Level Zero ecosystem. It shares no implementation code with the other QuixiCore backends.

It implements the contract defined by [QuixiAI/QuixiCore](https://github.com/QuixiAI/QuixiCore): the same operation names, quant formats, correctness expectations, benchmark methodology, and public library identity as the other QuixiCore backends.

**Native implementations. Shared contract. No shared code.**

## QuixiCore Standard Files

- Contract metadata: [`.quixicore/backend.yaml`](.quixicore/backend.yaml)
- Kernel coverage manifest: [`.quixicore/kernels.yaml`](.quixicore/kernels.yaml)
- Quant format manifest: [`.quixicore/quant-formats.yaml`](.quixicore/quant-formats.yaml)
- Repository structure: [`docs/repository-structure.md`](docs/repository-structure.md)
- Contribution workflow: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

Common developer entrypoints:

```bash
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
scripts/clean
```

These scripts keep the QuixiCore workflow consistent while wrapping CMake,
oneAPI/SYCL, Level Zero probes, and benchmark tooling.

## Status

QuixiCore XPU is an active native backend. It includes correctness-tested SYCL
and oneDNN implementations across activations, norms, attention, matrix
multiplication, quantization, MoE, sampling, state-space models, and serving
utilities. Operation-level maturity and supported variants are tracked in
[`.quixicore/kernels.yaml`](.quixicore/kernels.yaml).

The Qwen serving surface includes NVFP4 GEMM and MoE, FP8 W8A16 GEMM,
Qwen GDN decode, fused residual-add RMSNorm, and current-stream SYCL command
graph capture. These additions remain experimental while their ABI and
cross-model coverage settle; see
[`docs/qwen-serving-port.md`](docs/qwen-serving-port.md).

The current public surface is inference- and serving-first. The native C++ API
contains an explicit GELU backward kernel and AdamW consumes caller-provided
gradients, but the PyTorch binding does not register autograd formulas for its
forward operators. It exposes `gelu_backward` as a manual operation; it does
not claim training-complete backward coverage.

This repository is reserved for the native Intel GPU backend. Implementation work should live here, not in the QuixiCore umbrella repository and not in another backend repository.

## Target Platforms

- Intel Arc
- Intel Data Center GPU
- Future Intel XPU-class accelerators

## Scope

This backend is expected to own:

- oneAPI/SYCL/Level Zero source code
- Intel GPU build configuration
- Intel GPU runtime integration
- XPU-specific correctness tests
- XPU-specific benchmarks
- Backend compatibility metadata

The shared contract lives in [QuixiAI/QuixiCore](https://github.com/QuixiAI/QuixiCore).

## Quick Start

The default build does not require oneAPI. It validates the backend metadata and
project structure on any host with CMake and a C++20 compiler:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

On a machine with oneAPI DPC++ installed, enable the native kernels and tests:

```bash
source /opt/intel/oneapi/setvars.sh
cmake --preset sycl
cmake --build --preset sycl
ctest --preset sycl --output-on-failure
```

The SYCL preset expects `icpx` on `PATH`.

## Project Structure

```text
.quixicore/backend.yaml      Backend contract metadata
include/quixicore/xpu/       Public C++ headers
src/                         Native backend source
examples/                    Host and SYCL examples
tests/                       Correctness and smoke tests
perf/                        Benchmark harness notes and local results
docs/                        Backend development notes
```

## Performance Workflow

The XPU performance workflow follows the sibling backend pattern:

- [`perf/perf.md`](perf/perf.md) is the optimization handbook.
- [`perf/optimization_status.md`](perf/optimization_status.md) is the running
  implementation and tuning notebook.
- [`perf/baseline_status.md`](perf/baseline_status.md) records baseline
  snapshots.
- `perf/results/` stores raw local runs and is ignored by git.

Run the benchmark harness directly or use a focused configuration:

```bash
.venv/bin/python perf/bench_kernels.py --phase all --preset sycl
.venv/bin/python perf/bench_kernels.py --phase kernels --preset sycl
```

The kernel phase discovers the YAML sweeps in `perf/configs/`, including
`serving_qwen_decode.yaml`.

## Current Kernel Coverage

Implemented and experimental operations are listed in
[`.quixicore/kernels.yaml`](.quixicore/kernels.yaml). A family remains `partial`
until every operation required by the shared QuixiCore contract has native
correctness and benchmark coverage.

## License

MIT. See [`LICENSE`](LICENSE).
