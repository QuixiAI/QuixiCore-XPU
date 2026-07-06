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

QuixiCore XPU is planned. The repository currently contains the initial backend
scaffold: compatibility metadata, CMake build files, a small C++ status library,
smoke tests, and a gated SYCL device probe.

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

On a machine with oneAPI DPC++ installed, enable the SYCL probe:

```bash
cmake --preset sycl
cmake --build --preset sycl
ctest --preset sycl
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

The current scaffold harness records configure/build/test/probe health:

```bash
python3 perf/bench_kernels.py --phase all --preset dev
```

## Current Kernel Coverage

All QuixiCore kernel families are currently planned for XPU. No kernel family is
claimed complete until native implementation, correctness tests, and benchmark
coverage are present in this repository.
