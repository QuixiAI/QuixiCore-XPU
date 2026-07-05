# QuixiCore XPU

QuixiCore XPU is the Intel GPU implementation of the QuixiCore kernel library.

It is a standalone native implementation for Intel XPU-class accelerators using the oneAPI, SYCL, and Level Zero ecosystem. It shares no implementation code with the other QuixiCore backends.

It implements the contract defined by [QuixiAI/QuixiCore](https://github.com/QuixiAI/QuixiCore): the same operation names, quant formats, correctness expectations, benchmark methodology, and public library identity as the other QuixiCore backends.

**Native implementations. Shared contract. No shared code.**

## Status

QuixiCore XPU is planned.

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

