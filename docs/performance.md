# Performance

XPU performance work is tracked in `perf/`.

- `perf/perf.md` is the performance handbook.
- `perf/optimization_status.md` is the running tuning notebook.
- `perf/baseline_status.md` records baseline snapshots.
- `perf/bench_kernels.py` records scaffold health today and should grow into
  operation benchmarks as kernels land.

Benchmark reports should include Intel GPU model, driver/runtime versions,
compiler, Level Zero/SYCL environment, input shape, dtype, quant format, and
relevant kernel variant.
