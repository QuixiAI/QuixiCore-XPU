# Performance

XPU performance work is tracked in `perf/`.

- `perf/perf.md` is the performance handbook.
- `perf/optimization_status.md` is the running tuning notebook.
- `perf/baseline_status.md` records baseline snapshots.
- `perf/bench_kernels.py` runs build health checks and operation benchmarks.
- `perf/configs/serving_qwen_decode.yaml` is the focused Qwen decode sweep.

Benchmark reports should include Intel GPU model, driver/runtime versions,
compiler, Level Zero/SYCL environment, input shape, dtype, quant format, and
relevant kernel variant.
