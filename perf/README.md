# Benchmarking

QuixiCore XPU benchmarks should use native Intel timing and profiling tools.

The operating guide is [`perf.md`](perf.md). The running status notebook is
[`optimization_status.md`](optimization_status.md), with baseline snapshots in
[`baseline_status.md`](baseline_status.md).

The scaffold harness records configure/build/test/probe results in the same
layout used by the sibling backends:

```bash
python3 perf/bench_kernels.py --phase all --preset dev
```

When oneAPI is available:

```bash
python3 perf/bench_kernels.py --phase all --preset sycl
```

Initial reporting should include:

- QuixiCore XPU commit.
- QuixiCore contract version.
- Intel GPU target and driver/runtime versions.
- oneAPI compiler version.
- Kernel family, operation, dtype, quant format, and shape.
- Warmup iterations and measured iterations.
- Latency summary.
- Throughput summary where applicable.
- Correctness tolerance and observed error.

Raw benchmark output should be written under `perf/results/`, which is ignored by
git. Summaries that matter for future work should be copied into a tracked
status document.
