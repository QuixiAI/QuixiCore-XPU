# perf/ — QuixiCore XPU benchmarking

Native Intel-GPU timing (SYCL queue-profiling events), disciplined by
[`perf.md`](perf.md).

| File | What it is |
|---|---|
| [`perf.md`](perf.md) | the handbook: prime directive (smash truisms), roofline, harness, technique catalogue, decision rules |
| [`optimization_status.md`](optimization_status.md) | the running log — one dated entry per kernel/pass, baseline → after, keep/reject verdict |
| [`baseline_status.md`](baseline_status.md) | environment snapshot + current per-family roofline table |
| `harness/xpu_bench.cpp` | the C++ bench (one JSON line per run: median device ms + throughput) |
| `bench_kernels.py` | orchestrator — runs the C++ bench per `configs/*.yaml` entry, logs a dated run under `results/` |
| `configs/*.yaml` | tracked sweep definitions (variant × dtype × shape) for the kernels we baseline regularly |
| `results/` | raw dated JSON per run — git-ignored (machine-specific) |

## Run it

```bash
source /opt/intel/oneapi/setvars.sh

# one-off A/B during an optimization pass:
cmake --build --preset sycl --target quixicore_xpu_bench
./build-sycl/quixicore_xpu_bench --kernel rms_norm --dtype bf16 \
    --rows 8192 --dim 8192 --variant best --iters 50 --warmup 15

# tracked, logged sweep over the configured kernels:
python3 perf/bench_kernels.py --phase kernels --preset sycl
```

Every result carries: kernel/variant/dtype/shape, median device ms, derived
throughput (GB/s, GFLOP/s, TOPS), and device name. Correctness is gated
separately — `ctest --preset sycl` (fp64 oracle) and
`bindings/pytorch/test_parity.py` (torch.xpu). A number is not a win until both
gates pass; see the performance gate in `perf.md`.

Summaries that matter for the next pass go into `optimization_status.md` (tracked);
raw `results/` runs stay local.
