#!/usr/bin/env python3
"""QuixiCore XPU benchmark + build-health orchestrator.

Runs the native C++ bench (quixicore_xpu_bench, SYCL profiling-event timing) once
per entry in perf/configs/*.yaml -- or the built-in DEFAULT_KERNEL_BENCH matrix
when no config/PyYAML is present -- and logs a dated run under perf/results/.
Also captures build-health phases (configure/build/test/info/probe). See
perf/perf.md for the methodology and perf/optimization_status.md for the log.

Examples:

    python3 perf/bench_kernels.py --phase kernels --preset sycl   # kernel sweep
    python3 perf/bench_kernels.py --phase all     --preset sycl   # + build health
    python3 perf/bench_kernels.py --phase all     --preset dev    # non-SYCL scaffold

Results land in:

    perf/results/YYYY-MM-DD/<run-id>/run.json
    perf/results/YYYY-MM-DD/<run-id>/results.jsonl
    perf/results/YYYY-MM-DD/<run-id>/summary.md
    perf/results/YYYY-MM-DD/<run-id>/logs/*.log
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "perf" / "results"
SCHEMA_VERSION = 1

# Default kernel benchmark matrix, used when a perf/configs/*.yaml is not present
# or PyYAML is unavailable. Each entry drives one run of quixicore_xpu_bench.
DEFAULT_KERNEL_BENCH = [
    {"kernel": "gelu", "variant": "sycl", "dtype": "f32", "n": 4194304},
    {"kernel": "gelu", "variant": "vendor", "dtype": "f32", "n": 4194304},
    {"kernel": "gelu", "variant": "sycl", "dtype": "bf16", "n": 4194304},
    {"kernel": "gelu", "variant": "sycl", "dtype": "f16", "n": 4194304},
    {"kernel": "rms_norm", "variant": "sycl", "dtype": "f32", "rows": 8192, "dim": 4096},
    {"kernel": "rms_norm", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 4096},
    {"kernel": "layernorm", "variant": "sycl", "dtype": "f32", "rows": 8192, "dim": 4096},
    {"kernel": "layernorm", "variant": "vendor", "dtype": "f32", "rows": 8192, "dim": 4096},
    {"kernel": "layernorm", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 4096},
    {"kernel": "layernorm", "variant": "vendor", "dtype": "bf16", "rows": 8192, "dim": 4096},
    {"kernel": "softmax", "variant": "sycl", "dtype": "f32", "rows": 8192, "dim": 4096},
    {"kernel": "softmax", "variant": "vendor", "dtype": "f32", "rows": 8192, "dim": 4096},
    {"kernel": "softmax", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 4096},
    {"kernel": "softmax", "variant": "vendor", "dtype": "bf16", "rows": 8192, "dim": 4096},
    {"kernel": "silu", "variant": "sycl", "dtype": "bf16", "n": 4194304},
    {"kernel": "glu", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 4096},
    {"kernel": "gelu_backward", "variant": "sycl", "dtype": "bf16", "n": 4194304},
    {"kernel": "dense_gemm", "variant": "sycl", "dtype": "bf16", "M": 2048, "N": 2048, "K": 2048},
    {"kernel": "dense_gemm", "variant": "vendor", "dtype": "bf16", "M": 2048, "N": 2048, "K": 2048},
    {"kernel": "rope", "variant": "sycl", "dtype": "bf16", "rows": 65536, "dim": 128},
    {"kernel": "adamw", "variant": "sycl", "dtype": "f32", "n": 16777216},
    {"kernel": "argmax", "variant": "sycl", "dtype": "f32", "rows": 4096, "dim": 131072},
    {"kernel": "qgemv_int4", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "qgemm_int8", "variant": "vendor", "dtype": "bf16", "M": 2048, "N": 2048, "K": 2048},
    {"kernel": "fp8_gemm", "variant": "vendor", "dtype": "bf16", "approx": "e4m3", "M": 2048, "N": 2048, "K": 2048},
    {"kernel": "mxfp4_gemv", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "nvfp4_gemv", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "gguf_gemv", "variant": "sycl", "dtype": "bf16", "approx": "q4_0", "rows": 8192, "dim": 8192},
    {"kernel": "gguf_gemv", "variant": "sycl", "dtype": "bf16", "approx": "q4_K", "rows": 8192, "dim": 8192},
    {"kernel": "gguf_gemv", "variant": "sycl", "dtype": "bf16", "approx": "q6_K", "rows": 8192, "dim": 8192},
    {"kernel": "gguf_gemv", "variant": "sycl", "dtype": "bf16", "approx": "iq4_nl", "rows": 8192, "dim": 8192},
    {"kernel": "gguf_gemv", "variant": "sycl", "dtype": "bf16", "approx": "iq2_xxs", "rows": 8192, "dim": 8192},
    {"kernel": "act_quant", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "quantize_int4", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "attention", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "softmax", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "cross_entropy", "variant": "sycl", "dtype": "bf16", "rows": 8192, "dim": 8192},
    {"kernel": "hadamard", "variant": "sycl", "dtype": "f32", "rows": 8192, "dim": 4096},
    {"kernel": "dropout", "variant": "sycl", "dtype": "bf16", "n": 16777216},
    {"kernel": "sample_categorical", "variant": "sycl", "dtype": "f32", "rows": 64, "dim": 131072},
    {"kernel": "moe_route", "variant": "sycl", "dtype": "f32", "rows": 8192, "dim": 256},
    {"kernel": "selective_scan", "variant": "sycl", "dtype": "f32", "rows": 4096, "dim": 4096},
    {"kernel": "linear_attn", "variant": "sycl", "dtype": "f32", "rows": 4096, "dim": 4096},
]


def git_label() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            text=True,
        ).strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def compiler_version(exe: str) -> str | None:
    path = shutil.which(exe)
    if path is None:
        return None
    try:
        proc = subprocess.run(
            [path, "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = (proc.stdout or proc.stderr).splitlines()
        return first_line[0] if first_line else path
    except Exception as exc:
        return f"{path}: {type(exc).__name__}: {exc}"


def binary_dir_for_preset(preset: str) -> Path:
    if preset == "sycl":
        return REPO_ROOT / "build-sycl"
    return REPO_ROOT / "build"


def run_command(name: str, cmd: list[str], out_dir: Path, timeout: int) -> dict:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "ok" if proc.returncode == 0 else "error"
        output = proc.stdout + proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = (exc.stdout or "") + (exc.stderr or "")
        returncode = None
    except FileNotFoundError as exc:
        status = "missing"
        output = f"{type(exc).__name__}: {exc}\n"
        returncode = None

    seconds = time.perf_counter() - start
    log_path.write_text(
        f"# cmd: {' '.join(cmd)}\n# cwd: {REPO_ROOT}\n\n{output}",
        encoding="utf-8",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": name,
        "command": cmd,
        "status": status,
        "returncode": returncode,
        "seconds": round(seconds, 3),
        "log": str(log_path.relative_to(out_dir)),
    }


def command_plan(phase: str, preset: str, include_probe: bool) -> list[tuple[str, list[str]]]:
    build_dir = binary_dir_for_preset(preset)
    info_exe = build_dir / "quixicore_xpu_info"
    probe_exe = build_dir / "quixicore_xpu_sycl_device_probe"

    commands: list[tuple[str, list[str]]] = []
    selected = ["configure", "build", "test", "info"]
    if preset == "sycl" or include_probe:
        selected.append("probe")
    if phase != "all":
        selected = [phase]

    for item in selected:
        if item == "configure":
            commands.append(("configure", ["cmake", "--preset", preset]))
        elif item == "build":
            commands.append(("build", ["cmake", "--build", "--preset", preset]))
        elif item == "test":
            commands.append(("test", ["ctest", "--preset", preset]))
        elif item == "info":
            commands.append(("info", [str(info_exe)]))
        elif item == "probe":
            commands.append(("probe", [str(probe_exe)]))
        else:
            raise ValueError(item)

    return commands


def load_kernel_bench_matrix() -> list[dict]:
    """Full-coverage kernel matrix. DEFAULT_KERNEL_BENCH is the always-present
    base (one broad run per shipped kernel, no third-party dep). A
    perf/configs/<op>.yaml, when PyYAML is available, provides a richer sweep and
    *overrides* the default entry for the kernels it covers. So a run always
    covers the whole matrix, with detailed sweeps where a config exists."""
    configs_dir = REPO_ROOT / "perf" / "configs"
    config_runs: list[dict] = []
    try:
        import yaml  # type: ignore
        if configs_dir.is_dir():
            for path in sorted(configs_dir.glob("*.yaml")):
                try:
                    doc = yaml.safe_load(path.read_text()) or {}
                except Exception:
                    continue
                config_runs.extend(doc.get("runs", []))
    except Exception:
        pass  # PyYAML absent -> default matrix only
    covered = {r["kernel"] for r in config_runs if "kernel" in r}
    base = [r for r in DEFAULT_KERNEL_BENCH if r["kernel"] not in covered]
    return config_runs + base


def run_kernel_bench(preset: str, out_dir: Path, timeout: int) -> list[dict]:
    """Run the on-device kernel benchmark for each config and collect its JSON
    output. Only meaningful for the sycl preset (needs an Intel GPU)."""
    bench_exe = binary_dir_for_preset(preset) / "quixicore_xpu_bench"
    if not bench_exe.exists():
        return [{
            "schema_version": SCHEMA_VERSION, "phase": "kernels",
            "status": "missing", "returncode": None, "seconds": 0.0,
            "log": "kernels", "note": f"{bench_exe} not built",
        }]

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cfg in load_kernel_bench_matrix():
        cmd = [str(bench_exe)]
        for key in ("kernel", "variant", "dtype", "approx", "n", "rows", "dim",
                    "M", "N", "K", "iters", "warmup"):
            if key in cfg:
                cmd += [f"--{key}", str(cfg[key])]
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=timeout)
            status = "ok" if proc.returncode == 0 else "error"
            metrics = {}
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        metrics = json.loads(line)
                    except Exception:
                        pass
        except subprocess.TimeoutExpired:
            status, metrics = "timeout", {}
        seconds = round(time.perf_counter() - start, 3)
        row = {"schema_version": SCHEMA_VERSION, "phase": "kernels",
               "status": status, "seconds": seconds, "config": cfg}
        row.update({k: metrics[k] for k in metrics if k != "schema_version"})
        rows.append(row)
    return rows


def write_summary(rows: list[dict], out_dir: Path, meta: dict) -> None:
    lines = [
        "# QuixiCore XPU Harness Summary",
        "",
        f"- Run id: `{meta['run_id']}`",
        f"- Preset: `{meta['preset']}`",
        f"- Git: `{meta['git']}`",
        "",
        "| Phase | Status | Seconds | Log |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        log = row.get("log", "")
        lines.append(
            f"| {row['phase']} | {row['status']} | {row['seconds']:.3f} | `{log}` |"
        )

    kernel_rows = [r for r in rows if r["phase"] == "kernels" and "median_ms" in r]
    if kernel_rows:
        lines += [
            "",
            "## Kernel benchmarks",
            "",
            "| Kernel | Variant | Dtype | n | Median ms | GB/s |",
            "|---|---|---|---:|---:|---:|",
        ]
        for r in kernel_rows:
            lines.append(
                f"| {r.get('kernel','')} | {r.get('variant','')} | {r.get('dtype','')} "
                f"| {r.get('n','')} | {r.get('median_ms',0):.4f} | {r.get('gbps',0):.1f} |"
            )
    lines.append("")
    out_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "configure", "build", "test", "info", "probe", "kernels"],
        default="all",
    )
    parser.add_argument("--preset", default="dev", choices=["dev", "sycl"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--include-probe",
        action="store_true",
        help="Run the SYCL device probe even when the selected preset is not sycl.",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"{args.preset}-{uuid.uuid4().hex[:8]}"
    today = dt.date.today().isoformat()
    out_dir = RESULTS_ROOT / today / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_label(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "preset": args.preset,
        "phase": args.phase,
        "cmake": compiler_version("cmake"),
        "cxx": os.environ.get("CXX") or compiler_version("c++"),
        "icpx": compiler_version("icpx"),
    }
    out_dir.joinpath("run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    rows = []
    # Standard build-health phases (skip when only the kernel bench is requested).
    if args.phase != "kernels":
        for name, cmd in command_plan(args.phase, args.preset, args.include_probe):
            row = run_command(name, cmd, out_dir, args.timeout)
            rows.append(row)
            if row["status"] != "ok":
                break

    # On-device kernel benchmark (sycl preset only). Runs after a clean build.
    standard_ok = all(row["status"] == "ok" for row in rows)
    if args.preset == "sycl" and args.phase in ("all", "kernels") and (
        args.phase == "kernels" or standard_ok
    ):
        rows.extend(run_kernel_bench(args.preset, out_dir, args.timeout))

    with out_dir.joinpath("results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_summary(rows, out_dir, meta)

    for row in rows:
        print(f"{row['phase']}: {row['status']} ({row['seconds']:.3f}s)")

    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
