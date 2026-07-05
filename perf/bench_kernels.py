#!/usr/bin/env python3
"""QuixiCore XPU scaffold benchmark and build-health harness.

The sibling backends record build/test/perf output under perf/results/. This
script starts the same pattern for XPU. Today it records scaffold health:
configure, build, CTest, backend-info output, and optional SYCL device probe.

Examples:

    python3 perf/bench_kernels.py --phase all --preset dev
    python3 perf/bench_kernels.py --phase all --preset sycl

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
        lines.append(
            f"| {row['phase']} | {row['status']} | {row['seconds']:.3f} | `{row['log']}` |"
        )
    lines.append("")
    out_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "configure", "build", "test", "info", "probe"],
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
    for name, cmd in command_plan(args.phase, args.preset, args.include_probe):
        row = run_command(name, cmd, out_dir, args.timeout)
        rows.append(row)
        if row["status"] != "ok":
            break

    with out_dir.joinpath("results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_summary(rows, out_dir, meta)

    for row in rows:
        print(f"{row['phase']}: {row['status']} ({row['seconds']:.3f}s)")

    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
