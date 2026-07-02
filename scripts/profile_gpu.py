#!/usr/bin/env python3
"""Run a command while sampling GPU memory and utilization with nvidia-smi."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--gpus", default="", help="Comma-separated physical GPU ids to sample, e.g. 0,1.")
    parser.add_argument("--csv", type=Path, required=True, help="Path for raw samples.")
    parser.add_argument("--summary", type=Path, required=True, help="Path for JSON summary.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to profile")
    args.gpu_filter = parse_gpu_filter(args.gpus, parser)
    return args


def parse_gpu_filter(spec: str, parser: argparse.ArgumentParser) -> set[int] | None:
    if not spec.strip():
        return None
    gpus: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if not part.isdigit():
            parser.error("--gpus must be a comma-separated list of physical GPU ids, e.g. 0,1")
        gpus.add(int(part))
    return gpus or None


def sample_gpus(gpu_filter: set[int] | None = None) -> list[dict[str, int]]:
    query = "index,memory.used,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    samples: list[dict[str, int]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        index, memory_used, util = [item.strip() for item in row]
        gpu = int(index)
        if gpu_filter is not None and gpu not in gpu_filter:
            continue
        samples.append(
            {
                "gpu": gpu,
                "memory_used_mib": int(memory_used),
                "utilization_gpu_pct": int(util),
            }
        )
    return samples


def write_summary(
    summary_path: Path,
    command: list[str],
    returncode: int,
    elapsed: float,
    sample_interval: float,
    max_mem: dict[int, int],
    util_sum: dict[int, int],
    util_count: dict[int, int],
    gpu_filter: set[int] | None = None,
    interrupted: bool = False,
) -> None:
    summary = {
        "command": command,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "sample_interval_sec": sample_interval,
        "gpus": {
            str(gpu): {
                "peak_memory_mib": max_mem[gpu],
                "avg_utilization_gpu_pct": round(util_sum[gpu] / util_count[gpu], 2),
                "samples": util_count[gpu],
            }
            for gpu in sorted(max_mem)
            if util_count.get(gpu, 0)
        },
    }
    if gpu_filter is not None:
        summary["gpus_filter"] = sorted(gpu_filter)
    if interrupted:
        summary["interrupted"] = True
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    process = subprocess.Popen(args.command, start_new_session=True)
    max_mem: dict[int, int] = {}
    util_sum: dict[int, int] = {}
    util_count: dict[int, int] = {}
    interrupted = False
    forwarded_signal: int | None = None

    def _handle_signal(signum, _frame):
        nonlocal interrupted, forwarded_signal
        interrupted = True
        forwarded_signal = signum
        if process.poll() is None:
            os.killpg(process.pid, signum)

    old_sigint = signal.signal(signal.SIGINT, _handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    try:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["elapsed_sec", "gpu", "memory_used_mib", "utilization_gpu_pct"],
            )
            writer.writeheader()
            while process.poll() is None:
                elapsed = time.time() - start
                try:
                    samples = sample_gpus(args.gpu_filter)
                except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                    print(f"[profile_gpu] nvidia-smi failed: {exc}", file=sys.stderr)
                    samples = []
                for sample in samples:
                    row = {"elapsed_sec": round(elapsed, 3), **sample}
                    writer.writerow(row)
                    gpu = int(sample["gpu"])
                    mem = int(sample["memory_used_mib"])
                    util = int(sample["utilization_gpu_pct"])
                    max_mem[gpu] = max(max_mem.get(gpu, 0), mem)
                    util_sum[gpu] = util_sum.get(gpu, 0) + util
                    util_count[gpu] = util_count.get(gpu, 0) + 1
                f.flush()
                time.sleep(args.sample_interval)
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if interrupted and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)

    returncode = process.wait()
    elapsed = time.time() - start
    if interrupted and returncode == 0 and forwarded_signal is not None:
        returncode = -forwarded_signal
    write_summary(
        args.summary,
        args.command,
        returncode,
        elapsed,
        args.sample_interval,
        max_mem,
        util_sum,
        util_count,
        gpu_filter=args.gpu_filter,
        interrupted=interrupted,
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
