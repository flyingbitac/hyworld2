#!/usr/bin/env python3
"""Run a command while sampling GPU memory and utilization with nvidia-smi."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, required=True, help="Path for raw samples.")
    parser.add_argument("--summary", type=Path, required=True, help="Path for JSON summary.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to profile")
    return args


def sample_gpus() -> list[dict[str, int]]:
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
        samples.append(
            {
                "gpu": int(index),
                "memory_used_mib": int(memory_used),
                "utilization_gpu_pct": int(util),
            }
        )
    return samples


def main() -> int:
    args = parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    process = subprocess.Popen(args.command)
    rows: list[dict[str, float | int]] = []
    max_mem: dict[int, int] = {}
    util_sum: dict[int, int] = {}
    util_count: dict[int, int] = {}

    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["elapsed_sec", "gpu", "memory_used_mib", "utilization_gpu_pct"],
        )
        writer.writeheader()
        while process.poll() is None:
            elapsed = time.time() - start
            try:
                samples = sample_gpus()
            except subprocess.CalledProcessError as exc:
                print(f"[profile_gpu] nvidia-smi failed: {exc}", file=sys.stderr)
                samples = []
            for sample in samples:
                row = {"elapsed_sec": round(elapsed, 3), **sample}
                writer.writerow(row)
                rows.append(row)
                gpu = int(sample["gpu"])
                mem = int(sample["memory_used_mib"])
                util = int(sample["utilization_gpu_pct"])
                max_mem[gpu] = max(max_mem.get(gpu, 0), mem)
                util_sum[gpu] = util_sum.get(gpu, 0) + util
                util_count[gpu] = util_count.get(gpu, 0) + 1
            f.flush()
            time.sleep(args.sample_interval)

    returncode = process.wait()
    elapsed = time.time() - start
    summary = {
        "command": args.command,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 3),
        "sample_interval_sec": args.sample_interval,
        "gpus": {
            str(gpu): {
                "peak_memory_mib": max_mem[gpu],
                "avg_utilization_gpu_pct": round(util_sum[gpu] / util_count[gpu], 2),
                "samples": util_count[gpu],
            }
            for gpu in sorted(max_mem)
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
