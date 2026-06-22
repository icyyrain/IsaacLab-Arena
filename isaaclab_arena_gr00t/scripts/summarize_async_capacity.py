# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize asynchronous VLA capacity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Sort capacity runs and identify the zero-miss boundary."""
    sorted_runs = sorted(runs, key=lambda run: int(run["num_envs"]))
    zero_miss_counts = [int(run["num_envs"]) for run in sorted_runs if float(run["deadline_miss_rate"]) == 0.0]
    overloaded_counts = [int(run["num_envs"]) for run in sorted_runs if float(run["deadline_miss_rate"]) > 0.0]
    return {
        "max_zero_miss_num_envs": max(zero_miss_counts) if zero_miss_counts else None,
        "first_overloaded_num_envs": min(overloaded_counts) if overloaded_counts else None,
        "runs": sorted_runs,
    }


def _format_optional(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path, help="Async metrics JSON files")
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    summary = summarize_runs(runs)

    print("robots  miss_rate  inf_p95_wall_s  queue_p95_sim_s  hold_sim_s")
    for run in summary["runs"]:
        print(
            f"{int(run['num_envs']):7d}  "
            f"{float(run['deadline_miss_rate']):9.4f}  "
            f"{_format_optional(run['inference_wall_s']['p95']):14s}  "
            f"{_format_optional(run['virtual_queue_wait_sim_s']['p95']):15s}  "
            f"{float(run['hold_sim_time_s']):10.4f}"
        )
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, sort_keys=True))


if __name__ == "__main__":
    main()
