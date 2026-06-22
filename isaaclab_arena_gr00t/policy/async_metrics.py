# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Metric aggregation for control-time asynchronous VLA evaluation."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 9),
        "p50": round(float(np.percentile(array, 50)), 9),
        "p95": round(float(np.percentile(array, 95)), 9),
        "p99": round(float(np.percentile(array, 99)), 9),
        "max": round(float(array.max()), 9),
    }


def build_async_metrics(scheduler_metrics: dict[str, Any], wall_elapsed_s: float, step_dt: float) -> dict[str, Any]:
    """Combine scheduler control-time metrics with independently measured wall timing."""
    assert wall_elapsed_s >= 0.0, "wall_elapsed_s must be non-negative"
    inference_samples = [
        sample for env_metrics in scheduler_metrics["per_env"] for sample in env_metrics["inference_wall_s"]
    ]
    queue_samples = [
        sample for env_metrics in scheduler_metrics["per_env"] for sample in env_metrics["virtual_queue_wait_sim_s"]
    ]
    request_count = int(scheduler_metrics["request_count"])
    deadline_count = int(scheduler_metrics["deadline_count"])
    miss_count = int(scheduler_metrics["deadline_miss_count"])
    sim_time_s = float(scheduler_metrics["sim_time_s"])
    return {
        "num_envs": int(scheduler_metrics["num_envs"]),
        "sim_time_s": sim_time_s,
        "wall_elapsed_s": round(wall_elapsed_s, 9),
        "simulation_real_time_factor": round(sim_time_s / wall_elapsed_s, 9) if wall_elapsed_s else None,
        "deadline_window_sim_s": float(scheduler_metrics["deadline_window_s"]),
        "request_count": request_count,
        "deadline_count": deadline_count,
        "deadline_miss_count": miss_count,
        "deadline_miss_rate": round(miss_count / deadline_count, 9) if deadline_count else 0.0,
        "hold_steps": int(scheduler_metrics["hold_steps"]),
        "hold_sim_time_s": round(int(scheduler_metrics["hold_steps"]) * step_dt, 9),
        "inference_wall_s": _percentiles(inference_samples),
        "virtual_queue_wait_sim_s": _percentiles(queue_samples),
        "zero_miss": miss_count == 0,
        "per_env": scheduler_metrics["per_env"],
    }


def write_async_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    """Atomically write one async evaluation summary as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary_path.replace(output_path)


def build_async_trace(
    num_envs: int,
    step_dt: float,
    deadline_window_s: float,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the versioned per-control-step trace consumed by visualization tools."""
    return {
        "schema_version": 1,
        "num_envs": num_envs,
        "step_dt": step_dt,
        "deadline_window_sim_s": deadline_window_s,
        "frame_count": len(frames),
        "frames": frames,
    }


def write_async_trace(path: str | Path, trace: dict[str, Any]) -> None:
    """Atomically write a strict JSON asynchronous scheduler trace."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(trace, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary_path.replace(output_path)
