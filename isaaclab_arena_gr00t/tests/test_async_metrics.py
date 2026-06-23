# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for asynchronous VLA metric aggregation."""

from __future__ import annotations

import json

from isaaclab_arena_gr00t.policy.async_metrics import (
    build_async_metrics,
    build_async_trace,
    write_async_metrics,
    write_async_trace,
)
from isaaclab_arena_gr00t.policy.async_status_ui import build_status_rows


def test_build_async_metrics_keeps_control_and_wall_clocks_separate(tmp_path) -> None:
    scheduler_metrics = {
        "num_envs": 2,
        "sim_time_s": 2.0,
        "deadline_window_s": 0.5,
        "request_count": 8,
        "deadline_count": 4,
        "deadline_miss_count": 1,
        "hold_steps": 5,
        "per_env": [
            {"inference_wall_s": [0.1, 0.2], "virtual_queue_wait_sim_s": [0.0, 0.1]},
            {"inference_wall_s": [0.3, 0.4], "virtual_queue_wait_sim_s": [0.2, 0.3]},
        ],
    }

    metrics = build_async_metrics(scheduler_metrics, wall_elapsed_s=10.0, step_dt=0.02)

    assert metrics["deadline_window_sim_s"] == 0.5
    assert metrics["simulation_real_time_factor"] == 0.2
    assert metrics["deadline_miss_rate"] == 0.25
    assert metrics["hold_sim_time_s"] == 0.1
    assert metrics["inference_wall_s"]["p95"] == 0.385
    assert metrics["zero_miss"] is False

    output_path = tmp_path / "metrics.json"
    write_async_metrics(output_path, metrics)
    assert json.loads(output_path.read_text()) == metrics


def test_build_status_rows_marks_deadline_miss_red() -> None:
    rows = build_status_rows(["executing", "queued", "inference", "deadline_miss"])

    assert rows[0] == ("Robot 0  EXECUTING", "green")
    assert rows[1] == ("Robot 1  QUEUED", "yellow")
    assert rows[2] == ("Robot 2  INFERENCE", "blue")
    assert rows[3] == ("Robot 3  DEADLINE MISS / HOLD", "red")


def test_async_trace_preserves_per_step_robot_state(tmp_path) -> None:
    frames = [{
        "step": 0,
        "sim_time_s": 0.02,
        "queue_depth": 1,
        "active_inference_env_id": 0,
        "deadline_miss_count": 0,
        "robots": [{
            "env_id": 0,
            "status": "inference",
            "action_index": 26,
            "remaining_action_steps": 24,
            "deadline_remaining_sim_s": 0.48,
            "generation": 0,
        }],
    }]
    trace = build_async_trace(
        num_envs=1,
        step_dt=0.02,
        deadline_window_s=0.5,
        frames=frames,
        gpu_service_intervals=[
            {
                "env_id": 0,
                "generation": 0,
                "sequence": 0,
                "submit_sim_time_s": 0.5,
                "start_sim_time_s": 0.5,
                "finish_sim_time_s": 0.63,
                "inference_wall_s": 0.13,
            }
        ],
    )

    assert trace["schema_version"] == 1
    assert trace["frame_count"] == 1
    assert trace["frames"][0]["robots"][0]["deadline_remaining_sim_s"] == 0.48
    assert trace["gpu_service_intervals"][0]["finish_sim_time_s"] == 0.63

    path = tmp_path / "trace.json"
    write_async_trace(path, trace)
    assert json.loads(path.read_text()) == trace
