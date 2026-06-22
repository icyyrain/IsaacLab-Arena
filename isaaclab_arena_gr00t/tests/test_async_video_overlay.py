# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for rendering asynchronous scheduler state into videos and timelines."""

from __future__ import annotations

import numpy as np

from isaaclab_arena_gr00t.utils.async_video_overlay import build_status_timeline, overlay_status_frame


def _trace_frame() -> dict:
    statuses = ["executing", "queued", "inference", "gated", "deadline_miss", "bootstrap"]
    return {
        "step": 25,
        "sim_time_s": 0.5,
        "queue_depth": 3,
        "active_inference_env_id": 2,
        "deadline_miss_count": 1,
        "robots": [
            {
                "env_id": env_id,
                "status": status,
                "deadline_remaining_sim_s": 0.5 - env_id * 0.1,
            }
            for env_id, status in enumerate(statuses)
        ],
    }


def test_overlay_status_frame_draws_all_robot_state_colors() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    rendered = overlay_status_frame(frame, _trace_frame())

    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0
    for expected_bgr in ((85, 180, 45), (40, 200, 245), (220, 125, 45), (55, 55, 225)):
        assert np.any(np.all(rendered == expected_bgr, axis=2)), expected_bgr


def test_status_timeline_has_one_colored_lane_per_robot_and_queue_plot() -> None:
    trace = {
        "num_envs": 6,
        "step_dt": 0.02,
        "frames": [_trace_frame(), {**_trace_frame(), "step": 26, "sim_time_s": 0.52, "queue_depth": 4}],
    }

    timeline = build_status_timeline(trace, width=800)

    assert timeline.shape[1] == 800
    assert timeline.shape[0] >= 420
    assert np.count_nonzero(timeline) > 0
