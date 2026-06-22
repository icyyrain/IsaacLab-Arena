# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the async VLA capacity summary."""

from __future__ import annotations

from isaaclab_arena_gr00t.scripts.summarize_async_capacity import summarize_runs


def test_summarize_runs_finds_zero_miss_capacity_and_first_overload() -> None:
    runs = [
        {"num_envs": 4, "deadline_miss_rate": 0.25},
        {"num_envs": 2, "deadline_miss_rate": 0.0},
        {"num_envs": 3, "deadline_miss_rate": 0.0},
        {"num_envs": 5, "deadline_miss_rate": 0.4},
    ]

    summary = summarize_runs(runs)

    assert summary["max_zero_miss_num_envs"] == 3
    assert summary["first_overloaded_num_envs"] == 4
    assert [run["num_envs"] for run in summary["runs"]] == [2, 3, 4, 5]
