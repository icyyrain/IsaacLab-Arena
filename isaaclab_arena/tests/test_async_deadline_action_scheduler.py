# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the control-time asynchronous action chunk scheduler."""

from __future__ import annotations

import torch

import pytest

from isaaclab_arena.policy.action_scheduling import AsyncDeadlineActionScheduler, AsyncEnvStatus


def _make_scheduler(
    num_envs: int = 1,
    action_chunk_length: int = 50,
    action_horizon: int = 50,
    prefetch_lead_steps: int = 25,
    action_start_offset_steps: int = 0,
) -> AsyncDeadlineActionScheduler:
    return AsyncDeadlineActionScheduler(
        num_envs=num_envs,
        action_chunk_length=action_chunk_length,
        action_horizon=action_horizon,
        action_dim=2,
        step_dt=0.02,
        prefetch_lead_steps=prefetch_lead_steps,
        action_start_offset_steps=action_start_offset_steps,
        device="cpu",
    )


def test_bootstrap_plays_chunk_and_requests_at_control_time_lead() -> None:
    scheduler = _make_scheduler()
    chunk = torch.arange(100, dtype=torch.float32).reshape(1, 50, 2)
    scheduler.bootstrap(chunk)

    for index in range(25):
        torch.testing.assert_close(scheduler.step(torch.zeros(1, 2))[0], chunk[0, index])
        assert scheduler.take_pending_requests() == []

    torch.testing.assert_close(scheduler.step(torch.zeros(1, 2))[0], chunk[0, 25])
    requests = scheduler.take_pending_requests()

    assert len(requests) == 1
    assert requests[0].env_id == 0
    assert requests[0].submit_sim_time_s == 0.5
    assert requests[0].deadline_sim_time_s == 1.0
    assert scheduler.statuses == [AsyncEnvStatus.QUEUED]
    assert scheduler.metrics()["per_env"][0]["virtual_finish_sim_time_s"] is None


def test_holds_at_boundary_until_virtual_completion_then_resumes() -> None:
    scheduler = _make_scheduler()
    scheduler.bootstrap(torch.ones(1, 50, 2))

    for _ in range(26):
        scheduler.step(torch.zeros(1, 2))
    request = scheduler.take_pending_requests()[0]
    scheduler.mark_inference_started(request.env_id, request.generation)
    accepted = scheduler.accept_result(
        request=request,
        chunk=torch.full((50, 2), 2.0),
        inference_wall_s=0.6,
    )
    assert accepted
    assert scheduler.statuses == [AsyncEnvStatus.GATED]

    for _ in range(24):
        scheduler.step(torch.zeros(1, 2))
    hold = torch.full((1, 2), 9.0)
    torch.testing.assert_close(scheduler.step(hold)[0], hold[0])
    assert scheduler.statuses == [AsyncEnvStatus.DEADLINE_MISS]

    for _ in range(4):
        torch.testing.assert_close(scheduler.step(hold)[0], hold[0])
    torch.testing.assert_close(scheduler.step(hold)[0], torch.tensor([2.0, 2.0]))
    assert scheduler.statuses == [AsyncEnvStatus.EXECUTING]

    metrics = scheduler.metrics()
    assert metrics["deadline_count"] == 1
    assert metrics["deadline_miss_count"] == 1
    assert metrics["hold_steps"] == 5
    assert metrics["deadline_window_s"] == 0.5


def test_accept_result_skips_prefetch_lead_when_start_offset_is_configured() -> None:
    scheduler = _make_scheduler(
        action_chunk_length=50,
        action_horizon=75,
        prefetch_lead_steps=25,
        action_start_offset_steps=25,
    )
    scheduler.bootstrap(torch.zeros(1, 75, 2))

    for _ in range(26):
        scheduler.step(torch.zeros(1, 2))
    request = scheduler.take_pending_requests()[0]

    predicted_chunk = torch.arange(150, dtype=torch.float32).reshape(75, 2)
    assert scheduler.accept_result(request, predicted_chunk, inference_wall_s=0.1)

    for _ in range(24):
        scheduler.step(torch.zeros(1, 2))
    torch.testing.assert_close(scheduler.step(torch.zeros(1, 2))[0], predicted_chunk[25])
    torch.testing.assert_close(scheduler.step(torch.zeros(1, 2))[0], predicted_chunk[26])
    assert scheduler.metrics()["action_start_offset_steps"] == 25


def test_start_offset_requires_enough_action_horizon() -> None:
    with pytest.raises(AssertionError, match="action_start_offset_steps \\+ action_chunk_length"):
        _make_scheduler(
            action_chunk_length=50,
            action_horizon=50,
            prefetch_lead_steps=25,
            action_start_offset_steps=25,
        )


def test_virtual_gpu_timeline_serializes_same_time_requests() -> None:
    scheduler = _make_scheduler(num_envs=2)
    scheduler.bootstrap(torch.ones(2, 50, 2))
    for _ in range(26):
        scheduler.step(torch.zeros(2, 2))
    requests = scheduler.take_pending_requests()

    scheduler.accept_result(requests[0], torch.full((50, 2), 2.0), inference_wall_s=0.3)
    scheduler.accept_result(requests[1], torch.full((50, 2), 3.0), inference_wall_s=0.3)

    metrics = scheduler.metrics()
    assert metrics["per_env"][0]["virtual_finish_sim_time_s"] == 0.8
    assert metrics["per_env"][1]["virtual_finish_sim_time_s"] == 1.1
    assert metrics["per_env"][1]["virtual_queue_wait_sim_s"] == pytest.approx([0.3])


def test_gpu_service_intervals_preserve_control_time_order_and_duration() -> None:
    scheduler = _make_scheduler(num_envs=2)
    scheduler.bootstrap(torch.ones(2, 50, 2))
    for _ in range(26):
        scheduler.step(torch.zeros(2, 2))
    requests = scheduler.take_pending_requests()

    scheduler.accept_result(requests[0], torch.full((50, 2), 2.0), inference_wall_s=0.3)
    scheduler.accept_result(requests[1], torch.full((50, 2), 3.0), inference_wall_s=0.2)

    assert scheduler.metrics()["gpu_service_intervals"] == [
        {
            "env_id": 0,
            "generation": 0,
            "sequence": 0,
            "submit_sim_time_s": 0.5,
            "start_sim_time_s": 0.5,
            "finish_sim_time_s": 0.8,
            "inference_wall_s": 0.3,
        },
        {
            "env_id": 1,
            "generation": 0,
            "sequence": 1,
            "submit_sim_time_s": 0.5,
            "start_sim_time_s": 0.8,
            "finish_sim_time_s": 1.0,
            "inference_wall_s": 0.2,
        },
    ]


def test_reset_generation_rejects_stale_result() -> None:
    scheduler = _make_scheduler()
    scheduler.bootstrap(torch.ones(1, 50, 2))
    for _ in range(26):
        scheduler.step(torch.zeros(1, 2))
    request = scheduler.take_pending_requests()[0]

    scheduler.reset(torch.tensor([0]))

    assert not scheduler.accept_result(request, torch.full((50, 2), 2.0), inference_wall_s=0.1)
    assert scheduler.statuses == [AsyncEnvStatus.BOOTSTRAP]
    assert scheduler.metrics()["gpu_service_intervals"] == []


def test_state_snapshot_reports_deadline_countdown_and_hold() -> None:
    scheduler = _make_scheduler()
    scheduler.bootstrap(torch.ones(1, 50, 2))

    for _ in range(26):
        scheduler.step(torch.zeros(1, 2))
    scheduler.take_pending_requests()

    snapshot = scheduler.state_snapshot()[0]
    assert snapshot == {
        "env_id": 0,
        "status": "queued",
        "action_index": 26,
        "remaining_action_steps": 24,
        "deadline_remaining_sim_s": pytest.approx(0.48),
        "generation": 0,
    }

    for _ in range(24):
        scheduler.step(torch.zeros(1, 2))
    scheduler.step(torch.full((1, 2), 9.0))

    missed = scheduler.state_snapshot()[0]
    assert missed["status"] == "deadline_miss"
    assert missed["remaining_action_steps"] == 0
    assert missed["deadline_remaining_sim_s"] == 0.0
