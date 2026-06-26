# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Gr00tRemoteClosedloopPolicy.

Exercises the real obs/action translation pipeline and mocks only ``Gr00tPolicyClient`` -- the wire
boundary to a remote GR00T server -- so no server is required.

Verifies:
- the obs dict sent to the server has the expected language/video/state structure;
- the action dict returned by the server is converted to a tensor of the
  expected (num_envs, action_chunk_length, action_dim) shape;
- ``reset`` propagates to the client;
- a failing ping during construction raises ``ConnectionError``.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from typing import Any

import pytest

pytestmark = pytest.mark.gr00t_policy


from isaaclab_arena_gr00t.tests.utils.constants import TestConstants as Gr00tTestConstants

NUM_ENVS = 2
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
NUM_CHANNELS = 3
NUM_SIM_JOINTS = 43  # G1 43-DoF
ACTION_HORIZON = 50
ACTION_CHUNK_LENGTH = 50
EXPECTED_ACTION_DIM = 50  # 43 joints + 3 nav + 1 base height + 3 torso rpy

# Joint group sizes from gr00t_43dof_joint_space.yaml (must match the YAML).
POLICY_GROUP_SIZES = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "left_hand": 7,
    "right_arm": 7,
    "right_hand": 7,
}


# ----------------------------- fixtures ------------------------------ #


@pytest.fixture
def policy_config_yaml():
    """Return the g1 locomanip test config used for Arena-side obs/action translation."""
    return (
        Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )


@pytest.fixture
def synthetic_observation():
    """Env-shaped torch obs: one ego camera + 43 joint positions per env."""
    rgb = torch.randint(0, 255, (NUM_ENVS, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS), dtype=torch.uint8)
    joint_pos = torch.randn(NUM_ENVS, NUM_SIM_JOINTS, dtype=torch.float32)
    return {
        "camera_obs": {"robot_head_cam_rgb": rgb},
        "policy": {"robot_joint_pos": joint_pos},
    }


def _make_action_response(num_envs: int, horizon: int) -> dict[str, np.ndarray]:
    """Synthetic GR00T-server response in the format ``build_gr00t_action_np`` expects."""
    response: dict[str, np.ndarray] = {}
    for group, size in POLICY_GROUP_SIZES.items():
        response[group] = np.random.randn(num_envs, horizon, size).astype(np.float32)
    response["navigate_command"] = np.random.randn(num_envs, horizon, 3).astype(np.float32)
    response["base_height_command"] = np.random.randn(num_envs, horizon, 1).astype(np.float32)
    return response


class _FakePolicyClient:
    """Minimal stand-in for ``gr00t.policy.server_client.PolicyClient``."""

    def __init__(self, *args, ping_ok: bool = True, **kwargs):
        self.init_kwargs = kwargs
        self._ping_ok = ping_ok
        self.last_observation: dict[str, Any] | None = None
        self.observations: list[dict[str, Any]] = []
        self.get_action_calls = 0
        self.reset_called = False

    def ping(self) -> bool:
        return self._ping_ok

    def get_action(self, observation: dict[str, Any]):
        self.last_observation = observation
        self.observations.append(observation)
        self.get_action_calls += 1
        batch_size = observation["video"]["ego_view"].shape[0]
        # Match the real PolicyClient return signature: (action_dict, latency_or_meta).
        return _make_action_response(batch_size, ACTION_HORIZON), None

    def reset(self):
        self.reset_called = True


@pytest.fixture
def fake_client_factory(monkeypatch):
    """Patch ``Gr00tPolicyClient`` in the policy module's namespace and return
    a factory that gives access to the most recently constructed fake."""
    from isaaclab_arena_gr00t.policy import gr00t_remote_closedloop_policy as mod

    created: list[_FakePolicyClient] = []

    def factory(ping_ok: bool = True):
        def _ctor(*args, **kwargs):
            client = _FakePolicyClient(*args, ping_ok=ping_ok, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(mod, "Gr00tPolicyClient", _ctor)
        return created

    return factory


def _build_policy(
    policy_config_yaml: str,
    action_scheduler_cls=None,
    scheduler_mode: str | None = None,
    async_metrics_path: str | None = None,
    async_trace_path: str | None = None,
    async_time_aligned: bool = False,
    async_action_chunk_length: int | None = None,
    async_action_start_offset_steps: int = 0,
    async_prefetch_lead_steps: int = 25,
    async_hold_mode: str = "last_action",
):
    from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
        Gr00tRemoteClosedloopPolicy,
        Gr00tRemoteClosedloopPolicyArgs,
    )

    if action_scheduler_cls is None:
        from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

        action_scheduler_cls = ActionChunkScheduler

    args = Gr00tRemoteClosedloopPolicyArgs(
        policy_config_yaml_path=policy_config_yaml,
        policy_device="cpu",  # keep the test portable; remote policy does no GPU compute
        num_envs=NUM_ENVS,
        remote_host="unused",
        remote_port=0,
        remote_api_token=None,
        async_prefetch_lead_steps=async_prefetch_lead_steps,
        async_time_aligned=async_time_aligned,
        async_step_dt=0.02,
        async_network_delay_s=0.0,
        async_metrics_path=async_metrics_path,
        async_trace_path=async_trace_path,
        async_status_ui=False,
        async_action_chunk_length=async_action_chunk_length,
        async_action_start_offset_steps=async_action_start_offset_steps,
        async_hold_mode=async_hold_mode,
    )
    return Gr00tRemoteClosedloopPolicy(
        args,
        action_scheduler_cls=action_scheduler_cls,
        scheduler_mode=scheduler_mode,
    )


# ------------------------------- tests ------------------------------- #


def test_observation_sent_to_server_has_expected_structure(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """The dict passed to ``PolicyClient.get_action`` matches GR00T's expected
    language/video/state layout for the g1 sim WBC modality config."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    policy._get_action_chunk(synthetic_observation, ["robot_head_cam_rgb"])

    assert len(clients) == 1
    sent = clients[0].last_observation
    assert sent is not None and clients[0].get_action_calls == 1

    # language: one [task] entry per env
    assert "language" in sent
    lang = sent["language"]["annotation.human.task_description"]
    assert lang == [["pick up the brown box"]] * NUM_ENVS

    # video: one ego view, shaped (N, 1, H, W, C) and uint8
    assert "video" in sent and "ego_view" in sent["video"]
    ego = sent["video"]["ego_view"]
    assert ego.shape == (NUM_ENVS, 1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS)
    assert ego.dtype == np.uint8

    # state: one (N, 1, dof_in_group) entry per modality state key
    expected_state_keys = {"left_arm", "right_arm", "left_hand", "right_hand", "waist"}
    assert set(sent["state"].keys()) == expected_state_keys
    for key in expected_state_keys:
        arr = sent["state"][key]
        assert arr.ndim == 3
        assert arr.shape[0] == NUM_ENVS
        assert arr.shape[1] == 1
        assert arr.shape[2] == POLICY_GROUP_SIZES[key]


def test_action_response_is_translated_to_correct_tensor_shape(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """Server's grouped joint dict is reassembled into a (N, chunk, action_dim) tensor."""
    fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    action = policy._get_action_chunk(synthetic_observation, ["robot_head_cam_rgb"])

    assert isinstance(action, torch.Tensor)
    # _get_action_chunk asserts action_tensor.shape[1] >= action_chunk_length, but
    # the full horizon is returned; pin both dims explicitly.
    assert action.shape == (NUM_ENVS, ACTION_HORIZON, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_reset_propagates_to_client(policy_config_yaml, fake_client_factory):
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    assert clients[0].reset_called is False
    policy.reset()
    assert clients[0].reset_called is True


def test_construction_fails_when_server_unreachable(policy_config_yaml, fake_client_factory):
    fake_client_factory(ping_ok=False)
    with pytest.raises(ConnectionError):
        _build_policy(policy_config_yaml)


# ---------------- scheduler-switch tests ---------------- #


@pytest.mark.parametrize("scheduler_cls_name", ["ActionChunkScheduler", "SyncedBatchActionScheduler"])
def test_get_action_returns_correct_shape_for_each_scheduler(
    policy_config_yaml, synthetic_observation, fake_client_factory, scheduler_cls_name
):
    """Both ActionChunkScheduler and SyncedBatchActionScheduler should drive the policy
    end-to-end and produce one (num_envs, action_dim) action per get_action call."""
    from isaaclab_arena.policy import action_scheduling

    scheduler_cls = getattr(action_scheduling, scheduler_cls_name)

    fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, action_scheduler_cls=scheduler_cls)
    assert isinstance(policy._chunking_state, scheduler_cls)
    policy.set_task_description("pick up the brown box")

    action = policy.get_action(env=None, observation=synthetic_observation)

    assert isinstance(action, torch.Tensor)
    assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_synced_batch_holds_joint_position_for_env_after_partial_reset(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """After resetting a single env, SyncedBatchActionScheduler should hold that env on
    the joint-position-derived hold_action while the others continue stepping their chunk."""
    from isaaclab_arena.policy.action_scheduling import SyncedBatchActionScheduler

    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, action_scheduler_cls=SyncedBatchActionScheduler)
    policy.set_task_description("pick up the brown box")

    # Step 1: every env needs a chunk → exactly one fetch.
    policy.get_action(env=None, observation=synthetic_observation)
    assert clients[0].get_action_calls == 1

    # Reset env 1 only; env 0 still has its chunk.
    policy.reset(env_ids=torch.tensor([1]))

    action = policy.get_action(env=None, observation=synthetic_observation)
    # No new chunk fetched (env 0 not yet exhausted, so .all() is False).
    assert clients[0].get_action_calls == 1
    expected_hold = policy._extract_hold_action(synthetic_observation)
    torch.testing.assert_close(action[1], expected_hold[1])


def test_async_edf_bootstraps_each_environment_with_b1_requests(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, scheduler_mode="async_edf")
    policy.set_task_description("pick up the brown box")

    action = policy.get_action(env=None, observation=synthetic_observation)
    policy.close()

    assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
    assert len(clients) == 1
    assert clients[0].get_action_calls == NUM_ENVS
    assert all(obs["video"]["ego_view"].shape[0] == 1 for obs in clients[0].observations)


def test_async_edf_prefetches_individual_environment_requests(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, scheduler_mode="async_edf")
    policy.set_task_description("pick up the brown box")

    for _ in range(26):
        action = policy.get_action(env=None, observation=synthetic_observation)
        assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)

    results = policy._async_worker.wait_for_results(NUM_ENVS, timeout_s=2.0)
    policy._process_async_results(results)
    policy.close()

    assert clients[0].get_action_calls == NUM_ENVS * 2
    assert policy._async_scheduler.metrics()["request_count"] == NUM_ENVS


def test_async_edf_writes_control_time_metrics_on_close(
    tmp_path, policy_config_yaml, synthetic_observation, fake_client_factory
):
    fake_client_factory(ping_ok=True)
    metrics_path = tmp_path / "async_metrics.json"
    policy = _build_policy(
        policy_config_yaml,
        scheduler_mode="async_edf",
        async_metrics_path=str(metrics_path),
    )
    policy.set_task_description("pick up the brown box")

    policy.get_action(env=None, observation=synthetic_observation)
    metrics = policy.async_metrics()
    policy.close()

    assert metrics["deadline_window_sim_s"] == 0.5
    assert metrics["num_envs"] == NUM_ENVS
    assert metrics_path.exists()


def test_async_edf_can_execute_time_aligned_tail_of_policy_horizon(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    fake_client_factory(ping_ok=True)
    policy = _build_policy(
        policy_config_yaml,
        scheduler_mode="async_edf",
        async_action_chunk_length=25,
        async_action_start_offset_steps=25,
    )
    policy.set_task_description("pick up the brown box")

    action = policy.get_action(env=None, observation=synthetic_observation)
    metrics = policy._async_scheduler.metrics()
    policy.close()

    assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
    assert policy._async_worker is None
    assert metrics["deadline_window_s"] == pytest.approx(0.5)
    assert metrics["action_start_offset_steps"] == 25


def test_async_edf_time_aligned_derives_tail_chunk_from_lead(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    fake_client_factory(ping_ok=True)
    policy = _build_policy(
        policy_config_yaml,
        scheduler_mode="async_edf",
        async_time_aligned=True,
        async_prefetch_lead_steps=10,
    )
    policy.set_task_description("pick up the brown box")

    action = policy.get_action(env=None, observation=synthetic_observation)
    metrics = policy._async_scheduler.metrics()
    policy.close()

    assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
    assert policy.async_action_chunk_length == 40
    assert metrics["action_start_offset_steps"] == 10
    assert metrics["deadline_window_s"] == pytest.approx(0.2)


def test_async_edf_last_action_hold_replays_previous_action_target(
    policy_config_yaml, synthetic_observation, fake_client_factory, monkeypatch
):
    fake_client_factory(ping_ok=True)
    policy = _build_policy(
        policy_config_yaml,
        scheduler_mode="async_edf",
        async_action_chunk_length=2,
        async_prefetch_lead_steps=1,
        async_hold_mode="last_action",
    )
    policy.set_task_description("pick up the brown box")

    first_action = policy.get_action(env=None, observation=synthetic_observation)
    second_action = policy.get_action(env=None, observation=synthetic_observation)
    monkeypatch.setattr(policy._async_worker, "poll", lambda: [])
    missed_action = policy.get_action(env=None, observation=synthetic_observation)
    current_joint_hold = policy._extract_current_joint_hold_action(synthetic_observation)
    policy.close()

    torch.testing.assert_close(missed_action, second_action)
    assert not torch.allclose(missed_action, current_joint_hold)
    assert not torch.allclose(first_action, second_action)


def test_async_edf_writes_one_trace_frame_per_action_step(
    tmp_path, policy_config_yaml, synthetic_observation, fake_client_factory
):
    fake_client_factory(ping_ok=True)
    trace_path = tmp_path / "async_trace.json"
    policy = _build_policy(
        policy_config_yaml,
        scheduler_mode="async_edf",
        async_trace_path=str(trace_path),
    )
    policy.set_task_description("pick up the brown box")

    policy.get_action(env=None, observation=synthetic_observation)
    policy.get_action(env=None, observation=synthetic_observation)
    policy.close()

    trace = json.loads(trace_path.read_text())
    assert trace["frame_count"] == 2
    assert [frame["step"] for frame in trace["frames"]] == [0, 1]
    assert len(trace["frames"][0]["robots"]) == NUM_ENVS
    assert trace["frames"][0]["sim_time_s"] == pytest.approx(0.02)


def test_remote_lifecycle_routes_runner_shutdown_to_close(
    tmp_path, policy_config_yaml, synthetic_observation, fake_client_factory
):
    fake_client_factory(ping_ok=True)
    metrics_path = tmp_path / "runner_metrics.json"
    policy = _build_policy(
        policy_config_yaml,
        scheduler_mode="async_edf",
        async_metrics_path=str(metrics_path),
    )
    policy.set_task_description("pick up the brown box")
    policy.get_action(env=None, observation=synthetic_observation)

    assert policy.is_remote is True
    policy.shutdown_remote(kill_server=False)

    assert metrics_path.exists()
    assert policy._async_worker is None


def test_remote_parser_defines_runner_kill_server_flag(policy_config_yaml) -> None:
    from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicy

    parser = Gr00tRemoteClosedloopPolicy.add_args_to_parser(argparse.ArgumentParser())
    args = parser.parse_args(["--policy_config_yaml_path", policy_config_yaml])

    assert args.remote_kill_on_exit is False
    assert args.async_trace_path is None


def test_remote_parser_defines_async_time_alignment_flags(policy_config_yaml) -> None:
    from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicy

    parser = Gr00tRemoteClosedloopPolicy.add_args_to_parser(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--policy_config_yaml_path",
            policy_config_yaml,
            "--async_time_aligned",
            "--async_action_chunk_length",
            "25",
            "--async_action_start_offset_steps",
            "25",
            "--async_hold_mode",
            "last_action",
        ]
    )

    assert args.async_time_aligned is True
    assert args.async_action_chunk_length == 25
    assert args.async_action_start_offset_steps == 25
    assert args.async_hold_mode == "last_action"


def test_remote_parser_defaults_to_last_action_hold(policy_config_yaml) -> None:
    from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicy

    parser = Gr00tRemoteClosedloopPolicy.add_args_to_parser(argparse.ArgumentParser())
    args = parser.parse_args(["--policy_config_yaml_path", policy_config_yaml])

    assert args.async_hold_mode == "last_action"
