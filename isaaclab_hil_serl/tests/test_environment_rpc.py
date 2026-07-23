# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_hil_serl.protocol import EnvironmentRpcServer, RemoteEnv
from isaaclab_hil_serl.protocol.codec import space_from_spec, space_to_spec


class _InterventionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state.fill(0.0)
        return {"state": self.state.copy()}, {"seed": seed}

    def step(self, policy_action):
        policy_action = np.asarray(policy_action, dtype=np.float32)
        intervention_action = np.array([-0.25, 0.5], dtype=np.float32)
        self.state += intervention_action
        info = {
            "intervene_action": intervention_action,
            "policy_action_seen_by_server": policy_action,
        }
        return {"state": self.state.copy()}, 1.0, False, False, info

    def close(self):
        self.closed = True


def test_space_spec_round_trip_preserves_nested_box():
    original = gym.spaces.Dict({
        "state": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        "mode": gym.spaces.Discrete(3, start=1),
    })

    restored = space_from_spec(space_to_spec(original))

    assert isinstance(restored, gym.spaces.Dict)
    assert restored["state"].shape == (7,)
    assert restored["state"].dtype == np.float32
    assert restored["mode"].n == 3
    assert restored["mode"].start == 1


def test_remote_env_round_trip_and_intervention_action():
    backend = _InterventionEnv()
    server = EnvironmentRpcServer(backend)
    server.start_in_thread()
    client = RemoteEnv(*server.address)

    try:
        observation, info = client.reset(seed=7)
        np.testing.assert_array_equal(observation["state"], np.zeros(2, dtype=np.float32))
        assert info == {"seed": 7}
        assert client.observation_space.contains(observation)

        policy_action = np.array([0.75, -0.5], dtype=np.float32)
        next_observation, reward, terminated, truncated, info = client.step(policy_action)

        np.testing.assert_allclose(next_observation["state"], [-0.25, 0.5])
        np.testing.assert_array_equal(info["policy_action_seen_by_server"], policy_action)
        np.testing.assert_allclose(info["intervene_action"], [-0.25, 0.5])
        assert info["intervene_action"].dtype == np.float32
        assert reward == 1.0
        assert not terminated
        assert not truncated
        assert client.ping()["protocol_version"] == 1
    finally:
        client.close()
        server.stop()

    assert backend.closed


def test_remote_error_is_reported_to_client():
    backend = _InterventionEnv()
    server = EnvironmentRpcServer(backend)
    server.start_in_thread()
    client = RemoteEnv(*server.address)

    try:
        try:
            client._rpc("does-not-exist")
        except RuntimeError as error:
            assert "Unknown RPC method" in str(error)
        else:
            raise AssertionError("Expected the remote method to fail")
    finally:
        client.close()
        server.stop()
