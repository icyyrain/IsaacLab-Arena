# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab_hil_serl.data import collect_transitions


class _TwoStepEnv(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.episode_id = -1
        self.step_id = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_id += 1
        self.step_id = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        step_id = self.step_id
        self.step_id += 1
        truncated = self.step_id == 2
        observation = np.array([float(self.step_id)], dtype=np.float32)
        info = {
            "episode_id": self.episode_id,
            "step_id": step_id,
            "success": False,
            "intervention": False,
            "executed_action": action,
        }
        return observation, 0.0, False, truncated, info


def test_collector_resets_only_on_episode_boundary():
    env = _TwoStepEnv()
    transitions = collect_transitions(
        env,
        lambda observation: np.array([observation[0] / 10.0], dtype=np.float32),
        num_transitions=5,
    )

    assert [(item.episode_id, item.step_id) for item in transitions] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
    ]
    assert sum(item.truncated for item in transitions) == 2
