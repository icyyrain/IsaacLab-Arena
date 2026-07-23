# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import types

from isaaclab_hil_serl.envs.factory_adapter import (
    FACTORY_ACTION_DIM,
    FACTORY_STATE_DIM,
    FACTORY_STATE_FIELDS,
    FactoryStateEnvAdapter,
    OneShotInterventionProvider,
)


class _FakeFactoryEnv:
    def __init__(self):
        state_order = [name for name, _ in FACTORY_STATE_FIELDS[:-1]]
        self.unwrapped = types.SimpleNamespace(
            num_envs=1,
            device="cpu",
            cfg=types.SimpleNamespace(state_order=state_order),
            cfg_task=types.SimpleNamespace(name="peg_insert", success_threshold=0.9),
            _get_curr_successes=lambda success_threshold, check_rot: torch.tensor([self.success]),
        )
        self.state = torch.arange(FACTORY_STATE_DIM, dtype=torch.float32).unsqueeze(0)
        self.success = False
        self.closed = False
        self.last_action = None
        self.timeout = False

    def reset(self, seed=None, options=None):
        return {"critic": self.state.clone()}, {}

    def step(self, action):
        self.last_action = action.clone()
        self.state += 1.0
        done = torch.tensor([self.timeout])
        return {"critic": self.state.clone()}, torch.tensor([3.5]), done, done, {}

    def close(self):
        self.closed = True


def test_factory_state_contract_and_sparse_reward():
    env = _FakeFactoryEnv()
    adapter = FactoryStateEnvAdapter(env)

    observation, info = adapter.reset(seed=5)
    assert observation.shape == (FACTORY_STATE_DIM,)
    assert observation.dtype == np.float32
    assert info["episode_id"] == 0
    assert tuple(info["state_fields"]) == FACTORY_STATE_FIELDS

    action = np.zeros(FACTORY_ACTION_DIM, dtype=np.float32)
    next_observation, reward, terminated, truncated, info = adapter.step(action)

    assert next_observation.shape == (FACTORY_STATE_DIM,)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["factory_reward"] == 3.5
    assert not info["intervention"]
    np.testing.assert_array_equal(env.last_action.numpy()[0], action)


def test_intervention_is_executed_and_success_terminates():
    env = _FakeFactoryEnv()
    env.success = True
    intervention_action = np.full(FACTORY_ACTION_DIM, -0.25, dtype=np.float32)
    adapter = FactoryStateEnvAdapter(
        env,
        intervention_provider=lambda observation, policy_action: intervention_action,
    )
    adapter.reset()

    _, reward, terminated, truncated, info = adapter.step(np.zeros(FACTORY_ACTION_DIM, dtype=np.float32))

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["intervention"]
    np.testing.assert_array_equal(info["intervene_action"], intervention_action)
    np.testing.assert_array_equal(info["executed_action"], intervention_action)
    np.testing.assert_array_equal(env.last_action.numpy()[0], intervention_action)


def test_factory_timeout_is_only_truncated():
    env = _FakeFactoryEnv()
    env.timeout = True
    adapter = FactoryStateEnvAdapter(env)
    adapter.reset()

    _, _, terminated, truncated, _ = adapter.step(np.zeros(FACTORY_ACTION_DIM, dtype=np.float32))

    assert not terminated
    assert truncated


def test_one_shot_intervention_rearms_on_reset():
    env = _FakeFactoryEnv()
    intervention_action = np.full(FACTORY_ACTION_DIM, 0.5, dtype=np.float32)
    provider = OneShotInterventionProvider(intervention_action)
    adapter = FactoryStateEnvAdapter(env, intervention_provider=provider)
    policy_action = np.zeros(FACTORY_ACTION_DIM, dtype=np.float32)

    adapter.reset()
    *_, first_info = adapter.step(policy_action)
    *_, second_info = adapter.step(policy_action)
    adapter.reset()
    *_, reset_info = adapter.step(policy_action)

    assert first_info["intervention"]
    assert not second_info["intervention"]
    assert reset_info["intervention"]
    np.testing.assert_array_equal(first_info["executed_action"], intervention_action)
    np.testing.assert_array_equal(second_info["executed_action"], policy_action)
