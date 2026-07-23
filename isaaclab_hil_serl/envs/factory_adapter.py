# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-environment, state-based Gym adapter for Factory peg insertion."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Callable
from typing import Any, Literal

FACTORY_STATE_FIELDS = (
    ("fingertip_pos", 3),
    ("fingertip_quat", 4),
    ("ee_linvel", 3),
    ("ee_angvel", 3),
    ("joint_pos", 7),
    ("held_pos", 3),
    ("held_pos_rel_fixed", 3),
    ("held_quat", 4),
    ("fixed_pos", 3),
    ("fixed_quat", 4),
    ("prev_actions", 6),
)
"""Stable layout of the 43-dimensional state sent to HIL-SERL."""

FACTORY_STATE_DIM = sum(width for _, width in FACTORY_STATE_FIELDS)
FACTORY_ACTION_DIM = 6

InterventionProvider = Callable[[np.ndarray, np.ndarray], np.ndarray | None]


class OneShotInterventionProvider:
    """Override the first action of every episode with a configured action."""

    source = "scripted"

    def __init__(self, action: np.ndarray):
        self._action = np.asarray(action, dtype=np.float32).copy()
        assert self._action.shape == (
            FACTORY_ACTION_DIM,
        ), f"Intervention action must have shape {(FACTORY_ACTION_DIM,)}, received {self._action.shape}"
        assert np.isfinite(self._action).all(), "Intervention action contains NaN or Inf"
        assert np.all(np.abs(self._action) <= 1.0), f"Intervention action is outside [-1, 1]: {self._action}"
        self._used = False

    def reset(self) -> None:
        """Arm the intervention for a new episode."""
        self._used = False

    def __call__(self, observation: np.ndarray, policy_action: np.ndarray) -> np.ndarray | None:
        del observation, policy_action
        if self._used:
            return None
        self._used = True
        return self._action.copy()


class FactoryStateEnvAdapter(gym.Env):
    """Convert Isaac Lab's one-environment vector API into a standard Gym API.

    The action is a normalized six-dimensional end-effector delta:
    ``[dx, dy, dz, droll, dpitch, dyaw]`` in ``[-1, 1]``. Factory applies its
    own translation/rotation thresholds and EMA after this adapter.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_mode: Literal["sparse_success", "factory"] = "sparse_success",
        terminate_on_success: bool = True,
        intervention_provider: InterventionProvider | None = None,
    ):
        self._env = env
        self._base_env = env.unwrapped
        assert self._base_env.num_envs == 1, "HIL-SERL Factory adapter currently requires num_envs=1"
        assert reward_mode in ("sparse_success", "factory"), f"Unsupported reward mode: {reward_mode}"
        assert tuple(self._base_env.cfg.state_order) == tuple(
            name for name, _ in FACTORY_STATE_FIELDS[:-1]
        ), "Factory state_order changed; update the HIL-SERL wire contract before running"

        self.reward_mode = reward_mode
        self.terminate_on_success = terminate_on_success
        self.intervention_provider = intervention_provider
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(FACTORY_STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(FACTORY_ACTION_DIM,),
            dtype=np.float32,
        )
        self._last_observation: np.ndarray | None = None
        self._episode_id = -1
        self._step_id = 0

    @property
    def state_fields(self) -> tuple[tuple[str, int], ...]:
        return FACTORY_STATE_FIELDS

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, _ = self._env.reset(seed=seed, options=options)
        state = self._extract_state(observation)
        self._last_observation = state
        self._episode_id += 1
        self._step_id = 0
        reset_intervention = getattr(self.intervention_provider, "reset", None)
        if reset_intervention is not None:
            reset_intervention()
        return state, {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "state_fields": self.state_fields,
        }

    def step(self, policy_action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._last_observation is not None, "Call reset() before step()"
        policy_action = self._validate_action(policy_action, name="policy_action")
        intervention_action = None
        if self.intervention_provider is not None:
            intervention_action = self.intervention_provider(self._last_observation.copy(), policy_action.copy())
        if intervention_action is None:
            executed_action = policy_action
        else:
            intervention_action = self._validate_action(intervention_action, name="intervention_action")
            executed_action = intervention_action

        action_tensor = torch.as_tensor(
            executed_action,
            dtype=torch.float32,
            device=self._base_env.device,
        ).unsqueeze(0)
        observation, factory_reward, raw_terminated, raw_truncated, _ = self._env.step(action_tensor)
        state = self._extract_state(observation)
        success = self._get_success()
        factory_reward_value = self._first_scalar(factory_reward, float)
        raw_terminated_value = self._first_scalar(raw_terminated, bool)
        raw_truncated_value = self._first_scalar(raw_truncated, bool)

        # Factory reports timeout as both terminated and truncated. Normalize it
        # to Gymnasium's bootstrap-safe convention, and use success as terminal.
        timed_out = raw_truncated_value or raw_terminated_value
        terminated = bool(success and self.terminate_on_success)
        truncated = bool(timed_out and not terminated)
        reward = float(success) if self.reward_mode == "sparse_success" else factory_reward_value

        transition_step_id = self._step_id
        self._step_id += 1
        info = {
            "episode_id": self._episode_id,
            "step_id": transition_step_id,
            "success": success,
            "intervention": intervention_action is not None,
            "policy_action": policy_action,
            "executed_action": executed_action,
            "factory_reward": factory_reward_value,
        }
        if intervention_action is not None:
            # Upstream train_rlpd.py uses the presence of this key to replace
            # the proposed policy action in the stored Bellman transition.
            info["intervene_action"] = intervention_action
            info["intervention_source"] = getattr(self.intervention_provider, "source", "external")

        self._last_observation = state
        return state, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            close_intervention = getattr(self.intervention_provider, "close", None)
            if close_intervention is not None:
                close_intervention()
        finally:
            self._env.close()

    def _extract_state(self, observation: dict[str, torch.Tensor]) -> np.ndarray:
        assert "critic" in observation, f"Factory observation has no critic state: {tuple(observation)}"
        state = observation["critic"][0].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
        assert state.shape == (
            FACTORY_STATE_DIM,
        ), f"Expected Factory state shape {(FACTORY_STATE_DIM,)}, received {state.shape}"
        assert np.isfinite(state).all(), "Factory state contains NaN or Inf"
        return state

    def _get_success(self) -> bool:
        check_rotation = self._base_env.cfg_task.name == "nut_thread"
        successes = self._base_env._get_curr_successes(
            success_threshold=self._base_env.cfg_task.success_threshold,
            check_rot=check_rotation,
        )
        return bool(successes[0].item())

    def _validate_action(self, action: Any, *, name: str) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        assert value.shape == (
            FACTORY_ACTION_DIM,
        ), f"{name} must have shape {(FACTORY_ACTION_DIM,)}, received {value.shape}"
        assert np.isfinite(value).all(), f"{name} contains NaN or Inf"
        assert self.action_space.contains(value), f"{name} is outside [-1, 1]: {value}"
        return value.copy()

    @staticmethod
    def _first_scalar(value: Any, scalar_type: type) -> Any:
        if isinstance(value, (torch.Tensor, np.ndarray)):
            value = value.reshape(-1)[0].item()
        return scalar_type(value)
