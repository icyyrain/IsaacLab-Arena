# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rollout collection shared by random, learned, and human policies."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from collections.abc import Callable

from .schema import Transition, validate_episode_boundaries

Policy = Callable[[np.ndarray], np.ndarray]


def collect_transitions(
    env: gym.Env,
    policy: Policy,
    *,
    num_transitions: int,
    seed: int = 0,
) -> list[Transition]:
    """Collect validated transitions, resetting only at episode boundaries."""
    assert num_transitions > 0, "num_transitions must be positive"
    observation, _ = env.reset(seed=seed)
    transitions = []

    for index in range(num_transitions):
        policy_action = np.asarray(policy(observation), dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(policy_action)
        transitions.append(
            Transition.from_env_step(
                observation=observation,
                policy_action=policy_action,
                next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        if terminated or truncated:
            if index + 1 < num_transitions:
                observation, _ = env.reset()
        else:
            observation = next_observation

    validate_episode_boundaries(transitions)
    return transitions
