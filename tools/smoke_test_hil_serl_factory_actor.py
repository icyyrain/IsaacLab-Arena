# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run one upstream SAC action and gradient update against remote Factory."""

from __future__ import annotations

import argparse
import gymnasium as gym
import numpy as np

import jax
import jax.numpy as jnp
from flax.core import frozen_dict

from isaaclab_hil_serl.data import Transition
from isaaclab_hil_serl.learning import make_state_sac_agent
from isaaclab_hil_serl.protocol import RemoteEnv


class _StateDictWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"state": env.observation_space})

    def observation(self, observation):
        return {"state": observation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--expect_intervention_action",
        nargs=6,
        type=float,
        help="Assert that the server overrides the policy action with these six values.",
    )
    args = parser.parse_args()

    env = _StateDictWrapper(RemoteEnv(args.host, args.port))
    try:
        observation, _ = env.reset(seed=args.seed)
        agent = make_state_sac_agent(
            seed=args.seed,
            sample_observation=jax.tree.map(jnp.asarray, observation),
            sample_action=jnp.asarray(env.action_space.sample()),
        )
        action_key = jax.random.PRNGKey(args.seed + 1)
        policy_action = np.asarray(
            jax.device_get(agent.sample_actions(jax.tree.map(jnp.asarray, observation), seed=action_key))
        )
        next_observation, reward, terminated, truncated, info = env.step(policy_action)
        transition = Transition.from_env_step(
            observation=observation["state"],
            policy_action=policy_action,
            next_observation=next_observation["state"],
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        serl_transition = transition.to_serl_dict()

        def repeat(value):
            return np.repeat(np.asarray(value)[None, ...], args.batch_size, axis=0)

        batch = frozen_dict.freeze({
            "observations": {"state": repeat(serl_transition["observations"])},
            "actions": repeat(serl_transition["actions"]),
            "next_observations": {"state": repeat(serl_transition["next_observations"])},
            "rewards": repeat(serl_transition["rewards"]),
            "masks": repeat(serl_transition["masks"]),
            "dones": repeat(serl_transition["dones"]),
        })
        agent, update_info = agent.update(jax.device_put(batch))
        update_info = jax.tree.map(lambda value: np.asarray(jax.device_get(value)), update_info)
        print(f"jax_devices={jax.devices()}")
        print(f"observation_shape={observation['state'].shape}, action={policy_action}")
        print(f"reward={reward}, success={info['success']}, intervention={info['intervention']}")
        print(f"update_metrics={sorted(update_info)}")
        if args.expect_intervention_action is not None:
            expected_action = np.asarray(args.expect_intervention_action, dtype=np.float32)
            assert info["intervention"], "Expected the Factory server to report an intervention"
            np.testing.assert_allclose(info["intervene_action"], expected_action)
            np.testing.assert_allclose(info["executed_action"], expected_action)
            np.testing.assert_allclose(serl_transition["actions"], expected_action)
            print(f"executed_intervention_action={info['executed_action']}")
        for value in jax.tree.leaves(update_info):
            assert np.isfinite(np.asarray(value)).all(), f"Non-finite update metric: {value}"
    finally:
        env.close()


if __name__ == "__main__":
    main()
