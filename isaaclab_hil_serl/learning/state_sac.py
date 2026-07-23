# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""State-only constructor using upstream HIL-SERL's unmodified SAC agent."""

from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP


class StateEncoder(nn.Module):
    """Extract the flat state while retaining SERL's dictionary batch format."""

    @nn.compact
    def __call__(self, observations, train: bool = False, stop_gradient: bool = False):
        del train
        state = observations["state"]
        return jax.lax.stop_gradient(state) if stop_gradient else state


def make_state_sac_agent(
    *,
    seed: int,
    sample_observation,
    sample_action,
    discount: float = 0.97,
) -> SACAgent:
    """Build a state-only SAC agent from upstream HIL-SERL modules."""
    encoder = StateEncoder()
    critic_backbone = partial(
        MLP,
        hidden_dims=[256, 256],
        activations=nn.tanh,
        use_layer_norm=True,
        activate_final=True,
    )
    critic_backbone = ensemblize(critic_backbone, 2)(name="critic_ensemble")
    critic = Critic(encoder=encoder, network=critic_backbone, name="critic")
    policy = Policy(
        encoder=encoder,
        network=MLP(
            hidden_dims=[256, 256],
            activations=nn.tanh,
            use_layer_norm=True,
            activate_final=True,
        ),
        action_dim=sample_action.shape[-1],
        tanh_squash_distribution=True,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5.0,
        name="actor",
    )
    temperature = GeqLagrangeMultiplier(
        init_value=1e-2,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )
    return SACAgent.create(
        jax.random.PRNGKey(seed),
        sample_observation,
        sample_action,
        actor_def=policy,
        critic_def=critic,
        temperature_def=temperature,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        image_keys=("state",),
        augmentation_function=None,
    )
