# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise upstream Agentlace replay upload and parameter broadcast."""

from __future__ import annotations

import argparse
import gymnasium as gym
import numpy as np
import threading
import time

import jax
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer
from serl_launcher.data.data_store import ReplayBufferDataStore

from isaaclab_hil_serl.data import load_transition_dataset
from isaaclab_hil_serl.learning import make_state_sac_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--port", type=int, default=15588)
    parser.add_argument("--broadcast_port", type=int, default=15589)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    transitions = load_transition_dataset(args.dataset)
    sample = transitions[0]
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=sample.observation.shape,
                dtype=np.float32,
            )
        }
    )
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=sample.executed_action.shape,
        dtype=np.float32,
    )
    replay_buffer = ReplayBufferDataStore(observation_space, action_space, capacity=1024)
    actor_queue = QueuedDataStore(capacity=1024)
    config = TrainerConfig(
        port_number=args.port,
        broadcast_port=args.broadcast_port,
        request_types=[],
        version="isaaclab-hil-serl-m1",
    )
    server = TrainerServer(config)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)
    client = None

    try:
        client = TrainerClient(
            "actor_env",
            "127.0.0.1",
            config,
            data_stores={"actor_env": actor_queue},
            wait_for_server=True,
            timeout_ms=3000,
        )
        received_params = []
        received_event = threading.Event()

        def receive_network(params):
            received_params.append(params)
            received_event.set()

        client.recv_network_callback(receive_network)
        time.sleep(0.25)

        for transition in transitions:
            serl_transition = transition.to_serl_dict()
            serl_transition["observations"] = {"state": serl_transition["observations"]}
            serl_transition["next_observations"] = {"state": serl_transition["next_observations"]}
            actor_queue.insert(serl_transition)
        assert client.update(), "Agentlace actor failed to upload transitions"
        assert len(replay_buffer) == len(
            transitions
        ), f"Learner replay received {len(replay_buffer)} of {len(transitions)} transitions"

        agent = make_state_sac_agent(
            seed=args.seed,
            sample_observation={"state": jax.numpy.asarray(sample.observation)},
            sample_action=jax.numpy.asarray(sample.executed_action),
        )
        batch = jax.device_put(replay_buffer.sample(args.batch_size))
        agent, update_info = agent.update(batch)
        for value in jax.tree.leaves(update_info):
            assert np.isfinite(np.asarray(jax.device_get(value))).all()

        server.publish_network(agent.state.params)
        assert received_event.wait(timeout=5.0), "Actor did not receive learner parameter broadcast"
        assert received_params, "Agentlace callback did not receive parameters"
        assert jax.tree.structure(received_params[-1]) == jax.tree.structure(agent.state.params)

        print(f"jax_devices={jax.devices()}")
        print(f"uploaded_transitions={len(replay_buffer)}")
        print(f"update_metrics={sorted(update_info)}")
        print("parameter_broadcast=received")
    finally:
        if client is not None:
            client.stop()
        server.stop()


if __name__ == "__main__":
    main()
