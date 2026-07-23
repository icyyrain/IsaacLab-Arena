# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run separate upstream Agentlace actor and learner processes."""

from __future__ import annotations

import argparse
import gymnasium as gym
import json
import numpy as np
import subprocess
import sys
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer
from serl_launcher.data.data_store import ReplayBufferDataStore

from isaaclab_hil_serl.data import Transition, save_transition_dataset
from isaaclab_hil_serl.learning import make_state_sac_agent
from isaaclab_hil_serl.protocol import RemoteEnv

STATE_DIM = 43
ACTION_DIM = 6


def _spaces() -> tuple[gym.spaces.Dict, gym.spaces.Box]:
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(STATE_DIM,),
                dtype=np.float32,
            )
        }
    )
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(ACTION_DIM,),
        dtype=np.float32,
    )
    return observation_space, action_space


def _agent(seed: int):
    return make_state_sac_agent(
        seed=seed,
        sample_observation={"state": jnp.zeros(STATE_DIM, dtype=jnp.float32)},
        sample_action=jnp.zeros(ACTION_DIM, dtype=jnp.float32),
    )


def _trainer_config(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        port_number=args.agentlace_port,
        broadcast_port=args.broadcast_port,
        request_types=[],
        version="isaaclab-hil-serl-m1-distributed",
    )


def _parameter_fingerprint(params) -> float:
    total = 0.0
    for leaf in jax.tree.leaves(params):
        total += float(np.asarray(jax.device_get(leaf), dtype=np.float64).sum())
    return round(total, 8)


def run_learner(args: argparse.Namespace) -> None:
    observation_space, action_space = _spaces()
    replay_buffer = ReplayBufferDataStore(observation_space, action_space, capacity=4096)
    server = TrainerServer(_trainer_config(args))
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)
    agent = _agent(args.seed)
    deadline = time.monotonic() + args.timeout

    try:
        while len(replay_buffer) < args.transitions:
            assert (
                time.monotonic() < deadline
            ), f"Timed out with {len(replay_buffer)}/{args.transitions} replay transitions"
            server.publish_network(agent.state.params)
            time.sleep(0.25)

        metric_names = []
        for _ in range(args.updates):
            batch = jax.device_put(replay_buffer.sample(args.batch_size))
            agent, update_info = agent.update(batch)
            for value in jax.tree.leaves(update_info):
                assert np.isfinite(np.asarray(jax.device_get(value))).all()
            metric_names = sorted(update_info)
            server.publish_network(agent.state.params)
            time.sleep(0.25)

        time.sleep(1.0)
        result = {
            "replay_transitions": len(replay_buffer),
            "updates": args.updates,
            "metrics": metric_names,
            "parameter_fingerprint": _parameter_fingerprint(agent.state.params),
        }
        _write_json(args.output_dir / "learner-result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        server.stop()


def run_actor(args: argparse.Namespace) -> None:
    actor_queue = QueuedDataStore(capacity=4096)
    client = TrainerClient(
        "actor_env",
        args.learner_host,
        _trainer_config(args),
        data_stores={"actor_env": actor_queue},
        wait_for_server=True,
        timeout_ms=3000,
    )
    env = RemoteEnv(args.factory_host, args.factory_port)
    agent = _agent(args.seed)
    agent_lock = threading.Lock()
    network_event = threading.Event()
    fingerprints: set[float] = set()

    def receive_network(params) -> None:
        nonlocal agent
        with agent_lock:
            agent = agent.replace(state=agent.state.replace(params=params))
            fingerprints.add(_parameter_fingerprint(params))
        network_event.set()

    client.recv_network_callback(receive_network)
    transitions = []
    rng = jax.random.PRNGKey(args.seed + 1)

    try:
        assert network_event.wait(timeout=args.timeout), "Actor did not receive initial learner parameters"
        observation, _ = env.reset(seed=args.seed)
        for _ in range(args.transitions):
            rng, action_key = jax.random.split(rng)
            with agent_lock:
                actor_agent = agent
            policy_action = np.asarray(
                jax.device_get(
                    actor_agent.sample_actions(
                        {"state": jnp.asarray(observation)},
                        seed=action_key,
                    )
                )
            )
            next_observation, reward, terminated, truncated, info = env.step(policy_action)
            transition = Transition.from_env_step(
                observation=observation,
                policy_action=policy_action,
                next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            transitions.append(transition)
            serl_transition = transition.to_serl_dict()
            serl_transition["observations"] = {"state": serl_transition["observations"]}
            serl_transition["next_observations"] = {"state": serl_transition["next_observations"]}
            actor_queue.insert(serl_transition)
            assert client.update(), "Actor failed to upload a transition"
            if terminated or truncated:
                observation, _ = env.reset()
            else:
                observation = next_observation

        deadline = time.monotonic() + args.timeout
        while len(fingerprints) < 2:
            assert time.monotonic() < deadline, "Actor did not receive updated learner parameters"
            time.sleep(0.1)

        dataset_path = save_transition_dataset(
            args.output_dir / "distributed-factory-transitions.npz",
            transitions,
        )
        result = {
            "transitions": len(transitions),
            "interventions": sum(item.intervention for item in transitions),
            "unique_parameter_broadcasts": len(fingerprints),
            "dataset": str(dataset_path),
        }
        _write_json(args.output_dir / "actor-result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.close()
        client.stop()


def run_orchestrator(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    common = [
        "--factory_host",
        args.factory_host,
        "--factory_port",
        str(args.factory_port),
        "--learner_host",
        "127.0.0.1",
        "--agentlace_port",
        str(args.agentlace_port),
        "--broadcast_port",
        str(args.broadcast_port),
        "--transitions",
        str(args.transitions),
        "--updates",
        str(args.updates),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--timeout",
        str(args.timeout),
        "--output_dir",
        str(args.output_dir),
    ]
    learner = subprocess.Popen(
        [sys.executable, str(script), "learner", *common],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(1.0)
    actor = subprocess.Popen(
        [sys.executable, str(script), "actor", *common],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        actor_output, _ = actor.communicate(timeout=args.timeout + 60.0)
        learner_output, _ = learner.communicate(timeout=args.timeout + 60.0)
    except subprocess.TimeoutExpired:
        actor.kill()
        learner.kill()
        raise

    print("[actor]", actor_output, sep="\n", end="")
    print("[learner]", learner_output, sep="\n", end="")
    assert actor.returncode == 0, f"Actor exited with code {actor.returncode}"
    assert learner.returncode == 0, f"Learner exited with code {learner.returncode}"

    actor_result = json.loads((args.output_dir / "actor-result.json").read_text())
    learner_result = json.loads((args.output_dir / "learner-result.json").read_text())
    assert actor_result["transitions"] == args.transitions
    assert actor_result["unique_parameter_broadcasts"] >= 2
    assert learner_result["replay_transitions"] == args.transitions
    assert learner_result["updates"] == args.updates
    print(
        f"distributed_smoke=passed, transitions={args.transitions}, "
        f"updates={args.updates}, broadcasts={actor_result['unique_parameter_broadcasts']}"
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["orchestrator", "actor", "learner"])
    parser.add_argument("--factory_host", default="127.0.0.1")
    parser.add_argument("--factory_port", type=int, default=18765)
    parser.add_argument("--learner_host", default="127.0.0.1")
    parser.add_argument("--agentlace_port", type=int, default=16588)
    parser.add_argument("--broadcast_port", type=int, default=16589)
    parser.add_argument("--transitions", type=int, default=8)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.role == "learner":
        run_learner(cli_args)
    elif cli_args.role == "actor":
        run_actor(cli_args)
    else:
        run_orchestrator(cli_args)
