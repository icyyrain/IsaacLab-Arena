# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Host Factory peg insertion on Windows for an upstream HIL-SERL actor."""

from __future__ import annotations

import argparse
import contextlib
import gymnasium as gym
import sys

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

from isaaclab_hil_serl.envs import FactoryStateEnvAdapter, OneShotInterventionProvider
from isaaclab_hil_serl.protocol import EnvironmentRpcServer
from isaaclab_hil_serl.teleop import SpaceMouseInterventionProvider

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

parser = argparse.ArgumentParser(description="Run a Windows-native Factory environment RPC server.")
parser.add_argument("--task", default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=18765)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--reward_mode", choices=["sparse_success", "factory"], default="sparse_success")
parser.add_argument("--no_terminate_on_success", action="store_true")
parser.add_argument(
    "--scripted_intervention_action",
    nargs=6,
    type=float,
    metavar=("DX", "DY", "DZ", "DROLL", "DPITCH", "DYAW"),
    help="Override the first policy action of every episode; values must be in [-1, 1].",
)
parser.add_argument(
    "--intervention_device",
    choices=["none", "spacemouse"],
    default="none",
    help="Human input device that overrides policy actions while it is moving.",
)
parser.add_argument("--spacemouse_deadzone", type=float, default=0.05)
parser.add_argument("--spacemouse_translation_scale", type=float, default=1.0)
parser.add_argument("--spacemouse_rotation_scale", type=float, default=1.0)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        intervention_provider = None
        assert not (
            args_cli.scripted_intervention_action is not None and args_cli.intervention_device != "none"
        ), "Choose either scripted intervention or a human intervention device"
        if args_cli.scripted_intervention_action is not None:
            intervention_provider = OneShotInterventionProvider(args_cli.scripted_intervention_action)
        elif args_cli.intervention_device == "spacemouse":
            intervention_provider = SpaceMouseInterventionProvider.create_isaaclab(
                deadzone=args_cli.spacemouse_deadzone,
                translation_scale=args_cli.spacemouse_translation_scale,
                rotation_scale=args_cli.spacemouse_rotation_scale,
            )
        adapter = FactoryStateEnvAdapter(
            env,
            reward_mode=args_cli.reward_mode,
            terminate_on_success=not args_cli.no_terminate_on_success,
            intervention_provider=intervention_provider,
        )
        server = EnvironmentRpcServer(adapter, host=args_cli.host, port=args_cli.port)
        print(
            f"[INFO] Factory HIL-SERL RPC listening on {server.address[0]}:{server.address[1]} "
            f"(state_dim={adapter.observation_space.shape[0]}, action_dim={adapter.action_space.shape[0]})",
            flush=True,
        )
        if args_cli.host not in ("127.0.0.1", "localhost"):
            print("[WARN] RPC has no authentication; only expose this port to trusted local networks.", flush=True)
        try:
            with contextlib.suppress(KeyboardInterrupt):
                server.serve_forever()
        finally:
            server.stop()


if __name__ == "__main__":
    main()
