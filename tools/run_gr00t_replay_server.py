#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a GR00T ReplayPolicy server from a LeRobot dataset."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer


def load_modality_config(path: str | Path) -> dict[str, Any]:
    """Load a Python modality config module used by Arena GR00T policies."""

    config_path = Path(path)
    spec = importlib.util.spec_from_file_location("arena_gr00t_modality_config", config_path)
    assert spec is not None and spec.loader is not None, f"Unable to load {config_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "unitree_g1_sim_wbc_config"):
        return module.unitree_g1_sim_wbc_config
    raise AttributeError(f"{config_path} does not define unitree_g1_sim_wbc_config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="LeRobot dataset directory")
    parser.add_argument("--modality-config-path", required=True, help="Arena GR00T Python modality config")
    parser.add_argument("--execution-horizon", type=int, default=50, help="Steps consumed per policy call")
    parser.add_argument("--video-backend", default="decord", help="Video backend for LeRobot loader")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--strict", action="store_true", help="Enable strict ReplayPolicy validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modality_config = load_modality_config(args.modality_config_path)
    policy = ReplayPolicy(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        execution_horizon=args.execution_horizon,
        video_backend=args.video_backend,
        strict=args.strict,
    )
    print(
        "Starting GR00T replay server "
        f"dataset={args.dataset_path} execution_horizon={args.execution_horizon} "
        f"video_backend={args.video_backend} strict={args.strict}",
        flush=True,
    )
    PolicyServer(policy=policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
