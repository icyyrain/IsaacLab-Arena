# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect and verify a state-transition dataset from remote Factory."""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from isaaclab_hil_serl.data import collect_transitions, load_transition_dataset, save_transition_dataset
from isaaclab_hil_serl.protocol import RemoteEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_transitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = RemoteEnv(args.host, args.port)
    env.action_space.seed(args.seed)
    try:
        transitions = collect_transitions(
            env,
            lambda observation: env.action_space.sample(),
            num_transitions=args.num_transitions,
            seed=args.seed,
        )
        output_path = save_transition_dataset(args.output, transitions)
        loaded = load_transition_dataset(output_path)
        assert len(loaded) == len(transitions)
        interventions = sum(transition.intervention for transition in loaded)
        successes = sum(transition.success for transition in loaded)
        episodes = len({transition.episode_id for transition in loaded})
        print(f"dataset={output_path}")
        print(f"transitions={len(loaded)}, episodes={episodes}, interventions={interventions}, successes={successes}")
        print(f"observation_shape={loaded[0].observation.shape}, action_shape={loaded[0].executed_action.shape}")
        assert np.isfinite(np.stack([transition.observation for transition in loaded])).all()
    finally:
        env.close()


if __name__ == "__main__":
    main()
