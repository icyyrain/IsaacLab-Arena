# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Connect to a HIL-SERL environment RPC server and execute one step."""

from __future__ import annotations

import argparse

from isaaclab_hil_serl.protocol import RemoteEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    env = RemoteEnv(args.host, args.port)
    try:
        observation, reset_info = env.reset(seed=11)
        assert env.observation_space.contains(observation), "Reset observation is outside the declared space"

        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, step_info = env.step(action)
        assert env.observation_space.contains(next_observation), "Step observation is outside the declared space"
        print(f"reset_info={reset_info}")
        print(f"reward={reward}, terminated={terminated}, truncated={truncated}")
        print(f"step_info_keys={sorted(step_info)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
