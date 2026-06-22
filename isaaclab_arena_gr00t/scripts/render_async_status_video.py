# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Overlay asynchronous VLA trace state onto an Arena rollout video."""

from __future__ import annotations

import argparse
import json

from isaaclab_arena_gr00t.utils.async_video_overlay import render_status_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--timeline", required=True)
    args = parser.parse_args()

    summary = render_status_video(args.input_video, args.trace, args.output_video, args.timeline)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
