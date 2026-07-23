# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate that a Factory PPO demo contains visible frames and robot motion."""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

import cv2


def main() -> None:
    """Inspect the recorded MP4 and fail when rendering or motion is missing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Factory PPO MP4 to inspect")
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--require-motion", action="store_true")
    args = parser.parse_args()

    assert args.video.is_file(), f"Video not found: {args.video}"
    capture = cv2.VideoCapture(str(args.video))
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    capture.release()

    assert len(frames) >= args.min_frames, f"Expected at least {args.min_frames} frames, found {len(frames)}"
    visible_frames = sum(float(frame.std()) > 5.0 for frame in frames)
    assert visible_frames >= 0.95 * len(frames), f"Only {visible_frames}/{len(frames)} frames contain visible pixels"

    reference = frames[0].astype(np.int16)
    mean_differences = []
    changed_fractions = []
    for frame in frames[1:]:
        difference = np.abs(frame.astype(np.int16) - reference)
        mean_differences.append(float(difference.mean()))
        changed_fractions.append(float((difference > 20).mean()))

    max_mean_difference = max(mean_differences, default=0.0)
    max_changed_fraction = max(changed_fractions, default=0.0)
    if args.require_motion:
        assert (
            max_mean_difference > 4.0
        ), f"Video appears static (maximum mean frame difference {max_mean_difference:.2f})"
        assert max_changed_fraction > 0.02, f"Video appears static (changed-pixel fraction {max_changed_fraction:.3f})"

    print(
        f"Validated {len(frames)} visible frames; max mean difference {max_mean_difference:.2f}; "
        f"changed-pixel fraction {max_changed_fraction:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
