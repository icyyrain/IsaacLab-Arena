# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for policy-runner mosaic video CLI arguments."""

from __future__ import annotations

import argparse

import pytest

from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments


def test_policy_runner_mosaic_cli_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_policy_runner_arguments(parser)

    args = parser.parse_args(["--policy_type", "example.Policy"])

    assert args.mosaic_video is False
    assert args.mosaic_camera_width == 480
    assert args.mosaic_camera_height == 360
    assert args.mosaic_columns == 3
    assert args.mosaic_camera_mode == "planar"
    assert args.mosaic_camera_eye == [-2.2, -2.2, 1.7]
    assert args.mosaic_camera_target == [0.0, 0.0, 0.6]
    assert args.mosaic_camera_follow_tau == 0.25
    assert args.mosaic_camera_follow_deadband == 0.02


def test_policy_runner_rejects_unknown_mosaic_camera_mode() -> None:
    parser = argparse.ArgumentParser()
    add_policy_runner_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--policy_type", "example.Policy", "--mosaic_camera_mode", "orbit"])
