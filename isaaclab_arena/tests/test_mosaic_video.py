# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for batched third-person camera mosaic recording."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import cv2
import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab_arena.evaluation.mosaic_video import (
    TiledCameraMosaicRecorder,
    _as_torch_tensor,
    _smooth_planar_xy,
    _write_mp4,
    tile_camera_batch,
)


def test_tile_camera_batch_preserves_row_major_environment_order() -> None:
    frames = np.stack([np.full((4, 5, 3), env_id + 1, dtype=np.uint8) for env_id in range(6)])

    mosaic = tile_camera_batch(frames, columns=3, draw_labels=False)

    assert mosaic.shape == (8, 15, 3)
    for env_id in range(6):
        row, column = divmod(env_id, 3)
        np.testing.assert_array_equal(mosaic[row * 4 : (row + 1) * 4, column * 5 : (column + 1) * 5], frames[env_id])


def test_tile_camera_batch_converts_float_rgba_and_pads_unused_tiles() -> None:
    frames = np.zeros((5, 6, 8, 4), dtype=np.float32)
    for env_id in range(5):
        frames[env_id, ..., :3] = (env_id + 1) / 10.0
        frames[env_id, ..., 3] = 1.0

    mosaic = tile_camera_batch(frames, columns=3, draw_labels=False)

    assert mosaic.dtype == np.uint8
    assert mosaic.shape == (12, 24, 3)
    np.testing.assert_array_equal(mosaic[6:12, 16:24], np.zeros((6, 8, 3), dtype=np.uint8))
    assert tuple(mosaic[1, 1]) == (25, 25, 25)


def test_tile_camera_batch_draws_robot_labels() -> None:
    frames = np.full((2, 60, 80, 3), 127, dtype=np.uint8)

    mosaic = tile_camera_batch(frames, columns=2, draw_labels=True)

    assert np.any(mosaic[0:28, 0:58] != 127)
    assert np.any(mosaic[0:28, 80:138] != 127)


def test_write_mp4_preserves_exact_frame_count_at_30_seconds(tmp_path) -> None:
    path = tmp_path / "exact.mp4"
    frames = [np.full((16, 16, 3), frame_id % 256, dtype=np.uint8) for frame_id in range(1500)]

    _write_mp4(path, frames, fps=50)

    capture = cv2.VideoCapture(str(path))
    decoded_frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded_frames += 1
    capture.release()
    assert decoded_frames == 1500


def test_smooth_planar_xy_holds_inside_deadband() -> None:
    current = torch.tensor([[1.0, 2.0]])
    desired = torch.tensor([[1.01, 2.0]])

    actual = _smooth_planar_xy(current, desired, dt=0.02, tau=0.25, deadband=0.02)

    torch.testing.assert_close(actual, current)


def test_smooth_planar_xy_uses_exponential_filter_outside_deadband() -> None:
    current = torch.zeros((1, 2))
    desired = torch.tensor([[1.0, 0.0]])

    actual = _smooth_planar_xy(current, desired, dt=0.25, tau=0.25, deadband=0.02)

    torch.testing.assert_close(actual, torch.tensor([[1.0 - math.exp(-1.0), 0.0]]))


def test_as_torch_tensor_converts_warp_backed_state(monkeypatch) -> None:
    expected = torch.tensor([[1.0, 2.0]])
    warp_value = SimpleNamespace(tensor=expected)
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(to_torch=lambda value: value.tensor))

    actual = _as_torch_tensor(warp_value)

    assert actual is expected


class _FakeSensor:
    def __init__(self) -> None:
        self.data = SimpleNamespace(output={"rgb": torch.zeros(2, 6, 8, 4)})
        self.pose_updates = []

    def set_world_poses_from_view(self, eyes, targets) -> None:
        self.pose_updates.append((eyes.clone(), targets.clone()))


class _FakeRobot:
    body_names = ["pelvis"]

    def __init__(self) -> None:
        self.data = SimpleNamespace(
            body_link_state_w=torch.tensor(
                [
                    [[0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0]],
                    [[10.0, 20.2, 0.8, 0.0, 0.0, 0.0, 1.0]],
                ]
            )
        )


class _FakeScene:
    def __init__(self, sensor: _FakeSensor, include_sensor: bool) -> None:
        self.robot = _FakeRobot()
        self.env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 0.0]])
        self._entities = {"robot": self.robot}
        if include_sensor:
            self._entities["third_person_camera"] = sensor

    def __getitem__(self, key):
        if key not in self._entities:
            raise KeyError(key)
        return self._entities[key]


class _FakeEnv(gym.Env):
    metadata = {"render_fps": 50}
    step_dt = 0.02

    def __init__(self, include_sensor: bool = True) -> None:
        super().__init__()
        sensor = _FakeSensor()
        self.sensor = sensor
        self.scene = _FakeScene(sensor, include_sensor)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {}, {}

    def step(self, action):
        return {}, 0.0, False, False, {}


@pytest.mark.parametrize("camera_mode", ["fixed", "pelvis"])
def test_mosaic_recorder_does_not_override_sensor_pose(tmp_path, camera_mode: str) -> None:
    env = _FakeEnv()

    recorder = TiledCameraMosaicRecorder(
        env,
        video_folder=str(tmp_path),
        sensor_name="third_person_camera",
        step_trigger=lambda step: step == 0,
        video_length=2,
        columns=2,
        camera_mode=camera_mode,
    )
    recorder.reset()

    assert env.sensor.pose_updates == []


def test_planar_recorder_translates_eye_and_target_without_rotating(tmp_path) -> None:
    env = _FakeEnv()
    recorder = TiledCameraMosaicRecorder(
        env,
        video_folder=str(tmp_path),
        sensor_name="third_person_camera",
        step_trigger=lambda step: False,
        video_length=2,
        columns=2,
        camera_mode="planar",
        camera_eye=(-2.2, -2.2, 1.7),
        camera_target=(0.0, 0.0, 0.6),
        follow_tau=0.25,
        follow_deadband=0.0,
    )
    initial_eyes, initial_targets = env.sensor.pose_updates[-1]
    torch.testing.assert_close(initial_eyes, torch.tensor([[-2.2, -2.0, 1.7], [7.8, 18.0, 1.7]]))
    torch.testing.assert_close(initial_targets, torch.tensor([[0.0, 0.2, 0.6], [10.0, 20.2, 0.6]]))

    env.scene.robot.data.body_link_state_w[:, 0, 0] += 1.0
    recorder.step(None)

    moved_eyes, moved_targets = env.sensor.pose_updates[-1]
    expected_delta = 1.0 - math.exp(-env.step_dt / 0.25)
    torch.testing.assert_close(moved_eyes[:, 0], initial_eyes[:, 0] + expected_delta)
    torch.testing.assert_close(moved_targets - moved_eyes, initial_targets - initial_eyes)


def test_planar_recorder_reset_snaps_to_current_pelvis_position(tmp_path) -> None:
    env = _FakeEnv()
    recorder = TiledCameraMosaicRecorder(
        env,
        video_folder=str(tmp_path),
        sensor_name="third_person_camera",
        step_trigger=lambda step: False,
        video_length=2,
        columns=2,
        camera_mode="planar",
        camera_eye=(-2.2, -2.2, 1.7),
        camera_target=(0.0, 0.0, 0.6),
        follow_tau=0.25,
        follow_deadband=0.02,
    )
    env.scene.robot.data.body_link_state_w[:, 0, 0] += 5.0

    recorder.reset()

    eyes, _ = env.sensor.pose_updates[-1]
    torch.testing.assert_close(eyes[:, 0], torch.tensor([2.8, 12.8]))


def test_mosaic_recorder_rejects_missing_scene_sensor(tmp_path) -> None:
    with pytest.raises(AssertionError, match="third_person_camera"):
        TiledCameraMosaicRecorder(
            _FakeEnv(include_sensor=False),
            video_folder=str(tmp_path),
            sensor_name="third_person_camera",
            step_trigger=lambda step: step == 0,
            video_length=2,
            columns=2,
        )
