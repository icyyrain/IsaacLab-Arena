# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for batched third-person camera mosaic recording."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab_arena.evaluation.mosaic_video import TiledCameraMosaicRecorder, _write_mp4, tile_camera_batch


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


class _FakeSensor:
    def __init__(self) -> None:
        self.data = SimpleNamespace(output={"rgb": torch.zeros(2, 6, 8, 4)})
        self.pose_update_count = 0

    def set_world_poses_from_view(self, eyes, targets) -> None:
        self.pose_update_count += 1


class _FakeScene:
    def __init__(self, sensor: _FakeSensor, include_sensor: bool) -> None:
        self._entities = {"third_person_camera": sensor} if include_sensor else {}

    def __getitem__(self, key):
        if key not in self._entities:
            raise KeyError(key)
        return self._entities[key]


class _FakeEnv(gym.Env):
    metadata = {"render_fps": 50}

    def __init__(self, include_sensor: bool = True) -> None:
        super().__init__()
        sensor = _FakeSensor()
        self.sensor = sensor
        self.scene = _FakeScene(sensor, include_sensor)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {}, {}


def test_mosaic_recorder_does_not_override_sensor_pose(tmp_path) -> None:
    env = _FakeEnv()

    recorder = TiledCameraMosaicRecorder(
        env,
        video_folder=str(tmp_path),
        sensor_name="third_person_camera",
        step_trigger=lambda step: step == 0,
        video_length=2,
        columns=2,
    )
    recorder.reset()

    assert env.sensor.pose_update_count == 0


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
