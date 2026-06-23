# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record one row-major video mosaic from a batched Isaac Lab camera sensor."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import cv2
import gymnasium as gym
import numpy as np
import torch


def _to_uint8_rgb(frames: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    assert frames.ndim == 4 and frames.shape[-1] >= 3, f"expected [N,H,W,C>=3], got {frames.shape}"
    frames = frames[..., :3]
    if frames.dtype.kind == "f":
        scale = 255.0 if float(frames.max()) <= 1.0 else 1.0
        frames = np.clip(frames * scale, 0, 255).astype(np.uint8)
    elif frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def tile_camera_batch(
    frames: torch.Tensor | np.ndarray,
    columns: int = 3,
    draw_labels: bool = True,
) -> np.ndarray:
    """Tile an ``[N,H,W,C]`` camera batch in row-major environment order."""
    assert columns > 0, "mosaic columns must be positive"
    frames_rgb = _to_uint8_rgb(frames)
    num_frames, frame_height, frame_width, _ = frames_rgb.shape
    assert num_frames > 0, "camera batch must contain at least one environment"
    rows = math.ceil(num_frames / columns)
    mosaic = np.zeros((rows * frame_height, columns * frame_width, 3), dtype=np.uint8)
    for env_id, frame in enumerate(frames_rgb):
        row, column = divmod(env_id, columns)
        y0 = row * frame_height
        x0 = column * frame_width
        mosaic[y0 : y0 + frame_height, x0 : x0 + frame_width] = frame
        if draw_labels:
            cv2.rectangle(mosaic, (x0 + 5, y0 + 5), (x0 + 57, y0 + 29), (15, 15, 15), -1)
            cv2.putText(
                mosaic,
                f"R{env_id}",
                (x0 + 11, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                2,
            )
    return mosaic


def _write_mp4(path: str | os.PathLike[str], frames: list[np.ndarray], fps: int) -> None:
    """Write every RGB frame without MoviePy's exact-duration frame loss."""
    assert frames, "cannot encode an empty mosaic video"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"cannot open mosaic video writer: {path}"
    try:
        for frame in frames:
            assert frame.shape == (height, width, 3), "all mosaic frames must have the same RGB shape"
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


class TiledCameraMosaicRecorder(gym.Wrapper):
    """Record a single MP4 from one batched scene camera."""

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        sensor_name: str,
        step_trigger: Callable[[int], bool],
        video_length: int,
        columns: int = 3,
        name_prefix: str = "third-person-mosaic",
        fps: int | None = None,
    ) -> None:
        super().__init__(env)
        assert columns > 0, "mosaic columns must be positive"
        assert video_length > 0, "mosaic video length must be positive"
        os.makedirs(video_folder, exist_ok=True)
        self.video_folder = video_folder
        self.sensor_name = sensor_name
        self.step_trigger = step_trigger
        self.video_length = video_length
        self.columns = columns
        self.name_prefix = name_prefix
        self.fps = fps if fps is not None else int(env.metadata.get("render_fps", 30))
        self.step_id = -1
        self.recording = False
        self.recording_start_step = 0
        self.frames: list[np.ndarray] = []

        scene = env.unwrapped.scene
        try:
            self.sensor = scene[sensor_name]
        except KeyError as exc:
            raise AssertionError(f"mosaic camera sensor '{sensor_name}' is missing from the scene") from exc

    def step(self, action):
        result = self.env.step(action)
        self.step_id += 1
        if not self.recording and self.step_trigger(self.step_id):
            self.recording = True
            self.recording_start_step = self.step_id
            self.frames = []
        if self.recording:
            rgb = self.sensor.data.output.get("rgb")
            assert rgb is not None, f"mosaic camera sensor '{self.sensor_name}' has no rgb output"
            self.frames.append(tile_camera_batch(rgb, columns=self.columns, draw_labels=True))
            if len(self.frames) >= self.video_length:
                self._flush()
        return result

    def _flush(self) -> None:
        if not self.frames:
            return
        path = os.path.join(
            self.video_folder,
            f"{self.name_prefix}-step-{self.recording_start_step}.mp4",
        )
        _write_mp4(path, self.frames, self.fps)
        self.recording = False
        self.frames = []

    def close(self) -> None:
        try:
            if self.recording and self.frames:
                self._flush()
        finally:
            self.env.close()
