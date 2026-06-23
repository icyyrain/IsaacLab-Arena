# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render asynchronous VLA scheduler state into rollout videos and timeline images."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

import cv2

_STATUS_BGR = {
    "bootstrap": (220, 125, 45),
    "executing": (85, 180, 45),
    "queued": (40, 200, 245),
    "inference": (220, 125, 45),
    "gated": (220, 125, 45),
    "deadline_miss": (55, 55, 225),
}

_STATUS_LABEL = {
    "bootstrap": "BOOT",
    "executing": "EXEC",
    "queued": "QUEUE",
    "inference": "INFER",
    "gated": "GATED",
    "deadline_miss": "MISS / HOLD",
}

_GPU_ROBOT_BGR = (
    (214, 104, 48),
    (46, 139, 230),
    (72, 170, 72),
    (180, 105, 190),
    (55, 190, 210),
    (190, 145, 70),
)
_GPU_IDLE_BGR = (205, 205, 205)


def overlay_status_frame(frame: np.ndarray, trace_frame: dict[str, Any]) -> np.ndarray:
    """Draw a compact scheduler panel directly into one BGR video frame."""
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    robots = trace_frame["robots"]
    panel_height = min(112, height)
    cv2.rectangle(rendered, (0, 0), (width, panel_height), (25, 25, 25), thickness=-1)
    header = (
        f"sim {trace_frame['sim_time_s']:.2f}s   queue {trace_frame['queue_depth']}   "
        f"misses {trace_frame['deadline_miss_count']}"
    )
    cv2.putText(rendered, header, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)

    gap = 6
    tile_y0 = 36
    tile_height = panel_height - tile_y0 - 6
    tile_width = max(1, (width - gap * (len(robots) + 1)) // max(1, len(robots)))
    for index, robot in enumerate(robots):
        x0 = gap + index * (tile_width + gap)
        x1 = min(width - gap, x0 + tile_width)
        status = robot["status"]
        color = _STATUS_BGR[status]
        cv2.rectangle(rendered, (x0, tile_y0), (x1, tile_y0 + tile_height), color, thickness=-1)
        cv2.putText(
            rendered,
            f"R{robot['env_id']} {_STATUS_LABEL[status]}",
            (x0 + 7, tile_y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (15, 15, 15),
            2,
        )
        deadline = robot.get("deadline_remaining_sim_s")
        deadline_text = "deadline --" if deadline is None else f"deadline {deadline:.2f}s"
        cv2.putText(
            rendered,
            deadline_text,
            (x0 + 7, tile_y0 + 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (15, 15, 15),
            1,
        )
    return rendered


def build_status_timeline(trace: dict[str, Any], width: int = 1600) -> np.ndarray:
    """Build a color-lane timeline with a queue-depth plot below the robot lanes."""
    frames = trace["frames"]
    num_envs = int(trace["num_envs"])
    margin_left = 90
    margin_right = 24
    header_height = 55
    lane_height = 44
    gpu_lane_height = 44
    queue_height = 110
    height = header_height + num_envs * lane_height + gpu_lane_height + queue_height + 70
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    plot_width = max(1, width - margin_left - margin_right)
    cv2.putText(canvas, "Async VLA scheduler timeline", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2)

    for env_id in range(num_envs):
        y0 = header_height + env_id * lane_height
        cv2.putText(canvas, f"R{env_id}", (24, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
        for index, frame in enumerate(frames):
            x0 = margin_left + int(index * plot_width / max(1, len(frames)))
            x1 = margin_left + int((index + 1) * plot_width / max(1, len(frames)))
            status = frame["robots"][env_id]["status"]
            cv2.rectangle(canvas, (x0, y0 + 4), (max(x0, x1), y0 + lane_height - 4), _STATUS_BGR[status], -1)

    duration = float(frames[-1]["sim_time_s"]) if frames else 0.0
    gpu_top = header_height + num_envs * lane_height + 8
    gpu_bottom = gpu_top + gpu_lane_height - 8
    cv2.putText(canvas, "GPU", (18, gpu_top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)
    cv2.rectangle(canvas, (margin_left, gpu_top), (margin_left + plot_width, gpu_bottom), _GPU_IDLE_BGR, -1)
    if duration > 0.0:
        for interval in trace.get("gpu_service_intervals", []):
            start_s = max(0.0, min(duration, float(interval["start_sim_time_s"])))
            finish_s = max(start_s, min(duration, float(interval["finish_sim_time_s"])))
            x0 = margin_left + int(start_s * plot_width / duration)
            x1 = margin_left + int(finish_s * plot_width / duration)
            env_id = int(interval["env_id"])
            color = _GPU_ROBOT_BGR[env_id % len(_GPU_ROBOT_BGR)]
            cv2.rectangle(canvas, (x0, gpu_top), (max(x0 + 1, x1), gpu_bottom), color, -1)
            if x1 - x0 >= 24:
                cv2.putText(
                    canvas,
                    f"R{env_id}",
                    (x0 + 3, gpu_top + 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (20, 20, 20),
                    1,
                )

    queue_top = gpu_top + gpu_lane_height + 10
    cv2.putText(canvas, "queue", (18, queue_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)
    max_queue = max([int(frame["queue_depth"]) for frame in frames] + [1])
    points = []
    for index, frame in enumerate(frames):
        x = margin_left + int(index * plot_width / max(1, len(frames) - 1))
        y = queue_top + queue_height - int(int(frame["queue_depth"]) * (queue_height - 20) / max_queue)
        points.append((x, y))
    if len(points) >= 2:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (80, 50, 180), 2)
    elif points:
        cv2.circle(canvas, points[0], 3, (80, 50, 180), -1)

    cv2.putText(canvas, "0s", (margin_left, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    cv2.putText(
        canvas,
        f"{duration:.1f}s",
        (width - margin_right - 55, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (30, 30, 30),
        1,
    )
    return canvas


def render_status_video(
    input_video_path: str | Path,
    trace_path: str | Path,
    output_video_path: str | Path,
    timeline_path: str | Path,
) -> dict[str, int | float]:
    """Overlay a trace onto an MP4 and write the companion timeline PNG."""
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    frames = trace["frames"]
    assert frames, "async trace contains no frames"

    capture = cv2.VideoCapture(str(input_video_path))
    assert capture.isOpened(), f"cannot open input video: {input_video_path}"
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"cannot open output video: {output_video_path}"

    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        trace_frame = frames[min(frame_count, len(frames) - 1)]
        writer.write(overlay_status_frame(frame, trace_frame))
        frame_count += 1
    capture.release()
    writer.release()

    timeline = build_status_timeline(trace)
    timeline_output = Path(timeline_path)
    timeline_output.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(timeline_output), timeline), f"cannot write timeline: {timeline_path}"
    return {"frame_count": frame_count, "fps": fps, "duration_s": frame_count / fps}
