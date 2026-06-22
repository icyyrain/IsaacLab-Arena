# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional Kit status window for the multi-robot asynchronous VLA demo."""

from __future__ import annotations

from typing import Any

_STATUS_PRESENTATION = {
    "bootstrap": ("BOOTSTRAP", "blue"),
    "executing": ("EXECUTING", "green"),
    "queued": ("QUEUED", "yellow"),
    "inference": ("INFERENCE", "blue"),
    "gated": ("RESULT GATED", "blue"),
    "deadline_miss": ("DEADLINE MISS / HOLD", "red"),
}


def build_status_rows(statuses: list[Any]) -> list[tuple[str, str]]:
    """Return display text and semantic color for each robot."""
    rows = []
    for env_id, status in enumerate(statuses):
        status_value = str(status)
        label, color = _STATUS_PRESENTATION[status_value]
        rows.append((f"Robot {env_id}  {label}", color))
    return rows


class AsyncStatusWindow:
    """Small floating Kit window showing queue and per-robot scheduler state."""

    _COLORS = {
        "green": 0xFF6FD08C,
        "yellow": 0xFF5BD9F5,
        "blue": 0xFFF1B56B,
        "red": 0xFF6B6BF1,
    }

    def __init__(self, num_envs: int) -> None:
        import omni.ui as ui

        self._window = ui.Window("Async VLA Status", width=360, height=100 + 28 * num_envs)
        self._labels = []
        with self._window.frame:
            with ui.VStack(spacing=4, height=0):
                self._summary = ui.Label("Starting async policy...", height=24)
                for env_id in range(num_envs):
                    self._labels.append(ui.Label(f"Robot {env_id}  BOOTSTRAP", height=24))

    def update(
        self,
        statuses: list[Any],
        sim_time_s: float,
        queue_depth: int,
        miss_count: int,
        inference_p95_wall_s: float | None,
    ) -> None:
        p95_text = "NA" if inference_p95_wall_s is None else f"{inference_p95_wall_s:.3f}s"
        self._summary.text = (
            f"sim={sim_time_s:.2f}s  queue={queue_depth}  misses={miss_count}  infer p95={p95_text} wall"
        )
        for label, (text, semantic_color) in zip(self._labels, build_status_rows(statuses)):
            label.text = text
            label.style = {"color": self._COLORS[semantic_color]}

    def close(self) -> None:
        self._window.visible = False
        self._labels = []
