# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Control-time action chunk scheduling for asynchronous policy inference."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from enum import StrEnum


class AsyncEnvStatus(StrEnum):
    """Visible state of one environment in the asynchronous pipeline."""

    BOOTSTRAP = "bootstrap"
    EXECUTING = "executing"
    QUEUED = "queued"
    INFERENCE = "inference"
    GATED = "gated"
    DEADLINE_MISS = "deadline_miss"


@dataclass(frozen=True)
class AsyncChunkRequest:
    """Metadata needed to schedule one environment's next action chunk."""

    env_id: int
    generation: int
    submit_sim_time_s: float
    deadline_sim_time_s: float
    sequence: int


class AsyncDeadlineActionScheduler:
    """Replay per-env chunks while issuing prefetch requests on a control-time clock."""

    def __init__(
        self,
        num_envs: int,
        action_chunk_length: int,
        action_horizon: int,
        action_dim: int,
        step_dt: float,
        prefetch_lead_steps: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert num_envs > 0, "num_envs must be positive"
        assert action_horizon >= action_chunk_length > 0, "action horizon must cover the executed chunk"
        assert 1 <= prefetch_lead_steps <= action_chunk_length, "prefetch lead must fit inside the chunk"
        assert step_dt > 0.0, "step_dt must be positive"

        self.num_envs = num_envs
        self.action_chunk_length = action_chunk_length
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.step_dt = step_dt
        self.prefetch_lead_steps = prefetch_lead_steps
        self.device = torch.device(device)
        self.dtype = dtype

        self.current_action_chunk = torch.zeros((num_envs, action_horizon, action_dim), dtype=dtype, device=self.device)
        self.next_action_chunk = torch.zeros_like(self.current_action_chunk)
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._bootstrapped = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._request_outstanding = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._next_available = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._next_virtual_finish_s = torch.full((num_envs,), float("inf"), dtype=torch.float64)
        self._deadline_sim_time_s = torch.full((num_envs,), float("inf"), dtype=torch.float64)
        self._generation = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._miss_active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._deadline_recorded = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._pending_requests: list[AsyncChunkRequest] = []
        self._sequence = 0
        self._step_count = 0
        self._virtual_gpu_available_s = 0.0
        self._request_count = torch.zeros(num_envs, dtype=torch.int64)
        self._deadline_count = torch.zeros(num_envs, dtype=torch.int64)
        self._deadline_miss_count = torch.zeros(num_envs, dtype=torch.int64)
        self._hold_steps = torch.zeros(num_envs, dtype=torch.int64)
        self._inference_wall_s: list[list[float]] = [[] for _ in range(num_envs)]
        self._virtual_queue_wait_s: list[list[float]] = [[] for _ in range(num_envs)]
        self.statuses = [AsyncEnvStatus.BOOTSTRAP for _ in range(num_envs)]

    @property
    def sim_time_s(self) -> float:
        return self._step_count * self.step_dt

    @property
    def needs_bootstrap_env_ids(self) -> torch.Tensor:
        return (~self._bootstrapped).nonzero().flatten()

    def generation(self, env_id: int) -> int:
        return int(self._generation[env_id].item())

    def bootstrap(self, chunks: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        expected_shape = (len(env_ids), self.action_horizon, self.action_dim)
        assert chunks.shape == expected_shape, f"{chunks.shape=} != {expected_shape}"
        self.current_action_chunk[env_ids] = chunks.to(device=self.device, dtype=self.dtype)
        self.current_action_index[env_ids] = 0
        self._bootstrapped[env_ids] = True
        self._request_outstanding[env_ids] = False
        self._next_available[env_ids] = False
        self._miss_active[env_ids] = False
        self._deadline_recorded[env_ids] = False
        for env_id in env_ids.tolist():
            self.statuses[env_id] = AsyncEnvStatus.EXECUTING

    def step(self, hold_action: torch.Tensor) -> torch.Tensor:
        expected_hold_shape = (self.num_envs, self.action_dim)
        assert hold_action.shape == expected_hold_shape, f"{hold_action.shape=} != {expected_hold_shape}"

        for env_id in range(self.num_envs):
            if not self._bootstrapped[env_id]:
                continue
            if self.current_action_index[env_id] >= self.action_chunk_length:
                if not self._deadline_recorded[env_id]:
                    self._deadline_count[env_id] += 1
                    self._deadline_recorded[env_id] = True
                if self._next_available[env_id] and self.sim_time_s + 1e-9 >= self._next_virtual_finish_s[env_id]:
                    self.current_action_chunk[env_id] = self.next_action_chunk[env_id]
                    self.current_action_index[env_id] = 0
                    self._next_available[env_id] = False
                    self._deadline_sim_time_s[env_id] = float("inf")
                    self._miss_active[env_id] = False
                    self._deadline_recorded[env_id] = False
                    self.statuses[env_id] = AsyncEnvStatus.EXECUTING
                else:
                    if not self._miss_active[env_id]:
                        self._deadline_miss_count[env_id] += 1
                        self._miss_active[env_id] = True
                    self._hold_steps[env_id] += 1
                    self.statuses[env_id] = AsyncEnvStatus.DEADLINE_MISS
            elif self._next_available[env_id] and self.sim_time_s + 1e-9 >= self._next_virtual_finish_s[env_id]:
                self.statuses[env_id] = AsyncEnvStatus.EXECUTING

        trigger_index = self.action_chunk_length - self.prefetch_lead_steps
        for env_id in range(self.num_envs):
            if (
                self._bootstrapped[env_id]
                and self.current_action_index[env_id] == trigger_index
                and not self._request_outstanding[env_id]
            ):
                submit_time = self.sim_time_s
                self._pending_requests.append(
                    AsyncChunkRequest(
                        env_id=env_id,
                        generation=int(self._generation[env_id].item()),
                        submit_sim_time_s=submit_time,
                        deadline_sim_time_s=submit_time + self.prefetch_lead_steps * self.step_dt,
                        sequence=self._sequence,
                    )
                )
                self._deadline_sim_time_s[env_id] = submit_time + self.prefetch_lead_steps * self.step_dt
                self._sequence += 1
                self._request_outstanding[env_id] = True
                self._request_count[env_id] += 1
                self.statuses[env_id] = AsyncEnvStatus.QUEUED

        actions = hold_action.to(device=self.device, dtype=self.dtype).clone()
        for env_id in range(self.num_envs):
            index = int(self.current_action_index[env_id].item())
            if self._bootstrapped[env_id] and index < self.action_chunk_length:
                actions[env_id] = self.current_action_chunk[env_id, index]
                self.current_action_index[env_id] += 1

        self._step_count += 1
        return actions

    def state_snapshot(self) -> list[dict[str, int | float | str | None]]:
        """Return one JSON-serializable scheduler state row per environment."""
        snapshot = []
        for env_id in range(self.num_envs):
            action_index = int(self.current_action_index[env_id].item())
            deadline_sim_time_s = float(self._deadline_sim_time_s[env_id].item())
            deadline_remaining_sim_s = None
            if math.isfinite(deadline_sim_time_s):
                deadline_remaining_sim_s = round(max(0.0, deadline_sim_time_s - self.sim_time_s), 9)
            snapshot.append({
                "env_id": env_id,
                "status": str(self.statuses[env_id]),
                "action_index": action_index,
                "remaining_action_steps": max(0, self.action_chunk_length - action_index),
                "deadline_remaining_sim_s": deadline_remaining_sim_s,
                "generation": int(self._generation[env_id].item()),
            })
        return snapshot

    def take_pending_requests(self) -> list[AsyncChunkRequest]:
        requests = self._pending_requests
        self._pending_requests = []
        return requests

    def mark_inference_started(self, env_id: int, generation: int) -> None:
        if generation == int(self._generation[env_id].item()) and self._request_outstanding[env_id]:
            self.statuses[env_id] = AsyncEnvStatus.INFERENCE

    def accept_result(
        self,
        request: AsyncChunkRequest,
        chunk: torch.Tensor,
        inference_wall_s: float,
        network_delay_s: float = 0.0,
    ) -> bool:
        env_id = request.env_id
        if request.generation != int(self._generation[env_id].item()) or not self._request_outstanding[env_id]:
            return False
        expected_shape = (self.action_horizon, self.action_dim)
        assert chunk.shape == expected_shape, f"{chunk.shape=} != {expected_shape}"
        assert inference_wall_s >= 0.0, "inference latency must be non-negative"
        assert network_delay_s >= 0.0, "network delay must be non-negative"

        virtual_start_s = max(request.submit_sim_time_s, self._virtual_gpu_available_s)
        virtual_finish_s = virtual_start_s + inference_wall_s + network_delay_s
        self._virtual_gpu_available_s = virtual_finish_s
        self.next_action_chunk[env_id] = chunk.to(device=self.device, dtype=self.dtype)
        self._next_virtual_finish_s[env_id] = virtual_finish_s
        self._next_available[env_id] = True
        self._request_outstanding[env_id] = False
        self._inference_wall_s[env_id].append(inference_wall_s)
        self._virtual_queue_wait_s[env_id].append(virtual_start_s - request.submit_sim_time_s)
        self.statuses[env_id] = AsyncEnvStatus.GATED
        return True

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        reset_ids = set(env_ids.tolist())
        self._generation[env_ids] += 1
        self.current_action_chunk[env_ids] = 0.0
        self.next_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = 0
        self._bootstrapped[env_ids] = False
        self._request_outstanding[env_ids] = False
        self._next_available[env_ids] = False
        self._miss_active[env_ids] = False
        self._deadline_recorded[env_ids] = False
        self._next_virtual_finish_s[env_ids.cpu()] = float("inf")
        self._deadline_sim_time_s[env_ids.cpu()] = float("inf")
        self._pending_requests = [request for request in self._pending_requests if request.env_id not in reset_ids]
        for env_id in reset_ids:
            self.statuses[env_id] = AsyncEnvStatus.BOOTSTRAP

    def metrics(self) -> dict[str, object]:
        per_env = []
        for env_id in range(self.num_envs):
            virtual_finish = float(self._next_virtual_finish_s[env_id].item())
            per_env.append({
                "env_id": env_id,
                "request_count": int(self._request_count[env_id].item()),
                "deadline_count": int(self._deadline_count[env_id].item()),
                "deadline_miss_count": int(self._deadline_miss_count[env_id].item()),
                "hold_steps": int(self._hold_steps[env_id].item()),
                "inference_wall_s": list(self._inference_wall_s[env_id]),
                "virtual_queue_wait_sim_s": list(self._virtual_queue_wait_s[env_id]),
                "virtual_finish_sim_time_s": round(virtual_finish, 9) if math.isfinite(virtual_finish) else None,
            })
        return {
            "num_envs": self.num_envs,
            "sim_time_s": self.sim_time_s,
            "deadline_window_s": self.prefetch_lead_steps * self.step_dt,
            "request_count": int(self._request_count.sum().item()),
            "deadline_count": int(self._deadline_count.sum().item()),
            "deadline_miss_count": int(self._deadline_miss_count.sum().item()),
            "hold_steps": int(self._hold_steps.sum().item()),
            "per_env": per_env,
        }
