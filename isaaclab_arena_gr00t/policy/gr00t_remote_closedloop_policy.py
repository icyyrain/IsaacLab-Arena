# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote closed-loop policy using GR00T's native PolicyClient.

This policy connects to a GR00T policy server (launched via
``gr00t/eval/run_gr00t_server.py``) and uses its own observation/action translation pipeline.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import json
import os
import time
import torch
from dataclasses import dataclass, field
from typing import Any

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.policy.action_scheduling import (
    ActionChunkScheduler,
    ActionScheduler,
    AsyncChunkRequest,
    AsyncDeadlineActionScheduler,
    SyncedBatchActionScheduler,
)
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.async_metrics import (
    build_async_metrics,
    build_async_trace,
    write_async_metrics,
    write_async_trace,
)
from isaaclab_arena_gr00t.policy.async_status_ui import AsyncStatusWindow
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.policy.gr00t_async_worker import (
    Gr00tAsyncInferenceWorker,
    Gr00tWorkerRequest,
    Gr00tWorkerResult,
)
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyArgs,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_gr00t_modality_config_from_file


# TODO(xinjieyao, 2026-04-27): consider adding RemotePolicyArgs to inherit from BasePolicyArgs
# and then having Gr00tRemoteClosedloopPolicyArgs inherit from RemotePolicyArgs
@dataclass
class Gr00tRemoteClosedloopPolicyArgs(Gr00tBasePolicyArgs):
    """Configuration for Gr00tRemoteClosedloopPolicy.

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyArgs,
    and adds remote server connection parameters and num_envs.
    """

    num_envs: int = field(default=1, metadata={"help": "Number of environments to simulate"})
    remote_host: str = field(default="localhost", metadata={"help": "GR00T policy server hostname"})
    remote_port: int = field(default=5555, metadata={"help": "GR00T policy server port"})
    remote_api_token: str | None = field(default=None, metadata={"help": "API token for the policy server"})
    async_prefetch_lead_steps: int = field(default=25, metadata={"help": "Control steps reserved for prefetch"})
    async_step_dt: float = field(default=0.02, metadata={"help": "Control-time duration of one action step"})
    async_network_delay_s: float = field(default=0.0, metadata={"help": "Virtual one-way response delay"})
    async_metrics_path: str | None = field(default=None, metadata={"help": "Optional async metrics JSON path"})
    async_trace_path: str | None = field(default=None, metadata={"help": "Optional per-step async trace JSON path"})
    async_status_ui: bool = field(default=True, metadata={"help": "Show the Kit async status window"})

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Gr00tRemoteClosedloopPolicyArgs:
        """Create configuration from parsed CLI arguments."""
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            remote_api_token=getattr(args, "remote_api_token", None),
            async_prefetch_lead_steps=getattr(args, "async_prefetch_lead_steps", 25),
            async_step_dt=getattr(args, "async_step_dt", 0.02),
            async_network_delay_s=getattr(args, "async_network_delay_s", 0.0),
            async_metrics_path=getattr(args, "async_metrics_path", None),
            async_trace_path=getattr(args, "async_trace_path", None),
            async_status_ui=getattr(args, "async_status_ui", True),
        )


# TODO(xinjieyao, 2026-04-27): add policy registry
class Gr00tRemoteClosedloopPolicy(PolicyBase):
    """GR00T closed-loop policy that delegates inference to a remote GR00T server.

    Uses GR00T's native ``PolicyClient`` (from ``gr00t.policy.server_client``)
    to communicate with a GR00T policy server.
    """

    name = "gr00t_remote_closedloop"
    config_class = Gr00tRemoteClosedloopPolicyArgs

    @property
    def is_remote(self) -> bool:
        return True

    def __init__(
        self,
        config: Gr00tRemoteClosedloopPolicyArgs,
        action_scheduler_cls: type[ActionScheduler] = ActionChunkScheduler,
        scheduler_mode: str | None = None,
    ):
        super().__init__(config)

        # Policy config (for obs/action translation — no model loading)
        # TODO(xinjieyao, 2026-04-27): to be refactored
        self.policy_config: Gr00tClosedloopPolicyConfig = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyConfig
        )
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configs (for sim from/to policy joint space remapping)
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )

        # Action / chunk shapes
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length

        self._scheduler_mode = scheduler_mode or "chunk"
        self._chunking_state: ActionScheduler | None = None
        self._async_scheduler: AsyncDeadlineActionScheduler | None = None
        self._async_worker: Gr00tAsyncInferenceWorker | None = None
        self._async_status_window: AsyncStatusWindow | None = None
        self._client: Gr00tPolicyClient | None = None

        def client_factory() -> Gr00tPolicyClient:
            return Gr00tPolicyClient(
                host=config.remote_host,
                port=config.remote_port,
                api_token=config.remote_api_token,
                strict=False,
            )

        if self._scheduler_mode == "async_edf":
            assert (
                1 <= config.async_prefetch_lead_steps <= self.action_chunk_length
            ), "async prefetch lead must be between one and action_chunk_length"
            assert config.async_step_dt > 0.0, "async_step_dt must be positive"
            assert config.async_network_delay_s >= 0.0, "async_network_delay_s must be non-negative"
            self._async_scheduler = AsyncDeadlineActionScheduler(
                num_envs=self.num_envs,
                action_chunk_length=self.action_chunk_length,
                action_horizon=self.policy_config.action_horizon,
                action_dim=self.action_dim,
                step_dt=config.async_step_dt,
                prefetch_lead_steps=config.async_prefetch_lead_steps,
                device=self.device,
                dtype=torch.float,
            )
            self._async_worker = Gr00tAsyncInferenceWorker(client_factory=client_factory)
            if config.async_status_ui:
                try:
                    self._async_status_window = AsyncStatusWindow(self.num_envs)
                except Exception as exc:
                    print(f"[ASYNC_VLA] Status UI unavailable: {exc}", flush=True)
        else:
            self._chunking_state = action_scheduler_cls(
                num_envs=self.num_envs,
                action_chunk_length=self.action_chunk_length,
                action_horizon=self.policy_config.action_horizon,
                action_dim=self.action_dim,
                device=self.device,
                dtype=torch.float,
            )
            client = client_factory()
            self._client = client
            if not client.ping():
                raise ConnectionError(f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}")

        self.task_description: str | None = None
        self._timing_enabled = os.getenv("GR00T_TIMING", "").lower() in {"1", "true", "yes"}
        self._timing_chunk_id = 0
        self._timing_last_chunk_fetch_end: float | None = None
        self._async_rollout_wall_start: float | None = None
        self._async_trace_frames: list[dict[str, Any]] = []

    # ---------------------- CLI helpers -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Gr00t Remote Closedloop Policy",
            "Arguments for GR00T remote closed-loop policy evaluation.",
        )
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the Gr00t closedloop policy config YAML file",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device for Arena-side tensor operations (default: cuda)",
        )
        group.add_argument("--remote_host", type=str, default="localhost", help="GR00T policy server hostname")
        group.add_argument("--remote_port", type=int, default=5555, help="GR00T policy server port")
        group.add_argument("--remote_api_token", type=str, default=None, help="API token for the policy server")
        group.add_argument(
            "--remote_kill_on_exit",
            action="store_true",
            help="Stop the shared GR00T server when the rollout exits",
        )
        group.add_argument(
            "--scheduler",
            type=str,
            default="chunk",
            choices=["chunk", "synced_batch", "async_edf"],
            help=(
                "Action scheduler: 'chunk' fetches a new chunk for any env that needs one;"
                " 'synced_batch' waits until ALL envs need a new chunk and then issues a single"
                " full-batch inference call (envs that finish early hold their current robot state);"
                " 'async_edf' issues independent B=1 requests on a background EDF worker."
            ),
        )
        group.add_argument("--async_prefetch_lead_steps", type=int, default=25)
        group.add_argument("--async_step_dt", type=float, default=0.02)
        group.add_argument("--async_network_delay_s", type=float, default=0.0)
        group.add_argument("--async_metrics_path", type=str, default=None)
        group.add_argument("--async_trace_path", type=str, default=None)
        group.add_argument("--async_status_ui", action=argparse.BooleanOptionalAction, default=True)
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tRemoteClosedloopPolicy:
        config = Gr00tRemoteClosedloopPolicyArgs.from_cli_args(args)
        scheduler_cls: type[ActionScheduler] = (
            SyncedBatchActionScheduler
            if getattr(args, "scheduler", "chunk") == "synced_batch"
            else ActionChunkScheduler
        )
        return Gr00tRemoteClosedloopPolicy(
            config,
            action_scheduler_cls=scheduler_cls,
            scheduler_mode=getattr(args, "scheduler", "chunk"),
        )

    # ---------------------- Policy interface -------------------

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if not task_description:
            raise ValueError(
                "No language instruction provided. Set 'language_instruction' in the job config, "
                "pass --language_instruction on the CLI, or define 'task_description' on the task class."
            )
        self.task_description = task_description
        return self.task_description

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        if self._scheduler_mode == "async_edf":
            return self._get_async_action(observation)
        assert self._chunking_state is not None, "GR00T remote policy has been closed"

        def fetch_chunk() -> torch.Tensor:
            return self._get_action_chunk(observation, self.policy_config.pov_cam_name_sim)

        return self._chunking_state.get_action(
            fetch_chunk,
            hold_action=self._extract_hold_action(observation),
        )

    def _get_async_action(self, observation: dict[str, Any]) -> torch.Tensor:
        assert self._async_scheduler is not None, "Async scheduler has been closed"
        assert self._async_worker is not None, "Async worker has been closed"

        self._process_async_results(self._async_worker.poll())
        active_env_id = self._async_worker.active_env_id
        if active_env_id is not None:
            self._async_scheduler.mark_inference_started(active_env_id, self._async_scheduler.generation(active_env_id))

        policy_observations = None
        bootstrap_env_ids = self._async_scheduler.needs_bootstrap_env_ids
        if len(bootstrap_env_ids) > 0:
            policy_observations = self._build_policy_observations(observation, self.policy_config.pov_cam_name_sim)
            self._bootstrap_async(policy_observations, bootstrap_env_ids)
        if self._async_rollout_wall_start is None:
            self._async_rollout_wall_start = time.perf_counter()

        action = self._async_scheduler.step(self._extract_hold_action(observation))
        pending_requests = self._async_scheduler.take_pending_requests()
        if pending_requests:
            if policy_observations is None:
                policy_observations = self._build_policy_observations(observation, self.policy_config.pov_cam_name_sim)
            split_observations = _split_policy_observations(policy_observations)
            for request in pending_requests:
                self._async_worker.submit(
                    Gr00tWorkerRequest(
                        scheduler_request=request,
                        payload=split_observations[request.env_id],
                    )
                )
        self._update_async_status()
        self._record_async_trace()
        return action

    def _record_async_trace(self) -> None:
        if self.config.async_trace_path is None or self._async_scheduler is None or self._async_worker is None:
            return
        scheduler_metrics = self._async_scheduler.metrics()
        self._async_trace_frames.append({
            "step": len(self._async_trace_frames),
            "sim_time_s": round(self._async_scheduler.sim_time_s, 9),
            "queue_depth": self._async_worker.queue_depth,
            "active_inference_env_id": self._async_worker.active_env_id,
            "deadline_miss_count": int(scheduler_metrics["deadline_miss_count"]),
            "robots": self._async_scheduler.state_snapshot(),
        })

    def _update_async_status(self) -> None:
        if self._async_status_window is None or self._async_scheduler is None or self._async_worker is None:
            return
        metrics = self.async_metrics()
        try:
            self._async_status_window.update(
                statuses=self._async_scheduler.statuses,
                sim_time_s=self._async_scheduler.sim_time_s,
                queue_depth=self._async_worker.queue_depth,
                miss_count=int(metrics["deadline_miss_count"]),
                inference_p95_wall_s=metrics["inference_wall_s"]["p95"],
            )
        except Exception as exc:
            print(f"[ASYNC_VLA] Disabling status UI after update failure: {exc}", flush=True)
            self._async_status_window.close()
            self._async_status_window = None

    def _bootstrap_async(self, policy_observations: dict[str, Any], env_ids: torch.Tensor) -> None:
        assert self._async_scheduler is not None
        assert self._async_worker is not None
        split_observations = _split_policy_observations(policy_observations)
        target_env_ids = env_ids.tolist()
        for offset, env_id in enumerate(target_env_ids):
            request = AsyncChunkRequest(
                env_id=env_id,
                generation=self._async_scheduler.generation(env_id),
                submit_sim_time_s=self._async_scheduler.sim_time_s,
                deadline_sim_time_s=self._async_scheduler.sim_time_s,
                sequence=-len(target_env_ids) + offset,
            )
            self._async_worker.submit(
                Gr00tWorkerRequest(
                    scheduler_request=request,
                    payload=split_observations[env_id],
                    is_bootstrap=True,
                )
            )

        chunks_by_env: dict[int, torch.Tensor] = {}
        while len(chunks_by_env) < len(target_env_ids):
            result = self._async_worker.wait_for_results(count=1, timeout_s=120.0)[0]
            if result.request.is_bootstrap:
                if result.error is not None:
                    raise RuntimeError(f"GR00T bootstrap inference failed: {result.error}") from result.error
                assert result.action is not None
                chunk = self._translate_action_response(result.action, expected_num_envs=1)[0]
                chunks_by_env[result.request.scheduler_request.env_id] = chunk
            else:
                self._process_async_results([result])

        chunks = torch.stack([chunks_by_env[env_id] for env_id in target_env_ids])
        self._async_scheduler.bootstrap(chunks, env_ids=env_ids)

    def _process_async_results(self, results: list[Gr00tWorkerResult]) -> None:
        assert self._async_scheduler is not None
        for result in results:
            if result.error is not None:
                raise RuntimeError(f"GR00T asynchronous inference failed: {result.error}") from result.error
            assert result.action is not None
            chunk = self._translate_action_response(result.action, expected_num_envs=1)[0]
            self._async_scheduler.accept_result(
                request=result.request.scheduler_request,
                chunk=chunk,
                inference_wall_s=result.inference_wall_s,
                network_delay_s=self.config.async_network_delay_s,
            )

    def _extract_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Build the action vector that waiting envs should hold: their current sim joint positions
        copied into the action slots that share a joint name with the state config."""
        joint_pos_sim = observation["policy"]["robot_joint_pos"].to(device=self.device, dtype=torch.float)
        hold_action = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float, device=self.device)
        for joint_name, action_idx in self.robot_action_joints_config.items():
            state_idx = self.robot_state_joints_config.get(joint_name)
            if state_idx is not None:
                hold_action[:, action_idx] = joint_pos_sim[:, state_idx]
        return hold_action

    def _get_action_chunk(
        self, observation: dict[str, Any], camera_names: list[str] | str = "robot_head_cam_rgb"
    ) -> torch.Tensor:
        """Get an action chunk from the remote GR00T server.

        Calls GR00T's PolicyClient to get the action chunk.
        """
        total_start = time.perf_counter()
        assert self._client is not None, "GR00T remote policy has been closed"
        policy_observations = self._build_policy_observations(observation, camera_names)

        # 2. Call GR00T's own client
        fetch_start = time.perf_counter()
        previous_chunk_execution_s = None
        if self._timing_enabled and self._timing_last_chunk_fetch_end is not None:
            previous_chunk_execution_s = fetch_start - self._timing_last_chunk_fetch_end
        robot_action_policy, _ = self._client.get_action(policy_observations)
        fetch_end = time.perf_counter()

        # 3. Action translation from policy output to sim action tensor
        action_tensor = self._translate_action_response(robot_action_policy, expected_num_envs=self.num_envs)
        total_end = time.perf_counter()
        if self._timing_enabled:
            print(
                "[GR00T_TIMING] "
                f"chunk={self._timing_chunk_id} "
                f"inference_wall_s={fetch_end - fetch_start:.6f} "
                f"total_chunk_fetch_wall_s={total_end - total_start:.6f} "
                "prev_chunk_execution_wall_s="
                f"{previous_chunk_execution_s if previous_chunk_execution_s is not None else 'NA'} "
                f"chunk_len={self.action_chunk_length}",
                flush=True,
            )
            self._timing_chunk_id += 1
            self._timing_last_chunk_fetch_end = fetch_end
        return action_tensor

    def _build_policy_observations(self, observation: dict[str, Any], camera_names: list[str] | str) -> dict[str, Any]:
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        assert self.task_description is not None, "Task description is not set"
        rgb_list_np, joint_pos_sim_np = extract_obs_numpy_from_torch(nested_obs=observation, camera_names=camera_names)
        return build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
        )

    def _translate_action_response(self, robot_action_policy: dict[str, Any], expected_num_envs: int) -> torch.Tensor:
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )
        assert action_tensor.shape[0] == expected_num_envs
        assert action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        if self._scheduler_mode == "async_edf":
            assert self._async_scheduler is not None, "Async scheduler has been closed"
            scheduler_env_ids = None if isinstance(env_ids, slice) else env_ids
            self._async_scheduler.reset(scheduler_env_ids)
            return
        assert self._client is not None, "GR00T remote policy has been closed"
        assert self._chunking_state is not None, "GR00T remote policy has been closed"
        self._client.reset()
        self._chunking_state.reset(env_ids)

    def async_metrics(self) -> dict[str, Any]:
        assert self._async_scheduler is not None, "Async metrics are available only in async_edf mode"
        wall_elapsed_s = (
            time.perf_counter() - self._async_rollout_wall_start if self._async_rollout_wall_start is not None else 0.0
        )
        return build_async_metrics(
            self._async_scheduler.metrics(),
            wall_elapsed_s=wall_elapsed_s,
            step_dt=self.config.async_step_dt,
        )

    def close(self) -> None:
        """Release Arena-side resources for the remote GR00T policy client."""
        if self._scheduler_mode == "async_edf":
            if self._async_worker is not None:
                self._process_async_results(self._async_worker.poll())
                self._async_worker.close(drain=True)
                self._process_async_results(self._async_worker.poll())
                self._async_worker = None
            if self._async_scheduler is not None:
                metrics = self.async_metrics()
                print(f"[ASYNC_VLA_SUMMARY] {json.dumps(metrics, sort_keys=True)}", flush=True)
                if self.config.async_metrics_path:
                    write_async_metrics(self.config.async_metrics_path, metrics)
                if self.config.async_trace_path:
                    trace = build_async_trace(
                        num_envs=self.num_envs,
                        step_dt=self.config.async_step_dt,
                        deadline_window_s=float(metrics["deadline_window_sim_s"]),
                        frames=self._async_trace_frames,
                    )
                    write_async_trace(self.config.async_trace_path, trace)
            if self._async_status_window is not None:
                self._async_status_window.close()
                self._async_status_window = None
            return
        client = self._client
        try:
            if client is not None:
                socket = getattr(client, "socket", None)
                context = getattr(client, "context", None)
                try:
                    if socket is not None:
                        socket.close(linger=0)
                finally:
                    if context is not None:
                        context.term()
        finally:
            self._client = None
            self._chunking_state = None
            self.modality_configs = None

    def shutdown_remote(self, kill_server: bool = False) -> None:
        """Close Arena-side resources and optionally stop the shared remote server."""
        if kill_server and self._client is not None:
            self._client.kill_server()
        self.close()
        if kill_server and self._scheduler_mode == "async_edf":
            client = Gr00tPolicyClient(
                host=self.config.remote_host,
                port=self.config.remote_port,
                api_token=self.config.remote_api_token,
                strict=False,
            )
            try:
                client.kill_server()
            finally:
                Gr00tAsyncInferenceWorker._close_client(client)


def _split_policy_observations(policy_observations: dict[str, Any]) -> list[dict[str, Any]]:
    video_values = next(iter(policy_observations["video"].values()))
    num_envs = video_values.shape[0]
    return [
        {
            "video": {key: value[env_id : env_id + 1] for key, value in policy_observations["video"].items()},
            "state": {key: value[env_id : env_id + 1] for key, value in policy_observations["state"].items()},
            "language": {key: value[env_id : env_id + 1] for key, value in policy_observations["language"].items()},
        }
        for env_id in range(num_envs)
    ]
