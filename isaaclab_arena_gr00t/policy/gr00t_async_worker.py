# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-owner EDF worker for remote GR00T inference requests."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from isaaclab_arena.policy.action_scheduling import AsyncChunkRequest
from isaaclab_arena.policy.traffic_capture import capture_vla_call


@dataclass(frozen=True)
class Gr00tWorkerRequest:
    """One policy-formatted observation and its control-time scheduling metadata."""

    scheduler_request: AsyncChunkRequest
    payload: dict[str, Any]
    is_bootstrap: bool = False
    enqueued_wall_s: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class Gr00tWorkerResult:
    """Result of one remote inference request."""

    request: Gr00tWorkerRequest
    action: dict[str, Any] | None
    inference_wall_s: float
    queue_wait_wall_s: float
    error: Exception | None = None


class Gr00tAsyncInferenceWorker:
    """Serialize B=1 GR00T requests on a client owned by one worker thread."""

    def __init__(self, client_factory: Callable[[], Any], autostart: bool = True) -> None:
        self._client_factory = client_factory
        self._requests: queue.PriorityQueue[tuple[float, int, Gr00tWorkerRequest]] = queue.PriorityQueue()
        self._results: queue.SimpleQueue[Gr00tWorkerResult] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None
        self._drain_on_close = False
        self._active_env_id: int | None = None
        self._active_lock = threading.Lock()
        self._capture_host = "unknown"
        self._capture_port = 0
        if autostart:
            self.start()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gr00t-async-inference", daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=15.0)
        if self._startup_error is not None:
            raise self._startup_error
        if not self._ready_event.is_set():
            raise TimeoutError("GR00T async inference worker did not start within 15 seconds")

    def submit(self, request: Gr00tWorkerRequest) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("Cannot submit to a closed GR00T async inference worker")
        metadata = request.scheduler_request
        self._requests.put((metadata.deadline_sim_time_s, metadata.sequence, request))

    def poll(self) -> list[Gr00tWorkerResult]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def wait_for_results(self, count: int, timeout_s: float) -> list[Gr00tWorkerResult]:
        deadline = time.perf_counter() + timeout_s
        results = []
        while len(results) < count:
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0.0:
                raise TimeoutError(f"Timed out waiting for {count} GR00T inference results")
            try:
                results.append(self._results.get(timeout=remaining_s))
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting for {count} GR00T inference results") from exc
        return results

    @property
    def active_env_id(self) -> int | None:
        with self._active_lock:
            return self._active_env_id

    @property
    def queue_depth(self) -> int:
        return self._requests.qsize()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self, timeout_s: float = 5.0, drain: bool = False) -> None:
        self._drain_on_close = drain
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("GR00T async inference worker did not stop cleanly")

    def _run(self) -> None:
        client = None
        try:
            client = self._client_factory()
            self._capture_host = str(getattr(client, "host", "unknown"))
            self._capture_port = int(getattr(client, "port", 0))
            if not client.ping():
                raise ConnectionError("Cannot reach the GR00T policy server")
        except Exception as exc:
            self._startup_error = exc
            self._ready_event.set()
            return

        self._ready_event.set()
        try:
            while not self._stop_event.is_set() or (self._drain_on_close and not self._requests.empty()):
                try:
                    _, _, request = self._requests.get(timeout=0.05)
                except queue.Empty:
                    continue

                metadata = request.scheduler_request
                with self._active_lock:
                    self._active_env_id = metadata.env_id
                started_wall_s = time.perf_counter()
                try:
                    action, _ = capture_vla_call(
                        policy="gr00t_async",
                        transport="zmq_msgpack",
                        host=self._capture_host,
                        port=self._capture_port,
                        request_payload=request.payload,
                        call=lambda: client.get_action(request.payload),
                        response_payload=lambda response: response[0],
                        extra={
                            "env_id": metadata.env_id,
                            "generation": metadata.generation,
                            "deadline_sim_time_s": metadata.deadline_sim_time_s,
                            "is_bootstrap": request.is_bootstrap,
                        },
                    )
                    error = None
                except Exception as exc:
                    action = None
                    error = exc
                finished_wall_s = time.perf_counter()
                with self._active_lock:
                    self._active_env_id = None
                self._results.put(
                    Gr00tWorkerResult(
                        request=request,
                        action=action,
                        inference_wall_s=finished_wall_s - started_wall_s,
                        queue_wait_wall_s=started_wall_s - request.enqueued_wall_s,
                        error=error,
                    )
                )
                self._requests.task_done()
        finally:
            self._close_client(client)

    @staticmethod
    def _close_client(client: Any) -> None:
        socket = getattr(client, "socket", None)
        context = getattr(client, "context", None)
        try:
            if socket is not None:
                socket.close(linger=0)
        finally:
            if context is not None:
                context.term()
