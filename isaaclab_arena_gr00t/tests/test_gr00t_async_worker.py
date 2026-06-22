# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-owner GR00T asynchronous inference worker."""

from __future__ import annotations

import threading

from isaaclab_arena.policy.action_scheduling import AsyncChunkRequest
from isaaclab_arena_gr00t.policy.gr00t_async_worker import Gr00tAsyncInferenceWorker, Gr00tWorkerRequest


class _FakeClient:
    def __init__(self) -> None:
        self.owner_thread_id = threading.get_ident()
        self.call_thread_ids: list[int] = []
        self.payload_order: list[str] = []

    def ping(self) -> bool:
        return True

    def get_action(self, payload):
        self.call_thread_ids.append(threading.get_ident())
        self.payload_order.append(payload["name"])
        if payload["name"] == "error":
            raise RuntimeError("inference failed")
        return {"action": payload["name"]}, None


def _request(env_id: int, deadline: float, sequence: int, name: str) -> Gr00tWorkerRequest:
    return Gr00tWorkerRequest(
        scheduler_request=AsyncChunkRequest(
            env_id=env_id,
            generation=0,
            submit_sim_time_s=0.0,
            deadline_sim_time_s=deadline,
            sequence=sequence,
        ),
        payload={"name": name},
    )


def test_worker_owns_client_and_processes_earliest_deadline_first() -> None:
    created: list[_FakeClient] = []

    def client_factory() -> _FakeClient:
        client = _FakeClient()
        created.append(client)
        return client

    worker = Gr00tAsyncInferenceWorker(client_factory=client_factory, autostart=False)
    worker.submit(_request(env_id=1, deadline=1.0, sequence=1, name="late"))
    worker.submit(_request(env_id=0, deadline=0.5, sequence=2, name="early"))
    worker.start()

    results = worker.wait_for_results(count=2, timeout_s=2.0)
    worker.close()

    assert [result.request.payload["name"] for result in results] == ["early", "late"]
    assert all(result.error is None for result in results)
    assert len(created) == 1
    assert created[0].owner_thread_id != threading.get_ident()
    assert set(created[0].call_thread_ids) == {created[0].owner_thread_id}
    assert worker.is_alive is False


def test_worker_returns_request_error_without_dying() -> None:
    worker = Gr00tAsyncInferenceWorker(client_factory=_FakeClient, autostart=False)
    worker.submit(_request(env_id=0, deadline=0.5, sequence=0, name="error"))
    worker.submit(_request(env_id=1, deadline=1.0, sequence=1, name="ok"))
    worker.start()

    results = worker.wait_for_results(count=2, timeout_s=2.0)
    worker.close()

    assert isinstance(results[0].error, RuntimeError)
    assert results[1].action == {"action": "ok"}


def test_close_can_drain_submitted_requests() -> None:
    worker = Gr00tAsyncInferenceWorker(client_factory=_FakeClient, autostart=False)
    worker.submit(_request(env_id=0, deadline=0.5, sequence=0, name="first"))
    worker.submit(_request(env_id=1, deadline=1.0, sequence=1, name="second"))
    worker.start()

    worker.close(drain=True)
    results = worker.poll()

    assert [result.action for result in results] == [{"action": "first"}, {"action": "second"}]
    assert worker.queue_depth == 0
