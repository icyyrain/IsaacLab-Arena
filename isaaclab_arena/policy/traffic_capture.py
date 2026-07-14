# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Application-level VLA traffic capture.

The capture intentionally records metadata only: shapes, dtypes, estimated byte
counts, timing, and endpoint information. It does not persist raw images,
prompts, states, or actions.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

CAPTURE_PATH_ENV = "VLA_TRAFFIC_CAPTURE_PATH"


def capture_vla_call(
    *,
    policy: str,
    transport: str,
    host: str,
    port: int,
    request_payload: Any,
    call: Callable[[], Any],
    endpoint: str = "get_action",
    extra: dict[str, Any] | None = None,
    response_payload: Callable[[Any], Any] | None = None,
) -> Any:
    """Run a VLA remote call and append one JSONL metadata record when enabled."""

    capture_path = os.getenv(CAPTURE_PATH_ENV)
    if not capture_path:
        return call()

    started_wall_s = time.perf_counter()
    started_unix_s = time.time()
    status = "ok"
    error_type = None
    error_message = None
    result = None
    response_for_summary = None
    try:
        result = call()
        response_for_summary = response_payload(result) if response_payload is not None else result
        return result
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        raise
    finally:
        finished_wall_s = time.perf_counter()
        record = {
            "event": "vla_remote_call",
            "schema_version": 1,
            "timestamp_unix_s": round(started_unix_s, 9),
            "policy": policy,
            "transport": transport,
            "host": host,
            "port": int(port),
            "endpoint": endpoint,
            "status": status,
            "latency_s": round(finished_wall_s - started_wall_s, 9),
            "request_bytes": estimate_payload_bytes(request_payload),
            "response_bytes": estimate_payload_bytes(response_for_summary),
            "request": summarize_payload(request_payload),
            "response": summarize_payload(response_for_summary),
        }
        if extra:
            record["extra"] = extra
        if error_type is not None:
            record["error_type"] = error_type
            record["error_message"] = error_message
        _append_jsonl(Path(capture_path), record)


def summarize_payload(payload: Any) -> Any:
    """Return a JSON-safe metadata summary for payload values."""

    if payload is None:
        return None
    if isinstance(payload, np.ndarray):
        return {
            "shape": list(payload.shape),
            "dtype": str(payload.dtype),
            "bytes": int(payload.nbytes),
        }
    tensor_summary = _summarize_torch_tensor(payload)
    if tensor_summary is not None:
        return tensor_summary
    if isinstance(payload, dict):
        return {str(key): summarize_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        if _is_scalar_sequence(payload):
            return {
                "type": type(payload).__name__,
                "shape": _nested_sequence_shape(payload),
                "length": len(payload),
                "bytes": estimate_payload_bytes(payload),
            }
        return [summarize_payload(value) for value in payload]
    if isinstance(payload, str):
        return {"type": "str", "length": len(payload), "bytes": len(payload.encode("utf-8"))}
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return {"type": type(payload).__name__, "bytes": len(payload)}
    if isinstance(payload, (bool, int, float)):
        return {"type": type(payload).__name__, "bytes": estimate_payload_bytes(payload)}
    return {"type": type(payload).__name__, "bytes": estimate_payload_bytes(payload)}


def estimate_payload_bytes(payload: Any) -> int:
    """Estimate in-memory payload bytes for transport-level trend analysis."""

    if payload is None:
        return 0
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    tensor_bytes = _torch_tensor_bytes(payload)
    if tensor_bytes is not None:
        return tensor_bytes
    if isinstance(payload, dict):
        return sum(estimate_payload_bytes(key) + estimate_payload_bytes(value) for key, value in payload.items())
    if isinstance(payload, (list, tuple)):
        return sum(estimate_payload_bytes(value) for value in payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    if isinstance(payload, bool):
        return 1
    if isinstance(payload, int):
        return 8
    if isinstance(payload, float):
        return 8
    return 0


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _summarize_torch_tensor(value: Any) -> dict[str, Any] | None:
    if not _is_torch_tensor(value):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "bytes": _torch_tensor_bytes(value),
    }


def _torch_tensor_bytes(value: Any) -> int | None:
    if not _is_torch_tensor(value):
        return None
    return int(value.numel() * value.element_size())


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "numel") and hasattr(value, "element_size") and hasattr(value, "device")


def _is_scalar_sequence(value: list[Any] | tuple[Any, ...]) -> bool:
    if len(value) == 0:
        return True
    return all(isinstance(item, (str, int, float, bool, list, tuple)) for item in value)


def _nested_sequence_shape(value: Any) -> list[int]:
    shape = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape

