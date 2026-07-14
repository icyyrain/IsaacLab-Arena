# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np


def _load_traffic_capture():
    module_path = Path(__file__).resolve().parents[1] / "policy" / "traffic_capture.py"
    spec = importlib.util.spec_from_file_location("traffic_capture_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


capture_vla_call = _load_traffic_capture().capture_vla_call


def test_capture_vla_call_writes_shape_and_size_without_raw_payload(tmp_path, monkeypatch):
    capture_path = tmp_path / "traffic.jsonl"
    monkeypatch.setenv("VLA_TRAFFIC_CAPTURE_PATH", str(capture_path))
    request = {
        "image": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "prompt": "pick up the block",
    }
    response = {"actions": np.zeros((2, 16, 8), dtype=np.float32)}

    result = capture_vla_call(
        policy="example",
        transport="loopback",
        host="127.0.0.1",
        port=1234,
        request_payload=request,
        call=lambda: response,
    )

    assert result is response
    rows = [json.loads(line) for line in capture_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["policy"] == "example"
    assert row["transport"] == "loopback"
    assert row["status"] == "ok"
    assert row["request_bytes"] == 96 + len("image") + len("prompt") + len("pick up the block".encode("utf-8"))
    assert row["response_bytes"] == len("actions") + 2 * 16 * 8 * 4
    assert row["request"]["image"] == {"shape": [2, 4, 4, 3], "dtype": "uint8", "bytes": 96}
    assert row["request"]["prompt"] == {"type": "str", "length": 17, "bytes": 17}
    assert row["response"]["actions"]["shape"] == [2, 16, 8]
    assert "pick up the block" not in capture_path.read_text()


def test_capture_vla_call_records_error_and_reraises(tmp_path, monkeypatch):
    capture_path = tmp_path / "traffic.jsonl"
    monkeypatch.setenv("VLA_TRAFFIC_CAPTURE_PATH", str(capture_path))

    try:
        capture_vla_call(
            policy="example",
            transport="loopback",
            host="127.0.0.1",
            port=1234,
            request_payload={"state": np.zeros((1, 3), dtype=np.float32)},
            call=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("capture_vla_call should re-raise call failures")

    row = json.loads(capture_path.read_text().strip())
    assert row["status"] == "error"
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "boom"
    assert row["response_bytes"] == 0
