# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_plot_tool():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "plot_vla_traffic.py"
    spec = importlib.util.spec_from_file_location("plot_vla_traffic_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_tshark_plot_tool():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "plot_tshark_tcp_timeline.py"
    spec = importlib.util.spec_from_file_location("plot_tshark_tcp_timeline_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_vla_traffic_generates_timeline_html(tmp_path):
    plot_tool = _load_plot_tool()
    input_path = tmp_path / "traffic.jsonl"
    output_path = tmp_path / "traffic.html"
    rows = [
        {
            "timestamp_unix_s": 100.0,
            "policy": "gr00t",
            "transport": "zmq_msgpack",
            "host": "127.0.0.1",
            "port": 5555,
            "latency_s": 0.12,
            "request_bytes": 172800,
            "response_bytes": 896,
            "request": {"video": {"ego_view": {"shape": [1, 1, 180, 320, 3], "dtype": "uint8"}}},
            "response": {"right_arm": {"shape": [1, 32, 7], "dtype": "float32"}},
        },
        {
            "timestamp_unix_s": 100.5,
            "policy": "openpi",
            "transport": "websocket",
            "host": "127.0.0.1",
            "port": 8000,
            "latency_s": 0.05,
            "request_bytes": 301088,
            "response_bytes": 480,
            "request": {
                "observation/exterior_image_1_left": {"shape": [224, 224, 3], "dtype": "uint8"},
                "observation/wrist_image_left": {"shape": [224, 224, 3], "dtype": "uint8"},
            },
            "response": {"actions": {"shape": [15, 8], "dtype": "float32"}},
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    plot_tool.write_timeline_html(input_path, output_path, title="VLA test traffic")

    html = output_path.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "VLA test traffic" in html
    assert "Request bytes" in html
    assert "Response bytes" in html
    assert "Latency" in html
    assert "gr00t" in html
    assert "openpi" in html
    assert "ego_view: [1, 1, 180, 320, 3]" in html
    assert "actions: [15, 8]" in html
    assert "301.1 KB" in html


def test_plot_tshark_tcp_timeline_accepts_windows_csv(tmp_path):
    plot_tool = _load_tshark_plot_tool()
    input_path = tmp_path / "packets.csv"
    output_path = tmp_path / "packets.html"
    input_path.write_text(
        "\n".join(
            [
                "frame.time_relative,ip.src,tcp.srcport,ip.dst,tcp.dstport,frame.len,tcp.len,tcp.stream",
                "0.000000000,127.0.0.1,50000,127.0.0.1,8000,100,56,0",
                "0.100000000,127.0.0.1,8000,127.0.0.1,50000,80,36,0",
            ]
        )
        + "\n",
        encoding="utf-16",
    )

    plot_tool.write_timeline_html(input_path, output_path, server_port=8000, title="TCP test traffic", bin_s=1.0)

    html = output_path.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "TCP test traffic" in html
    assert "Client to server, max bucket" in html
    assert "Server to client, max bucket" in html
    assert "Each lane uses its own vertical scale" in html
    assert "2 packets" in html
    assert "TCP payload 92 B" in html


def test_plot_tshark_tcp_timeline_packet_mode(tmp_path):
    plot_tool = _load_tshark_plot_tool()
    input_path = tmp_path / "packets.csv"
    output_path = tmp_path / "packets.html"
    input_path.write_text(
        "\n".join(
            [
                "frame.time_relative,tcp.srcport,tcp.dstport,frame.len,tcp.len",
                "0.000000000,50000,5555,100,56",
                "0.010000000,5555,50000,80,36",
                "0.250000000,50000,5555,120,76",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plot_tool.write_timeline_html(
        input_path,
        output_path,
        server_port=5555,
        title="Packet mode TCP",
        mode="packets",
        packet_bar_width=4.0,
        width=1600,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "Packet mode TCP" in html
    assert "Wireshark TCP bytes per packet" in html
    assert "Packet mode: every vertical bar is one TCP packet" in html
    assert "packet bars 4px" in html
    assert '<svg width="1600"' in html
    assert "3 packets" in html


def test_plot_tshark_tcp_timeline_excludes_ack_by_default(tmp_path):
    plot_tool = _load_tshark_plot_tool()
    input_path = tmp_path / "packets.csv"
    default_output_path = tmp_path / "packets_default.html"
    include_ack_output_path = tmp_path / "packets_with_ack.html"
    input_path.write_text(
        "\n".join(
            [
                "frame.time_relative,tcp.srcport,tcp.dstport,frame.len,tcp.len",
                "0.000000000,50000,5555,100,56",
                "0.010000000,5555,50000,56,0",
                "0.250000000,5555,50000,80,36",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plot_tool.write_timeline_html(
        input_path,
        default_output_path,
        server_port=5555,
        title="No ACK by default",
        mode="packets",
    )
    plot_tool.write_timeline_html(
        input_path,
        include_ack_output_path,
        server_port=5555,
        title="ACK included",
        mode="packets",
        include_ack=True,
    )

    default_html = default_output_path.read_text(encoding="utf-8")
    include_ack_html = include_ack_output_path.read_text(encoding="utf-8")
    assert "2 packets" in default_html
    assert "ACK-only packets where tcp.len = 0 are excluded." in default_html
    assert "3 packets" in include_ack_html
    assert "ACK-only packets are included." in include_ack_html
