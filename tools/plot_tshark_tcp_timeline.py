#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render tshark TCP field CSV as a standalone HTML traffic timeline."""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_COLOR = "#2563eb"
DOWNSTREAM_COLOR = "#16a34a"
PAYLOAD_COLOR = "#c2410c"
AXIS_COLOR = "#334155"
GRID_COLOR = "#d8dee9"
TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#64748b"


@dataclass(frozen=True)
class Packet:
    time_s: float
    src_port: int
    dst_port: int
    frame_bytes: int
    payload_bytes: int


@dataclass
class Bucket:
    start_s: float
    upstream_frame_bytes: int = 0
    downstream_frame_bytes: int = 0
    upstream_payload_bytes: int = 0
    downstream_payload_bytes: int = 0
    frames: int = 0

    @property
    def total_frame_bytes(self) -> int:
        return self.upstream_frame_bytes + self.downstream_frame_bytes

    @property
    def total_payload_bytes(self) -> int:
        return self.upstream_payload_bytes + self.downstream_payload_bytes


def load_packets(path: str | Path) -> list[Packet]:
    """Load the CSV exported by tshark -T fields."""

    packets: list[Packet] = []
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                packets.append(
                    Packet(
                        time_s=float(row.get("frame.time_relative") or 0.0),
                        src_port=int(row.get("tcp.srcport") or 0),
                        dst_port=int(row.get("tcp.dstport") or 0),
                        frame_bytes=int(row.get("frame.len") or 0),
                        payload_bytes=int(row.get("tcp.len") or 0),
                    )
                )
            except ValueError:
                continue
    packets.sort(key=lambda packet: packet.time_s)
    return packets


def _open_csv(path: str | Path):
    csv_path = Path(path)
    raw = csv_path.read_bytes()[:4]
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return csv_path.open(newline="", encoding="utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return csv_path.open(newline="", encoding="utf-8-sig")
    return csv_path.open(newline="", encoding="utf-8")


def write_timeline_html(
    input_path: str | Path,
    output_path: str | Path,
    server_port: int,
    title: str = "TCP traffic timeline",
    bin_s: float | None = None,
    mode: str = "bucket",
    packet_bar_width: float = 2.5,
    width: int | None = None,
    include_ack: bool = False,
) -> None:
    """Read tshark CSV and write a standalone HTML timeline."""

    packets = load_packets(input_path)
    assert len(packets) > 0, f"No TCP packets found in {input_path}"
    html_text = build_timeline_html(
        packets,
        server_port=server_port,
        title=title,
        bin_s=bin_s,
        mode=mode,
        packet_bar_width=packet_bar_width,
        width=width,
        include_ack=include_ack,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_text, encoding="utf-8")


def build_timeline_html(
    packets: list[Packet],
    server_port: int,
    title: str = "TCP traffic timeline",
    bin_s: float | None = None,
    mode: str = "bucket",
    packet_bar_width: float = 2.5,
    width: int | None = None,
    include_ack: bool = False,
) -> str:
    """Build a standalone HTML document containing a TCP traffic chart."""

    assert mode in {"bucket", "packets"}, f"Unsupported timeline mode: {mode}"
    packets = _filter_packets(packets, include_ack=include_ack)
    assert len(packets) > 0, "No TCP payload packets found. Use --include-ack to render ACK-only packets."
    if mode == "packets":
        return _build_packet_timeline_html(
            packets,
            server_port=server_port,
            title=title,
            packet_bar_width=packet_bar_width,
            width=width,
            include_ack=include_ack,
        )

    return _build_bucket_timeline_html(
        packets,
        server_port=server_port,
        title=title,
        bin_s=bin_s,
        width=width,
        include_ack=include_ack,
    )


def _build_bucket_timeline_html(
    packets: list[Packet],
    server_port: int,
    title: str = "TCP traffic timeline",
    bin_s: float | None = None,
    width: int | None = None,
    include_ack: bool = False,
) -> str:
    """Build a standalone HTML document containing a binned TCP traffic chart."""

    duration_s = max(packet.time_s for packet in packets) - min(packet.time_s for packet in packets)
    bucket_s = bin_s if bin_s is not None else _default_bin_s(duration_s)
    buckets = _bucket_packets(packets, server_port=server_port, bucket_s=bucket_s)
    max_upstream_frame_bytes = max(max(bucket.upstream_frame_bytes, 1) for bucket in buckets)
    max_downstream_frame_bytes = max(max(bucket.downstream_frame_bytes, 1) for bucket in buckets)

    width = width if width is not None else max(960, 150 + 22 * len(buckets))
    width = max(960, width)
    height = 640
    margin_left = 88
    margin_right = 42
    margin_top = 78
    margin_bottom = 112
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    lane_gap = 54
    lane_height = (plot_height - lane_gap) / 2
    upstream_zero = margin_top + lane_height
    downstream_top = upstream_zero + lane_gap
    downstream_zero = downstream_top + lane_height
    bar_width = max(3, min(16, plot_width / max(len(buckets) * 1.35, 1)))

    def x_for(index: int) -> float:
        if len(buckets) == 1:
            return margin_left + plot_width / 2
        return margin_left + index * plot_width / (len(buckets) - 1)

    def y_for_upstream(value: float) -> float:
        return upstream_zero - (value / max_upstream_frame_bytes) * lane_height

    def y_for_downstream(value: float) -> float:
        return downstream_zero - (value / max_downstream_frame_bytes) * lane_height

    bars = []
    label_stride = max(1, math.ceil(len(buckets) / 12))
    for index, bucket in enumerate(buckets):
        x = x_for(index)
        upstream_y = y_for_upstream(bucket.upstream_frame_bytes)
        upstream_height = upstream_zero - upstream_y
        downstream_y = y_for_downstream(bucket.downstream_frame_bytes)
        downstream_height = downstream_zero - downstream_y
        tooltip = html.escape(_bucket_tooltip(bucket, bucket_s))
        maybe_tick = ""
        if index % label_stride == 0 or index == len(buckets) - 1:
            maybe_tick = (
                f'<text x="{x:.2f}" y="{downstream_zero + 20}" text-anchor="middle" class="tick-label">'
                f'{html.escape(_format_seconds(bucket.start_s))}</text>'
            )
        bars.append(
            f'<g><title>{tooltip}</title>'
            f'<rect x="{x - bar_width / 2:.2f}" y="{upstream_y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(upstream_height, 0):.2f}" fill="{UPSTREAM_COLOR}" />'
            f'<rect x="{x - bar_width / 2:.2f}" y="{downstream_y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(downstream_height, 0):.2f}" fill="{DOWNSTREAM_COLOR}" />'
            f"{maybe_tick}</g>"
        )

    upstream_grid = _lane_grid(
        y_for_value=y_for_upstream,
        max_value=max_upstream_frame_bytes,
        margin_left=margin_left,
        plot_right=width - margin_right,
    )
    downstream_grid = _lane_grid(
        y_for_value=y_for_downstream,
        max_value=max_downstream_frame_bytes,
        margin_left=margin_left,
        plot_right=width - margin_right,
    )

    summary = _summary(packets, buckets, server_port=server_port, bucket_s=bucket_s, include_ack=include_ack)
    ack_note = _ack_note(include_ack)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: {TEXT_COLOR}; background: #f8fafc; }}
    main {{ max-width: {width + 80}px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    .subtitle {{ color: {MUTED_TEXT_COLOR}; margin-bottom: 18px; }}
    .chart-wrap {{ overflow-x: auto; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }}
    svg {{ min-width: {width}px; display: block; }}
    .axis {{ stroke: {AXIS_COLOR}; stroke-width: 1.4; }}
    .grid {{ stroke: {GRID_COLOR}; stroke-width: 1; }}
    .tick-label {{ fill: {MUTED_TEXT_COLOR}; font-size: 12px; }}
    .payload-tick {{ fill: {PAYLOAD_COLOR}; font-size: 12px; }}
    .axis-label {{ fill: {AXIS_COLOR}; font-size: 13px; font-weight: 700; }}
    .lane-label {{ font-size: 13px; font-weight: 700; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0; color: {AXIS_COLOR}; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
    .legend .note {{ color: {MUTED_TEXT_COLOR}; }}
    .swatch {{ width: 13px; height: 13px; border-radius: 3px; display: inline-block; }}
  </style>
</head>
<body>
<main>
  <h1>{escaped_title}</h1>
  <div class="subtitle">{html.escape(summary)}</div>
  <div class="legend">
    <span><i class="swatch" style="background:{UPSTREAM_COLOR}"></i>Client to server</span>
    <span><i class="swatch" style="background:{DOWNSTREAM_COLOR}"></i>Server to client</span>
    <span class="note">Each lane uses its own vertical scale so small replies stay visible. {ack_note}</span>
  </div>
  <div class="chart-wrap">
    <svg width="{width}" height="{height}" role="img" aria-label="{escaped_title}">
      <text x="{margin_left}" y="30" class="axis-label">Wireshark TCP bytes per time bucket</text>
      <text x="{margin_left}" y="{margin_top - 14:.2f}" class="lane-label" fill="{UPSTREAM_COLOR}">Client to server, max bucket {_format_bytes(max_upstream_frame_bytes)}</text>
      <text x="{margin_left}" y="{downstream_top - 14:.2f}" class="lane-label" fill="{DOWNSTREAM_COLOR}">Server to client, max bucket {_format_bytes(max_downstream_frame_bytes)}</text>
      {upstream_grid}
      {downstream_grid}
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{upstream_zero}" class="axis" />
      <line x1="{margin_left}" y1="{upstream_zero}" x2="{width - margin_right}" y2="{upstream_zero}" class="axis" />
      <line x1="{margin_left}" y1="{downstream_top}" x2="{margin_left}" y2="{downstream_zero}" class="axis" />
      <line x1="{margin_left}" y1="{downstream_zero}" x2="{width - margin_right}" y2="{downstream_zero}" class="axis" />
      {"".join(bars)}
      <text x="{margin_left + plot_width / 2:.2f}" y="{height - 34}" text-anchor="middle" class="axis-label">Time since first packet</text>
      <text x="22" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 22,{margin_top + plot_height / 2:.2f})" text-anchor="middle" class="axis-label">Frame bytes per lane</text>
    </svg>
  </div>
</main>
</body>
</html>
"""


def _build_packet_timeline_html(
    packets: list[Packet],
    server_port: int,
    title: str = "TCP packet timeline",
    packet_bar_width: float = 2.5,
    width: int | None = None,
    include_ack: bool = False,
) -> str:
    """Build a standalone HTML document with one vertical bar per TCP packet."""

    start_s = min(packet.time_s for packet in packets)
    duration_s = max(packet.time_s for packet in packets) - start_s
    upstream_packets = [packet for packet in packets if packet.dst_port == server_port]
    downstream_packets = [packet for packet in packets if packet.src_port == server_port]
    max_upstream_frame_bytes = max([packet.frame_bytes for packet in upstream_packets] + [1])
    max_downstream_frame_bytes = max([packet.frame_bytes for packet in downstream_packets] + [1])

    bar_width = max(0.5, packet_bar_width)
    width = width if width is not None else int(150 + max(len(packets), 1) * (bar_width + 1.0))
    width = max(1200, width)
    height = 640
    margin_left = 88
    margin_right = 42
    margin_top = 78
    margin_bottom = 112
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    lane_gap = 54
    lane_height = (plot_height - lane_gap) / 2
    upstream_zero = margin_top + lane_height
    downstream_top = upstream_zero + lane_gap
    downstream_zero = downstream_top + lane_height

    def x_for(packet: Packet) -> float:
        if duration_s <= 1e-9:
            return margin_left + plot_width / 2
        return margin_left + ((packet.time_s - start_s) / duration_s) * plot_width

    def y_for_upstream(value: float) -> float:
        return upstream_zero - (value / max_upstream_frame_bytes) * lane_height

    def y_for_downstream(value: float) -> float:
        return downstream_zero - (value / max_downstream_frame_bytes) * lane_height

    bars = []
    for index, packet in enumerate(packets):
        is_upstream = packet.dst_port == server_port
        is_downstream = packet.src_port == server_port
        if not is_upstream and not is_downstream:
            continue
        x = x_for(packet)
        if is_upstream:
            y = y_for_upstream(packet.frame_bytes)
            zero = upstream_zero
            color = UPSTREAM_COLOR
            direction = "client to server"
        else:
            y = y_for_downstream(packet.frame_bytes)
            zero = downstream_zero
            color = DOWNSTREAM_COLOR
            direction = "server to client"
        tooltip = html.escape(_packet_tooltip(packet, index=index, start_s=start_s, direction=direction))
        bars.append(
            f'<g><title>{tooltip}</title>'
            f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(zero - y, 0):.2f}" fill="{color}" /></g>'
        )

    upstream_grid = _lane_grid(
        y_for_value=y_for_upstream,
        max_value=max_upstream_frame_bytes,
        margin_left=margin_left,
        plot_right=width - margin_right,
    )
    downstream_grid = _lane_grid(
        y_for_value=y_for_downstream,
        max_value=max_downstream_frame_bytes,
        margin_left=margin_left,
        plot_right=width - margin_right,
    )
    time_ticks = _time_ticks(
        start_s=0.0,
        duration_s=duration_s,
        margin_left=margin_left,
        plot_width=plot_width,
        y=downstream_zero + 20,
    )

    upstream = sum(packet.frame_bytes for packet in upstream_packets)
    downstream = sum(packet.frame_bytes for packet in downstream_packets)
    payload = sum(packet.payload_bytes for packet in packets)
    summary = (
        f"{len(packets)} packets over {_format_seconds(duration_s)}; server port {server_port}; "
        f"packet bars {bar_width:g}px; client to server {_format_bytes(upstream)}; "
        f"server to client {_format_bytes(downstream)}; TCP payload {_format_bytes(payload)}"
    )
    ack_note = _ack_note(include_ack)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: {TEXT_COLOR}; background: #f8fafc; }}
    main {{ max-width: {width + 80}px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    .subtitle {{ color: {MUTED_TEXT_COLOR}; margin-bottom: 18px; }}
    .chart-wrap {{ overflow-x: auto; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }}
    svg {{ min-width: {width}px; display: block; }}
    .axis {{ stroke: {AXIS_COLOR}; stroke-width: 1.4; }}
    .grid {{ stroke: {GRID_COLOR}; stroke-width: 1; }}
    .tick-label {{ fill: {MUTED_TEXT_COLOR}; font-size: 12px; }}
    .axis-label {{ fill: {AXIS_COLOR}; font-size: 13px; font-weight: 700; }}
    .lane-label {{ font-size: 13px; font-weight: 700; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0; color: {AXIS_COLOR}; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
    .legend .note {{ color: {MUTED_TEXT_COLOR}; }}
    .swatch {{ width: 13px; height: 13px; border-radius: 3px; display: inline-block; }}
  </style>
</head>
<body>
<main>
  <h1>{escaped_title}</h1>
  <div class="subtitle">{html.escape(summary)}</div>
  <div class="legend">
    <span><i class="swatch" style="background:{UPSTREAM_COLOR}"></i>Client to server</span>
    <span><i class="swatch" style="background:{DOWNSTREAM_COLOR}"></i>Server to client</span>
    <span class="note">Packet mode: every vertical bar is one TCP packet; each lane uses its own vertical scale. {ack_note}</span>
  </div>
  <div class="chart-wrap">
    <svg width="{width}" height="{height}" role="img" aria-label="{escaped_title}">
      <text x="{margin_left}" y="30" class="axis-label">Wireshark TCP bytes per packet</text>
      <text x="{margin_left}" y="{margin_top - 14:.2f}" class="lane-label" fill="{UPSTREAM_COLOR}">Client to server, max packet {_format_bytes(max_upstream_frame_bytes)}</text>
      <text x="{margin_left}" y="{downstream_top - 14:.2f}" class="lane-label" fill="{DOWNSTREAM_COLOR}">Server to client, max packet {_format_bytes(max_downstream_frame_bytes)}</text>
      {upstream_grid}
      {downstream_grid}
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{upstream_zero}" class="axis" />
      <line x1="{margin_left}" y1="{upstream_zero}" x2="{width - margin_right}" y2="{upstream_zero}" class="axis" />
      <line x1="{margin_left}" y1="{downstream_top}" x2="{margin_left}" y2="{downstream_zero}" class="axis" />
      <line x1="{margin_left}" y1="{downstream_zero}" x2="{width - margin_right}" y2="{downstream_zero}" class="axis" />
      {"".join(bars)}
      {time_ticks}
      <text x="{margin_left + plot_width / 2:.2f}" y="{height - 34}" text-anchor="middle" class="axis-label">Time since first packet</text>
      <text x="22" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 22,{margin_top + plot_height / 2:.2f})" text-anchor="middle" class="axis-label">Frame bytes per packet</text>
    </svg>
  </div>
</main>
</body>
</html>
"""


def _bucket_packets(packets: list[Packet], server_port: int, bucket_s: float) -> list[Bucket]:
    start_s = min(packet.time_s for packet in packets)
    end_s = max(packet.time_s for packet in packets)
    bucket_count = max(1, int(math.floor((end_s - start_s) / bucket_s)) + 1)
    buckets = [Bucket(start_s=index * bucket_s) for index in range(bucket_count)]
    for packet in packets:
        index = min(bucket_count - 1, int(math.floor((packet.time_s - start_s) / bucket_s)))
        bucket = buckets[index]
        bucket.frames += 1
        if packet.dst_port == server_port:
            bucket.upstream_frame_bytes += packet.frame_bytes
            bucket.upstream_payload_bytes += packet.payload_bytes
        elif packet.src_port == server_port:
            bucket.downstream_frame_bytes += packet.frame_bytes
            bucket.downstream_payload_bytes += packet.payload_bytes
    return buckets


def _filter_packets(packets: list[Packet], include_ack: bool) -> list[Packet]:
    if include_ack:
        return packets
    return [packet for packet in packets if packet.payload_bytes > 0]


def _lane_grid(y_for_value, max_value: float, margin_left: int, plot_right: int) -> str:
    return "\n".join(
        f'<line x1="{margin_left}" y1="{y_for_value(tick):.2f}" x2="{plot_right}" '
        f'y2="{y_for_value(tick):.2f}" class="grid" />'
        f'<text x="{margin_left - 12}" y="{y_for_value(tick) + 4:.2f}" text-anchor="end" class="tick-label">'
        f'{html.escape(_format_bytes(tick))}</text>'
        for tick in _nice_ticks(max_value, count=4)
    )


def _bucket_tooltip(bucket: Bucket, bucket_s: float) -> str:
    return "\n".join(
        [
            f"{_format_seconds(bucket.start_s)} - {_format_seconds(bucket.start_s + bucket_s)}",
            f"frames: {bucket.frames}",
            f"client to server frames: {_format_bytes(bucket.upstream_frame_bytes)}",
            f"server to client frames: {_format_bytes(bucket.downstream_frame_bytes)}",
            f"tcp payload: {_format_bytes(bucket.total_payload_bytes)}",
        ]
    )


def _packet_tooltip(packet: Packet, index: int, start_s: float, direction: str) -> str:
    return "\n".join(
        [
            f"packet: {index + 1}",
            f"time: {_format_seconds(packet.time_s - start_s)}",
            f"direction: {direction}",
            f"src port: {packet.src_port}",
            f"dst port: {packet.dst_port}",
            f"frame: {_format_bytes(packet.frame_bytes)}",
            f"tcp payload: {_format_bytes(packet.payload_bytes)}",
        ]
    )


def _time_ticks(start_s: float, duration_s: float, margin_left: int, plot_width: int, y: float) -> str:
    if duration_s <= 1e-9:
        return ""
    tick_count = 10
    ticks = []
    for index in range(tick_count + 1):
        value = start_s + duration_s * index / tick_count
        x = margin_left + plot_width * index / tick_count
        ticks.append(
            f'<line x1="{x:.2f}" y1="{y - 14:.2f}" x2="{x:.2f}" y2="{y - 8:.2f}" class="axis" />'
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" class="tick-label">'
            f'{html.escape(_format_seconds(value))}</text>'
        )
    return "\n".join(ticks)


def _summary(
    packets: list[Packet],
    buckets: list[Bucket],
    server_port: int,
    bucket_s: float,
    include_ack: bool,
) -> str:
    duration_s = max(packet.time_s for packet in packets) - min(packet.time_s for packet in packets)
    upstream = sum(bucket.upstream_frame_bytes for bucket in buckets)
    downstream = sum(bucket.downstream_frame_bytes for bucket in buckets)
    payload = sum(bucket.total_payload_bytes for bucket in buckets)
    return (
        f"{len(packets)} packets over {_format_seconds(duration_s)}; server port {server_port}; "
        f"bucket {_format_seconds(bucket_s)}; client to server {_format_bytes(upstream)}; "
        f"server to client {_format_bytes(downstream)}; TCP payload {_format_bytes(payload)}"
    )


def _ack_note(include_ack: bool) -> str:
    if include_ack:
        return "ACK-only packets are included."
    return "ACK-only packets where tcp.len = 0 are excluded."


def _default_bin_s(duration_s: float) -> float:
    if duration_s <= 2.0:
        return 0.05
    if duration_s <= 20.0:
        return 0.2
    return 1.0


def _nice_ticks(max_value: float, count: int) -> list[float]:
    if max_value <= 0:
        return [0.0]
    raw_step = max_value / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice_step = magnitude
    elif normalized <= 2:
        nice_step = 2 * magnitude
    elif normalized <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    top = math.ceil(max_value / nice_step) * nice_step
    ticks = []
    value = 0.0
    while value <= top + nice_step * 0.5:
        ticks.append(value)
        value += nice_step
    return ticks


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    amount = float(value)
    unit_index = 0
    while abs(amount) >= 1000.0 and unit_index < len(units) - 1:
        amount /= 1000.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(round(amount))} {units[unit_index]}"
    return f"{amount:.1f} {units[unit_index]}"


def _format_seconds(value: float) -> str:
    if abs(value) < 1.0:
        return f"{value * 1000.0:.0f} ms"
    return f"{value:.2f} s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to tshark CSV field export")
    parser.add_argument("--output", required=True, help="Path to write the standalone HTML timeline")
    parser.add_argument("--server-port", required=True, type=int, help="Server TCP port used to infer direction")
    parser.add_argument("--title", default="TCP traffic timeline", help="Chart title")
    parser.add_argument("--bin-s", type=float, default=None, help="Optional bucket size in seconds")
    parser.add_argument("--mode", choices=["bucket", "packets"], default="bucket", help="Timeline rendering mode")
    parser.add_argument("--packet-bar-width", type=float, default=2.5, help="Packet-mode bar width in pixels")
    parser.add_argument("--width", type=int, default=None, help="Optional fixed SVG width in pixels")
    parser.add_argument("--include-ack", action="store_true", help="Include ACK-only packets where tcp.len is 0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_timeline_html(
        args.input,
        args.output,
        server_port=args.server_port,
        title=args.title,
        bin_s=args.bin_s,
        mode=args.mode,
        packet_bar_width=args.packet_bar_width,
        width=args.width,
        include_ack=args.include_ack,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
