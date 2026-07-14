#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render VLA traffic-capture JSONL as a self-contained HTML timeline."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


REQUEST_COLOR = "#2f6fed"
RESPONSE_COLOR = "#21a67a"
LATENCY_COLOR = "#c2410c"
AXIS_COLOR = "#334155"
GRID_COLOR = "#d8dee9"
TEXT_COLOR = "#111827"
MUTED_TEXT_COLOR = "#64748b"


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load traffic-capture JSONL rows sorted by capture timestamp."""

    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_number"] = line_number
        rows.append(row)
    rows.sort(key=lambda row: (float(row.get("timestamp_unix_s", 0.0)), int(row.get("_line_number", 0))))
    return rows


def write_timeline_html(input_path: str | Path, output_path: str | Path, title: str = "VLA traffic timeline") -> None:
    """Read JSONL traffic records and write a standalone HTML timeline."""

    events = load_events(input_path)
    assert len(events) > 0, f"No traffic events found in {input_path}"
    html_text = build_timeline_html(events, title=title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html_text, encoding="utf-8")


def build_timeline_html(events: list[dict[str, Any]], title: str = "VLA traffic timeline") -> str:
    """Build a standalone HTML document containing an SVG traffic chart."""

    prepared = _prepare_events(events)
    max_transfer_bytes = max(max(event["total_bytes"], 1) for event in prepared)
    max_latency_s = max(max(event["latency_s"], 0.0) for event in prepared) or 1.0

    width = max(960, 160 + 86 * len(prepared))
    height = 560
    margin_left = 88
    margin_right = 110
    margin_top = 78
    margin_bottom = 118
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_zero = margin_top + plot_height

    duration_s = max(event["time_s"] for event in prepared) - min(event["time_s"] for event in prepared)
    use_index_spacing = duration_s <= 1e-9

    def x_for(index: int, event: dict[str, Any]) -> float:
        if use_index_spacing:
            if len(prepared) == 1:
                return margin_left + plot_width / 2
            return margin_left + index * plot_width / (len(prepared) - 1)
        return margin_left + (event["time_s"] / duration_s) * plot_width

    def y_for_transfer(value: float) -> float:
        return y_zero - (value / max_transfer_bytes) * plot_height

    def y_for_latency(value: float) -> float:
        return y_zero - (value / max_latency_s) * plot_height

    bars = []
    latency_points = []
    row_cards = []
    bar_width = min(42, max(14, plot_width / max(len(prepared) * 2.4, 1)))
    for index, event in enumerate(prepared):
        x = x_for(index, event)
        request_height = y_zero - y_for_transfer(event["request_bytes"])
        response_height = y_zero - y_for_transfer(event["response_bytes"])
        request_y = y_zero - request_height
        response_y = request_y - response_height
        tooltip = html.escape(_event_tooltip(event))
        bars.append(
            f'<g class="event-bar"><title>{tooltip}</title>'
            f'<rect x="{x - bar_width / 2:.2f}" y="{request_y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(request_height, 1):.2f}" fill="{REQUEST_COLOR}" rx="2" />'
            f'<rect x="{x - bar_width / 2:.2f}" y="{response_y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(response_height, 1):.2f}" fill="{RESPONSE_COLOR}" rx="2" />'
            f'<text x="{x:.2f}" y="{y_zero + 20}" text-anchor="middle" class="tick-label">'
            f'{html.escape(_format_seconds(event["time_s"]))}</text></g>'
        )
        latency_points.append((x, y_for_latency(event["latency_s"])))
        row_cards.append(_event_card(event))

    latency_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in latency_points)
    latency_markers = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{LATENCY_COLOR}"><title>Latency: '
        f'{html.escape(_format_seconds(prepared[index]["latency_s"]))}</title></circle>'
        for index, (x, y) in enumerate(latency_points)
    )

    y_ticks = _nice_ticks(max_transfer_bytes, count=5)
    y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y_for_transfer(tick):.2f}" x2="{width - margin_right}" '
        f'y2="{y_for_transfer(tick):.2f}" class="grid" />'
        f'<text x="{margin_left - 12}" y="{y_for_transfer(tick) + 4:.2f}" text-anchor="end" class="tick-label">'
        f'{html.escape(_format_bytes(tick))}</text>'
        for tick in y_ticks
    )
    latency_ticks = _nice_ticks(max_latency_s, count=5)
    latency_axis = "\n".join(
        f'<text x="{width - margin_right + 12}" y="{y_for_latency(tick) + 4:.2f}" '
        f'text-anchor="start" class="latency-tick">{html.escape(_format_seconds(tick))}</text>'
        for tick in latency_ticks
    )

    escaped_title = html.escape(title)
    summary = _summary(prepared)
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
    .latency-tick {{ fill: {LATENCY_COLOR}; font-size: 12px; }}
    .axis-label {{ fill: {AXIS_COLOR}; font-size: 13px; font-weight: 700; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0; color: {AXIS_COLOR}; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
    .swatch {{ width: 13px; height: 13px; border-radius: 3px; display: inline-block; }}
    .events {{ margin-top: 18px; display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .event-card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }}
    .event-title {{ font-weight: 700; margin-bottom: 7px; }}
    .event-meta {{ color: {MUTED_TEXT_COLOR}; font-size: 13px; line-height: 1.45; }}
    code {{ background: #eef2f7; border-radius: 4px; padding: 1px 4px; }}
  </style>
</head>
<body>
<main>
  <h1>{escaped_title}</h1>
  <div class="subtitle">{html.escape(summary)}</div>
  <div class="legend">
    <span><i class="swatch" style="background:{REQUEST_COLOR}"></i>Request bytes</span>
    <span><i class="swatch" style="background:{RESPONSE_COLOR}"></i>Response bytes</span>
    <span><i class="swatch" style="background:{LATENCY_COLOR}"></i>Latency</span>
  </div>
  <div class="chart-wrap">
    <svg width="{width}" height="{height}" role="img" aria-label="{escaped_title}">
      <text x="{margin_left}" y="30" class="axis-label">Transfer per VLA inference call</text>
      <text x="{width - margin_right}" y="30" text-anchor="end" class="axis-label" fill="{LATENCY_COLOR}">Latency axis</text>
      {y_grid}
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{y_zero}" class="axis" />
      <line x1="{margin_left}" y1="{y_zero}" x2="{width - margin_right}" y2="{y_zero}" class="axis" />
      <line x1="{width - margin_right}" y1="{margin_top}" x2="{width - margin_right}" y2="{y_zero}" class="axis" />
      {latency_axis}
      {"".join(bars)}
      <polyline points="{latency_polyline}" fill="none" stroke="{LATENCY_COLOR}" stroke-width="2.4" />
      {latency_markers}
      <text x="{margin_left + plot_width / 2:.2f}" y="{height - 34}" text-anchor="middle" class="axis-label">Time since first captured event</text>
      <text x="22" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 22,{margin_top + plot_height / 2:.2f})" text-anchor="middle" class="axis-label">Transfer volume</text>
    </svg>
  </div>
  <section class="events">
    {"".join(row_cards)}
  </section>
</main>
</body>
</html>
"""


def _prepare_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = min(float(event.get("timestamp_unix_s", 0.0)) for event in events)
    prepared = []
    for index, event in enumerate(events, start=1):
        request_bytes = int(event.get("request_bytes", 0) or 0)
        response_bytes = int(event.get("response_bytes", 0) or 0)
        prepared.append({
            "index": index,
            "time_s": float(event.get("timestamp_unix_s", 0.0)) - start,
            "policy": str(event.get("policy", "unknown")),
            "transport": str(event.get("transport", "unknown")),
            "host": str(event.get("host", "unknown")),
            "port": int(event.get("port", 0) or 0),
            "latency_s": float(event.get("latency_s", 0.0) or 0.0),
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "total_bytes": request_bytes + response_bytes,
            "request": event.get("request", {}),
            "response": event.get("response", {}),
            "status": str(event.get("status", "unknown")),
        })
    return prepared


def _event_tooltip(event: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"{event['policy']} via {event['transport']} ({event['host']}:{event['port']})",
            f"t={_format_seconds(event['time_s'])}, latency={_format_seconds(event['latency_s'])}",
            f"request={_format_bytes(event['request_bytes'])}, response={_format_bytes(event['response_bytes'])}",
            _shape_summary(event),
        ]
    )


def _event_card(event: dict[str, Any]) -> str:
    return (
        '<article class="event-card">'
        f'<div class="event-title">#{event["index"]} {html.escape(event["policy"])} '
        f'<span class="event-meta">({html.escape(event["transport"])})</span></div>'
        '<div class="event-meta">'
        f'time <code>{html.escape(_format_seconds(event["time_s"]))}</code> · '
        f'latency <code>{html.escape(_format_seconds(event["latency_s"]))}</code><br />'
        f'request <code>{html.escape(_format_bytes(event["request_bytes"]))}</code> · '
        f'response <code>{html.escape(_format_bytes(event["response_bytes"]))}</code><br />'
        f'{html.escape(_shape_summary(event))}'
        "</div></article>"
    )


def _shape_summary(event: dict[str, Any]) -> str:
    shapes = []
    _collect_shapes(event.get("request"), shapes)
    _collect_shapes(event.get("response"), shapes)
    if not shapes:
        return "no array shapes recorded"
    return "; ".join(shapes[:4])


def _collect_shapes(value: Any, shapes: list[str], prefix: str = "") -> None:
    if isinstance(value, dict):
        if "shape" in value:
            label = prefix.rsplit(".", maxsplit=1)[-1] if prefix else "array"
            shapes.append(f"{label}: {value['shape']}")
            return
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_shapes(child, shapes, child_prefix)


def _summary(events: list[dict[str, Any]]) -> str:
    total_request = sum(event["request_bytes"] for event in events)
    total_response = sum(event["response_bytes"] for event in events)
    max_latency = max(event["latency_s"] for event in events)
    duration = max(event["time_s"] for event in events)
    return (
        f"{len(events)} calls over {_format_seconds(duration)} · "
        f"requests {_format_bytes(total_request)} · responses {_format_bytes(total_response)} · "
        f"max latency {_format_seconds(max_latency)}"
    )


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
    parser.add_argument("--input", required=True, help="Path to VLA traffic JSONL produced by VLA_TRAFFIC_CAPTURE_PATH")
    parser.add_argument("--output", required=True, help="Path to write the standalone HTML timeline")
    parser.add_argument("--title", default="VLA traffic timeline", help="Chart title")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_timeline_html(args.input, args.output, title=args.title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

