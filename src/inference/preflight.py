#!/usr/bin/env python3
"""Send one saved M4 observation through the real M7 inference socket."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from src.inference.action_audit import summarize_action_range
from src.inference.protocol import (
    PROTOCOL_VERSION,
    unix_request,
    validate_request_header,
    validate_response_header,
)
from src.inference.state import build_policy_state_from_m4_record


EXECUTE_STEPS = 10
TIMEOUT_MS = 10_000.0


def build_preflight_request(
    *,
    record: dict[str, Any],
    jpeg: bytes,
    episode_id: str,
    instruction: str,
    replan_index: int,
) -> dict[str, Any]:
    header = {
        "version": PROTOCOL_VERSION,
        "request_id": f"{episode_id}:{replan_index}",
        "episode_id": episode_id,
        "replan_index": replan_index,
        "instruction": instruction,
        "state": build_policy_state_from_m4_record(record),
        "image_encoding": "jpeg",
        "image_height": 512,
        "image_width": 512,
        "image_channels": 3,
        "jpeg_size": len(jpeg),
    }
    validate_request_header(header, payload_size=len(jpeg))
    return header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/go2_smolvla_m7.sock")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate_preflight_response(response: dict[str, Any], roundtrip_ms: float) -> dict[str, Any]:
    """Independently decide whether one complete policy response can enter M7."""
    checks = {"response_valid": True, "latency_within_timeout": True, "raw_output_range": True}
    action_summary = summarize_action_range([], execute_steps=EXECUTE_STEPS)
    try:
        validate_response_header(response)
        action_summary = summarize_action_range([response["actions"]], execute_steps=EXECUTE_STEPS)
        checks["raw_output_range"] = action_summary.raw_output_range_passed
    except (KeyError, TypeError, ValueError):
        checks["response_valid"] = False
        checks["raw_output_range"] = False
    try:
        if not math.isfinite(float(roundtrip_ms)) or float(roundtrip_ms) > TIMEOUT_MS:
            checks["latency_within_timeout"] = False
    except (TypeError, ValueError):
        checks["latency_within_timeout"] = False
    return {
        "passed": all(checks.values()),
        "checks": checks,
        **action_summary.as_dict(),
    }


def main() -> None:
    args = parse_args()
    summary = json.loads((args.episode_dir / "summary.json").read_text(encoding="utf-8"))
    with (args.episode_dir / "steps.jsonl").open(encoding="utf-8") as stream:
        record = json.loads(next(line for line in stream if line.strip()))
    jpeg_path = args.episode_dir / record["front_rgb"]
    jpeg = jpeg_path.read_bytes()
    request = build_preflight_request(
        record=record,
        jpeg=jpeg,
        episode_id=str(summary["short_episode_id"]),
        instruction=str(summary["instruction"]),
        replan_index=0,
    )
    started = time.perf_counter()
    response = unix_request(args.socket, request, jpeg, timeout_s=args.timeout)
    wall_ms = (time.perf_counter() - started) * 1000.0
    evaluated = evaluate_preflight_response(response, wall_ms)
    report = {
        "format": "go2-short-vln-m7-preflight-v1",
        **evaluated,
        "episode_id": summary["short_episode_id"],
        "source_frame": int(record["frame_index"]),
        "source_jpeg": str(jpeg_path),
        "request_fields": sorted(request),
        "state_shape": [len(request["state"])],
        "response_action_shape": [1, len(response["actions"]), len(response["actions"][0])],
        "raw_range_violation_count": evaluated["raw_action_vector_violation_count"],
        "server_timings_ms": response["timings_ms"],
        "roundtrip_wall_ms": wall_ms,
        "checkpoint_sha256": response["checkpoint_sha256"],
        "seed": response["seed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
