#!/usr/bin/env python3
"""Fail-closed evidence check for a full-episode 3-D closed-loop rollout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.inference.full_protocol import PROTOCOL_VERSION, validate_request_header, validate_response_header
from src.inference.full_state import apply_action_safety, validate_action_chunk


LATENCY_P95_MS_MAX = 600.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate(summary: Mapping[str, Any], records: list[Mapping[str, Any]], requests: list[Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["full_protocol_version"] = summary.get("protocol_version") == PROTOCOL_VERSION
    checks["three_dimensional_states"] = all(isinstance(row.get("policy_state"), list) and len(row["policy_state"]) == 3 and all(math.isfinite(float(value)) for value in row["policy_state"]) for row in records)
    checks["ten_frames_per_replan"] = summary.get("execute_steps_per_chunk") == 10 and summary.get("temporal_contract", {}).get("execute_frames_per_replan") == 10
    checks["no_wall_clock_rate_claim"] = summary.get("temporal_contract", {}).get("wall_clock_control_rate_claim") is None
    checks["p95_latency_at_most_600ms"] = summary.get("latency_gate", {}).get("passed") is True and float(summary.get("p95_inference_roundtrip_ms", math.inf)) <= LATENCY_P95_MS_MAX
    checks["requests_valid"] = True
    checks["raw_actions_in_range"] = True
    for row in requests:
        try:
            request, response = row["request"], row["response"]
            validate_request_header(request, payload_size=int(request["jpeg_size"]))
            validate_response_header(response)
            for action in validate_action_chunk(response["actions"]):
                if not apply_action_safety(action).in_range:
                    checks["raw_actions_in_range"] = False
        except (KeyError, TypeError, ValueError):
            checks["requests_valid"] = False
    checks["summary_passed"] = summary.get("passed") is True
    return {
        "format": "navila-full-episode-closed-loop-check-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "record_count": len(records),
        "request_count": len(requests),
        "observed_p95_roundtrip_ms": summary.get("p95_inference_roundtrip_ms"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.episode_dir.resolve()
    result = evaluate(
        json.loads((root / "summary.json").read_text(encoding="utf-8")),
        load_jsonl(root / "steps.jsonl"), load_jsonl(root / "requests.jsonl"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
