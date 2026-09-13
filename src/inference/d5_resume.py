#!/usr/bin/env python3
"""Validate persisted D5 rollout evidence before a resumed run skips it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.check_rollout import evaluate_rollout_records, load_jsonl


def audit_completed_episode(
    episode_dir: Path,
    *,
    expected_episode_id: str,
    max_steps: int = 1500,
) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    errors: list[str] = []
    required = ("summary.json", "steps.jsonl", "requests.jsonl")
    for name in required:
        if not (episode_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return {"passed": False, "episode_dir": str(episode_dir), "errors": errors}

    try:
        summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
        records = load_jsonl(episode_dir / "steps.jsonl")
        requests = load_jsonl(episode_dir / "requests.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "passed": False,
            "episode_dir": str(episode_dir),
            "errors": [f"unreadable evidence: {error}"],
        }

    if summary.get("short_episode_id") != expected_episode_id:
        errors.append("episode ID mismatch")
    if summary.get("error") is not None:
        errors.append("summary contains an error")
    if int(summary.get("frame_count", -1)) != len(records):
        errors.append("summary frame count differs from steps.jsonl")
    if not requests:
        errors.append("requests.jsonl is empty")

    success = bool(summary.get("success"))
    termination_reason = summary.get("termination_reason")
    if success:
        if summary.get("status") != "complete":
            errors.append("successful episode status is not complete")
        if termination_reason != "evaluator_goal_stop":
            errors.append("successful episode lacks evaluator_goal_stop")
        if not 1 <= len(records) <= max_steps:
            errors.append("successful episode has an invalid frame count")
    else:
        if summary.get("status") != "terminated":
            errors.append("failed episode status is not terminated")
        if termination_reason != "client_max_steps" or len(records) != max_steps:
            errors.append("failed episode did not complete the max-step horizon")

    missing_frames = [
        str(record.get("front_rgb"))
        for record in records
        if not isinstance(record.get("front_rgb"), str)
        or not (episode_dir / str(record.get("front_rgb"))).is_file()
    ]
    if missing_frames:
        errors.append(f"missing referenced RGB frames: {len(missing_frames)}")

    try:
        checked = evaluate_rollout_records(summary, records, requests)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"rollout audit failed: {error}")
        checked = None
    if checked is not None and not checked["infrastructure_passed"]:
        errors.append("rollout infrastructure audit failed")

    return {
        "passed": not errors,
        "episode_dir": str(episode_dir),
        "episode_id": summary.get("short_episode_id"),
        "frame_count": len(records),
        "request_count": len(requests),
        "success": success,
        "termination_reason": termination_reason,
        "raw_output_range_passed": None if checked is None else checked["raw_output_range_passed"],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--expected-episode-id", required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_completed_episode(
        args.episode_dir,
        expected_episode_id=args.expected_episode_id,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
