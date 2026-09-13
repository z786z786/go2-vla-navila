#!/usr/bin/env python3
"""Evaluate the small R7 expert-collection canary before full recollection."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.collector.r7_supervision import command_semantics, pd_motion_semantics, terminal_hold_audit


CANARY_EPISODE_COUNT = 6
CANARY_PER_CATEGORY = 2
CANARY_MIN_SCENES = 4
CANARY_TRAIN_EPISODES = 4
CANARY_SEEN_VAL_EPISODES = 2


def category(instruction: str) -> str:
    text = instruction.lower()
    if "turn left" in text:
        return "left_turn"
    if "turn right" in text:
        return "right_turn"
    return "straight"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"non-object record at {path}:{line_number}")
        rows.append(row)
    return rows


def _config_hold_frames(episode_dir: Path) -> int:
    payload = json.loads((episode_dir / "collector_config.json").read_text(encoding="utf-8"))
    value = payload.get("terminal_hold_frames")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{episode_dir}: terminal_hold_frames must be a positive integer")
    return value


def check_episode(episode_dir: Path, *, expected_state_dim: int, hold_frames: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = episode_dir.resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    sanity = json.loads((directory / "sanity.json").read_text(encoding="utf-8"))
    records = _read_rows(directory / "steps.jsonl")
    if not records:
        raise ValueError(f"{directory}: canary episode has no records")
    state_ok = all(
        isinstance(row.get("robot_state"), list)
        and len(row["robot_state"]) == expected_state_dim
        and all(math.isfinite(float(value)) for value in row["robot_state"])
        for row in records
    )
    semantics = command_semantics(records, allow_legacy=False)
    terminal = terminal_hold_audit(records, requested_frames=hold_frames)
    passed = (
        summary.get("status") == "complete"
        and summary.get("success") is True
        and sanity.get("passed") is True
        and state_ok
        and semantics["passed"]
        and terminal["passed"]
    )
    return {
        "episode_id": summary.get("short_episode_id"),
        "split": summary.get("split"),
        "scene_id": summary.get("scene_id"),
        "category": category(str(summary.get("instruction", ""))),
        "record_count": len(records),
        "state_dimension": expected_state_dim,
        "state_finite_and_constant": state_ok,
        "success": summary.get("success") is True,
        "sanity_passed": sanity.get("passed") is True,
        "command_semantics": semantics,
        "terminal_hold": terminal,
        "passed": passed,
    }, records


def evaluate_canary(
    episode_dirs: Sequence[Path], *, expected_state_dim: int, expected_hold_frames: int
) -> dict[str, Any]:
    if len(episode_dirs) != CANARY_EPISODE_COUNT:
        raise ValueError(f"canary requires exactly {CANARY_EPISODE_COUNT} episode directories")
    if expected_state_dim <= 0 or expected_hold_frames <= 0:
        raise ValueError("expected state dimension and hold frames must be positive")
    checked = [check_episode(path, expected_state_dim=expected_state_dim, hold_frames=expected_hold_frames) for path in episode_dirs]
    episodes = [item[0] for item in checked]
    pd_motion = pd_motion_semantics([row for _, rows in checked for row in rows])
    categories = Counter(str(item["category"]) for item in episodes)
    splits = Counter(str(item["split"]) for item in episodes)
    unique_scenes = {str(item["scene_id"]) for item in episodes if item["scene_id"]}
    coverage = {
        "episode_count_exact": len(episodes) == CANARY_EPISODE_COUNT,
        "two_per_category": all(categories[name] == CANARY_PER_CATEGORY for name in ("straight", "left_turn", "right_turn")),
        "minimum_scene_coverage": len(unique_scenes) >= CANARY_MIN_SCENES,
        "split_balance": splits["train"] == CANARY_TRAIN_EPISODES and splits["seen-val"] == CANARY_SEEN_VAL_EPISODES,
    }
    return {
        "format": "go2-short-vln-m6_2-r7-canary-quality-v1",
        "expected_state_dimension": expected_state_dim,
        "expected_terminal_hold_frames": expected_hold_frames,
        "episodes": episodes,
        "coverage": {
            **coverage,
            "category_counts": dict(sorted(categories.items())),
            "split_counts": dict(sorted(splits.items())),
            "scene_count": len(unique_scenes),
        },
        "pd_motion_semantics": pd_motion,
        "passed": all(item["passed"] for item in episodes) and all(coverage.values()) and pd_motion["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-state-dim", type=int, default=31)
    parser.add_argument("--expected-terminal-hold-frames", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_canary(
        args.episode_dirs,
        expected_state_dim=args.expected_state_dim,
        expected_hold_frames=args.expected_terminal_hold_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
