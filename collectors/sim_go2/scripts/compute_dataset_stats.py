#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from collectors.sim_go2.utils.io import dump_json, iter_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute packed dataset stats.")
    parser.add_argument("--input", type=Path, required=True, help="Packed dataset root or dataset.jsonl path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--turn-threshold", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = args.input if args.input.is_file() else args.input / "dataset.jsonl"
    vx_values = []
    wz_values = []
    state_vx = []
    state_vy = []
    state_vz = []
    scene_counts = Counter()
    target_counts = Counter()
    split_counts = Counter()
    task_counts = Counter()
    visibility_counts = Counter()
    left_turns = 0
    right_turns = 0
    nonzero_turns = 0
    chunk_turns = 0
    visible_first = 0
    visible_3f = 0
    visible_10f = 0
    episodes = set()
    sample_count = 0
    for record in iter_jsonl(dataset_path):
        sample_count += 1
        episodes.add(str(record.get("episode_id", "")))
        split_counts[str(record.get("split", ""))] += 1
        scene_counts[str(record.get("scene_id", ""))] += 1
        target_counts[str(record.get("target_class", ""))] += 1
        task_counts[str(record.get("task_id", ""))] += 1
        visibility_counts[str(record.get("visibility_bucket", ""))] += 1
        visible_first += int(bool(record.get("target_visible_first_frame", False)))
        visible_3f += int(bool(record.get("target_visible_within_3f", False)))
        visible_10f += int(bool(record.get("target_visible_within_10f", False)))
        state = list(record.get("state", [0.0, 0.0, 0.0]))
        state_vx.append(float(state[0]))
        state_vy.append(float(state[1]))
        state_vz.append(float(state[2]))
        chunk = list(record.get("action_chunk", []))
        if chunk:
            first_vx = float(chunk[0][0])
            first_wz = float(chunk[0][1])
            vx_values.append(first_vx)
            wz_values.append(first_wz)
            if abs(first_wz) > args.turn_threshold:
                nonzero_turns += 1
                if first_wz > 0:
                    left_turns += 1
                else:
                    right_turns += 1
            if any(abs(float(step[1])) > args.turn_threshold for step in chunk):
                chunk_turns += 1
    def summarize(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {"mean": float(statistics.mean(values)), "min": float(min(values)), "max": float(max(values))}
    payload = {
        "dataset_path": str(dataset_path),
        "sample_count": sample_count,
        "episode_count": len(episodes),
        "vx": summarize(vx_values),
        "wz": summarize(wz_values),
        "state_vx": summarize(state_vx),
        "state_vy": summarize(state_vy),
        "state_vz": summarize(state_vz),
        "turning_ratio": (nonzero_turns / sample_count) if sample_count else 0.0,
        "left_turn_ratio": (left_turns / sample_count) if sample_count else 0.0,
        "right_turn_ratio": (right_turns / sample_count) if sample_count else 0.0,
        "chunk_turn_ratio": (chunk_turns / sample_count) if sample_count else 0.0,
        "visible_first_frame_ratio": (visible_first / sample_count) if sample_count else 0.0,
        "visible_within_3f_ratio": (visible_3f / sample_count) if sample_count else 0.0,
        "visible_within_10f_ratio": (visible_10f / sample_count) if sample_count else 0.0,
        "split_counts": dict(sorted(split_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "visibility_bucket_counts": dict(sorted(visibility_counts.items())),
    }
    if args.output:
        dump_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
