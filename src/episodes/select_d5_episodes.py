#!/usr/bin/env python3
"""Write the deterministic D5 short-VLN collection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.smolvla.d5_data_expansion import (
    DISTANCE_BIN_NAMES,
    episode_features,
    select_stratified_episodes,
    selection_hash,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(items),
        "categories": {name: sum(item["category"] == name for item in items) for name in ("straight", "left_turn", "right_turn")},
        "scene_count": len({item["scene_id"] for item in items}),
        "heading_bin_count": len({item["heading_bin"] for item in items}),
        "distance_bins": {name: sum(item["distance_bin"] == name for item in items) for name in DISTANCE_BIN_NAMES},
    }


def _candidate_order(episodes: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, list[str]]:
    selected_ids = {item["short_episode_id"] for item in selected}
    output: dict[str, list[str]] = {}
    for category in ("straight", "left_turn", "right_turn"):
        output[category] = [
            item["short_episode_id"]
            for item in sorted(episodes, key=lambda item: item["short_episode_id"])
            if item["category"] == category and item["short_episode_id"] not in selected_ids
        ]
    return output


def main() -> None:
    args = parse_args()
    source = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection manifest: {output}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("short-VLN JSON requires an episodes list")
    train = select_stratified_episodes(
        episodes,
        split="train",
        per_category=10,
        required_ids=("short_vln_v1_0000", "short_vln_v1_0004"),
        min_scenes=9,
        min_heading_bins=8,
        min_distance_per_bin=6,
        seed=args.seed,
    )
    seen_val = select_stratified_episodes(
        episodes,
        split="seen-val",
        per_category=4,
        required_ids=("short_vln_v1_0001", "short_vln_v1_0003"),
        min_scenes=7,
        min_heading_bins=6,
        min_distance_per_bin=2,
        seed=args.seed,
    )
    all_features = [episode_features(item) for item in episodes]
    result = {
        "format": "go2-short-vln-m6_2-d5-selection-v1",
        "source_dataset": str(source),
        "source_dataset_sha256": sha256(source),
        "seed": args.seed,
        "splits": {
            "train": {"primary": train, "summary": _summary(train), "candidate_order_by_category": _candidate_order([item for item in all_features if item["split"] == "train"], train)},
            "seen_val": {"primary": seen_val, "summary": _summary(seen_val), "candidate_order_by_category": _candidate_order([item for item in all_features if item["split"] == "seen-val"], seen_val)},
        },
    }
    result["selection_sha256"] = selection_hash(train + seen_val)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selection_sha256": result["selection_sha256"], "train": result["splits"]["train"]["summary"], "seen_val": result["splits"]["seen_val"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
