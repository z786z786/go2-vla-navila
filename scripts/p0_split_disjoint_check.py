#!/usr/bin/env python3
"""Mechanically verify that the R2R-VLNCE train and VLN-CE-Isaac scenes are disjoint."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


TRAIN_PATH = Path(
    "/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz"
)
EVALUATION_PATH = Path(
    "/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz"
)
OUTPUT_PATH = Path("/home/wxh/go2_short_vln/reports/split_disjoint_check.json")
EXPECTED_TRAIN_EPISODES = 10819
EXPECTED_EVALUATION_EPISODES = 1077


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        document = json.load(source)
    episodes = document.get("episodes") if isinstance(document, dict) else None
    if not isinstance(episodes, list):
        raise ValueError(f"{path} does not contain a list at top-level key 'episodes'")
    if not all(isinstance(episode, dict) for episode in episodes):
        raise ValueError(f"{path} contains a non-object episode")
    return episodes


def normalize_scene_stem(scene_id: str) -> str:
    """Return only the final filename stem; do not alter case or path aliases."""
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError(f"invalid scene_id: {scene_id!r}")
    return PurePosixPath(scene_id).stem


def scene_sets(episodes: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    raw_scene_ids: set[str] = set()
    normalized_scene_stems: set[str] = set()
    for episode in episodes:
        scene_id = episode.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"episode {episode.get('episode_id')!r} has invalid scene_id")
        raw_scene_ids.add(scene_id)
        normalized_scene_stems.add(normalize_scene_stem(scene_id))
    return sorted(raw_scene_ids), sorted(normalized_scene_stems)


def radius_evidence(episodes: list[dict[str, Any]]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Collect every goals[0].radius value and any episode that is not exactly 3.0."""
    values: list[Any] = []
    counterexamples: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = episode.get("episode_id")
        goals = episode.get("goals")
        if not isinstance(goals, list) or not goals or not isinstance(goals[0], dict):
            counterexamples.append(
                {
                    "episode_id": episode_id,
                    "reason": "goals[0] is missing or is not an object",
                }
            )
            continue
        if "radius" not in goals[0]:
            counterexamples.append(
                {"episode_id": episode_id, "reason": "goals[0].radius is missing"}
            )
            continue
        radius = goals[0]["radius"]
        values.append(radius)
        if radius != 3.0:
            counterexamples.append(
                {
                    "episode_id": episode_id,
                    "observed_radius": radius,
                    "reason": "goals[0].radius != 3.0",
                }
            )

    try:
        unique_values = sorted(set(values))
    except TypeError:
        unique_values = sorted({json.dumps(value, sort_keys=True) for value in values})
    return unique_values, counterexamples


def main() -> None:
    train_episodes = load_episodes(TRAIN_PATH)
    evaluation_episodes = load_episodes(EVALUATION_PATH)
    train_raw_scenes, train_scene_stems = scene_sets(train_episodes)
    evaluation_raw_scenes, evaluation_scene_stems = scene_sets(evaluation_episodes)
    intersection = sorted(set(train_scene_stems).intersection(evaluation_scene_stems))
    radius_values, radius_counterexamples = radius_evidence(evaluation_episodes)

    train_count_matches = len(train_episodes) == EXPECTED_TRAIN_EPISODES
    evaluation_count_matches = len(evaluation_episodes) == EXPECTED_EVALUATION_EPISODES
    intersection_empty = intersection == []
    assertion = (
        "PASSED"
        if train_count_matches and evaluation_count_matches and intersection_empty
        else "FAILED"
    )

    report = {
        "task_id": "P0-T2",
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "generator_command": "python3 /home/wxh/go2_short_vln/scripts/p0_split_disjoint_check.py",
        "inputs": {
            str(TRAIN_PATH): sha256(TRAIN_PATH),
            str(EVALUATION_PATH): sha256(EVALUATION_PATH),
        },
        "findings": {
            "source_selection": {
                "train": {
                    "path": str(TRAIN_PATH),
                    "episodes_key": "episodes",
                    "episode_count": len(train_episodes),
                    "expected_episode_count": EXPECTED_TRAIN_EPISODES,
                    "matches_expected": train_count_matches,
                },
                "evaluation": {
                    "path": str(EVALUATION_PATH),
                    "episodes_key": "episodes",
                    "episode_count": len(evaluation_episodes),
                    "expected_episode_count": EXPECTED_EVALUATION_EPISODES,
                    "matches_expected": evaluation_count_matches,
                },
                "alternative_source_search_required": False,
                "alternative_source_search_reason": "Both prescribed sources contain their expected episode counts.",
            },
            "normalization_rule": {
                "description": "For each scene_id, take its final POSIX path component and remove only that filename's final extension (PurePosixPath(scene_id).stem). No case folding, directory rewriting, alias mapping, or other normalization is performed.",
                "example_input": "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
                "example_output": "17DRP5sb8fy",
            },
            "scene_sets": {
                "train": {
                    "raw_scene_ids_sorted": train_raw_scenes,
                    "raw_scene_id_set_size": len(train_raw_scenes),
                    "normalized_scene_stems_sorted": train_scene_stems,
                    "normalized_scene_stem_set_size": len(train_scene_stems),
                    "expected_normalized_scene_stem_set_size": 61,
                    "matches_expected_normalized_size": len(train_scene_stems) == 61,
                },
                "evaluation": {
                    "raw_scene_ids_sorted": evaluation_raw_scenes,
                    "raw_scene_id_set_size": len(evaluation_raw_scenes),
                    "normalized_scene_stems_sorted": evaluation_scene_stems,
                    "normalized_scene_stem_set_size": len(evaluation_scene_stems),
                    "expected_normalized_scene_stem_set_size": 11,
                    "matches_expected_normalized_size": len(evaluation_scene_stems) == 11,
                },
            },
            "intersection": {
                "basis": "normalized scene stems",
                "size": len(intersection),
                "contents_sorted": intersection,
                "assertion_expression": "intersection == []",
                "assertion": "PASSED" if intersection_empty else "FAILED",
            },
            "goals_0_radius": {
                "checked_episodes": len(evaluation_episodes),
                "expected_value": 3.0,
                "unique_values_sorted": radius_values,
                "counterexamples": radius_counterexamples,
                "assertion_expression": "every evaluation episode has goals[0].radius == 3.0",
                "assertion": "PASSED" if not radius_counterexamples else "FAILED",
            },
            "assertion": assertion,
            "assertion_conditions": {
                "train_episode_count_equals_10819": train_count_matches,
                "evaluation_episode_count_equals_1077": evaluation_count_matches,
                "normalized_scene_intersection_equals_empty_list": intersection_empty,
            },
        },
        "missing": [],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
