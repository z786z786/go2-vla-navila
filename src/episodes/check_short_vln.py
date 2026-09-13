#!/usr/bin/env python3
"""Independent structural and semantic validator for short_vln_v1.json."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--spot-check-report", type=Path, required=True)
    return parser.parse_args()


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def polyline_length(path: Sequence[Sequence[float]]) -> float:
    return sum(distance(a, b) for a, b in zip(path, path[1:]))


def scene_name(scene_id: str) -> str:
    return scene_id.split("/")[1]


def yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrapped_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def fail(errors: list[str], episode: dict[str, Any], message: str) -> None:
    if len(errors) < 100:
        errors.append(f"{episode.get('short_episode_id', '<unknown>')}: {message}")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    with gzip.open(args.source_dataset, "rt", encoding="utf-8") as stream:
        source_payload = json.load(stream)
    episodes = payload["episodes"]
    source_episodes = source_payload["episodes"]
    source_by_id = {str(episode["episode_id"]): episode for episode in source_episodes}
    errors: list[str] = []

    if payload.get("schema_version") != "short_vln_v1":
        errors.append("wrong schema_version")
    if len(episodes) < 100:
        errors.append("fewer than 100 episodes")
    if len({episode["short_episode_id"] for episode in episodes}) != len(episodes):
        errors.append("duplicate short_episode_id")
    if len({str(episode["source_episode_id"]) for episode in episodes}) != len(episodes):
        errors.append("source episode reused")

    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for episode in episodes:
        split_counts[episode["split"]] += 1
        path = episode["reference_path"]
        if len(path) < 2:
            fail(errors, episode, "path has fewer than 2 points")
            continue
        length = polyline_length(path)
        if not 1.0 <= length <= 4.0:
            fail(errors, episode, f"length {length} outside [1, 4]")
        if not math.isclose(length, float(episode["path_length"]), abs_tol=1e-9):
            fail(errors, episode, "stored path length differs from independent sum")
        if distance(episode["start_pose"]["position"], path[0]) > 1e-9:
            fail(errors, episode, "start position differs from path start")
        if distance(episode["goal_pose"]["position"], path[-1]) > 1e-9:
            fail(errors, episode, "goal position differs from path end")
        quaternion = episode["start_pose"]["rotation_wxyz"]
        if not math.isclose(sum(float(value) ** 2 for value in quaternion), 1.0, abs_tol=1e-9):
            fail(errors, episode, "start quaternion is not normalized")
        tangent_yaw = math.atan2(
            float(path[1][1]) - float(path[0][1]),
            float(path[1][0]) - float(path[0][0]),
        )
        if abs(wrapped_delta(yaw_from_wxyz(quaternion), tangent_yaw)) > 1e-7:
            fail(errors, episode, "start yaw is not aligned with path tangent")

        source = source_by_id.get(str(episode["source_episode_id"]))
        if source is None:
            fail(errors, episode, "source episode does not exist")
        else:
            start_index, end_index = episode["source_path_index_range"]
            expected = source["gt_locations"][start_index : end_index + 1]
            if path != expected:
                fail(errors, episode, "path is not the declared contiguous source slice")
            if episode["scene_id"] != source["scene_id"]:
                fail(errors, episode, "scene differs from source")

        turns = episode["turn_statistics"]["major_turns"]
        turn_count = int(episode["turn_statistics"]["major_turn_count"])
        if turn_count != len(turns) or turn_count > 1:
            fail(errors, episode, "invalid major-turn count")
        metadata = episode["instruction_metadata"]
        category = metadata["category"]
        category_counts[category] += 1
        parameters = metadata["parameters"]
        if metadata["uses_external_llm"] or metadata["uses_landmark_label"]:
            fail(errors, episode, "instruction policy is not geometry-only")
        if category == "straight":
            expected_instruction = (
                f"Move forward along the available route for about {length:.1f} meters, then stop."
            )
            if turn_count != 0:
                fail(errors, episode, "straight instruction has a major turn")
            if not math.isclose(parameters["total_distance_m"], length, abs_tol=1e-9):
                fail(errors, episode, "straight instruction distance mismatch")
        elif category in {"left_turn", "right_turn"}:
            if turn_count != 1:
                fail(errors, episode, "turn instruction lacks exactly one major turn")
                continue
            direction = category.removesuffix("_turn")
            if turns[0]["direction"] != direction:
                fail(errors, episode, "instruction direction differs from turn event")
            before = float(parameters["distance_before_turn_m"])
            after = float(parameters["distance_after_turn_m"])
            if before < 0.75 or after < 0.75:
                fail(errors, episode, "turn leg is shorter than 0.75 m")
            if not math.isclose(before + after, length, abs_tol=1e-9):
                fail(errors, episode, "turn instruction legs do not sum to path length")
            expected_instruction = (
                f"Move forward about {before:.1f} meters, turn {direction}, then continue about "
                f"{after:.1f} meters and stop."
            )
        else:
            fail(errors, episode, f"unknown instruction category {category}")
            continue
        if episode["instruction"] != expected_instruction:
            fail(errors, episode, "instruction text differs from declared parameters")
        if (
            episode["instruction"]
            == episode["source_metadata"]["original_instruction_text_not_for_training"]
        ):
            fail(errors, episode, "original long instruction was reused")

    scenes = {
        split: {
            scene_name(episode["scene_id"])
            for episode in episodes
            if episode["split"] == split
        }
        for split in ("train", "seen-val", "unseen-test")
    }
    if scenes["train"] & scenes["unseen-test"]:
        errors.append("train/unseen-test scene leakage")
    if not scenes["seen-val"] <= scenes["train"]:
        errors.append("seen-val scene is absent from train")
    if any(split_counts[split] == 0 for split in scenes):
        errors.append("an output split is empty")
    spot_text = args.spot_check_report.read_text(encoding="utf-8")
    spot_passes = spot_text.count("| PASS |")
    if spot_passes < 20:
        errors.append(f"only {spot_passes} spot-check PASS rows")
    spot_ids = re.findall(r"\| (short_vln_v1_\d+) \|", spot_text)
    episode_by_short_id = {episode["short_episode_id"]: episode for episode in episodes}
    checked = [episode_by_short_id[short_id] for short_id in spot_ids if short_id in episode_by_short_id]
    checked_scenes = {scene_name(episode["scene_id"]) for episode in checked}
    checked_splits = {episode["split"] for episode in checked}
    checked_categories = {episode["instruction_metadata"]["category"] for episode in checked}
    all_scenes = {scene_name(episode["scene_id"]) for episode in episodes}
    if len(checked) != 20:
        errors.append(f"spot-check report resolves to {len(checked)} dataset episodes")
    if checked_scenes != all_scenes:
        errors.append("spot checks do not cover every source scene")
    if checked_splits != {"train", "seen-val", "unseen-test"}:
        errors.append("spot checks do not cover every split")
    if checked_categories != {"straight", "left_turn", "right_turn"}:
        errors.append("spot checks do not cover every instruction category")

    if errors:
        print("Validation: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Validation: PASS")
    print(f"Episodes: {len(episodes)}")
    print(f"Splits: {dict(split_counts)}")
    print(f"Scenes: { {key: len(value) for key, value in scenes.items()} }")
    print(f"Instruction categories: {dict(category_counts)}")
    print(f"Spot-check PASS rows: {spot_passes}")


if __name__ == "__main__":
    main()
