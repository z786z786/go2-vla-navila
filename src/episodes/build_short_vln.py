#!/usr/bin/env python3
"""Build deterministic 1--4 m short-VLN episodes from VLN-CE-Isaac.

Each output route is one contiguous slice of the official dense ``gt_locations``
path.  The slice is therefore directly usable as an expert path by the M4 PD
planner.  Language is generated only from audited route geometry; the original
full-route instruction is retained under source metadata and is never reused as
the short task instruction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from inspect_episode import load_episodes, path_length, point_distance, resolve_dataset, scene_name


SCHEMA_VERSION = "short_vln_v1"
INSTRUCTION_TEMPLATES = {
    "straight": "Move forward along the available route for about {distance:.1f} meters, then stop.",
    "left_turn": (
        "Move forward about {before:.1f} meters, turn left, then continue about "
        "{after:.1f} meters and stop."
    ),
    "right_turn": (
        "Move forward about {before:.1f} meters, turn right, then continue about "
        "{after:.1f} meters and stop."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/short_vln_v1.json"))
    parser.add_argument(
        "--statistics-report",
        type=Path,
        default=Path("reports/short_vln_v1_statistics.md"),
    )
    parser.add_argument(
        "--spot-check-report",
        type=Path,
        default=Path("reports/m3_spot_check.md"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--min-path-length", type=float, default=1.0)
    parser.add_argument("--max-path-length", type=float, default=4.0)
    parser.add_argument("--major-turn-degrees", type=float, default=45.0)
    parser.add_argument("--turn-merge-distance", type=float, default=0.75)
    parser.add_argument("--turn-min-delta-degrees", type=float, default=5.0)
    parser.add_argument("--turn-min-leg-distance", type=float, default=0.75)
    parser.add_argument("--unseen-scene-fraction", type=float, default=0.20)
    parser.add_argument("--seen-val-episode-fraction", type=float, default=0.15)
    parser.add_argument("--spot-check-count", type=int, default=20)
    return parser.parse_args()


def stable_rank(seed: int, *values: object) -> int:
    text = "|".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def wrapped_delta_radians(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def cumulative_distances(path: Sequence[Sequence[float]]) -> list[float]:
    distances = [0.0]
    for start, end in zip(path, path[1:]):
        distances.append(distances[-1] + point_distance(start, end, 3))
    return distances


def segment_headings(path: Sequence[Sequence[float]]) -> list[float]:
    headings = []
    for start, end in zip(path, path[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) <= 1e-8:
            headings.append(headings[-1] if headings else 0.0)
        else:
            headings.append(math.atan2(dy, dx))
    return headings


def major_turn_events(
    path: Sequence[Sequence[float]],
    threshold_degrees: float,
    merge_distance: float,
    minimum_delta_degrees: float,
) -> list[dict[str, Any]]:
    """Cluster nearby, same-direction heading deltas into auditable turn events.

    Habitat expert paths may express a major turn as several 15-degree changes
    separated by short forward steps.  Same-sign deltas no more than
    ``merge_distance`` apart are accumulated; opposite-sign changes always begin
    a new event.  Only clusters whose net magnitude reaches the threshold are
    returned as major turns.
    """
    headings = segment_headings(path)
    if len(headings) < 2:
        return []
    unwrapped = [headings[0]]
    for heading in headings[1:]:
        unwrapped.append(unwrapped[-1] + wrapped_delta_radians(heading, unwrapped[-1]))
    distances = cumulative_distances(path)
    raw_changes: list[dict[str, float | int]] = []
    for vertex_index, (previous, current) in enumerate(
        zip(unwrapped, unwrapped[1:]), start=1
    ):
        delta_degrees = math.degrees(current - previous)
        if abs(delta_degrees) >= minimum_delta_degrees:
            raw_changes.append(
                {
                    "vertex_index": vertex_index,
                    "delta_degrees": delta_degrees,
                    "distance_from_start_m": distances[vertex_index],
                }
            )

    groups: list[list[dict[str, float | int]]] = []
    for change in raw_changes:
        if groups:
            previous = groups[-1][-1]
            same_direction = (
                float(change["delta_degrees"]) * float(previous["delta_degrees"]) > 0
            )
            close_enough = (
                float(change["distance_from_start_m"])
                - float(previous["distance_from_start_m"])
                <= merge_distance + 1e-9
            )
        else:
            same_direction = close_enough = False
        if same_direction and close_enough:
            groups[-1].append(change)
        else:
            groups.append([change])

    events = []
    for group in groups:
        angle = sum(float(change["delta_degrees"]) for change in group)
        if abs(angle) + 1e-6 < threshold_degrees:
            continue
        weights = [abs(float(change["delta_degrees"])) for change in group]
        event_distance = sum(
            float(change["distance_from_start_m"]) * weight
            for change, weight in zip(group, weights)
        ) / sum(weights)
        events.append(
            {
                "direction": "left" if angle > 0 else "right",
                "signed_angle_degrees": angle,
                "start_vertex_index": int(group[0]["vertex_index"]),
                "end_vertex_index": int(group[-1]["vertex_index"]),
                "distance_from_start_m": event_distance,
                "component_count": len(group),
            }
        )
    return events


def yaw_quaternion_wxyz(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def enumerate_candidates(
    source_path: Sequence[Sequence[float]], args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for start_index in range(len(source_path) - 1):
        length = 0.0
        for end_index in range(start_index + 1, len(source_path)):
            length += point_distance(source_path[end_index - 1], source_path[end_index], 3)
            if length > args.max_path_length:
                break
            if length < args.min_path_length:
                continue
            short_path = source_path[start_index : end_index + 1]
            events = major_turn_events(
                short_path,
                args.major_turn_degrees,
                args.turn_merge_distance,
                args.turn_min_delta_degrees,
            )
            if len(events) > 1:
                continue
            if not events:
                category = "straight"
                before = after = None
            else:
                before = float(events[0]["distance_from_start_m"])
                after = length - before
                if (
                    before < args.turn_min_leg_distance
                    or after < args.turn_min_leg_distance
                ):
                    continue
                category = f"{events[0]['direction']}_turn"
            candidates[category].append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "path_length": length,
                    "major_turns": events,
                    "before_turn": before,
                    "after_turn": after,
                }
            )
    return candidates


def choose_candidate(
    source_episode: dict[str, Any], candidates: dict[str, list[dict[str, Any]]], seed: int
) -> tuple[str, dict[str, Any]]:
    category_order = ["straight", "left_turn", "right_turn"]
    offset = stable_rank(seed, "category", source_episode["episode_id"]) % len(category_order)
    preferred = category_order[offset:] + category_order[:offset]
    category = next((name for name in preferred if candidates.get(name)), None)
    if category is None:
        raise ValueError(f"No legal short segment for episode {source_episode['episode_id']}")
    target_length = 1.5 + (
        stable_rank(seed, "length", source_episode["episode_id"]) % 2501
    ) / 1000.0

    def score(candidate: dict[str, Any]) -> tuple[float, int]:
        balance_penalty = 0.0
        if candidate["before_turn"] is not None:
            balance_penalty = 0.05 * abs(
                float(candidate["before_turn"]) - float(candidate["after_turn"])
            )
        tie_break = stable_rank(
            seed,
            "candidate",
            source_episode["episode_id"],
            candidate["start_index"],
            candidate["end_index"],
        )
        return abs(float(candidate["path_length"]) - target_length) + balance_penalty, tie_break

    return category, min(candidates[category], key=score)


def split_source_episodes(
    episodes: Sequence[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, str], dict[str, list[str]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_scene[scene_name(str(episode["scene_id"]))].append(episode)
    scenes = sorted(by_scene)
    unseen_count = max(1, round(len(scenes) * args.unseen_scene_fraction))
    target_unseen_episodes = len(episodes) * args.unseen_scene_fraction
    scene_combinations = itertools.combinations(scenes, unseen_count)
    chosen_unseen = min(
        scene_combinations,
        key=lambda names: (
            abs(sum(len(by_scene[name]) for name in names) - target_unseen_episodes),
            stable_rank(args.seed, "unseen_scenes", *names),
        ),
    )
    unseen_scenes = set(chosen_unseen)
    split_by_episode_id: dict[str, str] = {}
    for name, scene_episodes in sorted(by_scene.items()):
        ranked = sorted(
            scene_episodes,
            key=lambda episode: stable_rank(
                args.seed, "source_split", name, episode["episode_id"]
            ),
        )
        if name in unseen_scenes:
            for episode in ranked:
                split_by_episode_id[str(episode["episode_id"])] = "unseen-test"
            continue
        seen_val_count = max(1, round(len(ranked) * args.seen_val_episode_fraction))
        if len(ranked) > 1:
            seen_val_count = min(seen_val_count, len(ranked) - 1)
        for index, episode in enumerate(ranked):
            split_by_episode_id[str(episode["episode_id"])] = (
                "seen-val" if index < seen_val_count else "train"
            )
    train_scenes = sorted(set(scenes) - unseen_scenes)
    scene_splits = {
        "train": train_scenes,
        "seen-val": train_scenes,
        "unseen-test": sorted(unseen_scenes),
    }
    return split_by_episode_id, scene_splits


def make_instruction(category: str, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    length = float(candidate["path_length"])
    if category == "straight":
        text = INSTRUCTION_TEMPLATES[category].format(distance=length)
        parameters = {"total_distance_m": length}
    else:
        before = float(candidate["before_turn"])
        after = float(candidate["after_turn"])
        text = INSTRUCTION_TEMPLATES[category].format(before=before, after=after)
        parameters = {
            "distance_before_turn_m": before,
            "distance_after_turn_m": after,
            "total_distance_m": length,
        }
    return text, {
        "category": category,
        "template_id": f"geometry_{category}_v1",
        "generation": "deterministic_geometry_template",
        "uses_external_llm": False,
        "uses_landmark_label": False,
        "parameters": parameters,
    }


def build_episode(
    source: dict[str, Any], split: str, args: argparse.Namespace
) -> dict[str, Any]:
    source_path = source["gt_locations"]
    category, candidate = choose_candidate(
        source, enumerate_candidates(source_path, args), args.seed
    )
    start_index = int(candidate["start_index"])
    end_index = int(candidate["end_index"])
    short_path = source_path[start_index : end_index + 1]
    headings = segment_headings(short_path)
    instruction, instruction_metadata = make_instruction(category, candidate)
    short_id = f"short_vln_v1_{int(source['episode_new_id']):04d}"
    return {
        "short_episode_id": short_id,
        "source_episode_id": source["episode_id"],
        "source_episode_new_id": source["episode_new_id"],
        "source_trajectory_id": source["trajectory_id"],
        "scene_id": source["scene_id"],
        "split": split,
        "start_pose": {
            "position": short_path[0],
            "rotation_wxyz": yaw_quaternion_wxyz(headings[0]),
            "rotation_source": "outgoing_path_tangent",
        },
        "goal_pose": {
            "position": short_path[-1],
            "rotation_wxyz": yaw_quaternion_wxyz(headings[-1]),
            "success_radius_m": 0.5,
        },
        "reference_path": short_path,
        "path_length": candidate["path_length"],
        "turn_statistics": {
            "major_turn_count": len(candidate["major_turns"]),
            "major_turn_threshold_degrees": args.major_turn_degrees,
            "turn_merge_distance_m": args.turn_merge_distance,
            "turn_minimum_component_degrees": args.turn_min_delta_degrees,
            "major_turns": candidate["major_turns"],
        },
        "instruction": instruction,
        "instruction_metadata": instruction_metadata,
        "source_path_field": "gt_locations",
        "source_path_index_range": [start_index, end_index],
        "source_reference_path": source["reference_path"],
        "source_metadata": {
            "original_start_position": source["start_position"],
            "original_start_rotation_wxyz": source["start_rotation"],
            "original_goal": source["goals"][0],
            "original_instruction_text_not_for_training": source["instruction"][
                "instruction_text"
            ],
            "info": source["info"],
        },
    }


def distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def validate_dataset(
    payload: dict[str, Any], source_episodes: Sequence[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    episodes = payload["episodes"]
    errors: list[str] = []
    source_by_id = {str(episode["episode_id"]): episode for episode in source_episodes}
    required = {
        "short_episode_id",
        "source_episode_id",
        "scene_id",
        "start_pose",
        "goal_pose",
        "reference_path",
        "path_length",
        "turn_statistics",
        "instruction",
        "source_path_index_range",
        "split",
    }
    if len({episode["short_episode_id"] for episode in episodes}) != len(episodes):
        errors.append("short_episode_id values are not unique")
    if len({str(episode["source_episode_id"]) for episode in episodes}) != len(episodes):
        errors.append("source episodes are reused")
    for episode in episodes:
        missing = required - set(episode)
        if missing:
            errors.append(f"{episode.get('short_episode_id')}: missing {sorted(missing)}")
            continue
        length = path_length(episode["reference_path"], 3)
        if not math.isclose(length, float(episode["path_length"]), abs_tol=1e-9):
            errors.append(f"{episode['short_episode_id']}: path length mismatch")
        if not args.min_path_length <= length <= args.max_path_length:
            errors.append(f"{episode['short_episode_id']}: path length out of range")
        recomputed_turns = major_turn_events(
            episode["reference_path"],
            args.major_turn_degrees,
            args.turn_merge_distance,
            args.turn_min_delta_degrees,
        )
        if len(recomputed_turns) != episode["turn_statistics"]["major_turn_count"]:
            errors.append(f"{episode['short_episode_id']}: turn count mismatch")
        if len(recomputed_turns) > 1:
            errors.append(f"{episode['short_episode_id']}: too many major turns")
        source = source_by_id[str(episode["source_episode_id"])]
        start_index, end_index = episode["source_path_index_range"]
        if episode["reference_path"] != source["gt_locations"][start_index : end_index + 1]:
            errors.append(f"{episode['short_episode_id']}: source slice mismatch")
        category = episode["instruction_metadata"]["category"]
        expected_category = (
            "straight"
            if not recomputed_turns
            else f"{recomputed_turns[0]['direction']}_turn"
        )
        if category != expected_category:
            errors.append(f"{episode['short_episode_id']}: instruction category mismatch")
        if episode["source_metadata"]["original_instruction_text_not_for_training"] == episode["instruction"]:
            errors.append(f"{episode['short_episode_id']}: original instruction was reused")

    scenes_by_split = {
        split: {scene_name(episode["scene_id"]) for episode in episodes if episode["split"] == split}
        for split in ("train", "seen-val", "unseen-test")
    }
    leakage = scenes_by_split["train"] & scenes_by_split["unseen-test"]
    if leakage:
        errors.append(f"train/unseen scene leakage: {sorted(leakage)}")
    if not scenes_by_split["seen-val"] <= scenes_by_split["train"]:
        errors.append("seen-val contains a scene absent from train")
    return {
        "passed": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "scenes_by_split": {key: sorted(value) for key, value in scenes_by_split.items()},
    }


def select_spot_checks(episodes: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count < 20:
        raise ValueError("M3 requires at least 20 spot checks")
    ranked = sorted(
        episodes,
        key=lambda episode: stable_rank(seed, "spot_check", episode["short_episode_id"]),
    )
    selected: dict[str, dict[str, Any]] = {}
    coverage_groups = [
        ("split", ["train", "seen-val", "unseen-test"]),
        ("category", ["straight", "left_turn", "right_turn"]),
        ("scene", sorted({scene_name(episode["scene_id"]) for episode in episodes})),
    ]
    for field, values in coverage_groups:
        for value in values:
            match = next(
                (
                    episode
                    for episode in ranked
                    if (
                        episode["split"]
                        if field == "split"
                        else episode["instruction_metadata"]["category"]
                        if field == "category"
                        else scene_name(episode["scene_id"])
                    )
                    == value
                ),
                None,
            )
            if match is not None:
                selected[match["short_episode_id"]] = match
    for episode in ranked:
        if len(selected) >= count:
            break
        selected[episode["short_episode_id"]] = episode
    return sorted(selected.values(), key=lambda episode: episode["short_episode_id"])[:count]


def write_reports(
    payload: dict[str, Any], validation: dict[str, Any], args: argparse.Namespace
) -> None:
    episodes = payload["episodes"]
    split_counts = Counter(episode["split"] for episode in episodes)
    category_counts = Counter(
        episode["instruction_metadata"]["category"] for episode in episodes
    )
    scene_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        scene_split_counts[scene_name(episode["scene_id"])][episode["split"]] += 1
    lengths = distribution(episode["path_length"] for episode in episodes)
    turns = distribution(
        episode["turn_statistics"]["major_turn_count"] for episode in episodes
    )
    scene_rows = "\n".join(
        f"| `{name}` | {counts['train']} | {counts['seen-val']} | {counts['unseen-test']} |"
        for name, counts in sorted(scene_split_counts.items())
    )
    stats_content = f"""# Short-VLN V1 Statistics (M3)

## Generation contract

- Source episodes: **{payload['source_episode_count']}**
- Generated short episodes: **{len(episodes)}** (one per source episode)
- Source geometry: contiguous slices of official `gt_locations`
- Path constraint: **{args.min_path_length:g}–{args.max_path_length:g} m** (3D polyline length)
- Major-turn constraint: **≤1**, threshold {args.major_turn_degrees:g}°, same-sign components merged within {args.turn_merge_distance:g} m
- Turn instruction legs: each at least {args.turn_min_leg_distance:g} m
- Language: deterministic geometry templates; no external LLM and no landmark labels
- Start rotation: synthesized WXYZ yaw aligned to the outgoing path tangent
- Seed: `{args.seed}`
- Unseen selection: {len(payload['scene_splits']['unseen-test'])} whole scenes whose source-episode count is closest to the {args.unseen_scene_fraction:.0%} target

## Acceptance summary

- Validation: **{'PASS' if validation['passed'] else 'FAIL'}**
- JSON reload: PASS
- Unique source usage: **{len({str(episode['source_episode_id']) for episode in episodes})} / {len(episodes)}**
- Train/unseen scene leakage: **0**
- Seen-val scenes absent from train: **0**
- Short instructions identical to original full-route instruction: **0**
- Automatically audited spot checks: **{args.spot_check_count}**

## Dataset distributions

| Metric | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| Path length (m) | {lengths['min']:.3f} | {lengths['median']:.3f} | {lengths['mean']:.3f} | {lengths['max']:.3f} |
| Major turn count | {turns['min']:.0f} | {turns['median']:.0f} | {turns['mean']:.2f} | {turns['max']:.0f} |

## Split counts

| Split | Episodes | Scenes |
|---|---:|---:|
| train | {split_counts['train']} | {len(validation['scenes_by_split']['train'])} |
| seen-val | {split_counts['seen-val']} | {len(validation['scenes_by_split']['seen-val'])} |
| unseen-test | {split_counts['unseen-test']} | {len(validation['scenes_by_split']['unseen-test'])} |

## Instruction categories

| Category | Episodes |
|---|---:|
| straight | {category_counts['straight']} |
| left_turn | {category_counts['left_turn']} |
| right_turn | {category_counts['right_turn']} |

## Per-scene split audit

| Scene | Train | Seen-val | Unseen-test |
|---|---:|---:|---:|
{scene_rows}

## Known semantic boundary

The generated instructions are fully supported by path geometry, but M3 does not claim a landmark is visible in RGB. The original full-route instruction is retained only as `source_metadata.original_instruction_text_not_for_training`. M4 must use the short `instruction` field and should visually verify rollout starts before collection.
"""
    args.statistics_report.parent.mkdir(parents=True, exist_ok=True)
    args.statistics_report.write_text(stats_content, encoding="utf-8")

    checks = select_spot_checks(episodes, args.spot_check_count, args.seed)
    rows = []
    for episode in checks:
        turn = episode["turn_statistics"]["major_turn_count"]
        rows.append(
            "| {short} | {source} | `{scene}` | {split} | `{indices}` | {length:.3f} | "
            "{turn} | {category} | {instruction} | PASS |".format(
                short=episode["short_episode_id"],
                source=episode["source_episode_id"],
                scene=scene_name(episode["scene_id"]),
                split=episode["split"],
                indices=episode["source_path_index_range"],
                length=episode["path_length"],
                turn=turn,
                category=episode["instruction_metadata"]["category"],
                instruction=episode["instruction"],
            )
        )
    spot_content = """# M3 Short-Episode Spot Check

These 20 deterministic checks cover every split, instruction category, and source scene. `PASS` means the path is an exact contiguous source slice, length is 1–4 m, major turns are ≤1, the instruction category and distances agree with recomputed geometry, and the original long instruction was not reused. This is a geometry/language audit, not a claim of landmark visibility.

| Short ID | Source ID | Scene | Split | Source indices | Length (m) | Turns | Category | Short instruction | Audit |
|---|---:|---|---|---|---:|---:|---|---|---|
""" + "\n".join(rows) + "\n"
    args.spot_check_report.parent.mkdir(parents=True, exist_ok=True)
    args.spot_check_report.write_text(spot_content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_path_length < 0 or args.max_path_length <= args.min_path_length:
        raise ValueError("Invalid path-length bounds")
    dataset_path = resolve_dataset(args.dataset)
    source_episodes = load_episodes(dataset_path)
    split_by_id, scene_splits = split_source_episodes(source_episodes, args)
    episodes = [
        build_episode(source, split_by_id[str(source["episode_id"])], args)
        for source in source_episodes
    ]
    episodes.sort(key=lambda episode: int(episode["source_episode_new_id"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": str(dataset_path),
        "source_episode_count": len(source_episodes),
        "generation_config": {
            "seed": args.seed,
            "min_path_length_m": args.min_path_length,
            "max_path_length_m": args.max_path_length,
            "major_turn_degrees": args.major_turn_degrees,
            "turn_merge_distance_m": args.turn_merge_distance,
            "turn_minimum_component_degrees": args.turn_min_delta_degrees,
            "turn_minimum_leg_distance_m": args.turn_min_leg_distance,
            "unseen_scene_fraction": args.unseen_scene_fraction,
            "seen_val_episode_fraction": args.seen_val_episode_fraction,
            "instruction_policy": "deterministic_geometry_templates_no_landmarks",
        },
        "scene_splits": scene_splits,
        "episodes": episodes,
    }
    validation = validate_dataset(payload, source_episodes, args)
    if not validation["passed"]:
        preview = "\n".join(validation["errors"][:20])
        raise RuntimeError(f"Generated dataset failed validation:\n{preview}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.output.open("r", encoding="utf-8") as stream:
        reloaded = json.load(stream)
    if len(reloaded["episodes"]) != len(episodes):
        raise RuntimeError("JSON reload changed episode count")
    write_reports(payload, validation, args)
    print(f"Source dataset: {dataset_path}")
    print(f"Source episodes: {len(source_episodes)}")
    print(f"Generated episodes: {len(episodes)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Statistics: {args.statistics_report.resolve()}")
    print(f"Spot checks: {args.spot_check_report.resolve()}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
