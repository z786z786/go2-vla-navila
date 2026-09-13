#!/usr/bin/env python3
"""Inspect and summarize real VLN-CE-Isaac episodes.

This tool intentionally has no dependency on Isaac Sim.  It reads the gzip JSON
dataset directly, reports the observed schema, computes path geometry, and makes
a top-down path plot that distinguishes the sparse reference path from the dense
expert trajectory (``gt_locations``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATASET_CANDIDATES = (
    Path("<external-data-root>"),
    Path(
        "<external-data-root>"
        "isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz"
    ),
    Path("third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Path to vln_ce_isaac_v1.json.gz")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--episode-id", help="Value of the episode_id field (not list index)")
    selector.add_argument("--episode-index", type=int, help="Zero-based episode list index")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m2"))
    parser.add_argument(
        "--statistics-report",
        type=Path,
        default=Path("reports/episode_statistics.md"),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--major-turn-degrees", type=float, default=45.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--statistics-only", action="store_true")
    return parser.parse_args()


def resolve_dataset(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        return path

    env_path = os.environ.get("VLN_CE_ISAAC_DATASET")
    candidates = ([Path(env_path).expanduser()] if env_path else []) + list(
        DEFAULT_DATASET_CANDIDATES
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find vln_ce_isaac_v1.json.gz. Use --dataset. Searched:\n  "
        + searched
    )


def load_episodes(dataset_path: Path) -> list[dict[str, Any]]:
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if set(payload) != {"episodes"}:
        raise ValueError(f"Expected top-level key 'episodes'; observed {list(payload)}")
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Dataset field 'episodes' must be a non-empty list")
    return episodes


def point_distance(a: Sequence[float], b: Sequence[float], dimensions: int = 3) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(dimensions)))


def path_length(path: Sequence[Sequence[float]], dimensions: int = 3) -> float:
    return sum(point_distance(a, b, dimensions) for a, b in zip(path, path[1:]))


def heading_changes(
    path: Sequence[Sequence[float]], minimum_segment_length: float = 1e-6
) -> list[float]:
    """Return signed XY heading changes in degrees, wrapped to [-180, 180]."""
    headings: list[float] = []
    for start, end in zip(path, path[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) > minimum_segment_length:
            headings.append(math.atan2(dy, dx))
    changes = []
    for previous, current in zip(headings, headings[1:]):
        wrapped = (current - previous + math.pi) % (2 * math.pi) - math.pi
        changes.append(math.degrees(wrapped))
    return changes


def scene_name(scene_id: str) -> str:
    parts = scene_id.rstrip("/").split("/")
    return parts[-1].removesuffix(".glb") if parts else scene_id


def summarize_episode(episode: dict[str, Any], major_turn_degrees: float) -> dict[str, Any]:
    reference_path = episode["reference_path"]
    expert_path = episode.get("gt_locations", [])
    start = episode["start_position"]
    configured_goal = episode["goals"][0]["position"]
    reference_goal = reference_path[-1]
    changes = heading_changes(reference_path)
    major_changes = [change for change in changes if abs(change) >= major_turn_degrees]
    return {
        "episode_id": episode["episode_id"],
        "episode_new_id": episode.get("episode_new_id"),
        "trajectory_id": episode.get("trajectory_id"),
        "scene_id": episode["scene_id"],
        "scene_name": scene_name(episode["scene_id"]),
        "start_position": start,
        "start_rotation_wxyz": episode["start_rotation"],
        "configured_goals": episode["goals"],
        "reference_path_goal": reference_goal,
        "instruction": episode["instruction"]["instruction_text"],
        "instruction_token_count_including_padding": len(
            episode["instruction"].get("instruction_tokens", [])
        ),
        "reference_path_waypoint_count": len(reference_path),
        "reference_path_length_3d_m": path_length(reference_path, 3),
        "reference_path_length_xy_m": path_length(reference_path, 2),
        "expert_waypoint_count": len(expert_path),
        "expert_path_length_3d_m": path_length(expert_path, 3),
        "start_to_reference_goal_3d_m": point_distance(start, reference_goal, 3),
        "start_to_configured_goal_3d_m": point_distance(start, configured_goal, 3),
        "configured_goal_to_reference_goal_3d_m": point_distance(
            configured_goal, reference_goal, 3
        ),
        "heading_changes_degrees": changes,
        "major_turn_threshold_degrees": major_turn_degrees,
        "major_heading_change_count": len(major_changes),
        "major_heading_changes_degrees": major_changes,
        "navigation_metadata": {
            "info": episode.get("info"),
            "gt_forward_steps": episode.get("gt_forward_steps"),
            "gt_action_count": len(episode.get("gt_actions", [])),
            "gt_actions": episode.get("gt_actions"),
        },
    }


def find_episode(
    episodes: Sequence[dict[str, Any]], episode_id: str | None, episode_index: int | None
) -> tuple[int, dict[str, Any]]:
    if episode_index is not None:
        if not -len(episodes) <= episode_index < len(episodes):
            raise IndexError(f"episode index {episode_index} is outside dataset")
        index = episode_index % len(episodes)
        return index, episodes[index]
    wanted = "1" if episode_id is None else str(episode_id)
    matches = [(index, episode) for index, episode in enumerate(episodes) if str(episode.get("episode_id")) == wanted]
    if not matches:
        raise KeyError(f"No episode has episode_id={wanted!r}")
    if len(matches) > 1:
        raise ValueError(f"episode_id={wanted!r} is not unique ({len(matches)} matches)")
    return matches[0]


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1 - weight) + float(sorted_values[upper]) * weight


def distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def observed_schema(episodes: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    episode_keys = sorted({key for episode in episodes for key in episode})
    instruction_keys = sorted(
        {key for episode in episodes for key in episode.get("instruction", {})}
    )
    goal_keys = sorted(
        {
            key
            for episode in episodes
            for goal in episode.get("goals", [])
            for key in goal
        }
    )
    info_keys = sorted({key for episode in episodes for key in episode.get("info", {})})
    return {
        "episode": episode_keys,
        "instruction": instruction_keys,
        "goal": goal_keys,
        "info": info_keys,
    }


def format_table_row(label: str, stats: dict[str, float], decimals: int) -> str:
    values = [stats[key] for key in ("min", "p25", "median", "mean", "p75", "max")]
    return "| " + label + " | " + " | ".join(f"{value:.{decimals}f}" for value in values) + " |"


def write_statistics_report(
    report_path: Path,
    dataset_path: Path,
    episodes: Sequence[dict[str, Any]],
    sample_size: int,
    seed: int,
    major_turn_degrees: float,
) -> list[dict[str, Any]]:
    if sample_size < 50:
        raise ValueError("M2 requires --sample-size >= 50")
    actual_size = min(sample_size, len(episodes))
    rng = random.Random(seed)
    indices_by_scene: dict[str, list[int]] = {}
    for index, episode in enumerate(episodes):
        indices_by_scene.setdefault(scene_name(str(episode["scene_id"])), []).append(index)
    if actual_size >= len(indices_by_scene):
        # Make scene coverage deterministic even when a scene has very few episodes.
        selected_set = {
            rng.choice(indices) for _, indices in sorted(indices_by_scene.items())
        }
        remaining = [index for index in range(len(episodes)) if index not in selected_set]
        selected_set.update(rng.sample(remaining, actual_size - len(selected_set)))
        selected_indices = sorted(selected_set)
        sampling_method = "scene-stratified random sample"
    else:
        selected_indices = sorted(rng.sample(range(len(episodes)), actual_size))
        sampling_method = "random sample"
    summaries = [summarize_episode(episodes[index], major_turn_degrees) for index in selected_indices]
    scene_counts = Counter(summary["scene_name"] for summary in summaries)
    all_scenes = {scene_name(str(episode["scene_id"])) for episode in episodes}

    metrics = {
        "Reference path length, 3D (m)": distribution(
            summary["reference_path_length_3d_m"] for summary in summaries
        ),
        "Reference waypoint count": distribution(
            summary["reference_path_waypoint_count"] for summary in summaries
        ),
        "Major turn count": distribution(
            summary["major_heading_change_count"] for summary in summaries
        ),
        "Expert waypoint count": distribution(
            summary["expert_waypoint_count"] for summary in summaries
        ),
        "Start-goal distance, 3D (m)": distribution(
            summary["start_to_reference_goal_3d_m"] for summary in summaries
        ),
    }
    schema = observed_schema(episodes)
    episode_key_sets = {tuple(sorted(episode)) for episode in episodes}
    instruction_key_sets = {
        tuple(sorted(episode.get("instruction", {}))) for episode in episodes
    }
    goal_key_sets = {
        tuple(sorted(goal))
        for episode in episodes
        for goal in episode.get("goals", [])
    }
    ids = [str(episode.get("episode_id")) for episode in episodes]
    empty_instructions = sum(
        not str(episode.get("instruction", {}).get("instruction_text", "")).strip()
        for episode in episodes
    )
    short_reference_paths = sum(len(episode.get("reference_path", [])) < 2 for episode in episodes)
    short_expert_paths = sum(len(episode.get("gt_locations", [])) < 2 for episode in episodes)
    start_mismatches = sum(
        point_distance(episode["start_position"], episode["reference_path"][0], 3) > 1e-6
        for episode in episodes
        if episode.get("reference_path")
    )
    goal_mismatches = sum(
        point_distance(episode["goals"][0]["position"], episode["reference_path"][-1], 3)
        > 1e-6
        for episode in episodes
        if episode.get("goals") and episode.get("reference_path")
    )
    sampled_ids = ", ".join(str(summary["episode_id"]) for summary in summaries)
    scene_rows = "\n".join(
        f"| `{name}` | {count} |" for name, count in sorted(scene_counts.items())
    )
    metric_rows = "\n".join(
        format_table_row(name, stats, 2 if "count" not in name.lower() else 1)
        for name, stats in metrics.items()
    )
    content = f"""# VLN-CE-Isaac Episode Statistics (M2)

## Dataset and method

- Dataset: `{dataset_path}`
- Dataset episodes: **{len(episodes)}**
- Dataset scenes: **{len(all_scenes)}**
- Deterministic {sampling_method}: **{actual_size} episodes**, seed `{seed}`
- Reference path length: sum of consecutive 3D Euclidean waypoint distances.
- Major turn: absolute wrapped change between consecutive non-zero **XY** segment headings, threshold **{major_turn_degrees:g} degrees**. Height changes are not turns.
- `reference_path` is the sparse task/reference route. `gt_locations` is the dense expert route consumed by the official PD planner and distance metric; they are reported separately.

## Observed schema

The gzip JSON top level contains only `episodes`. Fields below are the union observed over all {len(episodes)} episodes, not guessed names.

- Episode: `{', '.join(schema['episode'])}`
- Instruction: `{', '.join(schema['instruction'])}`
- Goal: `{', '.join(schema['goal'])}`
- Info: `{', '.join(schema['info'])}`

## Dataset integrity checks

| Check | Result |
|---|---:|
| Unique `episode_id` values | {len(set(ids))} / {len(ids)} |
| Distinct episode key sets | {len(episode_key_sets)} |
| Distinct instruction key sets | {len(instruction_key_sets)} |
| Distinct goal key sets | {len(goal_key_sets)} |
| Empty instruction text | {empty_instructions} |
| Reference paths with fewer than 2 points | {short_reference_paths} |
| Expert paths with fewer than 2 points | {short_expert_paths} |
| Start vs `reference_path[0]` mismatch (>1e-6 m) | {start_mismatches} |
| `goals[0]` vs `reference_path[-1]` mismatch (>1e-6 m) | {goal_mismatches} |

## Sample distributions

| Metric | Min | P25 | Median | Mean | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|
{metric_rows}

## Scene coverage in sample

The sample covers **{len(scene_counts)} scenes**.

| Scene | Episodes |
|---|---:|
{scene_rows}

## Fields needed by short-VLN M3

- Stable provenance: `episode_id`, `episode_new_id`, `trajectory_id`, `scene_id`.
- Start pose: `start_position`, `start_rotation` (official code passes it as WXYZ).
- Goal: `goals[0].position`; the official demo also places its goal marker at `reference_path[-1]`, so both must be retained and consistency-checked.
- Sparse geometry: `reference_path` for segment length and turn filtering.
- Dense expert geometry: `gt_locations`, plus `gt_actions` and `gt_forward_steps` for audit only.
- Language: `instruction.instruction_text`; it describes the full original route and must not be reused unchanged after arbitrary path cropping.
- Navigation metadata: `info.geodesic_distance`, goal `radius`.

## Deterministic sample IDs

{sampled_ids}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return summaries


def plot_episode(
    episode: dict[str, Any], summary: dict[str, Any], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = episode["reference_path"]
    expert = episode.get("gt_locations", [])
    configured_goal = episode["goals"][0]["position"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    if expert:
        ax.plot(
            [point[0] for point in expert],
            [point[1] for point in expert],
            color="#999999",
            linewidth=1.4,
            label=f"gt_locations ({len(expert)})",
            zorder=1,
        )
    ax.plot(
        [point[0] for point in reference],
        [point[1] for point in reference],
        "o-",
        color="#1565c0",
        linewidth=2.3,
        label=f"reference_path ({len(reference)})",
        zorder=2,
    )
    for number, point in enumerate(reference):
        ax.annotate(str(number), (point[0], point[1]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.scatter(reference[0][0], reference[0][1], marker="*", s=180, color="#2e7d32", label="start", zorder=4)
    ax.scatter(reference[-1][0], reference[-1][1], marker="*", s=180, color="#c62828", label="reference goal", zorder=4)
    ax.scatter(configured_goal[0], configured_goal[1], marker="x", s=100, color="#ef6c00", label="goals[0]", zorder=4)
    ax.set_title("Top-down XY path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    cumulative = [0.0]
    for start, end in zip(reference, reference[1:]):
        cumulative.append(cumulative[-1] + point_distance(start, end, 3))
    ax.plot(cumulative, [point[2] for point in reference], "o-", color="#6a1b9a")
    ax.set_title("Reference-path height profile")
    ax.set_xlabel("Cumulative 3D distance (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Episode {summary['episode_id']} | {summary['scene_name']} | "
        f"{summary['reference_path_length_3d_m']:.2f} m | "
        f"major turns={summary['major_heading_change_count']}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset(args.dataset)
    episodes = load_episodes(dataset_path)
    sampled = write_statistics_report(
        args.statistics_report,
        dataset_path,
        episodes,
        args.sample_size,
        args.seed,
        args.major_turn_degrees,
    )
    print(f"Dataset: {dataset_path}")
    print(f"Episodes: {len(episodes)}")
    print(f"Statistics sample: {len(sampled)}")
    print(f"Statistics report: {args.statistics_report.resolve()}")

    if args.statistics_only:
        return
    index, episode = find_episode(episodes, args.episode_id, args.episode_index)
    summary = summarize_episode(episode, args.major_turn_degrees)
    print(f"Episode list index: {index}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    safe_id = str(episode["episode_id"]).replace("/", "_")
    metadata_path = args.output_dir / f"episode_{safe_id}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Metadata JSON: {metadata_path.resolve()}")
    if not args.no_plot:
        output_path = args.output_dir / f"episode_{safe_id}_path.png"
        plot_episode(episode, summary, output_path)
        print(f"Path visualization: {output_path.resolve()}")


if __name__ == "__main__":
    main()
