"""Select simple *complete* official NaVILA episodes without rewriting language.

The manifest is source/audit metadata, not a policy dataset.  It keeps each
selected source annotation's original instruction, ``reference_path`` and
``gt_locations``.  The later LeRobot conversion copies only RGB, 3-D velocity
state, 3-D action, and the original instruction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.navila_full.contracts import DATASET_SCHEMA_VERSION, SELECTION_MANIFEST_FORMAT
from src.navila_n0.spatial import cluster_candidates


DEFAULT_SEED = 20260905
DEFAULT_UNSEEN_SCENE = "2azQ1b91cZZ"
CATEGORIES = ("straight", "left_turn", "right_turn")
PRIMARY_RULES = {
    "minimum_reference_length_m": 5.0,
    "maximum_reference_length_m": 10.0,
    "maximum_gt_length_m": 10.0,
    "maximum_height_span_m": 0.05,
    "maximum_instruction_words": 30,
    "maximum_major_turns": 1,
}
REPLACEMENT_RULES = {
    **PRIMARY_RULES,
    "maximum_height_span_m": 0.10,
    "maximum_instruction_words": 40,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def scene_name(scene_id: str) -> str:
    parts = str(scene_id).replace("\\", "/").split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def instruction_text(episode: Mapping[str, Any]) -> str:
    instruction = episode.get("instruction")
    if not isinstance(instruction, Mapping):
        raise ValueError("official episode has no instruction object")
    value = instruction.get("instruction_text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("official episode has an empty instruction_text")
    # Preserve the official annotation byte-for-byte. Whitespace normalization
    # is allowed for filtering/counting only, never for stored policy language
    # or source-provenance binding.
    return value


def _points(path: Any, field: str) -> list[tuple[float, float, float]]:
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or len(path) < 2:
        raise ValueError(f"{field} must contain at least two xyz points")
    result: list[tuple[float, float, float]] = []
    for point in path:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 3:
            raise ValueError(f"{field} contains a non-xyz point")
        xyz = tuple(float(axis) for axis in point)
        if not all(math.isfinite(axis) for axis in xyz):
            raise ValueError(f"{field} contains a non-finite coordinate")
        result.append(xyz)
    return result


def path_length_m(path: Any, field: str) -> float:
    points = _points(path, field)
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def height_span_m(path: Any, field: str) -> float:
    points = _points(path, field)
    return max(point[2] for point in points) - min(point[2] for point in points)


def normalized_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text))


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def major_turns(path: Any) -> list[dict[str, float | str]]:
    """Merge local heading changes into stable, ≥45-degree route events."""
    points = _points(path, "route")
    headings: list[tuple[float, float]] = []
    travelled = 0.0
    for first, second in zip(points, points[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        segment = math.hypot(dx, dy)
        travelled += segment
        if segment > 1e-5:
            headings.append((math.degrees(math.atan2(dy, dx)), travelled))
    changes: list[tuple[float, float]] = []
    for (previous, _), (current, location) in zip(headings, headings[1:]):
        delta = _wrap_degrees(current - previous)
        if abs(delta) >= 12.0:
            changes.append((delta, location))
    groups: list[list[tuple[float, float]]] = []
    for change in changes:
        if groups:
            previous = groups[-1][-1]
            if change[0] * previous[0] > 0 and change[1] - previous[1] <= 1.0:
                groups[-1].append(change)
                continue
        groups.append([change])
    result: list[dict[str, float | str]] = []
    for group in groups:
        angle = sum(change[0] for change in group)
        if abs(angle) < 45.0:
            continue
        weights = [abs(change[0]) for change in group]
        location = sum(change[1] * weight for change, weight in zip(group, weights)) / sum(weights)
        result.append({"direction": "left" if angle > 0 else "right", "signed_angle_degrees": angle, "arc_length_m": location})
    return result


def route_category(reference_path: Any) -> tuple[str, list[dict[str, float | str]]]:
    turns = major_turns(reference_path)
    if not turns:
        return "straight", turns
    if len(turns) != 1:
        raise ValueError("route contains more than one major turn")
    return f"{turns[0]['direction']}_turn", turns


def route_id_for(episode: Mapping[str, Any]) -> str:
    trajectory_id = str(episode.get("trajectory_id", ""))
    if not trajectory_id:
        raise ValueError("official episode has no trajectory_id")
    return _hash({"trajectory_id": trajectory_id, "gt_locations": episode.get("gt_locations")})[:24]


def _resampled_for_spatial(path: Any) -> list[dict[str, Any]]:
    points = _points(path, "reference_path")
    arclength = 0.0
    rows = [{"xyz": list(points[0]), "arc_length_m": 0.0}]
    for first, second in zip(points, points[1:]):
        arclength += math.dist(first, second)
        rows.append({"xyz": list(second), "arc_length_m": arclength})
    return rows


def load_official_episodes(dataset_path: Path) -> list[dict[str, Any]]:
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if set(payload) != {"episodes"} or not isinstance(payload["episodes"], list):
        raise ValueError("official NaVILA dataset must contain exactly an episodes array")
    required = {"episode_id", "trajectory_id", "scene_id", "instruction", "reference_path", "gt_locations"}
    episodes = payload["episodes"]
    for episode in episodes:
        if not isinstance(episode, dict) or not required <= set(episode):
            raise ValueError("official episode misses full-episode provenance fields")
        instruction_text(episode)
        _points(episode["reference_path"], "reference_path")
        _points(episode["gt_locations"], "gt_locations")
    return episodes


def _candidate_from_episode(episode: Mapping[str, Any], rules: Mapping[str, float | int], tier: str) -> dict[str, Any] | None:
    instruction = instruction_text(episode)
    reference_length = path_length_m(episode["reference_path"], "reference_path")
    gt_length = path_length_m(episode["gt_locations"], "gt_locations")
    height_span = max(height_span_m(episode["reference_path"], "reference_path"), height_span_m(episode["gt_locations"], "gt_locations"))
    words = normalized_word_count(instruction)
    try:
        category, turns = route_category(episode["reference_path"])
    except ValueError:
        return None
    if not (
        float(rules["minimum_reference_length_m"]) <= reference_length <= float(rules["maximum_reference_length_m"])
        and gt_length <= float(rules["maximum_gt_length_m"])
        and height_span <= float(rules["maximum_height_span_m"])
        and words <= int(rules["maximum_instruction_words"])
        and len(turns) <= int(rules["maximum_major_turns"])
    ):
        return None
    route_id = route_id_for(episode)
    return {
        "route_id": route_id,
        # Keep the historical deterministic annotation choice while storing
        # the exact source instruction. This avoids changing representatives
        # solely because an official annotation has padding.
        "annotation_id": _hash({"route_id": route_id, "episode_id": str(episode["episode_id"]), "instruction": instruction.strip()})[:24],
        "source_episode_id": str(episode["episode_id"]),
        "source_trajectory_id": str(episode["trajectory_id"]),
        "scene_id": str(episode["scene_id"]),
        "scene_name": scene_name(str(episode["scene_id"])),
        "instruction": instruction,
        "reference_path": episode["reference_path"],
        "gt_locations": episode["gt_locations"],
        "start_position": episode.get("start_position"),
        "start_rotation": episode.get("start_rotation"),
        "goals": episode.get("goals"),
        "episode_new_id": episode.get("episode_new_id"),
        "category": category,
        "turn_audit": turns,
        "route_metrics": {
            "reference_length_m": reference_length,
            "gt_length_m": gt_length,
            "height_span_m": height_span,
            "instruction_word_count": words,
        },
        "filter_tier": tier,
    }


def candidate_routes(episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate paraphrases, retaining a deterministic original annotation per route."""
    primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    replacement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        candidate = _candidate_from_episode(episode, PRIMARY_RULES, "primary")
        if candidate is not None:
            primary[candidate["route_id"]].append(candidate)
            continue
        candidate = _candidate_from_episode(episode, REPLACEMENT_RULES, "replacement")
        if candidate is not None:
            replacement[candidate["route_id"]].append(candidate)
    records: list[dict[str, Any]] = []
    for route_id in sorted(set(primary) | set(replacement)):
        choices = primary.get(route_id) or replacement[route_id]
        selected = sorted(choices, key=lambda item: (item["annotation_id"], item["source_episode_id"]))[0].copy()
        selected["source_annotation_count"] = len(choices)
        selected["source_annotation_ids"] = sorted(item["annotation_id"] for item in choices)
        records.append(selected)
    return records


def attach_spatial_clusters(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cluster only within a Matterport scene; house-local coordinates otherwise collide."""
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in routes:
        row = dict(record)
        row["candidate_id"] = str(row["route_id"])
        row["parent_episode_id"] = str(row["source_trajectory_id"])
        row["resampled_path"] = _resampled_for_spatial(row["reference_path"])
        by_scene[str(row["scene_name"])].append(row)
    output: list[dict[str, Any]] = []
    for name in sorted(by_scene):
        rows = by_scene[name]
        clusters = cluster_candidates(rows, buffer_m=0.5, overlap_m=0.5)
        for row in rows:
            row = dict(row)
            row["spatial_cluster_id"] = f"{name}:{clusters[row['candidate_id']]}"
            row.pop("candidate_id", None)
            row.pop("parent_episode_id", None)
            row.pop("resampled_path", None)
            output.append(row)
    return sorted(output, key=lambda row: str(row["route_id"]))


def _rank(seed: int, record: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|{record['route_id']}|{record['annotation_id']}".encode()).hexdigest()


def _take_category(
    records: Sequence[Mapping[str, Any]], *, category: str, amount: int, seed: int,
    forbidden_clusters: set[str], used_routes: set[str],
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    occupied = set(forbidden_clusters)
    for record in sorted(records, key=lambda row: _rank(seed, row)):
        if record["category"] != category or str(record["route_id"]) in used_routes:
            continue
        cluster = str(record["spatial_cluster_id"])
        if cluster in occupied:
            continue
        picked.append(dict(record))
        used_routes.add(str(record["route_id"]))
        occupied.add(cluster)
        if len(picked) == amount:
            return picked
    raise ValueError(f"insufficient spatially isolated {category} routes for quota {amount}")


def assign_splits(
    successful_routes: Sequence[Mapping[str, Any]], *, seed: int = DEFAULT_SEED,
    unseen_scene: str = DEFAULT_UNSEEN_SCENE, train_per_category: int = 10,
    seen_val_per_category: int = 4, unseen_test_per_category: int = 4,
) -> list[dict[str, Any]]:
    """Make the agreed 30/12/12 route split after PD-success filtering.

    Each route contributes one original human instruction.  Routes sharing a
    spatial cluster never cross train/seen-val, and the held-out house is used
    only for unseen test.
    """
    routes = attach_spatial_clusters(successful_routes)
    seen = [row for row in routes if row["scene_name"] != unseen_scene]
    unseen = [row for row in routes if row["scene_name"] == unseen_scene]
    used: set[str] = set()
    assigned: list[dict[str, Any]] = []
    # Lock seen-val first, then prevent its clusters from supplying train.
    seen_clusters: set[str] = set()
    for category in CATEGORIES:
        rows = _take_category(seen, category=category, amount=seen_val_per_category, seed=seed + 11, forbidden_clusters=seen_clusters, used_routes=used)
        for row in rows:
            row["split"] = "seen-val"
        assigned.extend(rows)
        seen_clusters.update(str(row["spatial_cluster_id"]) for row in rows)
    for category in CATEGORIES:
        rows = _take_category(seen, category=category, amount=train_per_category, seed=seed + 23, forbidden_clusters=seen_clusters, used_routes=used)
        for row in rows:
            row["split"] = "train"
        assigned.extend(rows)
    for category in CATEGORIES:
        rows = _take_category(unseen, category=category, amount=unseen_test_per_category, seed=seed + 37, forbidden_clusters=set(), used_routes=used)
        for row in rows:
            row["split"] = "unseen-test"
        assigned.extend(rows)
    errors = validate_split_records(assigned, unseen_scene=unseen_scene)
    if errors:
        raise RuntimeError("invalid full-episode split:\n" + "\n".join(errors))
    return sorted(assigned, key=lambda row: (str(row["split"]), str(row["route_id"])))


def validate_split_records(records: Sequence[Mapping[str, Any]], *, unseen_scene: str = DEFAULT_UNSEEN_SCENE) -> list[str]:
    errors: list[str] = []
    ids = [str(row.get("route_id", "")) for row in records]
    if len(ids) != len(set(ids)):
        errors.append("a physical route appears more than once")
    by_split_category: dict[tuple[str, str], int] = defaultdict(int)
    for row in records:
        split, category = str(row.get("split")), str(row.get("category"))
        by_split_category[(split, category)] += 1
        if split == "unseen-test" and row.get("scene_name") != unseen_scene:
            errors.append("unseen-test contains a scene other than the held-out house")
        if split in {"train", "seen-val"} and row.get("scene_name") == unseen_scene:
            errors.append("held-out house leaks into train/seen-val")
        if not str(row.get("instruction", "")).strip():
            errors.append("selected route has an empty original instruction")
        if not row.get("reference_path") or not row.get("gt_locations"):
            errors.append("selected route misses source reference_path or gt_locations")
    expected = {"train": 10, "seen-val": 4, "unseen-test": 4}
    for split, quota in expected.items():
        for category in CATEGORIES:
            if by_split_category[(split, category)] != quota:
                errors.append(f"{split}/{category} has {by_split_category[(split, category)]}, expected {quota}")
    parent_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split = str(row.get("split"))
        if split in {"train", "seen-val"}:
            parent_splits[str(row.get("source_trajectory_id"))].add(split)
            cluster_splits[str(row.get("spatial_cluster_id"))].add(split)
    if any(len(splits) > 1 for splits in parent_splits.values()):
        errors.append("a source trajectory crosses train/seen-val")
    if any(len(splits) > 1 for splits in cluster_splits.values()):
        errors.append("a spatial cluster crosses train/seen-val")
    return errors


def build_candidate_manifest(dataset_path: Path, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    episodes = load_official_episodes(dataset_path)
    routes = attach_spatial_clusters(candidate_routes(episodes))
    manifest = {
        "format": SELECTION_MANIFEST_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "stage": "candidate_bank_pending_go2_pd_collection",
        "source_provenance": {
            "official_dataset_path": str(dataset_path),
            "official_dataset_sha256": sha256(dataset_path),
            "source_episode_count": len(episodes),
            "source_fields_preserved": ["instruction.instruction_text", "reference_path", "gt_locations"],
        },
        "selection_contract": {
            "complete_episode_only": True,
            "original_instruction_only": True,
            "unseen_scene": DEFAULT_UNSEEN_SCENE,
            "final_route_quotas": {"train": 30, "seen-val": 12, "unseen-test": 12},
            "final_annotation_policy": "one deterministic original annotation per physical route",
            "spatial_isolation": "0.5m buffer/overlap, clustered independently per scene",
            "canary": "six successful balanced routes are required and excluded before final assignment",
        },
        "filter_tiers": {"primary": PRIMARY_RULES, "replacement": REPLACEMENT_RULES},
        "seed": seed,
        "routes": routes,
    }
    errors = validate_candidate_manifest(manifest)
    if errors:
        raise RuntimeError("invalid full-episode candidate manifest:\n" + "\n".join(errors))
    return manifest


def validate_candidate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("format") != SELECTION_MANIFEST_FORMAT:
        errors.append("wrong full-episode selection manifest format")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("wrong full-episode schema version")
    provenance = manifest.get("source_provenance", {})
    if not isinstance(provenance, Mapping) or not provenance.get("official_dataset_sha256"):
        errors.append("manifest lacks official dataset hash")
    routes = manifest.get("routes", [])
    route_ids: set[str] = set()
    for route in routes if isinstance(routes, list) else []:
        identifier = str(route.get("route_id", ""))
        if not identifier or identifier in route_ids:
            errors.append("route identifiers are missing or duplicated")
        route_ids.add(identifier)
        if route.get("category") not in CATEGORIES:
            errors.append(f"{identifier}: invalid route category")
        if not route.get("instruction") or not route.get("reference_path") or not route.get("gt_locations"):
            errors.append(f"{identifier}: original instruction/reference_path/gt_locations must be retained")
        if not str(route.get("spatial_cluster_id", "")).startswith(str(route.get("scene_name", "")) + ":"):
            errors.append(f"{identifier}: spatial cluster must be scene-local")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_candidate_manifest(args.dataset, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = defaultdict(int)
    for row in manifest["routes"]:
        counts[str(row["category"])] += 1
    print(json.dumps({"output": str(args.output), "route_count": len(manifest["routes"]), "categories": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
