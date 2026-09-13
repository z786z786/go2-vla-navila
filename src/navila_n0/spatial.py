"""Spatial-cluster split isolation for resampled short routes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _xy(point: Mapping[str, Any]) -> tuple[float, float]:
    xyz = point["xyz"]
    return float(xyz[0]), float(xyz[1])


def _point_distance_xy(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.hypot(ax - bx, ay - by)


def buffered_overlap_length_m(
    path_a: Sequence[Mapping[str, Any]], path_b: Sequence[Mapping[str, Any]], *, buffer_m: float = 0.5
) -> float:
    """Conservative shared centerline length under the stipulated 0.5-m buffer."""
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive")
    if len(path_a) < 2 or len(path_b) < 2:
        return 0.0

    def covered_length(source: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]) -> float:
        total = 0.0
        for first, second in zip(source, source[1:]):
            midpoint = {"xyz": [(float(a) + float(b)) / 2.0 for a, b in zip(first["xyz"], second["xyz"])]}
            if min(_point_distance_xy(midpoint, target) for target in other) <= buffer_m + 1e-9:
                total += float(second["arc_length_m"]) - float(first["arc_length_m"])
        return total

    return min(covered_length(path_a, path_b), covered_length(path_b, path_a))


def cluster_candidates(
    candidates: Sequence[Mapping[str, Any]], *, buffer_m: float = 0.5, overlap_m: float = 0.5
) -> dict[str, int]:
    """Union parent-identical or spatially-overlapping candidates into clusters."""
    if overlap_m <= 0:
        raise ValueError("overlap_m must be positive")
    ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, a in enumerate(candidates):
        for right in range(left + 1, len(candidates)):
            b = candidates[right]
            same_parent = str(a["parent_episode_id"]) == str(b["parent_episode_id"])
            overlap = buffered_overlap_length_m(a["resampled_path"], b["resampled_path"], buffer_m=buffer_m)
            if same_parent or overlap >= overlap_m - 1e-9:
                union(left, right)
    roots = {index: find(index) for index in range(len(candidates))}
    root_to_cluster = {root: cluster for cluster, root in enumerate(sorted(set(roots.values())), start=1)}
    return {ids[index]: root_to_cluster[root] for index, root in roots.items()}


def validate_train_seen_isolation(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject a parent trajectory or spatial cluster that crosses train/seen-val."""
    errors: list[str] = []
    parent_splits: dict[str, set[str]] = {}
    cluster_splits: dict[str, set[str]] = {}
    for record in records:
        split = str(record.get("split"))
        if split not in {"train", "seen-val"}:
            continue
        parent_splits.setdefault(str(record["parent_episode_id"]), set()).add(split)
        cluster_splits.setdefault(str(record["spatial_cluster_id"]), set()).add(split)
    for parent, splits in sorted(parent_splits.items()):
        if len(splits) > 1:
            errors.append(f"parent dense trajectory crosses train/seen-val: {parent}")
    for cluster, splits in sorted(cluster_splits.items()):
        if len(splits) > 1:
            errors.append(f"spatial cluster crosses train/seen-val: {cluster}")
    return errors
