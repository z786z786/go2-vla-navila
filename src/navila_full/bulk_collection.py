#!/usr/bin/env python3
"""Build a balanced, resumable queue for full-episode Go2 PD collection."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.navila_full.finalize_selection import validate_canary_manifest
from src.navila_full.selection import (
    CATEGORIES,
    DEFAULT_UNSEEN_SCENE,
    assign_splits,
    sha256,
    validate_candidate_manifest,
)


BULK_QUEUE_FORMAT = "navila-full-episode-bulk-queue-v1"
SCOPES = ("unseen", "seen")


def route_difficulty(route: Mapping[str, Any]) -> tuple[float, float, float, str]:
    metrics = route["route_metrics"]
    start = route["start_position"]
    goal = route["goals"][0]["position"]
    return (
        float(metrics["reference_length_m"]),
        float(metrics["gt_length_m"]),
        math.dist(start, goal),
        str(route["route_id"]),
    )


def _scope(route: Mapping[str, Any]) -> str:
    return "unseen" if route.get("scene_name") == DEFAULT_UNSEEN_SCENE else "seen"


def plan_bulk_queue(candidate_manifest_path: Path, canary_manifest_path: Path) -> dict[str, Any]:
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    canary = json.loads(canary_manifest_path.read_text(encoding="utf-8"))
    candidate_errors = validate_candidate_manifest(candidate)
    canary_errors = validate_canary_manifest(canary)
    if candidate_errors or canary_errors:
        raise ValueError("invalid candidate/canary manifest")
    if canary.get("candidate_manifest_sha256") != sha256(candidate_manifest_path):
        raise ValueError("canary belongs to another candidate manifest")

    canary_ids = {str(route["route_id"]) for route in canary["routes"]}
    rejected_ids = {str(route_id) for route_id in canary.get("rejected_pd_route_ids", [])}
    if canary_ids & rejected_ids:
        raise ValueError("canary and rejected route IDs overlap")
    excluded = canary_ids | rejected_ids
    remaining = [dict(route) for route in candidate["routes"] if str(route["route_id"]) not in excluded]
    # Prove the complete queue still contains at least one valid final split.
    assign_splits(remaining, seed=int(canary["seed"]))

    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = {}
    for scope in SCOPES:
        for category in CATEGORIES:
            routes = sorted(
                (route for route in remaining if _scope(route) == scope and route["category"] == category),
                key=route_difficulty,
            )
            buckets[(scope, category)] = deque(routes)

    queue: list[dict[str, Any]] = []
    # Round-robin prevents easy routes from one category or scene scope from
    # starving the other five quota buckets after PD failures.
    while any(buckets.values()):
        for scope in SCOPES:
            for category in CATEGORIES:
                bucket = buckets[(scope, category)]
                if bucket:
                    route = bucket.popleft()
                    queue.append({**route, "collection_scope": scope, "difficulty": list(route_difficulty(route)[:3])})

    manifest = {
        "format": BULK_QUEUE_FORMAT,
        "dataset_schema_version": candidate["dataset_schema_version"],
        "stage": "approved_after_go2_pd_canary",
        "candidate_manifest_path": str(candidate_manifest_path.resolve()),
        "candidate_manifest_sha256": sha256(candidate_manifest_path),
        "canary_manifest_path": str(canary_manifest_path.resolve()),
        "canary_manifest_sha256": sha256(canary_manifest_path),
        "seed": int(canary["seed"]),
        "excluded_canary_route_ids": sorted(canary_ids),
        "excluded_rejected_pd_route_ids": sorted(rejected_ids),
        "ordering": "round-robin unseen/seen x straight/left_turn/right_turn; simplest first within each bucket",
        "routes": queue,
    }
    errors = validate_bulk_queue(manifest)
    if errors:
        raise RuntimeError("invalid bulk queue: " + "; ".join(errors))
    return manifest


def validate_bulk_queue(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("format") != BULK_QUEUE_FORMAT:
        errors.append("wrong bulk queue format")
    if manifest.get("stage") != "approved_after_go2_pd_canary":
        errors.append("bulk queue is not approved")
    routes = manifest.get("routes")
    if not isinstance(routes, list):
        return [*errors, "bulk routes must be a list"]
    ids = [str(route.get("route_id")) for route in routes]
    excluded = {
        *(str(route_id) for route_id in manifest.get("excluded_canary_route_ids", [])),
        *(str(route_id) for route_id in manifest.get("excluded_rejected_pd_route_ids", [])),
    }
    if len(ids) != len(set(ids)):
        errors.append("bulk queue contains duplicate physical routes")
    if set(ids) & excluded:
        errors.append("bulk queue contains canary or rejected routes")
    for route in routes:
        if route.get("category") not in CATEGORIES or route.get("collection_scope") not in SCOPES:
            errors.append(f"{route.get('route_id')}: invalid category or scene scope")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = plan_bulk_queue(args.candidate_manifest, args.canary_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = defaultdict(int)
    for route in manifest["routes"]:
        counts[f"{route['collection_scope']}/{route['category']}"] += 1
    print(json.dumps({"output": str(args.output), "routes": len(manifest["routes"]), "buckets": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
