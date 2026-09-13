#!/usr/bin/env python3
"""Plan the next canary round from PD evidence plus the simplest remaining routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.navila_full.finalize_selection import CANARY_MANIFEST_FORMAT, validate_canary_manifest
from src.navila_full.selection import CATEGORIES, DEFAULT_SEED, DEFAULT_UNSEEN_SCENE, assign_splits, sha256, validate_candidate_manifest


def difficulty(route: Mapping[str, Any]) -> tuple[float, float, float, str]:
    metrics = route["route_metrics"]
    start = route["start_position"]
    goal = route["goals"][0]["position"]
    return (
        float(metrics["reference_length_m"]),
        float(metrics["gt_length_m"]),
        math.dist(start, goal),
        str(route["route_id"]),
    )


def plan_evidence_aware_canary(
    candidate_manifest_path: Path, *, proven_success_route_ids: set[str], rejected_pd_route_ids: set[str], seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    errors = validate_candidate_manifest(candidate)
    if errors:
        raise ValueError("invalid candidate manifest: " + "; ".join(errors))
    by_id = {str(route["route_id"]): route for route in candidate["routes"]}
    if not proven_success_route_ids <= set(by_id):
        raise ValueError("a pinned successful route is outside the candidate bank")
    if proven_success_route_ids & rejected_pd_route_ids:
        raise ValueError("a route cannot be both successful and rejected")
    chosen = [dict(by_id[route_id]) for route_id in sorted(proven_success_route_ids)]
    counts = Counter(str(route["category"]) for route in chosen)
    if any(counts[category] > 2 for category in CATEGORIES):
        raise ValueError("pinned evidence overfills a canary category")
    used = set(proven_success_route_ids) | set(rejected_pd_route_ids)
    for category in CATEGORIES:
        for route in sorted(candidate["routes"], key=difficulty):
            if counts[category] == 2:
                break
            route_id = str(route["route_id"])
            if route["category"] != category or route["scene_name"] == DEFAULT_UNSEEN_SCENE or route_id in used:
                continue
            chosen.append(dict(route))
            used.add(route_id)
            counts[category] += 1
        if counts[category] != 2:
            raise RuntimeError(f"not enough simple seen-house routes for {category}")
    held_out = set(rejected_pd_route_ids) | {str(route["route_id"]) for route in chosen}
    # Do not collect an easy-looking route unless it also leaves the final split viable.
    assign_splits([dict(route) for route in candidate["routes"] if str(route["route_id"]) not in held_out], seed=seed)
    manifest = {
        "format": CANARY_MANIFEST_FORMAT,
        "dataset_schema_version": candidate["dataset_schema_version"],
        "stage": "evidence_aware_simplest_route_replan",
        "candidate_manifest_path": str(candidate_manifest_path.resolve()),
        "candidate_manifest_sha256": sha256(candidate_manifest_path),
        "seed": seed,
        "routes": chosen,
        "pinned_previous_go2_pd_success_route_ids": sorted(proven_success_route_ids),
        "rejected_pd_route_ids": sorted(rejected_pd_route_ids),
        "selection_order": "pin Go2-PD successes; fill remaining slots by reference length, GT length, then start-goal distance",
        "route_difficulty": {str(route["route_id"]): {"reference_length_m": difficulty(route)[0], "gt_length_m": difficulty(route)[1], "start_goal_distance_m": difficulty(route)[2]} for route in chosen},
        "acceptance": "all six routes are recollected in one fresh run and must complete Go2 PD collection and sanity checks; rejected and canary routes are excluded from final split assignment",
    }
    validation = validate_canary_manifest(manifest)
    if validation:
        raise RuntimeError("replanned canary is invalid: " + "; ".join(validation))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--proven-success-route-id", action="append", default=[])
    parser.add_argument("--rejected-pd-route-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = plan_evidence_aware_canary(args.candidate_manifest, proven_success_route_ids=set(args.proven_success_route_id), rejected_pd_route_ids=set(args.rejected_pd_route_id), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"routes": [(route["category"], route["source_episode_id"], route["route_id"]) for route in manifest["routes"]], "rejected": manifest["rejected_pd_route_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
