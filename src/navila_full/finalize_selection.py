#!/usr/bin/env python3
"""Bind PD-successful collections to the canary and final 30/12/12 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from src.collector.full_episode_contract import validate_collection_provenance
from src.navila_full.contracts import DATASET_SCHEMA_VERSION
from src.navila_full.selection import CATEGORIES, DEFAULT_SEED, DEFAULT_UNSEEN_SCENE, assign_splits, sha256, validate_candidate_manifest, validate_split_records


FINAL_MANIFEST_FORMAT = "navila-full-episode-final-split-v1"
CANARY_MANIFEST_FORMAT = "navila-full-episode-pd-canary-v1"


def _rank(seed: int, route: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|{route['route_id']}|{route['source_episode_id']}".encode()).hexdigest()


def select_canary(routes: list[Mapping[str, Any]], *, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Choose six *seen-house* routes while proving the final split remains viable.

    The held-out scene has exactly the final unseen-test quota, so it is never
    consumed by the canary.  Repeated deterministic rankings handle the case
    where an early spatial cluster choice would starve the later 30/12/12 split.
    """
    seen = [route for route in routes if route.get("scene_name") != DEFAULT_UNSEEN_SCENE]
    for attempt in range(4096):
        result: list[dict[str, Any]] = []
        used: set[str] = set()
        for category in CATEGORIES:
            selected = []
            for route in sorted(
                (item for item in seen if item.get("category") == category),
                key=lambda item: _rank(seed + attempt, item),
            ):
                if str(route["route_id"]) in used:
                    continue
                selected.append(dict(route))
                used.add(str(route["route_id"]))
                if len(selected) == 2:
                    break
            if len(selected) != 2:
                break
            result.extend(selected)
        if len(result) != 6:
            continue
        try:
            assign_splits(
                [dict(route) for route in routes if str(route["route_id"]) not in used],
                seed=seed,
            )
        except (RuntimeError, ValueError):
            continue
        return result
    raise ValueError("could not reserve a balanced canary without starving the final spatial split")


def canary_manifest(candidate_manifest_path: Path, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    errors = validate_candidate_manifest(candidate)
    if errors:
        raise ValueError("invalid candidate manifest: " + "; ".join(errors))
    routes = select_canary(candidate["routes"], seed=seed)
    return {
        "format": CANARY_MANIFEST_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "stage": "planned_before_pd_collection",
        "candidate_manifest_path": str(candidate_manifest_path.resolve()),
        "candidate_manifest_sha256": sha256(candidate_manifest_path),
        "seed": seed,
        "routes": routes,
        "acceptance": "all six routes must complete Go2 PD collection and sanity checks; they are excluded from final splits",
    }


def validate_canary_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("format") != CANARY_MANIFEST_FORMAT:
        errors.append("wrong canary manifest format")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("wrong full-episode schema")
    routes = manifest.get("routes")
    if not isinstance(routes, list):
        return [*errors, "canary routes must be a list"]
    counts: dict[str, int] = defaultdict(int)
    identifiers: set[str] = set()
    for row in routes:
        counts[str(row.get("category"))] += 1
        identifiers.add(str(row.get("route_id")))
        if row.get("scene_name") == DEFAULT_UNSEEN_SCENE:
            errors.append("canary must not consume the held-out unseen scene")
    if len(identifiers) != 6:
        errors.append("canary must contain six distinct physical routes")
    for category in CATEGORIES:
        if counts[category] != 2:
            errors.append(f"canary has {counts[category]} {category} routes, expected two")
    return errors


def successful_collections(collection_root: Path, candidate_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept only source-bound, sanity-passing, successful Go2 PD rollouts."""
    source_hash = str(candidate_manifest["source_provenance"]["official_dataset_sha256"])
    candidate_by_episode = {str(row["source_episode_id"]): row for row in candidate_manifest["routes"]}
    accepted: list[dict[str, Any]] = []
    provenance_paths: list[Path] = []
    # RGB directories contain thousands of files per episode. Prune them from
    # evidence discovery so repeated incremental finalization stays bounded.
    for current, directories, files in os.walk(collection_root):
        directories[:] = [name for name in directories if name != "front_rgb"]
        if "full_episode_provenance.json" in files:
            provenance_paths.append(Path(current) / "full_episode_provenance.json")
    for provenance_path in sorted(provenance_paths):
        root = provenance_path.parent
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            sanity = json.loads((root / "sanity.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if validate_collection_provenance(provenance):
            continue
        source_episode_id = str(provenance["source_episode_id"])
        candidate = candidate_by_episode.get(source_episode_id)
        if candidate is None:
            continue
        if (
            provenance.get("source_dataset_sha256") != source_hash
            or provenance.get("route_id") != candidate.get("route_id")
            or provenance.get("original_instruction") != candidate.get("instruction")
            or summary.get("collection_format") != provenance.get("format")
            or summary.get("status") != "complete" or summary.get("success") is not True
            or sanity.get("passed") is not True
            or not isinstance(summary.get("expert_path_marker_guard"), Mapping)
            or summary["expert_path_marker_guard"].get("passed") is not True
        ):
            continue
        accepted.append({**candidate, "collection_dir": str(root.resolve()), "collection_summary_sha256": sha256(root / "summary.json")})
    seen_routes: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in sorted(accepted, key=lambda item: str(item["collection_dir"])):
        route_id = str(row["route_id"])
        if route_id not in seen_routes:
            unique.append(row)
            seen_routes.add(route_id)
    return unique


def final_manifest(candidate_manifest_path: Path, collection_root: Path, canary_manifest_path: Path, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    errors = validate_candidate_manifest(candidate_manifest)
    if errors:
        raise ValueError("invalid candidate manifest: " + "; ".join(errors))
    planned_canary = json.loads(canary_manifest_path.read_text(encoding="utf-8"))
    canary_errors = validate_canary_manifest(planned_canary)
    if canary_errors:
        raise ValueError("invalid canary manifest: " + "; ".join(canary_errors))
    if planned_canary.get("candidate_manifest_sha256") != sha256(candidate_manifest_path):
        raise ValueError("canary belongs to another candidate manifest")
    successful = successful_collections(collection_root, candidate_manifest)
    successful_by_route = {str(row["route_id"]): row for row in successful}
    canary_ids = {str(row["route_id"]) for row in planned_canary["routes"]}
    missing_canary = sorted(canary_ids - set(successful_by_route))
    if missing_canary:
        raise ValueError(f"planned canary has not passed collection/sanity: {missing_canary}")
    canary = [successful_by_route[route_id] for route_id in sorted(canary_ids)]
    splits = assign_splits([row for row in successful if str(row["route_id"]) not in canary_ids], seed=seed)
    errors = validate_split_records(splits)
    if errors:
        raise RuntimeError("final split failed validation: " + "; ".join(errors))
    return {
        "format": FINAL_MANIFEST_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "stage": "approved_for_fresh_full_episode_conversion",
        "candidate_manifest_path": str(candidate_manifest_path.resolve()),
        "candidate_manifest_sha256": sha256(candidate_manifest_path),
        "canary_manifest_path": str(canary_manifest_path.resolve()),
        "canary_manifest_sha256": sha256(canary_manifest_path),
        "collection_root": str(collection_root.resolve()),
        "seed": seed,
        "canary": canary,
        "splits": splits,
        "policy_data_source": "only full_episode_provenance-bound successful Go2 PD collections",
    }


def validate_final_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("format") != FINAL_MANIFEST_FORMAT:
        errors.append("wrong final full-episode manifest format")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("wrong full-episode schema")
    if manifest.get("stage") != "approved_for_fresh_full_episode_conversion":
        errors.append("manifest is not approved for conversion")
    canary = manifest.get("canary")
    splits = manifest.get("splits")
    if not isinstance(canary, list) or not isinstance(splits, list):
        return [*errors, "canary/splits must be lists"]
    counts: dict[str, int] = defaultdict(int)
    for row in canary:
        counts[str(row.get("category"))] += 1
    for category in CATEGORIES:
        if counts[category] != 2:
            errors.append(f"canary has {counts[category]} {category} routes, expected two")
    canary_ids = {str(row.get("route_id")) for row in canary}
    split_ids = {str(row.get("route_id")) for row in splits}
    if canary_ids & split_ids:
        errors.append("canary routes leak into final split")
    errors.extend(validate_split_records(splits))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("canary", "final"), default="final")
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument("--canary-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "canary":
        manifest = canary_manifest(args.candidate_manifest, seed=args.seed)
        errors = validate_canary_manifest(manifest)
    else:
        if args.collection_root is None or args.canary_manifest is None:
            raise ValueError("final stage requires --collection-root and --canary-manifest")
        manifest = final_manifest(args.candidate_manifest, args.collection_root, args.canary_manifest, seed=args.seed)
        errors = validate_final_manifest(manifest)
    if errors:
        raise RuntimeError("invalid final manifest: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.stage == "canary":
        print(json.dumps({"output": str(args.output), "canary": len(manifest["routes"])}, ensure_ascii=False))
    else:
        print(json.dumps({"output": str(args.output), "canary": len(manifest["canary"]), "final_routes": len(manifest["splits"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
