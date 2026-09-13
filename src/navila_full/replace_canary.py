#!/usr/bin/env python3
"""Replace one failed Go2-PD canary route without weakening final split gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.navila_full.finalize_selection import CANARY_MANIFEST_FORMAT, validate_canary_manifest
from src.navila_full.selection import DEFAULT_SEED, DEFAULT_UNSEEN_SCENE, assign_splits, sha256, validate_candidate_manifest


def _rank(seed: int, route: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|replacement|{route['route_id']}".encode()).hexdigest()


def replace_failed_canary(
    candidate_manifest_path: Path, previous_canary_path: Path, failed_route_id: str, *, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    previous = json.loads(previous_canary_path.read_text(encoding="utf-8"))
    candidate_errors = validate_candidate_manifest(candidate)
    previous_errors = validate_canary_manifest(previous)
    if candidate_errors or previous_errors:
        raise ValueError("invalid source manifest")
    if previous.get("candidate_manifest_sha256") != sha256(candidate_manifest_path):
        raise ValueError("previous canary belongs to a different candidate bank")
    failed = [route for route in previous["routes"] if str(route["route_id"]) == failed_route_id]
    if len(failed) != 1:
        raise ValueError("failed route must occur exactly once in the previous canary")
    failed_route = failed[0]
    retained = [dict(route) for route in previous["routes"] if str(route["route_id"]) != failed_route_id]
    retained_ids = {str(route["route_id"]) for route in retained}
    for candidate_route in sorted(candidate["routes"], key=lambda route: _rank(seed, route)):
        route_id = str(candidate_route["route_id"])
        if (
            route_id == failed_route_id
            or route_id in retained_ids
            or candidate_route["category"] != failed_route["category"]
            or candidate_route["scene_name"] == DEFAULT_UNSEEN_SCENE
        ):
            continue
        chosen = [*retained, dict(candidate_route)]
        held_out = {failed_route_id, *(str(route["route_id"]) for route in chosen)}
        try:
            # Prove the rejected route and all six canary routes still leave a
            # viable final 30/12/12 physical split before doing any new PD run.
            assign_splits(
                [dict(route) for route in candidate["routes"] if str(route["route_id"]) not in held_out],
                seed=seed,
            )
        except (RuntimeError, ValueError):
            continue
        manifest = {
            "format": CANARY_MANIFEST_FORMAT,
            "dataset_schema_version": candidate["dataset_schema_version"],
            "stage": "replacement_after_failed_go2_pd_canary",
            "candidate_manifest_path": str(candidate_manifest_path.resolve()),
            "candidate_manifest_sha256": sha256(candidate_manifest_path),
            "seed": seed,
            "routes": chosen,
            "supersedes_canary_manifest": str(previous_canary_path.resolve()),
            "supersedes_canary_manifest_sha256": sha256(previous_canary_path),
            "rejected_pd_route_ids": [failed_route_id],
            "replacement_for": {"route_id": failed_route_id, "category": failed_route["category"], "replacement_route_id": route_id},
            "acceptance": "all six routes must complete Go2 PD collection and sanity checks; rejected and canary routes are excluded from final split assignment",
        }
        errors = validate_canary_manifest(manifest)
        if errors:
            raise RuntimeError("replacement canary failed validation: " + "; ".join(errors))
        return manifest
    raise RuntimeError("no same-category seen-house replacement preserves the final split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--previous-canary-manifest", type=Path, required=True)
    parser.add_argument("--failed-route-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = replace_failed_canary(args.candidate_manifest, args.previous_canary_manifest, args.failed_route_id, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["replacement_for"], ensure_ascii=False))


if __name__ == "__main__":
    main()
