"""Build a fail-closed N1 geometry candidate bank from official gt_locations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from src.navila_n0.contracts import DATASET_SCHEMA_VERSION
from src.navila_n0.geometry import RouteRules, select_geometry_precandidates


MANIFEST_FORMAT = "navila-semantic-short-vln-n1-geometry-candidates-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_name(scene_id: str) -> str:
    values = str(scene_id).replace("\\", "/").split("/")
    return values[1] if len(values) > 1 else values[0]


def load_official_episodes(dataset_path: Path) -> list[dict[str, Any]]:
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if set(payload) != {"episodes"} or not isinstance(payload["episodes"], list):
        raise ValueError("official NaVILA dataset must contain exactly an episodes array")
    episodes = payload["episodes"]
    required = {"episode_id", "scene_id", "gt_locations"}
    for episode in episodes:
        if not isinstance(episode, dict) or not required <= set(episode):
            raise ValueError("official episode is missing gt_locations provenance fields")
    return episodes


def _navila_commit(navila_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(navila_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_geometry_manifest(
    dataset_path: Path,
    navila_root: Path,
    *,
    seed: int = 20260905,
    max_candidates_per_parent: int = 4,
    rules: RouteRules = RouteRules(),
) -> dict[str, Any]:
    """Create N1 pre-candidates; no semantic or policy data is accepted here."""
    dataset_path = dataset_path.resolve()
    navila_root = navila_root.resolve()
    episodes = load_official_episodes(dataset_path)
    candidates: list[dict[str, Any]] = []
    parent_without_geometry: list[str] = []
    for episode in episodes:
        parent_id = str(episode["episode_id"])
        selected = select_geometry_precandidates(
            parent_id,
            episode["gt_locations"],
            maximum_candidates=max_candidates_per_parent,
            selection_seed=seed,
            rules=rules,
        )
        if not selected:
            parent_without_geometry.append(parent_id)
            continue
        for candidate in selected:
            candidate["scene_id"] = str(episode["scene_id"])
            candidate["scene_name"] = _scene_name(str(episode["scene_id"]))
        candidates.extend(selected)
    manifest = {
        "format": MANIFEST_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "stage": "N1_geometry_preview_bank",
        "acceptance_status": "not_training_data_pending_surface_and_semantic_preview",
        "source_provenance": {
            "official_dataset_path": str(dataset_path),
            "official_dataset_sha256": sha256(dataset_path),
            "navila_root": str(navila_root),
            "navila_bench_commit": _navila_commit(navila_root),
            "source_path_field": "gt_locations",
            "source_episode_count": len(episodes),
        },
        "generation_config": {
            "seed": seed,
            "max_candidates_per_parent": max_candidates_per_parent,
            "route_rules": vars(rules),
            "policy_data_written": False,
            "legacy_m6_m7_d5_input_allowed": False,
        },
        "candidates": candidates,
        "parents_without_geometry_precandidate": parent_without_geometry,
    }
    errors = validate_geometry_manifest(manifest)
    if errors:
        raise RuntimeError("invalid N1 geometry manifest:\n" + "\n".join(errors[:20]))
    return manifest


def validate_geometry_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("format") != MANIFEST_FORMAT:
        errors.append("wrong N1 geometry manifest format")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("wrong dataset schema version")
    if manifest.get("acceptance_status") != "not_training_data_pending_surface_and_semantic_preview":
        errors.append("N1 manifest must not claim accepted training data")
    provenance = manifest.get("source_provenance", {})
    if provenance.get("source_path_field") != "gt_locations" or not provenance.get("official_dataset_sha256"):
        errors.append("source provenance must bind official gt_locations and its dataset hash")
    identifiers: set[str] = set()
    for candidate in manifest.get("candidates", []):
        identifier = str(candidate.get("candidate_id", ""))
        if not identifier or identifier in identifiers:
            errors.append("candidate identifiers are missing or duplicate")
        identifiers.add(identifier)
        if candidate.get("acceptance_status") != "pending_surface_and_semantic_preview":
            errors.append(f"{identifier}: candidate is not fail-closed pending preview")
        if candidate.get("instruction") or candidate.get("task"):
            errors.append(f"{identifier}: geometry manifest may not contain an instruction/task")
        if candidate.get("flatness_audit", {}).get("passed") is not False:
            errors.append(f"{identifier}: surface flatness cannot be accepted before preview")
        if len(candidate.get("dense_index_range", [])) != 2 or not candidate.get("parent_dense_path_sha256"):
            errors.append(f"{identifier}: missing dense-path audit")
        if not 1.5 <= float(candidate.get("path_length_m", 0.0)) <= 4.0:
            errors.append(f"{identifier}: path length violates N1 range")
        if candidate.get("category") not in {"straight", "left_turn", "right_turn"}:
            errors.append(f"{identifier}: invalid route category")
    return errors


def write_report(manifest: dict[str, Any], output_path: Path) -> None:
    candidates = manifest["candidates"]
    category_counts = Counter(candidate["category"] for candidate in candidates)
    scene_counts = Counter(candidate["scene_name"] for candidate in candidates)
    lines = [
        "# N1 Geometry Preview Bank",
        "",
        "Status: **PASS (geometry only; not training data)**.",
        "",
        f"- Official dataset SHA-256: `{manifest['source_provenance']['official_dataset_sha256']}`",
        f"- NaVILA-Bench commit: `{manifest['source_provenance']['navila_bench_commit']}`",
        f"- Parents scanned: **{manifest['source_provenance']['source_episode_count']}**",
        f"- Pending geometry pre-candidates: **{len(candidates)}**",
        f"- Parents without a geometry pre-candidate: **{len(manifest['parents_without_geometry_precandidate'])}**",
        "- No RGB, semantic preview, instruction, action, normalizer, checkpoint, or policy dataset is written by this stage.",
        "",
        "## Categories",
        "",
        "| Category | Candidates |",
        "|---|---:|",
        *[f"| {name} | {category_counts[name]} |" for name in ("straight", "left_turn", "right_turn")],
        "",
        "## Scene coverage",
        "",
        "| Scene | Candidates |",
        "|---|---:|",
        *[f"| `{scene}` | {count} |" for scene, count in sorted(scene_counts.items())],
        "",
        "## Required next gate",
        "",
        "Each row remains rejected until a preview-only Matterport semantic camera supplies an auditable floor probe and three consecutive, unambiguous whitelist landmark frames. No N1 row may be converted to policy data directly.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--navila-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-candidates-per-parent", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_geometry_manifest(
        args.dataset,
        args.navila_root,
        seed=args.seed,
        max_candidates_per_parent=args.max_candidates_per_parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(manifest, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "candidates": len(manifest["candidates"])}))


if __name__ == "__main__":
    main()
