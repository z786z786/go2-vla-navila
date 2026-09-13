#!/usr/bin/env python3
"""Consolidate D5 evidence while preserving diagnostic population labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _offline_summary(path: Path) -> dict[str, Any]:
    metrics = _load(path / "metrics.json")
    first = metrics["first_step"]["ensemble_mean"]["per_action"]
    return {
        "path": str(path),
        "vx_mse": first["vx"]["mse"], "wz_mse": first["wz"]["mse"],
        "vx_correlation": first["vx"]["correlation_pearson"], "wz_correlation": first["wz"]["correlation_pearson"],
    }


def collection_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate and summarize the accepted-only D5 expert collection evidence."""
    if manifest.get("status") != "complete":
        raise ValueError(f"D5 collection is not complete: {manifest.get('status')!r}")
    accepted = manifest.get("accepted_ids")
    rejected = manifest.get("rejected_ids")
    attempts = manifest.get("attempts")
    if not isinstance(accepted, dict) or not isinstance(rejected, dict) or not isinstance(attempts, list):
        raise ValueError("invalid D5 collection manifest")
    minimum = {"train": 30, "seen-val": 12}
    maximum = {"train": 34, "seen-val": 16}
    counts = {split: len(accepted.get(split, [])) for split in minimum}
    if any(not minimum[split] <= counts[split] <= maximum[split] for split in minimum):
        raise ValueError(f"D5 accepted counts must be within {minimum}..{maximum}, got {counts}")
    if sum(counts.values()) > 50:
        raise ValueError(f"D5 accepted count exceeds the D5-R3 cap: {sum(counts.values())}")
    accepted_summary = manifest.get("accepted_summary")
    if not isinstance(accepted_summary, dict) or not isinstance(accepted_summary.get("splits"), dict):
        raise ValueError("D5 collection manifest lacks accepted-only coverage summary")
    coverage: dict[str, Any] = {}
    for split in minimum:
        summary = accepted_summary["splits"].get(split)
        if not isinstance(summary, dict) or not summary.get("category_minimum_met") or not summary.get("coverage_met"):
            raise ValueError(f"D5 accepted {split} evidence does not meet category and coverage requirements")
        coverage[split] = {
            "category_counts": summary.get("category_counts"), "scene_count": summary.get("scene_count"),
            "heading_bin_count": summary.get("heading_bin_count"),
            "distance_bin_counts": summary.get("distance_bin_counts"),
            "supplemental_count": summary.get("supplemental_count", 0),
        }
    accepted_values = {item for values in accepted.values() for item in values}
    rejected_values = {item for values in rejected.values() for item in values}
    if accepted_values & rejected_values:
        raise ValueError("accepted and rejected D5 IDs overlap")
    planner_rejects = sum(
        isinstance(item, dict) and item.get("decision", {}).get("kind") == "planner_reject"
        for item in attempts
    )
    rollout_rejects = sum(
        isinstance(item, dict) and item.get("decision", {}).get("kind") == "expert_rollout_reject"
        for item in attempts
    )
    reset_rejects = sum(
        isinstance(item, dict) and item.get("decision", {}).get("kind") == "expert_reset_reject"
        for item in attempts
    )
    assignment_policy = manifest.get("assignment_overrides", {})
    assignments = assignment_policy.get("assignments", {}) if isinstance(assignment_policy, dict) else {}
    if not isinstance(assignments, dict):
        raise ValueError("D5 assignment override provenance is invalid")
    ruleset = manifest.get("collection_ruleset")
    if ruleset is not None and not isinstance(ruleset, dict):
        raise ValueError("D5 collection ruleset provenance is invalid")
    return {
        "accepted_by_split": counts,
        "accepted_episode_count": sum(counts.values()),
        "accepted_coverage": coverage,
        "rejected_episode_count": len(rejected_values),
        "planner_reject_count": planner_rejects,
        "expert_rollout_reject_count": rollout_rejects,
        "expert_reset_reject_count": reset_rejects,
        "assignment_overrides": sorted(assignments.values(), key=lambda item: item["short_episode_id"]),
        "collection_ruleset": ruleset,
        "collection_manifest_sha256": manifest.get("collection_manifest_sha256"),
    }


def _markdown(report: dict[str, Any]) -> str:
    collection = report["collection"]
    lines = ["# D5 — Small Data Expansion Gate", "", "**Stage:** D5  ", f"**Status:** {report['status']}  ", f"**Evidence root:** `{report['run_root']}`", "", "Frame-level action-regime, sampler, normalizer, offline and closed-loop evidence are retained in the D5 root. All success/error thresholds below are diagnostic decision criteria, not statistical significance.", "", "## Expert collection", "", f"- Accepted episodes: `{collection['accepted_episode_count']}` (train `{collection['accepted_by_split']['train']}`, seen-val `{collection['accepted_by_split']['seen-val']}`).", f"- Retained rejected attempts: `{collection['rejected_episode_count']}`; planner rejects: `{collection['planner_reject_count']}`; expert-rollout rejects: `{collection['expert_rollout_reject_count']}`; expert-reset rejects: `{collection['expert_reset_reject_count']}.", "", "## Dataset and normalizer", "", f"- Sampler unit: `{report['sampler']['sampling_unit']}`; trainable anchors baseline/expanded: `{report['sampler']['baseline_trainable_anchor_count']}` / `{report['sampler']['expanded_trainable_anchor_count']}`.", f"- Equal-exposure final step count: `{report['equal_exposure_final_steps']}`.", f"- Substantial action-normalizer shift: `{report['normalizer_shift_substantial']}`.", "", "## Evaluation labels", "", "- `0000/0004`: TRAIN-ROUTE DIAGNOSTIC", "- `0001/0003`: SEEN-VAL GENERALIZATION", "- `0006`: UNSEEN-SCENE DIAGNOSTIC", "", "No merged train-route/generalization success rate is reported.", "", "## Offline first-step MSE", "", "| Model | Split | vx MSE | wz MSE |", "|---|---|---:|---:|"]
    for assignment in collection["assignment_overrides"]:
        lines.insert(13, f"- Assignment override: `{assignment['short_episode_id']}` source `{assignment['source_split']}` → collection `{assignment['collection_split']}`; excluded from seen-val evaluation.")
    ruleset = collection.get("collection_ruleset")
    if isinstance(ruleset, dict):
        change = ruleset.get("train_min_scenes_change", {})
        if isinstance(change, dict):
            lines.insert(13, f"- Train accepted-scene threshold: `{change.get('current')}` (approved D5-R5 change from `{change.get('previous')}`; this is a coverage criterion, not a generalization claim).")
    for model, splits in report["offline"].items():
        for split, values in splits.items():
            lines.append(f"| {model} | {split} | {values['vx_mse']:.8f} | {values['wz_mse']:.8f} |")
    lines.extend(["", "## Closed-loop", ""])
    for model, labels in report["closed_loop_summary"].items():
        for label, values in labels.items():
            lines.append(f"- `{model}` / `{label}`: macro success `{values['macro_success_rate']:.3f}`, macro final error `{values['macro_median_final_navigation_error_m']:.3f} m`.")
    lines.extend(["", f"**Conclusion:** {report['conclusion']}", "", "**Stop here.** Wait for explicit direction before any D6 or remedial experiment.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    collection = collection_summary(_load(root / "collection_manifest.json"))
    audit = _load(root / "dataset_audit" / "d5_dataset_audit.json")
    train = _load(root / "training" / "data_expanded" / "m6_report.json")
    closed = _load(root / "closed_loop" / "d5_closed_loop.json")
    offline: dict[str, dict[str, Any]] = {}
    for model_root in sorted((root / "offline").iterdir()):
        if model_root.is_dir():
            offline[model_root.name] = {split.name: _offline_summary(split) for split in sorted(model_root.iterdir()) if split.is_dir()}
    rows = closed["rows"]
    infrastructure = all(row["infrastructure_passed"] and row["raw_output_range_passed"] for row in rows)
    shift = bool(audit["normalizer_shift"]["substantial"])
    conclusion = (
        "D5 completed with a substantial normalizer shift; interpret coverage only against the retained normalizer ablation."
        if shift else "D5 completed without a substantial action-normalizer shift; compare the separated train-route and seen-val evidence to judge data coverage."
    )
    report = {
        "format": "go2-short-vln-m6_2-d5-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "status": "PASS" if infrastructure and bool(train["passed"]) else "FAIL",
        "collection": collection,
        "sampler": {
            "sampling_unit": audit["sampler_audit"]["sampling_unit"],
            "baseline_trainable_anchor_count": audit["baseline_sampler_audit"]["trainable_anchor_count"],
            "expanded_trainable_anchor_count": audit["sampler_audit"]["trainable_anchor_count"],
        },
        "equal_exposure_final_steps": train["training"]["steps"],
        "normalizer_shift_substantial": shift,
        "offline": offline,
        "closed_loop_summary": closed["summary"],
        "closed_loop_infrastructure_passed": infrastructure,
        "conclusion": conclusion,
    }
    (root / "d5_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    (root / "d5_report.md").write_text(markdown, encoding="utf-8")
    with args.diagnosis_report.open("a", encoding="utf-8") as stream:
        stream.write("\n" + markdown)
    print(json.dumps({"status": report["status"], "report": str(root / "d5_report.md")}, indent=2))


if __name__ == "__main__":
    main()
