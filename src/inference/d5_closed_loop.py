#!/usr/bin/env python3
"""Aggregate D5 closed-loop evidence without merging diagnostic populations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.inference.check_rollout import evaluate_rollout_records, load_jsonl
from src.smolvla.d5_data_expansion import diagnostic_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _row(model: str, seed: int, episode_dir: Path, oracle_root: Path) -> dict[str, Any]:
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
    records = load_jsonl(episode_dir / "steps.jsonl")
    requests = load_jsonl(episode_dir / "requests.jsonl")
    checked = evaluate_rollout_records(summary, records, requests)
    episode_id = str(summary["short_episode_id"])
    return {
        "model": model,
        "policy_seed": seed,
        "episode_id": episode_id,
        "evaluation_label": diagnostic_labels().get(episode_id, "OTHER HELD-OUT DIAGNOSTIC"),
        "infrastructure_passed": checked["infrastructure_passed"],
        "raw_output_range_passed": checked["raw_output_range_passed"],
        "success": bool(summary["success"]),
        "final_navigation_error_m": float(summary["final_navigation_error_m"]),
        "minimum_navigation_error_m": min(float(item["evaluator_distance_to_goal_m"]) for item in records),
        "collision": bool(summary["collision"]),
        "termination_reason": summary["termination_reason"],
        "frame_count": len(records),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["model"]][row["evaluation_label"]].append(row)
    result: dict[str, Any] = {}
    for model, by_label in grouped.items():
        result[model] = {}
        for label, items in by_label.items():
            by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in items:
                by_episode[item["episode_id"]].append(item)
            episode_metrics = {
                episode: {
                    "run_count": len(values),
                    "success_rate": statistics.fmean(float(value["success"]) for value in values),
                    "median_final_navigation_error_m": statistics.median(value["final_navigation_error_m"] for value in values),
                    "all_infrastructure_passed": all(value["infrastructure_passed"] for value in values),
                    "all_raw_ranges_passed": all(value["raw_output_range_passed"] for value in values),
                }
                for episode, values in by_episode.items()
            }
            result[model][label] = {
                "episode_count": len(by_episode),
                "macro_success_rate": statistics.fmean(item["success_rate"] for item in episode_metrics.values()),
                "macro_median_final_navigation_error_m": statistics.fmean(item["median_final_navigation_error_m"] for item in episode_metrics.values()),
                "episodes": episode_metrics,
                "thresholds_note": "D5 diagnostic improvement triggers only; not statistical significance.",
            }
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# D5 Closed-Loop Diagnostic", "", "All thresholds here are diagnostic decision criteria, not statistical significance.", ""]
    for model, labels in report["summary"].items():
        lines.extend([f"## {model}", "", "| Evaluation label | Episodes | Macro success | Macro final error (m) |", "|---|---:|---:|---:|"])
        for label, metrics in labels.items():
            lines.append(f"| {label} | {metrics['episode_count']} | {metrics['macro_success_rate']:.3f} | {metrics['macro_median_final_navigation_error_m']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    rows: list[dict[str, Any]] = []
    for model_root in sorted(path for path in root.iterdir() if path.is_dir()):
        for seed_root in sorted(path for path in model_root.glob("seed_*") if path.is_dir()):
            seed = int(seed_root.name.removeprefix("seed_"))
            for episode_dir in sorted(path for path in seed_root.iterdir() if (path / "summary.json").is_file()):
                rows.append(_row(model_root.name, seed, episode_dir, args.oracle_root.resolve()))
    if not rows:
        raise ValueError("no D5 episode evidence found")
    report = {"format": "go2-short-vln-m6_2-d5-closed-loop-v1", "rows": rows, "summary": summarize(rows)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(rows), "models": sorted(report["summary"])}, indent=2))


if __name__ == "__main__":
    main()
