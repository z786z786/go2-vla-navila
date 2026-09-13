#!/usr/bin/env python3
"""Verify the six planned full-episode Go2 PD canary collections."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.collector.full_episode_contract import TERMINAL_HOLD_FRAMES, validate_collection_provenance, validate_record
from src.collector.r7_supervision import command_semantics, terminal_hold_audit
from src.navila_full.finalize_selection import validate_canary_manifest


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate(canary_manifest: dict[str, Any], collection_root: Path) -> dict[str, Any]:
    errors = validate_canary_manifest(canary_manifest)
    rows: list[dict[str, Any]] = []
    for planned in canary_manifest.get("routes", []):
        route_id = str(planned["route_id"])
        root = collection_root / route_id
        checks: dict[str, bool] = {}
        try:
            provenance = json.loads((root / "full_episode_provenance.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            sanity = json.loads((root / "sanity.json").read_text(encoding="utf-8"))
            records = load_records(root / "steps.jsonl")
            checks["provenance_valid"] = not validate_collection_provenance(provenance)
            checks["route_bound"] = provenance.get("route_id") == route_id and provenance.get("source_episode_id") == planned.get("source_episode_id") and provenance.get("original_instruction") == planned.get("instruction")
            guard = summary.get("expert_path_marker_guard")
            checks["expert_path_markers_hidden"] = isinstance(guard, dict) and guard.get("passed") is True
            checks["success"] = summary.get("status") == "complete" and summary.get("success") is True and sanity.get("passed") is True
            checks["terminal_50"] = terminal_hold_audit(records, requested_frames=TERMINAL_HOLD_FRAMES)["passed"]
            checks["command_semantics"] = command_semantics(records, allow_legacy=False)["passed"]
            try:
                for index, record in enumerate(records):
                    validate_record(record, expected_index=index)
                checks["three_dimensional_policy_fields"] = True
            except (ValueError, TypeError, KeyError):
                checks["three_dimensional_policy_fields"] = False
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            checks = {"collection_readable": False}
        rows.append({"route_id": route_id, "source_episode_id": planned.get("source_episode_id"), "category": planned.get("category"), "scene_name": planned.get("scene_name"), "collection_dir": str(root), "checks": checks, "passed": all(checks.values())})
    coverage = Counter(str(row["category"]) for row in rows if row["passed"])
    coverage_ok = all(coverage[category] == 2 for category in ("straight", "left_turn", "right_turn"))
    return {"format": "navila-full-episode-go2-pd-canary-quality-v1", "passed": not errors and coverage_ok and len(rows) == 6 and all(row["passed"] for row in rows), "manifest_errors": errors, "coverage": dict(coverage), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.canary_manifest.read_text(encoding="utf-8")), args.collection_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "coverage": report["coverage"], "output": str(args.output)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
