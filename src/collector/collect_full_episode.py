#!/usr/bin/env python3
"""Recollect one complete official NaVILA episode with the immutable Go2 PD expert.

The legacy collector is used only as an instrumentation engine.  A temporary,
single-episode adapter supplies *original* ``gt_locations`` as the PD route;
the source ``reference_path`` and ``gt_locations`` are kept verbatim in an
immutable provenance sidecar.  No M3/D5/M6/M7 dataset or checkpoint is read.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.collector.full_episode_contract import (
    TERMINAL_HOLD_FRAMES,
    build_collection_provenance,
    validate_collection_provenance,
)
from src.navila_full.selection import load_official_episodes, path_length_m, route_id_for, sha256


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-episode-id", required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--expected-route-id")
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--terminal-hold-frames", type=int, default=TERMINAL_HOLD_FRAMES)
    parser.add_argument(
        "--max-pd-frames",
        type=int,
        default=None,
        help="reject a full-episode PD attempt after this many non-terminal records",
    )
    parser.add_argument("--skip-sanity", action="store_true")
    args, passthrough = parser.parse_known_args()
    if args.terminal_hold_frames != TERMINAL_HOLD_FRAMES:
        raise ValueError(f"full-episode collection fixes --terminal-hold-frames={TERMINAL_HOLD_FRAMES}")
    if args.max_pd_frames is not None and args.max_pd_frames <= 0:
        raise ValueError("--max-pd-frames must be positive when set")
    return args, passthrough


def select_episode(dataset: Path, episode_id: str) -> dict[str, Any]:
    matches = [episode for episode in load_official_episodes(dataset) if str(episode["episode_id"]) == episode_id]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one official source episode {episode_id!r}, found {len(matches)}")
    return matches[0]


def adapter_episode(source: dict[str, Any]) -> dict[str, Any]:
    """Adapt the immutable planner input, deliberately making gt_locations canonical."""
    goals = source.get("goals")
    if not isinstance(goals, list) or len(goals) != 1 or not isinstance(goals[0], dict):
        raise ValueError("official episode must have exactly one goal")
    goal = goals[0]
    if "position" not in goal or "radius" not in goal:
        raise ValueError("official goal needs position and radius")
    return {
        "short_episode_id": f"full_{route_id_for(source)}_{hashlib.sha256(str(source['episode_id']).encode()).hexdigest()[:8]}",
        "source_episode_id": str(source["episode_id"]),
        "source_trajectory_id": str(source["trajectory_id"]),
        "source_episode_new_id": source.get("episode_new_id"),
        "scene_id": source["scene_id"],
        # Required only by the legacy instrumentation's summary schema; it is
        # never sent to the policy and does not define a final train/test split.
        "split": "full-episode-pending-final-assignment",
        "start_pose": {"position": source["start_position"], "rotation_wxyz": source["start_rotation"]},
        "goal_pose": {"position": goal["position"], "success_radius_m": goal["radius"]},
        "path_length": path_length_m(source["gt_locations"], "gt_locations"),
        "instruction": source["instruction"]["instruction_text"],
        # collect_expert maps this field to official reference_path and
        # gt_locations; this is why the planner sees the original GT route.
        "reference_path": source["gt_locations"],
    }


def run() -> None:
    args, passthrough = parse_args()
    dataset = args.dataset.resolve()
    episode_dir = args.episode_dir.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if episode_dir.exists():
        raise FileExistsError(f"refusing to overwrite collection directory: {episode_dir}")
    source = select_episode(dataset, args.source_episode_id)
    route_id = route_id_for(source)
    if args.expected_route_id is not None and args.expected_route_id != route_id:
        raise ValueError("--expected-route-id does not bind the requested source episode")
    provenance = build_collection_provenance(
        source_dataset_sha256=sha256(dataset), source_episode=source
    )
    errors = validate_collection_provenance(provenance)
    if errors:
        raise RuntimeError("invalid collection provenance: " + "; ".join(errors))
    with tempfile.TemporaryDirectory(prefix="navila_full_episode_") as temporary:
        adapter_path = Path(temporary) / "one_full_episode.json"
        adapter_path.write_text(json.dumps({"episodes": [adapter_episode(source)]}, ensure_ascii=False), encoding="utf-8")
        command = [
            sys.executable, "-m", "src.collector.collect_expert",
            "--short-dataset", str(adapter_path), "--short-index", "0",
            "--episode-dir", str(episode_dir), "--terminal-hold-frames", str(TERMINAL_HOLD_FRAMES),
            "--jpeg-quality", str(args.jpeg_quality), "--hide-expert-path-markers",
        ]
        if args.max_pd_frames is not None:
            command.extend(["--max-pd-frames", str(args.max_pd_frames)])
        if args.official_source is not None:
            command.extend(["--official-source", str(args.official_source.resolve())])
        if args.asset_root is not None:
            command.extend(["--asset-root", str(args.asset_root.resolve())])
        completed = subprocess.run(command + passthrough, check=False)
    # Preserve provenance even for unsuccessful attempted candidates, so that
    # a failure can never be silently reclassified as a different route.
    if episode_dir.is_dir():
        (episode_dir / "full_episode_provenance.json").write_text(
            json.dumps({**provenance, "route_id": route_id}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    summary_path = episode_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("instrumented PD collector completed without summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    marker_guard = summary.get("expert_path_marker_guard")
    if not isinstance(marker_guard, dict) or marker_guard.get("passed") is not True:
        raise RuntimeError("full-episode collection did not prove expert-path markers were hidden")
    summary.update({
        "collection_format": provenance["format"], "dataset_schema_version": provenance["dataset_schema_version"],
        "full_route_id": route_id, "source_dataset_sha256": provenance["source_dataset_sha256"],
        "original_instruction": provenance["original_instruction"],
        "full_episode_provenance": "full_episode_provenance.json",
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.skip_sanity:
        report_json = episode_dir / "full_episode_sanity_report.json"
        report_md = episode_dir / "full_episode_sanity_report.md"
        check = subprocess.run([
            sys.executable, "-m", "src.collector.check_expert", "--episode-dirs", str(episode_dir),
            "--report-json", str(report_json), "--report-md", str(report_md),
        ], check=False)
        if check.returncode != 0:
            raise SystemExit(check.returncode)


if __name__ == "__main__":
    run()
