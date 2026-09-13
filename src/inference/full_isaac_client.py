#!/usr/bin/env python3
"""Run the full-episode 3-D policy in Go2 Isaac closed loop.

The environment runner is reused process-locally, while its policy-facing
protocol/state helpers are replaced before startup.  Thus the runner receives
only current RGB + [body_vx, body_vy, body_yaw_rate] + original instruction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.collector.full_episode_contract import build_collection_provenance
from src.inference import isaac_client as runner
from src.inference import full_protocol, full_state
from src.navila_full.contracts import POLICY_PROTOCOL_VERSION
from src.navila_full.selection import load_official_episodes, path_length_m, route_id_for, sha256


LATENCY_P95_MS_MAX = 600.0


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-episode-id", required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--expected-route-id")
    args, passthrough = parser.parse_known_args()
    forbidden = {"--short-dataset", "--short-episode-id", "--short-index"}
    if forbidden & set(passthrough):
        raise ValueError("full client selects only --dataset + --source-episode-id")
    return args, passthrough


def select_episode(dataset: Path, episode_id: str) -> dict[str, Any]:
    matches = [episode for episode in load_official_episodes(dataset) if str(episode["episode_id"]) == episode_id]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one official episode {episode_id!r}, found {len(matches)}")
    return matches[0]


def adapter_episode(source: dict[str, Any]) -> dict[str, Any]:
    goal = source["goals"][0]
    return {
        "short_episode_id": f"full_{route_id_for(source)}_{hashlib.sha256(str(source['episode_id']).encode()).hexdigest()[:8]}",
        "source_episode_id": str(source["episode_id"]),
        "source_trajectory_id": str(source["trajectory_id"]),
        "source_episode_new_id": source.get("episode_new_id"),
        "scene_id": source["scene_id"], "split": "full-episode",
        "start_pose": {"position": source["start_position"], "rotation_wxyz": source["start_rotation"]},
        "goal_pose": {"position": goal["position"], "success_radius_m": goal["radius"]},
        "path_length": path_length_m(source["gt_locations"], "gt_locations"),
        "instruction": source["instruction"]["instruction_text"],
        # The official VLN wrapper receives gt_locations for its expert route.
        "reference_path": source["gt_locations"],
    }


def _patch_runner() -> None:
    runner.PROTOCOL_VERSION = full_protocol.PROTOCOL_VERSION
    runner.validate_request_header = full_protocol.validate_request_header
    runner.unix_request = full_protocol.unix_request
    runner.build_policy_state = full_state.build_policy_state
    runner.validate_action_chunk = full_state.validate_action_chunk
    runner.apply_action_safety = full_state.apply_action_safety


def _annotate_summary(episode_dir: Path, source: dict[str, Any], dataset_sha256: str, route_id: str) -> None:
    provenance = build_collection_provenance(source_dataset_sha256=dataset_sha256, source_episode=source)
    provenance["route_id"] = route_id
    provenance["use"] = "full_episode_closed_loop_evaluation_only"
    (episode_dir / "full_episode_inference_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = episode_dir / "summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    p95 = summary.get("p95_inference_roundtrip_ms")
    latency_passed = isinstance(p95, (int, float)) and float(p95) <= LATENCY_P95_MS_MAX
    summary.update({
        "format": "navila-full-episode-go2-closed-loop-v1",
        "dataset_schema_version": provenance["dataset_schema_version"],
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "full_route_id": route_id,
        "source_dataset_sha256": dataset_sha256,
        "original_instruction": provenance["original_instruction"],
        "full_episode_inference_provenance": "full_episode_inference_provenance.json",
        "latency_gate": {"p95_roundtrip_ms_max": LATENCY_P95_MS_MAX, "observed_p95_roundtrip_ms": p95, "passed": latency_passed},
        "temporal_contract": {"dataset_rate_hz": 50, "execute_frames_per_replan": 10, "wall_clock_control_rate_claim": None},
    })
    summary["passed"] = bool(summary.get("passed")) and latency_passed
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path = episode_dir / "client_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update({
            "format": "navila-full-episode-go2-client-config-v1",
            "protocol_version": POLICY_PROTOCOL_VERSION,
            "policy_request_fields": ["current_rotated_rgb_jpeg", "body_vx_body_vy_body_yaw_rate", "original_instruction"],
            "forbidden_policy_inputs": ["reference_path", "gt_locations", "goal_pose", "distance_to_goal", "next_waypoint", "pd_state"],
            "full_route_id": route_id,
        })
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args, passthrough = parse_args()
    dataset, episode_dir = args.dataset.resolve(), args.episode_dir.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    source = select_episode(dataset, args.source_episode_id)
    route_id = route_id_for(source)
    if args.expected_route_id is not None and args.expected_route_id != route_id:
        raise ValueError("--expected-route-id does not bind the requested source episode")
    if episode_dir.exists():
        raise FileExistsError(f"refusing to overwrite episode directory: {episode_dir}")
    source_hash = sha256(dataset)
    _patch_runner()
    with tempfile.TemporaryDirectory(prefix="navila_full_client_") as temporary:
        adapter_path = Path(temporary) / "one_full_episode.json"
        adapter_path.write_text(json.dumps({"episodes": [adapter_episode(source)]}, ensure_ascii=False), encoding="utf-8")
        original_argv = sys.argv
        try:
            sys.argv = [sys.argv[0], "--short-dataset", str(adapter_path), "--short-index", "0", "--episode-dir", str(episode_dir), *passthrough]
            runner.main()
        finally:
            sys.argv = original_argv
            if episode_dir.is_dir():
                _annotate_summary(episode_dir, source, source_hash, route_id)


if __name__ == "__main__":
    main()
