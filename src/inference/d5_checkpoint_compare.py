#!/usr/bin/env python3
"""Build checked videos for the three D5 checkpoints on one fixed rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.inference.check_rollout import (
    EXPECTED_M4_PLANNER_SHA256,
    EXPECTED_SHORT_DATASET_SHA256,
    check_episode,
    load_jsonl,
)
from src.inference.d3_closed_loop import audit_request_seeds
from src.inference.d5_resume import audit_completed_episode


MODELS = ("m61", "expanded_step_002000", "expanded_equal_exposure")
EVIDENCE_FILES = ("summary.json", "steps.jsonl", "requests.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_checkpoint_sha256(requests: list[dict[str, Any]]) -> str:
    values = {
        item.get("response", {}).get("checkpoint_sha256")
        for item in requests
        if item.get("status") == "ok" and isinstance(item.get("response"), dict)
    }
    values.discard(None)
    if len(values) != 1:
        raise ValueError(f"expected one response checkpoint hash, got {sorted(values)}")
    return str(next(iter(values)))


def build_comparison(
    run_root: Path,
    oracle_root: Path,
    *,
    episode_id: str = "short_vln_v1_0000",
    seed: int = 20260831,
) -> dict[str, Any]:
    closed_loop_root = run_root.resolve() / "closed_loop"
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        episode_dir = closed_loop_root / model / f"seed_{seed}" / episode_id
        completion = audit_completed_episode(
            episode_dir, expected_episode_id=episode_id, max_steps=1500
        )
        if not completion["passed"]:
            raise ValueError(f"incomplete comparison episode {model}: {completion['errors']}")
        before = {name: sha256(episode_dir / name) for name in EVIDENCE_FILES}
        requests = load_jsonl(episode_dir / "requests.jsonl")
        checkpoint_sha256 = response_checkpoint_sha256(requests)
        seed_audit = audit_request_seeds(requests, base_seed=seed)
        checked = check_episode(
            episode_dir,
            oracle_root.resolve(),
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_planner_sha256=EXPECTED_M4_PLANNER_SHA256,
            expected_short_dataset_sha256=EXPECTED_SHORT_DATASET_SHA256,
        )
        after = {name: sha256(episode_dir / name) for name in EVIDENCE_FILES}
        if before != after:
            raise RuntimeError(f"source evidence changed while rendering {model}")
        if not seed_audit["passed"]:
            raise ValueError(f"policy seed audit failed for {model}")
        if not checked["artifacts_complete"] or not checked["infrastructure_passed"]:
            raise ValueError(f"artifact or infrastructure audit failed for {model}")
        rows.append(
            {
                "model": model,
                "episode_id": episode_id,
                "policy_seed": seed,
                "checkpoint_sha256": checkpoint_sha256,
                "success": bool(checked["summary"]["success"]),
                "final_navigation_error_m": checked["summary"]["final_navigation_error_m"],
                "frame_count": completion["frame_count"],
                "raw_output_range_passed": checked["raw_output_range_passed"],
                "video": checked["artifacts"]["video"],
                "video_decoded_frames": checked["artifacts"]["video_decoded_frames"],
                "plot": checked["artifacts"]["plot"],
                "source_sha256": before,
            }
        )
    return {
        "format": "go2-short-vln-m6_2-d5-three-checkpoint-comparison-v1",
        "episode_id": episode_id,
        "policy_seed": seed,
        "models": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_comparison(args.run_root, args.oracle_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
