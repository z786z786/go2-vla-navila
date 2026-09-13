#!/usr/bin/env python3
"""Convert successful full-episode Go2 PD data to a new 3-D LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.collector.full_episode_contract import action3_from_record, policy_state3_from_velocity, validate_record
from src.collector.r7_supervision import TERMINAL_HOLD_ACTION_SOURCE, terminal_hold_audit
from src.navila_full.contracts import (
    ACTION_KEY, ACTION_NAMES, CONVERSION_MANIFEST_FORMAT, DATASET_SCHEMA_VERSION,
    IMAGE_KEY, POLICY_INPUTS, STATE_KEY, STATE_NAMES,
)
from src.navila_full.finalize_selection import validate_final_manifest
from src.navila_full.selection import sha256
from src.navila_n0.sampling import cap_terminal_near_zero_samples, is_near_zero


FPS = 50
SPLIT_DIRS = {"train": "train", "seen-val": "seen_val", "unseen-test": "unseen_test"}
POLICY_FIELDS = [*POLICY_INPUTS[:-1], ACTION_KEY, "task"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-prefix", default="local/navila_full_episode_go2")
    parser.add_argument("--lerobot-version", default="v0.6.0")
    parser.add_argument("--lerobot-commit", default="30da8e687a6dfc617fcd94afc367ac7071c376ce")
    parser.add_argument("--lerobot-archive-sha256", default="a4451766b7b450c7067a7e45671117511212a51c693b05aa711df2e101e4f7fd")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def feature_spec() -> dict[str, dict[str, Any]]:
    return {
        IMAGE_KEY: {"dtype": "video", "shape": (512, 512, 3), "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": (3,), "names": list(STATE_NAMES)},
        ACTION_KEY: {"dtype": "float32", "shape": (3,), "names": list(ACTION_NAMES)},
    }


def load_verified_episode(row: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]], str]:
    root = Path(str(row["collection_dir"])).resolve()
    provenance = json.loads((root / "full_episode_provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    sanity = json.loads((root / "sanity.json").read_text(encoding="utf-8"))
    if (
        provenance.get("route_id") != row.get("route_id")
        or provenance.get("source_episode_id") != row.get("source_episode_id")
        or provenance.get("original_instruction") != row.get("instruction")
        or provenance.get("original_reference_path") != row.get("reference_path")
        or provenance.get("original_gt_locations") != row.get("gt_locations")
        or summary.get("success") is not True or summary.get("status") != "complete"
        or sanity.get("passed") is not True
    ):
        raise ValueError(f"collection provenance/success mismatch: {root}")
    records = load_records(root / "steps.jsonl")
    if len(records) < 52 or len(records) != int(summary.get("record_count", -1)):
        raise ValueError(f"invalid complete full episode record count: {root}")
    terminal = terminal_hold_audit(records, requested_frames=50)
    if not terminal["passed"]:
        raise ValueError(f"full episode misses exact 50-frame terminal supervision: {root}")
    for index, record in enumerate(records):
        validate_record(record, expected_index=index)
    task = str(provenance["original_instruction"]).strip()
    if not task:
        raise ValueError(f"empty original instruction: {root}")
    return root, records, task


def convert_episode(dataset: Any, row: Mapping[str, Any], *, global_start: int) -> dict[str, Any]:
    import cv2
    import numpy as np

    root, records, task = load_verified_episode(row)
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        state = np.asarray(policy_state3_from_velocity(record["current_velocity"]), dtype=np.float32)
        action = np.asarray(action3_from_record(record), dtype=np.float32)
        if state.shape != (3,) or action.shape != (3,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"non-finite 3-D policy data in {root} frame {index}")
        image_path = root / str(record["front_rgb"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.shape != (512, 512, 3) or image_bgr.dtype != np.uint8:
            raise ValueError(f"invalid current RGB observation: {image_path}")
        dataset.add_frame({
            IMAGE_KEY: cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            STATE_KEY: state,
            ACTION_KEY: action,
            "task": task,
        })
        samples.append({
            "sample_id": f"{row['route_id']}:{index}",
            "global_index": global_start + index,
            "terminal": record.get("action_source") == TERMINAL_HOLD_ACTION_SOURCE,
            "near_zero_chunk": is_near_zero(action.tolist()),
        })
    dataset.save_episode()
    return {
        "route_id": row["route_id"],
        "source_episode_id": row["source_episode_id"],
        "source_trajectory_id": row["source_trajectory_id"],
        "scene_id": row["scene_id"],
        "category": row["category"],
        "collection_dir": str(root),
        "task": task,
        "frame_count": len(records),
        "steps_sha256": sha256(root / "steps.jsonl"),
        "summary_sha256": sha256(root / "summary.json"),
        "samples": samples,
    }


def _make_dataset(*, repo_id: str, root: Path) -> Any:
    # Import lazily: pure contract/state tests should not need an Isaac/LeRobot install.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, features=feature_spec(), root=root, robot_type="unitree_go2",
        use_videos=True, image_writer_threads=4, encoder_threads=4,
    )


def convert(final_manifest_path: Path, output_root: Path, *, repo_prefix: str, lerobot: Mapping[str, str]) -> dict[str, Any]:
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    errors = validate_final_manifest(final_manifest)
    if errors:
        raise ValueError("invalid final selection manifest: " + "; ".join(errors))
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_DIRS}
    for row in final_manifest["splits"]:
        grouped[str(row["split"])].append(row)
    converted: dict[str, Any] = {}
    train_candidates: list[dict[str, Any]] = []
    for split, rows in grouped.items():
        if not rows:
            raise ValueError(f"required final split is empty: {split}")
        split_root = output_root / SPLIT_DIRS[split]
        dataset = _make_dataset(repo_id=f"{repo_prefix}_{SPLIT_DIRS[split]}", root=split_root)
        episodes: list[dict[str, Any]] = []
        frame_offset = 0
        for episode_index, row in enumerate(rows):
            item = convert_episode(dataset, row, global_start=frame_offset)
            item["lerobot_episode_index"] = episode_index
            frame_offset += int(item["frame_count"])
            if split == "train":
                train_candidates.extend(item["samples"])
            item.pop("samples")
            episodes.append(item)
        dataset.finalize()
        converted[split] = {
            "root": str(split_root), "repo_id": f"{repo_prefix}_{SPLIT_DIRS[split]}",
            "episode_count": len(episodes), "frame_count": frame_offset, "episodes": episodes,
        }
    capped = cap_terminal_near_zero_samples(train_candidates, maximum_ratio=0.10)
    selected_ids = set(capped["selected_sample_ids"])
    selected_train_indices = [item["global_index"] for item in train_candidates if item["sample_id"] in selected_ids]
    manifest = {
        "format": CONVERSION_MANIFEST_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_selection_manifest": str(final_manifest_path.resolve()),
        "final_selection_manifest_sha256": sha256(final_manifest_path),
        "output_root": str(output_root), "fps": FPS,
        "policy_fields": POLICY_FIELDS,
        "forbidden_source_fields_not_copied": ["reference_path", "gt_locations", "goal_pose", "planner_command", "pd_planner_command", "robot_pose", "joint_position", "joint_velocity"],
        "image_mapping": {"source": "current front_rgb JPEG", "target": IMAGE_KEY, "color": "RGB"},
        "state_mapping": {"source": "current_velocity.linear_body[x,y], current_velocity.angular_body[z]", "target": STATE_KEY, "output_dimension": 3, "names": list(STATE_NAMES)},
        "action_mapping": {"source": "expert_vx, expert_vy, expert_wz", "target": ACTION_KEY, "names": list(ACTION_NAMES)},
        "task_mapping": {"source": "full_episode_provenance.original_instruction", "target": "task", "rewrite": "forbidden"},
        "terminal_sampler": {**capped, "selected_train_indices": selected_train_indices},
        "temporal_storage": "one 50-Hz action per frame; the policy creates 50-action chunks at load time",
        "splits": converted,
        "lerobot": dict(lerobot),
    }
    (output_root / "conversion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = convert(
        args.final_selection_manifest.resolve(), args.output_root.resolve(), repo_prefix=args.repo_prefix,
        lerobot={"version": args.lerobot_version, "commit": args.lerobot_commit, "source_archive_sha256": args.lerobot_archive_sha256},
    )
    print(json.dumps({"output": str(args.output_root), "train_frames": manifest["splits"]["train"]["frame_count"], "terminal_ratio": manifest["terminal_sampler"]["terminal_or_near_zero_selected_ratio"]}))


if __name__ == "__main__":
    main()
