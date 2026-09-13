#!/usr/bin/env python3
"""Convert successful M4 Go2 expert rollouts to isolated LeRobot datasets.

The converter intentionally writes only the four policy-facing fields required by
M5: front RGB, a 30-D proprioceptive state, the 3-D velocity action, and task text.
Planner/reference/goal fields remain in the source audit trail and are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


FPS = 50
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
POLICY_FIELDS = [IMAGE_KEY, STATE_KEY, ACTION_KEY, "task"]
SPLIT_DIRS = {"train": "train", "seen-val": "seen_val", "unseen-test": "unseen_test"}
D5_ASSIGNMENT_SIDECAR = "d5_assignment.json"

# M4 intended to record a 33-D vector, but its Isaac Lab RPY call returned a
# tuple and `[0]` serialized only roll. All 2,446 gate frames therefore contain
# 31 values: velocities [0:6], roll [6], relative joint position [7:19], and
# joint velocity [19:31]. Reconstruct full RPY from the same frame's legitimate
# proprioceptive WXYZ quaternion. SmolVLA v0.6.0 accepts at most 32 state dims,
# so retain planar body motion, full base attitude, and all joint state while
# omitting z linear and roll/pitch angular velocity, yielding 30 dimensions.
M4_ROBOT_STATE_DIMS = (31, 33)
STATE_NAMES = [
    "body_linear_velocity_x",
    "body_linear_velocity_y",
    "body_angular_velocity_yaw",
    "base_roll",
    "base_pitch",
    "base_yaw",
    *[f"joint_{i:02d}_position_relative" for i in range(12)],
    *[f"joint_{i:02d}_velocity" for i in range(12)],
]
ACTION_NAMES = ["vx", "vy", "wz"]
OMITTED_RAW_STATE = {
    "names": [
        "body_linear_velocity_z",
        "body_angular_velocity_roll",
        "body_angular_velocity_pitch",
    ],
    "reason": "preserve pretrained SmolVLA max_state_dim=32 compatibility",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-prefix", default="local/go2_short_vln")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_DIRS.values()),
        default=tuple(SPLIT_DIRS.values()),
        help="LeRobot split directories to convert; default preserves the M5 three-split contract.",
    )
    parser.add_argument("--lerobot-version", default="v0.6.0")
    parser.add_argument(
        "--lerobot-commit", default="30da8e687a6dfc617fcd94afc367ac7071c376ce"
    )
    parser.add_argument(
        "--lerobot-archive-sha256",
        default="a4451766b7b450c7067a7e45671117511212a51c693b05aa711df2e101e4f7fd",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_d5_assignment(summary: dict[str, Any], episode_dir: Path) -> dict[str, Any] | None:
    path = episode_dir / D5_ASSIGNMENT_SIDECAR
    if not path.is_file():
        return None
    assignment = load_json(path)
    required = {
        "format", "short_episode_id", "source_split", "collection_split", "reason",
        "exclude_from_seen_val_evaluation", "assignment_policy_sha256", "source_dataset_sha256",
    }
    if not isinstance(assignment, dict) or set(assignment) != required:
        raise ValueError(f"Invalid D5 assignment sidecar: {path}")
    if assignment["format"] != "go2-short-vln-m6_2-d5-assignment-sidecar-v1":
        raise ValueError(f"Unknown D5 assignment sidecar format: {path}")
    if assignment["short_episode_id"] != summary.get("short_episode_id"):
        raise ValueError(f"D5 assignment ID mismatch: {path}")
    if assignment["source_split"] != summary.get("split") or assignment["source_split"] not in SPLIT_DIRS:
        raise ValueError(f"D5 assignment source split mismatch: {path}")
    if assignment["collection_split"] not in SPLIT_DIRS or assignment["collection_split"] == assignment["source_split"]:
        raise ValueError(f"D5 assignment collection split is invalid: {path}")
    if assignment["source_split"] != "seen-val" or assignment["collection_split"] != "train":
        raise ValueError(f"D5 assignment is outside the approved train supplement scope: {path}")
    if assignment["exclude_from_seen_val_evaluation"] is not True:
        raise ValueError(f"D5 assignment does not exclude seen-val evaluation: {path}")
    if not isinstance(assignment["assignment_policy_sha256"], str) or not isinstance(assignment["source_dataset_sha256"], str):
        raise ValueError(f"D5 assignment provenance is invalid: {path}")
    return {**assignment, "path": str(path), "sha256": sha256(path)}


def discover_episodes(root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        episode_dir = summary_path.parent
        steps_path = episode_dir / "steps.jsonl"
        sanity_path = episode_dir / "sanity.json"
        if not steps_path.is_file() or not sanity_path.is_file():
            continue
        summary, sanity = load_json(summary_path), load_json(sanity_path)
        if summary.get("status") != "complete" or not summary.get("success"):
            raise ValueError(f"Source episode is not successful: {episode_dir}")
        if not sanity.get("passed"):
            raise ValueError(f"Source episode failed M4 sanity checks: {episode_dir}")
        source_split = summary.get("split")
        if source_split not in SPLIT_DIRS:
            raise ValueError(f"Unknown source split {source_split!r}: {episode_dir}")
        assignment = _load_d5_assignment(summary, episode_dir)
        collection_split = assignment["collection_split"] if assignment else source_split
        episodes.append(
            {
                "dir": episode_dir,
                "summary_path": summary_path,
                "steps_path": steps_path,
                "sanity_path": sanity_path,
                "summary": summary,
                "source_split": source_split,
                "collection_split": collection_split,
                "assignment": assignment,
            }
        )
    if not episodes:
        raise ValueError(f"No complete, sanity-checked M4 episodes found below {root}")
    return episodes


def feature_spec() -> dict[str, dict[str, Any]]:
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (512, 512, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {"dtype": "float32", "shape": (30,), "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (3,), "names": ACTION_NAMES},
    }


def validate_record(record: dict[str, Any], episode_dir: Path, expected_index: int) -> None:
    if int(record["frame_index"]) != expected_index:
        raise ValueError(f"Non-contiguous frame index in {episode_dir}: {record['frame_index']}")
    if not math.isclose(float(record["timestamp"]), expected_index / FPS, abs_tol=1e-8):
        raise ValueError(f"Timestamp/FPS mismatch at {episode_dir} frame {expected_index}")
    if not math.isclose(float(record["control_dt_s"]), 1.0 / FPS, abs_tol=1e-8):
        raise ValueError(f"Control period mismatch at {episode_dir} frame {expected_index}")


def quaternion_wxyz_to_rpy(quaternion: Any) -> np.ndarray:
    """Match Isaac Lab's XYZ Euler conversion and [0, 2*pi) wrapping."""
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"Invalid WXYZ quaternion: {quaternion}")
    norm = float(np.linalg.norm(quat))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"Non-unit WXYZ quaternion norm={norm}")
    w, x, y, z = quat / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.mod(np.asarray([roll, pitch, yaw], dtype=np.float32), 2.0 * np.pi)


def extract_policy_state(record: dict[str, Any]) -> np.ndarray:
    """Build the audited 30-D SmolVLA state from one M4 record."""
    raw_state = np.asarray(record["robot_state"], dtype=np.float32)
    linear_body = np.asarray(record["current_velocity"]["linear_body"], dtype=np.float32)
    angular_body = np.asarray(record["current_velocity"]["angular_body"], dtype=np.float32)
    joint_velocity = np.asarray(record["joint_velocity"], dtype=np.float32)
    if raw_state.shape not in {(31,), (33,)}:
        raise ValueError(f"Expected M4 robot_state[31 or 33], got {raw_state.shape}")
    if linear_body.shape != (3,) or angular_body.shape != (3,) or joint_velocity.shape != (12,):
        raise ValueError("Invalid velocity field dimensions in M4 record")
    rpy = quaternion_wxyz_to_rpy(record["robot_pose"]["quaternion_wxyz"])
    joint_offset_start = 7 if raw_state.shape == (31,) else 9
    state = np.concatenate(
        [linear_body[:2], angular_body[2:3], rpy, raw_state[joint_offset_start : joint_offset_start + 12], joint_velocity]
    ).astype(np.float32)
    if state.shape != (30,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid reconstructed policy state: shape={state.shape}")
    return state


def convert_episode(dataset: LeRobotDataset, episode: dict[str, Any]) -> dict[str, Any]:
    episode_dir: Path = episode["dir"]
    summary = episode["summary"]
    records = load_records(episode["steps_path"])
    if len(records) != int(summary["record_count"]):
        raise ValueError(f"Record count mismatch: {episode_dir}")
    if len(records) < 2:
        raise ValueError(f"Episode too short: {episode_dir}")
    task = str(summary["instruction"]).strip()
    if not task:
        raise ValueError(f"Empty instruction: {episode_dir}")

    robot_state_length_counts: dict[str, int] = {}
    for expected_index, record in enumerate(records):
        validate_record(record, episode_dir, expected_index)
        try:
            state = extract_policy_state(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid M4 robot state: {episode_dir} frame {expected_index}: {exc}"
            ) from exc
        action = np.asarray(
            [record["expert_vx"], record["expert_vy"], record["expert_wz"]], dtype=np.float32
        )
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid 3-D expert action: {episode_dir} frame {expected_index}")
        image_path = episode_dir / record["front_rgb"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.shape != (512, 512, 3) or image_bgr.dtype != np.uint8:
            raise ValueError(f"Invalid front RGB image: {image_path}")
        dataset.add_frame(
            {
                IMAGE_KEY: cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                STATE_KEY: state,
                ACTION_KEY: action,
                "task": task,
            }
        )
        raw_length = len(record["robot_state"])
        robot_state_length_counts[str(raw_length)] = robot_state_length_counts.get(str(raw_length), 0) + 1
    dataset.save_episode()
    return {
        "source_dir": str(episode_dir),
        "source_short_episode_id": summary["short_episode_id"],
        "source_episode_id": summary["source_episode_id"],
        "scene_id": summary["scene_id"],
        "source_split": episode["source_split"],
        "collection_split": episode["collection_split"],
        "assignment_override": episode["assignment"],
        "task": task,
        "frame_count": len(records),
        "summary_sha256": sha256(episode["summary_path"]),
        "steps_sha256": sha256(episode["steps_path"]),
        "sanity_sha256": sha256(episode["sanity_path"]),
        "robot_state_length_counts": robot_state_length_counts,
    }


def main() -> None:
    args = parse_args()
    expert_root = args.expert_root.resolve()
    output_root = args.output_root.resolve()
    if not expert_root.is_dir():
        raise FileNotFoundError(expert_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    selected_splits = tuple(args.splits)
    if len(set(selected_splits)) != len(selected_splits):
        raise ValueError("--splits must not contain duplicates")
    discovered = discover_episodes(expert_root)
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in selected_splits}
    for episode in discovered:
        target = SPLIT_DIRS[episode["collection_split"]]
        if target in grouped:
            grouped[target].append(episode)

    converted: dict[str, Any] = {}
    observed_state_length_counts: dict[str, int] = {}
    for split, episodes in grouped.items():
        if not episodes:
            raise ValueError(f"Required split is empty: {split}")
        split_root = output_root / split
        repo_id = f"{args.repo_prefix}_{split}"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            features=feature_spec(),
            root=split_root,
            robot_type="unitree_go2",
            use_videos=True,
            image_writer_threads=4,
            encoder_threads=4,
        )
        episode_manifest = []
        for lerobot_episode_index, episode in enumerate(episodes):
            item = convert_episode(dataset, episode)
            for length, count in item["robot_state_length_counts"].items():
                observed_state_length_counts[length] = observed_state_length_counts.get(length, 0) + int(count)
            item["lerobot_episode_index"] = lerobot_episode_index
            episode_manifest.append(item)
            print(
                f"converted split={split} episode={lerobot_episode_index} "
                f"frames={item['frame_count']} scene={item['scene_id']}",
                flush=True,
            )
        dataset.finalize()
        converted[split] = {
            "root": str(split_root),
            "repo_id": repo_id,
            "episode_count": len(episode_manifest),
            "frame_count": sum(item["frame_count"] for item in episode_manifest),
            "episodes": episode_manifest,
        }

    manifest = {
        "format": "go2-short-vln-lerobot-m5-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_expert_root": str(expert_root),
        "output_root": str(output_root),
        "fps": FPS,
        "temporal_storage": "one action per source frame; SmolVLA action chunks are sampled at load time",
        "policy_fields": POLICY_FIELDS,
        "forbidden_source_fields_not_copied": [
            "reference_path",
            "goal_pose",
            "distance_to_goal_xy_m",
            "distance_to_goal_path_m",
            "planner_command_history",
            "ground_truth_goal_direction",
        ],
        "image_mapping": {"source": "front_rgb JPEG", "target": IMAGE_KEY, "color": "RGB"},
        "state_mapping": {
            "source": {
                "planar_velocity": "current_velocity.linear_body[x,y]",
                "yaw_rate": "current_velocity.angular_body[z]",
                "base_rpy": "reconstructed from robot_pose.quaternion_wxyz with Isaac Lab XYZ convention",
                "relative_joint_position": "robot_state[7:19]",
                "joint_velocity": "joint_velocity[0:12]",
            },
            "target": STATE_KEY,
            "output_dimension": 30,
            "names": STATE_NAMES,
            "source_schema_correction": {
                "declared_dimension": 33,
                "observed_dimension_counts": observed_state_length_counts,
                "legacy_31_cause": "collector indexed tuple returned by euler_xyz_from_quat and serialized roll only",
                "complete_33_contract": "D5 collector serializes all roll/pitch/yaw values",
            },
            "omitted": OMITTED_RAW_STATE,
        },
        "action_mapping": {
            "source": ["expert_vx", "expert_vy", "expert_wz"],
            "target": ACTION_KEY,
            "names": ACTION_NAMES,
        },
        "task_mapping": {"source": "summary.instruction", "target": "task"},
        "lerobot": {
            "version": args.lerobot_version,
            "commit": args.lerobot_commit,
            "source_archive_sha256": args.lerobot_archive_sha256,
        },
        "splits": converted,
        "converted_collection_splits": list(selected_splits),
        "assignment_overrides": [
            item["assignment_override"]
            for split in converted.values() for item in split["episodes"]
            if item["assignment_override"] is not None
        ],
    }
    manifest_path = output_root / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"conversion complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
