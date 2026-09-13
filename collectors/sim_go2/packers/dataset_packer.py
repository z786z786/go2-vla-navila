from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from collectors.sim_go2.utils.action_utils import build_action_chunk
from collectors.sim_go2.utils.io import dump_json, dump_jsonl, ensure_dir, ensure_symlink_or_copytree, iter_jsonl, load_json
from collectors.sim_go2.utils.randomization import assign_split_by_layout
from collectors.sim_go2.utils.schema import PACKED_SCHEMA_VERSION


class DatasetPacker:
    """Convert raw per-step trajectories into packed VLA training samples."""

    def __init__(self, raw_root: str | Path, output_dir: str | Path, packing_cfg: dict[str, Any], runtime_cfg: dict[str, Any]):
        self.raw_root = Path(raw_root)
        self.output_dir = ensure_dir(output_dir)
        self.packing_cfg = packing_cfg
        self.runtime_cfg = runtime_cfg
        self.chunk_len = int(packing_cfg.get("chunk_len", 4))
        self.control_hz = int(round(1.0 / float(runtime_cfg.get("control_dt", 0.05))))
        self.turn_threshold = float(packing_cfg.get("turning_threshold", 0.1))
        self.rng = random.Random(int(runtime_cfg.get("seed", 0)))

    def pack(self) -> dict[str, Any]:
        episodes = self._load_episodes()
        filtered = [episode for episode in episodes if self._episode_passes_filters(episode)]
        self._materialize_episode_images()
        packed_rows: list[dict[str, Any]] = []
        for episode in filtered:
            rows = self._pack_episode(episode)
            packed_rows.extend(rows)
        packed_rows = self._apply_visibility_sample_quota(packed_rows)
        split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for row in packed_rows:
            split_rows[row["split"]].append(row)
        dump_jsonl(self.output_dir / "dataset.jsonl", packed_rows)
        for split_name, rows in split_rows.items():
            dump_jsonl(self.output_dir / f"{split_name}.jsonl", rows)
        summary = self._summary_payload(filtered, packed_rows)
        dump_json(self.output_dir / "pack_summary.json", summary)
        return summary

    def _materialize_episode_images(self) -> None:
        source_dir = self.raw_root / "episodes"
        if source_dir.exists():
            ensure_symlink_or_copytree(self.output_dir / "episodes", source_dir)

    def _load_episodes(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        episodes_dir = self.raw_root / "episodes"
        for episode_dir in sorted(episodes_dir.glob("ep_*")):
            meta_path = episode_dir / "meta.json"
            steps_path = episode_dir / "steps.jsonl"
            if not meta_path.exists() or not steps_path.exists():
                continue
            meta = load_json(meta_path)
            if meta.get("status") != "completed":
                continue
            steps = list(iter_jsonl(steps_path))
            if not steps:
                continue
            episodes.append({"meta": meta, "steps": steps, "episode_dir": episode_dir})
        return episodes

    def _episode_passes_filters(self, episode: dict[str, Any]) -> bool:
        meta = episode["meta"]
        steps = episode["steps"]
        min_steps = int(self.packing_cfg.get("min_episode_steps", 8))
        if len(steps) < min_steps:
            return False
        if self.packing_cfg.get("require_success", False) and not bool(meta.get("success", False)):
            return False
        if self.packing_cfg.get("drop_collision_episodes", True) and bool(meta.get("collision", False)):
            return False
        if self.packing_cfg.get("drop_fall_episodes", False) and bool(meta.get("fallen", False)):
            return False
        require_visibility = bool(self.packing_cfg.get("require_visibility_metadata", True))
        visibility_bucket = str(meta.get("visibility_bucket", ""))
        if require_visibility and not visibility_bucket:
            return False
        if visibility_bucket == "visible_early" and not bool(meta.get("target_visible_within_3f", False)):
            return False
        if visibility_bucket == "visible_after_small_turn" and not bool(meta.get("target_visible_within_10f", False)):
            return False
        if visibility_bucket == "search" and not bool(self.packing_cfg.get("allow_search_episodes", True)):
            return False
        min_progress = float(self.packing_cfg.get("min_progress_distance", 0.2))
        robot_poses = [step.get("robot_pose") or {} for step in steps]
        if robot_poses:
            start_x = float(robot_poses[0].get("x", 0.0))
            start_y = float(robot_poses[0].get("y", 0.0))
            end_x = float(robot_poses[-1].get("x", 0.0))
            end_y = float(robot_poses[-1].get("y", 0.0))
            if math.hypot(end_x - start_x, end_y - start_y) < min_progress:
                return False
        return True

    def _pack_episode(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        meta = episode["meta"]
        steps = episode["steps"]
        layout_group = str(meta.get("layout_group_id") or steps[0].get("layout_group_id") or meta.get("episode_id"))
        split_cfg = self.packing_cfg.get("split", {})
        split = assign_split_by_layout(
            layout_group,
            train_ratio=float(split_cfg.get("train_ratio", 0.8)),
            val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        )
        action_rows = [list(step.get("raw_cmd_train", [0.0, 0.0])) for step in steps]
        packed_rows: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            action_chunk, action_mask = build_action_chunk(action_rows, index, self.chunk_len, pad_action=(0.0, 0.0))
            row = {
                "schema_version": PACKED_SCHEMA_VERSION,
                "image": str(step["image_path"]),
                "instruction": str(step["instruction"]),
                "state": [float(value) for value in step["state"]],
                "action_chunk": action_chunk,
                "action_mask": action_mask,
                "task_id": str(step.get("task_id", meta.get("task_id", "goal_navigation"))),
                "target_class": str(step.get("target_class", meta.get("target_class", "black_box"))),
                "target_color": str(step.get("target_color", meta.get("target_color", "black"))),
                "target_shape": str(step.get("target_shape", meta.get("target_shape", "box"))),
                "episode_id": str(step["episode_id"]),
                "step_id": int(step["step_id"]),
                "control_hz": self.control_hz,
                "dt": float(self.runtime_cfg.get("control_dt", 0.05)),
                "source": "isaac_sim",
                "scene_id": str(step.get("scene_id", meta.get("scene_id", ""))),
                "split": split,
                "instruction_template_id": str(step.get("instruction_template_id", meta.get("instruction_template_id", ""))),
                "scene_kind": str(step.get("scene_kind", meta.get("scene_kind", ""))),
                "scene_mode": str(step.get("scene_mode", meta.get("scene_mode", ""))),
                "layout_id": str(step.get("layout_id", meta.get("layout_id", ""))),
                "layout_template_id": str(step.get("layout_template_id", meta.get("layout_template_id", ""))),
                "layout_group_id": layout_group,
                "turn_bucket": str(step.get("turn_bucket", meta.get("turn_bucket", ""))),
                "visibility_bucket": str(step.get("visibility_bucket", meta.get("visibility_bucket", ""))),
                "target_visible_first_frame": bool(meta.get("target_visible_first_frame", False)),
                "target_visible_within_3f": bool(meta.get("target_visible_within_3f", False)),
                "target_visible_within_10f": bool(meta.get("target_visible_within_10f", False)),
                "target_pixel_ratio_first": float(meta.get("target_pixel_ratio_first", 0.0)),
                "split_hint": str(step.get("split_hint", meta.get("split_hint", "train_candidate"))),
                "raw_cmd_full_first": [float(value) for value in step.get("raw_cmd_full", [0.0, 0.0, 0.0])],
                "raw_cmd_train_first": [float(value) for value in step.get("raw_cmd_train", [0.0, 0.0])],
                "timestamp": float(step.get("timestamp", 0.0)),
                "debug": {
                    "success": bool(step.get("success", False)),
                    "collision": bool(step.get("collision", False)),
                    "timeout": bool(step.get("timeout", False)),
                    "robot_pose": step.get("robot_pose", {}),
                    "target_pose": step.get("target_pose", {}),
                },
            }
            packed_rows.append(row)
        return packed_rows

    def _apply_visibility_sample_quota(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        quota_cfg = self.packing_cfg.get("visibility_sample_quota", {})
        if not quota_cfg.get("enabled", False):
            return rows
        bucket_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            bucket_rows.setdefault(str(row.get("visibility_bucket", "")), []).append(row)
        total_rows = len(rows)
        desired_ratios = {
            "visible_early": float(quota_cfg.get("visible_early_ratio", 0.75)),
            "visible_after_small_turn": float(quota_cfg.get("visible_after_small_turn_ratio", 0.17)),
            "search": float(quota_cfg.get("search_ratio", 0.08)),
        }
        kept: list[dict[str, Any]] = []
        for bucket, group in bucket_rows.items():
            if bucket not in desired_ratios:
                kept.extend(group)
                continue
            max_count = int(round(desired_ratios[bucket] * total_rows))
            if len(group) <= max_count or max_count <= 0:
                kept.extend(group)
                continue
            shuffled = list(group)
            self.rng.shuffle(shuffled)
            kept.extend(shuffled[:max_count])
        kept.sort(key=lambda row: (str(row.get("episode_id", "")), int(row.get("step_id", 0))))
        return kept

    def _summary_payload(self, episodes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
        split_counts = Counter(row["split"] for row in rows)
        target_counts = Counter(row["target_class"] for row in rows)
        scene_counts = Counter(row["scene_id"] for row in rows)
        turn_counts = Counter(row.get("turn_bucket", "") for row in rows if row.get("turn_bucket"))
        visibility_counts = Counter(row.get("visibility_bucket", "") for row in rows if row.get("visibility_bucket"))
        state_lengths = Counter(len(row.get("state", [])) for row in rows)
        turn_ratio = 0.0
        left_ratio = 0.0
        right_ratio = 0.0
        chunk_turn_ratio = 0.0
        if rows:
            turn_steps = 0
            left_steps = 0
            right_steps = 0
            chunk_turns = 0
            for row in rows:
                first_wz = float(row["action_chunk"][0][1])
                if abs(first_wz) > self.turn_threshold:
                    turn_steps += 1
                    if first_wz > 0:
                        left_steps += 1
                    else:
                        right_steps += 1
                if any(abs(float(step[1])) > self.turn_threshold for step in row["action_chunk"]):
                    chunk_turns += 1
            turn_ratio = turn_steps / len(rows)
            left_ratio = left_steps / len(rows)
            right_ratio = right_steps / len(rows)
            chunk_turn_ratio = chunk_turns / len(rows)
        return {
            "episodes_total": len(episodes),
            "samples_total": len(rows),
            "split_counts": dict(sorted(split_counts.items())),
            "target_counts": dict(sorted(target_counts.items())),
            "scene_counts": dict(sorted(scene_counts.items())),
            "turn_bucket_counts": dict(sorted(turn_counts.items())),
            "visibility_bucket_counts": dict(sorted(visibility_counts.items())),
            "state_length_counts": dict(sorted(state_lengths.items())),
            "chunk_len": self.chunk_len,
            "turn_threshold": self.turn_threshold,
            "turn_ratio_first": turn_ratio,
            "left_ratio_first": left_ratio,
            "right_ratio_first": right_ratio,
            "chunk_turn_ratio": chunk_turn_ratio,
            "visible_first_frame_ratio": self._bool_ratio(rows, "target_visible_first_frame"),
            "visible_within_3f_ratio": self._bool_ratio(rows, "target_visible_within_3f"),
            "visible_within_10f_ratio": self._bool_ratio(rows, "target_visible_within_10f"),
            "search_ratio": visibility_counts.get("search", 0) / len(rows) if rows else 0.0,
        }

    @staticmethod
    def _bool_ratio(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(1 for row in rows if bool(row.get(key, False))) / len(rows)
