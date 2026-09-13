from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from collectors.sim_go2.recorders.image_writer import save_depth_image, save_rgb_image
from collectors.sim_go2.utils.io import dump_json, ensure_dir


UNIFIED_RAW_SCHEMA_VERSION = "go2_local_dataset_v1"
SIM_SOURCE_TYPE = "isaac_sim"
SIM_CAMERA_INTERFACE_SOURCE = "isaaclab.camera.rgb"
SIM_DEPTH_INTERFACE_SOURCE = "isaaclab.camera.distance_to_image_plane"


class RawEpisodeLogger:
    """Write sim collection output directly in the unified raw session schema.

    The output layout matches the new real collector family:

    output_dir/
      index.json
      episodes/ep_xxxxxx.json
      images/ep_xxxxxx/frame_xxxxxx.jpg
      images_depth/ep_xxxxxx/frame_xxxxxx.png|npy
    """

    def __init__(
        self,
        output_dir: str | Path,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        *,
        save_depth: bool = False,
        depth_format: str = "png",
        depth_scale: float = 1000.0,
        depth_min: float | None = None,
        depth_max: float | None = None,
        depth_invalid_fill_value: float = 0.0,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.episodes_dir = ensure_dir(self.output_dir / "episodes")
        self.images_dir = ensure_dir(self.output_dir / "images")
        self.depth_dir = ensure_dir(self.output_dir / "images_depth")
        self.image_format = str(image_format).lower().lstrip(".")
        self.jpeg_quality = int(jpeg_quality)
        self.save_depth = bool(save_depth)
        self.depth_format = str(depth_format).lower().lstrip(".")
        self.depth_scale = float(depth_scale)
        self.depth_min = float(depth_min) if depth_min is not None else None
        self.depth_max = float(depth_max) if depth_max is not None else None
        self.depth_invalid_fill_value = float(depth_invalid_fill_value)
        self.session_id = self.output_dir.name
        self.current_episode_id: str | None = None
        self.current_episode_meta: dict[str, Any] | None = None
        self.current_frames: list[dict[str, Any]] = []
        self.current_image_dir: Path | None = None
        self.current_depth_dir: Path | None = None

        index_path = self.output_dir / "index.json"
        if index_path.exists():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
        self.index_payload = payload if isinstance(payload, dict) else {}
        self.index_payload.setdefault("schema_version", UNIFIED_RAW_SCHEMA_VERSION)
        self.index_payload.setdefault("session_id", self.session_id)
        self.index_payload.setdefault("source_type", SIM_SOURCE_TYPE)
        self.index_payload.setdefault("episodes", [])
        self._flush_index()

    def _flush_index(self) -> None:
        dump_json(self.output_dir / "index.json", self.index_payload)

    def list_completed_episode_ids(self) -> list[str]:
        completed: list[str] = []
        for item in self.index_payload.get("episodes", []):
            if not isinstance(item, dict):
                continue
            episode_id = str(item.get("episode_id") or "")
            if episode_id and (self.episodes_dir / f"{episode_id}.json").exists():
                completed.append(episode_id)
        return sorted(set(completed))

    def start_episode(self, episode_meta: Mapping[str, Any]) -> None:
        if self.current_episode_id is not None:
            raise RuntimeError("Previous episode not closed.")
        self.current_episode_id = str(episode_meta["episode_id"])
        self.current_episode_meta = deepcopy(dict(episode_meta))
        self.current_frames = []
        self.current_image_dir = ensure_dir(self.images_dir / self.current_episode_id)
        self.current_depth_dir = ensure_dir(self.depth_dir / self.current_episode_id) if self.save_depth else None

    def record_step(self, observation: Mapping[str, Any], command: Mapping[str, Any], step_meta: Mapping[str, Any]) -> None:
        if self.current_episode_id is None or self.current_episode_meta is None or self.current_image_dir is None:
            raise RuntimeError("start_episode() must be called before record_step().")

        step_id = int(step_meta["step_id"])
        image_name = f"frame_{step_id:06d}.{self.image_format}"
        image_path = self.current_image_dir / image_name
        save_rgb_image(observation["image"], image_path, image_format=self.image_format, jpeg_quality=self.jpeg_quality)
        depth_rel_path = None
        if self.save_depth and self.current_depth_dir is not None and observation.get("depth_image") is not None:
            depth_name = f"frame_{step_id:06d}.{self.depth_format}"
            depth_path = self.current_depth_dir / depth_name
            save_depth_image(
                observation["depth_image"],
                depth_path,
                image_format=self.depth_format,
                depth_scale=self.depth_scale,
                invalid_fill_value=self.depth_invalid_fill_value,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
            )
            depth_rel_path = str(Path("images_depth") / self.current_episode_id / depth_name)

        debug_state = observation.get("debug_state") or {}
        task = observation.get("task") or {}
        timestamp = float(step_meta["timestamp"])
        action_train = [float(value) for value in command["raw_cmd_train"]]
        action_full = [float(value) for value in command["raw_cmd_full"]]
        if self.current_frames:
            previous = self.current_frames[-1].get("control_action") if isinstance(self.current_frames[-1], dict) else None
            previous = previous if isinstance(previous, dict) else {}
            previous_action = {
                "vx": float(previous.get("vx", 0.0)),
                "vy": float(previous.get("vy", 0.0)),
                "wz": float(previous.get("wz", 0.0)),
            }
        else:
            previous_action = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        instruction_source = str(
            step_meta.get("instruction_source")
            or task.get("instruction_source")
            or self.current_episode_meta.get("instruction_source")
            or "auto_template"
        )
        frame = {
            "timestamp": timestamp,
            "image": str(Path("images") / self.current_episode_id / image_name),
            "instruction": str(observation["instruction"]),
            "state": {
                "vx": float(debug_state.get("vx", 0.0)),
                "vy": float(debug_state.get("vy", 0.0)),
                "wz": float(debug_state.get("yaw_rate", 0.0)),
                "yaw": float(debug_state.get("yaw", 0.0)),
                "vz": float(debug_state.get("vz", 0.0)),
                "x": float(debug_state.get("robot_x", 0.0)),
                "y": float(debug_state.get("robot_y", 0.0)),
                "z": float(debug_state.get("robot_z", 0.0)),
                "body_height": float(debug_state.get("robot_z", 0.0)),
                "roll": float(debug_state.get("roll", 0.0)),
                "pitch": float(debug_state.get("pitch", 0.0)),
                "error_code": float(debug_state.get("error_code", 0.0)),
                "mode": float(debug_state.get("mode", 0.0)),
                "gait_type": float(debug_state.get("gait_type", 0.0)),
            },
            "raw_action": {
                "vx": float(action_full[0]),
                "vy": float(action_full[1]),
                "wz": float(action_full[2]),
                "camera_pitch": 0.0,
                "keys": 0,
            },
            "control_action": {
                "vx": float(action_train[0]),
                "vy": 0.0,
                "wz": float(action_train[1]),
            },
            "previous_action": previous_action,
            "meta": {
                "schema_version": UNIFIED_RAW_SCHEMA_VERSION,
                "session_id": self.session_id,
                "episode_id": self.current_episode_id,
                "source_type": SIM_SOURCE_TYPE,
                "camera_interface_source": SIM_CAMERA_INTERFACE_SOURCE,
                "capture_mode": "trajectory",
                "task_family": str(task.get("task_id") or "goal_navigation"),
                "target_type": str(step_meta.get("target_type", "box")),
                "target_label": str(step_meta.get("target_label", task.get("target_label", task.get("target_class", "black box")))),
                "target_description": str(step_meta.get("target_description", task.get("target_description", task.get("target_class", "black box")))),
                "active_target_id": str(step_meta.get("active_target_id", task.get("active_target_id", ""))),
                "instruction_source": instruction_source,
                "segment_status": "clean",
                "success": "success" if bool(step_meta.get("success", False)) else "",
                "termination_reason": str(step_meta.get("termination_reason", "")),
                "scene_id": str(step_meta.get("scene_id", "")),
                "contrast_group_id": str(step_meta.get("contrast_group_id", "")),
                "contrast_variant": str(step_meta.get("contrast_variant", "")),
                "operator_id": str(step_meta.get("operator_id", "isaac_sim")),
                "state_timestamp": timestamp,
                "action_timestamp": timestamp,
                "raw_action_timestamp": timestamp,
                "control_action_timestamp": timestamp,
                "image_timestamp": timestamp,
                "turn_bucket": str(step_meta.get("turn_bucket", "")),
                "visibility_bucket": str(step_meta.get("visibility_bucket", "")),
                "layout_id": str(step_meta.get("layout_id", "")),
                "layout_template_id": str(step_meta.get("layout_template_id", "")),
                "layout_group_id": str(step_meta.get("layout_group_id", "")),
                "target_visible": bool((step_meta.get("debug", {}) or {}).get("target_visible", False)),
                "target_visible_within_3f": bool(task.get("target_visible_within_3f", False)),
                "target_visible_within_10f": bool(task.get("target_visible_within_10f", False)),
                "target_pixel_ratio": float((step_meta.get("debug", {}) or {}).get("target_pixel_ratio", 0.0)),
            },
        }
        if depth_rel_path is not None:
            frame["depth_image"] = depth_rel_path
            frame["meta"]["depth_interface_source"] = SIM_DEPTH_INTERFACE_SOURCE
            frame["meta"]["depth_format"] = self.depth_format
            frame["meta"]["depth_scale"] = self.depth_scale
            if self.depth_min is not None:
                frame["meta"]["depth_min"] = self.depth_min
            if self.depth_max is not None:
                frame["meta"]["depth_max"] = self.depth_max
        self.current_frames.append(frame)

    def end_episode(self, summary: Mapping[str, Any]) -> None:
        if self.current_episode_id is None or self.current_episode_meta is None:
            raise RuntimeError("No active episode to close.")

        episode_meta = dict(self.current_episode_meta)
        summary_dict = dict(summary)
        instruction = str(episode_meta.get("instruction", ""))
        instruction_source = str(episode_meta.get("instruction_source") or "auto_template")
        success_label = "success" if bool(summary_dict.get("success", False)) else "fail"
        termination_reason = str(summary_dict.get("termination_reason") or ("goal_reached" if success_label == "success" else "timeout"))
        target_label = str(episode_meta.get("target_label", episode_meta.get("target_class", "black box")))
        target_description = str(episode_meta.get("target_description", target_label))
        target_type = str(episode_meta.get("target_type", "box"))

        payload = {
            "schema_version": UNIFIED_RAW_SCHEMA_VERSION,
            "session_id": self.session_id,
            "episode_id": self.current_episode_id,
            "instruction": instruction,
            "capture_mode": "trajectory",
            "task_family": str(episode_meta.get("task_id", "goal_navigation")),
            "target_type": target_type,
            "target_label": target_label,
            "target_description": target_description,
            "active_target_id": str(episode_meta.get("active_target_id", "")),
            "collector_notes": str(episode_meta.get("collector_notes", "")),
            "instruction_source": instruction_source,
            "segment_status": str(episode_meta.get("segment_status", "clean")),
            "success": success_label,
            "termination_reason": termination_reason,
            "scene_id": str(episode_meta.get("scene_id", "")),
            "contrast_group_id": str(episode_meta.get("contrast_group_id", "")),
            "contrast_variant": str(episode_meta.get("contrast_variant", "")),
            "operator_id": str(episode_meta.get("operator_id", "isaac_sim")),
            "source_type": SIM_SOURCE_TYPE,
            "turn_bucket": str(summary_dict.get("turn_bucket", episode_meta.get("turn_bucket", ""))),
            "visibility_bucket": str(summary_dict.get("visibility_bucket", episode_meta.get("visibility_bucket", ""))),
            "layout_id": str(summary_dict.get("layout_id", episode_meta.get("layout_id", ""))),
            "layout_template_id": str(summary_dict.get("layout_template_id", episode_meta.get("layout_template_id", ""))),
            "layout_group_id": str(summary_dict.get("layout_group_id", episode_meta.get("layout_group_id", ""))),
            "target_visible_first_frame": bool(summary_dict.get("target_visible_first_frame", episode_meta.get("target_visible_first_frame", False))),
            "target_visible_within_3f": bool(summary_dict.get("target_visible_within_3f", episode_meta.get("target_visible_within_3f", False))),
            "target_visible_within_10f": bool(summary_dict.get("target_visible_within_10f", episode_meta.get("target_visible_within_10f", False))),
            "target_pixel_ratio_first": float(summary_dict.get("target_pixel_ratio_first", episode_meta.get("target_pixel_ratio_first", 0.0))),
            "scene_targets": list(episode_meta.get("scene_targets", [])),
            "frames": list(self.current_frames),
        }
        dump_json(self.episodes_dir / f"{self.current_episode_id}.json", payload)

        episode_index_entry = {
            "episode_id": self.current_episode_id,
            "instruction": instruction,
            "capture_mode": "trajectory",
            "task_family": str(episode_meta.get("task_id", "goal_navigation")),
            "target_type": target_type,
            "target_label": target_label,
            "target_description": target_description,
            "active_target_id": str(episode_meta.get("active_target_id", "")),
            "collector_notes": str(episode_meta.get("collector_notes", "")),
            "instruction_source": instruction_source,
            "segment_status": str(episode_meta.get("segment_status", "clean")),
            "success": success_label,
            "termination_reason": termination_reason,
            "scene_id": str(episode_meta.get("scene_id", "")),
            "contrast_group_id": str(episode_meta.get("contrast_group_id", "")),
            "contrast_variant": str(episode_meta.get("contrast_variant", "")),
            "operator_id": str(episode_meta.get("operator_id", "isaac_sim")),
            "num_frames": len(self.current_frames),
            "start_timestamp": float(self.current_frames[0]["timestamp"]) if self.current_frames else 0.0,
            "end_timestamp": float(self.current_frames[-1]["timestamp"]) if self.current_frames else 0.0,
            "source_type": SIM_SOURCE_TYPE,
            "turn_bucket": str(summary_dict.get("turn_bucket", "")),
            "visibility_bucket": str(summary_dict.get("visibility_bucket", "")),
        }
        episodes = [item for item in self.index_payload.get("episodes", []) if str(item.get("episode_id", "")) != self.current_episode_id]
        episodes.append(episode_index_entry)
        episodes.sort(key=lambda item: str(item.get("episode_id", "")))
        self.index_payload["episodes"] = episodes
        self._flush_index()

        self.current_episode_id = None
        self.current_episode_meta = None
        self.current_frames = []
        self.current_image_dir = None
        self.current_depth_dir = None

    def close(self) -> None:
        self._flush_index()
