"""Pure validation helpers shared by full-episode collection and conversion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.collector.collection_limits import EXPERT_PATH_MARKER_PRIM_PATH
from src.collector.r7_supervision import PD_ACTION_SOURCE, TERMINAL_HOLD_ACTION_SOURCE
from src.navila_full.contracts import ACTION_NAMES, DATASET_SCHEMA_VERSION, STATE_NAMES


FULL_COLLECTION_FORMAT = "navila-full-episode-go2-pd-v2-marker-hidden"
TERMINAL_HOLD_FRAMES = 50
CONTROL_FREQUENCY_HZ = 50


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-D sequence")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def policy_state3_from_velocity(current_velocity: Mapping[str, Any]) -> list[float]:
    """The sole policy state: current measured body planar velocity and yaw rate."""
    if not isinstance(current_velocity, Mapping):
        raise ValueError("current_velocity must be an object")
    linear = _finite_vector(current_velocity.get("linear_body"), 3, "linear_body")
    angular = _finite_vector(current_velocity.get("angular_body"), 3, "angular_body")
    return [linear[0], linear[1], angular[2]]


def action3_from_record(record: Mapping[str, Any]) -> list[float]:
    declared = [record.get("expert_vx"), record.get("expert_vy"), record.get("expert_wz")]
    action = _finite_vector(declared, 3, "expert action")
    planner = _finite_vector(record.get("planner_command"), 3, "planner_command")
    locomotion = _finite_vector(record.get("locomotion_command"), 3, "locomotion_command")
    if any(abs(left - right) > 1e-8 for left, right in zip(action, planner)):
        raise ValueError("expert action does not equal planner_command")
    if any(abs(left - right) > 1e-8 for left, right in zip(action, locomotion)):
        raise ValueError("planner_command does not equal locomotion_command")
    if not -1e-6 <= action[0] <= 0.500001 or abs(action[1]) > 0.500001 or abs(action[2]) > 0.500001:
        raise ValueError("Go2 PD action exceeds the declared navigation bounds")
    source = record.get("action_source")
    if source not in {PD_ACTION_SOURCE, TERMINAL_HOLD_ACTION_SOURCE}:
        raise ValueError("action_source must identify PD or terminal zero supervision")
    if source == TERMINAL_HOLD_ACTION_SOURCE and any(abs(value) > 1e-8 for value in action):
        raise ValueError("terminal hold action must be an exact zero label")
    return action


def validate_record(record: Mapping[str, Any], *, expected_index: int) -> None:
    if int(record.get("frame_index", -1)) != expected_index:
        raise ValueError("frame indices must be contiguous")
    timestamp = float(record.get("timestamp", math.nan))
    control_dt = float(record.get("control_dt_s", math.nan))
    if not math.isclose(control_dt, 1.0 / CONTROL_FREQUENCY_HZ, abs_tol=1e-8):
        raise ValueError("collection must remain at 50 Hz")
    if not math.isclose(timestamp, expected_index / CONTROL_FREQUENCY_HZ, abs_tol=1e-8):
        raise ValueError("frame timestamp does not agree with 50 Hz collection")
    if not isinstance(record.get("front_rgb"), str) or not record["front_rgb"]:
        raise ValueError("record has no front RGB reference")
    policy_state3_from_velocity(record.get("current_velocity", {}))
    action3_from_record(record)


def build_collection_provenance(*, source_dataset_sha256: str, source_episode: Mapping[str, Any], pd_route_field: str = "gt_locations") -> dict[str, Any]:
    if pd_route_field != "gt_locations":
        raise ValueError("full-episode PD collection must follow original gt_locations")
    required = {"episode_id", "trajectory_id", "scene_id", "instruction", "reference_path", "gt_locations"}
    if not required <= set(source_episode):
        raise ValueError("source episode misses original full-episode fields")
    instruction = source_episode["instruction"]
    if not isinstance(instruction, Mapping) or not isinstance(instruction.get("instruction_text"), str):
        raise ValueError("source instruction must be the original instruction_text")
    return {
        "format": FULL_COLLECTION_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset_sha256": source_dataset_sha256,
        "source_episode_id": str(source_episode["episode_id"]),
        "source_trajectory_id": str(source_episode["trajectory_id"]),
        "scene_id": str(source_episode["scene_id"]),
        "original_instruction": source_episode["instruction"]["instruction_text"],
        "original_reference_path": source_episode["reference_path"],
        "original_gt_locations": source_episode["gt_locations"],
        "pd_route_field": pd_route_field,
        "expert_path_markers_hidden": True,
        "expert_path_marker_prim_path": EXPERT_PATH_MARKER_PRIM_PATH,
        "policy_state": {"names": list(STATE_NAMES), "source": "current_velocity body frame"},
        "policy_action": {"names": list(ACTION_NAMES), "source": "official Go2 PD planner command"},
        "terminal_hold_frames": TERMINAL_HOLD_FRAMES,
    }


def validate_collection_provenance(provenance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if provenance.get("format") != FULL_COLLECTION_FORMAT:
        errors.append("wrong full-episode collection format")
    if provenance.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("wrong full-episode data schema")
    if not provenance.get("source_dataset_sha256"):
        errors.append("source dataset hash is missing")
    if not str(provenance.get("original_instruction", "")).strip():
        errors.append("original instruction is missing")
    if not provenance.get("original_reference_path") or not provenance.get("original_gt_locations"):
        errors.append("original reference_path/gt_locations are missing")
    if provenance.get("pd_route_field") != "gt_locations":
        errors.append("PD route is not original gt_locations")
    if provenance.get("expert_path_markers_hidden") is not True:
        errors.append("RGB must hide expert-path markers")
    if provenance.get("expert_path_marker_prim_path") != EXPERT_PATH_MARKER_PRIM_PATH:
        errors.append("wrong expert-path marker namespace")
    if provenance.get("terminal_hold_frames") != TERMINAL_HOLD_FRAMES:
        errors.append("terminal supervision must be exactly 50 frames")
    return errors
