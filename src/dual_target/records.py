"""Append-only DT1 episode evidence records; learner inputs stay separate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class RecordValidationError(ValueError):
    """Evidence is incomplete or crosses the learner/audit boundary."""


PRE_ACTION_FIELDS = frozenset({
    "event", "phase", "sim_time_s", "physics_step", "observation_seq", "camera_sensor_frame",
    "camera_timestamp_s", "render_request_seq", "rgb_path", "body_velocity_body", "raw_action",
    "applied_action", "task",
})
POST_STEP_FIELDS = frozenset({
    "event", "phase", "sim_time_before_s", "sim_time_after_s", "physics_step_before",
    "physics_step_after", "observation_seq_after", "camera_sensor_frame", "camera_timestamp_s",
    "render_request_seq", "robot_pose_w", "body_velocity_body", "collision_latched_substeps",
    "target_contact_force_max", "in_correct_parking_region", "in_other_parking_region",
    "fallen", "terminated", "truncated", "evaluator_stop", "wrong_target_stop", "warmup",
    "pre_reset_snapshot_ref", "scorer_status", "scorer_detail",
})


def _finite_vector(value: Sequence[object], length: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise RecordValidationError(f"{label} must contain {length} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise RecordValidationError(f"{label} must contain finite numeric values")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise RecordValidationError(f"{label} must contain finite numeric values") from exc
        if not math.isfinite(number):
            raise RecordValidationError(f"{label} must contain finite numeric values")
        result.append(number)
    return result


def _require_exact_keys(record: Mapping[str, Any], expected: frozenset[str], kind: str) -> None:
    actual = set(record)
    if actual != expected:
        raise RecordValidationError(f"{kind} keys differ; missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}")


def validate_pre_action(record: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(record, PRE_ACTION_FIELDS, "pre-action record")
    if record["event"] != "pre_action" or record["phase"] not in {"expert", "policy"}:
        raise RecordValidationError("pre-action record has invalid event or phase")
    for field in ("sim_time_s", "camera_timestamp_s"):
        if not isinstance(record[field], (int, float)) or isinstance(record[field], bool) or not math.isfinite(float(record[field])):
            raise RecordValidationError(f"{field} must be finite")
    for field in ("physics_step", "observation_seq", "camera_sensor_frame", "render_request_seq"):
        if not isinstance(record[field], int) or isinstance(record[field], bool) or record[field] < 0:
            raise RecordValidationError(f"{field} must be a non-negative integer")
    if not isinstance(record["rgb_path"], str) or not record["rgb_path"]:
        raise RecordValidationError("rgb_path is required")
    if not isinstance(record["task"], str) or not record["task"]:
        raise RecordValidationError("task is required")
    normalized = dict(record)
    normalized["body_velocity_body"] = _finite_vector(record["body_velocity_body"], 3, "body_velocity_body")
    normalized["raw_action"] = _finite_vector(record["raw_action"], 3, "raw_action")
    normalized["applied_action"] = _finite_vector(record["applied_action"], 3, "applied_action")
    return normalized


def validate_post_step(record: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(record, POST_STEP_FIELDS, "post-step record")
    if record["event"] != "post_step" or record["phase"] not in {"expert", "policy"}:
        raise RecordValidationError("post-step record has invalid event or phase")
    for field in ("sim_time_before_s", "sim_time_after_s", "camera_timestamp_s"):
        if not isinstance(record[field], (int, float)) or isinstance(record[field], bool) or not math.isfinite(float(record[field])):
            raise RecordValidationError(f"{field} must be finite")
    if float(record["sim_time_after_s"]) < float(record["sim_time_before_s"]):
        raise RecordValidationError("post-step simulation time must not go backwards")
    for field in ("physics_step_before", "physics_step_after", "observation_seq_after", "camera_sensor_frame", "render_request_seq"):
        if not isinstance(record[field], int) or isinstance(record[field], bool) or record[field] < 0:
            raise RecordValidationError(f"{field} must be a non-negative integer")
    if int(record["physics_step_after"]) < int(record["physics_step_before"]):
        raise RecordValidationError("post-step physics step must not go backwards")
    for field in (
        "in_correct_parking_region", "in_other_parking_region", "fallen", "terminated", "truncated",
        "evaluator_stop", "wrong_target_stop", "warmup",
    ):
        if not isinstance(record[field], bool):
            raise RecordValidationError(f"{field} must be a bool")
    collision = record["collision_latched_substeps"]
    if not isinstance(collision, list) or len(collision) != 4 or not all(isinstance(value, bool) for value in collision):
        raise RecordValidationError("collision_latched_substeps must be exactly four bool values")
    forces = record["target_contact_force_max"]
    if not isinstance(forces, Mapping) or set(forces) != {"red", "blue"}:
        raise RecordValidationError("target_contact_force_max must contain red and blue")
    for color, value in forces.items():
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0):
            raise RecordValidationError(f"target_contact_force_max[{color}] must be a finite non-negative number")
    normalized = dict(record)
    normalized["robot_pose_w"] = _finite_vector(record["robot_pose_w"], 7, "robot_pose_w")
    normalized["body_velocity_body"] = _finite_vector(record["body_velocity_body"], 3, "body_velocity_body")
    normalized["target_contact_force_max"] = {color: float(value) for color, value in forces.items()}
    if not isinstance(record["pre_reset_snapshot_ref"], str):
        raise RecordValidationError("pre_reset_snapshot_ref must be a string")
    return normalized


@dataclass
class EpisodeEvidenceWriter:
    """Create an append-only per-episode evidence directory without data conversion."""

    root: Path
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.root.exists():
            raise FileExistsError(f"refusing to overwrite episode evidence: {self.root}")
        metadata = {"format": "go2-dual-target-dt1-episode-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), **dict(self.metadata)}
        # Validate every metadata value before creating a partially initialized
        # evidence directory.  A NaN sidecar must fail closed, not leave a
        # directory that another run could mistake for a valid episode.
        manifest = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        self.root.mkdir(parents=True)
        # A paired-reset audit requests one real front RGB before warmup.  It
        # is deliberately separate from learner pre/post observations, but it
        # is still a fixed evidence directory created before any capture.
        (self.root / "rgb_reset").mkdir()
        (self.root / "rgb_pre").mkdir()
        (self.root / "rgb_post").mkdir()
        (self.root / "episode_manifest.json").write_text(manifest, encoding="utf-8")

    def _append(self, name: str, record: Mapping[str, Any]) -> None:
        with (self.root / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def pre_action(self, record: Mapping[str, Any]) -> None:
        self._append("pre_action.jsonl", validate_pre_action(record))

    def post_step(self, record: Mapping[str, Any]) -> None:
        self._append("post_step_events.jsonl", validate_post_step(record))

    def warmup(self, record: Mapping[str, Any]) -> None:
        required = {"event", "warmup_step", "sim_time_s", "physics_step", "command"}
        _require_exact_keys(record, frozenset(required), "warmup record")
        if record["event"] != "warmup" or not isinstance(record["warmup_step"], int):
            raise RecordValidationError("invalid warmup record")
        clean = dict(record)
        clean["command"] = _finite_vector(record["command"], 3, "command")
        self._append("warmup.jsonl", clean)
