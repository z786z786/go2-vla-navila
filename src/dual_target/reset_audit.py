"""Strict CPU-side contract for paired DT1 physical resets.

Each color instruction is evaluated from a separately constructed Isaac
environment.  That is useful for isolation, but it means equality of the
starting physics state cannot be assumed from a shared command-line ``--seed``.
This module defines the small, serializable record that the live runner must
write immediately after reset and its first front-camera capture.  It contains
no target-derived control signal and is never passed to a policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

from .low_level import HISTORY_LENGTH, PROPRIO_DIM


RESET_AUDIT_SCHEMA = "go2-dual-target-dt1-paired-reset-audit-v1"
PAIR_SEED_ALGORITHM = "sha256-go2-dual-target-dt1-pair-v1-u31"
_CONFIGURATIONS = frozenset({"A_red_B_blue", "A_blue_B_red"})
_COLORS = frozenset({"red", "blue"})
_HEX64 = re.compile(r"[0-9a-f]{64}")
_STATE_ABS_TOL = 1e-6
# Task comparability, not bitwise simulator determinism. Frozen before the
# revised matrix runs; unrelated to navigation success/parking thresholds.
PAIR_COMPARISON_POLICY = "task_equivalence_v2"
PAIR_POSITION_TOL_M = 0.02
PAIR_ORIENTATION_TOL_RAD = math.radians(3.0)


class ResetAuditError(ValueError):
    """A reset record is incomplete or cannot prove a paired initial state."""


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResetAuditError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ResetAuditError(f"{label} must be a finite number")
    return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResetAuditError(f"{label} must be an integer >= {minimum}")
    return value


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ResetAuditError(f"{label} must contain exactly {length} finite values")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


def _history(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != HISTORY_LENGTH:
        raise ResetAuditError(f"proprio_history must contain exactly {HISTORY_LENGTH} frames")
    return tuple(
        _vector(frame, PROPRIO_DIM, f"proprio_history[{index}]")
        for index, frame in enumerate(value)
    )


def _target_poses(value: Any) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not isinstance(value, Mapping) or set(value) != _COLORS:
        raise ResetAuditError("target_poses_w must contain exactly red and blue target poses")
    return tuple((color, _vector(value[color], 7, f"target_poses_w.{color}")) for color in ("red", "blue"))


def _same_vector(left: Sequence[float], right: Sequence[float], label: str) -> None:
    if len(left) != len(right) or any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=_STATE_ABS_TOL)
        for a, b in zip(left, right)
    ):
        raise ResetAuditError(f"paired resets disagree on {label}")


def _comparable_pose(left: Sequence[float], right: Sequence[float], label: str) -> None:
    distance = math.dist(left[:3], right[:3])
    norms = [math.sqrt(sum(v*v for v in pose[3:])) for pose in (left, right)]
    if any(abs(n-1.0) > .001 for n in norms):
        raise ResetAuditError(f"invalid orientation quaternion: {label}")
    dot = abs(sum(a*b for a, b in zip(left[3:], right[3:])) / math.prod(norms))
    angle = 2.0 * math.acos(min(1.0, dot))
    if distance > PAIR_POSITION_TOL_M or angle > PAIR_ORIENTATION_TOL_RAD:
        raise ResetAuditError(f"paired reset pose is not comparable: {label}")


def _valid_start(audit: "ResetAudit") -> None:
    # The audited wrapper explicitly zeroes its history on reset. Check that
    # operation per episode, not equality of two potentially stale histories.
    if any(abs(v) > 1e-6 for frame in audit.proprio_history for v in frame):
        raise ResetAuditError("reset history was not cleared")
    if (math.sqrt(sum(v*v for v in audit.robot_root_lin_vel_b)) > .03
            or math.sqrt(sum(v*v for v in audit.robot_root_ang_vel_b)) > .05):
        raise ResetAuditError("robot was moving immediately after reset")
    vx, vy, wz = audit.learner_start_body_velocity_body
    if math.hypot(vx, vy) > .03 or abs(wz) > .05:
        raise ResetAuditError("learner start is not stationary")


def derive_pair_seed(*, geometry_group_id: str, color_configuration: str, repeat: int) -> int:
    """Derive the one simulator seed for the red/blue instruction pair.

    Target color and natural-language instruction intentionally are not
    arguments.  A caller cannot therefore make either one a source of reset
    noise without first changing this auditable API.
    """
    group = _string(geometry_group_id, "geometry_group_id")
    if color_configuration not in _CONFIGURATIONS:
        raise ResetAuditError("unknown color configuration")
    count = _integer(repeat, "repeat")
    canonical = "\0".join((PAIR_SEED_ALGORITHM, group, color_configuration, str(count))).encode("utf-8")
    # Isaac environment seeds are represented as signed 31-bit integers.
    return int.from_bytes(hashlib.sha256(canonical).digest()[:4], "big") & 0x7FFF_FFFF


@dataclass(frozen=True)
class ResetAudit:
    """One color-task's post-reset state and first non-advancing RGB sample."""

    schema_version: str
    geometry_group_id: str
    color_configuration: str
    repeat: int
    target_color: str
    scene_metadata_sha256: str
    pair_seed_algorithm: str
    pair_seed: int
    physics_step_after_reset: int
    sim_time_after_reset_s: float
    robot_root_pose_w: tuple[float, ...]
    robot_root_lin_vel_b: tuple[float, ...]
    robot_root_ang_vel_b: tuple[float, ...]
    joint_pos: tuple[float, ...]
    joint_vel: tuple[float, ...]
    proprio_history: tuple[tuple[float, ...], ...]
    target_poses_w: tuple[tuple[str, tuple[float, ...]], ...]
    first_observation_seq: int
    first_render_request_seq: int
    first_camera_sensor_frame: int
    first_rgb_path: str
    first_rgb_sha256: str
    render_physics_step_before: int
    render_physics_step_after: int
    render_sim_time_before_s: float
    render_sim_time_after_s: float
    render_did_not_advance_physics: bool
    learner_start_physics_step: int
    learner_start_sim_time_s: float
    learner_start_robot_root_pose_w: tuple[float, ...]
    learner_start_body_velocity_body: tuple[float, ...]
    learner_first_observation_seq: int
    learner_first_render_request_seq: int
    learner_first_camera_sensor_frame: int
    learner_first_rgb_path: str
    learner_first_rgb_sha256: str
    learner_render_physics_step_before: int
    learner_render_physics_step_after: int
    learner_render_sim_time_before_s: float
    learner_render_sim_time_after_s: float
    learner_render_did_not_advance_physics: bool

    def validate(self) -> None:
        if self.schema_version != RESET_AUDIT_SCHEMA:
            raise ResetAuditError("unexpected paired reset audit schema")
        _string(self.geometry_group_id, "geometry_group_id")
        if self.color_configuration not in _CONFIGURATIONS:
            raise ResetAuditError("unknown color configuration")
        _integer(self.repeat, "repeat")
        if self.target_color not in _COLORS:
            raise ResetAuditError("target_color must be red or blue")
        if not isinstance(self.scene_metadata_sha256, str) or _HEX64.fullmatch(self.scene_metadata_sha256) is None:
            raise ResetAuditError("scene_metadata_sha256 must be a lowercase SHA-256 hex digest")
        if self.pair_seed_algorithm != PAIR_SEED_ALGORITHM:
            raise ResetAuditError("unexpected pair seed algorithm")
        expected_seed = derive_pair_seed(
            geometry_group_id=self.geometry_group_id,
            color_configuration=self.color_configuration,
            repeat=self.repeat,
        )
        if _integer(self.pair_seed, "pair_seed") != expected_seed:
            raise ResetAuditError("pair_seed is not derived from the color-independent pair identity")
        _integer(self.physics_step_after_reset, "physics_step_after_reset")
        _finite(self.sim_time_after_reset_s, "sim_time_after_reset_s")
        _vector(self.robot_root_pose_w, 7, "robot_root_pose_w")
        _vector(self.robot_root_lin_vel_b, 3, "robot_root_lin_vel_b")
        _vector(self.robot_root_ang_vel_b, 3, "robot_root_ang_vel_b")
        _vector(self.joint_pos, 12, "joint_pos")
        _vector(self.joint_vel, 12, "joint_vel")
        _history(self.proprio_history)
        _target_poses(dict(self.target_poses_w))
        _integer(self.first_observation_seq, "first_observation_seq")
        _integer(self.first_render_request_seq, "first_render_request_seq")
        _integer(self.first_camera_sensor_frame, "first_camera_sensor_frame")
        _string(self.first_rgb_path, "first_rgb_path")
        if not isinstance(self.first_rgb_sha256, str) or _HEX64.fullmatch(self.first_rgb_sha256) is None:
            raise ResetAuditError("first_rgb_sha256 must be a lowercase SHA-256 hex digest")
        _integer(self.render_physics_step_before, "render_physics_step_before")
        _integer(self.render_physics_step_after, "render_physics_step_after")
        before_time = _finite(self.render_sim_time_before_s, "render_sim_time_before_s")
        after_time = _finite(self.render_sim_time_after_s, "render_sim_time_after_s")
        if type(self.render_did_not_advance_physics) is not bool or not self.render_did_not_advance_physics:
            raise ResetAuditError("first RGB must explicitly prove its render did not advance physics")
        if self.render_physics_step_before != self.render_physics_step_after or before_time != after_time:
            raise ResetAuditError("first RGB render advanced physical time")
        _integer(self.learner_start_physics_step, "learner_start_physics_step")
        _finite(self.learner_start_sim_time_s, "learner_start_sim_time_s")
        _vector(self.learner_start_robot_root_pose_w, 7, "learner_start_robot_root_pose_w")
        _vector(self.learner_start_body_velocity_body, 3, "learner_start_body_velocity_body")
        _integer(self.learner_first_observation_seq, "learner_first_observation_seq")
        _integer(self.learner_first_render_request_seq, "learner_first_render_request_seq")
        _integer(self.learner_first_camera_sensor_frame, "learner_first_camera_sensor_frame")
        _string(self.learner_first_rgb_path, "learner_first_rgb_path")
        if not isinstance(self.learner_first_rgb_sha256, str) or _HEX64.fullmatch(self.learner_first_rgb_sha256) is None:
            raise ResetAuditError("learner_first_rgb_sha256 must be a lowercase SHA-256 hex digest")
        _integer(self.learner_render_physics_step_before, "learner_render_physics_step_before")
        _integer(self.learner_render_physics_step_after, "learner_render_physics_step_after")
        learner_before_time = _finite(self.learner_render_sim_time_before_s, "learner_render_sim_time_before_s")
        learner_after_time = _finite(self.learner_render_sim_time_after_s, "learner_render_sim_time_after_s")
        if (type(self.learner_render_did_not_advance_physics) is not bool
                or not self.learner_render_did_not_advance_physics):
            raise ResetAuditError("learner's first RGB must explicitly prove its render did not advance physics")
        if (self.learner_render_physics_step_before != self.learner_render_physics_step_after
                or learner_before_time != learner_after_time):
            raise ResetAuditError("learner's first RGB render advanced physical time")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        result = asdict(self)
        # JSON makes tuples lists; spelling this out keeps the on-disk schema
        # clear for independent validators that deliberately do not import us.
        result["target_poses_w"] = {color: list(pose) for color, pose in self.target_poses_w}
        return result


def reset_audit_from_dict(payload: Mapping[str, Any]) -> ResetAudit:
    """Parse only the exact sidecar schema; reject omitted or added fields."""
    expected = set(ResetAudit.__dataclass_fields__)
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ResetAuditError("paired reset audit keys do not match the frozen schema")
    raw = dict(payload)
    raw["robot_root_pose_w"] = _vector(raw["robot_root_pose_w"], 7, "robot_root_pose_w")
    raw["robot_root_lin_vel_b"] = _vector(raw["robot_root_lin_vel_b"], 3, "robot_root_lin_vel_b")
    raw["robot_root_ang_vel_b"] = _vector(raw["robot_root_ang_vel_b"], 3, "robot_root_ang_vel_b")
    raw["joint_pos"] = _vector(raw["joint_pos"], 12, "joint_pos")
    raw["joint_vel"] = _vector(raw["joint_vel"], 12, "joint_vel")
    raw["proprio_history"] = _history(raw["proprio_history"])
    raw["target_poses_w"] = _target_poses(raw["target_poses_w"])
    raw["learner_start_robot_root_pose_w"] = _vector(
        raw["learner_start_robot_root_pose_w"], 7, "learner_start_robot_root_pose_w"
    )
    raw["learner_start_body_velocity_body"] = _vector(
        raw["learner_start_body_velocity_body"], 3, "learner_start_body_velocity_body"
    )
    audit = ResetAudit(**raw)
    audit.validate()
    return audit


def validate_paired_resets(left: ResetAudit | Mapping[str, Any], right: ResetAudit | Mapping[str, Any]) -> dict[str, object]:
    """Check task-equivalent starts; retain per-file RGB integrity, not equality.

    Full joint/history data remain in audit files. Joint jitter and independent
    renderer noise are not task failures. Each history must actually be reset,
    each start stationary, and the physical scene unchanged by the instruction.
    """
    first = reset_audit_from_dict(left) if isinstance(left, Mapping) else left
    second = reset_audit_from_dict(right) if isinstance(right, Mapping) else right
    first.validate()
    second.validate()
    _valid_start(first)
    _valid_start(second)
    if {first.target_color, second.target_color} != _COLORS:
        raise ResetAuditError("a paired reset must contain exactly one red and one blue task")
    identity = (first.geometry_group_id, first.color_configuration, first.repeat, first.pair_seed_algorithm, first.pair_seed)
    other_identity = (second.geometry_group_id, second.color_configuration, second.repeat, second.pair_seed_algorithm, second.pair_seed)
    if identity != other_identity:
        raise ResetAuditError("paired resets use different group/configuration/repeat/seed identities")
    if first.scene_metadata_sha256 != second.scene_metadata_sha256:
        raise ResetAuditError("paired resets use different canonical scene metadata")
    for label, one, two in (
        ("physics_step_after_reset", (float(first.physics_step_after_reset),), (float(second.physics_step_after_reset),)),
        ("sim_time_after_reset_s", (first.sim_time_after_reset_s,), (second.sim_time_after_reset_s,)),
    ):
        _same_vector(one, two, label)
    _comparable_pose(first.robot_root_pose_w, second.robot_root_pose_w, "reset")
    _comparable_pose(first.learner_start_robot_root_pose_w, second.learner_start_robot_root_pose_w, "learner_start")
    if len(first.target_poses_w) != len(second.target_poses_w):  # defensive after exact validation
        raise ResetAuditError("paired resets disagree on target pose count")
    for (one_color, one_pose), (two_color, two_pose) in zip(first.target_poses_w, second.target_poses_w):
        if one_color != two_color:
            raise ResetAuditError("paired resets reorder target pose colors")
        _same_vector(one_pose, two_pose, f"target_poses_w.{one_color}")
    for label, one, two in (
        ("learner_start_physics_step", (float(first.learner_start_physics_step),), (float(second.learner_start_physics_step),)),
        ("learner_start_sim_time_s", (first.learner_start_sim_time_s,), (second.learner_start_sim_time_s,)),
    ):
        _same_vector(one, two, label)
    return {
        "schema_version": RESET_AUDIT_SCHEMA,
        "status": "PAIRED_RESET_MATCHED_NOT_TASK_APPROVED",
        "geometry_group_id": first.geometry_group_id,
        "color_configuration": first.color_configuration,
        "repeat": first.repeat,
        "pair_seed_algorithm": PAIR_SEED_ALGORITHM,
        "pair_seed": first.pair_seed,
        "task_colors": ["red", "blue"],
        "comparison_policy": PAIR_COMPARISON_POLICY,
        "position_tolerance_m": PAIR_POSITION_TOL_M,
        "orientation_tolerance_rad": PAIR_ORIENTATION_TOL_RAD,
        "cross_run_rgb_equality_required": False,
        "rgb_integrity_sha256_by_color": {
            a.target_color: {"reset": a.first_rgb_sha256, "learner_start": a.learner_first_rgb_sha256}
            for a in (first, second)
        },
        "scene_metadata_sha256": first.scene_metadata_sha256,
        "physics_and_reset_state_matched": True,
        "render_did_not_advance_physics": True,
        "navigation_success_approved": False,
        "dt1_approved": False,
    }
