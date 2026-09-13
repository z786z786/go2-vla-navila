#!/usr/bin/env python3
"""M5-compatible state construction and M7 action safety checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass


STATE_DIM = 30
ACTION_STEPS = 50
ACTION_DIM = 3
ACTION_NAMES = ("vx", "vy", "wz")
ACTION_BOUNDS = ((0.0, 0.5), (0.0, 0.0), (-0.5, 0.5))


def _finite_vector(value: Sequence[float], length: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{label}[{index}] must be finite")
        try:
            converted = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{index}] must be finite") from exc
        if not math.isfinite(converted):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(converted)
    return result


def quaternion_wxyz_to_rpy(quaternion_wxyz: Sequence[float]) -> tuple[float, float, float]:
    quat = _finite_vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = math.sqrt(sum(value * value for value in quat))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"expected unit quaternion, got norm={norm}")
    w, x, y, z = (value / norm for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(value % (2.0 * math.pi) for value in (roll, pitch, yaw))


def build_policy_state(
    linear_body: Sequence[float],
    angular_body: Sequence[float],
    quaternion_wxyz: Sequence[float],
    joint_position: Sequence[float],
    default_joint_position: Sequence[float],
    joint_velocity: Sequence[float],
) -> list[float]:
    linear = _finite_vector(linear_body, 3, "linear_body")
    angular = _finite_vector(angular_body, 3, "angular_body")
    position = _finite_vector(joint_position, 12, "joint_position")
    default = _finite_vector(default_joint_position, 12, "default_joint_position")
    velocity = _finite_vector(joint_velocity, 12, "joint_velocity")
    rpy = list(quaternion_wxyz_to_rpy(quaternion_wxyz))
    state = linear[:2] + angular[2:3] + rpy + [a - b for a, b in zip(position, default)] + velocity
    if len(state) != STATE_DIM or not all(math.isfinite(value) for value in state):
        raise ValueError(f"constructed policy state must be finite and {STATE_DIM}-D")
    return state


def build_policy_state_from_m4_record(record: Mapping[str, object]) -> list[float]:
    try:
        raw_values = record["robot_state"]
        if isinstance(raw_values, (str, bytes)) or len(raw_values) not in (31, 33):  # type: ignore[arg-type]
            raise ValueError("robot_state must contain either 31 legacy or 33 complete values")
        raw_state = _finite_vector(raw_values, len(raw_values), "robot_state")  # type: ignore[arg-type]
        current_velocity = record["current_velocity"]
        robot_pose = record["robot_pose"]
        if not isinstance(current_velocity, Mapping) or not isinstance(robot_pose, Mapping):
            raise ValueError("current_velocity and robot_pose must be objects")
        linear = _finite_vector(current_velocity["linear_body"], 3, "linear_body")  # type: ignore[arg-type]
        angular = _finite_vector(current_velocity["angular_body"], 3, "angular_body")  # type: ignore[arg-type]
        rpy = list(quaternion_wxyz_to_rpy(robot_pose["quaternion_wxyz"]))  # type: ignore[arg-type]
        joint_velocity = _finite_vector(record["joint_velocity"], 12, "joint_velocity")  # type: ignore[arg-type]
    except KeyError as exc:
        raise ValueError(f"M4 record is missing {exc.args[0]!r}") from exc
    # The original M4 hook indexed the `[1, 3]` RPY tensor and serialized only
    # roll: legacy records therefore place joint offsets at [7:19].  Fixed D5
    # records retain all three RPY values and place them at [9:21].  Both use
    # quaternion-derived RPY in the 30-D policy interface.
    joint_offset_start = 7 if len(raw_state) == 31 else 9
    state = linear[:2] + angular[2:3] + rpy + raw_state[joint_offset_start : joint_offset_start + 12] + joint_velocity
    if len(state) != STATE_DIM or not all(math.isfinite(value) for value in state):
        raise ValueError(f"M4-derived policy state must be finite and {STATE_DIM}-D")
    return state


def validate_action_chunk(actions: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(actions, (str, bytes)) or len(actions) != ACTION_STEPS:
        raise ValueError(f"action chunk must contain {ACTION_STEPS} actions")
    return [
        _finite_vector(action, ACTION_DIM, f"action[{index}]")
        for index, action in enumerate(actions)
    ]


@dataclass(frozen=True)
class ActionSafetyResult:
    raw: tuple[float, float, float]
    applied: tuple[float, float, float]
    in_range: bool
    clipped_dimensions: tuple[str, ...]


def apply_action_safety(action: Sequence[float]) -> ActionSafetyResult:
    raw_values = _finite_vector(action, ACTION_DIM, "action")
    applied = tuple(
        max(lower, min(upper, value))
        for value, (lower, upper) in zip(raw_values, ACTION_BOUNDS)
    )
    raw = tuple(raw_values)
    clipped = tuple(
        name for name, raw_value, applied_value in zip(ACTION_NAMES, raw, applied)
        if raw_value != applied_value
    )
    return ActionSafetyResult(
        raw=raw,
        applied=applied,
        in_range=not clipped,
        clipped_dimensions=clipped,
    )
