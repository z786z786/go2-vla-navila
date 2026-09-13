"""Fail-closed learner-facing contract for the dual-target task.

The policy may see only an RGB observation, three body-frame velocity values,
and the fixed natural-language task.  Geometry and evaluator information is
kept out of this module's policy-input builder by construction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
TASK_KEY = "task"
POLICY_INPUTS = (IMAGE_KEY, STATE_KEY, TASK_KEY)
STATE_NAMES = ("body_vx", "body_vy", "body_yaw_rate")
ACTION_NAMES = ("vx", "vy", "wz")
RED_TASK = "Go to the red box and stop in front of it."
BLUE_TASK = "Go to the blue box and stop in front of it."
ALLOWED_TASKS = frozenset({RED_TASK, BLUE_TASK})
ACTION_CHUNK_SIZE = 50
EXECUTE_ACTION_STEPS = 10
HIGH_LEVEL_RATE_HZ = 50
ACTION_BOUNDS = ((0.0, 0.5), (0.0, 0.0), (-0.5, 0.5))

# This list is intentionally broader than the known legacy request fields.
# A new privileged field must be added to a separate evaluator sidecar, never
# silently allowed through the learner input mapping.
PROHIBITED_POLICY_SIGNALS = frozenset(
    {
        "episode_id", "frame_index", "time", "timestamp", "sim_time",
        "absolute_pose", "position", "robot_pose", "goal", "goal_pose",
        "goal_distance", "goal_direction", "target", "target_color",
        "target_position", "parking_region", "reference_path", "path",
        "waypoint", "gt_locations", "expert_action", "expert_command",
        "semantic_segmentation", "segmentation", "evaluator_stop",
        "collision", "geometry_group_id", "split", "seed",
    }
)


class ContractValidationError(ValueError):
    """Raised before a privileged, malformed, or unsafe value can be used."""


def _finite_vector(value: Sequence[object], length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ContractValidationError(f"{label} must contain exactly {length} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ContractValidationError(f"{label}[{index}] must be a finite number")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(f"{label}[{index}] must be a finite number") from exc
        if not math.isfinite(number):
            raise ContractValidationError(f"{label}[{index}] must be a finite number")
        result.append(number)
    return tuple(result)


def validate_policy_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical, strict policy input mapping.

    Audit identifiers belong in a transport envelope, not in ``payload``.  An
    exact allowlist avoids accidentally passing a future evaluator field to the
    policy during a convenient ``dict.update``.
    """
    keys = set(payload)
    forbidden = sorted(str(key) for key in keys & PROHIBITED_POLICY_SIGNALS)
    if forbidden:
        raise ContractValidationError(f"prohibited policy signals: {forbidden}")
    if keys != set(POLICY_INPUTS):
        unexpected = sorted(str(key) for key in keys - set(POLICY_INPUTS))
        missing = sorted(set(POLICY_INPUTS) - keys)
        raise ContractValidationError(
            f"policy inputs must be exactly {list(POLICY_INPUTS)!r}; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    image = payload[IMAGE_KEY]
    if image is None:
        raise ContractValidationError("current RGB image must not be None")
    state = _finite_vector(payload[STATE_KEY], len(STATE_NAMES), STATE_KEY)
    task = payload[TASK_KEY]
    if not isinstance(task, str):
        raise ContractValidationError("task must be an instruction string")
    # Normalize whitespace only, then compare against the two frozen task
    # texts.  This prevents a goal coordinate or a target slot from entering
    # the language prompt under the guise of a descriptive instruction.
    normalized_task = " ".join(task.split())
    if normalized_task not in ALLOWED_TASKS:
        raise ContractValidationError("task must be one of the two frozen dual-target instructions")
    return {IMAGE_KEY: image, STATE_KEY: list(state), TASK_KEY: normalized_task}


def build_policy_envelope(
    *, image: Any, body_velocity: Sequence[object], task: str, audit: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a request that keeps audit metadata structurally separate."""
    policy_input = validate_policy_input(
        {IMAGE_KEY: image, STATE_KEY: body_velocity, TASK_KEY: task}
    )
    envelope: dict[str, Any] = {"policy_input": policy_input}
    if audit:
        envelope["audit"] = dict(audit)
    return envelope


def validate_action_chunk(actions: Sequence[Sequence[object]]) -> list[tuple[float, float, float]]:
    """Validate, but do not clamp, a SmolVLA 50×3 response."""
    if isinstance(actions, (str, bytes)) or len(actions) != ACTION_CHUNK_SIZE:
        raise ContractValidationError(f"action chunk must have {ACTION_CHUNK_SIZE} actions")
    return [
        _finite_vector(action, len(ACTION_NAMES), f"action[{index}]")
        for index, action in enumerate(actions)
    ]


@dataclass(frozen=True)
class AppliedAction:
    """Raw output and deployment command; raw values remain available to scoring."""

    raw: tuple[float, float, float]
    applied: tuple[float, float, float]
    clipped_dimensions: tuple[str, ...]


def apply_deployment_bounds(action: Sequence[object]) -> AppliedAction:
    """Apply only the fixed deployment bounds after preserving finite raw output."""
    raw = _finite_vector(action, len(ACTION_NAMES), "action")
    applied = tuple(
        max(lower, min(upper, value))
        for value, (lower, upper) in zip(raw, ACTION_BOUNDS)
    )
    clipped = tuple(
        name for name, before, after in zip(ACTION_NAMES, raw, applied)
        if before != after
    )
    return AppliedAction(raw=raw, applied=applied, clipped_dimensions=clipped)


@dataclass(frozen=True)
class TaskContract:
    """DT0 draft values; DT1 simulation calibration must freeze the replacement lock."""

    schema_version: str = "go2-dual-target-task-contract-v1-draft"
    status: str = "pending_sim_validation"
    action_chunk_size: int = ACTION_CHUNK_SIZE
    execute_action_steps: int = EXECUTE_ACTION_STEPS
    high_level_rate_hz: int = HIGH_LEVEL_RATE_HZ
    max_vx_mps: float = ACTION_BOUNDS[0][1]
    max_abs_wz_radps: float = ACTION_BOUNDS[2][1]
    parking_radius_m: float = 0.3
    stop_raw_vx_mps: float = 0.03
    stop_raw_wz_radps: float = 0.05
    required_stop_duration_s: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
