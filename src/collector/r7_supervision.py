"""Pure validation rules for R7 terminal-stop supervision.

The official PD planner remains the expert during navigation.  After its first
successful action, an explicitly labelled zero-command hold is collected so the
policy is supervised to stop rather than merely to enter the success radius.
This module is runtime-independent: it defines the evidence contract used both
by the Isaac collector and by the canary quality gate.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ACTION_DIM = 3
PD_ACTION_SOURCE = "pd_planner"
TERMINAL_HOLD_ACTION_SOURCE = "terminal_hold_zero"
ZERO_ACTION = (0.0, 0.0, 0.0)
_EPSILON = 1e-9


def command_vector(value: Any, *, name: str) -> tuple[float, float, float]:
    """Validate and canonicalize one finite Go2 ``[vx, vy, wz]`` command."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != ACTION_DIM:
        raise ValueError(f"{name} must be a length-{ACTION_DIM} sequence")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def is_zero_action(value: Any, *, tolerance: float = _EPSILON) -> bool:
    try:
        return all(abs(item) <= tolerance for item in command_vector(value, name="action"))
    except (TypeError, ValueError):
        return False


def commands_match(left: Any, right: Any, *, tolerance: float = _EPSILON) -> bool:
    try:
        return all(abs(a - b) <= tolerance for a, b in zip(command_vector(left, name="left"), command_vector(right, name="right"), strict=True))
    except (TypeError, ValueError):
        return False


def supervised_action(pd_command: Any, *, terminal_hold: bool) -> tuple[tuple[float, float, float], str]:
    """Return the label/executed action while retaining the unmodified PD command."""
    pd = command_vector(pd_command, name="pd_command")
    return (ZERO_ACTION, TERMINAL_HOLD_ACTION_SOURCE) if terminal_hold else (pd, PD_ACTION_SOURCE)


def command_semantics(records: Sequence[Mapping[str, Any]], *, allow_legacy: bool) -> dict[str, Any]:
    """Check the R7 command provenance without treating a terminal override as PD drift."""
    failures: list[dict[str, Any]] = []
    sources: dict[str, int] = {PD_ACTION_SOURCE: 0, TERMINAL_HOLD_ACTION_SOURCE: 0, "legacy": 0}
    for index, record in enumerate(records):
        source = record.get("action_source")
        planner = record.get("planner_command")
        locomotion = record.get("locomotion_command")
        label = [record.get("expert_vx"), record.get("expert_vy"), record.get("expert_wz")]
        if source is None:
            source = "legacy"
        if source not in sources:
            failures.append({"frame_index": index, "reason": "unknown_action_source"})
            continue
        try:
            command_vector(planner, name="planner_command")
            command_vector(locomotion, name="locomotion_command")
            command_vector(label, name="expert_action")
        except (TypeError, ValueError):
            failures.append({"frame_index": index, "reason": "nonfinite_or_wrong_shape_command"})
            continue
        sources[source] += 1
        if not commands_match(planner, locomotion) or not commands_match(planner, label):
            failures.append({"frame_index": index, "reason": "label_or_locomotion_differs_from_planner"})
            continue
        if source == "legacy":
            if not allow_legacy:
                failures.append({"frame_index": index, "reason": "missing_r7_action_source"})
            continue
        pd_command = record.get("pd_planner_command")
        try:
            command_vector(pd_command, name="pd_planner_command")
        except (TypeError, ValueError):
            failures.append({"frame_index": index, "reason": "missing_or_invalid_pd_planner_command"})
            continue
        if source == PD_ACTION_SOURCE and not commands_match(pd_command, planner):
            failures.append({"frame_index": index, "reason": "pd_planner_command_changed_before_execution"})
        if source == TERMINAL_HOLD_ACTION_SOURCE and not is_zero_action(planner):
            failures.append({"frame_index": index, "reason": "terminal_hold_label_not_zero"})
    return {"passed": not failures, "sources": sources, "failures": failures}


def terminal_hold_audit(records: Sequence[Mapping[str, Any]], *, requested_frames: int) -> dict[str, Any]:
    """Validate that a requested R7 terminal hold is real, contiguous, and terminal."""
    if not isinstance(requested_frames, int) or isinstance(requested_frames, bool) or requested_frames < 0:
        raise ValueError("requested_frames must be a non-negative integer")
    hold_indices = [index for index, row in enumerate(records) if row.get("action_source") == TERMINAL_HOLD_ACTION_SOURCE]
    expected_indices = list(range(len(records) - requested_frames, len(records))) if requested_frames else []
    exact_count = len(hold_indices) == requested_frames
    tail_contiguous = hold_indices == expected_indices
    zero_labels = all(is_zero_action(row.get("planner_command")) for row in (records[index] for index in hold_indices))
    environment_done = all(row.get("raw_environment_done") is True for row in (records[index] for index in hold_indices))
    inside_radius = True
    for index in hold_indices:
        row = records[index]
        try:
            distance = float(row["next_distance_to_goal_xy_m"])
            radius = float(row["goal_pose"]["success_radius_m"])
            inside_radius &= math.isfinite(distance) and math.isfinite(radius) and distance < radius
        except (KeyError, TypeError, ValueError):
            inside_radius = False
    final_success = bool(records) and bool(records[-1].get("termination")) and bool(records[-1].get("success"))
    passed = exact_count and tail_contiguous and zero_labels and environment_done and inside_radius and final_success
    return {
        "requested_frames": requested_frames,
        "recorded_frames": len(hold_indices),
        "frame_indices": hold_indices,
        "exact_count": exact_count,
        "tail_contiguous": tail_contiguous,
        "zero_labels": zero_labels,
        "raw_environment_done_each_frame": environment_done,
        "inside_success_radius": inside_radius,
        "final_record_success_and_termination": final_success,
        "passed": passed,
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if left_scale <= _EPSILON or right_scale <= _EPSILON:
        return None
    return numerator / (left_scale * right_scale)


def pd_motion_semantics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit that signed PD commands cause the expected signed body motion.

    It uses only normal PD-navigation frames; terminal zero holds intentionally
    do not participate because their label is an evaluator-stop supervisor.
    Thresholds are conservative relative to the frozen D5-R6 evidence
    (``vx`` sign 0.996 and ``wz`` sign 0.826).
    """
    definitions = {
        "vx": {"action_index": 0, "velocity_key": "linear_body", "velocity_index": 0, "command_deadband": 0.05, "sign_min": 0.90},
        "wz": {"action_index": 2, "velocity_key": "angular_body", "velocity_index": 2, "command_deadband": 0.10, "sign_min": 0.70},
    }
    result: dict[str, Any] = {}
    for name, definition in definitions.items():
        commands: list[float] = []
        responses: list[float] = []
        for row in records:
            if row.get("action_source") not in {PD_ACTION_SOURCE, None}:
                continue
            try:
                command = command_vector(row.get("pd_planner_command", row.get("planner_command")), name="pd_command")[definition["action_index"]]
                response = float(row["next_current_velocity"][definition["velocity_key"]][definition["velocity_index"]])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(command) > definition["command_deadband"] and abs(response) > 0.005 and math.isfinite(response):
                commands.append(command)
                responses.append(response)
        agreement = (
            sum((command > 0.0) == (response > 0.0) for command, response in zip(commands, responses, strict=True)) / len(commands)
            if commands else 0.0
        )
        correlation = _pearson(commands, responses)
        passed = len(commands) >= 10 and agreement >= definition["sign_min"] and correlation is not None and correlation >= 0.5
        result[name] = {
            "eligible_frames": len(commands),
            "sign_agreement": agreement,
            "correlation": correlation,
            "thresholds": {"minimum_eligible_frames": 10, "minimum_sign_agreement": definition["sign_min"], "minimum_correlation": 0.5},
            "passed": passed,
        }
    return {"per_dimension": result, "passed": all(item["passed"] for item in result.values())}
