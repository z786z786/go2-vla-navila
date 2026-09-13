"""3-D live-state construction and Go2 command safety for full episodes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


STATE_DIM, ACTION_STEPS, ACTION_DIM = 3, 50, 3
ACTION_NAMES = ("vx", "vy", "wz")
ACTION_BOUNDS = ((0.0, 0.5), (0.0, 0.0), (-0.5, 0.5))


def _vector(value: Sequence[float], length: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    output = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{label}[{index}] must be finite")
        result = float(item)
        if not math.isfinite(result):
            raise ValueError(f"{label}[{index}] must be finite")
        output.append(result)
    return output


def build_policy_state(linear_body: Sequence[float], angular_body: Sequence[float], *_unused: object) -> list[float]:
    """Accept legacy call shape but ignore all non-policy state by construction."""
    linear, angular = _vector(linear_body, 3, "linear_body"), _vector(angular_body, 3, "angular_body")
    return [linear[0], linear[1], angular[2]]


def validate_action_chunk(actions: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(actions, (str, bytes)) or len(actions) != ACTION_STEPS:
        raise ValueError(f"action chunk must contain {ACTION_STEPS} actions")
    return [_vector(action, ACTION_DIM, f"action[{index}]") for index, action in enumerate(actions)]


@dataclass(frozen=True)
class ActionSafetyResult:
    raw: tuple[float, float, float]
    applied: tuple[float, float, float]
    in_range: bool
    clipped_dimensions: tuple[str, ...]


def apply_action_safety(action: Sequence[float]) -> ActionSafetyResult:
    raw = tuple(_vector(action, ACTION_DIM, "action"))
    applied = tuple(max(low, min(high, value)) for value, (low, high) in zip(raw, ACTION_BOUNDS))
    clipped = tuple(name for name, before, after in zip(ACTION_NAMES, raw, applied) if before != after)
    return ActionSafetyResult(raw=raw, applied=applied, in_range=not clipped, clipped_dimensions=clipped)
