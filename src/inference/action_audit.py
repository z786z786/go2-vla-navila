"""Independent raw-action range accounting for M7 action chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.inference.state import ACTION_BOUNDS, apply_action_safety, validate_action_chunk


DIMENSION_KEYS = (
    "vx_below_min",
    "vx_above_max",
    "vy_not_zero",
    "wz_below_min",
    "wz_above_max",
)


@dataclass(frozen=True)
class ActionRangeSummary:
    total_chunks: int
    total_vectors: int
    violating_vectors: int
    chunks_with_violation: int
    executed_vectors: int
    executed_violating_vectors: int
    tail_vectors: int
    tail_violating_vectors: int
    per_dimension: dict[str, int]

    @property
    def raw_output_range_passed(self) -> bool:
        return self.violating_vectors == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_action_chunk_count": self.total_chunks,
            "raw_action_vector_count": self.total_vectors,
            "raw_action_vector_violation_count": self.violating_vectors,
            "raw_chunk_with_violation_count": self.chunks_with_violation,
            "executed_policy_action_count": self.executed_vectors,
            "executed_policy_action_violation_count": self.executed_violating_vectors,
            "discarded_action_count": self.tail_vectors,
            "discarded_action_violation_count": self.tail_violating_vectors,
            "raw_action_dimension_violation_count": dict(self.per_dimension),
            "raw_output_range_passed": self.raw_output_range_passed,
        }


def _dimension_violations(action: Sequence[float]) -> tuple[str, ...]:
    result: list[str] = []
    for index, value in enumerate(action):
        lower, upper = ACTION_BOUNDS[index]
        if value < lower:
            result.append(("vx_below_min", "vy_not_zero", "wz_below_min")[index])
        elif value > upper:
            result.append(("vx_above_max", "vy_not_zero", "wz_above_max")[index])
    return tuple(result)


def summarize_action_range(
    chunks: Sequence[Sequence[Sequence[float]]], *, execute_steps: int
) -> ActionRangeSummary:
    """Count raw action-range violations without altering the actions."""
    if isinstance(execute_steps, bool) or execute_steps < 1:
        raise ValueError("execute_steps must be a positive integer")

    total_vectors = 0
    violating_vectors = 0
    chunks_with_violation = 0
    executed_vectors = 0
    executed_violating_vectors = 0
    tail_vectors = 0
    tail_violating_vectors = 0
    per_dimension = {key: 0 for key in DIMENSION_KEYS}

    for chunk_index, chunk in enumerate(chunks):
        actions = validate_action_chunk(chunk)
        if execute_steps > len(actions):
            raise ValueError(
                f"execute_steps must be at most chunk length {len(actions)} for chunk {chunk_index}"
            )
        chunk_has_violation = False
        for action_index, action in enumerate(actions):
            decision = apply_action_safety(action)
            dimensions = _dimension_violations(decision.raw)
            is_violation = not decision.in_range
            total_vectors += 1
            if action_index < execute_steps:
                executed_vectors += 1
                if is_violation:
                    executed_violating_vectors += 1
            else:
                tail_vectors += 1
                if is_violation:
                    tail_violating_vectors += 1
            if is_violation:
                violating_vectors += 1
                chunk_has_violation = True
                for key in dimensions:
                    per_dimension[key] += 1
        if chunk_has_violation:
            chunks_with_violation += 1

    return ActionRangeSummary(
        total_chunks=len(chunks),
        total_vectors=total_vectors,
        violating_vectors=violating_vectors,
        chunks_with_violation=chunks_with_violation,
        executed_vectors=executed_vectors,
        executed_violating_vectors=executed_violating_vectors,
        tail_vectors=tail_vectors,
        tail_violating_vectors=tail_violating_vectors,
        per_dimension=per_dimension,
    )
