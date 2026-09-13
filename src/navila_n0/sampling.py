"""Terminal-zero audits and deterministic ≤10% sampler cap."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


NEAR_ZERO_THRESHOLDS = (0.02, 0.02, 0.03)


def is_near_zero(action: Sequence[float]) -> bool:
    if len(action) != 3:
        raise ValueError("action must have exactly three elements")
    return all(abs(float(value)) < threshold for value, threshold in zip(action, NEAR_ZERO_THRESHOLDS))


def terminal_action_statistics(actions: Sequence[Sequence[float]], terminal_start_frame: int, *, chunk_size: int = 50) -> dict[str, Any]:
    if terminal_start_frame < 0 or terminal_start_frame > len(actions):
        raise ValueError("terminal_start_frame is outside the action sequence")
    if chunk_size != 50:
        raise ValueError("N0 fixes chunk_size at 50")
    near = [is_near_zero(action) for action in actions]
    chunks = [near[start : start + chunk_size] for start in range(0, max(0, len(near) - chunk_size + 1))]
    ratios = [sum(chunk) / chunk_size for chunk in chunks]
    terminal = actions[terminal_start_frame:]
    exact_tail = len(terminal) == 50 and all(is_near_zero(action) for action in terminal)
    return {
        "frame_count": len(actions),
        "zero_action_frame_ratio": sum(near) / len(near) if near else 0.0,
        "near_zero_chunk_ratio_ge_80pct": sum(ratio >= 0.8 for ratio in ratios) / len(ratios) if ratios else 0.0,
        "near_zero_chunk_ratio_100pct": sum(ratio == 1.0 for ratio in ratios) / len(ratios) if ratios else 0.0,
        "terminal_zero_tail_exactly_50": exact_tail,
        "terminal_frame_count": len(terminal),
    }


def cap_terminal_near_zero_samples(samples: Sequence[Mapping[str, Any]], *, maximum_ratio: float = 0.10) -> dict[str, Any]:
    """Keep all normal anchors and a deterministic capped subset of terminal/near-zero anchors."""
    if not 0 <= maximum_ratio < 1:
        raise ValueError("maximum_ratio must be in [0, 1)")
    flagged = [item for item in samples if bool(item.get("terminal", False)) or bool(item.get("near_zero_chunk", False))]
    normal = [item for item in samples if item not in flagged]
    maximum_flagged = int((maximum_ratio * len(normal)) // (1.0 - maximum_ratio))
    ranked = sorted(flagged, key=lambda item: hashlib.sha256(str(item["sample_id"]).encode()).hexdigest())
    selected = [*normal, *ranked[:maximum_flagged]]
    actual_ratio = len(ranked[:maximum_flagged]) / len(selected) if selected else 0.0
    return {
        "selected_sample_ids": [str(item["sample_id"]) for item in selected],
        "total_candidates": len(samples),
        "terminal_or_near_zero_candidates": len(flagged),
        "terminal_or_near_zero_selected": len(ranked[:maximum_flagged]),
        "terminal_or_near_zero_selected_ratio": actual_ratio,
        "maximum_ratio": maximum_ratio,
        "passed": actual_ratio <= maximum_ratio + 1e-12,
    }
