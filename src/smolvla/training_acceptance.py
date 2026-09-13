"""Pure acceptance rules shared by live and post-hoc SmolVLA training audits."""

from __future__ import annotations

import math
from typing import Any


def smoke_acceptance_checks(
    steps: int, history: list[dict[str, Any]], reloaded: dict[str, Any]
) -> dict[str, bool]:
    """Accept both the original 2k smoke run and longer equal-exposure runs."""
    return {
        "minimum_steps_met": steps >= 2000,
        "history_length_matches_steps": len(history) == steps,
        "all_losses_finite": all(math.isfinite(float(row["loss"])) for row in history),
        "checkpoint_reloaded": bool(reloaded.get("finite")),
        "output_action_shape": reloaded.get("output_shape") == [1, 50, 3],
    }
