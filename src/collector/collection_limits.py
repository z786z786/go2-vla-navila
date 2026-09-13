"""Bounded-collection helpers shared by full-episode Go2 PD runs.

The official NaVILA planner is deliberately not edited.  These helpers only
bound the instrumentation process so a PD command that never reaches an
environment terminal state cannot consume the collection worker indefinitely.
"""

from __future__ import annotations

from typing import Any


EXPERT_PATH_MARKER_PRIM_PATH = "/Visuals/Command/pos_goal_command"


def is_expert_path_marker(prim_path: object) -> bool:
    """Identify the official planner's GT expert-path marker namespace."""
    return str(prim_path).startswith(EXPERT_PATH_MARKER_PRIM_PATH)


def pd_frame_limit_reached(max_pd_frames: int | None, next_record_count: int) -> bool:
    """Return whether the next non-terminal PD record exhausts the run budget."""
    return max_pd_frames is not None and next_record_count >= max_pd_frames


def result_with_forced_done(result: Any) -> tuple[Any, Any, bool, Any]:
    """Return an environment-step result that stops the immutable outer loop."""
    if not isinstance(result, tuple) or len(result) != 4:
        raise TypeError("VLNEnvWrapper.step must return a four-tuple")
    observation, reward, _done, info = result
    return observation, reward, True, info
