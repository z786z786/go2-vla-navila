from __future__ import annotations

from typing import Iterable, Mapping


def compute_target_stop_distance(
    *,
    size: Iterable[float],
    shape: str,
    robot_goal_clearance_radius: float,
    target_stop_extra_margin: float,
    controller_cfg: Mapping[str, object] | None = None,
) -> float:
    size_values = [float(value) for value in size]
    if len(size_values) < 2:
        return float(robot_goal_clearance_radius) + float(target_stop_extra_margin)

    cfg = controller_cfg or {}
    shape_key = str(shape).strip().lower()
    extent_mode = str(cfg.get("default_target_stop_extent_mode", "max")).strip().lower()
    extra_margin = float(target_stop_extra_margin)

    if shape_key == "door":
        extent_mode = str(cfg.get("door_target_stop_extent_mode", "min")).strip().lower()
        extra_margin = float(cfg.get("door_target_stop_extra_margin", target_stop_extra_margin))
    elif shape_key in {"box", "suitcase", "dark_gray_suitcase"}:
        extent_mode = str(cfg.get("box_target_stop_extent_mode", "min")).strip().lower()
        extra_margin = float(cfg.get("box_target_stop_extra_margin", max(0.0, float(target_stop_extra_margin) - 0.04)))

    if extent_mode == "min":
        target_extent = min(size_values[0], size_values[1]) / 2.0
    else:
        target_extent = max(size_values[0], size_values[1]) / 2.0
    return float(robot_goal_clearance_radius) + target_extent + max(0.0, extra_margin)
