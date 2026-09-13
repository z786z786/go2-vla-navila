from __future__ import annotations

import math
from typing import Any, Mapping

from collectors.sim_go2.controllers.heuristic_nav_controller import PrivilegedNavExpert
from collectors.sim_go2.utils.state_utils import build_debug_nav_state, world_to_body_xy


def _camera_pose(robot_position: tuple[float, float, float], robot_yaw: float, camera_cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    offset_cfg = camera_cfg.get("offset", {})
    offset = offset_cfg.get("pos", (0.0, 0.0, 0.0))
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)
    camera_x = float(robot_position[0]) + cos_yaw * float(offset[0]) - sin_yaw * float(offset[1])
    camera_y = float(robot_position[1]) + sin_yaw * float(offset[0]) + cos_yaw * float(offset[1])
    camera_z = float(robot_position[2]) + float(offset[2])
    return camera_x, camera_y, camera_z


def camera_fovs(camera_cfg: Mapping[str, Any]) -> tuple[float, float]:
    focal_length = float(camera_cfg.get("focal_length", 24.0))
    horizontal_aperture = float(camera_cfg.get("horizontal_aperture", 20.955))
    width = max(1, int(camera_cfg.get("width", 640)))
    height = max(1, int(camera_cfg.get("height", 384)))
    vertical_aperture = horizontal_aperture * (height / width)
    hfov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
    vfov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
    return hfov, vfov


def estimate_target_visibility(
    *,
    robot_position: tuple[float, float, float],
    robot_yaw: float,
    goal_position: tuple[float, float, float],
    camera_cfg: Mapping[str, Any],
    target_cfg: Mapping[str, Any],
    visibility_cfg: Mapping[str, Any],
) -> dict[str, float | bool]:
    camera_x, camera_y, camera_z = _camera_pose(robot_position, robot_yaw, camera_cfg)
    goal_dx = float(goal_position[0]) - camera_x
    goal_dy = float(goal_position[1]) - camera_y
    goal_dz = float(goal_position[2]) - camera_z
    rel_x, rel_y = world_to_body_xy(goal_dx, goal_dy, robot_yaw)
    rel_z = goal_dz

    width = max(1, int(camera_cfg.get("width", 640)))
    height = max(1, int(camera_cfg.get("height", 384)))
    hfov, vfov = camera_fovs(camera_cfg)
    min_depth = float(visibility_cfg.get("min_depth", 0.35))
    heading_margin = math.radians(float(visibility_cfg.get("horizontal_margin_deg", 3.0)))
    pitch_margin = math.radians(float(visibility_cfg.get("vertical_margin_deg", 3.0)))
    min_pixel_ratio = float(visibility_cfg.get("min_pixel_ratio", 0.002))

    if rel_x <= min_depth:
        return {
            "visible": False,
            "pixel_ratio": 0.0,
            "heading_deg": math.degrees(math.atan2(rel_y, max(rel_x, 1.0e-6))),
            "pitch_deg": math.degrees(math.atan2(rel_z, max(rel_x, 1.0e-6))),
            "distance": math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy + goal_dz * goal_dz),
        }

    heading = math.atan2(rel_y, rel_x)
    pitch = math.atan2(rel_z, rel_x)
    fx = width / (2.0 * math.tan(hfov * 0.5))
    fy = height / (2.0 * math.tan(vfov * 0.5))
    target_width = float(target_cfg.get("size", (0.35, 0.35, 0.50))[0])
    target_height = float(target_cfg.get("size", (0.35, 0.35, 0.50))[2])
    bbox_width = fx * target_width / rel_x
    bbox_height = fy * target_height / rel_x
    pixel_ratio = max(0.0, min(1.0, (bbox_width * bbox_height) / float(width * height)))
    visible = (
        abs(heading) <= max(0.0, hfov * 0.5 - heading_margin)
        and abs(pitch) <= max(0.0, vfov * 0.5 - pitch_margin)
        and pixel_ratio >= min_pixel_ratio
    )
    return {
        "visible": visible,
        "pixel_ratio": pixel_ratio,
        "heading_deg": math.degrees(heading),
        "pitch_deg": math.degrees(pitch),
        "distance": math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy + goal_dz * goal_dz),
    }


def rollout_visibility(
    *,
    robot_position: tuple[float, float, float],
    robot_yaw: float,
    goal_position: tuple[float, float, float],
    camera_cfg: Mapping[str, Any],
    controller_cfg: Mapping[str, Any],
    target_cfg: Mapping[str, Any],
    visibility_cfg: Mapping[str, Any],
    horizon_steps: int,
    control_dt: float,
) -> list[dict[str, float | bool]]:
    controller = PrivilegedNavExpert(controller_cfg, control_dt=control_dt)
    controller.reset()
    pose = [float(robot_position[0]), float(robot_position[1]), float(robot_position[2])]
    yaw = float(robot_yaw)
    trace: list[dict[str, float | bool]] = []
    for _ in range(int(horizon_steps) + 1):
        metrics = estimate_target_visibility(
            robot_position=(pose[0], pose[1], pose[2]),
            robot_yaw=yaw,
            goal_position=goal_position,
            camera_cfg=camera_cfg,
            target_cfg=target_cfg,
            visibility_cfg=visibility_cfg,
        )
        trace.append(metrics)
        debug_state = build_debug_nav_state(
            robot_position=(pose[0], pose[1], pose[2]),
            robot_yaw=yaw,
            base_lin_vel_b=(0.0, 0.0, 0.0),
            base_ang_vel_b=(0.0, 0.0, 0.0),
            goal_position=goal_position,
        )
        command = controller.compute_command({"debug_state": debug_state, "task": {}})
        vx, wz = [float(value) for value in command["raw_cmd_train"]]
        yaw = ((yaw + wz * control_dt) + math.pi) % (2.0 * math.pi) - math.pi
        pose[0] += vx * math.cos(yaw) * control_dt
        pose[1] += vx * math.sin(yaw) * control_dt
    return trace

