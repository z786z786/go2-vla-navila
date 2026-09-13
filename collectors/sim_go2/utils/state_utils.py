from __future__ import annotations

import math
from typing import Sequence

STATE_FIELDS = ("vx", "vy", "vz")


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return math.cos(half), 0.0, 0.0, math.sin(half)


def quat_wxyz_to_yaw(quaternion: Sequence[float]) -> float:
    w, x, y, z = [float(value) for value in quaternion]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_body_xy(dx_world: float, dy_world: float, yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    rel_x = cos_yaw * float(dx_world) + sin_yaw * float(dy_world)
    rel_y = -sin_yaw * float(dx_world) + cos_yaw * float(dy_world)
    return rel_x, rel_y


def build_debug_nav_state(robot_position: Sequence[float], robot_yaw: float, base_lin_vel_b: Sequence[float], base_ang_vel_b: Sequence[float], goal_position: Sequence[float]) -> dict[str, float]:
    dx_world = float(goal_position[0]) - float(robot_position[0])
    dy_world = float(goal_position[1]) - float(robot_position[1])
    goal_rel_x, goal_rel_y = world_to_body_xy(dx_world, dy_world, robot_yaw)
    goal_distance = math.hypot(goal_rel_x, goal_rel_y)
    goal_heading = wrap_to_pi(math.atan2(goal_rel_y, goal_rel_x))
    return {
        "vx": float(base_lin_vel_b[0]),
        "vy": float(base_lin_vel_b[1]),
        "vz": float(base_lin_vel_b[2]),
        "yaw": float(robot_yaw),
        "yaw_rate": float(base_ang_vel_b[2]),
        "robot_x": float(robot_position[0]),
        "robot_y": float(robot_position[1]),
        "robot_z": float(robot_position[2]),
        "goal_x": float(goal_position[0]),
        "goal_y": float(goal_position[1]),
        "goal_z": float(goal_position[2]),
        "goal_rel_x": float(goal_rel_x),
        "goal_rel_y": float(goal_rel_y),
        "goal_distance": float(goal_distance),
        "goal_heading": float(goal_heading),
    }


def extract_train_state_from_debug(debug_state: dict[str, float]) -> list[float]:
    return [float(debug_state[field]) for field in STATE_FIELDS]
