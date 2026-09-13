from __future__ import annotations

import math
from typing import Any, Mapping

from collectors.sim_go2.utils.action_utils import clip_train_action, raw_full_from_train_action
from collectors.sim_go2.utils.state_utils import wrap_to_pi
from collectors.sim_go2.utils.target_stop import compute_target_stop_distance


class PrivilegedNavExpert:
    """Privileged expert for Go2 navigation collection.

    The learner only sees image + instruction + state=[vx, vy, vz],
    but the expert is allowed to use privileged goal-relative signals.
    """

    def __init__(self, controller_cfg: Mapping[str, Any], control_dt: float = 0.05):
        self.controller_cfg = controller_cfg
        self.control_dt = float(control_dt)
        self.vx_min = float(controller_cfg.get("vx_min", -0.1))
        self.vx_max = float(controller_cfg.get("vx_max", 1.0))
        self.wz_min = float(controller_cfg.get("wz_min", -1.4))
        self.wz_max = float(controller_cfg.get("wz_max", 1.4))
        self.k_vx = float(controller_cfg.get("k_vx", 0.7))
        self.k_wz = float(controller_cfg.get("k_wz", 2.0))
        self.goal_tolerance = float(controller_cfg.get("goal_tolerance", 0.4))
        self.robot_goal_clearance_radius = float(controller_cfg.get("robot_goal_clearance_radius", 0.18))
        self.target_stop_extra_margin = float(controller_cfg.get("target_stop_extra_margin", 0.05))
        self.goal_stop_buffer = float(controller_cfg.get("goal_stop_buffer", 0.10))
        self.goal_stop_heading = math.radians(float(controller_cfg.get("goal_stop_heading_deg", 18.0)))
        self.near_goal_slow_radius = float(
            controller_cfg.get("near_goal_slow_radius", max(self.goal_tolerance + 0.25, self.goal_tolerance * 2.0))
        )
        self.near_goal_heading_gain = float(controller_cfg.get("near_goal_heading_gain", 0.65))
        self.turn_in_place_angle = math.radians(float(controller_cfg.get("turn_in_place_angle_deg", 65.0)))
        self.slow_angle = math.radians(float(controller_cfg.get("slow_angle_deg", 22.0)))
        self.min_turning_wz = float(controller_cfg.get("min_turning_wz", 0.2))
        self.vx_acc_limit = float(controller_cfg.get("vx_acc_limit", 0.8))
        self.wz_acc_limit = float(controller_cfg.get("wz_acc_limit", 2.5))
        self.vx_dec_limit = float(controller_cfg.get("vx_dec_limit", max(self.vx_acc_limit * 2.0, 1.6)))
        self.wz_dec_limit = float(controller_cfg.get("wz_dec_limit", max(self.wz_acc_limit * 2.0, 4.0)))
        self.startup_heading_threshold = math.radians(float(controller_cfg.get("startup_heading_threshold_deg", 8.0)))
        self.startup_wz_lead_steps = max(0, int(controller_cfg.get("startup_wz_lead_steps", 2)))
        self.startup_vx_hold_steps = max(0, int(controller_cfg.get("startup_vx_hold_steps", 1)))
        self.startup_vx_cap = float(controller_cfg.get("startup_vx_cap", 0.08))
        self.startup_min_wz = float(controller_cfg.get("startup_min_wz", min(self.wz_max, 0.30)))
        self.prev_vx = 0.0
        self.prev_wz = 0.0
        self.policy_step_index = 0

    def reset(self) -> None:
        self.prev_vx = 0.0
        self.prev_wz = 0.0
        self.policy_step_index = 0

    def compute_command(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        debug_state = observation.get("debug_state") or {}
        task = observation.get("task") or {}
        goal_distance = float(debug_state.get("goal_distance", 0.0))
        goal_clearance = max(0.0, goal_distance - self._target_stop_distance(task))
        goal_heading = wrap_to_pi(float(debug_state.get("goal_heading", 0.0)))
        yaw_rate = float(debug_state.get("yaw_rate", 0.0))

        if goal_clearance <= self.goal_tolerance:
            target_vx = 0.0
            target_wz = 0.0
        else:
            target_wz = self.k_wz * goal_heading - 0.1 * yaw_rate
            heading_abs = abs(goal_heading)
            if heading_abs >= self.turn_in_place_angle:
                target_vx = 0.0
            else:
                heading_scale = max(0.0, math.cos(heading_abs))
                if heading_abs > self.slow_angle:
                    heading_scale *= 0.5
                target_vx = self.k_vx * goal_clearance * max(0.08, heading_scale)
                target_vx = min(self.vx_max, target_vx)
                if target_vx < self.vx_min and goal_clearance > self.goal_tolerance:
                    target_vx = self.vx_min
            if abs(target_wz) > 1.0e-6 and heading_abs > self.slow_angle:
                target_wz = math.copysign(max(abs(target_wz), self.min_turning_wz), target_wz)

            stop_radius = self.goal_tolerance + self.goal_stop_buffer
            if goal_clearance <= stop_radius:
                if heading_abs <= self.goal_stop_heading:
                    target_vx = 0.0
                    target_wz = 0.0
                else:
                    target_vx = 0.0
                    target_wz *= min(1.0, heading_abs / max(self.goal_stop_heading, 1.0e-6))

            if goal_clearance < self.near_goal_slow_radius:
                distance_span = max(1.0e-6, self.near_goal_slow_radius - self.goal_tolerance)
                near_goal_ratio = max(0.0, min(1.0, (goal_clearance - self.goal_tolerance) / distance_span))
                target_vx *= near_goal_ratio * near_goal_ratio
                target_wz *= max(near_goal_ratio, self.near_goal_heading_gain)
                safe_distance = max(0.0, goal_clearance - stop_radius)
                distance_safe_vx = safe_distance / max(self.control_dt, 1.0e-6)
                target_vx = min(target_vx, distance_safe_vx)

            target_vx, target_wz = self._apply_startup_stagger(
                target_vx=target_vx,
                target_wz=target_wz,
                goal_clearance=goal_clearance,
                goal_heading=goal_heading,
            )

        smooth_vx = self._slew(self.prev_vx, target_vx, self.vx_acc_limit, self.vx_dec_limit)
        smooth_wz = self._slew(self.prev_wz, target_wz, self.wz_acc_limit, self.wz_dec_limit)
        raw_cmd_train = clip_train_action(
            [smooth_vx, smooth_wz],
            vx_min=self.vx_min,
            vx_max=self.vx_max,
            wz_min=self.wz_min,
            wz_max=self.wz_max,
        )
        self.prev_vx, self.prev_wz = raw_cmd_train
        self.policy_step_index += 1
        return {
            "raw_cmd_full": raw_full_from_train_action(raw_cmd_train, default_vy=0.0),
            "raw_cmd_train": raw_cmd_train,
            "expert_debug": {
                "target_vx": float(target_vx),
                "target_wz": float(target_wz),
                "goal_distance": goal_distance,
                "goal_clearance": goal_clearance,
                "goal_heading": goal_heading,
                "near_goal_slow_radius": float(self.near_goal_slow_radius),
                "startup_step_index": int(self.policy_step_index),
                "turn_bucket": str(task.get("turn_bucket", "")),
            },
        }

    def _target_stop_distance(self, task: Mapping[str, Any]) -> float:
        scene_targets = list(task.get("scene_targets") or [])
        active_target_id = str(task.get("active_target_id", ""))
        active_shape = str(task.get("target_shape", "")).strip().lower()
        for target in scene_targets:
            if str(target.get("target_id", "")) != active_target_id:
                continue
            return compute_target_stop_distance(
                size=list(target.get("size") or []),
                shape=active_shape,
                robot_goal_clearance_radius=self.robot_goal_clearance_radius,
                target_stop_extra_margin=self.target_stop_extra_margin,
                controller_cfg=self.controller_cfg,
            )
        return self.robot_goal_clearance_radius + self.target_stop_extra_margin

    def _apply_startup_stagger(
        self,
        *,
        target_vx: float,
        target_wz: float,
        goal_clearance: float,
        goal_heading: float,
    ) -> tuple[float, float]:
        if goal_clearance <= self.goal_tolerance:
            return target_vx, target_wz
        if self.policy_step_index >= self.startup_wz_lead_steps:
            return target_vx, target_wz

        heading_abs = abs(goal_heading)
        if heading_abs < self.startup_heading_threshold:
            return target_vx, target_wz

        signed_min_wz = math.copysign(self.startup_min_wz, target_wz if abs(target_wz) > 1.0e-6 else goal_heading)
        target_wz = math.copysign(max(abs(target_wz), abs(signed_min_wz)), signed_min_wz)
        if self.policy_step_index < self.startup_vx_hold_steps:
            return 0.0, target_wz
        return min(target_vx, self.startup_vx_cap), target_wz

    def _slew(self, previous: float, target: float, acc_limit: float, dec_limit: float) -> float:
        max_increase = float(acc_limit) * self.control_dt
        max_decrease = float(dec_limit) * self.control_dt
        if target > previous + max_increase:
            return previous + max_increase
        if target < previous - max_decrease:
            return previous - max_decrease
        return float(target)


HeuristicNavController = PrivilegedNavExpert
