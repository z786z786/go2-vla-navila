from __future__ import annotations

import math
import random
from typing import Any, Mapping

import isaaclab.sim as sim_utils
import torch
from isaaclab.sim import SimulationContext

from collectors.sim_go2.backends.motion_backends import build_motion_backend
from collectors.sim_go2.envs.scene_builder import Go2SceneHandles, build_scene
from collectors.sim_go2.envs.task_generator import GoalNavigationTaskGenerator, TaskSpec
from collectors.sim_go2.utils.state_utils import build_debug_nav_state, extract_train_state_from_debug, quat_wxyz_to_yaw
from collectors.sim_go2.utils.target_stop import compute_target_stop_distance
from collectors.sim_go2.utils.visibility import estimate_target_visibility


class Go2NavCollectionEnv:
    """Standalone Isaac Sim goal-navigation collection environment for Go2."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.runtime_cfg = config.get("runtime", {})
        self.sim_cfg = config.get("sim", {})
        self.scene_cfg = config.get("scene", {})
        self.robot_cfg = config.get("robot", {})
        self.task_cfg = config.get("task", {})
        self.data_cfg = config.get("data", {})
        self.controller_cfg = config.get("controller", {})
        self.camera_cfg = config.get("camera", {})
        self.dr_cfg = config.get("domain_randomization", {})

        self.seed = int(self.runtime_cfg.get("seed", 0))
        self.rng = random.Random(self.seed)
        self.physics_dt = float(self.sim_cfg.get("physics_dt", 0.01))
        self.control_dt = float(self.sim_cfg.get("control_dt", 0.05))
        self.control_decimation = max(1, int(round(self.control_dt / self.physics_dt)))
        self.control_dt = self.control_decimation * self.physics_dt
        self.max_steps_per_episode = int(self.runtime_cfg.get("max_steps_per_episode", 240))
        self.motion_backend = str(self.robot_cfg.get("motion_backend", "root_kinematic"))
        self.collision_enabled = bool(self.robot_cfg.get("collision_enabled", True))
        self.collision_radius = float(self.robot_cfg.get("collision_radius", 0.32))
        self.target_collision_margin = float(self.robot_cfg.get("target_collision_margin", 0.08))
        self.static_prop_collision_margin = float(self.robot_cfg.get("static_prop_collision_margin", 0.04))

        sim_context_cfg = sim_utils.SimulationCfg(
            dt=self.physics_dt,
            device=str(self.sim_cfg.get("device", "cuda:0")),
        )
        self.sim = SimulationContext(sim_context_cfg)
        self.sim.set_camera_view(
            eye=self.scene_cfg.get("viewer_eye", [5.5, 5.5, 4.0]),
            target=self.scene_cfg.get("viewer_lookat", [0.0, 0.0, 0.3]),
        )

        self.handles: Go2SceneHandles = build_scene(self.scene_cfg, self.robot_cfg, self.camera_cfg, self.dr_cfg, self.control_dt, self.rng)
        self.robot = self.handles.robot
        self.front_camera = self.handles.front_camera
        self.task_generator = GoalNavigationTaskGenerator(
            self.task_cfg,
            self.robot_cfg,
            self.scene_cfg,
            self.camera_cfg,
            self.controller_cfg,
            self.control_dt,
            self.rng,
        )

        self.sim.reset()
        self.handles.scene.reset()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.backend = build_motion_backend(
            self.motion_backend,
            robot=self.robot,
            scene=self.handles.scene,
            sim=self.sim,
            physics_dt=self.physics_dt,
            control_decimation=self.control_decimation,
            robot_cfg=self.robot_cfg,
            controller_cfg=self.controller_cfg,
            default_joint_pos=self.default_joint_pos,
            default_joint_vel=self.default_joint_vel,
        )
        self.goal_tolerance = float(self.controller_cfg.get("goal_tolerance", 0.4))
        self.robot_goal_clearance_radius = float(self.controller_cfg.get("robot_goal_clearance_radius", 0.18))
        self.target_stop_extra_margin = float(self.controller_cfg.get("target_stop_extra_margin", 0.05))
        self.success_acceptance_margin = float(self.controller_cfg.get("success_acceptance_margin", 0.0))
        self.success_hold_steps = max(1, int(self.controller_cfg.get("success_hold_steps", 3)))
        self.success_stop_vx_threshold = float(self.controller_cfg.get("success_stop_vx_threshold", 0.05))
        self.success_stop_wz_threshold = float(self.controller_cfg.get("success_stop_wz_threshold", 0.12))
        self.navigation_region = self.scene_cfg.get("navigation_region")
        self.forbidden_zones = list(self.scene_cfg.get("forbidden_zones", []))
        self._warmup_steps = int(self.sim_cfg.get("warmup_steps_after_reset", 3))
        self._post_reset_settle_steps = int(self.sim_cfg.get("post_reset_settle_steps", 6))
        self._stabilization_max_steps = int(self.sim_cfg.get("stabilization_max_steps", 48))
        self._stabilization_required_frames = int(self.sim_cfg.get("stabilization_required_frames", 3))
        self._stabilization_max_frame_delta = float(self.sim_cfg.get("stabilization_max_frame_delta", 0.02))
        self._stabilization_min_mean = float(self.sim_cfg.get("stabilization_min_mean", 0.05))
        self._stabilization_min_std = float(self.sim_cfg.get("stabilization_min_std", 0.02))

        self.episode_index = 0
        self.step_index = 0
        self.episode_id = ""
        self.current_task: TaskSpec | None = None
        self._success_hold_counter = 0

    def reset(self) -> dict[str, Any]:
        self.episode_index += 1
        self.episode_id = f"ep_{self.episode_index:06d}"
        self.step_index = 0
        self._success_hold_counter = 0
        self.current_task = self.task_generator.sample()
        self.handles.scene.reset()
        self._write_scene_targets(self.current_task.scene_targets)
        self.backend.reset(
            x=float(self.current_task.robot_position[0]),
            y=float(self.current_task.robot_position[1]),
            yaw=float(self.current_task.robot_yaw),
        )
        self.handles.scene.write_data_to_sim()
        self.handles.scene.update(0.0)
        self.backend.warmup(self._warmup_steps)
        return self._stabilize_after_reset()

    def step(self, action_train: list[float] | tuple[float, float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        applied_action = self.backend.apply_action(action_train)
        self.step_index += 1
        observation = self.get_observation()
        debug_state = observation["debug_state"]
        goal_distance = float(debug_state["goal_distance"])
        goal_clearance = max(0.0, goal_distance - self._target_stop_distance())
        stop_ready = (
            abs(float(debug_state.get("vx", 0.0))) <= self.success_stop_vx_threshold
            and abs(float(debug_state.get("yaw_rate", 0.0))) <= self.success_stop_wz_threshold
        )
        success_candidate = goal_clearance <= (self.goal_tolerance + self.success_acceptance_margin) and stop_ready
        if success_candidate:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0
        success = self._success_hold_counter >= self.success_hold_steps
        timeout = self.step_index >= self.max_steps_per_episode
        out_of_bounds = not self._point_in_navigation_region(float(debug_state["robot_x"]), float(debug_state["robot_y"]))
        entered_forbidden_zone = self._point_in_forbidden_zone(float(debug_state["robot_x"]), float(debug_state["robot_y"]))
        collision = self._has_collision(float(debug_state["robot_x"]), float(debug_state["robot_y"]))
        done = bool(success or timeout or out_of_bounds or entered_forbidden_zone or collision)
        info = {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "out_of_bounds": out_of_bounds,
            "entered_forbidden_zone": entered_forbidden_zone,
            "goal_distance": goal_distance,
            "goal_clearance": goal_clearance,
            "goal_heading": float(debug_state["goal_heading"]),
            "success_candidate": success_candidate,
            "success_hold_counter": self._success_hold_counter,
            "stop_ready": stop_ready,
            "applied_action": [float(applied_action[0]), float(applied_action[1])],
            "scene_kind": self.handles.scene_kind,
            "scene_asset_path": self.handles.scene_asset_path,
            "motion_backend": self.motion_backend,
            "target_visible": bool(debug_state["target_visible"]),
            "target_visible_within_3f": bool(self.current_task.target_visible_within_3f) if self.current_task is not None else False,
            "target_visible_within_10f": bool(self.current_task.target_visible_within_10f) if self.current_task is not None else False,
            "target_pixel_ratio": float(debug_state["target_pixel_ratio"]),
        }
        if self.current_task is not None:
            info.update(
                {
                    "task_id": self.current_task.task_id,
                    "active_target_id": self.current_task.active_target_id,
                    "target_label": self.current_task.target_label,
                    "target_description": self.current_task.target_description,
                    "turn_bucket": self.current_task.turn_bucket,
                    "visibility_bucket": self.current_task.visibility_bucket,
                    "scene_id": self.current_task.scene_id,
                    "scene_mode": self.current_task.scene_mode,
                    "layout_id": self.current_task.layout_id,
                    "layout_template_id": self.current_task.layout_template_id,
                    "layout_group_id": self.current_task.layout_group_id,
                    "contrast_group_id": self.current_task.contrast_group_id,
                    "contrast_variant": self.current_task.contrast_variant,
                }
            )
        reward = -goal_distance
        return observation, reward, done, info

    def get_observation(self) -> dict[str, Any]:
        task = self.current_task
        if task is None:
            raise RuntimeError("Environment must be reset before calling get_observation().")
        debug_state = self.get_debug_state()
        depth_image = self._get_depth_image()
        return {
            "image": self.front_camera.data.output["rgb"][0],
            "depth_image": depth_image,
            "instruction": task.instruction,
            "state": extract_train_state_from_debug(debug_state),
            "debug_state": debug_state,
            "task": {
                "task_id": task.task_id,
                "instruction_template_id": task.instruction_template_id,
                "active_target_id": task.active_target_id,
                "target_class": task.target_class,
                "target_color": task.target_color,
                "target_shape": task.target_shape,
                "target_label": task.target_label,
                "target_description": task.target_description,
                "scene_id": task.scene_id,
                "scene_kind": task.scene_kind,
                "scene_mode": task.scene_mode,
                "turn_bucket": task.turn_bucket,
                "visibility_bucket": task.visibility_bucket,
                "layout_id": task.layout_id,
                "layout_template_id": task.layout_template_id,
                "layout_group_id": task.layout_group_id,
                "split_hint": task.split_hint,
                "contrast_group_id": task.contrast_group_id,
                "contrast_variant": task.contrast_variant,
                "target_visible_first_frame": task.target_visible_first_frame,
                "target_visible_within_3f": task.target_visible_within_3f,
                "target_visible_within_10f": task.target_visible_within_10f,
                "target_pixel_ratio_first": task.target_pixel_ratio_first,
                "goal_position": [float(v) for v in task.goal_position],
                "robot_init_position": [float(v) for v in task.robot_position],
                "robot_init_yaw": float(task.robot_yaw),
                "scene_targets": list(task.scene_targets),
            },
        }

    def get_debug_state(self) -> dict[str, float]:
        robot_pos = self.robot.data.root_pos_w[0].detach().cpu().tolist()
        robot_quat = self.robot.data.root_quat_w[0].detach().cpu().tolist()
        if self.current_task is None:
            raise RuntimeError("Environment must be reset before calling get_debug_state().")
        active_target = self.handles.target(self.current_task.active_target_id)
        goal_pos = active_target.data.root_pos_w[0].detach().cpu().tolist()
        base_lin_vel_b = self.robot.data.root_lin_vel_b[0].detach().cpu().tolist()
        base_ang_vel_b = self.robot.data.root_ang_vel_b[0].detach().cpu().tolist()
        robot_yaw = quat_wxyz_to_yaw(robot_quat)
        debug_state = build_debug_nav_state(
            robot_position=robot_pos,
            robot_yaw=robot_yaw,
            base_lin_vel_b=base_lin_vel_b,
            base_ang_vel_b=base_ang_vel_b,
            goal_position=goal_pos,
        )
        visibility_cfg = self.task_cfg.get("visibility_probe", {})
        visibility = estimate_target_visibility(
            robot_position=(float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])),
            robot_yaw=robot_yaw,
            goal_position=(float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2])),
            camera_cfg=self.camera_cfg,
            target_cfg=self.current_task.active_target_cfg,
            visibility_cfg=visibility_cfg,
        )
        debug_state["target_visible"] = bool(visibility["visible"])
        debug_state["target_pixel_ratio"] = float(visibility["pixel_ratio"])
        debug_state["target_heading_camera_deg"] = float(visibility["heading_deg"])
        return debug_state

    def _get_depth_image(self):
        output = getattr(self.front_camera.data, "output", {}) or {}
        for key in ("distance_to_image_plane", "distance_to_camera", "depth"):
            if key in output:
                value = output[key]
                if value is not None:
                    return value[0]
        return None

    def close(self) -> None:
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    def _write_scene_targets(self, scene_targets: tuple[dict[str, Any], ...]) -> None:
        for target_state in scene_targets:
            target_id = str(target_state["target_id"])
            target_asset = self.handles.target(target_id)
            goal_pose = target_asset.data.default_root_state.clone()
            position = target_state["position"]
            goal_pose[:, 0] = float(position[0])
            goal_pose[:, 1] = float(position[1])
            goal_pose[:, 2] = float(position[2])
            goal_pose[:, 3] = 1.0
            goal_pose[:, 4] = 0.0
            goal_pose[:, 5] = 0.0
            goal_pose[:, 6] = 0.0
            goal_velocity = target_asset.data.default_root_state[:, 7:].clone()
            goal_velocity[:, :] = 0.0
            target_asset.write_root_pose_to_sim(goal_pose[:, :7])
            target_asset.write_root_velocity_to_sim(goal_velocity)

    def _stabilize_after_reset(self) -> dict[str, Any]:
        if self._post_reset_settle_steps > 0:
            self.backend.warmup(self._post_reset_settle_steps)

        previous_frame: torch.Tensor | None = None
        stable_frames = 0
        observation = self.get_observation()
        for _ in range(max(1, self._stabilization_max_steps)):
            current_frame = self._normalized_rgb_tensor(observation["image"])
            if self._frame_is_visually_ready(current_frame, previous_frame):
                stable_frames += 1
                if stable_frames >= self._stabilization_required_frames:
                    return observation
            else:
                stable_frames = 0
            previous_frame = current_frame
            self.backend.warmup(1)
            observation = self.get_observation()
        return observation

    @staticmethod
    def _normalized_rgb_tensor(image: Any) -> torch.Tensor:
        tensor = torch.as_tensor(image).detach().float()
        if tensor.numel() == 0:
            return tensor.reshape(0)
        if tensor.ndim >= 3 and tensor.shape[-1] in {3, 4}:
            tensor = tensor[..., :3]
        if float(tensor.max().item()) > 1.5:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def _frame_is_visually_ready(self, current: torch.Tensor, previous: torch.Tensor | None) -> bool:
        if current.numel() == 0 or not torch.isfinite(current).all():
            return False
        mean = float(current.mean().item())
        std = float(current.std().item())
        if mean < self._stabilization_min_mean or std < self._stabilization_min_std:
            return False
        if previous is None or previous.shape != current.shape:
            return True
        delta = float((current - previous).abs().mean().item())
        return delta <= self._stabilization_max_frame_delta

    def _point_in_navigation_region(self, x: float, y: float) -> bool:
        if not self.navigation_region:
            return True
        x_range = self.navigation_region.get("x", (-math.inf, math.inf))
        y_range = self.navigation_region.get("y", (-math.inf, math.inf))
        return float(x_range[0]) <= x <= float(x_range[1]) and float(y_range[0]) <= y <= float(y_range[1])

    def _point_in_forbidden_zone(self, x: float, y: float) -> bool:
        for zone in self.forbidden_zones:
            x_range = zone.get("x", ())
            y_range = zone.get("y", ())
            if len(x_range) != 2 or len(y_range) != 2:
                continue
            if float(x_range[0]) <= x <= float(x_range[1]) and float(y_range[0]) <= y <= float(y_range[1]):
                return True
        return False

    def _has_collision(self, x: float, y: float) -> bool:
        if not self.collision_enabled:
            return False
        for target_state in getattr(self.current_task, "scene_targets", ()) or ():
            target_id = str(target_state.get("target_id", ""))
            if target_id == getattr(self.current_task, "active_target_id", ""):
                continue
            size = target_state.get("size") or ()
            position = target_state.get("position") or ()
            if len(size) >= 2 and len(position) >= 2 and self._point_hits_aabb(
                x,
                y,
                float(position[0]),
                float(position[1]),
                float(size[0]) + self.target_collision_margin,
                float(size[1]) + self.target_collision_margin,
            ):
                return True
        for prop_cfg in self.handles.static_props:
            size = prop_cfg.get("size") or ()
            position = prop_cfg.get("position") or ()
            if len(size) >= 2 and len(position) >= 2 and self._point_hits_aabb(
                x,
                y,
                float(position[0]),
                float(position[1]),
                float(size[0]) + self.static_prop_collision_margin,
                float(size[1]) + self.static_prop_collision_margin,
            ):
                return True
        return False

    def _target_stop_distance(self) -> float:
        task = self.current_task
        if task is None:
            return self.robot_goal_clearance_radius + self.target_stop_extra_margin
        return compute_target_stop_distance(
            size=list(task.active_target_cfg.get("size") or []),
            shape=str(task.target_shape),
            robot_goal_clearance_radius=self.robot_goal_clearance_radius,
            target_stop_extra_margin=self.target_stop_extra_margin,
            controller_cfg=self.controller_cfg,
        )

    def _point_hits_aabb(self, x: float, y: float, center_x: float, center_y: float, size_x: float, size_y: float) -> bool:
        half_x = max(0.0, size_x / 2.0) + self.collision_radius
        half_y = max(0.0, size_y / 2.0) + self.collision_radius
        return (center_x - half_x) <= x <= (center_x + half_x) and (center_y - half_y) <= y <= (center_y + half_y)
