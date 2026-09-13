from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

from collectors.sim_go2.utils.io import stable_hash
from collectors.sim_go2.utils.randomization import point_in_region, sample_point_in_region, sample_uniform, sample_weighted_label
from collectors.sim_go2.utils.visibility import rollout_visibility


VISIBLE_EARLY = "visible_early"
VISIBLE_AFTER_TURN = "visible_after_small_turn"
SEARCH_STYLE = "search"


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    task_id: str
    instruction_template_id: str
    active_target_id: str
    target_class: str
    target_color: str
    target_shape: str
    target_label: str
    target_description: str
    active_target_cfg: dict[str, Any]
    goal_position: tuple[float, float, float]
    scene_targets: tuple[dict[str, Any], ...]
    contrast_group_id: str
    contrast_variant: str
    robot_position: tuple[float, float, float]
    robot_yaw: float
    turn_bucket: str
    goal_distance: float
    relative_angle: float
    scene_id: str
    scene_kind: str
    scene_mode: str
    layout_template_id: str
    layout_id: str
    layout_group_id: str
    split_hint: str
    visibility_bucket: str
    target_visible_first_frame: bool
    target_visible_within_3f: bool
    target_visible_within_10f: bool
    target_pixel_ratio_first: float


class GoalNavigationTaskGenerator:
    """Sample goal-navigation tasks with optional dual-target contrast pairs."""

    def __init__(
        self,
        task_cfg: Mapping[str, Any],
        robot_cfg: Mapping[str, Any],
        scene_cfg: Mapping[str, Any],
        camera_cfg: Mapping[str, Any],
        controller_cfg: Mapping[str, Any],
        control_dt: float,
        rng: random.Random,
    ):
        self.task_cfg = task_cfg
        self.robot_cfg = robot_cfg
        self.scene_cfg = scene_cfg
        self.camera_cfg = camera_cfg
        self.controller_cfg = controller_cfg
        self.control_dt = float(control_dt)
        self.rng = rng
        self.scene_target_specs = self._scene_target_specs()
        self._pending_tasks: list[TaskSpec] = []

    def sample(self) -> TaskSpec:
        if self._pending_tasks:
            return self._pending_tasks.pop(0)

        target_catalog = self._target_catalog()
        contrast_cfg = dict(self.task_cfg.get("contrast_pairs", {}))
        contrast_enabled = bool(contrast_cfg.get("enabled", False)) and len(target_catalog) >= 2
        if contrast_enabled:
            tasks = self._sample_contrast_tasks(target_catalog, contrast_cfg)
            if tasks:
                self._pending_tasks.extend(tasks[1:])
                return tasks[0]

        task = self._sample_single_target_task(target_catalog)
        return task

    def _scene_target_specs(self) -> dict[str, dict[str, Any]]:
        configured = list(self.scene_cfg.get("targets", []))
        if configured:
            specs: dict[str, dict[str, Any]] = {}
            for index, item in enumerate(configured):
                visual_cfg = item.get("visual", {}) if isinstance(item.get("visual"), Mapping) else {}
                size = item.get("size", visual_cfg.get("size", (0.35, 0.35, 0.50)))
                color = item.get("color", visual_cfg.get("color", (0.05, 0.05, 0.05)))
                height = item.get("height", visual_cfg.get("height", 0.25))
                target_id = str(item.get("id") or item.get("target_id") or f"target_{index}")
                specs[target_id] = {
                    "target_id": target_id,
                    "target_class": str(item.get("class") or item.get("target_class") or target_id),
                    "target_color": str(item.get("color_name") or item.get("target_color") or ""),
                    "target_shape": str(item.get("shape") or item.get("target_shape") or "box"),
                    "target_label": str(item.get("label") or item.get("target_label") or target_id.replace("_", " ")),
                    "target_description": str(item.get("description") or item.get("target_description") or item.get("label") or target_id.replace("_", " ")),
                    "size": tuple(float(value) for value in size),
                    "color": tuple(float(value) for value in color),
                    "height": float(height),
                }
            return specs

        goal_box_cfg = self.scene_cfg.get("goal_box", {})
        return {
            "goal_box": {
                "target_id": "goal_box",
                "target_class": "goal_box",
                "target_color": "black",
                "target_shape": "box",
                "target_label": "goal box",
                "target_description": "goal box",
                "size": tuple(float(value) for value in goal_box_cfg.get("size", (0.35, 0.35, 0.50))),
                "color": tuple(float(value) for value in goal_box_cfg.get("color", (0.05, 0.05, 0.05))),
                "height": float(goal_box_cfg.get("height", 0.25)),
            }
        }

    def _target_catalog(self) -> list[dict[str, Any]]:
        configured = list(self.task_cfg.get("target_catalog", []))
        if not configured:
            configured = [
                {
                    "id": str(self.task_cfg.get("target_id") or next(iter(self.scene_target_specs.keys()), "goal_box")),
                    "class": str(self.task_cfg.get("target_class", "goal_box")),
                    "color": str(self.task_cfg.get("target_color", "black")),
                    "shape": str(self.task_cfg.get("target_shape", "box")),
                    "label": str(self.task_cfg.get("goal_label", "goal box")),
                    "description": str(self.task_cfg.get("goal_description", self.task_cfg.get("goal_label", "goal box"))),
                }
            ]

        catalog: list[dict[str, Any]] = []
        for index, item in enumerate(configured):
            target_id = str(
                item.get("id")
                or item.get("target_id")
                or item.get("scene_target_id")
                or next(iter(self.scene_target_specs.keys()), f"target_{index}")
            )
            scene_spec = dict(self.scene_target_specs.get(target_id) or {})
            catalog.append(
                {
                    "target_id": target_id,
                    "target_class": str(item.get("class") or item.get("target_class") or scene_spec.get("target_class") or target_id),
                    "target_color": str(item.get("color") or item.get("target_color") or scene_spec.get("target_color") or ""),
                    "target_shape": str(item.get("shape") or item.get("target_shape") or scene_spec.get("target_shape") or "box"),
                    "target_label": str(item.get("label") or item.get("target_label") or scene_spec.get("target_label") or target_id.replace("_", " ")),
                    "target_description": str(item.get("description") or item.get("target_description") or scene_spec.get("target_description") or item.get("label") or target_id.replace("_", " ")),
                    "size": tuple(scene_spec.get("size") or (0.35, 0.35, 0.50)),
                    "color_rgb": tuple(scene_spec.get("color") or (0.05, 0.05, 0.05)),
                    "height": float(scene_spec.get("height", 0.25)),
                }
            )
        return catalog

    def _sample_single_target_task(self, target_catalog: list[dict[str, Any]]) -> TaskSpec:
        target_cfg = dict(target_catalog[0])
        sampled = self._sample_layout(target_catalog=[target_cfg], validate_target_ids=[target_cfg["target_id"]])
        if sampled is None:
            return self._fallback_task(target_cfg)
        return self._build_task(
            target_cfg=target_cfg,
            instruction_template=sampled["instruction_template"],
            shared=sampled,
        )

    def _sample_contrast_tasks(self, target_catalog: list[dict[str, Any]], contrast_cfg: Mapping[str, Any]) -> list[TaskSpec]:
        requested_ids = list(contrast_cfg.get("target_ids", []))
        contrast_targets = [
            dict(target)
            for target in target_catalog
            if not requested_ids or target["target_id"] in requested_ids
        ]
        if len(contrast_targets) < 2:
            return []

        sampled = self._sample_layout(
            target_catalog=contrast_targets,
            validate_target_ids=[target["target_id"] for target in contrast_targets],
        )
        if sampled is None:
            # Some turn-pilot layouts cannot satisfy a shared visibility contract
            # for every contrast target from one identical rollout start state.
            # Rather than silently degrading to the hard-coded fallback task,
            # retry with per-target validation so online evaluation still uses a
            # semantically valid active-target rollout.
            per_target_tasks: list[TaskSpec] = []
            for target_cfg in contrast_targets:
                target_sampled = self._sample_layout(
                    target_catalog=contrast_targets,
                    validate_target_ids=[target_cfg["target_id"]],
                )
                if target_sampled is None:
                    per_target_tasks.append(self._fallback_task(target_cfg))
                    continue
                per_target_tasks.append(
                    self._build_task(
                        target_cfg=target_cfg,
                        instruction_template=target_sampled["instruction_template"],
                        shared=target_sampled,
                    )
                )
            return per_target_tasks

        return [
            self._build_task(
                target_cfg=target_cfg,
                instruction_template=sampled["instruction_template"],
                shared=sampled,
            )
            for target_cfg in contrast_targets
        ]

    def _sample_layout(
        self,
        *,
        target_catalog: list[dict[str, Any]],
        validate_target_ids: list[str],
    ) -> dict[str, Any] | None:
        instruction_templates = list(self.task_cfg.get("instruction_templates", [])) or [
            {"id": "go_to_target", "text": "go to the {target_label}"}
        ]
        scene_kind = str(self.scene_cfg.get("kind", "plane"))
        scene_mode = str(self.task_cfg.get("scene_mode", scene_kind))
        scene_id = str(self.scene_cfg.get("scene_id", scene_mode))
        navigation_region = self.task_cfg.get("workspace", self.scene_cfg.get("navigation_region", {}))
        forbidden_zones = list(self.scene_cfg.get("forbidden_zones", []))
        min_clearance = float(self.scene_cfg.get("min_obstacle_clearance", 0.0))
        default_spawn_region = self.scene_cfg.get("spawn_region", {})
        default_goal_region = self.scene_cfg.get("goal_region", {})
        default_distance_range = self.task_cfg.get("goal_distance_range", (1.0, 4.0))
        init_range = self.robot_cfg.get("init_pose_range", {})
        max_attempts = int(self.task_cfg.get("max_resample_attempts", 192))
        visibility_cfg = self.task_cfg.get("visibility_curriculum", {})
        probe_cfg = self.task_cfg.get("visibility_probe", {})
        layout_templates = list(self.task_cfg.get("layout_templates", []))
        instruction_template = dict(self.rng.choice(instruction_templates))
        instruction_template_id = str(instruction_template.get("id", "template_0"))

        for _ in range(max_attempts):
            visibility_bucket = self._sample_visibility_bucket(visibility_cfg)
            candidate_templates = self._candidate_templates(layout_templates, visibility_bucket)
            if candidate_templates:
                layout_template = dict(self.rng.choice(candidate_templates))
            else:
                layout_template = {
                    "id": f"default_{visibility_bucket}",
                    "spawn_region": default_spawn_region,
                    "distance_range": default_distance_range,
                    "target_slots": {},
                }
            spawn_region = layout_template.get("spawn_region", default_spawn_region)
            if spawn_region:
                robot_x, robot_y = sample_point_in_region(self.rng, spawn_region)
            else:
                robot_x = sample_uniform(self.rng, init_range.get("x", (-2.0, 2.0)))
                robot_y = sample_uniform(self.rng, init_range.get("y", (-2.0, 2.0)))
            if self._point_hits_forbidden_zone(robot_x, robot_y, forbidden_zones, min_clearance):
                continue

            scene_targets = self._sample_scene_targets(
                robot_x=robot_x,
                robot_y=robot_y,
                target_catalog=target_catalog,
                layout_template=layout_template,
                default_goal_region=default_goal_region,
                default_distance_range=default_distance_range,
                forbidden_zones=forbidden_zones,
                min_clearance=min_clearance,
            )
            if scene_targets is None:
                continue

            reference_target_ids = list(layout_template.get("visibility_reference_target_ids", [])) or list(validate_target_ids)
            reference_positions = [
                scene_targets[target_id]["goal_position"]
                for target_id in reference_target_ids
                if target_id in scene_targets
            ]
            if not reference_positions:
                continue
            reference_x = sum(position[0] for position in reference_positions) / len(reference_positions)
            reference_y = sum(position[1] for position in reference_positions) / len(reference_positions)
            turn_bucket, relative_angle, robot_yaw = self._sample_robot_yaw(
                robot_x=robot_x,
                robot_y=robot_y,
                goal_x=reference_x,
                goal_y=reference_y,
                visibility_bucket=visibility_bucket,
                layout_template=layout_template,
            )
            visibility_by_target: dict[str, dict[str, float | bool]] = {}
            for target_id in validate_target_ids:
                target_state = scene_targets.get(target_id)
                if target_state is None:
                    visibility_by_target = {}
                    break
                visibility_trace = rollout_visibility(
                    robot_position=(robot_x, robot_y, float(self.robot_cfg.get("base_height", 0.4))),
                    robot_yaw=robot_yaw,
                    goal_position=target_state["goal_position"],
                    camera_cfg=self.camera_cfg,
                    controller_cfg=self.controller_cfg,
                    target_cfg={
                        "size": target_state["size"],
                        "height": target_state["height"],
                    },
                    visibility_cfg=probe_cfg,
                    horizon_steps=int(probe_cfg.get("horizon_steps", 10)),
                    control_dt=self.control_dt,
                )
                visibility_summary = self._visibility_summary(visibility_trace)
                if not self._visibility_bucket_matches(visibility_bucket, visibility_summary):
                    visibility_by_target = {}
                    break
                visibility_by_target[target_id] = visibility_summary
            if not visibility_by_target:
                continue

            spawn_bin = self._spatial_bin(robot_x, robot_y, spawn_region or navigation_region)
            target_bins = [
                f"{target_id}:{self._spatial_bin(scene_targets[target_id]['goal_position'][0], scene_targets[target_id]['goal_position'][1], scene_targets[target_id]['region'])}"
                for target_id in sorted(scene_targets)
            ]
            layout_template_id = str(layout_template.get("id", f"default_{visibility_bucket}"))
            layout_seed = "|".join(
                [
                    scene_id,
                    scene_mode,
                    layout_template_id,
                    visibility_bucket,
                    turn_bucket,
                    spawn_bin,
                    *target_bins,
                    instruction_template_id,
                ]
            )
            layout_group_id = stable_hash(layout_seed, digits=16)
            layout_id = stable_hash(layout_seed + f"|{robot_yaw:.4f}", digits=16)
            contrast_group_id = layout_group_id
            return {
                "instruction_template": instruction_template,
                "instruction_template_id": instruction_template_id,
                "scene_id": scene_id,
                "scene_kind": scene_kind,
                "scene_mode": scene_mode,
                "layout_template_id": layout_template_id,
                "layout_id": layout_id,
                "layout_group_id": layout_group_id,
                "contrast_group_id": contrast_group_id,
                "visibility_bucket": visibility_bucket,
                "turn_bucket": turn_bucket,
                "reference_relative_angle": relative_angle,
                "robot_position": (robot_x, robot_y, float(self.robot_cfg.get("base_height", 0.4))),
                "robot_yaw": robot_yaw,
                "scene_targets": scene_targets,
                "visibility_by_target": visibility_by_target,
                "split_hint": str(self.task_cfg.get("default_split_hint", "train_candidate")),
            }
        return None

    def _sample_scene_targets(
        self,
        *,
        robot_x: float,
        robot_y: float,
        target_catalog: list[dict[str, Any]],
        layout_template: Mapping[str, Any],
        default_goal_region: Mapping[str, Any],
        default_distance_range: tuple[float, float] | list[float],
        forbidden_zones: list[Mapping[str, Any]],
        min_clearance: float,
    ) -> dict[str, dict[str, Any]] | None:
        target_slots = layout_template.get("target_slots", {}) if isinstance(layout_template.get("target_slots"), Mapping) else {}
        min_target_separation = float(layout_template.get("min_target_separation", self.task_cfg.get("min_target_separation", 0.75)))
        sampled: dict[str, dict[str, Any]] = {}
        for target_cfg in target_catalog:
            target_id = target_cfg["target_id"]
            slot_cfg = target_slots.get(target_id)
            if not isinstance(slot_cfg, Mapping):
                slot_cfg = {}
            region = slot_cfg.get("region") or slot_cfg.get("goal_region") or layout_template.get("goal_region", default_goal_region)
            distance_range = slot_cfg.get("distance_range") or layout_template.get("distance_range", default_distance_range)
            if not region or not distance_range:
                return None
            sample = self._sample_target_point(
                robot_x=robot_x,
                robot_y=robot_y,
                goal_region=region,
                goal_distance_range=distance_range,
                forbidden_zones=forbidden_zones,
                min_clearance=min_clearance,
                existing_positions=[item["goal_position"] for item in sampled.values()],
                min_target_separation=min_target_separation,
            )
            if sample is None:
                return None
            goal_x, goal_y, goal_distance = sample
            sampled[target_id] = {
                "target_id": target_id,
                "target_class": target_cfg["target_class"],
                "target_color": target_cfg["target_color"],
                "target_shape": target_cfg["target_shape"],
                "target_label": target_cfg["target_label"],
                "target_description": target_cfg["target_description"],
                "goal_position": (goal_x, goal_y, float(target_cfg["height"])),
                "goal_distance": goal_distance,
                "size": tuple(target_cfg["size"]),
                "color": tuple(target_cfg["color_rgb"]),
                "height": float(target_cfg["height"]),
                "region": region,
            }
        return sampled

    def _sample_target_point(
        self,
        *,
        robot_x: float,
        robot_y: float,
        goal_region: Mapping[str, Any],
        goal_distance_range: tuple[float, float] | list[float],
        forbidden_zones: list[Mapping[str, Any]],
        min_clearance: float,
        existing_positions: list[tuple[float, float, float]],
        min_target_separation: float,
    ) -> tuple[float, float, float] | None:
        min_distance = float(goal_distance_range[0])
        max_distance = float(goal_distance_range[1])
        attempts = int(self.task_cfg.get("goal_region_resample_attempts", 256))
        for _ in range(attempts):
            goal_x, goal_y = sample_point_in_region(self.rng, goal_region)
            if self._point_hits_forbidden_zone(goal_x, goal_y, forbidden_zones, min_clearance):
                continue
            goal_distance = math.hypot(goal_x - robot_x, goal_y - robot_y)
            if not (min_distance <= goal_distance <= max_distance):
                continue
            if any(math.hypot(goal_x - existing[0], goal_y - existing[1]) < min_target_separation for existing in existing_positions):
                continue
            return goal_x, goal_y, goal_distance
        return None

    def _build_task(
        self,
        *,
        target_cfg: Mapping[str, Any],
        instruction_template: Mapping[str, Any],
        shared: Mapping[str, Any],
    ) -> TaskSpec:
        target_state = dict(shared["scene_targets"][target_cfg["target_id"]])
        goal_x, goal_y, goal_z = target_state["goal_position"]
        robot_x, robot_y, robot_z = shared["robot_position"]
        goal_heading = math.atan2(goal_y - robot_y, goal_x - robot_x)
        relative_angle = ((goal_heading - float(shared["robot_yaw"]) + math.pi) % (2.0 * math.pi)) - math.pi
        instruction = str(instruction_template.get("text", "go to the {target_label}")).format(
            target=target_cfg["target_label"],
            target_label=target_cfg["target_label"],
            target_class=target_cfg["target_class"],
            target_color=target_cfg["target_color"],
            target_shape=target_cfg["target_shape"],
        )
        scene_targets_payload = tuple(
            {
                "target_id": state["target_id"],
                "target_class": state["target_class"],
                "target_color": state["target_color"],
                "target_shape": state["target_shape"],
                "target_label": state["target_label"],
                "target_description": state["target_description"],
                "position": [float(value) for value in state["goal_position"]],
                "size": [float(value) for value in state["size"]],
                "color": [float(value) for value in state["color"]],
                "height": float(state["height"]),
            }
            for _target_id, state in sorted(shared["scene_targets"].items())
        )
        visibility_summary = dict(shared["visibility_by_target"][target_cfg["target_id"]])
        return TaskSpec(
            instruction=instruction,
            task_id=str(self.task_cfg.get("task_id", "goal_navigation")),
            instruction_template_id=str(shared["instruction_template_id"]),
            active_target_id=str(target_cfg["target_id"]),
            target_class=str(target_cfg["target_class"]),
            target_color=str(target_cfg["target_color"]),
            target_shape=str(target_cfg["target_shape"]),
            target_label=str(target_cfg["target_label"]),
            target_description=str(target_cfg["target_description"]),
            active_target_cfg={
                "target_id": str(target_cfg["target_id"]),
                "size": [float(value) for value in target_cfg["size"]],
                "color": [float(value) for value in target_cfg["color_rgb"]],
                "height": float(target_cfg["height"]),
            },
            goal_position=(float(goal_x), float(goal_y), float(goal_z)),
            scene_targets=scene_targets_payload,
            contrast_group_id=str(shared["contrast_group_id"]),
            contrast_variant=str(target_cfg["target_id"]),
            robot_position=(float(robot_x), float(robot_y), float(robot_z)),
            robot_yaw=float(shared["robot_yaw"]),
            turn_bucket=str(shared["turn_bucket"]),
            goal_distance=float(target_state["goal_distance"]),
            relative_angle=float(relative_angle),
            scene_id=str(shared["scene_id"]),
            scene_kind=str(shared["scene_kind"]),
            scene_mode=str(shared["scene_mode"]),
            layout_template_id=str(shared["layout_template_id"]),
            layout_id=str(shared["layout_id"]),
            layout_group_id=str(shared["layout_group_id"]),
            split_hint=str(shared["split_hint"]),
            visibility_bucket=str(shared["visibility_bucket"]),
            target_visible_first_frame=bool(visibility_summary["visible_first_frame"]),
            target_visible_within_3f=bool(visibility_summary["visible_within_3f"]),
            target_visible_within_10f=bool(visibility_summary["visible_within_10f"]),
            target_pixel_ratio_first=float(visibility_summary["pixel_ratio_first"]),
        )

    def _fallback_task(self, target_cfg: Mapping[str, Any]) -> TaskSpec:
        robot_position = (0.0, -1.5, float(self.robot_cfg.get("base_height", 0.4)))
        robot_yaw = 0.0
        fallback_goal = (3.5, -1.5, float(target_cfg["height"]))
        visibility_summary = self._visibility_summary(
            rollout_visibility(
                robot_position=robot_position,
                robot_yaw=robot_yaw,
                goal_position=fallback_goal,
                camera_cfg=self.camera_cfg,
                controller_cfg=self.controller_cfg,
                target_cfg={
                    "size": tuple(target_cfg["size"]),
                    "height": float(target_cfg["height"]),
                },
                visibility_cfg=self.task_cfg.get("visibility_probe", {}),
                horizon_steps=int(self.task_cfg.get("visibility_probe", {}).get("horizon_steps", 10)),
                control_dt=self.control_dt,
            )
        )
        instruction = f"go to the {target_cfg['target_label']}"
        return TaskSpec(
            instruction=instruction,
            task_id=str(self.task_cfg.get("task_id", "goal_navigation")),
            instruction_template_id="fallback",
            active_target_id=str(target_cfg["target_id"]),
            target_class=str(target_cfg["target_class"]),
            target_color=str(target_cfg["target_color"]),
            target_shape=str(target_cfg["target_shape"]),
            target_label=str(target_cfg["target_label"]),
            target_description=str(target_cfg["target_description"]),
            active_target_cfg={
                "target_id": str(target_cfg["target_id"]),
                "size": [float(value) for value in target_cfg["size"]],
                "color": [float(value) for value in target_cfg["color_rgb"]],
                "height": float(target_cfg["height"]),
            },
            goal_position=fallback_goal,
            scene_targets=(
                {
                    "target_id": str(target_cfg["target_id"]),
                    "target_class": str(target_cfg["target_class"]),
                    "target_color": str(target_cfg["target_color"]),
                    "target_shape": str(target_cfg["target_shape"]),
                    "target_label": str(target_cfg["target_label"]),
                    "target_description": str(target_cfg["target_description"]),
                    "position": [float(v) for v in fallback_goal],
                    "size": [float(value) for value in target_cfg["size"]],
                    "color": [float(value) for value in target_cfg["color_rgb"]],
                    "height": float(target_cfg["height"]),
                },
            ),
            contrast_group_id=stable_hash(str(target_cfg["target_id"]) + "|fallback", digits=16),
            contrast_variant=str(target_cfg["target_id"]),
            robot_position=robot_position,
            robot_yaw=robot_yaw,
            turn_bucket="fallback",
            goal_distance=3.5,
            relative_angle=0.0,
            scene_id=str(self.scene_cfg.get("scene_id", "fallback_scene")),
            scene_kind=str(self.scene_cfg.get("kind", "plane")),
            scene_mode=str(self.task_cfg.get("scene_mode", self.scene_cfg.get("kind", "plane"))),
            layout_template_id="fallback",
            layout_id=stable_hash(str(target_cfg["target_id"]) + "|fallback_layout", digits=16),
            layout_group_id=stable_hash(str(target_cfg["target_id"]) + "|fallback_group", digits=16),
            split_hint=str(self.task_cfg.get("default_split_hint", "train_candidate")),
            visibility_bucket=VISIBLE_EARLY,
            target_visible_first_frame=bool(visibility_summary["visible_first_frame"]),
            target_visible_within_3f=bool(visibility_summary["visible_within_3f"]),
            target_visible_within_10f=bool(visibility_summary["visible_within_10f"]),
            target_pixel_ratio_first=float(visibility_summary["pixel_ratio_first"]),
        )

    def _sample_visibility_bucket(self, visibility_cfg: Mapping[str, Any]) -> str:
        return sample_weighted_label(
            self.rng,
            {
                VISIBLE_EARLY: float(visibility_cfg.get("visible_early_ratio", 0.75)),
                VISIBLE_AFTER_TURN: float(visibility_cfg.get("visible_after_small_turn_ratio", 0.17)),
                SEARCH_STYLE: float(visibility_cfg.get("search_ratio", 0.08)),
            },
        )

    @staticmethod
    def _candidate_templates(layout_templates: list[Mapping[str, Any]], visibility_bucket: str) -> list[Mapping[str, Any]]:
        candidates: list[Mapping[str, Any]] = []
        for template in layout_templates:
            buckets = list(template.get("visibility_buckets", []))
            if not buckets or visibility_bucket in buckets:
                candidates.append(template)
        return candidates

    def _sample_robot_yaw(
        self,
        *,
        robot_x: float,
        robot_y: float,
        goal_x: float,
        goal_y: float,
        visibility_bucket: str,
        layout_template: Mapping[str, Any],
    ) -> tuple[str, float, float]:
        goal_heading = math.atan2(goal_y - robot_y, goal_x - robot_x)
        turn_bucket, relative_angle = self._sample_turn_bucket_for_visibility(visibility_bucket)
        yaw_jitter_deg = float(layout_template.get("yaw_jitter_deg", 4.0))
        base_yaw = goal_heading - relative_angle
        robot_yaw = base_yaw + math.radians(self.rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        robot_yaw = ((robot_yaw + math.pi) % (2.0 * math.pi)) - math.pi
        return turn_bucket, relative_angle, robot_yaw

    def _sample_turn_bucket_for_visibility(self, visibility_bucket: str) -> tuple[str, float]:
        if visibility_bucket == VISIBLE_EARLY:
            bucket = sample_weighted_label(
                self.rng,
                {"straight": 0.32, "left_small": 0.28, "right_small": 0.24, "left_large": 0.10, "right_large": 0.06},
            )
            ranges = {
                "straight": (-10.0, 10.0),
                "left_small": (12.0, 26.0),
                "right_small": (-26.0, -12.0),
                "left_large": (24.0, 34.0),
                "right_large": (-34.0, -24.0),
            }
        elif visibility_bucket == VISIBLE_AFTER_TURN:
            bucket = sample_weighted_label(
                self.rng,
                {"left_large": 0.42, "right_large": 0.38, "left_small": 0.10, "right_small": 0.10},
            )
            ranges = {
                "left_small": (24.0, 34.0),
                "right_small": (-34.0, -24.0),
                "left_large": (34.0, 55.0),
                "right_large": (-55.0, -34.0),
            }
        else:
            bucket = sample_weighted_label(self.rng, {"left_large": 0.35, "right_large": 0.35, "reverse_reorient": 0.30})
            ranges = {
                "left_large": (55.0, 90.0),
                "right_large": (-90.0, -55.0),
                "reverse_reorient": (110.0, 165.0),
            }
        angle_deg = sample_uniform(self.rng, ranges[bucket])
        if bucket == "reverse_reorient" and self.rng.random() < 0.5:
            angle_deg = -angle_deg
        return bucket, math.radians(angle_deg)

    @staticmethod
    def _visibility_summary(trace: list[dict[str, float | bool]]) -> dict[str, float | bool]:
        visible_flags = [bool(item.get("visible", False)) for item in trace]
        first_window = visible_flags[:3]
        reveal_window = visible_flags[:10]
        return {
            "visible_first_frame": bool(visible_flags[0]) if visible_flags else False,
            "visible_within_3f": any(first_window),
            "visible_within_10f": any(reveal_window),
            "pixel_ratio_first": float(trace[0].get("pixel_ratio", 0.0)) if trace else 0.0,
        }

    @staticmethod
    def _visibility_bucket_matches(visibility_bucket: str, summary: Mapping[str, float | bool]) -> bool:
        if visibility_bucket == VISIBLE_EARLY:
            return bool(summary["visible_first_frame"]) or bool(summary["visible_within_3f"])
        if visibility_bucket == VISIBLE_AFTER_TURN:
            return (not bool(summary["visible_first_frame"])) and bool(summary["visible_within_10f"])
        return not bool(summary["visible_within_3f"])

    @staticmethod
    def _point_hits_forbidden_zone(x: float, y: float, forbidden_zones: list[Mapping[str, Any]], min_clearance: float) -> bool:
        for zone in forbidden_zones:
            if point_in_region(x, y, zone, margin=-float(min_clearance)):
                return True
        return False

    @staticmethod
    def _spatial_bin(x: float, y: float, region: Mapping[str, Any]) -> str:
        x_range = region.get("x", (-1.0, 1.0))
        y_range = region.get("y", (-1.0, 1.0))
        x_span = max(1.0e-6, float(x_range[1]) - float(x_range[0]))
        y_span = max(1.0e-6, float(y_range[1]) - float(y_range[0]))
        x_bin = min(2, max(0, int(3.0 * (float(x) - float(x_range[0])) / x_span)))
        y_bin = min(2, max(0, int(3.0 * (float(y) - float(y_range[0])) / y_span)))
        return f"x{x_bin}_y{y_bin}"
