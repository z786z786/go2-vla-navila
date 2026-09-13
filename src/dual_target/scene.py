"""Small, explicit DT1 scene adapter for the real Go2 flat-ground precheck.

This module deliberately keeps its scene specification CPU importable.  Isaac
objects are imported only by :func:`apply_flat_dual_target_scene`, immediately
before the caller makes the one environment instance.  The targets remain
physical cuboids in the renderer/physics world, but the old locomotion
ray-caster sees the single ground mesh only.  That is an intentional
flat-terrain compatibility choice, *not* a hidden obstacle/navigation input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal

from .layouts import GeometryGroup, GeometryValidationError, TargetSlot, Vec2


GROUND_PRIM_PATH = "/World/ground"
GROUND_ONLY_MESH_PATHS = (GROUND_PRIM_PATH,)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# This is source-controlled rather than Isaac's ``default_environment.usd``.
# The latter is absent from the audited local asset cache on neu-3070.
SELF_CONTAINED_GROUND_USD = _PROJECT_ROOT / "config/dual_target_v1/assets/dt1_flat_ground.usda"
TARGET_BOX_SIZE_M = (0.50, 0.50, 0.50)
TARGET_BOX_CENTER_Z_M = TARGET_BOX_SIZE_M[2] / 2.0
GO2_RESET_HEIGHT_M = 0.40
# These names were enumerated from the audited local Go2 USD's
# ``/go2_description`` default prim and verified against the 19 live rigid
# bodies.  ContactSensor's RigidContactView accepts one rigid body per filter
# expression; ``Robot/.*`` therefore silently created an invalid 19-body
# filter view in the first DT1 smoke and is prohibited here.
GO2_CONTACT_BODY_NAMES = (
    "base",
    "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
    "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
    "Head_upper", "Head_lower",
    "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
    "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
)
ROBOT_CONTACT_FILTER_PATHS = tuple(f"{{ENV_REGEX_NS}}/Robot/{body}" for body in GO2_CONTACT_BODY_NAMES)

Color = Literal["red", "blue"]
ColorConfiguration = Literal["A_red_B_blue", "A_blue_B_red"]


class SceneAdapterError(ValueError):
    """The frozen scene adapter cannot safely patch this environment config."""


def validate_robot_contact_filter_contract() -> tuple[str, ...]:
    """Return the audited one-rigid-body-per-filter ContactSensor contract."""
    expected_count = 19
    if len(GO2_CONTACT_BODY_NAMES) != expected_count or len(set(GO2_CONTACT_BODY_NAMES)) != expected_count:
        raise SceneAdapterError("Go2 contact body contract must contain exactly 19 unique rigid bodies")
    if len(ROBOT_CONTACT_FILTER_PATHS) != expected_count:
        raise SceneAdapterError("Go2 contact filter count must equal the audited rigid-body count")
    for body, path in zip(GO2_CONTACT_BODY_NAMES, ROBOT_CONTACT_FILTER_PATHS):
        if not body or not path.endswith(f"/Robot/{body}") or ".*" in path or "*" in path:
            raise SceneAdapterError("DT1 contact filters must be exact single-body prim paths, never wildcards")
    return ROBOT_CONTACT_FILTER_PATHS


def _self_contained_ground_usd() -> tuple[Path, str]:
    """Return the checked-in collision mesh; never fall back to a Kit asset."""
    path = SELF_CONTAINED_GROUND_USD.resolve()
    if not path.is_file():
        raise SceneAdapterError(f"self-contained DT1 ground mesh is absent: {path}")
    contents = path.read_bytes()
    # Keep this deliberately small and auditable: the flat terrain must be a
    # real collision Mesh and must not add a hidden reference to Nucleus/Grid.
    required = (b'def Mesh "mesh"', b"PhysicsCollisionAPI", b"physics:collisionEnabled")
    if any(token not in contents for token in required):
        raise SceneAdapterError("self-contained DT1 ground USD lacks an explicit collision mesh")
    if b"default_environment.usd" in contents or b"@" in contents:
        raise SceneAdapterError("DT1 ground USD must not reference an external Isaac/Grid asset")
    return path, hashlib.sha256(contents).hexdigest()


@dataclass(frozen=True)
class DualTargetSceneSpec:
    """A physical scene plus one color assignment, never a policy feature."""

    group: GeometryGroup
    color_configuration: ColorConfiguration

    def __post_init__(self) -> None:
        self.group.validate_geometry()
        if self.color_configuration not in {"A_red_B_blue", "A_blue_B_red"}:
            raise SceneAdapterError("unknown DT1 color configuration")

    def slot_for_color(self, color: Color) -> TargetSlot:
        if color not in {"red", "blue"}:
            raise SceneAdapterError("target color must be red or blue")
        if self.color_configuration == "A_red_B_blue":
            return self.group.slot_a if color == "red" else self.group.slot_b
        return self.group.slot_b if color == "red" else self.group.slot_a

    def color_for_slot(self, slot_id: str) -> Color:
        if slot_id not in {"A", "B"}:
            raise SceneAdapterError("target slot must be A or B")
        if self.color_configuration == "A_red_B_blue":
            return "red" if slot_id == "A" else "blue"
        return "blue" if slot_id == "A" else "red"

    def audit_metadata(self) -> dict[str, object]:
        ground_usd, ground_sha256 = _self_contained_ground_usd()
        contact_filters = validate_robot_contact_filter_contract()
        return {
            "geometry_group_id": self.group.geometry_group_id,
            "lineage_root_id": self.group.lineage_root_id,
            "color_configuration": self.color_configuration,
            "slot_a_color": self.color_for_slot("A"),
            "slot_b_color": self.color_for_slot("B"),
            "start_xy": [self.group.start.x, self.group.start.y],
            "slot_a_center_xy": [self.group.slot_a.center.x, self.group.slot_a.center.y],
            "slot_b_center_xy": [self.group.slot_b.center.x, self.group.slot_b.center.y],
            "parking_centers_xy": {
                "A": [self.group.parking_region("A").center.x, self.group.parking_region("A").center.y],
                "B": [self.group.parking_region("B").center.x, self.group.parking_region("B").center.y],
            },
            "parking_radius_m": self.group.parking_radius_m,
            "box_size_m": list(TARGET_BOX_SIZE_M),
            "terrain_mesh_paths": list(GROUND_ONLY_MESH_PATHS),
            "terrain_type": "usd_static_mesh",
            "terrain_usd_path": str(ground_usd),
            "terrain_usd_sha256": ground_sha256,
            "terrain_excludes_targets": True,
            "target_contact_body_names": list(GO2_CONTACT_BODY_NAMES),
            "target_contact_filter_paths": list(contact_filters),
            "target_contact_filter_count": len(contact_filters),
        }


def dt1_development_groups() -> tuple[GeometryGroup, GeometryGroup]:
    """Two auditable development geometries for the DT1 2×4×2 precheck.

    The robot faces +X from the origin.  Both boxes have their identifiable
    approach face toward -X, so their parking regions are fixed by box
    geometry rather than by the expert's route.  Color is assigned later,
    which balances each color across left/right slots inside every group.
    """
    groups = (
        GeometryGroup(
            geometry_group_id="dt1_dev_000",
            lineage_root_id="dt1_dev_000",
            slot_a=TargetSlot("A", Vec2(2.00, 1.20), Vec2(-1.0, 0.0)),
            slot_b=TargetSlot("B", Vec2(2.00, -1.20), Vec2(-1.0, 0.0)),
            start=Vec2(0.0, 0.0),
        ),
        GeometryGroup(
            geometry_group_id="dt1_dev_001",
            lineage_root_id="dt1_dev_001",
            slot_a=TargetSlot("A", Vec2(2.35, 0.90), Vec2(-1.0, 0.0)),
            slot_b=TargetSlot("B", Vec2(2.35, -1.45), Vec2(-1.0, 0.0)),
            start=Vec2(0.0, 0.0),
        ),
    )
    for group in groups:
        group.validate_geometry()
    return groups


def contact_precheck_scene(target_color: Color) -> DualTargetSceneSpec:
    """Controlled physical box-contact layout, never a navigation task.

    The selected color occupies slot A on the robot's +X axis.  Slot B stays
    far enough away to retain two real physical boxes but cannot be reached by
    the straight low-level command used in this diagnostic.  This geometry is
    separate from the fixed 2×4 DT1 navigation layouts and is not a learner
    input or a task-success fixture.
    """
    if target_color not in {"red", "blue"}:
        raise SceneAdapterError("contact precheck target must be red or blue")
    group = GeometryGroup(
        geometry_group_id="dt1_contact_precheck_v1",
        lineage_root_id="dt1_contact_precheck_v1",
        slot_a=TargetSlot("A", Vec2(1.60, 0.0), Vec2(-1.0, 0.0)),
        slot_b=TargetSlot("B", Vec2(1.60, -1.40), Vec2(-1.0, 0.0)),
        start=Vec2(0.0, 0.0),
        # This diagnostic drives into the physical A box; its parking disc is
        # never a navigation success area.  Preserve the prior validation
        # geometry solely so its controlled collision path stays at x=1.60.
        stand_off_m=0.45,
        min_box_edge_clearance_m=0.10,
    )
    group.validate_geometry()
    configuration: ColorConfiguration = "A_red_B_blue" if target_color == "red" else "A_blue_B_red"
    return DualTargetSceneSpec(group, configuration)


def _replace_ray_mesh(ray_cfg: Any, label: str) -> None:
    if ray_cfg is None:
        raise SceneAdapterError(f"{label} is missing; old low-level terrain contract cannot be verified")
    ray_cfg.mesh_prim_paths = list(GROUND_ONLY_MESH_PATHS)


def _cuboid_cfg(sim_utils: Any, *, center: Vec2, color: Color) -> Any:
    palette = {"red": (0.85, 0.05, 0.05), "blue": (0.05, 0.10, 0.85)}
    return sim_utils.CuboidCfg(
        size=TARGET_BOX_SIZE_M,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=palette[color]),
        activate_contact_sensors=True,
    )


def apply_flat_dual_target_scene(
    env_cfg: Any,
    scene: DualTargetSceneSpec,
    *,
    go2_usd_path: str,
) -> dict[str, object]:
    """Patch the final Go2 config before ``gym.make`` for one DT1 scene.

    This is intentionally a one-way configuration adapter.  It does not call
    legacy planners, write a goal command, add a path marker, or inspect a
    task instruction.  The runner applies an external velocity only after the
    old locomotion observation/history has been rebuilt.
    """
    if not go2_usd_path or not isinstance(go2_usd_path, str):
        raise SceneAdapterError("a verified local Go2 USD path is required before gym.make")
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import RigidObjectCfg
        from omni.isaac.lab.sensors import ContactSensorCfg
        from omni.isaac.lab.terrains import TerrainImporterCfg
    except ImportError as exc:  # pragma: no cover - exercised only in Isaac
        raise RuntimeError("Isaac Lab imports are required only to apply the live scene patch") from exc

    if not hasattr(env_cfg, "scene") or not hasattr(env_cfg, "sim"):
        raise SceneAdapterError("expected a Go2 environment config with scene and sim")
    if not hasattr(env_cfg.scene, "robot") or not hasattr(env_cfg.scene.robot, "spawn"):
        raise SceneAdapterError("Go2 robot spawn config is unavailable")
    num_envs = getattr(env_cfg.scene, "num_envs", None)
    if not isinstance(num_envs, int) or num_envs != 1:
        raise SceneAdapterError("DT1 uses exactly one physical environment per audited episode")
    sim_dt = getattr(env_cfg.sim, "dt", None)
    if not isinstance(sim_dt, (int, float)) or sim_dt <= 0.0:
        raise SceneAdapterError("simulation dt must be positive before scene patching")
    ground_usd, _ = _self_contained_ground_usd()
    contact_filters = validate_robot_contact_filter_contract()

    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path=GROUND_PRIM_PATH,
        num_envs=num_envs,
        # ``plane`` resolves GroundPlaneCfg through Isaac's optional
        # Environments/Grid/default_environment.usd.  The audited cache does
        # not contain that file, so use our explicit finite static collision
        # mesh.  It remains the sole ray-caster mesh; targets stay physical
        # scene objects rather than a hidden terrain/navigation input.
        terrain_type="usd",
        usd_path=str(ground_usd),
        env_spacing=8.0,
        debug_vis=False,
    )
    # The old low-level height map expects one mesh path.  Ground-only is
    # explicit: targets are physical/rendered objects but intentionally not
    # obstacle input for this open flat-ground reach-and-stop task.
    _replace_ray_mesh(getattr(env_cfg.scene, "lidar_sensor", None), "lidar_sensor")
    _replace_ray_mesh(getattr(env_cfg.scene, "height_scanner", None), "height_scanner")
    env_cfg.scene.lidar_sensor.update_period = 4 * float(sim_dt)
    env_cfg.scene.height_scanner.update_period = 4 * float(sim_dt)
    env_cfg.sim.disable_contact_processing = False

    # This must modify the final config object immediately before gym.make;
    # changing a global asset alias earlier is not sufficient for NaVILA.
    env_cfg.scene.robot.spawn.usd_path = go2_usd_path
    env_cfg.scene.robot.init_state.pos = (scene.group.start.x, scene.group.start.y, GO2_RESET_HEIGHT_M)
    env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    for color in ("red", "blue"):
        target = scene.slot_for_color(color)
        setattr(
            env_cfg.scene,
            f"{color}_target",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{color}_target",
                spawn=_cuboid_cfg(sim_utils, center=target.center, color=color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(target.center.x, target.center.y, TARGET_BOX_CENTER_Z_M),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        )
        # The target side has exactly one rigid body; the filters each name
        # one audited robot rigid body.  Do not compress this list into a
        # wildcard: ContactSensor's backend would accept a bad view instead
        # of producing a 19-column force matrix.
        setattr(
            env_cfg.scene,
            f"{color}_target_contacts",
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{color}_target",
                filter_prim_paths_expr=list(contact_filters),
                history_length=1,
                track_air_time=False,
                update_period=float(sim_dt),
                debug_vis=False,
            ),
        )
    return scene.audit_metadata()


def validate_scene_metadata(metadata: dict[str, object]) -> None:
    """CPU-side guard used before an evidence manifest records a scene."""
    if metadata.get("terrain_mesh_paths") != list(GROUND_ONLY_MESH_PATHS):
        raise SceneAdapterError("DT1 terrain contract must use the one ground mesh path")
    if metadata.get("terrain_excludes_targets") is not True:
        raise SceneAdapterError("ground-only terrain limitation must be explicit in metadata")
    ground_usd, ground_sha256 = _self_contained_ground_usd()
    if metadata.get("terrain_type") != "usd_static_mesh":
        raise SceneAdapterError("DT1 terrain must identify the self-contained static-mesh mode")
    if metadata.get("terrain_usd_path") != str(ground_usd) or metadata.get("terrain_usd_sha256") != ground_sha256:
        raise SceneAdapterError("DT1 terrain metadata does not bind the checked-in ground USD")
    contact_filters = validate_robot_contact_filter_contract()
    if (metadata.get("target_contact_body_names") != list(GO2_CONTACT_BODY_NAMES)
            or metadata.get("target_contact_filter_paths") != list(contact_filters)
            or metadata.get("target_contact_filter_count") != len(contact_filters)):
        raise SceneAdapterError("DT1 metadata does not bind the audited exact robot contact filters")
    if metadata.get("slot_a_color") == metadata.get("slot_b_color"):
        raise SceneAdapterError("the two physical targets must have different colors")
