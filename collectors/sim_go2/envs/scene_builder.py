from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import carb
import isaaclab.sim as sim_utils
import omni.client
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path

from collectors.sim_go2.utils.randomization import sample_camera_offset, sample_light_intensity
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TARGET_SLOT_NAMES = ("target_0", "target_1", "target_2", "target_3")
STATIC_PROP_SLOT_NAMES = (
    "static_prop_0",
    "static_prop_1",
    "static_prop_2",
    "static_prop_3",
    "static_prop_4",
    "static_prop_5",
    "static_prop_6",
    "static_prop_7",
    "static_prop_8",
    "static_prop_9",
    "static_prop_10",
    "static_prop_11",
)
HIDDEN_TARGET_XY = 256.0
HIDDEN_TARGET_Z = -5.0


def _default_target_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Target{index}",
        spawn=sim_utils.CuboidCfg(
            size=(0.35, 0.35, 0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), roughness=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(HIDDEN_TARGET_XY + index * 4.0, HIDDEN_TARGET_XY, HIDDEN_TARGET_Z)),
    )


def _default_static_prop_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/StaticProp{index}",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8), roughness=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(HIDDEN_TARGET_XY + index * 4.0, HIDDEN_TARGET_XY + 64.0, HIDDEN_TARGET_Z)),
    )


@configclass
class Go2CollectionSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75), roughness=0.95),
        debug_vis=False,
    )
    robot = copy.deepcopy(UNITREE_GO2_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    environment = AssetBaseCfg(
        prim_path="/World/Environment",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"),
    )
    target_0 = _default_target_cfg(0)
    target_1 = _default_target_cfg(1)
    target_2 = _default_target_cfg(2)
    target_3 = _default_target_cfg(3)
    static_prop_0 = _default_static_prop_cfg(0)
    static_prop_1 = _default_static_prop_cfg(1)
    static_prop_2 = _default_static_prop_cfg(2)
    static_prop_3 = _default_static_prop_cfg(3)
    static_prop_4 = _default_static_prop_cfg(4)
    static_prop_5 = _default_static_prop_cfg(5)
    static_prop_6 = _default_static_prop_cfg(6)
    static_prop_7 = _default_static_prop_cfg(7)
    static_prop_8 = _default_static_prop_cfg(8)
    static_prop_9 = _default_static_prop_cfg(9)
    static_prop_10 = _default_static_prop_cfg(10)
    static_prop_11 = _default_static_prop_cfg(11)
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera_sensor",
        update_period=0.05,
        height=384,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.32715, -0.00003, 0.04297),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )


@dataclass
class Go2SceneHandles:
    scene: InteractiveScene
    scene_kind: str
    scene_asset_path: str
    target_name_by_id: dict[str, str]
    target_cfg_by_id: dict[str, dict[str, Any]]
    static_props: tuple[dict[str, Any], ...]
    robot_name: str = "robot"
    camera_name: str = "front_camera"

    @property
    def robot(self):
        return self.scene[self.robot_name]

    def target(self, target_id: str) -> RigidObject:
        return self.scene[self.target_name_by_id[target_id]]

    @property
    def primary_target_id(self) -> str:
        return next(iter(self.target_name_by_id), "")

    @property
    def goal_box(self) -> RigidObject:
        return self.target(self.primary_target_id)

    @property
    def front_camera(self) -> Camera:
        return self.scene[self.camera_name]


def build_scene(
    scene_cfg_dict: Mapping[str, Any],
    robot_cfg_dict: Mapping[str, Any],
    camera_cfg_dict: Mapping[str, Any],
    dr_cfg_dict: Mapping[str, Any],
    control_dt: float,
    rng,
) -> Go2SceneHandles:
    scene_kind = str(scene_cfg_dict.get("kind", "plane")).strip().lower()
    scene_cfg = Go2CollectionSceneCfg(
        num_envs=int(scene_cfg_dict.get("num_envs", 1)),
        env_spacing=float(scene_cfg_dict.get("env_spacing", 6.0)),
        lazy_sensor_update=False,
    )

    terrain_cfg = scene_cfg_dict.get("terrain", {})
    scene_cfg.terrain.physics_material.static_friction = float(terrain_cfg.get("static_friction", 1.0))
    scene_cfg.terrain.physics_material.dynamic_friction = float(terrain_cfg.get("dynamic_friction", 1.0))
    scene_cfg.terrain.physics_material.restitution = float(terrain_cfg.get("restitution", 0.0))
    if scene_kind == "plane":
        # IsaacLab 5.1 ground-plane spawning can return no matching Plane child prim for
        # bind_physics_material(), which crashes local closed-loop startup with
        # Stage.GetPrimAtPath(... NoneType). Keep plane scenes loadable by skipping the
        # explicit material bind and falling back to the default ground-plane material.
        scene_cfg.terrain.physics_material = None

    configured_targets = _normalize_scene_targets(scene_cfg_dict)
    if len(configured_targets) > len(TARGET_SLOT_NAMES):
        raise ValueError(f"Configured {len(configured_targets)} scene targets, max supported is {len(TARGET_SLOT_NAMES)}")
    target_name_by_id: dict[str, str] = {}
    target_cfg_by_id: dict[str, dict[str, Any]] = {}
    for index, slot_name in enumerate(TARGET_SLOT_NAMES):
        slot_cfg: RigidObjectCfg = getattr(scene_cfg, slot_name)
        if index < len(configured_targets):
            target_cfg = configured_targets[index]
            slot_cfg.spawn.size = tuple(float(value) for value in target_cfg.get("size", (0.35, 0.35, 0.50)))
            slot_cfg.spawn.visual_material.diffuse_color = tuple(float(value) for value in target_cfg.get("color", (0.05, 0.05, 0.05)))
            slot_cfg.init_state.pos = (
                HIDDEN_TARGET_XY + index * 4.0,
                HIDDEN_TARGET_XY,
                float(target_cfg.get("height", 0.25)),
            )
            target_id = str(target_cfg["id"])
            target_name_by_id[target_id] = slot_name
            target_cfg_by_id[target_id] = target_cfg
        else:
            slot_cfg.init_state.pos = (HIDDEN_TARGET_XY + index * 4.0, HIDDEN_TARGET_XY, HIDDEN_TARGET_Z)

    configured_static_props = _normalize_static_props(scene_cfg_dict)
    if len(configured_static_props) > len(STATIC_PROP_SLOT_NAMES):
        raise ValueError(
            f"Configured {len(configured_static_props)} static props, max supported is {len(STATIC_PROP_SLOT_NAMES)}"
        )
    for index, slot_name in enumerate(STATIC_PROP_SLOT_NAMES):
        slot_cfg: RigidObjectCfg = getattr(scene_cfg, slot_name)
        if index < len(configured_static_props):
            prop_cfg = configured_static_props[index]
            slot_cfg.spawn.size = tuple(float(value) for value in prop_cfg.get("size", (0.5, 0.5, 0.5)))
            slot_cfg.spawn.visual_material.diffuse_color = tuple(float(value) for value in prop_cfg.get("color", (0.8, 0.8, 0.8)))
            position = tuple(float(value) for value in prop_cfg.get("position", (0.0, 0.0, 0.25)))
            slot_cfg.init_state.pos = position
        else:
            slot_cfg.init_state.pos = (HIDDEN_TARGET_XY + index * 4.0, HIDDEN_TARGET_XY + 64.0, HIDDEN_TARGET_Z)

    camera_dr_cfg = (dr_cfg_dict.get("camera") if isinstance(dr_cfg_dict, dict) else {}) or {}
    sampled_offset = sample_camera_offset(camera_cfg_dict.get("offset", {}), camera_dr_cfg, rng)
    scene_cfg.front_camera.prim_path = str(camera_cfg_dict.get("mount_prim_path", scene_cfg.front_camera.prim_path))
    scene_cfg.front_camera.update_period = float(control_dt)
    scene_cfg.front_camera.height = int(camera_cfg_dict.get("height", 384))
    scene_cfg.front_camera.width = int(camera_cfg_dict.get("width", 640))
    scene_cfg.front_camera.data_types = _resolve_camera_data_types(camera_cfg_dict)
    scene_cfg.front_camera.spawn.focal_length = float(camera_cfg_dict.get("focal_length", 24.0))
    scene_cfg.front_camera.spawn.focus_distance = float(camera_cfg_dict.get("focus_distance", 400.0))
    scene_cfg.front_camera.spawn.horizontal_aperture = float(camera_cfg_dict.get("horizontal_aperture", 20.955))
    clipping = camera_cfg_dict.get("clipping_range", (0.05, 100.0))
    scene_cfg.front_camera.spawn.clipping_range = (float(clipping[0]), float(clipping[1]))
    scene_cfg.front_camera.offset = CameraCfg.OffsetCfg(
        pos=tuple(float(value) for value in sampled_offset["pos"]),
        rot=tuple(float(value) for value in sampled_offset["rot"]),
        convention=str(sampled_offset["convention"]),
    )

    light_dr_cfg = (dr_cfg_dict.get("lighting") if isinstance(dr_cfg_dict, dict) else {}) or {}
    light_cfg = scene_cfg_dict.get("lighting", {})
    scene_cfg.light.spawn.intensity = sample_light_intensity(float(light_cfg.get("intensity", 1500.0)), light_dr_cfg or light_cfg, rng)
    scene_cfg.light.spawn.color = tuple(float(value) for value in light_cfg.get("color", (1.0, 1.0, 1.0)))

    scene_asset_path = ""
    if scene_kind == "plane":
        scene_cfg.environment = None
    else:
        scene_cfg.terrain = None
        scene_asset_path = _resolve_scene_asset_path(scene_cfg_dict)
        scene_cfg.environment.spawn = sim_utils.UsdFileCfg(usd_path=scene_asset_path)
        scene_cfg.environment.init_state.pos = tuple(float(value) for value in scene_cfg_dict.get("environment_offset", (0.0, 0.0, 0.0)))
        print(f"[INFO] resolved scene usd_path={scene_asset_path}", flush=True)

    scene_cfg.robot.spawn.asset_path = _resolve_go2_asset_path(robot_cfg_dict)
    scene_cfg.robot.spawn.usd_dir = _resolve_go2_cache_dir(robot_cfg_dict)
    scene_cfg.robot.init_state.pos = (0.0, 0.0, float(robot_cfg_dict.get("base_height", 0.4)))
    print(f"[INFO] resolved Go2 asset_path={scene_cfg.robot.spawn.asset_path}", flush=True)
    print(f"[INFO] resolved Go2 usd_dir={scene_cfg.robot.spawn.usd_dir}", flush=True)

    scene = InteractiveScene(scene_cfg)
    return Go2SceneHandles(
        scene=scene,
        scene_kind=scene_kind,
        scene_asset_path=scene_asset_path,
        target_name_by_id=target_name_by_id,
        target_cfg_by_id=target_cfg_by_id,
        static_props=tuple(configured_static_props),
    )


def _normalize_scene_targets(scene_cfg_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    configured = list(scene_cfg_dict.get("targets", []))
    if configured:
        return [_normalize_scene_target(item, index) for index, item in enumerate(configured)]

    goal_box_cfg = scene_cfg_dict.get("goal_box", {})
    return [
        {
            "id": "goal_box",
            "class": "goal_box",
            "label": "goal box",
            "shape": "box",
            "color_name": "black",
            "size": tuple(float(value) for value in goal_box_cfg.get("size", (0.35, 0.35, 0.50))),
            "color": tuple(float(value) for value in goal_box_cfg.get("color", (0.05, 0.05, 0.05))),
            "height": float(goal_box_cfg.get("height", 0.25)),
        }
    ]


def _normalize_static_props(scene_cfg_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    configured = list(scene_cfg_dict.get("static_props", []))
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(configured):
        normalized.append(
            {
                "id": str(item.get("id") or f"static_prop_{index}"),
                "size": tuple(float(value) for value in item.get("size", (0.5, 0.5, 0.5))),
                "color": tuple(float(value) for value in item.get("color", (0.8, 0.8, 0.8))),
                "position": tuple(float(value) for value in item.get("position", (0.0, 0.0, 0.25))),
            }
        )
    return normalized


def _normalize_scene_target(target_cfg: Mapping[str, Any], index: int) -> dict[str, Any]:
    visual_cfg = target_cfg.get("visual", {}) if isinstance(target_cfg.get("visual"), Mapping) else {}
    size = target_cfg.get("size", visual_cfg.get("size", (0.35, 0.35, 0.50)))
    color = target_cfg.get("color", visual_cfg.get("color", (0.05, 0.05, 0.05)))
    height = target_cfg.get("height", visual_cfg.get("height", 0.25))
    target_id = str(target_cfg.get("id") or target_cfg.get("target_id") or f"target_{index}")
    return {
        "id": target_id,
        "class": str(target_cfg.get("class") or target_cfg.get("target_class") or target_id),
        "label": str(target_cfg.get("label") or target_cfg.get("target_label") or target_id.replace("_", " ")),
        "description": str(target_cfg.get("description") or target_cfg.get("target_description") or target_cfg.get("label") or target_id.replace("_", " ")),
        "shape": str(target_cfg.get("shape") or target_cfg.get("target_shape") or "box"),
        "color_name": str(target_cfg.get("color_name") or target_cfg.get("target_color") or ""),
        "size": tuple(float(value) for value in size),
        "color": tuple(float(value) for value in color),
        "height": float(height),
    }


def _resolve_camera_data_types(camera_cfg_dict: Mapping[str, Any]) -> list[str]:
    explicit = camera_cfg_dict.get("data_types")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit]
    data_types = ["rgb"]
    if bool(camera_cfg_dict.get("enable_depth", False)):
        data_types.append(str(camera_cfg_dict.get("depth_data_type", "distance_to_image_plane")))
    return data_types


def _resolve_go2_asset_path(robot_cfg_dict: Mapping[str, Any]) -> str:
    configured = str(robot_cfg_dict.get("asset_path", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    env_candidates = [
        os.environ.get("GO2_DESCRIPTION_URDF", "").strip(),
        os.environ.get("UNITREE_GO2_URDF", "").strip(),
    ]
    for value in env_candidates:
        if value:
            candidates.append(Path(value).expanduser())

    unitree_ros_dir = os.environ.get("UNITREE_ROS_DIR", "").strip()
    if unitree_ros_dir:
        candidates.append(Path(unitree_ros_dir) / "robots" / "go2_description" / "urdf" / "go2_description.urdf")

    repo_root = PACKAGE_ROOT.parent.parent
    candidates.extend(
        [
            repo_root / "unitree_ros" / "robots" / "go2_description" / "urdf" / "go2_description.urdf",
            repo_root / "third_party" / "unitree_ros" / "robots" / "go2_description" / "urdf" / "go2_description.urdf",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    searched = "\n".join(f"  - {candidate}" for candidate in candidates) or "  - <none>"
    raise FileNotFoundError(
        "Unable to resolve the Unitree Go2 URDF asset.\n"
        "Please either set UNITREE_ROS_DIR or configure robot.asset_path.\n"
        f"Searched:\n{searched}"
    )


def _resolve_go2_cache_dir(robot_cfg_dict: Mapping[str, Any]) -> str:
    configured = str(robot_cfg_dict.get("asset_cache_dir", "")).strip()
    cache_dir = Path(configured).expanduser() if configured else PACKAGE_ROOT / "outputs" / "asset_cache" / "go2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir.resolve())


def _resolve_scene_asset_path(scene_cfg_dict: Mapping[str, Any]) -> str:
    asset_source = str(scene_cfg_dict.get("asset_source", "isaac_builtin")).strip().lower()
    scene_kind = str(scene_cfg_dict.get("kind", "office")).strip().lower()
    explicit_path = str(scene_cfg_dict.get("usd_path", "")).strip()
    if explicit_path:
        candidate = _normalize_scene_path(explicit_path, asset_source)
        if check_file_path(candidate):
            return candidate
        raise FileNotFoundError(f"Scene USD path does not exist: {candidate}")
    candidates = [_normalize_scene_path(item, asset_source) for item in scene_cfg_dict.get("usd_path_candidates", [])]
    if not candidates:
        candidates = _default_scene_candidates(scene_kind)
    for candidate in candidates:
        if check_file_path(candidate):
            return candidate
    discovered_candidates = _discover_scene_candidates(scene_kind)
    for candidate in discovered_candidates:
        if check_file_path(candidate):
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates) or "  - <none>"
    discovered = "\n".join(f"  - {candidate}" for candidate in discovered_candidates[:12]) or "  - <none>"
    raise FileNotFoundError(
        "Unable to resolve indoor scene USD.\n"
        f"scene.kind={scene_kind}\nConfigured candidates:\n{searched}\nDiscovered candidates:\n{discovered}"
    )


def _normalize_scene_path(path: str, asset_source: str) -> str:
    normalized = path.strip()
    if not normalized:
        return normalized
    if asset_source == "local":
        return str(Path(normalized).expanduser().resolve())
    if normalized.startswith("/Isaac") or normalized.startswith("omniverse://"):
        root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
        if normalized.startswith("/Isaac") and root:
            return f"{root}{normalized}"
    return normalized


def _default_scene_candidates(scene_kind: str) -> list[str]:
    if scene_kind == "warehouse":
        return [
            f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
            f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
        ]
    return [
        f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Office/full_office.usd",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Office/room.usd",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Office/conference_room.usd",
    ]


def _discover_scene_candidates(scene_kind: str) -> list[str]:
    base_dir = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse" if scene_kind == "warehouse" else f"{ISAAC_NUCLEUS_DIR}/Environments/Office"
    try:
        return _rank_scene_candidates(_list_usd_files(base_dir, max_depth=2), scene_kind)
    except Exception:
        return []


def _list_usd_files(base_dir: str, max_depth: int) -> list[str]:
    result, entries = omni.client.list(base_dir)
    if result != omni.client.Result.OK:
        return []
    discovered: list[str] = []
    for entry in entries:
        rel_path = entry.relative_path.strip("/")
        full_path = f"{base_dir}/{rel_path}"
        if rel_path.startswith((".", "Props", "Materials", "Looks")):
            continue
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            if max_depth > 0:
                discovered.extend(_list_usd_files(full_path, max_depth=max_depth - 1))
        elif rel_path.endswith((".usd", ".usda")):
            discovered.append(full_path)
    return discovered


def _rank_scene_candidates(candidates: list[str], scene_kind: str) -> list[str]:
    preferred = []
    fallback = []
    keywords = ["office", "room", "conference"] if scene_kind != "warehouse" else ["warehouse", "full"]
    for candidate in candidates:
        lowered = candidate.lower()
        if any(keyword in lowered for keyword in keywords):
            preferred.append(candidate)
        else:
            fallback.append(candidate)
    return preferred + fallback
