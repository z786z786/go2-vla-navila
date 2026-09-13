#!/usr/bin/env python3
"""Render and measure NaVILA dual-target initial-frame visibility by scene.

The masks come from IsaacLab semantic segmentation (not RGB thresholding, as
Matterport textures can contain arbitrary red/blue pixels).  A second render
hides the terrain and the other box to provide the same-pose unobstructed
projected silhouette.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_ASSET_ROOT = Path("/mnt/wxh/go2_short_vln/assets/isaac_sim_4_1")
DEFAULT_NAVILA_ROOT = Path("/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench")
# Chosen so that the farthest retained straight-route poses can still meet the
# hard 0.5% (512x512) visible-pixel gate; the gate remains the authority.
BOX_SIZE_M = (0.7, 0.7, 0.7)


def parse_args() -> argparse.Namespace:
    from omni.isaac.lab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--limit", type=int, default=None, help="bounded smoke/debug limit within the selected scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--rollout-video", type=Path, default=None,
                        help="optional true camera-following preview along the candidate GT segment")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--navila-root", type=Path, default=DEFAULT_NAVILA_ROOT)
    parser.add_argument("--task", default="go2_matterport_vision")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_mask(camera: Any, label: str) -> Any:
    import numpy as np

    data = camera.data.output["semantic_segmentation"][0]
    info = camera.data.info[0].get("semantic_segmentation") or {}
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    array = np.asarray(data)
    if array.ndim == 3:
        array = array[..., 0]
    mapping = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError:
            mapping = {}
    ids = []
    for key, value in (mapping.items() if isinstance(mapping, dict) else []):
        text = json.dumps(value, ensure_ascii=False).lower()
        if label.lower() in text:
            try:
                ids.append(int(key))
            except (TypeError, ValueError):
                pass
    if not ids:
        raise RuntimeError(f"semantic label {label!r} not present in idToLabels={mapping!r}")
    return np.isin(array, ids)


def _semantic_info(camera: Any) -> Any:
    """Return raw label mapping for audit/debugging without relying on RGB."""
    info = camera.data.info[0].get("semantic_segmentation") or {}
    mapping = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError:
            return mapping
    return mapping


def _rgb(camera: Any) -> Any:
    import numpy as np

    output = camera.data.output
    value = output["rgb"][0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"camera RGB has invalid shape {array.shape}")
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    # IsaacLab's camera output is already in the intended 512x512 image
    # orientation.  Do not rotate it: rotating here makes verticals lean and
    # invalidates the visual audit (the segmentation masks use the same frame
    # convention after their own explicit conversion).
    return array


def _cuboid_cfg(sim_utils: Any, color: str) -> Any:
    palette = {"red": (0.85, 0.05, 0.05), "blue": (0.05, 0.10, 0.85)}
    return sim_utils.CuboidCfg(
        size=BOX_SIZE_M,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=palette[color]),
        semantic_tags=[("class", f"dual_target_{color}_v3")],
    )


def _yaw_quaternion(degrees: float) -> list[float]:
    half = math.radians(float(degrees)) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _measurement(mask: Any, projected_pixels: int, projected_mask: Any) -> dict[str, Any]:
    import numpy as np

    y, x = np.nonzero(mask)
    py, px = np.nonzero(projected_mask)
    center_x = float(px.mean()) if len(px) else math.nan
    return {
        "visible_pixels": int(len(x)),
        "projected_pixels": int(projected_pixels),
        "projection_center_x_px": center_x,
        "visible_bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None,
        "projected_bbox_xyxy": [int(px.min()), int(py.min()), int(px.max()), int(py.max())] if len(px) else None,
    }


def run() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = [item for item in payload["candidates"] if str(item["scene"]) == args.scene]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        candidates = candidates[: args.limit]
    if not candidates:
        raise ValueError(f"no candidates for scene {args.scene}")

    from src.inference.isaac_client import configure_app_launcher, configure_runtime_assets

    configure_app_launcher(args.asset_root)
    from omni.isaac.lab.app import AppLauncher

    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import cv2
        import gymnasium as gym
        import numpy as np
        import torch
        import omni.isaac.lab.sim as sim_utils
        import omni.isaac.vlnce.config  # noqa: F401
        import omni.usd
        from omni.isaac.lab.assets import RigidObjectCfg
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        from omni.isaac.vlnce.utils import ASSETS_DIR
        from pxr import UsdGeom

        local_go2 = configure_runtime_assets(args.asset_root)
        first = candidates[0]
        env_cfg = parse_env_cfg(args.task, num_envs=1)
        env_cfg.seed = 20260908
        env_cfg.scene.robot.spawn.usd_path = str(local_go2)
        start = first["start_position"]
        env_cfg.scene.robot.init_state.pos = (start[0], start[1], start[2] + 0.4)
        env_cfg.scene.robot.init_state.rot = tuple(_yaw_quaternion(first["camera_yaw_degrees"]))
        env_cfg.scene_id = args.scene
        env_cfg.episode_id = str(first["source_episode_id"])
        env_cfg.traj_id = str(first["source_route_id"])
        env_cfg.instruction_text = "Go to the red box and stop in front of it."
        env_cfg.instruction_tokens = [0] * 200
        env_cfg.goals = [{"position": first["box_a_position"], "radius": 0.3}]
        env_cfg.reference_path = np.asarray(first["source_gt_segment"])
        env_cfg.expert_path = np.asarray(first["source_gt_segment"])
        env_cfg.expert_path_length = len(env_cfg.expert_path)
        env_cfg.expert_time = np.arange(env_cfg.expert_path_length)
        usd_path = Path(ASSETS_DIR) / "matterport_usd" / args.scene / f"{args.scene}.usd"
        if not usd_path.is_file():
            raise FileNotFoundError(usd_path)
        env_cfg.scene.terrain.obj_filepath = str(usd_path)
        env_cfg.scene.rgbd_camera.data_types = ["rgb", "distance_to_image_plane", "semantic_segmentation"]
        env_cfg.scene.rgbd_camera.colorize_semantic_segmentation = False
        # Hide legacy goal/path visualization well above the scene.
        for name in ("disk_1", "disk_2"):
            value = getattr(env_cfg.scene, name, None)
            if value is not None:
                value.init_state.pos = (start[0], start[1], start[2] + 100.0)
        for color, key in (("red", "box_a_position"), ("blue", "box_b_position")):
            position = first[key]
            setattr(env_cfg.scene, f"{color}_target", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{color}_target",
                spawn=_cuboid_cfg(sim_utils, color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(position), rot=(1.0, 0.0, 0.0, 0.0)),
            ))
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env.reset()
        base = env.unwrapped
        scene = base.scene
        camera = scene["rgbd_camera"]
        robot = scene["robot"]
        targets = {"A": scene["red_target"], "B": scene["blue_target"]}
        terrain_path = str(env_cfg.scene.terrain.prim_path)
        terrain_prim = omni.usd.get_context().get_stage().GetPrimAtPath(terrain_path)
        if not terrain_prim.IsValid():
            raise RuntimeError(f"terrain prim is unavailable: {terrain_path}")
        terrain_imageable = UsdGeom.Imageable(terrain_prim)
        args.image_dir.mkdir(parents=True, exist_ok=True)

        def place(candidate: dict[str, Any], isolate: str | None = None) -> None:
            start_position = candidate["start_position"]
            pose = [start_position[0], start_position[1], start_position[2] + 0.4, *_yaw_quaternion(candidate["camera_yaw_degrees"])]
            robot.write_root_pose_to_sim(torch.tensor([pose], device=base.device, dtype=torch.float32))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
            for slot, key in (("A", "box_a_position"), ("B", "box_b_position")):
                position = candidate[key]
                if isolate is not None and slot != isolate:
                    position = [start_position[0], start_position[1], start_position[2] - 100.0]
                target_pose = [*position, 1.0, 0.0, 0.0, 0.0]
                targets[slot].write_root_pose_to_sim(torch.tensor([target_pose], device=base.device, dtype=torch.float32))
                targets[slot].write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
            scene.write_data_to_sim()
            for _ in range(2):
                base.sim.step(render=True)
                scene.update(float(env_cfg.sim.dt))

        if args.rollout_video is not None:
            # This is a visual validation rollout only: the robot is placed at
            # each recorded GT waypoint and the camera frame is rendered after
            # each pose update.  It is deliberately separate from the static
            # visibility receipts used for training/splitting.
            import cv2
            preview_candidate = candidates[0]
            waypoints = preview_candidate.get("source_gt_segment") or []
            if len(waypoints) < 2:
                raise RuntimeError("candidate has no complete GT segment for rollout preview")
            frames = []
            for index, waypoint in enumerate(waypoints):
                next_point = waypoints[min(index + 1, len(waypoints) - 1)]
                if index == len(waypoints) - 1:
                    next_point = waypoints[max(0, index - 1)]
                yaw = math.degrees(math.atan2(float(next_point[1]) - float(waypoint[1]),
                                               float(next_point[0]) - float(waypoint[0])))
                rollout_candidate = {**preview_candidate, "start_position": waypoint,
                                     "camera_yaw_degrees": yaw}
                place(rollout_candidate)
                frames.append(_rgb(camera))
            args.rollout_video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.rollout_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10.0, (512, 512))
            if not writer.isOpened():
                raise RuntimeError(f"failed to open rollout video {args.rollout_video}")
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            print(json.dumps({"rollout_video": str(args.rollout_video),
                              "rollout_frames": len(frames),
                              "rollout_waypoints": len(waypoints)}), flush=True)

        results = []
        for index, candidate in enumerate(candidates):
            terrain_imageable.MakeVisible()
            place(candidate)
            normal = _rgb(camera)
            visible_masks = {"A": _semantic_mask(camera, "dual_target_red_v3"),
                             "B": _semantic_mask(camera, "dual_target_blue_v3")}
            image_path = args.image_dir / f"{candidate['candidate_id']}.jpg"
            if not cv2.imwrite(str(image_path), cv2.cvtColor(normal, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"failed to write {image_path}")
            projected_masks = {}
            terrain_imageable.MakeInvisible()
            for slot, color in (("A", "red"), ("B", "blue")):
                place(candidate, isolate=slot)
                projected_masks[slot] = _semantic_mask(camera, f"dual_target_{color}_v3")
            terrain_imageable.MakeVisible()
            visibility = {"image_width": 512, "image_height": 512}
            for slot in ("A", "B"):
                # Isaac's render-product update can lag one frame after a root
                # pose write.  Keep the raw isolated count, but conservatively
                # never claim fewer projected pixels than the observed mask.
                raw_projected = int(projected_masks[slot].sum())
                visibility[slot] = _measurement(
                    visible_masks[slot], max(raw_projected, int(visible_masks[slot].sum())), projected_masks[slot]
                )
                visibility[slot]["projected_pixels_raw"] = raw_projected
            results.append({**candidate, "status": "live_visibility_measured", "visibility": visibility,
                            "semantic_id_to_labels": _semantic_info(camera),
                            "initial_rgb": str(image_path), "initial_rgb_sha256": sha256(image_path)})
            if (index + 1) % 25 == 0:
                print(json.dumps({"scene": args.scene, "measured": index + 1, "total": len(candidates)}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "format": "navila-dual-target-live-visibility-v1", "scene": args.scene,
            "source_candidates_sha256": sha256(args.candidates), "candidates": results,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "scene": args.scene, "measured": len(results)}))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    run()
