#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def _env_root(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _bootstrap_candidates() -> list[Path]:
    candidates = [
        PROJECT_ROOT,
        PACKAGE_ROOT,
        Path("/home/wxh/unitree_rl_lab"),
        Path("/home/wxh/unitree_rl_lab/source"),
        Path("/home/wxh/unitree_rl_lab/source/unitree_rl_lab"),
        Path("/home/wxh/IsaacLab"),
        Path("/home/wxh/IsaacLab/source"),
        Path("/home/wxh/IsaacLab/source/isaaclab"),
        Path("/home/zxq/zxq/unitree_rl_lab"),
        Path("/home/zxq/zxq/unitree_rl_lab/source"),
        Path("/home/zxq/zxq/unitree_rl_lab/source/unitree_rl_lab"),
        Path("/home/zxq/zxq/IsaacLab"),
        Path("/home/zxq/zxq/IsaacLab/source"),
        Path("/home/zxq/zxq/IsaacLab/source/isaaclab"),
    ]
    unitree_root = _env_root("UNITREE_RL_LAB_ROOT")
    if unitree_root is not None:
        candidates.extend([unitree_root, unitree_root / "source", unitree_root / "source" / "unitree_rl_lab"])
    isaaclab_root = _env_root("ISAACLAB_ROOT")
    if isaaclab_root is not None:
        candidates.extend([isaaclab_root, isaaclab_root / "source", isaaclab_root / "source" / "isaaclab"])
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _bootstrap_paths() -> None:
    for candidate in _bootstrap_candidates():
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_bootstrap_paths()

from collectors.sim_go2.utils.io import apply_cli_overrides, dump_json, dump_yaml, load_resolved_config, resolve_output_dir


def _parse_optional_bool(raw_value: str | None) -> bool:
    if raw_value is None:
        return True
    lowered = raw_value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw_value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Isaac Sim raw trajectories for Go2 VLA data.")
    parser.add_argument("--config", type=Path, required=True, help="Top-level collection YAML config.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override raw trajectory output directory.")
    parser.add_argument("--packed_output_dir", type=Path, default=None, help="Deprecated legacy option; raw-only collector ignores it.")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--auto_pack", nargs="?", default=None, const=True, type=_parse_optional_bool, help="Deprecated legacy option; raw-only collector ignores it.")
    parser.add_argument("--headless", nargs="?", default=None, const=True, type=_parse_optional_bool)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_resolved_config(args.config)
    config = apply_cli_overrides(
        config,
        {
            "output_dir": str(args.output_dir) if args.output_dir is not None else None,
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "headless": args.headless,
            "max_steps_per_episode": args.max_steps_per_episode,
            "auto_pack": args.auto_pack,
        },
    )
    runtime_cfg = config.setdefault("runtime", {})
    raw_output_dir = resolve_output_dir(runtime_cfg.get("output_dir", "../data/isaac_sim_raw/default_run"), PACKAGE_ROOT)
    runtime_cfg["output_dir"] = str(raw_output_dir)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(runtime_cfg.get("headless", True)), enable_cameras=True)
    simulation_app = app_launcher.app
    from collectors.sim_go2.controllers.heuristic_nav_controller import PrivilegedNavExpert
    from collectors.sim_go2.envs.go2_nav_env import Go2NavCollectionEnv
    from collectors.sim_go2.recorders.raw_episode_logger import RawEpisodeLogger
    env = None
    logger = None
    progress_log_path = raw_output_dir / "collector_progress.log"
    error_path = raw_output_dir / "collection_error.txt"

    def log_progress(message: str) -> None:
        formatted = f"[INFO] {message}"
        print(formatted, flush=True)
        progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_log_path.open("a", encoding="utf-8") as handle:
            handle.write(formatted + "\n")

    try:
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        progress_log_path.write_text("", encoding="utf-8")
        if error_path.exists():
            error_path.unlink()
        dump_yaml(raw_output_dir / "resolved_config.yaml", config)
        env = Go2NavCollectionEnv(config)
        controller = PrivilegedNavExpert(config.get("controller", {}), control_dt=float(config.get("sim", {}).get("control_dt", 0.05)))
        logger = RawEpisodeLogger(
            raw_output_dir,
            image_format=str(config.get("data", {}).get("image_format", "jpg")),
            jpeg_quality=int(config.get("data", {}).get("jpeg_quality", 95)),
            save_depth=bool(config.get("camera", {}).get("enable_depth", False)),
            depth_format=str(config.get("data", {}).get("depth_format", "png")),
            depth_scale=float(config.get("data", {}).get("depth_scale", 1000.0)),
            depth_min=config.get("camera", {}).get("depth_min"),
            depth_max=config.get("camera", {}).get("depth_max"),
            depth_invalid_fill_value=float(config.get("camera", {}).get("depth_invalid_fill_value", 0.0)),
        )
        completed_ids = set(logger.list_completed_episode_ids())
        log_progress(f"raw_output_dir={raw_output_dir}")
        log_progress(f"completed_episodes_found={len(completed_ids)}")
        scene_metadata = {
            "dataset_name": str(runtime_cfg.get("dataset_name", "go2_isaac_collection")),
            "seed": int(runtime_cfg.get("seed", 0)),
            "scene_kind": env.handles.scene_kind,
            "scene_asset_path": env.handles.scene_asset_path,
            "motion_backend": env.motion_backend,
            "camera": {
                "rgb_interface_source": "isaaclab.camera.rgb",
                "width": int(config.get("camera", {}).get("width", 640)),
                "height": int(config.get("camera", {}).get("height", 384)),
                "enable_depth": bool(config.get("camera", {}).get("enable_depth", False)),
                "depth_interface_source": "isaaclab.camera.distance_to_image_plane" if bool(config.get("camera", {}).get("enable_depth", False)) else None,
                "depth_data_type": str(config.get("camera", {}).get("depth_data_type", "distance_to_image_plane")),
                "depth_min": config.get("camera", {}).get("depth_min"),
                "depth_max": config.get("camera", {}).get("depth_max"),
                "depth_format": str(config.get("data", {}).get("depth_format", "png")),
                "depth_scale": float(config.get("data", {}).get("depth_scale", 1000.0)),
            },
            "state_fields": list(config.get("data", {}).get("state_fields", ["vx", "vy", "vz"])),
            "train_action_fields": list(config.get("data", {}).get("train_action_fields", ["vx", "wz"])),
            "raw_action_fields": list(config.get("data", {}).get("raw_action_fields", ["vx", "vy", "wz"])),
            "visibility_curriculum": dict(config.get("task", {}).get("visibility_curriculum", {})),
        }
        dump_json(raw_output_dir / "scene_metadata.json", scene_metadata)

        requested_episodes = int(runtime_cfg.get("num_episodes", 1))
        pre_roll_frames = int(runtime_cfg.get("pre_roll_frames", 3))
        post_success_settle_frames = int(runtime_cfg.get("post_success_settle_frames", 6))
        episode_summaries: list[dict[str, Any]] = []
        total_collected = 0

        def zero_command() -> dict[str, Any]:
            return {
                "raw_cmd_full": [0.0, 0.0, 0.0],
                "raw_cmd_train": [0.0, 0.0],
                "expert_debug": {"phase": "zero_hold"},
            }

        def build_step_meta(
            *,
            task: dict[str, Any],
            episode_meta: dict[str, Any],
            debug_state: dict[str, Any],
            step_id: int,
            phase: str,
            success: bool = False,
            collision: bool = False,
            timeout: bool = False,
            termination_reason: str = "",
            extra_debug: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            debug_payload = {
                "goal_distance": float(debug_state["goal_distance"]),
                "goal_heading": float(debug_state["goal_heading"]),
                "yaw_rate": float(debug_state["yaw_rate"]),
                "target_visible": bool(debug_state["target_visible"]),
                "target_pixel_ratio": float(debug_state["target_pixel_ratio"]),
                "target_heading_camera_deg": float(debug_state["target_heading_camera_deg"]),
            }
            if extra_debug:
                debug_payload.update(extra_debug)
            return {
                "step_id": step_id,
                "timestamp": step_id * float(config.get("sim", {}).get("control_dt", 0.05)),
                "task_id": task["task_id"],
                "instruction_template_id": task["instruction_template_id"],
                "active_target_id": task["active_target_id"],
                "target_class": task["target_class"],
                "target_color": task["target_color"],
                "target_shape": task["target_shape"],
                "target_type": episode_meta["target_type"],
                "target_label": episode_meta["target_label"],
                "target_description": episode_meta["target_description"],
                "scene_id": task["scene_id"],
                "scene_kind": task["scene_kind"],
                "scene_mode": task["scene_mode"],
                "layout_id": task["layout_id"],
                "layout_template_id": task["layout_template_id"],
                "layout_group_id": task["layout_group_id"],
                "contrast_group_id": task.get("contrast_group_id", ""),
                "contrast_variant": task.get("contrast_variant", ""),
                "turn_bucket": task["turn_bucket"],
                "visibility_bucket": task["visibility_bucket"],
                "split_hint": task["split_hint"],
                "seed": int(runtime_cfg.get("seed", 0)),
                "operator_id": episode_meta["operator_id"],
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "termination_reason": termination_reason,
                "phase": phase,
                "robot_pose": {
                    "x": float(debug_state["robot_x"]),
                    "y": float(debug_state["robot_y"]),
                    "z": float(debug_state["robot_z"]),
                    "yaw": float(debug_state["yaw"]),
                },
                "target_pose": {
                    "x": float(debug_state["goal_x"]),
                    "y": float(debug_state["goal_y"]),
                    "z": float(debug_state["goal_z"]),
                },
                "dr_params": dict(config.get("domain_randomization", {})),
                "debug": debug_payload,
            }

        while total_collected < requested_episodes:
            observation = env.reset()
            controller.reset()
            if env.episode_id in completed_ids:
                log_progress(f"skip existing episode={env.episode_id}")
                continue
            task = observation["task"]
            episode_meta = {
                "episode_id": env.episode_id,
                "task_id": task["task_id"],
                "instruction": observation["instruction"],
                "instruction_source": "auto_template",
                "instruction_template_id": task["instruction_template_id"],
                "active_target_id": task["active_target_id"],
                "target_class": task["target_class"],
                "target_color": task["target_color"],
                "target_shape": task["target_shape"],
                "target_type": "box" if str(task["target_shape"]).lower() == "box" else str(task["target_shape"]).lower(),
                "target_label": str(task.get("target_label") or task["target_class"]).replace("_", " "),
                "target_description": str(task.get("target_description") or task["target_class"]).replace("_", " "),
                "scene_id": task["scene_id"],
                "scene_kind": task["scene_kind"],
                "scene_mode": task["scene_mode"],
                "layout_id": task["layout_id"],
                "layout_template_id": task["layout_template_id"],
                "layout_group_id": task["layout_group_id"],
                "contrast_group_id": task.get("contrast_group_id", ""),
                "contrast_variant": task.get("contrast_variant", ""),
                "turn_bucket": task["turn_bucket"],
                "visibility_bucket": task["visibility_bucket"],
                "split_hint": task["split_hint"],
                "target_visible_first_frame": bool(task["target_visible_first_frame"]),
                "target_visible_within_3f": bool(task["target_visible_within_3f"]),
                "target_visible_within_10f": bool(task["target_visible_within_10f"]),
                "target_pixel_ratio_first": float(task["target_pixel_ratio_first"]),
                "scene_targets": list(task.get("scene_targets", [])),
                "seed": int(runtime_cfg.get("seed", 0)),
                "control_dt": float(config.get("sim", {}).get("control_dt", 0.05)),
                "operator_id": str(runtime_cfg.get("operator_id", "isaac_sim")),
                "state_fields": list(config.get("data", {}).get("state_fields", ["vx", "vy", "vz"])),
                "train_action_fields": list(config.get("data", {}).get("train_action_fields", ["vx", "wz"])),
                "raw_action_fields": list(config.get("data", {}).get("raw_action_fields", ["vx", "vy", "wz"])),
            }
            logger.start_episode(episode_meta)
            done = False
            final_info: dict[str, Any] = {}
            step_id = 0
            for _ in range(max(0, pre_roll_frames)):
                debug_state = observation["debug_state"]
                command = zero_command()
                step_meta = build_step_meta(
                    task=task,
                    episode_meta=episode_meta,
                    debug_state=debug_state,
                    step_id=step_id,
                    phase="pre_roll",
                    extra_debug={"expert": command.get("expert_debug", {})},
                )
                logger.record_step(observation, command, step_meta)
                observation, _, done, final_info = env.step(command["raw_cmd_train"])
                step_id += 1
                if done:
                    break
            while not done:
                command = controller.compute_command(observation)
                debug_state = observation["debug_state"]
                step_meta = build_step_meta(
                    task=task,
                    episode_meta=episode_meta,
                    debug_state=debug_state,
                    step_id=step_id,
                    phase="policy",
                    extra_debug={"expert": command.get("expert_debug", {})},
                )
                logger.record_step(observation, command, step_meta)
                observation, _, done, final_info = env.step(command["raw_cmd_train"])
                step_id += 1
            if bool(final_info.get("success", False)) and post_success_settle_frames > 0:
                settle_observation = observation
                for settle_index in range(post_success_settle_frames):
                    command = zero_command()
                    settle_observation, _, _, settle_info = env.step(command["raw_cmd_train"])
                    settle_debug_state = settle_observation["debug_state"]
                    settle_step_meta = build_step_meta(
                        task=task,
                        episode_meta=episode_meta,
                        debug_state=settle_debug_state,
                        step_id=step_id,
                        phase="post_success_settle",
                        success=(settle_index == post_success_settle_frames - 1),
                        termination_reason="goal_reached" if settle_index == post_success_settle_frames - 1 else "settling_after_success",
                        extra_debug={"expert": command.get("expert_debug", {}), "settle_index": settle_index + 1},
                    )
                    logger.record_step(settle_observation, command, settle_step_meta)
                    observation = settle_observation
                    final_info = dict(final_info)
                    final_info["goal_distance"] = float(settle_info.get("goal_distance", final_info.get("goal_distance", 0.0)))
                    final_info["success"] = True
                    step_id += 1
            summary = {
                "success": bool(final_info.get("success", False)),
                "collision": bool(final_info.get("collision", False)),
                "timeout": bool(final_info.get("timeout", False)),
                "out_of_bounds": bool(final_info.get("out_of_bounds", False)),
                "entered_forbidden_zone": bool(final_info.get("entered_forbidden_zone", False)),
                "goal_distance": float(final_info.get("goal_distance", 0.0)),
                "scene_id": str(task["scene_id"]),
                "scene_kind": str(task["scene_kind"]),
                "scene_mode": str(task["scene_mode"]),
                "layout_id": str(task["layout_id"]),
                "layout_template_id": str(task["layout_template_id"]),
                "layout_group_id": str(task["layout_group_id"]),
                "active_target_id": str(task["active_target_id"]),
                "contrast_group_id": str(task.get("contrast_group_id", "")),
                "contrast_variant": str(task.get("contrast_variant", "")),
                "turn_bucket": str(task["turn_bucket"]),
                "visibility_bucket": str(task["visibility_bucket"]),
                "target_visible_first_frame": bool(task["target_visible_first_frame"]),
                "target_visible_within_3f": bool(task["target_visible_within_3f"]),
                "target_visible_within_10f": bool(task["target_visible_within_10f"]),
                "target_pixel_ratio_first": float(task["target_pixel_ratio_first"]),
            }
            logger.end_episode(summary)
            episode_summaries.append({"episode_id": env.episode_id, "steps": step_id, **summary})
            total_collected += 1
            log_progress(
                "episode="
                f"{env.episode_id} steps={step_id} success={summary['success']} "
                f"turn_bucket={summary['turn_bucket']} visibility_bucket={summary['visibility_bucket']} "
                f"visible3={summary['target_visible_within_3f']}"
            )

        collection_summary = {
            "dataset_name": str(runtime_cfg.get("dataset_name", "go2_isaac_collection")),
            "episodes_requested": requested_episodes,
            "episodes_completed": len(episode_summaries),
            "raw_output_dir": str(raw_output_dir),
            "episodes": episode_summaries,
        }
        dump_json(raw_output_dir / "collection_summary.json", collection_summary)
        if bool(runtime_cfg.get("auto_pack", False)):
            log_progress("auto_pack requested, but legacy packing/pipeline export has been removed; keeping raw session only")
    except BaseException as exc:
        with error_path.open("w", encoding="utf-8") as handle:
            handle.write("Collector failed with an exception.\n")
            handle.write(f"Type: {type(exc).__name__}\n")
            handle.write(f"Message: {exc}\n\n")
            handle.write(traceback.format_exc())
        print(f"[ERROR] collector failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print(f"[ERROR] traceback written to {error_path}", file=sys.stderr, flush=True)
        raise
    finally:
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
