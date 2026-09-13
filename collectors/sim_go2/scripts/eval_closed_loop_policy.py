#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

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
        Path("/home/wxh/IsaacLab"),
        Path("/home/wxh/IsaacLab/source"),
        Path("/home/wxh/IsaacLab/source/isaaclab"),
        Path("/home/wxh/unitree_rl_lab"),
        Path("/home/wxh/unitree_rl_lab/source"),
        Path("/home/wxh/unitree_rl_lab/source/unitree_rl_lab"),
        Path("/root/autodl-tmp/IsaacLab"),
        Path("/root/autodl-tmp/IsaacLab/source"),
        Path("/root/autodl-tmp/IsaacLab/source/isaaclab"),
        Path("/root/autodl-tmp/unitree_rl_lab"),
        Path("/root/autodl-tmp/unitree_rl_lab/source"),
        Path("/root/autodl-tmp/unitree_rl_lab/source/unitree_rl_lab"),
        Path("/home/zxq/zxq/IsaacLab"),
        Path("/home/zxq/zxq/IsaacLab/source"),
        Path("/home/zxq/zxq/IsaacLab/source/isaaclab"),
        Path("/home/zxq/zxq/unitree_rl_lab"),
        Path("/home/zxq/zxq/unitree_rl_lab/source"),
        Path("/home/zxq/zxq/unitree_rl_lab/source/unitree_rl_lab"),
    ]
    for env_name in ("ISAACLAB_ROOT", "UNITREE_RL_LAB_ROOT"):
        root = _env_root(env_name)
        if root is None:
            continue
        candidates.extend([root, root / "source", root / "source" / root.name])
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
        try:
            exists = candidate.exists()
        except OSError:
            exists = False
        if exists and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_bootstrap_paths()

import numpy as np
import torch

from collectors.sim_go2.recorders.image_writer import to_uint8_rgb
from collectors.sim_go2.utils.io import apply_cli_overrides, dump_json, load_resolved_config
from deploy.go2.adapters.base import BaseRobotAdapter, RobotObservation
from deploy.go2.controller_loop import Go2ControllerLoop
from deploy.go2.go2_state_reader import Go2StateReader
from deploy.go2.go2_velocity_commander import Go2VelocityCommander
from deploy.go2.safety_filter import VelocitySafetyFilter
from robotics.inference.predict import load_model


class SimNavAdapter(BaseRobotAdapter):
    def __init__(self, env):
        self.env = env
        self.observation: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None
        self.last_model_debug: dict[str, Any] | None = None
        self.done = False
        self.command_history: list[dict[str, float]] = []
        self.trace_records: list[dict[str, Any]] = []

    def reset(self) -> RobotObservation:
        self.observation = self.env.reset()
        self.last_info = None
        self.done = False
        self.command_history = []
        self.trace_records = []
        self.last_model_debug = None
        self._record_trace(command=None, info=None, phase="reset")
        return self.read_observation()

    def read_observation(self) -> RobotObservation:
        if self.observation is None:
            raise RuntimeError("Adapter must be reset before reading observations.")
        state = self._build_state(self.observation)
        image = self._build_image(self.observation["image"])
        return RobotObservation(state=state, image=image, timestamp=time.time())

    def send_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        if self.observation is None:
            raise RuntimeError("Adapter must be reset before sending commands.")
        next_obs, _reward, done, info = self.env.step([float(vx), float(wz)])
        self.observation = next_obs
        self.last_info = info
        self.done = bool(done)
        command = {"vx": float(vx), "vy": float(vy), "wz": float(wz)}
        self.command_history.append(command)
        self._record_trace(command=command, info=info, phase="step")

    def set_model_debug(self, debug_payload: dict[str, Any]) -> None:
        self.last_model_debug = copy.deepcopy(debug_payload)

    @staticmethod
    def _build_image(image: Any) -> Image.Image:
        return Image.fromarray(to_uint8_rgb(image), mode="RGB")

    @staticmethod
    def _build_state_from_observation(observation: dict[str, Any], previous_command: dict[str, float] | None) -> dict[str, float]:
        debug_state = observation.get("debug_state") or {}
        state = observation.get("state")
        if isinstance(state, dict):
            state_dict = dict(state)
        elif isinstance(state, (list, tuple)) and len(state) >= 3:
            state_dict = {
                "vx": float(state[0]),
                "vy": float(state[1]),
                "vz": float(state[2]),
            }
        else:
            state_dict = {}
        state_dict.setdefault("vx", float(debug_state.get("vx", 0.0)))
        state_dict.setdefault("vy", float(debug_state.get("vy", 0.0)))
        state_dict.setdefault("vz", float(debug_state.get("vz", 0.0)))
        state_dict["yaw"] = float(debug_state.get("yaw", 0.0))
        state_dict["yaw_speed"] = float(debug_state.get("yaw_rate", debug_state.get("wz", 0.0)))
        state_dict["vx_prev"] = float(previous_command["vx"]) if previous_command is not None else 0.0
        state_dict["vy_prev"] = float(previous_command["vy"]) if previous_command is not None else 0.0
        state_dict["wz_prev"] = float(previous_command["wz"]) if previous_command is not None else 0.0
        state_dict["mode"] = 3.0
        state_dict["gait_type"] = 1.0
        return state_dict

    def _build_state(self, observation: dict[str, Any]) -> dict[str, float]:
        previous_command = self.command_history[-1] if self.command_history else None
        return self._build_state_from_observation(observation, previous_command)

    def _record_trace(self, command: dict[str, float] | None, info: dict[str, Any] | None, phase: str) -> None:
        if self.observation is None:
            return
        debug_state = self.observation.get("debug_state") or {}
        task = self.observation.get("task") or {}
        record = {
            "phase": phase,
            "step_index": int((info or {}).get("step_index", len(self.trace_records))),
            "instruction": str(self.observation.get("instruction", "")),
            "command": command,
            "model_debug": copy.deepcopy(self.last_model_debug),
            "robot_x": float(debug_state.get("robot_x", 0.0)),
            "robot_y": float(debug_state.get("robot_y", 0.0)),
            "robot_z": float(debug_state.get("robot_z", 0.0)),
            "yaw": float(debug_state.get("yaw", 0.0)),
            "vx": float(debug_state.get("vx", 0.0)),
            "vy": float(debug_state.get("vy", 0.0)),
            "yaw_rate": float(debug_state.get("yaw_rate", 0.0)),
            "goal_x": float(debug_state.get("goal_x", 0.0)),
            "goal_y": float(debug_state.get("goal_y", 0.0)),
            "goal_z": float(debug_state.get("goal_z", 0.0)),
            "goal_distance": float((info or {}).get("goal_distance", debug_state.get("goal_distance", 0.0))),
            "goal_clearance": float((info or {}).get("goal_clearance", 0.0)),
            "goal_heading": float((info or {}).get("goal_heading", debug_state.get("goal_heading", 0.0))),
            "target_visible": bool(debug_state.get("target_visible", False)),
            "target_pixel_ratio": float(debug_state.get("target_pixel_ratio", 0.0)),
            "active_target_id": task.get("active_target_id"),
            "turn_bucket": task.get("turn_bucket"),
            "visibility_bucket": task.get("visibility_bucket"),
            "done": bool((info or {}).get("success", False) or (info or {}).get("timeout", False) or (info or {}).get("collision", False)),
            "success": bool((info or {}).get("success", False)),
            "collision": bool((info or {}).get("collision", False)),
            "timeout": bool((info or {}).get("timeout", False)),
            "out_of_bounds": bool((info or {}).get("out_of_bounds", False)),
            "entered_forbidden_zone": bool((info or {}).get("entered_forbidden_zone", False)),
        }
        self.trace_records.append(record)


def _summarize_episode(index: int, adapter: SimNavAdapter, instruction: str) -> dict[str, Any]:
    info = dict(adapter.last_info or {})
    command_count = len(adapter.command_history)
    predict_ms = []
    safety_ms = []
    control_loop_ms = []
    for row in adapter.trace_records:
        debug = row.get("model_debug") or {}
        if not isinstance(debug, dict):
            continue
        if "predict_ms" in debug:
            predict_ms.append(float(debug["predict_ms"]))
        if "safety_ms" in debug:
            safety_ms.append(float(debug["safety_ms"]))
        if "control_loop_ms" in debug:
            control_loop_ms.append(float(debug["control_loop_ms"]))

    def _stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    summary = {
        "episode_index": index,
        "instruction": instruction,
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(info.get("timeout", False)),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "entered_forbidden_zone": bool(info.get("entered_forbidden_zone", False)),
        "goal_distance": float(info.get("goal_distance", 0.0)),
        "goal_clearance": float(info.get("goal_clearance", 0.0)),
        "steps": int(info.get("step_index", command_count)),
        "command_count": command_count,
        "active_target_id": info.get("active_target_id"),
        "turn_bucket": info.get("turn_bucket"),
        "visibility_bucket": info.get("visibility_bucket"),
        "scene_id": info.get("scene_id"),
        "layout_id": info.get("layout_id"),
        "predict_ms": _stats(predict_ms),
        "safety_ms": _stats(safety_ms),
        "control_loop_ms": _stats(control_loop_ms),
    }
    return summary


def _write_trace_svg(trace_records: list[dict[str, Any]], output_path: Path) -> None:
    coords = []
    for row in trace_records:
        coords.append((float(row["robot_x"]), float(row["robot_y"])))
        coords.append((float(row["goal_x"]), float(row["goal_y"])))
    if not coords:
        return
    min_x = min(x for x, _ in coords)
    max_x = max(x for x, _ in coords)
    min_y = min(y for _, y in coords)
    max_y = max(y for _, y in coords)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    width = 720
    height = 720
    pad = 40

    def proj(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = pad + (x - min_x) / span_x * (width - 2 * pad)
        py = height - pad - (y - min_y) / span_y * (height - 2 * pad)
        return px, py

    robot_points = [proj((float(row["robot_x"]), float(row["robot_y"]))) for row in trace_records]
    goal_point = proj((float(trace_records[-1]["goal_x"]), float(trace_records[-1]["goal_y"])))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in robot_points)
    start_x, start_y = robot_points[0]
    end_x, end_y = robot_points[-1]
    goal_x, goal_y = goal_point
    labels = []
    for idx, (x, y) in enumerate(robot_points[:: max(1, len(robot_points) // 8 or 1)]):
        labels.append(f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="12" fill="#444">{idx}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#d0d7de"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{polyline}"/>
<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="7" fill="#2ca02c"/>
<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="7" fill="#d62728"/>
<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="8" fill="#ff7f0e"/>
<text x="{start_x + 10:.1f}" y="{start_y:.1f}" font-size="13" fill="#2ca02c">start</text>
<text x="{end_x + 10:.1f}" y="{end_y:.1f}" font-size="13" fill="#d62728">end</text>
<text x="{goal_x + 10:.1f}" y="{goal_y:.1f}" font-size="13" fill="#ff7f0e">goal</text>
{''.join(labels)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop Isaac Sim rollout for a velocity-control checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backbone-model-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sim-device", default=None)
    parser.add_argument("--execute-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_resolved_config(args.config)
    overrides: dict[str, Any] = {
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "headless": args.headless,
        "max_steps_per_episode": args.max_steps_per_episode,
    }
    config = apply_cli_overrides(config, overrides)
    if args.sim_device is not None:
        config.setdefault("sim", {})["device"] = args.sim_device

    tokenizer, image_processor, model, runtime_device = load_model(
        args.model_path,
        device=args.device,
        backbone_model_path=args.backbone_model_path,
    )
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(config.get("runtime", {}).get("headless", True)), enable_cameras=True)
    _simulation_app = app_launcher.app
    from collectors.sim_go2.envs.go2_nav_env import Go2NavCollectionEnv

    env: Go2NavCollectionEnv | None = None
    adapter: SimNavAdapter | None = None
    summaries: list[dict[str, Any]] = []
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = Go2NavCollectionEnv(config)
        adapter = SimNavAdapter(env)
        for episode_idx in range(args.num_episodes):
            first_observation = adapter.reset()
            instruction = str(adapter.observation.get("instruction", "go to the target"))  # type: ignore[union-attr]
            loop = Go2ControllerLoop(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                reader=Go2StateReader(adapter),
                commander=Go2VelocityCommander(adapter),
                safety_filter=VelocitySafetyFilter(model.velocity_config),
                instruction=instruction,
                execute_steps=args.execute_steps,
            )
            del first_observation
            max_steps = int(config.get("runtime", {}).get("max_steps_per_episode", 180))
            for _ in range(max_steps):
                loop.run_once()
                if adapter.done:
                    break
            episode_summary = _summarize_episode(episode_idx, adapter, instruction)
            trace_payload = {
                "summary": episode_summary,
                "trace": adapter.trace_records,
            }
            dump_json(episodes_dir / f"episode_{episode_idx:03d}.json", trace_payload)
            _write_trace_svg(adapter.trace_records, episodes_dir / f"episode_{episode_idx:03d}.svg")
            summaries.append(episode_summary)
            print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
        success_count = sum(1 for item in summaries if item["success"])
        collision_count = sum(1 for item in summaries if item["collision"])
        timeout_count = sum(1 for item in summaries if item["timeout"])
        result = {
            "model_path": args.model_path,
            "backbone_model_path": args.backbone_model_path,
            "config_path": str(args.config.resolve()),
            "runtime_device": runtime_device,
            "sim_device": config.get("sim", {}).get("device"),
            "num_episodes": len(summaries),
            "success_count": success_count,
            "success_rate": float(success_count / max(len(summaries), 1)),
            "collision_count": collision_count,
            "timeout_count": timeout_count,
            "mean_goal_clearance": float(sum(item["goal_clearance"] for item in summaries) / max(len(summaries), 1)),
            "mean_steps": float(sum(item["steps"] for item in summaries) / max(len(summaries), 1)),
            "mean_predict_ms": float(
                sum(item["predict_ms"]["mean"] for item in summaries if item.get("predict_ms")) / max(sum(1 for item in summaries if item.get("predict_ms")), 1)
            ),
            "mean_control_loop_ms": float(
                sum(item["control_loop_ms"]["mean"] for item in summaries if item.get("control_loop_ms"))
                / max(sum(1 for item in summaries if item.get("control_loop_ms")), 1)
            ),
            "episodes": summaries,
        }
        dump_json(output_dir / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if env is not None:
            env.close()
        _simulation_app.close()


if __name__ == "__main__":
    main()
