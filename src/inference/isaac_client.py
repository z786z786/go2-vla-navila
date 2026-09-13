#!/usr/bin/env python3
"""Run an independent SmolVLA closed loop in the official Go2 VLN environment."""

from __future__ import annotations

import argparse
import builtins
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.inference.action_audit import summarize_action_range
from src.inference.protocol import PROTOCOL_VERSION, validate_request_header, unix_request
from src.inference.state import (
    ActionSafetyResult,
    apply_action_safety,
    build_policy_state,
    validate_action_chunk,
)


DEFAULT_SHORT_DATASET = Path("<external-workspace>")
DEFAULT_ASSET_ROOT = Path("<external-data-root>")
DEFAULT_NAVILA_ROOT = Path("<external-data-root>")
DEFAULT_SOCKET = "/tmp/go2_smolvla_m7.sock"
EXECUTE_STEPS = 10
MAX_STEPS = 1500
SUCCESS_STREAK_REQUIRED = 10
JPEG_QUALITY = 90
SUCCESS_RADIUS_M = 0.5


# Isaac Kit owns this subscription.  Keeping the handle process-global prevents
# garbage collection from silently disabling the diagnostic callback.
timeline_stop_subscription: Any | None = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_policy_request(
    *,
    state: Sequence[float],
    jpeg: bytes,
    episode_id: str,
    instruction: str,
    replan_index: int,
) -> dict[str, Any]:
    header = {
        "version": PROTOCOL_VERSION,
        "request_id": f"{episode_id}:{replan_index}",
        "episode_id": episode_id,
        "replan_index": replan_index,
        "instruction": instruction,
        "state": list(state),
        "image_encoding": "jpeg",
        "image_height": 512,
        "image_width": 512,
        "image_channels": 3,
        "jpeg_size": len(jpeg),
    }
    validate_request_header(header, payload_size=len(jpeg))
    return header


def select_executed_actions(
    actions: Sequence[Sequence[float]], *, execute_steps: int = EXECUTE_STEPS
) -> list[ActionSafetyResult]:
    chunk = validate_action_chunk(actions)
    if isinstance(execute_steps, bool) or not 1 <= execute_steps <= len(chunk):
        raise ValueError(f"execute_steps must be in [1, {len(chunk)}]")
    return [apply_action_safety(action) for action in chunk[:execute_steps]]


def advance_success_streak(current: int, distance: float, *, radius: float) -> int:
    if current < 0 or not math.isfinite(distance) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("invalid success streak input")
    return current + 1 if distance < radius else 0


def make_official_episode(short_episode: dict[str, Any]) -> dict[str, Any]:
    reference_path = short_episode["reference_path"]
    return {
        "episode_id": short_episode["source_episode_id"],
        "trajectory_id": short_episode["source_trajectory_id"],
        "scene_id": short_episode["scene_id"],
        "start_position": short_episode["start_pose"]["position"],
        "start_rotation": short_episode["start_pose"]["rotation_wxyz"],
        "info": {"geodesic_distance": short_episode["path_length"]},
        "goals": [
            {
                "position": short_episode["goal_pose"]["position"],
                "radius": short_episode["goal_pose"]["success_radius_m"],
            }
        ],
        "instruction": {
            "instruction_text": short_episode["instruction"],
            "instruction_tokens": [0] * 200,
        },
        "reference_path": reference_path,
        "episode_new_id": short_episode["source_episode_new_id"],
        "gt_locations": reference_path,
        "gt_actions": [],
        "gt_forward_steps": len(reference_path) - 1,
    }


def jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def evaluator_distance_from_measurements(measurements: Any) -> float:
    """Read the success distance exclusively from the official VLN evaluator."""
    if not isinstance(measurements, dict) or "distance_to_goal" not in measurements:
        raise ValueError("official evaluator measurements must contain distance_to_goal")
    try:
        distance = float(jsonable(measurements["distance_to_goal"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("official evaluator distance_to_goal must be finite") from exc
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("official evaluator distance_to_goal must be finite and non-negative")
    return distance


def percentile_95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def add_official_runner_compat_args(parser: argparse.ArgumentParser) -> None:
    """Mirror the two NaVILA demo flags read by ``parse_rsl_rl_cfg``.

    These controls configure only the frozen low-level locomotion runner.
    They are not supplied to, or serialized in, the SmolVLA request.
    """
    parser.add_argument("--use_cnn", action="store_true", default=None)
    parser.add_argument("--use_rnn", action="store_true", default=False)


def parse_args() -> argparse.Namespace:
    navila_scripts = DEFAULT_NAVILA_ROOT / "scripts"
    sys.path.insert(0, str(navila_scripts))
    import cli_args
    from omni.isaac.lab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--short-episode-id")
    selection.add_argument("--short-index", type=int)
    parser.add_argument("--short-dataset", type=Path, default=DEFAULT_SHORT_DATASET)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--execute-steps", type=int, default=EXECUTE_STEPS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--success-streak", type=int, default=SUCCESS_STREAK_REQUIRED)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--navila-root", type=Path, default=DEFAULT_NAVILA_ROOT)
    parser.add_argument("--task", default="go2_matterport_vision")
    parser.add_argument("--num_envs", type=int, default=1)
    # The official NaVILA RSL-RL configuration reads this field even when its
    # value is ``None``.  It is simulator setup only and never crosses the
    # policy socket boundary.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--history_length", type=int, default=9)
    add_official_runner_compat_args(parser)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def select_short_episode(dataset_path: Path, episode_id: str | None, index: int | None) -> dict[str, Any]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    if index is not None:
        return episodes[index]
    matches = [episode for episode in episodes if episode["short_episode_id"] == episode_id]
    if len(matches) != 1:
        raise KeyError(f"expected one short episode {episode_id!r}, found {len(matches)}")
    return matches[0]


def configure_app_launcher(asset_root: Path) -> None:
    from omni.isaac.lab.app import AppLauncher

    original_init = AppLauncher.__init__
    kit_asset_args = (
        f"--/persistent/isaac/asset_root/default={asset_root}",
        f"--/persistent/isaac/asset_root/cloud={asset_root}",
        f"--/persistent/isaac/asset_root/nvidia={asset_root}",
        "--/persistent/isaac/asset_root/timeout=1.0",
    )

    def init_with_local_assets(self: Any, *args: Any, **kwargs: Any) -> None:
        for argument in kit_asset_args:
            prefix = argument.split("=", 1)[0] + "="
            if not any(item.startswith(prefix) for item in sys.argv):
                sys.argv.append(argument)
        original_init(self, *args, **kwargs)

        # This ordering follows the successful M4 collector.  The camera
        # experience starts a finite two-second Kit timeline while the scene
        # is being materialized.  Install the reset hook immediately after
        # AppLauncher is live, before task configuration can construct the
        # first SimulationContext.
        import gymnasium as gym
        import omni.timeline
        import omni.usd
        from omni.isaac.lab.sim import SimulationContext

        local_go2 = configure_runtime_assets(asset_root)
        original_gym_make = gym.make

        def make_with_local_go2(*make_args: Any, **make_kwargs: Any) -> Any:
            env_cfg = make_kwargs.get("cfg")
            robot_spawn = getattr(
                getattr(getattr(env_cfg, "scene", None), "robot", None),
                "spawn",
                None,
            )
            if robot_spawn is not None and hasattr(robot_spawn, "usd_path"):
                previous = robot_spawn.usd_path
                robot_spawn.usd_path = str(local_go2)
                print(f"M7 final task Go2 USD: {previous} -> {robot_spawn.usd_path}", flush=True)
                if robot_spawn.usd_path != str(local_go2):
                    raise RuntimeError("failed to pin final task Go2 USD")
            return original_gym_make(*make_args, **make_kwargs)

        gym.make = make_with_local_go2
        original_reset = SimulationContext.reset
        timeline_extended = False

        def reset_with_long_timeline(context: Any, *reset_args: Any, **reset_kwargs: Any) -> Any:
            nonlocal timeline_extended
            if not timeline_extended:
                timeline = omni.timeline.get_timeline_interface()
                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    raise RuntimeError("Kit stage is unavailable while extending the M7 timeline")
                before_end = timeline.get_end_time()
                before_stage_end = stage.GetEndTimeCode()
                stage.SetEndTimeCode(10000.0 * stage.GetTimeCodesPerSecond())
                timeline.set_end_time(10000.0)
                timeline.commit()
                timeline_extended = True
                print(
                    "M7 process-local Kit timeline extended: "
                    f"timeline {before_end} -> {timeline.get_end_time()} seconds; "
                    f"stage {before_stage_end} -> {stage.GetEndTimeCode()} time codes",
                    flush=True,
                )
            return original_reset(context, *reset_args, **reset_kwargs)

        SimulationContext.reset = reset_with_long_timeline

        global timeline_stop_subscription

        def log_timeline_stop(event: Any) -> None:
            timeline = omni.timeline.get_timeline_interface()
            stage = omni.usd.get_context().get_stage()
            stage_end = stage.GetEndTimeCode() if stage is not None else "none"
            print(
                "M7 timeline STOP observed: "
                f"current={timeline.get_current_time()} end={timeline.get_end_time()} "
                f"stage_end={stage_end}",
                flush=True,
            )

        timeline_stop_subscription = (
            omni.timeline.get_timeline_interface()
            .get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), log_timeline_stop, order=0
            )
        )

    AppLauncher.__init__ = init_with_local_assets


def configure_runtime_assets(asset_root: Path) -> Path:
    import carb

    settings = carb.settings.get_settings()
    settings.set_string("/persistent/isaac/asset_root/default", str(asset_root))
    settings.set_string("/persistent/isaac/asset_root/cloud", str(asset_root))
    settings.set_float("/persistent/isaac/asset_root/timeout", 30.0)
    settings.set_bool("/app/hangDetector/enabled", False)
    local_go2 = asset_root / "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
    if not local_go2.is_file():
        raise FileNotFoundError(local_go2)
    from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG

    UNITREE_GO2_CFG.spawn.usd_path = str(local_go2)
    return local_go2


def configure_env_episode(env_cfg: Any, episode: dict[str, Any], local_go2: Path) -> None:
    import numpy as np

    start = episode["start_position"]
    goal = episode["goals"][0]["position"]
    env_cfg.scene.robot.init_state.pos = (start[0], start[1], start[2] + 0.4)
    env_cfg.scene.robot.init_state.rot = tuple(episode["start_rotation"])
    env_cfg.scene.robot.spawn.usd_path = str(local_go2)
    env_cfg.scene.disk_1.init_state.pos = (start[0], start[1], start[2] + 2.5)
    env_cfg.scene.disk_2.init_state.pos = (goal[0], goal[1], goal[2] + 2.5)
    env_cfg.goals = episode["goals"]
    env_cfg.episode_id = episode["episode_id"]
    env_cfg.scene_id = episode["scene_id"].split("/")[1]
    env_cfg.traj_id = episode["trajectory_id"]
    env_cfg.instruction_text = episode["instruction"]["instruction_text"]
    env_cfg.instruction_tokens = episode["instruction"]["instruction_tokens"]
    env_cfg.reference_path = np.asarray(episode["reference_path"])
    env_cfg.expert_path = np.asarray(episode["gt_locations"])
    env_cfg.expert_path_length = len(env_cfg.expert_path)
    env_cfg.expert_time = np.arange(env_cfg.expert_path_length)


def capture_live_state(wrapper: Any) -> tuple[list[float], dict[str, Any]]:
    import omni.isaac.lab.utils.math as math_utils

    robot = wrapper.unwrapped.scene["robot"]
    data = robot.data
    quaternion = data.root_quat_w[0]
    linear_body = getattr(data, "root_lin_vel_b", None)
    if linear_body is None:
        linear_body = math_utils.quat_rotate_inverse(quaternion, data.root_lin_vel_w[0])
    else:
        linear_body = linear_body[0]
    angular_body = getattr(data, "root_ang_vel_b", None)
    if angular_body is None:
        angular_body = math_utils.quat_rotate_inverse(quaternion, data.root_ang_vel_w[0])
    else:
        angular_body = angular_body[0]
    state = build_policy_state(
        jsonable(linear_body),
        jsonable(angular_body),
        jsonable(quaternion),
        jsonable(data.joint_pos[0]),
        jsonable(data.default_joint_pos[0]),
        jsonable(data.joint_vel[0]),
    )
    audit = {
        "robot_pose": {
            "position_w": jsonable(data.root_pos_w[0]),
            "quaternion_wxyz": jsonable(quaternion),
        },
        "linear_velocity_body": jsonable(linear_body),
        "angular_velocity_body": jsonable(angular_body),
        "joint_position": jsonable(data.joint_pos[0]),
        "joint_velocity": jsonable(data.joint_vel[0]),
    }
    return state, audit


def rotated_rgb(observation: Any) -> Any:
    import cv2
    import numpy as np

    frame = jsonable(observation[0, :, :, :3])
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    rotated = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
    if rotated.shape != (512, 512, 3):
        raise ValueError(f"unexpected RGB shape {rotated.shape}")
    return rotated


def encode_jpeg(rgb: Any, quality: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("failed to encode RGB frame as JPEG")
    return encoded.tobytes()


def termination_terms(wrapper: Any) -> dict[str, bool]:
    manager = wrapper.unwrapped.termination_manager
    return {name: bool(manager.get_term(name)[0].item()) for name in manager.active_terms}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def finalize_summary(
    *,
    episode_dir: Path,
    short_episode: dict[str, Any],
    status: str,
    termination_reason: str,
    records: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    started_wall: float,
    error: str | None = None,
) -> dict[str, Any]:
    latencies = [float(row["roundtrip_wall_ms"]) for row in requests if row.get("status") == "ok"]
    response_chunks = [
        row["response"]["actions"]
        for row in requests
        if isinstance(row.get("response"), dict) and isinstance(row["response"].get("actions"), list)
    ]
    range_summary = summarize_action_range(response_chunks, execute_steps=EXECUTE_STEPS)
    last = records[-1] if records else {}
    measurements = last.get("measurements", {})
    success = bool(float(measurements.get("success", 0.0)) >= 1.0)
    collision = any(bool(row.get("termination_terms", {}).get("base_contact")) for row in records)
    final_error = last.get("next_evaluator_distance_to_goal_m")
    summary = {
        "format": "go2-short-vln-m7-episode-v1",
        "status": status,
        "passed": bool(status == "complete" and success and range_summary.raw_output_range_passed),
        "termination_reason": termination_reason,
        "error": error,
        "short_episode_id": short_episode["short_episode_id"],
        "source_episode_id": short_episode["source_episode_id"],
        "split": short_episode["split"],
        "scene_id": short_episode["scene_id"],
        "instruction": short_episode["instruction"],
        "goal_pose": short_episode["goal_pose"],
        "frame_count": len(records),
        "request_count": len(requests),
        "execute_steps_per_chunk": EXECUTE_STEPS,
        "success": success,
        "final_navigation_error_m": final_error,
        "collision": collision,
        "bad_orientation": any(
            bool(row.get("termination_terms", {}).get("bad_orientation")) for row in records
        ),
        **range_summary.as_dict(),
        # Kept for older evidence readers. The historical name implied chunks,
        # but it has always been a sum of action-vector violations.
        "raw_chunk_range_violation_count": range_summary.violating_vectors,
        "executed_action_range_violation_count": range_summary.executed_violating_vectors,
        "deprecated_compatibility_fields": {
            "raw_chunk_range_violation_count": "alias of raw_action_vector_violation_count",
            "executed_action_range_violation_count": "alias of executed_policy_action_violation_count",
        },
        "mean_inference_roundtrip_ms": statistics.fmean(latencies) if latencies else None,
        "p95_inference_roundtrip_ms": percentile_95(latencies),
        "max_inference_roundtrip_ms": max(latencies) if latencies else None,
        "elapsed_wall_seconds": time.monotonic() - started_wall,
        "measurements": measurements,
        "created_at_utc": now_utc(),
    }
    write_json(episode_dir / "summary.json", summary)
    return summary


def finalize_startup_failure(
    *,
    episode_dir: Path,
    short_episode: dict[str, Any],
    started_wall: float,
    error: Exception,
) -> dict[str, Any]:
    """Persist a checkable rollout failure even if the first reset never ran."""
    for filename in ("steps.jsonl", "requests.jsonl"):
        (episode_dir / filename).touch(exist_ok=True)
    return finalize_summary(
        episode_dir=episode_dir,
        short_episode=short_episode,
        status="initialization_error",
        termination_reason="startup_error",
        records=[],
        requests=[],
        started_wall=started_wall,
        error=f"{type(error).__name__}: {error}",
    )


def run_loop(
    wrapper: Any,
    observation: Any,
    short_episode: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    episode_dir = args.episode_dir
    rgb_dir = episode_dir / "front_rgb"
    rgb_dir.mkdir()
    steps_stream = (episode_dir / "steps.jsonl").open("w", encoding="utf-8", buffering=1)
    requests_stream = (episode_dir / "requests.jsonl").open("w", encoding="utf-8", buffering=1)
    records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    action_queue: list[ActionSafetyResult] = []
    queue_index = 0
    replan_index = 0
    success_streak = 0
    started_wall = time.monotonic()
    status = "incomplete"
    reason = "unknown"
    failure: str | None = None

    try:
        for frame_index in range(args.max_steps):
            state, pre_audit = capture_live_state(wrapper)
            current_measurements = wrapper.measure_manager.get_measurements()
            evaluator_distance = evaluator_distance_from_measurements(current_measurements)
            success_streak = advance_success_streak(
                success_streak, evaluator_distance, radius=SUCCESS_RADIUS_M
            )
            rgb = rotated_rgb(observation)
            jpeg = encode_jpeg(rgb, args.jpeg_quality)
            rgb_relative = f"front_rgb/{frame_index:06d}.jpg"
            (episode_dir / rgb_relative).write_bytes(jpeg)

            evaluator_stop = success_streak >= args.success_streak
            if evaluator_stop:
                wrapper.set_stop_called(True)
                decision = apply_action_safety([0.0, 0.0, 0.0])
                request_id = None
                action_offset = None
                action_source = "evaluator_stop"
            else:
                if not action_queue:
                    request = build_policy_request(
                        state=state,
                        jpeg=jpeg,
                        episode_id=short_episode["short_episode_id"],
                        instruction=short_episode["instruction"],
                        replan_index=replan_index,
                    )
                    request_started = time.perf_counter()
                    try:
                        response = unix_request(
                            args.socket, request, jpeg, timeout_s=args.request_timeout
                        )
                    except Exception as error:
                        zero = torch.tensor([0.0, 0.0, 0.0], device=wrapper.unwrapped.device)
                        wrapper.step(zero)
                        status = "inference_error"
                        reason = "zero_stop_after_inference_error"
                        failure = f"{type(error).__name__}: {error}"
                        break
                    roundtrip_ms = (time.perf_counter() - request_started) * 1000.0
                    request_range_summary = summarize_action_range(
                        [response["actions"]], execute_steps=args.execute_steps
                    )
                    request_row = {
                        "status": "ok",
                        "requested_at_utc": now_utc(),
                        "request": request,
                        "response": response,
                        "roundtrip_wall_ms": roundtrip_ms,
                        **request_range_summary.as_dict(),
                        "raw_range_violation_count": request_range_summary.violating_vectors,
                        "deprecated_compatibility_fields": {
                            "raw_range_violation_count": "alias of raw_action_vector_violation_count"
                        },
                    }
                    requests.append(request_row)
                    append_jsonl(requests_stream, request_row)
                    action_queue = select_executed_actions(
                        response["actions"], execute_steps=args.execute_steps
                    )
                    queue_index = 0
                    replan_index += 1
                decision = action_queue.pop(0)
                request_id = f"{short_episode['short_episode_id']}:{replan_index - 1}"
                action_offset = queue_index
                queue_index += 1
                action_source = "smolvla"

            command = torch.tensor(decision.applied, device=wrapper.unwrapped.device)
            observation, reward, done, info = wrapper.step(command)
            _, post_audit = capture_live_state(wrapper)
            terms = termination_terms(wrapper)
            measurements = {key: jsonable(value) for key, value in info["measurements"].items()}
            next_evaluator_distance = evaluator_distance_from_measurements(measurements)
            record = {
                "frame_index": frame_index,
                "timestamp": frame_index / 50.0,
                "front_rgb": rgb_relative,
                "action_source": action_source,
                "request_id": request_id,
                "action_offset": action_offset,
                "raw_action": list(decision.raw),
                "applied_action": list(decision.applied),
                "action_in_range": decision.in_range,
                "clipped_dimensions": list(decision.clipped_dimensions),
                "policy_state": state,
                **pre_audit,
                "evaluator_distance_to_goal_m": evaluator_distance,
                "success_streak": success_streak,
                "next_robot_pose": post_audit["robot_pose"],
                "next_evaluator_distance_to_goal_m": next_evaluator_distance,
                "reward": jsonable(reward),
                "termination": bool(done),
                "termination_terms": terms,
                "measurements": measurements,
            }
            records.append(record)
            append_jsonl(steps_stream, record)
            if bool(done):
                if evaluator_stop:
                    status, reason = "complete", "evaluator_goal_stop"
                elif terms.get("base_contact"):
                    status, reason = "terminated", "collision"
                elif terms.get("bad_orientation"):
                    status, reason = "terminated", "bad_orientation"
                elif terms.get("time_out"):
                    status, reason = "terminated", "environment_timeout"
                else:
                    status, reason = "terminated", "environment_done"
                break
        else:
            status, reason = "terminated", "client_max_steps"
    finally:
        steps_stream.close()
        requests_stream.close()

    return finalize_summary(
        episode_dir=episode_dir,
        short_episode=short_episode,
        status=status,
        termination_reason=reason,
        records=records,
        requests=requests,
        started_wall=started_wall,
        error=failure,
    )


def main() -> None:
    args = parse_args()
    if args.num_envs != 1 or args.execute_steps != EXECUTE_STEPS:
        raise ValueError("M7 v1 requires num_envs=1 and execute_steps=10")
    if args.max_steps != MAX_STEPS or args.success_streak != SUCCESS_STREAK_REQUIRED:
        raise ValueError("M7 v1 requires max_steps=1500 and success_streak=10")
    if not 1 <= args.jpeg_quality <= 100 or args.request_timeout != 10.0:
        raise ValueError("M7 v1 requires JPEG quality in [1,100] and request timeout=10s")
    args.short_dataset = args.short_dataset.expanduser().resolve()
    args.episode_dir = args.episode_dir.expanduser().resolve()
    args.asset_root = args.asset_root.expanduser().resolve()
    args.navila_root = args.navila_root.expanduser().resolve()
    if args.episode_dir.exists():
        raise FileExistsError(f"refusing to overwrite episode directory: {args.episode_dir}")
    args.episode_dir.mkdir(parents=True)
    short_episode = select_short_episode(
        args.short_dataset, args.short_episode_id, args.short_index
    )
    official_episode = make_official_episode(short_episode)
    write_json(
        args.episode_dir / "client_config.json",
        {
            "format": "go2-short-vln-m7-client-config-v1",
            "created_at_utc": now_utc(),
            "short_episode_id": short_episode["short_episode_id"],
            "short_dataset": str(args.short_dataset),
            "short_dataset_sha256": sha256(args.short_dataset),
            "split": short_episode["split"],
            "instruction": short_episode["instruction"],
            "socket": args.socket,
            "request_timeout_s": args.request_timeout,
            "execute_steps": args.execute_steps,
            "max_steps": args.max_steps,
            "success_streak_required": args.success_streak,
            "jpeg_quality": args.jpeg_quality,
            "policy_request_fields": [
                "current_rotated_rgb_jpeg",
                "m5_30d_state",
                "short_instruction",
            ],
            "forbidden_policy_inputs": [
                "reference_path",
                "next_waypoint",
                "goal_pose",
                "goal_direction",
                "distance_to_goal",
                "pd_state",
            ],
        },
    )

    configure_app_launcher(args.asset_root)
    from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    wrapper = None
    started_wall = time.monotonic()
    try:
        import cv2  # noqa: F401
        import gymnasium as gym
        import numpy as np
        import torch  # noqa: F401
        from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
        import omni.isaac.vlnce.config  # noqa: F401
        from omni.isaac.vlnce.utils import (
            ASSETS_DIR,
            RslRlVecEnvHistoryWrapper,
            VLNEnvWrapper,
        )
        from rsl_rl.runners import OnPolicyRunner
        import cli_args

        print(f"M7 phase: AppLauncher ready (running={simulation_app.is_running()})", flush=True)
        local_go2 = configure_runtime_assets(args.asset_root)
        env_cfg = parse_env_cfg(args.task, num_envs=1)
        configure_env_episode(env_cfg, official_episode, local_go2)
        usd_path = Path(ASSETS_DIR) / "matterport_usd" / env_cfg.scene_id / f"{env_cfg.scene_id}.usd"
        if not usd_path.is_file():
            raise FileNotFoundError(usd_path)
        env_cfg.scene.terrain.obj_filepath = str(usd_path)
        print("M7 phase: constructing official environment", flush=True)
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        print(f"M7 phase: environment constructed (running={simulation_app.is_running()})", flush=True)
        env = (
            RslRlVecEnvHistoryWrapper(env, history_length=args.history_length)
            if args.history_length > 0
            else RslRlVecEnvWrapper(env)
        )
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
        log_root = args.navila_root / "logs" / "rsl_rl" / agent_cfg.experiment_name
        resume_path = get_checkpoint_path(
            str(log_root.resolve()), args.load_run, agent_cfg.load_checkpoint
        )
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        low_level_policy = runner.get_inference_policy(device=env.unwrapped.device)
        measures = [
            "PathLength",
            "DistanceToGoal",
            "Success",
            "SPL",
            "OracleNavigationError",
            "OracleSuccess",
        ]
        wrapper = VLNEnvWrapper(
            env,
            low_level_policy,
            args.task,
            official_episode,
            max_length=args.max_steps + 100,
            high_level_obs_key="camera_obs",
            measure_names=measures,
        )
        if not simulation_app.is_running():
            raise RuntimeError("Kit stopped before VLN wrapper reset; inspect M7 timeline diagnostics")
        print("M7 phase: resetting official VLN wrapper", flush=True)
        observation, _ = wrapper.reset()
        print("M7 phase: reset complete; starting policy rollout", flush=True)
        summary = run_loop(wrapper, observation, short_episode, args)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        failure_summary = finalize_startup_failure(
            episode_dir=args.episode_dir,
            short_episode=short_episode,
            started_wall=started_wall,
            error=error,
        )
        print(json.dumps(failure_summary, indent=2), flush=True)
        raise
    finally:
        if wrapper is not None:
            try:
                wrapper.close()
            except Exception as error:
                print(f"M7 wrapper close warning: {error}", flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
