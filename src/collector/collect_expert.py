#!/usr/bin/env python3
"""Collect one short-VLN expert rollout around the official NaVILA PD planner.

The official ``demo_planner.py`` remains unmodified.  This process adapts one
short_vln_v1 episode to the schema expected by that script, then instruments the
``VLNEnvWrapper.step`` boundary to capture pre-action RGB/state, the PD command,
the command injected into locomotion, post-action motion, and the low-level joint
action.
"""

from __future__ import annotations

import argparse
import atexit
import builtins
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np

from src.collector.r7_supervision import (
    PD_ACTION_SOURCE,
    TERMINAL_HOLD_ACTION_SOURCE,
    command_vector,
)
from src.collector.collection_limits import (
    EXPERT_PATH_MARKER_PRIM_PATH,
    is_expert_path_marker,
    pd_frame_limit_reached,
    result_with_forced_done,
)
from src.collector.state_serialization import serialize_batched_euler_xyz
from src.collector.d5_collection import OFFICIAL_LARGE_ORIENTATION_RAD, official_large_orientation


DEFAULT_OFFICIAL_SOURCE = Path(
    "<external-data-root>"
)
DEFAULT_SHORT_DATASET = Path("<external-workspace>")
DEFAULT_ASSET_ROOT = Path("<external-data-root>")


def parse_collector_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--short-dataset", type=Path, default=DEFAULT_SHORT_DATASET)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--short-episode-id")
    selection.add_argument("--short-index", type=int)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, default=DEFAULT_OFFICIAL_SOURCE)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--terminal-hold-frames",
        type=int,
        default=0,
        help="after a successful PD action, collect this many real zero-command hold frames (R7 only)",
    )
    parser.add_argument(
        "--max-pd-frames",
        type=int,
        default=None,
        help="optional hard cap on non-terminal PD records; capped runs are rejected, never accepted",
    )
    parser.add_argument(
        "--hide-expert-path-markers",
        action="store_true",
        help="process-locally hide official GT expert-path cuboids from captured RGB",
    )
    parser.add_argument("--collector-help", action="store_true")
    args, official_args = parser.parse_known_args()
    if args.collector_help:
        parser.print_help()
        raise SystemExit(0)
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if not 0 <= args.terminal_hold_frames <= 1_000:
        raise ValueError("--terminal-hold-frames must be in [0, 1000]")
    if args.max_pd_frames is not None and args.max_pd_frames <= 0:
        raise ValueError("--max-pd-frames must be positive when set")
    return args, official_args


ARGS, OFFICIAL_ARGS = parse_collector_args()
SOURCE = ARGS.official_source.expanduser().resolve()
SHORT_DATASET = ARGS.short_dataset.expanduser().resolve()
EPISODE_DIR = ARGS.episode_dir.expanduser().resolve()
ASSET_ROOT = ARGS.asset_root.expanduser().resolve()

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
if not SHORT_DATASET.is_file():
    raise FileNotFoundError(SHORT_DATASET)
if not ASSET_ROOT.is_dir():
    raise FileNotFoundError(ASSET_ROOT)
if EPISODE_DIR.exists():
    raise FileExistsError(f"Refusing to overwrite existing episode directory: {EPISODE_DIR}")

payload = json.loads(SHORT_DATASET.read_text(encoding="utf-8"))
short_episodes = payload["episodes"]
if ARGS.short_index is not None:
    short_episode = short_episodes[ARGS.short_index]
else:
    matches = [
        episode
        for episode in short_episodes
        if episode["short_episode_id"] == ARGS.short_episode_id
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one short episode named {ARGS.short_episode_id!r}; found {len(matches)}"
        )
    short_episode = matches[0]

short_id = short_episode["short_episode_id"]
EPISODE_DIR.mkdir(parents=True)
RGB_DIR = EPISODE_DIR / "front_rgb"
RGB_DIR.mkdir()
RECORDS_PATH = EPISODE_DIR / "steps.jsonl"
SUMMARY_PATH = EPISODE_DIR / "summary.json"
TERMINAL_EVENT_PATH = EPISODE_DIR / "terminal_event.json"
ADAPTED_DATASET = EPISODE_DIR / "official_episode_adapter.json.gz"
CONFIG_PATH = EPISODE_DIR / "collector_config.json"

source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
dataset_hash = hashlib.sha256(SHORT_DATASET.read_bytes()).hexdigest()
reference_path = short_episode["reference_path"]
adapted_episode = {
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
with gzip.open(ADAPTED_DATASET, "wt", encoding="utf-8") as stream:
    json.dump({"episodes": [adapted_episode]}, stream)

CONFIG_PATH.write_text(
    json.dumps(
        {
            "status": "collecting",
            "short_episode_id": short_id,
            "source_episode_id": short_episode["source_episode_id"],
            "split": short_episode["split"],
            "scene_id": short_episode["scene_id"],
            "instruction": short_episode["instruction"],
            "goal_pose": short_episode["goal_pose"],
            "short_dataset": str(SHORT_DATASET),
            "short_dataset_sha256": dataset_hash,
            "official_planner": str(SOURCE),
            "official_planner_sha256": source_hash,
            "asset_root": str(ASSET_ROOT),
            "jpeg_quality": ARGS.jpeg_quality,
            "rgb_rotation": "90_degrees_clockwise_matching_official_demo",
            "planner_args": OFFICIAL_ARGS,
            "terminal_hold_frames": ARGS.terminal_hold_frames,
            "max_pd_frames": ARGS.max_pd_frames,
            "expert_path_marker_guard": {
                "enabled": ARGS.hide_expert_path_markers,
                "marker_prim_path": EXPERT_PATH_MARKER_PRIM_PATH,
                "official_pd_source_unchanged": True,
            },
            "terminal_hold_contract": {
                "format": "go2-short-vln-r7-terminal-stop-v1",
                "action_source": TERMINAL_HOLD_ACTION_SOURCE,
                "action": [0.0, 0.0, 0.0],
                "activation": "inside_the_successful_pd_step_before_the_official_loop_returns",
                "official_pd_source_unchanged": True,
            },
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

records_stream = RECORDS_PATH.open("w", encoding="utf-8", buffering=1)
records: list[dict[str, Any]] = []
pending_rgb: np.ndarray | None = None
last_applied_command: list[float] | None = None
terminal_hold_record_count = 0
terminal_hold_activated = False
finalized = False
timeline_extended = False
timeline_stop_subscription = None
instrumentation_installed = False
original_rotate = cv2.rotate
original_gzip_open = gzip.open
original_import = builtins.__import__
original_print = builtins.print
collection_started = time.time()
official_large_orientation_event: dict[str, Any] | None = None
expert_path_marker_suppression_installed = False
expert_path_marker_suppression_observed = False


def as_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def zero_action_like(value: Any) -> Any:
    """Create a zero command in the official planner's original tensor/array type."""
    if hasattr(value, "new_zeros") and hasattr(value, "shape"):
        return value.new_zeros(value.shape)
    return np.zeros_like(np.asarray(value))


def finite_scalar(value: Any) -> float | None:
    try:
        values = np.asarray(as_list(value), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.size != 1 or not np.isfinite(values).all():
        return None
    return float(values.reshape(-1)[0])


def capture_official_print(*args: Any, **kwargs: Any) -> Any:
    """Record the immutable planner's large-orientation guard before it returns."""
    global official_large_orientation_event
    if (
        official_large_orientation_event is None
        and len(args) >= 3
        and args[0] == "Large orientation: "
    ):
        roll, pitch = finite_scalar(args[1]), finite_scalar(args[2])
        if (
            roll is not None
            and pitch is not None
            and (abs(roll) > OFFICIAL_LARGE_ORIENTATION_RAD or abs(pitch) > OFFICIAL_LARGE_ORIENTATION_RAD)
        ):
            official_large_orientation_event = {
                "format": "go2-short-vln-official-orientation-event-v1",
                "event": "official_large_orientation",
                "stage": "before_first_action" if not records else "after_action",
                "record_count_at_event": len(records),
                "roll_rad": roll,
                "pitch_rad": pitch,
                "official_threshold_rad": OFFICIAL_LARGE_ORIENTATION_RAD,
                "strict_threshold_exceeded": True,
                "elapsed_wall_seconds": time.time() - collection_started,
                "official_planner_sha256": source_hash,
            }
            TERMINAL_EVENT_PATH.write_text(
                json.dumps(official_large_orientation_event, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    return original_print(*args, **kwargs)


def xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def path_remaining_xy(position: Sequence[float]) -> float:
    closest_index = min(
        range(len(reference_path)),
        key=lambda index: xy_distance(position, reference_path[index]),
    )
    remaining = xy_distance(position, reference_path[closest_index])
    remaining += sum(
        xy_distance(start, end)
        for start, end in zip(reference_path[closest_index:], reference_path[closest_index + 1 :])
    )
    return remaining


def body_velocity(data: Any, name: str, world_name: str, quaternion: Any) -> Any:
    value = getattr(data, name, None)
    if value is not None:
        return value[0]
    import omni.isaac.lab.utils.math as math_utils

    return math_utils.quat_rotate_inverse(quaternion, getattr(data, world_name)[0])


def capture_robot_state(wrapper: Any) -> dict[str, Any]:
    import omni.isaac.lab.utils.math as math_utils

    robot = wrapper.unwrapped.scene["robot"]
    data = robot.data
    quaternion = data.root_quat_w[0]
    root_lin_vel_b = body_velocity(data, "root_lin_vel_b", "root_lin_vel_w", quaternion)
    root_ang_vel_b = body_velocity(data, "root_ang_vel_b", "root_ang_vel_w", quaternion)
    rpy = serialize_batched_euler_xyz(math_utils.euler_xyz_from_quat(quaternion.unsqueeze(0)))
    joint_pos_relative = data.joint_pos[0] - data.default_joint_pos[0]
    state_vector = np.concatenate(
        [
            np.asarray(as_list(root_lin_vel_b)),
            np.asarray(as_list(root_ang_vel_b)),
            np.asarray(rpy),
            np.asarray(as_list(joint_pos_relative)),
            np.asarray(as_list(data.joint_vel[0])),
        ]
    ).tolist()
    return {
        "robot_pose": {
            "position_w": as_list(data.root_pos_w[0]),
            "quaternion_wxyz": as_list(quaternion),
        },
        "robot_state": state_vector,
        "robot_state_schema": (
            "root_linear_velocity_body[3],root_angular_velocity_body[3],"
            "base_rpy[3],joint_position_relative[12],joint_velocity[12]"
        ),
        "current_velocity": {
            "linear_body": as_list(root_lin_vel_b),
            "angular_body": as_list(root_ang_vel_b),
            "linear_world": as_list(data.root_lin_vel_w[0]),
            "angular_world": as_list(data.root_ang_vel_w[0]),
        },
        "joint_position": as_list(data.joint_pos[0]),
        "joint_velocity": as_list(data.joint_vel[0]),
    }


def capture_rotate(image: Any, rotate_code: int) -> Any:
    global pending_rgb
    frame = original_rotate(image, rotate_code)
    if frame is not None:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] >= 3:
            pending_rgb = array[:, :, :3].copy()
    return frame


def normalize_rgb(frame: Any) -> np.ndarray:
    """Validate the collector's rotated camera convention and return RGB uint8."""
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"camera frame must be HxWx3+, got {array.shape}")
    return array[:, :, :3].copy()


def rgb_from_wrapper_info(info: Any) -> np.ndarray:
    """Extract the next real camera observation for an in-step terminal hold."""
    try:
        camera = info["observations"]["camera_obs"]
        if hasattr(camera, "detach"):
            camera = camera.detach().cpu().numpy()
        camera = np.asarray(camera)
        if camera.ndim != 4 or camera.shape[0] < 1:
            raise ValueError(f"camera_obs must be [B,H,W,C], got {camera.shape}")
        return normalize_rgb(original_rotate(camera[0, :, :, :3], cv2.ROTATE_90_CLOCKWISE))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError("R7 terminal hold cannot extract a fresh wrapper camera observation") from error


def write_rgb_frame(rgb: np.ndarray, frame_index: int) -> str:
    rgb_relative = f"front_rgb/{frame_index:06d}.jpg"
    rgb_path = EPISODE_DIR / rgb_relative
    written = cv2.imwrite(
        str(rgb_path),
        cv2.cvtColor(normalize_rgb(rgb), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, ARGS.jpeg_quality],
    )
    if not written:
        raise RuntimeError(f"Failed to write RGB frame {rgb_path}")
    return rgb_relative


def adapted_gzip_open(filename: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        path = Path(filename)
        if path.name == "vln_ce_isaac_v1.json.gz" and path.resolve() != ADAPTED_DATASET:
            print(f"M4 dataset adapter: {path} -> {ADAPTED_DATASET}", flush=True)
            return original_gzip_open(ADAPTED_DATASET, *args, **kwargs)
    except (TypeError, OSError):
        pass
    return original_gzip_open(filename, *args, **kwargs)


def finalize(
    status: str, termination_reason: str, *, terminal_event: dict[str, Any] | None = None
) -> None:
    global finalized
    if finalized:
        return
    finalized = True
    records_stream.flush()
    os.fsync(records_stream.fileno())
    records_stream.close()
    if records:
        first_position = records[0]["robot_pose"]["position_w"]
        last_position = records[-1]["next_robot_pose"]["position_w"]
        final_goal_distance = records[-1]["next_distance_to_goal_xy_m"]
        initial_goal_distance = records[0]["distance_to_goal_xy_m"]
        control_dt = records[0]["control_dt_s"]
        success = bool(records[-1]["success"])
    else:
        first_position = last_position = None
        final_goal_distance = initial_goal_distance = None
        control_dt = None
        success = False
    marker_guard_ok = not ARGS.hide_expert_path_markers or expert_path_marker_suppression_observed
    if not marker_guard_ok:
        status = "terminated_without_success"
        termination_reason = "expert_path_marker_suppression_not_observed"
        success = False
    summary = {
        "status": status,
        "termination_reason": termination_reason,
        "short_episode_id": short_id,
        "source_episode_id": short_episode["source_episode_id"],
        "scene_id": short_episode["scene_id"],
        "split": short_episode["split"],
        "instruction": short_episode["instruction"],
        "goal_pose": short_episode["goal_pose"],
        "frame_count": len(records),
        "record_count": len(records),
        "control_dt_s": control_dt,
        "control_frequency_hz": (1.0 / control_dt if control_dt else None),
        "initial_robot_position_w": first_position,
        "final_robot_position_w": last_position,
        "robot_displacement_xy_m": (
            xy_distance(first_position, last_position) if first_position is not None else None
        ),
        "initial_distance_to_goal_xy_m": initial_goal_distance,
        "final_distance_to_goal_xy_m": final_goal_distance,
        "success": success,
        "elapsed_wall_seconds": time.time() - collection_started,
        "max_pd_frames": ARGS.max_pd_frames,
        "expert_path_marker_guard": {
            "enabled": ARGS.hide_expert_path_markers,
            "marker_prim_path": EXPERT_PATH_MARKER_PRIM_PATH,
            "installed": expert_path_marker_suppression_installed,
            "observed": expert_path_marker_suppression_observed,
            "passed": marker_guard_ok,
        },
        "official_planner_sha256": source_hash,
        "short_dataset_sha256": dataset_hash,
        "terminal_supervision": {
            "format": "go2-short-vln-r7-terminal-stop-v1",
            "requested_hold_frames": ARGS.terminal_hold_frames,
            "activated": terminal_hold_activated,
            "recorded_hold_frames": terminal_hold_record_count,
            "exact_requested_count": (
                terminal_hold_activated
                and terminal_hold_record_count == ARGS.terminal_hold_frames
            ),
            "action_source": TERMINAL_HOLD_ACTION_SOURCE,
        },
    }
    if terminal_event is not None:
        summary["terminal_event"] = terminal_event
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["status"] = status
    config["summary"] = str(SUMMARY_PATH)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"M4 collector finalized: {json.dumps(summary)}", flush=True)


def install_vln_instrumentation() -> None:
    global instrumentation_installed
    if instrumentation_installed:
        return
    instrumentation_installed = True
    from omni.isaac.vlnce.utils import VLNEnvWrapper

    original_update_command = VLNEnvWrapper.update_command
    original_step = VLNEnvWrapper.step

    def instrumented_update_command(wrapper: Any, command: Any) -> None:
        global last_applied_command
        last_applied_command = as_list(command)
        return original_update_command(wrapper, command)

    def instrumented_step(wrapper: Any, action: Any) -> Any:
        """Capture the PD transition and, on success, an in-step zero hold.

        ``demo_planner.py`` breaks its own loop as soon as its distance counter
        reaches the stop condition.  Returning a synthetic ``done=False`` is
        therefore insufficient: the planner never requests another observation.
        The collector instead performs the opt-in zero-command tail *inside the
        same successful PD call*, before returning the untouched result to the
        immutable planner.
        """
        global pending_rgb, last_applied_command
        global terminal_hold_record_count, terminal_hold_activated
        if pending_rgb is None:
            raise RuntimeError("Official planner called step before a valid rotated RGB frame")

        control_dt = float(wrapper.unwrapped.cfg.decimation * wrapper.unwrapped.cfg.sim.dt)
        goal = short_episode["goal_pose"]["position"]
        success_radius = float(short_episode["goal_pose"]["success_radius_m"])
        pd_planner_command = list(command_vector(as_list(action), name="official_pd_command"))

        def done_bool(value: Any) -> bool:
            return bool(value.item()) if hasattr(value, "item") else bool(value)

        def reward_value(value: Any) -> float:
            return float(value.reshape(-1)[0].item()) if hasattr(value, "reshape") else float(value)

        def append_record(
            *,
            rgb: np.ndarray,
            pre_state: dict[str, Any],
            post_state: dict[str, Any],
            reward: Any,
            applied_command: list[float] | None,
            planner_command: list[float],
            action_source: str,
            raw_environment_done: bool,
            raw_environment_success: bool,
            termination: bool,
            success: bool,
            terminal_hold_index: int | None = None,
        ) -> None:
            if applied_command is None:
                raise RuntimeError("No locomotion command was observed at update_command")
            applied = list(command_vector(applied_command, name="locomotion_command"))
            pre_position = pre_state["robot_pose"]["position_w"]
            post_position = post_state["robot_pose"]["position_w"]
            frame_index = len(records)
            record = {
                "episode_id": short_id,
                "source_episode_id": short_episode["source_episode_id"],
                "frame_index": frame_index,
                "timestamp": frame_index * control_dt,
                "control_dt_s": control_dt,
                "front_rgb": write_rgb_frame(rgb, frame_index),
                **pre_state,
                "instruction": short_episode["instruction"],
                "expert_vx": planner_command[0],
                "expert_vy": planner_command[1],
                "expert_wz": planner_command[2],
                "planner_command": planner_command,
                "locomotion_command": applied,
                "pd_planner_command": pd_planner_command,
                "action_source": action_source,
                "raw_environment_done": raw_environment_done,
                "raw_environment_success": raw_environment_success,
                "low_level_joint_action": as_list(wrapper.low_level_action),
                "goal_pose": short_episode["goal_pose"],
                "distance_to_goal_xy_m": xy_distance(pre_position, goal),
                "distance_to_goal_path_m": path_remaining_xy(pre_position),
                "next_robot_pose": post_state["robot_pose"],
                "next_current_velocity": post_state["current_velocity"],
                "next_distance_to_goal_xy_m": xy_distance(post_position, goal),
                "next_distance_to_goal_path_m": path_remaining_xy(post_position),
                "reward": reward_value(reward),
                "termination": termination,
                "success": success,
            }
            if terminal_hold_index is not None:
                record["terminal_hold_index"] = terminal_hold_index
                record["terminal_hold_total_frames"] = ARGS.terminal_hold_frames
            records.append(record)
            records_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            records_stream.flush()

        # Capture the official PD transition exactly as before.
        pre_state = capture_robot_state(wrapper)
        rgb = pending_rgb
        pending_rgb = None
        last_applied_command = None
        result = original_step(wrapper, action)
        _, reward, done, info = result
        post_state = capture_robot_state(wrapper)
        post_position = post_state["robot_pose"]["position_w"]
        raw_environment_done = done_bool(done)
        raw_environment_success = raw_environment_done and xy_distance(post_position, goal) < success_radius

        # Keep the successful PD action as evidence but do not label it terminal:
        # terminal supervision is the following, explicitly labelled zero tail.
        hold_requested = raw_environment_success and ARGS.terminal_hold_frames > 0
        forced_frame_limit = (
            not raw_environment_done
            and pd_frame_limit_reached(ARGS.max_pd_frames, len(records) + 1)
        )
        append_record(
            rgb=rgb,
            pre_state=pre_state,
            post_state=post_state,
            reward=reward,
            applied_command=last_applied_command,
            planner_command=pd_planner_command,
            action_source=PD_ACTION_SOURCE,
            raw_environment_done=raw_environment_done,
            raw_environment_success=raw_environment_success,
            termination=(raw_environment_done and not hold_requested) or forced_frame_limit,
            success=raw_environment_success and not hold_requested,
        )
        if forced_frame_limit:
            finalize("terminated_without_success", "collector_pd_frame_limit")
            # The official outer loop is unmodified.  Returning done here only
            # stops that loop after the final real PD transition has been
            # recorded and marked as rejected evidence.
            return result_with_forced_done(result)
        if not hold_requested:
            if raw_environment_done:
                finalize(
                    "complete" if raw_environment_success else "terminated_without_success",
                    "goal_reached" if raw_environment_success else "environment_done",
                )
            return result

        terminal_hold_activated = True
        hold_info = info
        zero_action = zero_action_like(action)
        for terminal_hold_index in range(1, ARGS.terminal_hold_frames + 1):
            # ``hold_info`` is the simulator-produced observation after the
            # preceding action.  It is not copied from an earlier image.
            hold_rgb = rgb_from_wrapper_info(hold_info)
            hold_pre_state = capture_robot_state(wrapper)
            last_applied_command = None
            _, hold_reward, hold_done, hold_info = original_step(wrapper, zero_action)
            hold_post_state = capture_robot_state(wrapper)
            hold_post_position = hold_post_state["robot_pose"]["position_w"]
            hold_raw_done = done_bool(hold_done)
            hold_raw_success = hold_raw_done and xy_distance(hold_post_position, goal) < success_radius
            if not hold_raw_success:
                raise RuntimeError(
                    "R7 terminal hold did not remain environment-done inside the success radius"
                )
            terminal_hold_record_count += 1
            final_hold = terminal_hold_index == ARGS.terminal_hold_frames
            append_record(
                rgb=hold_rgb,
                pre_state=hold_pre_state,
                post_state=hold_post_state,
                reward=hold_reward,
                applied_command=last_applied_command,
                planner_command=[0.0, 0.0, 0.0],
                action_source=TERMINAL_HOLD_ACTION_SOURCE,
                raw_environment_done=True,
                raw_environment_success=True,
                termination=final_hold,
                success=final_hold,
                terminal_hold_index=terminal_hold_index,
            )

        finalize("complete", "goal_reached")
        # Preserve the real terminal result for the official outer loop.
        return result

    VLNEnvWrapper.update_command = instrumented_update_command
    VLNEnvWrapper.step = instrumented_step
    print("M4 VLNEnvWrapper instrumentation installed", flush=True)


def configure_process() -> None:
    from omni.isaac.lab.app import AppLauncher

    original_init = AppLauncher.__init__
    kit_asset_args = (
        f"--/persistent/isaac/asset_root/default={ASSET_ROOT}",
        f"--/persistent/isaac/asset_root/cloud={ASSET_ROOT}",
        f"--/persistent/isaac/asset_root/nvidia={ASSET_ROOT}",
        "--/persistent/isaac/asset_root/timeout=1.0",
    )

    def init_with_collection(self: Any, *init_args: Any, **init_kwargs: Any) -> None:
        for kit_arg in kit_asset_args:
            key = kit_arg.split("=", 1)[0] + "="
            if not any(argument.startswith(key) for argument in sys.argv):
                sys.argv.append(kit_arg)
        original_init(self, *init_args, **init_kwargs)
        # The official planner renders a cuboid at every GT expert-path point.
        # It is useful for a human demo but a privileged visual leakage for a
        # vision-language policy. Patch only this process's marker visibility;
        # the immutable planner still computes exactly the same PD command.
        global expert_path_marker_suppression_installed
        if ARGS.hide_expert_path_markers and not expert_path_marker_suppression_installed:
            from omni.isaac.lab.markers import VisualizationMarkers

            original_marker_set_visibility = VisualizationMarkers.set_visibility

            def set_visibility_without_expert_path(self: Any, visible: bool) -> Any:
                global expert_path_marker_suppression_observed
                if is_expert_path_marker(getattr(getattr(self, "cfg", None), "prim_path", "")):
                    expert_path_marker_suppression_observed = True
                    return original_marker_set_visibility(self, False)
                return original_marker_set_visibility(self, visible)

            VisualizationMarkers.set_visibility = set_visibility_without_expert_path
            expert_path_marker_suppression_installed = True
            print(
                f"M4 expert-path marker guard enabled for {EXPERT_PATH_MARKER_PRIM_PATH}",
                flush=True,
            )
        import carb
        import omni.timeline
        import omni.usd
        from omni.isaac.lab.sim import SimulationContext

        settings = carb.settings.get_settings()
        settings.set_string("/persistent/isaac/asset_root/default", str(ASSET_ROOT))
        settings.set_string("/persistent/isaac/asset_root/cloud", str(ASSET_ROOT))
        settings.set_float("/persistent/isaac/asset_root/timeout", 30.0)
        settings.set_bool("/app/hangDetector/enabled", False)

        # The collector's import hooks add work between AppLauncher startup and
        # task-config import.  Pin the already-verified Go2 asset explicitly so
        # UNITREE_GO2_CFG cannot retain the unavailable cloud URL if its nucleus
        # constant was resolved during that interval.  This mutates only the
        # current process's config object; neither IsaacLab nor NaVILA is edited.
        local_go2_usd = ASSET_ROOT / "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
        if not local_go2_usd.is_file():
            raise FileNotFoundError(local_go2_usd)
        from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG

        previous_go2_usd = UNITREE_GO2_CFG.spawn.usd_path
        UNITREE_GO2_CFG.spawn.usd_path = str(local_go2_usd)
        print(
            f"M4 process-local Go2 USD: {previous_go2_usd} -> {local_go2_usd}",
            flush=True,
        )

        # omni.isaac.vlnce defines and copies its own UNITREE_GO2_CFG rather
        # than reusing lab_assets.unitree.  Patch the fully materialized task
        # config at the final construction boundary so every earlier copy is
        # covered, and verify the value before any stage prim is spawned.
        import gymnasium as gym

        original_gym_make = gym.make

        def make_with_local_go2(*make_args: Any, **make_kwargs: Any) -> Any:
            env_cfg = make_kwargs.get("cfg")
            robot_spawn = getattr(
                getattr(getattr(env_cfg, "scene", None), "robot", None),
                "spawn",
                None,
            )
            if robot_spawn is not None and hasattr(robot_spawn, "usd_path"):
                previous_task_go2_usd = robot_spawn.usd_path
                robot_spawn.usd_path = str(local_go2_usd)
                print(
                    "M4 final task Go2 USD: "
                    f"{previous_task_go2_usd} -> {robot_spawn.usd_path}",
                    flush=True,
                )
                if robot_spawn.usd_path != str(local_go2_usd):
                    raise RuntimeError("Failed to pin final task Go2 USD")
            return original_gym_make(*make_args, **make_kwargs)

        gym.make = make_with_local_go2
        original_reset = SimulationContext.reset

        def reset_with_long_timeline(context: Any, *reset_args: Any, **reset_kwargs: Any) -> Any:
            global timeline_extended
            if not timeline_extended:
                timeline = omni.timeline.get_timeline_interface()
                stage = omni.usd.get_context().get_stage()
                stage.SetEndTimeCode(10000.0 * stage.GetTimeCodesPerSecond())
                timeline.set_end_time(10000.0)
                timeline.commit()
                timeline_extended = True
                print("M4 process-local Kit timeline extended to 10000 seconds", flush=True)
            return original_reset(context, *reset_args, **reset_kwargs)

        SimulationContext.reset = reset_with_long_timeline

        global timeline_stop_subscription

        def log_timeline_stop(event: Any) -> None:
            timeline = omni.timeline.get_timeline_interface()
            print(
                f"M4 timeline STOP current={timeline.get_current_time()} end={timeline.get_end_time()}",
                flush=True,
            )
            # The immutable official planner returns directly on its own
            # large-orientation guard, before VLNEnvWrapper.step can return a
            # done flag.  The print hook records the exact guard values even
            # when the first action has not yet been issued.  Other unexpected
            # timeline stops remain explicitly unclassified and are not
            # replaceable by D5.
            if not finalized and official_large_orientation_event is not None:
                reason = (
                    "official_large_orientation_before_first_action"
                    if not records
                    else "official_large_orientation"
                )
                finalize(
                    "terminated_without_success",
                    reason,
                    terminal_event=official_large_orientation_event,
                )
            elif not finalized and records:
                pose = records[-1].get("next_robot_pose", {})
                quaternion = pose.get("quaternion_wxyz") if isinstance(pose, dict) else None
                try:
                    reason = (
                        "official_large_orientation"
                        if isinstance(quaternion, list) and official_large_orientation(quaternion)
                        else "timeline_stop_unclassified"
                    )
                except ValueError:
                    reason = "timeline_stop_unclassified"
                finalize("terminated_without_success", reason)

        timeline_stop_subscription = (
            omni.timeline.get_timeline_interface()
            .get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), log_timeline_stop, order=0
            )
        )
    AppLauncher.__init__ = init_with_collection


def delayed_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Install the VLN step hook only when the official script imports it.

    Importing the VLN package during AppLauncher initialization resolves the
    Go2 asset constant before the process-local asset root is fully active.  A
    delayed hook preserves the M1 import order and patches the class before its
    first instance is created.
    """
    module = original_import(name, globals_, locals_, fromlist, level)
    if name == "omni.isaac.vlnce.utils":
        install_vln_instrumentation()
    return module


def fallback_finalize() -> None:
    if not finalized:
        try:
            finalize("interrupted", "process_exit_before_done")
        except Exception as error:
            print(f"M4 fallback finalize failed: {error}", flush=True)


atexit.register(fallback_finalize)
cv2.rotate = capture_rotate
gzip.open = adapted_gzip_open
builtins.print = capture_official_print
builtins.__import__ = delayed_import
configure_process()

official_args = [argument for argument in OFFICIAL_ARGS if not argument.startswith("--episode_index")]
sys.argv = [str(SOURCE), *official_args, "--episode_index=0"]
print(f"M4 collecting {short_id} into {EPISODE_DIR}", flush=True)
print(f"M4 official planner SHA-256: {source_hash}", flush=True)
sys.path.insert(0, str(SOURCE.parent))
runpy.run_path(str(SOURCE), run_name="__main__")
