#!/usr/bin/env python3
"""Sanity-check expert rollouts, then create trajectory plots and MP4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.collector.r7_supervision import command_semantics, terminal_hold_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def finite_vector(values: Any) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def make_plot(episode_dir: Path, records: list[dict[str, Any]]) -> Path:
    timestamps = np.asarray([record["timestamp"] for record in records])
    actions = np.asarray([record["planner_command"] for record in records])
    distances = np.asarray([record["distance_to_goal_xy_m"] for record in records])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(timestamps, actions[:, 0], label="vx")
    axes[0].plot(timestamps, actions[:, 1], label="vy")
    axes[0].plot(timestamps, actions[:, 2], label="wz")
    axes[0].set_ylabel("Expert command")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(timestamps, distances, color="#c62828", label="distance_to_goal")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("XY distance (m)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(episode_dir.name)
    output = episode_dir / "trajectory.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def make_video(episode_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, int]:
    first = cv2.imread(str(episode_dir / records[0]["front_rgb"]))
    if first is None:
        raise ValueError("Cannot read first RGB frame for video")
    height, width = first.shape[:2]
    fps = 1.0 / float(records[0]["control_dt_s"])
    output = episode_dir / "rollout.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output}")
    written = 0
    for record in records:
        frame = cv2.imread(str(episode_dir / record["front_rgb"]))
        if frame is None or frame.shape[:2] != (height, width):
            writer.release()
            raise ValueError(f"Unreadable or inconsistent frame {record['front_rgb']}")
        command = record["planner_command"]
        text = (
            f"t={record['timestamp']:.2f}s  vx={command[0]:.2f} "
            f"vy={command[1]:.2f} wz={command[2]:.2f} "
            f"d={record['distance_to_goal_xy_m']:.2f}m"
        )
        cv2.putText(
            frame,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        written += 1
    writer.release()
    capture = cv2.VideoCapture(str(output))
    readable = capture.isOpened()
    decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
    capture.release()
    if not readable or decoded_frames < max(1, written - 1):
        raise ValueError(
            f"Video verification failed: written={written}, decoded={decoded_frames}"
        )
    return output, decoded_frames


def zero_frame_orientation_result(
    episode_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate a pre-action official orientation exit without inventing rollout media."""
    event_path = episode_dir / "terminal_event.json"
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        event = None
    config_path = episode_dir / "collector_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        config = None
    threshold = 0.6
    event_valid = False
    if isinstance(event, dict):
        try:
            roll, pitch = float(event["roll_rad"]), float(event["pitch_rad"])
            event_valid = (
                event.get("event") == "official_large_orientation"
                and event.get("stage") == "before_first_action"
                and event.get("record_count_at_event") == 0
                and event.get("strict_threshold_exceeded") is True
                and math.isclose(float(event["official_threshold_rad"]), threshold, abs_tol=1e-12)
                and finite_vector([roll, pitch])
                and (abs(roll) > threshold or abs(pitch) > threshold)
            )
        except (KeyError, TypeError, ValueError):
            event_valid = False
    checks = {
        "zero_frame_official_orientation_evidence": (
            summary.get("status") == "terminated_without_success"
            and summary.get("termination_reason") == "official_large_orientation_before_first_action"
            and summary.get("success") is False
            and summary.get("record_count") == 0
            and summary.get("frame_count") == 0
            and not records
            and event_valid
            and isinstance(config, dict)
            and config.get("official_planner_sha256") == summary.get("official_planner_sha256")
        ),
        "zero_frame_steps_empty": not records,
        "zero_frame_no_rgb": not any((episode_dir / "front_rgb").glob("*.jpg")),
    }


def terminal_supervision_result(
    episode_dir: Path, summary: Mapping[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate opt-in R7 stop supervision while preserving legacy M4 checks."""
    try:
        config = json.loads((episode_dir / "collector_config.json").read_text(encoding="utf-8"))
        requested = config.get("terminal_hold_frames", 0)
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 0:
            raise ValueError("invalid terminal_hold_frames")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        requested = -1
    semantics = command_semantics(records, allow_legacy=requested == 0)
    terminal = terminal_hold_audit(records, requested_frames=requested) if requested >= 0 else {
        "passed": False,
        "exact_count": False,
        "tail_contiguous": False,
        "zero_labels": False,
        "raw_environment_done_each_frame": False,
        "inside_success_radius": False,
        "final_record_success_and_termination": False,
    }
    summary_contract = summary.get("terminal_supervision")
    if requested == 0:
        summary_ok = summary_contract is None or (
            isinstance(summary_contract, Mapping)
            and summary_contract.get("requested_hold_frames") == 0
        )
    else:
        summary_ok = (
            isinstance(summary_contract, Mapping)
            and summary_contract.get("requested_hold_frames") == requested
            and summary_contract.get("activated") is True
            and summary_contract.get("recorded_hold_frames") == requested
            and summary_contract.get("exact_requested_count") is True
        )
    return {
        "requested_frames": requested,
        "command_semantics": semantics,
        "terminal_hold": terminal,
        "summary_contract_consistent": summary_ok,
        "passed": semantics["passed"] and terminal["passed"] and summary_ok,
    }
    return {
        "episode_id": summary["short_episode_id"],
        "episode_dir": str(episode_dir),
        "passed": False,
        "checks": checks,
        "record_count": 0,
        "control_frequency_hz": None,
        "robot_displacement_xy_m": None,
        "initial_distance_to_goal_xy_m": None,
        "final_distance_to_goal_xy_m": None,
        "action_min": None,
        "action_max": None,
        "mean_forward_speed_when_commanded_mps": None,
        "trajectory_plot": None,
        "video": None,
        "video_decoded_frames": 0,
    }


def check_episode(episode_dir: Path) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
    records = load_records(episode_dir / "steps.jsonl")
    if (
        summary.get("termination_reason") == "official_large_orientation_before_first_action"
        and summary.get("record_count") == 0
    ):
        result = zero_frame_orientation_result(episode_dir, summary, records)
        (episode_dir / "sanity.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    checks: dict[str, bool] = {}
    checks["record_count_positive"] = len(records) > 1
    checks["record_summary_count_match"] = len(records) == summary["record_count"]
    indices = [record["frame_index"] for record in records]
    checks["frame_indices_contiguous"] = indices == list(range(len(records)))

    frame_paths = [episode_dir / record["front_rgb"] for record in records]
    checks["no_missing_frames"] = all(path.is_file() and path.stat().st_size > 0 for path in frame_paths)
    valid_rgb = True
    sampled_hashes = set()
    for index in np.linspace(0, len(frame_paths) - 1, min(10, len(frame_paths)), dtype=int):
        path = frame_paths[int(index)]
        frame = cv2.imread(str(path))
        if frame is None or frame.ndim != 3 or frame.shape[:2] != (512, 512):
            valid_rgb = False
            continue
        if float(frame.std()) < 1.0:
            valid_rgb = False
        sampled_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    checks["rgb_sequence_normal"] = valid_rgb and len(sampled_hashes) > 1

    timestamps = [float(record["timestamp"]) for record in records]
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    checks["timestamps_strictly_monotonic"] = all(delta > 0 for delta in deltas)
    expected_dt = float(records[0]["control_dt_s"])
    checks["timestamp_period_matches_metadata"] = bool(deltas) and all(
        math.isclose(delta, expected_dt, abs_tol=1e-9) for delta in deltas
    )

    actions = np.asarray([record["planner_command"] for record in records], dtype=np.float64)
    applied = np.asarray([record["locomotion_command"] for record in records], dtype=np.float64)
    checks["actions_finite"] = finite_vector(actions)
    checks["actions_nonzero"] = bool(np.any(np.abs(actions) > 1e-4))
    checks["actions_in_expected_range"] = bool(
        np.all((actions[:, 0] >= -1e-6) & (actions[:, 0] <= 0.500001))
        and np.all(np.abs(actions[:, 1]) <= 0.500001)
        and np.all(np.abs(actions[:, 2]) <= 0.500001)
    )
    checks["planner_equals_locomotion_command"] = bool(
        np.allclose(actions, applied, atol=1e-9)
    )
    terminal_supervision = terminal_supervision_result(episode_dir, summary, records)
    checks["r7_command_semantics"] = bool(terminal_supervision["command_semantics"]["passed"])
    checks["r7_terminal_hold_exact_count"] = bool(terminal_supervision["terminal_hold"]["exact_count"])
    checks["r7_terminal_hold_tail_contiguous"] = bool(terminal_supervision["terminal_hold"]["tail_contiguous"])
    checks["r7_terminal_hold_zero_actions"] = bool(terminal_supervision["terminal_hold"]["zero_labels"])
    checks["r7_terminal_hold_inside_success_radius"] = bool(terminal_supervision["terminal_hold"]["inside_success_radius"])
    checks["r7_terminal_summary_contract"] = bool(terminal_supervision["summary_contract_consistent"])
    checks["robot_state_finite"] = all(
        finite_vector(record["robot_state"]) and finite_vector(record["current_velocity"]["linear_body"])
        for record in records
    )
    checks["robot_state_dimension_constant"] = len(
        {len(record["robot_state"]) for record in records}
    ) == 1
    checks["robot_moved"] = float(summary["robot_displacement_xy_m"]) > 0.2
    moving_forward_speeds = [
        float(record["current_velocity"]["linear_body"][0])
        for record in records
        if float(record["planner_command"][0]) > 0.05
    ]
    checks["command_motion_consistent"] = bool(moving_forward_speeds) and statistics.fmean(
        moving_forward_speeds
    ) > 0.01
    checks["normal_termination"] = bool(records[-1]["termination"]) and not any(
        record["termination"] for record in records[:-1]
    )
    checks["success"] = bool(summary["success"]) and summary["status"] == "complete"
    initial_distance = float(summary["initial_distance_to_goal_xy_m"])
    final_distance = float(summary["final_distance_to_goal_xy_m"])
    checks["distance_to_goal_overall_decreased"] = final_distance < initial_distance
    checks["final_distance_within_success_radius"] = final_distance < float(
        records[-1]["goal_pose"]["success_radius_m"]
    )

    plot_path = make_plot(episode_dir, records)
    video_path, decoded_frames = make_video(episode_dir, records)
    passed = all(checks.values())
    result = {
        "episode_id": summary["short_episode_id"],
        "episode_dir": str(episode_dir),
        "passed": passed,
        "checks": checks,
        "record_count": len(records),
        "control_frequency_hz": summary["control_frequency_hz"],
        "robot_displacement_xy_m": summary["robot_displacement_xy_m"],
        "initial_distance_to_goal_xy_m": initial_distance,
        "final_distance_to_goal_xy_m": final_distance,
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "mean_forward_speed_when_commanded_mps": (
            statistics.fmean(moving_forward_speeds) if moving_forward_speeds else None
        ),
        "trajectory_plot": str(plot_path),
        "video": str(video_path),
        "video_decoded_frames": decoded_frames,
        "terminal_supervision": terminal_supervision,
    }
    (episode_dir / "sanity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    results = [check_episode(path) for path in args.episode_dirs]
    report = {
        "passed": all(result["passed"] for result in results),
        "episode_count": len(results),
        "passed_count": sum(result["passed"] for result in results),
        "episodes": results,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    def number(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    for result in results:
        failed = [name for name, passed in result["checks"].items() if not passed]
        rows.append(
            "| {episode} | {status} | {frames} | {hz} | {motion} | {initial} | "
            "{final} | {failed} |".format(
                episode=result["episode_id"],
                status="PASS" if result["passed"] else "FAIL",
                frames=result["record_count"],
                hz="n/a" if result["control_frequency_hz"] is None else f"{float(result['control_frequency_hz']):.1f}",
                motion=number(result["robot_displacement_xy_m"]),
                initial=number(result["initial_distance_to_goal_xy_m"]),
                final=number(result["final_distance_to_goal_xy_m"]),
                failed=", ".join(failed) if failed else "none",
            )
        )
    markdown = """# M4 Expert Collection Sanity Report

| Episode | Status | Frames | Control Hz | Displacement (m) | Initial goal distance (m) | Final goal distance (m) | Failed checks |
|---|---|---:|---:|---:|---:|---:|---|
""" + "\n".join(rows) + f"\n\nOverall: **{'PASS' if report['passed'] else 'FAIL'}** ({report['passed_count']}/{report['episode_count']}).\n"
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
