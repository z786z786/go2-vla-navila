#!/usr/bin/env python3
"""Validate M7 rollouts and create evidence videos and diagnostic plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.inference.action_audit import ActionRangeSummary, summarize_action_range
from src.inference.protocol import validate_request_header, validate_response_header
from src.inference.state import ACTION_BOUNDS, apply_action_safety


FORBIDDEN_REQUEST_FIELDS = frozenset(
    {
        "reference_path",
        "next_waypoint",
        "goal_pose",
        "goal_direction",
        "distance_to_goal",
        "pd_state",
        "planner_command",
        "oracle_heading",
    }
)
DEFAULT_EXECUTE_STEPS = 10
EXPECTED_M6_CHECKPOINT_SHA256 = "8e72db4f1bf46c12dc3eb0f6e58b2cfc18d97e448dea1ecd9116d439a03145dd"
EXPECTED_M4_PLANNER_SHA256 = "20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432"
EXPECTED_SHORT_DATASET_SHA256 = "88255ffb8e175372048c4948531d7ad059a9d8aa9217b7be95e2f60f520b34b6"
FIRST_GATE_EPISODE_IDS = ("short_vln_v1_0004",)
FULL_GATE_EPISODE_IDS = (
    "short_vln_v1_0000",
    "short_vln_v1_0001",
    "short_vln_v1_0003",
    "short_vln_v1_0004",
    "short_vln_v1_0006",
)


def _finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return all(not isinstance(item, bool) and math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_policy_requests(requests: Sequence[dict[str, Any]]) -> dict[str, bool]:
    checks = {
        "request_count_positive": bool(requests),
        "request_schema_valid": True,
        "no_oracle_input_leakage": True,
        "responses_valid": True,
        "response_request_id_match": True,
        "latency_within_timeout": True,
    }
    for row in requests:
        request = row.get("request")
        response = row.get("response")
        if not isinstance(request, dict):
            checks["request_schema_valid"] = False
            checks["no_oracle_input_leakage"] = False
        else:
            if FORBIDDEN_REQUEST_FIELDS & set(request):
                checks["no_oracle_input_leakage"] = False
            try:
                validate_request_header(request, payload_size=int(request["jpeg_size"]))
            except (KeyError, TypeError, ValueError):
                checks["request_schema_valid"] = False
        if not isinstance(response, dict):
            checks["responses_valid"] = False
        else:
            try:
                validate_response_header(response)
            except ValueError:
                checks["responses_valid"] = False
            if not isinstance(request, dict) or response.get("request_id") != request.get("request_id"):
                checks["response_request_id_match"] = False
        try:
            latency = float(row["roundtrip_wall_ms"])
            if not math.isfinite(latency) or latency > 10_000.0:
                checks["latency_within_timeout"] = False
        except (KeyError, TypeError, ValueError):
            checks["latency_within_timeout"] = False
    return checks


def summarize_response_actions(
    requests: Sequence[dict[str, Any]], *, execute_steps: int
) -> ActionRangeSummary:
    """Count complete response chunks, excluding no-response inference failures."""
    chunks: list[Sequence[Sequence[float]]] = []
    for row in requests:
        response = row.get("response")
        if isinstance(response, dict) and isinstance(response.get("actions"), list):
            chunks.append(response["actions"])
    return summarize_action_range(chunks, execute_steps=execute_steps)


def _same_vector(actual: Any, expected: Sequence[float]) -> bool:
    if not _finite_vector(actual, len(expected)):
        return False
    return all(math.isclose(float(value), target, rel_tol=0.0, abs_tol=1e-9) for value, target in zip(actual, expected))


def audit_executed_action_mapping(
    records: Sequence[dict[str, Any]], requests: Sequence[dict[str, Any]], *, execute_steps: int
) -> dict[str, bool]:
    """Verify that each executed policy action is the audited response action at its offset."""
    responses: dict[str, Sequence[Sequence[float]]] = {}
    checks = {"executed_actions_match_response": True, "response_request_ids_unique": True}
    for row in requests:
        request = row.get("request")
        response = row.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            checks["executed_actions_match_response"] = False
            continue
        request_id = request.get("request_id")
        actions = response.get("actions")
        if not isinstance(request_id, str) or not isinstance(actions, list):
            checks["executed_actions_match_response"] = False
            continue
        if request_id in responses:
            checks["response_request_ids_unique"] = False
            checks["executed_actions_match_response"] = False
            continue
        responses[request_id] = actions

    for record in records:
        if record.get("action_source") != "smolvla":
            continue
        request_id = record.get("request_id")
        action_offset = record.get("action_offset")
        if (
            not isinstance(request_id, str)
            or isinstance(action_offset, bool)
            or not isinstance(action_offset, int)
            or not 0 <= action_offset < execute_steps
            or request_id not in responses
            or action_offset >= len(responses[request_id])
        ):
            checks["executed_actions_match_response"] = False
            continue
        expected = apply_action_safety(responses[request_id][action_offset])
        if not (
            _same_vector(record.get("raw_action"), expected.raw)
            and _same_vector(record.get("applied_action"), expected.applied)
            and record.get("action_in_range") is expected.in_range
            and record.get("clipped_dimensions") == list(expected.clipped_dimensions)
        ):
            checks["executed_actions_match_response"] = False
    return checks


def summary_range_counts_match(summary: dict[str, Any], action_summary: ActionRangeSummary) -> bool:
    expected = {
        **action_summary.as_dict(),
        "raw_chunk_range_violation_count": action_summary.violating_vectors,
        "executed_action_range_violation_count": action_summary.executed_violating_vectors,
    }
    return all(summary.get(key) == value for key, value in expected.items())


def audit_provenance(
    requests: Sequence[dict[str, Any]],
    oracle: dict[str, Any] | None,
    *,
    expected_checkpoint_sha256: str,
    expected_planner_sha256: str,
    expected_short_dataset_sha256: str,
) -> dict[str, bool]:
    checkpoint_hashes = [
        row["response"].get("checkpoint_sha256")
        for row in requests
        if isinstance(row.get("response"), dict)
    ]
    return {
        "checkpoint_hash_present": bool(checkpoint_hashes),
        "checkpoint_hash_matches": bool(checkpoint_hashes)
        and all(value == expected_checkpoint_sha256 for value in checkpoint_hashes),
        "oracle_available": bool(isinstance(oracle, dict) and oracle.get("available")),
        "planner_hash_matches": bool(
            isinstance(oracle, dict) and oracle.get("planner_sha256") == expected_planner_sha256
        ),
        "short_dataset_hash_matches": bool(
            isinstance(oracle, dict)
            and oracle.get("short_dataset_sha256") == expected_short_dataset_sha256
        ),
    }


def validate_gate_stage(episode_ids: Sequence[str], *, gate_stage: str) -> dict[str, Any]:
    if gate_stage == "first":
        expected = list(FIRST_GATE_EPISODE_IDS)
    elif gate_stage == "full":
        expected = list(FULL_GATE_EPISODE_IDS)
    else:
        raise ValueError(f"unknown gate stage {gate_stage!r}")
    actual = list(episode_ids)
    return {
        "gate_stage": gate_stage,
        "expected_episode_ids": expected,
        "actual_episode_ids": actual,
        "episode_ids_exact": actual == expected,
    }


def evaluate_rollout_records(
    summary: dict[str, Any], records: Sequence[dict[str, Any]], requests: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    request_checks = audit_policy_requests(requests)
    execute_steps = int(summary.get("execute_steps_per_chunk", DEFAULT_EXECUTE_STEPS))
    action_summary = summarize_response_actions(requests, execute_steps=execute_steps)
    indices = [record.get("frame_index") for record in records]
    action_mapping_checks = audit_executed_action_mapping(
        records, requests, execute_steps=execute_steps
    )
    record_checks = {
        "record_count_positive": bool(records),
        "record_summary_count_match": len(records) == int(summary.get("frame_count", -1)),
        "request_summary_count_match": len(requests) == int(summary.get("request_count", -1)),
        "frame_indices_contiguous": indices == list(range(len(records))),
        "state_shape_and_finite": all(_finite_vector(record.get("policy_state"), 30) for record in records),
        "raw_action_shape_and_finite": all(_finite_vector(record.get("raw_action"), 3) for record in records),
        "applied_action_shape_and_finite": all(
            _finite_vector(record.get("applied_action"), 3) for record in records
        ),
        "applied_actions_safe": all(
            _finite_vector(record.get("applied_action"), 3)
            and all(
                lower - 1e-9 <= float(value) <= upper + 1e-9
                for value, (lower, upper) in zip(record["applied_action"], ACTION_BOUNDS)
            )
            for record in records
        ),
        "summary_range_counts_match": summary_range_counts_match(summary, action_summary),
    }
    checks = {**request_checks, **record_checks, **action_mapping_checks}
    infrastructure_passed = all(checks.values())
    executed_range_violations = sum(not bool(record.get("action_in_range")) for record in records)
    official_navigation_success = bool(summary.get("status") == "complete" and summary.get("success"))
    range_compliant_autonomous_success = bool(
        infrastructure_passed and official_navigation_success and action_summary.raw_output_range_passed
    )
    return {
        "checks": checks,
        "infrastructure_passed": infrastructure_passed,
        "official_navigation_success": official_navigation_success,
        "range_compliant_autonomous_success": range_compliant_autonomous_success,
        "valid_autonomous_success": range_compliant_autonomous_success,
        "raw_output_range_passed": action_summary.raw_output_range_passed,
        "raw_action_chunk_count": action_summary.total_chunks,
        "raw_action_vector_count": action_summary.total_vectors,
        "raw_action_vector_violation_count": action_summary.violating_vectors,
        "raw_chunk_with_violation_count": action_summary.chunks_with_violation,
        "executed_policy_action_count": action_summary.executed_vectors,
        "executed_policy_action_violation_count": action_summary.executed_violating_vectors,
        "discarded_action_count": action_summary.tail_vectors,
        "discarded_action_violation_count": action_summary.tail_violating_vectors,
        "raw_action_dimension_violation_count": action_summary.per_dimension,
        "raw_range_violation_count": action_summary.violating_vectors,
        "executed_range_violation_count": executed_range_violations,
    }


def make_plot(episode_dir: Path, records: Sequence[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [float(row["timestamp"]) for row in records]
    raw = [[float(value) for value in row["raw_action"]] for row in records]
    applied = [[float(value) for value in row["applied_action"]] for row in records]
    distances = [float(row["evaluator_distance_to_goal_m"]) for row in records]
    positions = [row["robot_pose"]["position_w"] for row in records]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for index, label in enumerate(("vx", "vy", "wz")):
        axes[0].plot(times, [row[index] for row in raw], label=f"raw {label}", alpha=0.7)
        axes[0].plot(times, [row[index] for row in applied], label=f"applied {label}", linestyle="--")
    axes[0].set_ylabel("velocity command")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(times, distances, color="#c62828")
    axes[1].set_ylabel("goal distance (m)")
    axes[1].grid(alpha=0.25)
    axes[2].plot([float(pos[0]) for pos in positions], [float(pos[1]) for pos in positions], color="#286090")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("world x (m)")
    axes[2].set_ylabel("world y (m)")
    axes[2].grid(alpha=0.25)
    output = episode_dir / "m7_rollout_diagnostics.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def make_video(episode_dir: Path, records: Sequence[dict[str, Any]]) -> tuple[Path, int]:
    import cv2

    first = cv2.imread(str(episode_dir / records[0]["front_rgb"]))
    if first is None:
        raise ValueError("first RGB frame is unreadable")
    height, width = first.shape[:2]
    output = episode_dir / "rollout.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output}")
    for row in records:
        frame = cv2.imread(str(episode_dir / row["front_rgb"]))
        if frame is None or frame.shape[:2] != (height, width):
            writer.release()
            raise ValueError(f"unreadable or mismatched RGB frame {row['front_rgb']}")
        raw = row["raw_action"]
        applied = row["applied_action"]
        label = (
            f"t={float(row['timestamp']):.2f}s raw=[{raw[0]:.2f},{raw[1]:.2f},{raw[2]:.2f}] "
            f"applied=[{applied[0]:.2f},{applied[1]:.2f},{applied[2]:.2f}] "
            f"evaluator_d={float(row['evaluator_distance_to_goal_m']):.2f}m"
        )
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(output))
    decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    if decoded < max(1, len(records) - 1):
        raise ValueError(f"video decode count {decoded} is invalid for {len(records)} frames")
    return output, decoded


def validate_frames(episode_dir: Path, records: Sequence[dict[str, Any]]) -> bool:
    import cv2

    for row in records:
        frame = cv2.imread(str(episode_dir / row["front_rgb"]))
        if frame is None or frame.shape != (512, 512, 3):
            return False
    return True


def check_episode(
    episode_dir: Path,
    oracle_root: Path | None,
    *,
    expected_checkpoint_sha256: str,
    expected_planner_sha256: str,
    expected_short_dataset_sha256: str,
) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
    records = load_jsonl(episode_dir / "steps.jsonl")
    requests = load_jsonl(episode_dir / "requests.jsonl")
    evaluated = evaluate_rollout_records(summary, records, requests)
    artifacts: dict[str, Any] = {"frames_valid": False, "plot": None, "video": None, "video_decoded_frames": 0}
    if records:
        artifacts["frames_valid"] = validate_frames(episode_dir, records)
        if artifacts["frames_valid"]:
            artifacts["plot"] = str(make_plot(episode_dir, records))
            video, decoded = make_video(episode_dir, records)
            artifacts["video"] = str(video)
            artifacts["video_decoded_frames"] = decoded
    artifacts_complete = bool(artifacts["frames_valid"] and artifacts["plot"] and artifacts["video"])
    oracle: dict[str, Any] | None = None
    if oracle_root is not None:
        candidate = oracle_root / str(summary["short_episode_id"]) / "summary.json"
        if candidate.is_file():
            oracle_summary = json.loads(candidate.read_text(encoding="utf-8"))
            oracle = {
                "available": True,
                "success": bool(oracle_summary.get("success")),
                "final_navigation_error_m": oracle_summary.get("final_distance_to_goal_xy_m"),
                "planner_sha256": oracle_summary.get("official_planner_sha256"),
                "short_dataset_sha256": oracle_summary.get("short_dataset_sha256"),
            }
        else:
            oracle = {"available": False}
    provenance_checks = audit_provenance(
        requests,
        oracle,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_planner_sha256=expected_planner_sha256,
        expected_short_dataset_sha256=expected_short_dataset_sha256,
    )
    checks = {**evaluated["checks"], **provenance_checks}
    range_compliant_autonomous_success = bool(
        all(checks.values())
        and evaluated["official_navigation_success"]
        and evaluated["raw_output_range_passed"]
    )
    result = {
        "episode_id": summary["short_episode_id"],
        "episode_dir": str(episode_dir),
        "infrastructure_passed": all(checks.values()),
        "official_navigation_success": evaluated["official_navigation_success"],
        "range_compliant_autonomous_success": range_compliant_autonomous_success,
        "valid_autonomous_success": range_compliant_autonomous_success,
        "raw_output_range_passed": evaluated["raw_output_range_passed"],
        "raw_action_chunk_count": evaluated["raw_action_chunk_count"],
        "raw_action_vector_count": evaluated["raw_action_vector_count"],
        "raw_action_vector_violation_count": evaluated["raw_action_vector_violation_count"],
        "raw_chunk_with_violation_count": evaluated["raw_chunk_with_violation_count"],
        "executed_policy_action_count": evaluated["executed_policy_action_count"],
        "executed_policy_action_violation_count": evaluated[
            "executed_policy_action_violation_count"
        ],
        "discarded_action_count": evaluated["discarded_action_count"],
        "discarded_action_violation_count": evaluated["discarded_action_violation_count"],
        "raw_action_dimension_violation_count": evaluated["raw_action_dimension_violation_count"],
        "raw_range_violation_count": evaluated["raw_range_violation_count"],
        "executed_range_violation_count": evaluated["executed_range_violation_count"],
        "checks": checks,
        "artifacts": artifacts,
        "artifacts_complete": artifacts_complete,
        "summary": summary,
        "pd_oracle": oracle,
    }
    (episode_dir / "sanity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--gate-stage", choices=("first", "full"), required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_M6_CHECKPOINT_SHA256)
    parser.add_argument("--expected-planner-sha256", default=EXPECTED_M4_PLANNER_SHA256)
    parser.add_argument("--expected-short-dataset-sha256", default=EXPECTED_SHORT_DATASET_SHA256)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    return parser.parse_args()


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for episode in report["episodes"]:
        summary = episode["summary"]
        rows.append(
            "| {id} | {infra} | {official} | {range_ok} | {compliant} | {violations} | {error} | {latency} | {artifacts} |".format(
                id=episode["episode_id"],
                infra="PASS" if episode["infrastructure_passed"] else "FAIL",
                official="PASS" if episode["official_navigation_success"] else "FAIL",
                range_ok="PASS" if episode["raw_output_range_passed"] else "FAIL",
                compliant=(
                    "PASS" if episode["range_compliant_autonomous_success"] else "FAIL"
                ),
                violations=(
                    f"{episode['raw_action_vector_violation_count']} / "
                    f"{episode['raw_chunk_with_violation_count']}"
                ),
                error=(
                    f"{float(summary['final_navigation_error_m']):.3f}"
                    if summary.get("final_navigation_error_m") is not None
                    else "N/A"
                ),
                latency=(
                    f"{float(summary['mean_inference_roundtrip_ms']):.1f}"
                    if summary.get("mean_inference_roundtrip_ms") is not None
                    else "N/A"
                ),
                artifacts="PASS" if episode["artifacts_complete"] else "FAIL",
            )
        )
    return (
        "# M7 Closed-Loop Gate Report\n\n"
        "| Episode | Integration | Official success | Raw range | Range-compliant success | Raw vectors / chunks | Final error (m) | Mean latency (ms) | Artifacts |\n"
        "|---|---|---|---|---|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n\nOverall M7 gate: **{}**.\n".format("PASS" if report["passed"] else "FAIL")
    )


def main() -> None:
    args = parse_args()
    results = [
        check_episode(
            path,
            args.oracle_root,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_planner_sha256=args.expected_planner_sha256,
            expected_short_dataset_sha256=args.expected_short_dataset_sha256,
        )
        for path in args.episode_dirs
    ]
    stage_checks = validate_gate_stage(
        [item["episode_id"] for item in results], gate_stage=args.gate_stage
    )
    report = {
        "format": "go2-short-vln-m7-gate-v1",
        "episode_count": len(results),
        "infrastructure_passed": all(item["infrastructure_passed"] for item in results),
        "artifacts_complete": all(item["artifacts_complete"] for item in results),
        "official_navigation_success_count": sum(
            item["official_navigation_success"] for item in results
        ),
        "range_compliant_autonomous_success_count": sum(
            item["range_compliant_autonomous_success"] for item in results
        ),
        "stage_checks": stage_checks,
        "episodes": results,
    }
    report["valid_autonomous_success_count"] = report[
        "range_compliant_autonomous_success_count"
    ]
    report["passed"] = bool(
        report["infrastructure_passed"]
        and report["artifacts_complete"]
        and report["stage_checks"]["episode_ids_exact"]
        and report["valid_autonomous_success_count"] >= 1
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
