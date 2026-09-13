"""Bounded DT1 motion and real zero-command calibration probe.

This is a physics/control diagnostic, not a navigation episode.  It sends a
small fixed sequence through the audited old low-level adapter and records
actual clocks, body telemetry and target-contact latches.  Its result can
calibrate the parking scorer's body-speed thresholds, but never records a
task success or approves DT1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .calibration import (
    DECIMATION,
    ENV_STEP_S,
    MotionCalibrationError,
    ZERO_SETTLE_CASES,
    derive_motion_calibration,
)
from .runner import (
    GO2_WARMUP_STEPS,
    Dt1RunnerError,
    SharedSmokeResourceMonitor,
    _bool_at_zero,
    _default_runtime_snapshot,
    _fallen_from_pose,
    _scene_for_task,
    _target_contact_force,
    _yaw_from_wxyz,
    build_live_low_level_runtime,
    validate_live_target_contact_sensors,
)
from .runtime import PreResetEvidenceCapture, SubstepContactLatch


INITIAL_ZERO_STEPS = 75
FORWARD_STEPS = 75
RETURN_ZERO_STEPS = 100
TURN_STEPS = 75
# Exercise each actual deployment bound, not only a convenient interior
# command.  The selected 1.5s segment remains well before the dev000 boxes.
FORWARD_COMMAND = (0.50, 0.0, 0.0)
LEFT_TURN_COMMAND = (0.0, 0.0, 0.50)
RIGHT_TURN_COMMAND = (0.0, 0.0, -0.50)
MIN_FORWARD_DISPLACEMENT_M = 0.08
MIN_FORWARD_BODY_VX_MPS = 0.05
MIN_TURN_ABS_YAW_RAD = 0.12


class MotionPrecheckError(Dt1RunnerError):
    """The bounded physical motion probe cannot make trustworthy claims."""


@dataclass(frozen=True)
class MotionCase:
    name: str
    command: tuple[float, float, float]
    steps: int


MOTION_CASES = (
    MotionCase("initial_zero", (0.0, 0.0, 0.0), INITIAL_ZERO_STEPS),
    MotionCase("forward", FORWARD_COMMAND, FORWARD_STEPS),
    MotionCase("forward_return_zero", (0.0, 0.0, 0.0), RETURN_ZERO_STEPS),
    MotionCase("left_turn", LEFT_TURN_COMMAND, TURN_STEPS),
    MotionCase("left_return_zero", (0.0, 0.0, 0.0), RETURN_ZERO_STEPS),
    MotionCase("right_turn", RIGHT_TURN_COMMAND, TURN_STEPS),
    MotionCase("right_return_zero", (0.0, 0.0, 0.0), RETURN_ZERO_STEPS),
)


class _MotionEvidence:
    def __init__(self, root: Path, manifest: dict[str, object]) -> None:
        if root.exists():
            raise FileExistsError(f"refusing to overwrite motion-precheck evidence: {root}")
        self.root = root
        self.events_path = root / "motion_precheck_events.jsonl"
        self.root.mkdir(parents=True)
        (root / "motion_precheck_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self._pre_reset_index = 0

    def event(self, payload: dict[str, object]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def pre_reset(self, payload: dict[str, object]) -> str:
        relative = f"pre_reset_{self._pre_reset_index:04d}.json"
        self._pre_reset_index += 1
        (self.root / relative).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return relative

    def result(self, payload: dict[str, object]) -> None:
        (self.root / "motion_precheck_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _target_poses(base_env: Any) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for color in ("red", "blue"):
        raw = base_env.scene[f"{color}_target"].data.root_state_w[0, :7]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu()
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if not isinstance(raw, list) or len(raw) != 7:
            raise MotionPrecheckError(f"{color} target pose is unavailable")
        pose: list[float] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MotionPrecheckError(f"{color} target pose is non-finite")
            pose.append(float(value))
        result[color] = pose
    return result


def _motion_snapshot(base_env: Any) -> dict[str, object]:
    return {**_default_runtime_snapshot(base_env), "target_poses_w": _target_poses(base_env)}


def _assert_motion_response(events_by_case: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Check commanded forward/left/right produce measured, signed response."""
    reference_targets: dict[str, list[float]] | None = None
    for case in MOTION_CASES:
        events = events_by_case.get(case.name, [])
        if len(events) != case.steps:
            raise MotionPrecheckError(f"{case.name} did not retain every commanded environment step")
        for event in events:
            if event["raw_action"] != list(case.command) or event["applied_action"] != list(case.command):
                raise MotionPrecheckError(f"{case.name} command did not survive the fixed deployment mapping")
            if event["environment_done"] is not False or event["pre_reset_snapshot_refs"] != []:
                raise MotionPrecheckError(f"{case.name} has done/reset evidence")
            hits = event["collision_latched_substeps"]
            if (not isinstance(hits, list) or len(hits) != DECIMATION
                    or any(type(hit) is not bool for hit in hits) or any(hits)):
                raise MotionPrecheckError(f"{case.name} contacted a target; motion calibration is invalid")
            substeps = event.get("substeps")
            if not isinstance(substeps, list) or len(substeps) != DECIMATION:
                raise MotionPrecheckError(f"{case.name} lacks four physical substep records")
            if [substep.get("collision_latched") for substep in substeps] != hits:
                raise MotionPrecheckError(f"{case.name} substep collision records disagree with its latch")
            targets = event.get("target_poses_w")
            if not isinstance(targets, dict) or set(targets) != {"red", "blue"}:
                raise MotionPrecheckError(f"{case.name} lacks both physical target poses")
            if reference_targets is None:
                reference_targets = targets
            if any(
                not isinstance(targets[color], list) or len(targets[color]) != 7
                or any(not math.isclose(float(current), float(initial), rel_tol=0.0, abs_tol=1e-6)
                       for current, initial in zip(targets[color], reference_targets[color]))
                for color in ("red", "blue")
            ):
                raise MotionPrecheckError("motion probe targets moved or teleported")
    forward = events_by_case["forward"]
    start = forward[0]["robot_pose_before"]
    end = forward[-1]["robot_pose_w"]
    displacement = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    mean_forward_vx = sum(float(event["body_velocity_body"][0]) for event in forward) / len(forward)
    if displacement < MIN_FORWARD_DISPLACEMENT_M or mean_forward_vx < MIN_FORWARD_BODY_VX_MPS:
        raise MotionPrecheckError("forward command did not create the required real forward response")

    def signed_turn(case: str) -> float:
        events = events_by_case[case]
        first = events[0]["robot_pose_before"]
        last = events[-1]["robot_pose_w"]
        return math.atan2(
            math.sin(_yaw_from_wxyz(last[3:]) - _yaw_from_wxyz(first[3:])),
            math.cos(_yaw_from_wxyz(last[3:]) - _yaw_from_wxyz(first[3:])),
        )

    left_delta = signed_turn("left_turn")
    right_delta = signed_turn("right_turn")
    if left_delta < MIN_TURN_ABS_YAW_RAD or right_delta > -MIN_TURN_ABS_YAW_RAD:
        raise MotionPrecheckError("left/right commands did not create the required signed yaw responses")
    return {
        "forward_displacement_m": displacement,
        "forward_mean_body_vx_mps": mean_forward_vx,
        "left_yaw_delta_rad": left_delta,
        "right_yaw_delta_rad": right_delta,
    }


def run_live_motion_precheck(
    args: Any,
    simulation_app: Any,
    *,
    resource_monitor: SharedSmokeResourceMonitor | None = None,
) -> dict[str, object]:
    """Run the fixed physical primitive sequence and write calibration evidence."""
    del simulation_app
    scene, _ = _scene_for_task(args.group_id, args.color_configuration, "red")
    runtime = build_live_low_level_runtime(args, scene)
    adapter = runtime.adapter
    base_env = adapter.base_env
    root = Path(args.output_dir).resolve() / args.run_id / "motion_precheck"
    evidence = _MotionEvidence(root, {
        "format": "go2-dual-target-dt1-motion-precheck-v1",
        "run_id": args.run_id,
        "physics_dt_s": 0.005,
        "decimation": DECIMATION,
        "warmup_steps": GO2_WARMUP_STEPS,
        "cases": [
            {"name": case.name, "command": list(case.command), "steps": case.steps}
            for case in MOTION_CASES
        ],
        "checkpoint_path": str(runtime.checkpoint),
        "checkpoint_sha256": runtime.checkpoint_hash,
        "agent_policy_history_length": runtime.policy_history_length,
        "scene": runtime.scene_metadata,
        "navigation_success_approved": False,
        "dt1_approved": False,
    })
    latch = SubstepContactLatch(threshold_n=args.contact_threshold_n)
    substep_state: dict[str, object] = {"record": False, "substeps": []}

    def capture_substep() -> None:
        maxima = {
            "red": _target_contact_force(base_env.scene["red_target_contacts"], color="red"),
            "blue": _target_contact_force(base_env.scene["blue_target_contacts"], color="blue"),
        }
        hit = latch.capture(maxima)
        if substep_state["record"] is True:
            clock = adapter.live_clock()
            substep_state["substeps"].append({
                "physics_step": clock.sim_step,
                "sim_time_s": clock.sim_time_s,
                "target_contact_force_max": maxima,
                "collision_latched": hit,
            })

    capture = PreResetEvidenceCapture(
        base_env,
        snapshot=lambda env: _motion_snapshot(env),
        on_substep=capture_substep,
    )
    events_by_case: dict[str, list[dict[str, object]]] = {case.name: [] for case in MOTION_CASES}
    result: dict[str, object] = {
        "format": "go2-dual-target-dt1-motion-precheck-result-v1",
        "run_id": args.run_id,
        "status": "MOTION_PRECHECK_FAILED",
        "navigation_success_approved": False,
        "dt1_approved": False,
    }
    try:
        if resource_monitor is not None:
            resource_monitor.check()
        adapter.reset()
        contract = validate_live_target_contact_sensors(base_env)
        (root / "target_contact_view_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        initial_targets = _target_poses(base_env)
        capture.install()
        for _ in range(GO2_WARMUP_STEPS):
            substep_state["record"] = False
            substep_state["substeps"] = []
            _, _, done, _, _, hits, _ = adapter.step((0.0, 0.0, 0.0), latch)
            if resource_monitor is not None:
                resource_monitor.check()
            if any(hits) or _bool_at_zero(done, "motion_warmup_done") or capture.records:
                raise MotionPrecheckError("target contact or reset during motion-precheck warmup")

        for case in MOTION_CASES:
            for case_step in range(case.steps):
                if resource_monitor is not None:
                    resource_monitor.check()
                substep_state["record"] = True
                substep_state["substeps"] = []
                before_clock = adapter.live_clock()
                before = _motion_snapshot(base_env)
                applied, _, done, _, _, hits, forces = adapter.step(case.command, latch)
                if resource_monitor is not None:
                    resource_monitor.check()
                after_clock = adapter.live_clock()
                post = _motion_snapshot(base_env)
                pre_reset_refs: list[str] = []
                while capture.records:
                    pre_reset_refs.append(evidence.pre_reset(capture.records.pop(0)))
                substeps = substep_state["substeps"]
                if not isinstance(substeps, list) or len(substeps) != DECIMATION:
                    raise MotionPrecheckError("motion precheck did not retain exactly four physical substep records")
                for offset, substep in enumerate(substeps, 1):
                    if (substep["physics_step"] != before_clock.sim_step + offset
                            or not math.isclose(substep["sim_time_s"] - before_clock.sim_time_s, offset * 0.005,
                                                rel_tol=0.0, abs_tol=2e-6)):
                        raise MotionPrecheckError("motion precheck substep clock does not match the real physics grid")
                event: dict[str, object] = {
                    "event": "motion_step",
                    "case": case.name,
                    "case_step": case_step,
                    "raw_action": list(applied.raw),
                    "applied_action": list(applied.applied),
                    "sim_time_before_s": before_clock.sim_time_s,
                    "sim_time_after_s": after_clock.sim_time_s,
                    "physics_step_before": before_clock.sim_step,
                    "physics_step_after": after_clock.sim_step,
                    "robot_pose_before": before["robot_pose_w"],
                    "robot_pose_w": post["robot_pose_w"],
                    "body_velocity_body": post["body_velocity_body"],
                    "target_poses_w": post["target_poses_w"],
                    "initial_target_poses_w": initial_targets,
                    "collision_latched_substeps": hits,
                    "target_contact_force_max": forces,
                    "substeps": substeps,
                    "environment_done": _bool_at_zero(done, "motion_done"),
                    "warmup": False,
                    "pre_reset_snapshot_refs": pre_reset_refs,
                }
                events_by_case[case.name].append(event)
                evidence.event(event)
                if (any(hits) or event["environment_done"] or pre_reset_refs
                        or _fallen_from_pose(post["robot_pose_w"])):
                    raise MotionPrecheckError(f"{case.name} ended with collision, done/reset, or fall evidence")
        response = _assert_motion_response(events_by_case)
        try:
            calibration = derive_motion_calibration({case: events_by_case[case] for case in ZERO_SETTLE_CASES})
        except MotionCalibrationError as exc:
            raise MotionPrecheckError(f"zero-command calibration rejected: {exc}") from exc
        calibration_path = root / "motion_calibration.json"
        calibration_path.write_text(
            json.dumps(calibration.as_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        result.update({
            "status": "MOTION_PRECHECK_PASSED",
            "motion_response": response,
            "motion_calibration_path": str(calibration_path),
            "motion_calibration": calibration.as_dict(),
            "scope": "physical command-direction and zero-command calibration only; no task success or DT1 approval",
        })
        evidence.result(result)
        return result
    except BaseException as exc:
        result.update({"status": "MOTION_PRECHECK_FAILED", "reason": f"{type(exc).__name__}: {exc}", "evidence_dir": str(root)})
        evidence.result(result)
        raise
    finally:
        capture.restore()
        runtime.history_env.close()
