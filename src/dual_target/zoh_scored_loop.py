"""V2 ZOH loop: fresh expert commands at 5 Hz, physical scoring at 50 Hz."""
from __future__ import annotations
import argparse
import math
from .zoh_control import ZohClock, ZohSpec, probe_command
from .zoh_scoring import ZohActionAudit, ZohScoreFrame
from .expert import ParkingExpert
from .layouts import Vec2
from .runner import _yaw_from_wxyz
import json
from pathlib import Path
from dataclasses import asdict
from typing import Any
from .runner import (
    SharedSmokeResourceMonitor, DualTargetSceneSpec, _scene_for_task,
    build_live_low_level_runtime, MotionCalibration, load_motion_calibration,
    ParkingThresholds, EpisodeEvidenceWriter, PHYSICS_DT_S, DECIMATION,
    PAIR_SEED_ALGORITHM, SubstepContactLatch, PreResetEvidenceCapture,
    _default_runtime_snapshot, _target_contact_force, AutonomousParkingScorer,
    validate_live_target_contact_sensors, _render_without_physics, _camera_rgb,
    camera_audit_fields, Dt1RunnerError, _write_rgb, _paired_reset_physical_state,
    GO2_WARMUP_STEPS, _bool_at_zero, verify_pause_does_not_advance_physics,
    _write_paired_reset_audit, apply_deployment_bounds, _inside, _fallen_from_pose,
    ScoreFrame, ScoreStatus,
)

def run_live_zoh(
    args: argparse.Namespace,
    simulation_app: Any,
    *,
    resource_monitor: SharedSmokeResourceMonitor | None = None,
    scene_task: tuple[DualTargetSceneSpec, Any],
    policy_client=None,
    mode="expert",
    spec=ZohSpec(),
) -> dict[str, object]:
    """Reviewed DT1 physical recording/scoring loop with RGB-only model control."""
    if mode not in ("expert", "policy", "probe") or (mode == "policy" and policy_client is None):
        raise ValueError("explicit ZOH source required")
    if mode == "policy" and (policy_client.metadata.get("chunk_size") != spec.chunk_size
            or policy_client.metadata.get("execute_steps") != spec.execute_steps
            or policy_client.metadata.get("action_dt_s") != spec.action_dt_s):
        raise ValueError("model client must explicitly match the new ZOH action semantics")
    zoh = ZohClock(spec)
    action_audit = ZohActionAudit(spec)
    expert = ParkingExpert() if mode == "expert" else None
    phase = "policy" if mode == "policy" else "expert"
    scene, task = scene_task if scene_task is not None else _scene_for_task(args.group_id, args.color_configuration, args.target_color)
    live_runtime = build_live_low_level_runtime(args, scene)
    checkpoint = live_runtime.checkpoint
    checkpoint_hash = live_runtime.checkpoint_hash
    scene_metadata = live_runtime.scene_metadata
    history_env = live_runtime.history_env
    adapter = live_runtime.adapter
    episode_id = f"{scene.group.geometry_group_id}_{scene.color_configuration}_{task.target_color}_r{args.repeat:02d}"
    episode_root = Path(args.output_dir).resolve() / args.run_id / "episodes" / episode_id
    high_log = None
    calibration: MotionCalibration | None = None
    calibration_hash = ""
    calibration_path = getattr(args, "motion_calibration", None)
    if calibration_path is not None:
        calibration, calibration_hash = load_motion_calibration(calibration_path)
        thresholds = ParkingThresholds(
            body_linear_speed_mps=calibration.derived_body_planar_speed_mps,
            body_yaw_rate_radps=calibration.derived_body_yaw_rate_radps,
        )
    else:
        thresholds = ParkingThresholds()
    writer = EpisodeEvidenceWriter(episode_root, {
        **scene_metadata,
        "episode_id": episode_id,
        "seed": args.seed,
        "paired_reset_required": bool(args.paired_reset),
        "pair_seed_algorithm": PAIR_SEED_ALGORITHM if args.paired_reset else "",
        "repeat": args.repeat,
        "target_color": task.target_color,
        "target_slot": task.target_slot,
        "instruction": task.instruction,
        "physics_dt_s": PHYSICS_DT_S,
        "decimation": DECIMATION,
        "thresholds": asdict(thresholds),
        "motion_calibration": calibration.as_dict() if calibration is not None else None,
        "motion_calibration_path": str(Path(calibration_path).resolve()) if calibration_path is not None else "",
        "motion_calibration_sha256": calibration_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "agent_policy_history_length": live_runtime.policy_history_length,
        "policy_phase": phase,
        "zoh_source": mode,
        "scoring_action_validation": "independent_5hz_slew_and_50hz_hold_v1",
        "zoh_contract": spec.metadata(),
        "learner_inputs": ["RGB", "body_vx", "body_vy", "body_yaw_rate", "task"],
        "truth_is_audit_only": mode != "expert",
        "seed_protocol": "python_random,numpy,torch,ManagerBasedEnv.seed set before gym.make/reset",
        "physics_clock_source": "ManagerBasedRLEnv._sim_step_counter verified to increment inside each physics substep",
        "fall_detection": {"min_root_height_m": 0.18, "max_abs_roll_or_pitch_rad": 0.85, "env_terminated_is_fall": True},
        "model_controller": policy_client.metadata if mode == "policy" else {"source": mode},
    })
    high_log = (episode_root / "high_level_pre_action.jsonl").open("x", buffering=1)
    base_env = adapter.base_env
    latch = SubstepContactLatch(threshold_n=args.contact_threshold_n)
    snapshot_capture = PreResetEvidenceCapture(
        base_env,
        snapshot=lambda env: _default_runtime_snapshot(env),
        on_substep=lambda: latch.capture({
            "red": _target_contact_force(base_env.scene["red_target_contacts"], color="red"),
            "blue": _target_contact_force(base_env.scene["blue_target_contacts"], color="blue"),
        }),
    )
    scorer = AutonomousParkingScorer(thresholds)
    wrong_target_scorer = AutonomousParkingScorer(thresholds)
    render_request_seq = observation_seq = 0
    pre_reset_index = 0
    paired_reset_audit_path = ""
    paired_reset_audit_hash = ""
    result: dict[str, object] = {"episode_id": episode_id, "status": "FAILED", "reason": "not_started"}
    try:
        if resource_monitor is not None:
            resource_monitor.check()
        adapter.reset()
        contact_view_contract = validate_live_target_contact_sensors(base_env)
        (episode_root / "target_contact_view_contract.json").write_text(
            json.dumps(contact_view_contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        initial_clock = adapter.live_clock()
        (episode_root / "runtime_clock_initial.json").write_text(json.dumps({
            "manager_sim_step": initial_clock.manager_step, "simulation_context_step": initial_clock.sim_step,
            "simulation_context_time_s": initial_clock.sim_time_s,
            "frozen_manager_to_sim_step_offset": initial_clock.sim_step - initial_clock.manager_step,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        camera = base_env.scene["rgbd_camera"]

        def capture_rgb(kind: str) -> dict[str, Any]:
            """Request/render one audit observation without advancing physics."""
            nonlocal render_request_seq, observation_seq
            before_clock, after_clock, _, _ = _render_without_physics(base_env)
            render_request_seq += 1
            rgb = _camera_rgb(camera)
            fields = camera_audit_fields(camera)
            if fields["camera_sensor_frame"] is None or fields["camera_timestamp_s"] is None:
                raise Dt1RunnerError("camera audit bookkeeping is unavailable")
            observation_seq += 1
            relative = f"rgb_{kind}/{observation_seq:06d}.png"
            _write_rgb(episode_root / relative, rgb)
            return {
                "observation_seq": observation_seq, "render_request_seq": render_request_seq,
                "rgb_path": relative, "render_physics_step_before": before_clock.sim_step,
                "render_physics_step_after": after_clock.sim_step,
                "render_sim_time_before_s": before_clock.sim_time_s,
                "render_sim_time_after_s": after_clock.sim_time_s,
                "render_did_not_advance_physics": True,
                **fields,
            }

        reset_state: dict[str, object] | None = None
        reset_observation: dict[str, Any] | None = None
        if args.paired_reset:
            reset_state = _paired_reset_physical_state(base_env, history_env)
            reset_observation = capture_rgb("reset")
            reset_observation["reset_physics_step"] = initial_clock.sim_step
            reset_observation["reset_sim_time_s"] = initial_clock.sim_time_s

        snapshot_capture.install()
        # Legacy Go2 warmup is evidence only.  It never reaches scorer/pre-action logs.
        for warmup_step in range(GO2_WARMUP_STEPS):
            before = adapter.physics_step
            _, _, done, _, _, hits, forces = adapter.step((0.0, 0.0, 0.0), latch)
            if resource_monitor is not None:
                resource_monitor.check()
            if any(hits) or _bool_at_zero(done, "warmup_done") or snapshot_capture.records:
                raise Dt1RunnerError("target contact or auto-reset occurred during warmup")
            writer.warmup({
                "event": "warmup", "warmup_step": warmup_step,
                "sim_time_s": adapter.sim_time_s,
                "physics_step": adapter.physics_step, "command": [0.0, 0.0, 0.0],
            })
        if snapshot_capture.records:
            raise Dt1RunnerError("environment auto-reset during warmup; no scored episode may continue")
        pause_probe = verify_pause_does_not_advance_physics(base_env, duration_s=1.0)
        (episode_root / "pause_no_physics.json").write_text(
            json.dumps(pause_probe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # The initial observation is the RGB/state directly before its action.
        # Each following pre-action record reuses the immediately post-action
        # observation from the preceding action; no fake time or camera change
        # is invented between them.
        observation = capture_rgb("pre")
        if args.paired_reset:
            if reset_state is None or reset_observation is None:
                raise Dt1RunnerError("paired-reset audit lost its reset state before learner start")
            learner_state = _default_runtime_snapshot(base_env)
            observation["learner_start_physics_step"] = adapter.physics_step
            observation["learner_start_sim_time_s"] = adapter.sim_time_s
            audit_path, audit_hash = _write_paired_reset_audit(
                episode_root,
                args=args,
                reset_physical_state=reset_state,
                reset_observation=reset_observation,
                learner_physical_state=learner_state,
                learner_observation=observation,
                scene_metadata=scene_metadata,
            )
            paired_reset_audit_path = str(audit_path)
            paired_reset_audit_hash = audit_hash
        for step_index in range(args.max_env_steps):
            if resource_monitor is not None:
                resource_monitor.check()
            before_snapshot = _default_runtime_snapshot(base_env)
            clock_before_inference = adapter.live_clock()
            fresh = zoh.needs_command
            proposal = None
            if fresh:
                if mode == "policy":
                    proposal = policy_client.action(
                        episode_root / observation["rgb_path"],
                        before_snapshot["body_velocity_body"], task.instruction)
                elif mode == "expert":
                    pose = before_snapshot["robot_pose_w"]
                    target_slot = scene.group.slot_a if task.target_slot == "A" else scene.group.slot_b
                    proposal = expert.command(
                        robot_xy=Vec2(pose[0], pose[1]),
                        robot_yaw_rad=_yaw_from_wxyz(pose[3:]),
                        parking_center=task.parking_region.center,
                        terminal_heading_rad=math.atan2(-target_slot.front_unit.y, -target_slot.front_unit.x),
                    ).as_list()
                else:
                    proposal = probe_command(step_index // spec.hold_control_steps)
            raw_command, held_command = zoh.advance(proposal)
            command = list(held_command)
            expected_applied = action_audit.observe(raw_command, held_command, step_index)
            if fresh:
                high_log.write(json.dumps({
                    "high_level_index": step_index // spec.hold_control_steps,
                    "low_level_start_index": step_index,
                    "sim_time_s": adapter.sim_time_s, "physics_step": adapter.physics_step,
                    "rgb_path": observation["rgb_path"],
                    "body_velocity_body": before_snapshot["body_velocity_body"],
                    "task": task.instruction, "raw_action": raw_command,
                    "applied_action": held_command, "nominal_hold_steps": spec.hold_control_steps,
                }, allow_nan=False) + "\n")
            if adapter.live_clock() != clock_before_inference:
                raise Dt1RunnerError("physical clock advanced during model wait")
            physics_before = adapter.physics_step
            time_before = adapter.sim_time_s
            body_before = before_snapshot["body_velocity_body"]
            writer.pre_action({
                "event": "pre_action", "phase": phase, "sim_time_s": time_before,
                "physics_step": physics_before, "observation_seq": observation["observation_seq"],
                "camera_sensor_frame": observation["camera_sensor_frame"],
                "camera_timestamp_s": observation["camera_timestamp_s"],
                "render_request_seq": observation["render_request_seq"], "rgb_path": observation["rgb_path"],
                "body_velocity_body": body_before, "raw_action": list(raw_command),
                "applied_action": list(apply_deployment_bounds(command).applied), "task": task.instruction,
            })
            applied, _, done, _, _, hits, forces = adapter.step(command, latch)
            if tuple(applied.applied) != tuple(held_command):
                raise Dt1RunnerError("deployment changed shared ZOH boundary output")
            if resource_monitor is not None:
                resource_monitor.check()
            post_snapshot = _default_runtime_snapshot(base_env)
            pre_reset_ref = ""
            if snapshot_capture.records:
                # ManagerBasedRLEnv has already reset before its public step
                # returns.  Persist the hook's pre-reset values and use them
                # for this terminal post-step instead of reset pose/velocity.
                snapshot = snapshot_capture.records.pop(0)
                pre_reset_ref = f"pre_reset_{pre_reset_index:04d}.json"
                (episode_root / pre_reset_ref).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                pre_reset_index += 1
                post_snapshot = snapshot
            # This is a genuinely post-action renderer request.  Its RGB is
            # retained even for terminal failures, but never used as the
            # current action's learner input.
            post_observation = capture_rgb("post")
            correct = _inside(task.parking_region.center, scene.group.parking_radius_m, post_snapshot["robot_pose_w"])
            other_slot = "B" if task.target_slot == "A" else "A"
            other = _inside(scene.group.parking_region(other_slot).center, scene.group.parking_radius_m, post_snapshot["robot_pose_w"])
            body = post_snapshot["body_velocity_body"]
            terminated = bool(post_snapshot["terminated"])
            truncated = bool(post_snapshot["truncated"])
            environment_done = _bool_at_zero(done, "low_level_done")
            fallen = _fallen_from_pose(post_snapshot["robot_pose_w"]) or terminated
            # An environment reset/timeout is terminal failure evidence before
            # the scorer sees this frame.  It therefore cannot be re-labelled
            # as success merely because a pre-reset pose happened to be parked.
            evaluator_stop = truncated or (environment_done and not fallen)
            # Wrong target is a failure only after the same real, continuous
            # one-second autonomous stop evidence used for the correct target.
            # A transient zero command near the other box cannot end a task.
            wrong_decision = wrong_target_scorer.observe(ZohScoreFrame(expected_applied=expected_applied,
                sim_time_s=adapter.sim_time_s, physics_step=adapter.physics_step,
                observation_seq=post_observation["observation_seq"], raw_action=raw_command, applied_action=applied.applied,
                body_vx_mps=float(body[0]), body_vy_mps=float(body[1]), body_yaw_rate_radps=float(body[2]),
                in_correct_parking_region=other, warmup=False, collision=any(hits), fallen=fallen,
                wrong_target_stop=False, evaluator_stop=evaluator_stop,
            ))
            wrong_stop = wrong_decision.status is ScoreStatus.SUCCESS
            frame = ZohScoreFrame(expected_applied=expected_applied,
                sim_time_s=adapter.sim_time_s, physics_step=adapter.physics_step,
                observation_seq=post_observation["observation_seq"], raw_action=raw_command, applied_action=applied.applied,
                body_vx_mps=float(body[0]), body_vy_mps=float(body[1]), body_yaw_rate_radps=float(body[2]),
                in_correct_parking_region=correct, warmup=False, collision=any(hits), fallen=fallen,
                wrong_target_stop=wrong_stop, evaluator_stop=evaluator_stop,
            )
            decision = scorer.observe(frame)
            if decision.status is ScoreStatus.INVALID_SAMPLE or wrong_decision.status is ScoreStatus.INVALID_SAMPLE:
                raise Dt1RunnerError("ZOH scoring invalid sample: "+decision.detail+" / "+wrong_decision.detail)
            writer.post_step({
                "event": "post_step", "phase": phase, "sim_time_before_s": time_before,
                "sim_time_after_s": adapter.sim_time_s,
                "physics_step_before": physics_before, "physics_step_after": adapter.physics_step,
                "observation_seq_after": post_observation["observation_seq"],
                "camera_sensor_frame": post_observation["camera_sensor_frame"],
                "camera_timestamp_s": post_observation["camera_timestamp_s"],
                "render_request_seq": post_observation["render_request_seq"],
                "robot_pose_w": post_snapshot["robot_pose_w"], "body_velocity_body": body,
                "collision_latched_substeps": hits, "target_contact_force_max": forces,
                "in_correct_parking_region": correct, "in_other_parking_region": other, "fallen": fallen,
                "terminated": terminated, "truncated": truncated,
                "evaluator_stop": evaluator_stop, "wrong_target_stop": wrong_stop, "warmup": False,
                "pre_reset_snapshot_ref": pre_reset_ref, "scorer_status": decision.status.value, "scorer_detail": decision.detail,
            })
            if decision.terminal:
                result = {"episode_id": episode_id, "status": decision.status.value, "reason": decision.detail,
                          "steps": step_index + 1, "sim_time_s": adapter.sim_time_s}
                break
            if environment_done:
                result = {"episode_id": episode_id, "status": "FAILED_AUTO_RESET", "reason": "environment_done_before_scorer_success"}
                break
            observation = post_observation
        else:
            result = {"episode_id": episode_id, "status": "FAILED_TIMEOUT", "reason": "model did not finish before max_env_steps"}
        if paired_reset_audit_path:
            result["paired_reset_audit_path"] = paired_reset_audit_path
            result["paired_reset_audit_sha256"] = paired_reset_audit_hash
            result["pair_seed"] = args.seed
            result["pair_seed_algorithm"] = PAIR_SEED_ALGORITHM
        if mode == "probe" and result["status"] == "FAILED_TIMEOUT":
            result = {"episode_id": episode_id, "status": "PROBE_SEQUENCE_COMPLETE_NOT_APPROVED",
                      "steps": args.max_env_steps}
        result["zoh_source"] = mode
        result["high_level_command_count"] = (zoh.step + spec.hold_control_steps - 1) // spec.hold_control_steps
        result["complete_hold_count"] = zoh.step // spec.hold_control_steps
        return result
    finally:
        if high_log is not None:
            high_log.close()
        snapshot_capture.restore()
        history_env.close()
