"""Bounded physical contact diagnostic for DT1; never a navigation episode.

Each invocation proves one color only: 50 stationary steps must show ordinary
foot-to-ground force but zero target force, then a straight low-level command
must physically contact the selected colored box within 300 environment steps.
The other color is kept as a real scene body and must remain below threshold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from .contracts import AppliedAction
from .runner import (
    DECIMATION,
    GO2_WARMUP_STEPS,
    PHYSICS_DT_S,
    Dt1RunnerError,
    SharedSmokeResourceMonitor,
    _camera_rgb,
    _default_runtime_snapshot,
    _fallen_from_pose,
    _live_clock,
    _same_pose,
    _write_rgb,
    build_live_low_level_runtime,
    target_contact_body_norms,
    validate_live_target_contact_sensors,
)
from .runtime import PreResetEvidenceCapture, SubstepContactLatch, camera_audit_fields
from .scene import GO2_CONTACT_BODY_NAMES, contact_precheck_scene
from .scoring import AutonomousParkingScorer, ParkingThresholds, ScoreFrame, ScoreStatus


FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
NEGATIVE_STEPS = 50
MAX_POSITIVE_STEPS = 300
CONTACT_PRECHECK_FORWARD_MPS = 0.35


class ContactPrecheckError(Dt1RunnerError):
    """A contact diagnostic cannot produce trustworthy physical evidence."""


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != length:
        raise ContactPrecheckError(f"{label} must contain {length} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ContactPrecheckError(f"{label} contains non-finite telemetry")
        result.append(float(item))
    return result


def _target_poses_w(base_env: Any) -> dict[str, list[float]]:
    poses: dict[str, list[float]] = {}
    for color in ("red", "blue"):
        try:
            poses[color] = _finite_vector(base_env.scene[f"{color}_target"].data.root_state_w[0, :7], 7, f"{color} target pose")
        except (KeyError, AttributeError, TypeError) as exc:
            raise ContactPrecheckError(f"{color} target physical pose is unavailable") from exc
    return poses


def _done_at_zero(value: Any) -> bool:
    """Require one boolean/integer termination value for this one-env probe."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) != 1:
            raise ContactPrecheckError("contact precheck received a non-singleton done signal")
        value = value[0]
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ContactPrecheckError("contact precheck done signal must be one bool or 0/1 integer")


def _foot_ground_force_norms(base_env: Any) -> dict[str, float]:
    """Read the existing Go2 multi-body sensor only for the four named feet."""
    try:
        sensor = base_env.scene["contact_forces"]
        indices, names = sensor.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)
        net_forces = sensor.data.net_forces_w
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ContactPrecheckError("existing Go2 foot contact sensor is unavailable") from exc
    if hasattr(indices, "detach"):
        indices = indices.detach().cpu()
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if not isinstance(indices, list) or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
        raise ContactPrecheckError("foot contact sensor returned invalid body indices")
    if tuple(names) != FOOT_BODY_NAMES or len(indices) != len(FOOT_BODY_NAMES):
        raise ContactPrecheckError("foot contact sensor body-name lookup does not exactly match the four Go2 feet")
    if getattr(net_forces, "ndim", None) != 3 or net_forces.shape[0] != 1 or net_forces.shape[-1] != 3:
        raise ContactPrecheckError(f"foot contact force tensor has unexpected shape {tuple(getattr(net_forces, 'shape', ())) }")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Isaac host dependency
        raise ContactPrecheckError("torch is required for foot contact inspection") from exc
    if not bool(torch.all(torch.isfinite(net_forces))):
        raise ContactPrecheckError("foot contact sensor supplied non-finite telemetry")
    norms = torch.linalg.vector_norm(net_forces[0, indices], dim=-1).detach().cpu().tolist()
    result: dict[str, float] = {}
    for name, raw in zip(FOOT_BODY_NAMES, norms):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) < 0.0:
            raise ContactPrecheckError("foot contact norm is invalid")
        result[name] = float(raw)
    if len(result) != len(FOOT_BODY_NAMES):
        raise ContactPrecheckError("foot contact norm count is wrong")
    return result


class _ContactEvidence:
    """Small append-only writer for the parent-owned independent checker schema."""

    def __init__(self, root: Path, manifest: dict[str, object]) -> None:
        if root.exists():
            raise FileExistsError(f"refusing to overwrite contact precheck evidence: {root}")
        self.root = root
        self.events_path = root / "contact_precheck_events.jsonl"
        self.root.mkdir(parents=True)
        (root / "contact_precheck_manifest.json").write_text(
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
        (self.root / "contact_precheck_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _capture_standard_rgb(base_env: Any, root: Path, *, label: str) -> dict[str, object]:
    """Write one ordinary front-camera observation without stepping physics."""
    try:
        camera = base_env.scene["rgbd_camera"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ContactPrecheckError("standard RGB camera is unavailable for contact precheck") from exc
    before_clock = _live_clock(base_env)
    before_pose = _default_runtime_snapshot(base_env)["robot_pose_w"]
    base_env.sim.render()
    after_clock = _live_clock(base_env)
    after_pose = _default_runtime_snapshot(base_env)["robot_pose_w"]
    if after_clock != before_clock or not _same_pose(before_pose, after_pose):
        raise ContactPrecheckError("explicit contact-precheck render advanced physical time or robot pose")
    relative = f"rgb_{label}.png"
    _write_rgb(root / relative, _camera_rgb(camera))
    camera_fields = camera_audit_fields(camera)
    if camera_fields["camera_sensor_frame"] is None or camera_fields["camera_timestamp_s"] is None:
        raise ContactPrecheckError("standard RGB camera audit bookkeeping is unavailable")
    return {
        "label": label,
        "path": relative,
        "physics_step": before_clock.sim_step,
        "sim_time_s": before_clock.sim_time_s,
        "robot_pose_w": before_pose,
        **camera_fields,
        "render_did_not_advance_physics": True,
    }


def _contact_snapshot(base_env: Any, *, case: str) -> dict[str, object]:
    return {"case": case, **_default_runtime_snapshot(base_env), "target_poses_w": _target_poses_w(base_env)}


def run_live_contact_precheck(
    args: Any,
    simulation_app: Any,
    *,
    resource_monitor: SharedSmokeResourceMonitor | None = None,
) -> dict[str, object]:
    """Run one color's foot-negative/box-positive physics diagnostic."""
    del simulation_app  # The caller owns AppLauncher lifecycle; no Kit calls occur here.
    if args.target_color not in {"red", "blue"}:
        raise ContactPrecheckError("contact precheck needs a red or blue selected target")
    if not isinstance(args.max_env_steps, int) or not 1 <= args.max_env_steps <= MAX_POSITIVE_STEPS:
        raise ContactPrecheckError(f"contact precheck positive budget must be 1..{MAX_POSITIVE_STEPS}")
    if (isinstance(args.contact_threshold_n, bool)
            or not isinstance(args.contact_threshold_n, (int, float))
            or not math.isclose(float(args.contact_threshold_n), 1.0, rel_tol=0.0, abs_tol=1e-12)):
        raise ContactPrecheckError("contact precheck uses the independently audited 1.0 N contact threshold exactly")
    selected = args.target_color
    other = "blue" if selected == "red" else "red"
    scene = contact_precheck_scene(selected)
    runtime = build_live_low_level_runtime(args, scene)
    adapter = runtime.adapter
    base_env = adapter.base_env
    root = Path(args.output_dir).resolve() / args.run_id / "contact_precheck" / selected
    evidence = _ContactEvidence(root, {
        "format": "go2-dual-target-dt1-contact-precheck-v1",
        "run_id": args.run_id,
        "case_color": selected,
        "physics_dt_s": PHYSICS_DT_S,
        "decimation": DECIMATION,
        "contact_threshold_n": args.contact_threshold_n,
        "negative_steps": NEGATIVE_STEPS,
        "positive_max_steps": args.max_env_steps,
        "robot_contact_body_names": list(GO2_CONTACT_BODY_NAMES),
        "foot_body_names": list(FOOT_BODY_NAMES),
        "checkpoint_path": str(runtime.checkpoint),
        "checkpoint_sha256": runtime.checkpoint_hash,
        "agent_policy_history_length": runtime.policy_history_length,
        "scene": scene.audit_metadata(),
        "navigation_success_approved": False,
        "dt1_approved": False,
    })
    latch = SubstepContactLatch(threshold_n=args.contact_threshold_n)
    state: dict[str, object] = {"case": "warmup", "record": False, "substeps": []}

    def capture_substep() -> None:
        by_target = {
            color: target_contact_body_norms(base_env.scene[f"{color}_target_contacts"], color=color)
            for color in ("red", "blue")
        }
        maxima = {color: max(norms.values()) for color, norms in by_target.items()}
        hit = latch.capture(maxima)
        if state["record"] is True:
            clock = adapter.live_clock()
            state["substeps"].append({
                "physics_step": clock.sim_step,
                "sim_time_s": clock.sim_time_s,
                "target_force_per_robot_body_n": by_target,
                "target_force_max_n": maxima,
                "foot_force_n": _foot_ground_force_norms(base_env),
                "collision_latched": hit,
                "pre_reset_seen": False,
            })

    capture = PreResetEvidenceCapture(
        base_env,
        snapshot=lambda env: _contact_snapshot(env, case=str(state["case"])),
        on_substep=capture_substep,
    )
    scorer = AutonomousParkingScorer(ParkingThresholds())
    observation_seq = 0
    result: dict[str, object] = {
        "format": "go2-dual-target-dt1-contact-precheck-result-v1",
        "run_id": args.run_id,
        "case_color": selected,
        "status": "CONTACT_PRECHECK_FAILED",
        "navigation_success_approved": False,
        "dt1_approved": False,
    }
    try:
        if resource_monitor is not None:
            resource_monitor.check()
        adapter.reset()
        contact_contract = validate_live_target_contact_sensors(base_env)
        (root / "target_contact_view_contract.json").write_text(
            json.dumps(contact_contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        initial_targets = _target_poses_w(base_env)
        capture.install()

        def step(case: str, command: Sequence[float], *, record: bool) -> tuple[AppliedAction, list[bool], dict[str, float], dict[str, object]]:
            nonlocal observation_seq
            state["case"] = case
            state["record"] = record
            state["substeps"] = []
            before_clock = adapter.live_clock()
            applied, _, done, _, _, hits, maxima = adapter.step(command, latch)
            if resource_monitor is not None:
                resource_monitor.check()
            after_clock = adapter.live_clock()
            done_seen = _done_at_zero(done)
            substeps = state["substeps"]
            if record and len(substeps) != DECIMATION:
                raise ContactPrecheckError("contact precheck did not capture exactly four physical substeps")
            pre_reset_refs: list[str] = []
            while capture.records:
                pre_reset_refs.append(evidence.pre_reset(capture.records.pop(0)))
            post_snapshot = _contact_snapshot(base_env, case=case)
            observation_seq += 1
            event: dict[str, object] = {
                "case": case,
                "case_color": selected,
                "step_index": observation_seq - 1,
                "sim_time_before_s": before_clock.sim_time_s,
                "sim_time_after_s": after_clock.sim_time_s,
                "physics_step_before": before_clock.sim_step,
                "physics_step_after": after_clock.sim_step,
                "robot_pose_w": post_snapshot["robot_pose_w"],
                "target_poses_w": post_snapshot["target_poses_w"],
                "initial_target_poses_w": initial_targets,
                "applied_action": list(applied.applied),
                "substeps": substeps,
                "pre_reset_snapshot_refs": pre_reset_refs,
                "environment_done": done_seen,
                "scorer_status": "not_evaluated",
            }
            if pre_reset_refs:
                for substep in substeps:
                    substep["pre_reset_seen"] = True
            # A reset/done cannot be allowed to vanish merely because the
            # caller then raises.  Normal evidence is appended only after its
            # scorer outcome is known, but terminal raw evidence is durable.
            if record and (pre_reset_refs or done_seen):
                evidence.event(event)
            if pre_reset_refs:
                raise ContactPrecheckError("environment auto-reset during contact precheck; pre-reset evidence was retained")
            if done_seen:
                raise ContactPrecheckError("environment signaled done during contact precheck")
            return applied, hits, maxima, event

        # Preserve the normal Go2 warmup without placing warmup records in the
        # independently checked 50-step negative trace.
        for _ in range(GO2_WARMUP_STEPS):
            _, hits, _, _ = step("warmup", (0.0, 0.0, 0.0), record=False)
            if any(hits):
                raise ContactPrecheckError("target contact during contact-precheck warmup")

        negative_target_peak = 0.0
        negative_foot_peak = 0.0
        for _ in range(NEGATIVE_STEPS):
            _, hits, maxima, event = step("foot_ground_negative", (0.0, 0.0, 0.0), record=True)
            # Preserve an unexpected target contact as raw negative evidence
            # before failing the diagnostic.
            evidence.event(event)
            if any(hits) or any(value > 1e-5 for value in maxima.values()):
                raise ContactPrecheckError("ordinary foot-ground negative case reported target contact")
            negative_target_peak = max(negative_target_peak, *maxima.values())
            for substep in event["substeps"]:
                negative_foot_peak = max(negative_foot_peak, *substep["foot_force_n"].values())
        if negative_foot_peak < args.contact_threshold_n:
            raise ContactPrecheckError("foot-ground negative case did not observe a physical foot force above threshold")
        # The negative trace shows ordinary feet touching the collision mesh;
        # this single standard RGB is a human-readable companion, not an
        # invented new physics observation.
        negative_rgb = _capture_standard_rgb(base_env, root, label="foot_ground_negative_end")

        first_collision_step: int | None = None
        first_collision_env_step: int | None = None
        positive_rgb: dict[str, object] | None = None
        positive_peak = 0.0
        scorer_status = "not_observed"
        for _ in range(args.max_env_steps):
            applied, hits, maxima, event = step(f"{selected}_collision_positive", (CONTACT_PRECHECK_FORWARD_MPS, 0.0, 0.0), record=True)
            positive_peak = max(positive_peak, maxima[selected])
            post = event["robot_pose_w"]
            velocity = _default_runtime_snapshot(base_env)["body_velocity_body"]
            decision = scorer.observe(ScoreFrame(
                sim_time_s=float(event["sim_time_after_s"]),
                physics_step=int(event["physics_step_after"]),
                observation_seq=observation_seq,
                raw_action=applied.raw,
                applied_action=applied.applied,
                body_vx_mps=float(velocity[0]),
                body_vy_mps=float(velocity[1]),
                body_yaw_rate_radps=float(velocity[2]),
                in_correct_parking_region=True,
                collision=any(hits),
                fallen=_fallen_from_pose(post),
            ))
            event["scorer_status"] = decision.status.value
            event["scorer_detail"] = decision.detail
            scorer_status = decision.status.value
            if any(hits):
                selected_substeps = [
                    int(substep["physics_step"])
                    for substep in event["substeps"]
                    if float(substep["target_force_max_n"][selected]) >= args.contact_threshold_n
                ]
                if selected_substeps:
                    event["first_collision_physics_step"] = selected_substeps[0]
                event["collision_env_physics_step_after"] = int(event["physics_step_after"])
                # Keep the raw event even for wrong-color/malformed contact;
                # no collision failure is allowed to disappear on the error
                # branch below.
                evidence.event(event)
                (root / f"positive_decision_{observation_seq:04d}.json").write_text(
                    json.dumps(event, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                if maxima[selected] < args.contact_threshold_n or maxima[other] >= args.contact_threshold_n:
                    raise ContactPrecheckError("positive collision did not uniquely hit the selected physical target")
                if decision.status is not ScoreStatus.FAILED_COLLISION:
                    raise ContactPrecheckError("collision did not take precedence over all non-success scorer paths")
                if not selected_substeps:
                    raise ContactPrecheckError("latch reported target collision without a selected-target physical substep")
                first_collision_step = selected_substeps[0]
                first_collision_env_step = int(event["physics_step_after"])
                positive_rgb = _capture_standard_rgb(base_env, root, label=f"{selected}_collision_first")
                break
            evidence.event(event)
            (root / f"positive_decision_{observation_seq:04d}.json").write_text(
                json.dumps(event, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            if decision.terminal:
                raise ContactPrecheckError(
                    f"positive probe terminated as {decision.status.value} before a selected-target physical collision"
                )
        if first_collision_step is None:
            raise ContactPrecheckError("selected target was not physically contacted inside the bounded positive probe")
        if first_collision_env_step is None or positive_rgb is None:
            raise ContactPrecheckError("first physical collision lacks required bounded evidence")

        result.update({
            "status": "CONTACT_PRECHECK_PASSED",
            "negative_steps": NEGATIVE_STEPS,
            "negative_foot_peak_n": negative_foot_peak,
            "negative_target_peak_n": negative_target_peak,
            "positive_target_peak_n": positive_peak,
            "first_collision_physics_step": first_collision_step,
            "first_collision_env_physics_step_after": first_collision_env_step,
            "scorer_status": scorer_status,
            "standard_rgb": {
                "foot_ground_negative_end": negative_rgb,
                f"{selected}_collision_first": positive_rgb,
            },
            "evidence_dir": str(root),
            "scope": "physical contact precheck only; no navigation result or DT1 approval",
        })
        evidence.result(result)
        return result
    except BaseException as exc:
        result.update({"status": "CONTACT_PRECHECK_FAILED", "reason": f"{type(exc).__name__}: {exc}", "evidence_dir": str(root)})
        evidence.result(result)
        raise
    finally:
        capture.restore()
        runtime.history_env.close()
