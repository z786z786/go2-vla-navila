"""V2 independent parking trace review derived from DT1's approved checker.

Only the expected command mapping changes to rate-limited ZOH. Raw requests,
applied commands, measured speed, contact and continuous stopping remain checked.
"""

from __future__ import annotations

import math


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite number")
    return float(value)


def vector(value, size, label):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label}: expected {size}-element list")
    return [number(v, label) for v in value]


def flag(record, key):
    value = record[key]
    if type(value) is not bool:
        raise ValueError(f"{key}: expected bool")
    return value


def review_trace(pre, post, metadata, *, smoke_only=False):
    """Recompute correctness from recorded world pose, not region/success flags.

    metadata keys are an explicit parent-side review contract; an adapter may
    map implementation metadata to these names without changing the evidence.
    """
    if not pre or len(pre) != len(post):
        raise ValueError("nonempty pre/post action records must have equal length")
    if smoke_only and len(pre) != 50:
        raise ValueError("authorized smoke must contain exactly 50 scored steps")
    center = vector(metadata["correct_parking_center_xy"], 2, "parking center")
    other = vector(metadata["other_parking_center_xy"], 2, "other center")
    threshold = metadata["thresholds"]
    required = (
        "parking_radius_m", "raw_vx_mps", "raw_wz_radps", "applied_vx_mps",
        "applied_wz_radps", "body_linear_speed_mps", "body_yaw_rate_radps",
        "required_duration_s", "max_observation_gap_s",
    )
    limits = {key: number(threshold[key], key) for key in required}
    if any(v <= 0 for v in limits.values()):
        raise ValueError("thresholds must be positive")
    if not math.isclose(limits["required_duration_s"], 1.0, abs_tol=1e-12):
        raise ValueError("DT1 required stop duration must remain 1 second")
    dt = number(metadata["physics_dt_s"], "physics dt")
    contact_limit = number(metadata["contact_threshold_n"], "contact threshold")
    if contact_limit <= 0:
        raise ValueError("contact threshold must be positive")
    decimation = metadata["decimation"]
    if type(decimation) is not int or decimation != 4 or not math.isclose(dt, 0.005, abs_tol=1e-12):
        raise ValueError("trace does not match audited 200 Hz physics / decimation 4")
    stop_start = previous_end = previous_step = previous_obs = None
    previous_camera = previous_render = None
    independent_success_index = None
    max_duration = 0.0
    for index, (before, after) in enumerate(zip(pre, post)):
        label = f"frame {index}"
        start = number(before["sim_time_s"], label)
        end = number(after["sim_time_after_s"], label)
        if not math.isclose(start, number(after["sim_time_before_s"], label), abs_tol=1e-8):
            raise ValueError(f"{label}: action timestamp mismatch")
        if not math.isclose(end - start, dt * decimation, rel_tol=0, abs_tol=2e-6):
            raise ValueError(f"{label}: wrong physical duration")
        step = before["physics_step"]
        end_step = after["physics_step_after"]
        obs = before["observation_seq"]
        end_obs = after["observation_seq_after"]
        integers = (step, end_step, obs, end_obs, before["camera_sensor_frame"], before["render_request_seq"])
        if any(type(v) is not int or v < 0 for v in integers):
            raise ValueError(f"{label}: invalid step/frame identifiers")
        if step != after["physics_step_before"] or end_step - step != decimation or end_obs <= obs:
            raise ValueError(f"{label}: wrong action/physics/observation alignment")
        if previous_end is not None:
            if not math.isclose(start, previous_end, rel_tol=0, abs_tol=2e-6) or step != previous_step or obs != previous_obs:
                raise ValueError(f"{label}: gap, duplication or reset in action stream")
            if end - previous_end > limits["max_observation_gap_s"]:
                raise ValueError(f"{label}: insufficient temporal coverage")
            if before["camera_sensor_frame"] <= previous_camera or before["render_request_seq"] <= previous_render:
                raise ValueError(f"{label}: stale camera sampling/render request")
        previous_end, previous_step, previous_obs = end, end_step, end_obs
        previous_camera, previous_render = before["camera_sensor_frame"], before["render_request_seq"]
        # These are auxiliary timestamps only: actual render timing needs audit.
        number(before["camera_timestamp_s"], label)
        vector(before["body_velocity_body"], 3, label)
        if before["phase"] == "warmup" or flag(after, "warmup"):
            raise ValueError(f"{label}: warmup mixed into scored trace")
        raw = vector(before["raw_action"], 3, label)
        applied = vector(before["applied_action"], 3, label)
        bounds = ((0.0, 0.5), (0.0, 0.0), (-0.5, 0.5))
        # V2-only change: independent rate-limited ZOH mapping, preserving
        # the original raw/applied/body/position/contact/continuous-time tests.
        from .zoh_control import ZohSpec
        spec = ZohSpec()
        previous = [0., 0., 0.] if index == 0 else pre[index-1]["applied_action"]
        if index % 10 == 0:
            clipped = [max(lo, min(hi, v)) for v, (lo, hi) in zip(raw, bounds)]
            expected = [max(old-rate*.2, min(old+rate*.2, value))
                        for old,rate,value in zip(previous,spec.rate_limits,clipped)]
        else:
            if raw != pre[index-1]["raw_action"]:
                raise ValueError("raw command changed inside ZOH interval")
            expected = previous
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-7) for a, b in zip(applied, expected)):
            raise ValueError(f"{label}: unexpected external action remapping")
        if applied[1] != 0:
            raise ValueError(f"{label}: deployed lateral command is nonzero")
        collision = after["collision_latched_substeps"]
        if not isinstance(collision, list) or len(collision) != decimation or any(type(v) is not bool for v in collision):
            raise ValueError(f"{label}: missing per-substep collision evidence")
        events = ("fallen", "terminated", "truncated", "evaluator_stop", "wrong_target_stop")
        if any(collision) or any(flag(after, key) for key in events):
            raise ValueError(f"{label}: terminal failure cannot be accepted as success")
        if after["pre_reset_snapshot_ref"] != "":
            raise ValueError(f"{label}: reset event within successful episode")
        force_map = after["target_contact_force_max"]
        if not isinstance(force_map, dict) or set(force_map) != {"red", "blue"}:
            raise ValueError(f"{label}: missing per-target contact force maxima")
        forces = [number(force_map[color], label) for color in ("red", "blue")]
        if any(force < 0 for force in forces):
            raise ValueError(f"{label}: invalid force norm")
        if any(force >= contact_limit for force in forces):
            raise ValueError(f"{label}: physical contact force contradicts collision-free success")
        pose = vector(after["robot_pose_w"], 7, label)
        if not math.isclose(sum(q * q for q in pose[3:]), 1.0, abs_tol=1e-3):
            raise ValueError(f"{label}: non-unit robot quaternion")
        correct = math.hypot(pose[0] - center[0], pose[1] - center[1]) <= limits["parking_radius_m"]
        wrong = math.hypot(pose[0] - other[0], pose[1] - other[1]) <= limits["parking_radius_m"]
        if correct != flag(after, "in_correct_parking_region") or wrong != flag(after, "in_other_parking_region"):
            raise ValueError(f"{label}: recorded region flag disagrees with world pose")
        velocity = vector(after["body_velocity_body"], 3, label)
        stationary = (
            abs(raw[0]) < limits["raw_vx_mps"] and abs(raw[2]) < limits["raw_wz_radps"]
            and abs(applied[0]) < limits["applied_vx_mps"] and abs(applied[2]) < limits["applied_wz_radps"]
            and math.hypot(*velocity[:2]) < limits["body_linear_speed_mps"]
            and abs(velocity[2]) < limits["body_yaw_rate_radps"]
        )
        if correct and stationary:
            if stop_start is None:
                stop_start = end
            duration = end - stop_start
            max_duration = max(max_duration, duration)
            if duration >= limits["required_duration_s"] - 1e-9 and independent_success_index is None:
                independent_success_index = index
        else:
            stop_start = None
        if after["scorer_status"] == "success" and independent_success_index is None:
            raise ValueError(f"{label}: scorer claims success before independently valid stop")
    if smoke_only:
        return {"smoke_trace_integrity_passed": True, "scored_frames": len(post),
                "physical_duration_s": post[-1]["sim_time_after_s"] - pre[0]["sim_time_s"],
                "navigation_success_approved": False, "dt1_approved": False,
                "scope": "50-step trace integrity only; no task success or milestone approval"}
    if independent_success_index is None:
        raise ValueError("no independently valid continuous 1-second autonomous stop")
    if post[-1]["scorer_status"] != "success":
        raise ValueError("runtime did not terminate with success")
    return {"trace_passed": True, "scored_frames": len(post), "first_success_index": independent_success_index,
            "max_continuous_stop_s": max_duration, "scope": "trace only; not full DT1 approval"}
