"""CPU-only derivation of DT1 stop thresholds from real zero-command traces.

The draft parking scorer thresholds are deliberately not treated as measured
robot behaviour.  A live motion probe supplies only post-step body telemetry
from explicitly zero-applied command segments; this module validates those
samples and derives one bounded, reviewable replacement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


PHYSICS_DT_S = 0.005
DECIMATION = 4
ENV_STEP_S = PHYSICS_DT_S * DECIMATION
REQUIRED_ZERO_WINDOW_S = 1.0
REQUIRED_ZERO_SAMPLES = 51
ZERO_SETTLE_CASES = (
    "initial_zero",
    "forward_return_zero",
    "left_return_zero",
    "right_return_zero",
)
DRAFT_BODY_PLANAR_SPEED_MPS = 0.03
DRAFT_BODY_YAW_RATE_RADPS = 0.05
CALIBRATION_MARGIN_MULTIPLIER = 1.20
CALIBRATION_MARGIN_ABSOLUTE = 0.002
MAX_CALIBRATED_BODY_PLANAR_SPEED_MPS = 0.09
MAX_CALIBRATED_BODY_YAW_RATE_RADPS = 0.15


class MotionCalibrationError(ValueError):
    """A trace cannot safely define the DT1 stop contract."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MotionCalibrationError(f"{label} must be a finite number")
    return float(value)


def _zero_action(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise MotionCalibrationError(f"{label} must be a three-element action")
    if any(_finite(item, label) != 0.0 for item in value):
        raise MotionCalibrationError(f"{label} must be exactly [0,0,0] in a calibration segment")


def _velocity(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise MotionCalibrationError(f"{label} must be [body_vx,body_vy,body_yaw_rate]")
    return tuple(_finite(item, label) for item in value)  # type: ignore[return-value]


def _round_up_milli(value: float) -> float:
    return math.ceil(value * 1000.0) / 1000.0


@dataclass(frozen=True)
class MotionCalibration:
    """The only body-speed thresholds eligible for the DT1 contract lock."""

    schema_version: str
    physics_dt_s: float
    decimation: int
    env_step_s: float
    required_zero_window_s: float
    required_zero_samples: int
    zero_cases: tuple[str, ...]
    observed_planar_speed_max_mps: float
    observed_yaw_rate_max_radps: float
    derived_body_planar_speed_mps: float
    derived_body_yaw_rate_radps: float
    margin_multiplier: float
    margin_absolute: float
    status: str = "MOTION_CALIBRATED_NOT_TASK_CONTRACT_LOCKED"

    def validate(self) -> None:
        if self.schema_version != "go2-dual-target-dt1-motion-calibration-v1":
            raise MotionCalibrationError("unknown motion calibration schema")
        if self.status != "MOTION_CALIBRATED_NOT_TASK_CONTRACT_LOCKED":
            raise MotionCalibrationError("motion calibration must not claim a task-contract or DT1 approval")
        if (not math.isclose(self.physics_dt_s, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1e-12)
                or self.decimation != DECIMATION
                or not math.isclose(self.env_step_s, ENV_STEP_S, rel_tol=0.0, abs_tol=1e-12)):
            raise MotionCalibrationError("motion calibration timing does not match frozen DT1 physics")
        if (not math.isclose(self.required_zero_window_s, REQUIRED_ZERO_WINDOW_S, rel_tol=0.0, abs_tol=1e-12)
                or self.required_zero_samples != REQUIRED_ZERO_SAMPLES
                or self.zero_cases != ZERO_SETTLE_CASES):
            raise MotionCalibrationError("motion calibration lacks every required real zero-command window")
        if (not math.isclose(self.margin_multiplier, CALIBRATION_MARGIN_MULTIPLIER, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(self.margin_absolute, CALIBRATION_MARGIN_ABSOLUTE, rel_tol=0.0, abs_tol=1e-12)):
            raise MotionCalibrationError("motion calibration margin formula changed")
        for name, value in (
            ("observed_planar_speed_max_mps", self.observed_planar_speed_max_mps),
            ("observed_yaw_rate_max_radps", self.observed_yaw_rate_max_radps),
            ("derived_body_planar_speed_mps", self.derived_body_planar_speed_mps),
            ("derived_body_yaw_rate_radps", self.derived_body_yaw_rate_radps),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise MotionCalibrationError(f"{name} must be finite and non-negative")
        if not (DRAFT_BODY_PLANAR_SPEED_MPS <= self.derived_body_planar_speed_mps <= MAX_CALIBRATED_BODY_PLANAR_SPEED_MPS):
            raise MotionCalibrationError("calibrated planar-speed threshold exceeds the fixed safety envelope")
        if not (DRAFT_BODY_YAW_RATE_RADPS <= self.derived_body_yaw_rate_radps <= MAX_CALIBRATED_BODY_YAW_RATE_RADPS):
            raise MotionCalibrationError("calibrated yaw-rate threshold exceeds the fixed safety envelope")
        if self.derived_body_planar_speed_mps < self.observed_planar_speed_max_mps:
            raise MotionCalibrationError("calibrated planar threshold is below observed zero-command telemetry")
        if self.derived_body_yaw_rate_radps < self.observed_yaw_rate_max_radps:
            raise MotionCalibrationError("calibrated yaw threshold is below observed zero-command telemetry")
        expected_planar = _round_up_milli(max(
            DRAFT_BODY_PLANAR_SPEED_MPS,
            self.observed_planar_speed_max_mps * CALIBRATION_MARGIN_MULTIPLIER + CALIBRATION_MARGIN_ABSOLUTE,
        ))
        expected_yaw = _round_up_milli(max(
            DRAFT_BODY_YAW_RATE_RADPS,
            self.observed_yaw_rate_max_radps * CALIBRATION_MARGIN_MULTIPLIER + CALIBRATION_MARGIN_ABSOLUTE,
        ))
        if (not math.isclose(self.derived_body_planar_speed_mps, expected_planar, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(self.derived_body_yaw_rate_radps, expected_yaw, rel_tol=0.0, abs_tol=1e-12)):
            raise MotionCalibrationError("calibrated thresholds do not match the frozen measured-margin formula")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def derive_motion_calibration(events_by_case: Mapping[str, Sequence[Mapping[str, Any]]]) -> MotionCalibration:
    """Derive bounded thresholds from the final real 1s of each zero segment.

    The caller must retain all samples separately.  Taking the final 51
    observations of four independent zero segments gives each window a full
    one-second physical span at the frozen 50 Hz environment rate; warmup is
    never accepted as calibration input.
    """
    if set(events_by_case) != set(ZERO_SETTLE_CASES):
        raise MotionCalibrationError("calibration requires exactly the four named zero-command segments")
    planar_samples: list[float] = []
    yaw_samples: list[float] = []
    for case in ZERO_SETTLE_CASES:
        events = events_by_case[case]
        if not isinstance(events, Sequence) or len(events) < REQUIRED_ZERO_SAMPLES:
            raise MotionCalibrationError(f"{case} lacks {REQUIRED_ZERO_SAMPLES} real zero-command observations")
        previous_step: int | None = None
        previous_time: float | None = None
        for event in events[-REQUIRED_ZERO_SAMPLES:]:
            _zero_action(event.get("raw_action"), f"{case}.raw_action")
            _zero_action(event.get("applied_action"), f"{case}.applied_action")
            if event.get("warmup") is not False or event.get("environment_done") is not False:
                raise MotionCalibrationError(f"{case} contains warmup or done telemetry")
            hits = event.get("collision_latched_substeps")
            if (not isinstance(hits, list) or len(hits) != DECIMATION
                    or any(type(hit) is not bool for hit in hits) or any(hits)):
                raise MotionCalibrationError(f"{case} contains a target collision")
            if event.get("pre_reset_snapshot_refs") not in ([], ()):
                raise MotionCalibrationError(f"{case} contains pre-reset evidence")
            step = event.get("physics_step_after")
            stamp = event.get("sim_time_after_s")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise MotionCalibrationError(f"{case} has invalid physical step")
            stamp = _finite(stamp, f"{case}.sim_time_after_s")
            if previous_step is not None:
                if step - previous_step != DECIMATION or not math.isclose(stamp - previous_time, ENV_STEP_S, rel_tol=0.0, abs_tol=2e-6):
                    raise MotionCalibrationError(f"{case} zero-command observations are not physically continuous")
            previous_step, previous_time = step, stamp
            vx, vy, yaw = _velocity(event.get("body_velocity_body"), f"{case}.body_velocity_body")
            planar_samples.append(math.hypot(vx, vy))
            yaw_samples.append(abs(yaw))
    observed_planar = max(planar_samples)
    observed_yaw = max(yaw_samples)
    derived_planar = _round_up_milli(max(
        DRAFT_BODY_PLANAR_SPEED_MPS,
        observed_planar * CALIBRATION_MARGIN_MULTIPLIER + CALIBRATION_MARGIN_ABSOLUTE,
    ))
    derived_yaw = _round_up_milli(max(
        DRAFT_BODY_YAW_RATE_RADPS,
        observed_yaw * CALIBRATION_MARGIN_MULTIPLIER + CALIBRATION_MARGIN_ABSOLUTE,
    ))
    result = MotionCalibration(
        schema_version="go2-dual-target-dt1-motion-calibration-v1",
        physics_dt_s=PHYSICS_DT_S,
        decimation=DECIMATION,
        env_step_s=ENV_STEP_S,
        required_zero_window_s=REQUIRED_ZERO_WINDOW_S,
        required_zero_samples=REQUIRED_ZERO_SAMPLES,
        zero_cases=ZERO_SETTLE_CASES,
        observed_planar_speed_max_mps=observed_planar,
        observed_yaw_rate_max_radps=observed_yaw,
        derived_body_planar_speed_mps=derived_planar,
        derived_body_yaw_rate_radps=derived_yaw,
        margin_multiplier=CALIBRATION_MARGIN_MULTIPLIER,
        margin_absolute=CALIBRATION_MARGIN_ABSOLUTE,
    )
    result.validate()
    return result
