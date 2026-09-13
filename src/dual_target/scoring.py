"""Independent autonomous-parking scorer with simulation-time continuity checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .contracts import ACTION_BOUNDS

class ScoreStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    INVALID_SAMPLE = "invalid_sample"
    FAILED_COLLISION = "failed_collision"
    FAILED_FALLEN = "failed_fallen"
    FAILED_WRONG_TARGET_STOP = "failed_wrong_target_stop"
    FAILED_EXTERNAL_STOP = "failed_external_stop"


@dataclass(frozen=True)
class ParkingThresholds:
    parking_radius_m: float = 0.30
    raw_vx_mps: float = 0.03
    raw_wz_radps: float = 0.05
    applied_vx_mps: float = 0.03
    applied_wz_radps: float = 0.05
    body_linear_speed_mps: float = 0.03
    body_yaw_rate_radps: float = 0.05
    required_duration_s: float = 1.0
    max_observation_gap_s: float = 0.05

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class ScoreFrame:
    """Evaluator-side observation; none of these fields is a learner input."""

    sim_time_s: float
    physics_step: int
    observation_seq: int
    raw_action: Sequence[float]
    applied_action: Sequence[float]
    body_vx_mps: float
    body_vy_mps: float
    body_yaw_rate_radps: float
    in_correct_parking_region: bool
    warmup: bool = False
    collision: bool = False
    fallen: bool = False
    wrong_target_stop: bool = False
    evaluator_stop: bool = False

    def validate_event_flags(self) -> None:
        """Check terminal-event types before using their truth values."""
        for name in (
            "in_correct_parking_region", "warmup", "collision", "fallen",
            "wrong_target_stop", "evaluator_stop",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

    def validate(self) -> None:
        numbers = [
            self.sim_time_s, self.body_vx_mps, self.body_vy_mps, self.body_yaw_rate_radps,
            *self.raw_action, *self.applied_action,
        ]
        if len(self.raw_action) != 3 or len(self.applied_action) != 3:
            raise ValueError("raw_action and applied_action must each be [vx, vy, wz]")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in numbers):
            raise ValueError("score frame contains a non-finite numeric value")
        if isinstance(self.physics_step, bool) or isinstance(self.observation_seq, bool):
            raise ValueError("physics_step and observation_seq must be integers")
        if not isinstance(self.physics_step, int) or not isinstance(self.observation_seq, int):
            raise ValueError("physics_step and observation_seq must be integers")
        if self.physics_step < 0 or self.observation_seq < 0 or self.sim_time_s < 0.0:
            raise ValueError("time and sequence fields must be non-negative")
        self.validate_event_flags()
        raw = tuple(float(value) for value in self.raw_action)
        applied = tuple(float(value) for value in self.applied_action)
        expected_applied = tuple(
            max(lower, min(upper, value))
            for value, (lower, upper) in zip(raw, ACTION_BOUNDS)
        )
        if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12) for actual, expected in zip(applied, expected_applied)):
            raise ValueError("applied_action does not match the fixed deployment mapping")
        if applied[1] != 0.0:
            raise ValueError("applied vy must be exactly zero in the fixed deployment mapping")


@dataclass(frozen=True)
class ScoreDecision:
    status: ScoreStatus
    window_duration_s: float
    detail: str

    @property
    def terminal(self) -> bool:
        return self.status in {
            ScoreStatus.SUCCESS,
            ScoreStatus.FAILED_COLLISION,
            ScoreStatus.FAILED_FALLEN,
            ScoreStatus.FAILED_WRONG_TARGET_STOP,
            ScoreStatus.FAILED_EXTERNAL_STOP,
        }


class AutonomousParkingScorer:
    """Score a real, continuous stop without sending a stop command itself."""

    def __init__(self, thresholds: ParkingThresholds = ParkingThresholds()) -> None:
        thresholds.validate()
        self.thresholds = thresholds
        self._window_start: ScoreFrame | None = None
        self._previous: ScoreFrame | None = None
        self._terminal: ScoreDecision | None = None

    def _reset_window(self) -> None:
        self._window_start = None

    def _decision(self, status: ScoreStatus, detail: str) -> ScoreDecision:
        duration = 0.0
        if self._window_start is not None and self._previous is not None:
            duration = max(0.0, self._previous.sim_time_s - self._window_start.sim_time_s)
        return ScoreDecision(status=status, window_duration_s=duration, detail=detail)

    def _continuous_new_observation(self, frame: ScoreFrame) -> tuple[bool, str]:
        if self._previous is None:
            return True, "first_valid_observation"
        previous = self._previous
        if frame.sim_time_s <= previous.sim_time_s:
            return False, "sim_time did not strictly advance"
        if frame.physics_step <= previous.physics_step:
            return False, "physics_step did not strictly advance"
        if frame.observation_seq <= previous.observation_seq:
            return False, "observation_seq did not strictly advance"
        if frame.sim_time_s - previous.sim_time_s > self.thresholds.max_observation_gap_s:
            return False, "observation gap is too large to prove continuous stopping"
        return True, "continuous"

    def _stop_conditions_hold(self, frame: ScoreFrame) -> tuple[bool, str]:
        raw_vx, _, raw_wz = (float(value) for value in frame.raw_action)
        applied_vx, _, applied_wz = (float(value) for value in frame.applied_action)
        body_speed = math.hypot(float(frame.body_vx_mps), float(frame.body_vy_mps))
        if not frame.in_correct_parking_region:
            return False, "outside correct parking region"
        if abs(raw_vx) >= self.thresholds.raw_vx_mps or abs(raw_wz) >= self.thresholds.raw_wz_radps:
            return False, "raw vx/wz is not near zero"
        if abs(applied_vx) >= self.thresholds.applied_vx_mps or abs(applied_wz) >= self.thresholds.applied_wz_radps:
            return False, "applied vx/wz is not near zero"
        if body_speed >= self.thresholds.body_linear_speed_mps:
            return False, "body planar speed is not near zero"
        if abs(float(frame.body_yaw_rate_radps)) >= self.thresholds.body_yaw_rate_radps:
            return False, "body yaw rate is not near zero"
        return True, "stop conditions hold"

    def observe(self, frame: ScoreFrame) -> ScoreDecision:
        """Consume one evaluator frame; terminal errors can never be converted to success."""
        if self._terminal is not None:
            return self._terminal
        # Reject malformed flags before interpreting any event.  In
        # particular, strings such as ``'false'`` cannot use Python truthiness
        # to end or extend an episode.
        try:
            frame.validate_event_flags()
        except (TypeError, ValueError) as exc:
            self._reset_window()
            self._previous = None
            return ScoreDecision(ScoreStatus.INVALID_SAMPLE, 0.0, str(exc))
        # Terminal events take precedence over non-terminal numeric telemetry.
        # A collision must not disappear merely because another field in that
        # same frame became NaN before the runner serialized it.
        terminal_events = (
            (frame.collision, ScoreStatus.FAILED_COLLISION, "collision observed"),
            (frame.fallen, ScoreStatus.FAILED_FALLEN, "fall observed"),
            (frame.wrong_target_stop, ScoreStatus.FAILED_WRONG_TARGET_STOP, "wrong target stopped"),
            (frame.evaluator_stop, ScoreStatus.FAILED_EXTERNAL_STOP, "external evaluator stop observed"),
        )
        for happened, status, detail in terminal_events:
            if happened:
                self._reset_window()
                self._terminal = ScoreDecision(status, 0.0, detail)
                return self._terminal
        try:
            frame.validate()
        except (TypeError, ValueError) as exc:
            self._reset_window()
            self._previous = None
            return ScoreDecision(ScoreStatus.INVALID_SAMPLE, 0.0, str(exc))
        continuous, continuity_detail = self._continuous_new_observation(frame)
        if not continuous:
            self._reset_window()
            self._previous = frame
            return ScoreDecision(ScoreStatus.INVALID_SAMPLE, 0.0, continuity_detail)
        self._previous = frame
        if frame.warmup:
            self._reset_window()
            return ScoreDecision(ScoreStatus.IN_PROGRESS, 0.0, "warmup never contributes to parking window")
        holds, detail = self._stop_conditions_hold(frame)
        if not holds:
            self._reset_window()
            return ScoreDecision(ScoreStatus.IN_PROGRESS, 0.0, detail)
        if self._window_start is None:
            self._window_start = frame
        duration = frame.sim_time_s - self._window_start.sim_time_s
        if duration >= self.thresholds.required_duration_s:
            self._terminal = ScoreDecision(ScoreStatus.SUCCESS, duration, "continuous autonomous stop confirmed")
            return self._terminal
        return ScoreDecision(ScoreStatus.IN_PROGRESS, duration, detail)
