"""Truth-side-only proportional expert for DT1 reachability and stop checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .layouts import Vec2


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class ExpertCommand:
    vx: float
    vy: float
    wz: float
    phase: str

    def as_list(self) -> list[float]:
        return [self.vx, self.vy, self.wz]


@dataclass(frozen=True)
class ExpertGains:
    max_vx_mps: float = 0.35
    max_wz_radps: float = 0.5
    heading_gain: float = 1.5
    stop_radius_m: float = 0.12
    align_heading_rad: float = 0.18
    terminal_heading_tolerance_rad: float = 0.12


class ParkingExpert:
    """Produces commands from evaluator truth; it never constructs learner input."""

    def __init__(self, gains: ExpertGains = ExpertGains()) -> None:
        self.gains = gains
        self._terminal_hold = False

    def command(
        self,
        *,
        robot_xy: Vec2,
        robot_yaw_rad: float,
        parking_center: Vec2,
        terminal_heading_rad: float | None = None,
    ) -> ExpertCommand:
        """Return an expert-only approach command and an autonomous stop.

        ``terminal_heading_rad`` is the fixed orientation of the box-facing
        parking pose.  Once inside the small centre radius the expert first
        turns in place to that heading; it never begins the scoreable zero
        command interval while the robot is still looking away from the box.
        ``None`` preserves the CPU-only legacy command behaviour used by
        non-terminal unit tests.
        """
        dx, dy = parking_center.x - robot_xy.x, parking_center.y - robot_xy.y
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        heading_error = _wrap_angle(desired_yaw - robot_yaw_rad)
        if distance <= self.gains.stop_radius_m:
            if terminal_heading_rad is not None:
                if not math.isfinite(terminal_heading_rad):
                    raise ValueError("terminal_heading_rad must be finite")
                terminal_error = _wrap_angle(terminal_heading_rad - robot_yaw_rad)
                # The physical low-level policy rebounds slightly when a
                # turning command becomes zero. Use a wider exit boundary so
                # this transient cannot repeatedly interrupt settling. This
                # changes expert commands only, never the parking scorer.
                tolerance = self.gains.terminal_heading_tolerance_rad
                if self._terminal_hold and abs(terminal_error) <= 2.0 * tolerance:
                    return ExpertCommand(0.0, 0.0, 0.0, "hold_stop")
                self._terminal_hold = False
                if abs(terminal_error) > tolerance:
                    wz = max(-self.gains.max_wz_radps, min(self.gains.max_wz_radps, self.gains.heading_gain * terminal_error))
                    return ExpertCommand(0.0, 0.0, wz, "align_terminal_heading")
                self._terminal_hold = True
            return ExpertCommand(0.0, 0.0, 0.0, "hold_stop")
        self._terminal_hold = False
        wz = max(-self.gains.max_wz_radps, min(self.gains.max_wz_radps, self.gains.heading_gain * heading_error))
        # Do not drive forward while the target is behind the robot.  The
        # low-level controller receives only this velocity command; the pose is
        # not serialised in any learner-facing record.
        forward_scale = max(0.0, math.cos(heading_error))
        vx = min(self.gains.max_vx_mps, distance) * forward_scale
        phase = "approach" if abs(heading_error) <= self.gains.align_heading_rad else "turn_then_approach"
        return ExpertCommand(vx, 0.0, wz, phase)
