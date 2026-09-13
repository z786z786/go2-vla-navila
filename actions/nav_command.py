"""Validated, duration-aware navigation commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi


CONTROL_PERIOD_SECONDS = 0.02
PI_OVER_SIX = pi / 6.0
VALID_HOLD_STEPS = frozenset({25, 50, 75})
VALID_DURATIONS_SECONDS = frozenset({0.5, 1.0, 1.5})


def duration_to_hold_steps(duration_seconds: float) -> int:
    """Convert one of the contract's durations to exact 50 Hz control steps."""
    if duration_seconds not in VALID_DURATIONS_SECONDS:
        raise ValueError(
            "duration_seconds must be one of "
            f"{sorted(VALID_DURATIONS_SECONDS)}, got {duration_seconds!r}"
        )
    # The allowed values are fixed; rounding protects the interface against
    # binary floating-point representation while retaining the 0.02 s formula.
    return int(round(duration_seconds / CONTROL_PERIOD_SECONDS))


@dataclass(frozen=True)
class NavCommand:
    """A bounded planar navigation intent for the 50 Hz low-level controller."""

    vx: float
    vy: float
    wz: float
    hold_steps: int
    stop: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.vx <= 0.5:
            raise ValueError(f"vx must be in [0.0, 0.5], got {self.vx!r}")
        if self.vy != 0.0:
            raise ValueError(f"vy must be exactly 0.0, got {self.vy!r}")
        if not -PI_OVER_SIX <= self.wz <= PI_OVER_SIX:
            raise ValueError(
                f"wz must be in [-pi/6, +pi/6], got {self.wz!r}"
            )
        if not isinstance(self.hold_steps, int) or isinstance(self.hold_steps, bool):
            raise TypeError(
                f"hold_steps must be an int, got {type(self.hold_steps).__name__}"
            )
        if not isinstance(self.stop, bool):
            raise TypeError(f"stop must be bool, got {type(self.stop).__name__}")
        if self.stop:
            if (self.vx, self.vy, self.wz, self.hold_steps) != (0.0, 0.0, 0.0, 0):
                raise ValueError("a stop command must have zero velocity and hold_steps=0")
        elif self.hold_steps not in VALID_HOLD_STEPS:
            raise ValueError(
                "a non-stop command must have hold_steps in "
                f"{sorted(VALID_HOLD_STEPS)}, got {self.hold_steps!r}"
            )
