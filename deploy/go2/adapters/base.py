"""Abstract adapter interfaces for Go2 high-level control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RobotObservation:
    state: Dict[str, object]
    image: object = None
    timestamp: float = 0.0


@dataclass(frozen=True)
class TrajectoryWaypoint:
    time_from_start: float
    x: float
    y: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass
class TrajectoryStatus:
    trajectory_id: str | None = None
    active: bool = False
    phase: str = "idle"
    progress_index: int = 0
    total_points: int = 0
    elapsed: float = 0.0
    position_error: float | None = None
    yaw_error: float | None = None
    message: str = ""
    target_pose: Dict[str, float] = field(default_factory=dict)
    robot_pose: Dict[str, float] = field(default_factory=dict)


class BaseRobotAdapter:
    def read_observation(self) -> RobotObservation:
        raise NotImplementedError

    def send_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        raise NotImplementedError

    def follow_trajectory(self, waypoints: list[TrajectoryWaypoint]) -> TrajectoryStatus:
        raise NotImplementedError

    def trajectory_status(self) -> TrajectoryStatus:
        return TrajectoryStatus()

    def stop(self) -> TrajectoryStatus:
        raise NotImplementedError

    def stand_up(self) -> None:
        raise NotImplementedError

    def stand_down(self) -> None:
        raise NotImplementedError

    def set_model_debug(self, debug_payload: Dict[str, object]) -> None:
        return None

    def close(self) -> None:
        return None
