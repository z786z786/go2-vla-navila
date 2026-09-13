"""ROS2-style adapter stub for Unitree Go2 high-level control.

This adapter intentionally keeps the interface thin so SDK/topic differences can
be contained here without complicating the first robotics baseline.
"""

from __future__ import annotations

from typing import Callable

from deploy.go2.adapters.base import BaseRobotAdapter, RobotObservation, TrajectoryStatus, TrajectoryWaypoint


class Ros2LikeAdapter(BaseRobotAdapter):
    def __init__(
        self,
        state_reader: Callable[[], RobotObservation],
        velocity_sender: Callable[[float, float, float], None],
        trajectory_sender: Callable[[list[TrajectoryWaypoint]], TrajectoryStatus] | None = None,
        stop_sender: Callable[[], TrajectoryStatus] | None = None,
        stand_up_sender: Callable[[], None] | None = None,
        stand_down_sender: Callable[[], None] | None = None,
        trajectory_status_reader: Callable[[], TrajectoryStatus] | None = None,
    ):
        self._state_reader = state_reader
        self._velocity_sender = velocity_sender
        self._trajectory_sender = trajectory_sender
        self._stop_sender = stop_sender
        self._stand_up_sender = stand_up_sender
        self._stand_down_sender = stand_down_sender
        self._trajectory_status_reader = trajectory_status_reader

    def read_observation(self) -> RobotObservation:
        return self._state_reader()

    def send_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        self._velocity_sender(vx, vy, wz)

    def follow_trajectory(self, waypoints: list[TrajectoryWaypoint]) -> TrajectoryStatus:
        if self._trajectory_sender is None:
            raise NotImplementedError("trajectory sender not configured")
        return self._trajectory_sender(waypoints)

    def trajectory_status(self) -> TrajectoryStatus:
        if self._trajectory_status_reader is None:
            return super().trajectory_status()
        return self._trajectory_status_reader()

    def stop(self) -> TrajectoryStatus:
        if self._stop_sender is None:
            raise NotImplementedError("stop sender not configured")
        return self._stop_sender()

    def stand_up(self) -> None:
        if self._stand_up_sender is None:
            raise NotImplementedError("stand_up sender not configured")
        self._stand_up_sender()

    def stand_down(self) -> None:
        if self._stand_down_sender is None:
            raise NotImplementedError("stand_down sender not configured")
        self._stand_down_sender()
