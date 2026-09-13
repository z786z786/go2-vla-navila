"""High-level velocity command sender."""

from __future__ import annotations

from deploy.go2.adapters.base import BaseRobotAdapter, TrajectoryStatus, TrajectoryWaypoint


class Go2VelocityCommander:
    def __init__(self, adapter: BaseRobotAdapter):
        self.adapter = adapter

    def send(self, vx: float, vy: float, wz: float) -> None:
        self.adapter.send_velocity_command(vx, vy, wz)

    def follow_trajectory(self, waypoints: list[TrajectoryWaypoint]) -> TrajectoryStatus:
        return self.adapter.follow_trajectory(waypoints)

    def trajectory_status(self) -> TrajectoryStatus:
        return self.adapter.trajectory_status()

    def stop(self) -> TrajectoryStatus:
        return self.adapter.stop()

    def stand_up(self) -> None:
        self.adapter.stand_up()

    def stand_down(self) -> None:
        self.adapter.stand_down()
