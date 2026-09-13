"""Thin adapter wrapper around the in-process mock robot."""

from __future__ import annotations

from deploy.go2.adapters.base import BaseRobotAdapter, RobotObservation, TrajectoryStatus, TrajectoryWaypoint
from deploy.go2.mock_robot import MockRobot


class MockRobotAdapter(BaseRobotAdapter):
    def __init__(self, robot: MockRobot):
        self.robot = robot

    def read_observation(self) -> RobotObservation:
        return self.robot.read_observation()

    def send_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        self.robot.send_velocity_command(vx, vy, wz)

    def follow_trajectory(self, waypoints: list[TrajectoryWaypoint]) -> TrajectoryStatus:
        return self.robot.follow_trajectory(waypoints)

    def trajectory_status(self) -> TrajectoryStatus:
        return self.robot.trajectory_status()

    def stop(self) -> TrajectoryStatus:
        return self.robot.stop()

    def stand_up(self) -> None:
        self.robot.stand_up()

    def stand_down(self) -> None:
        self.robot.stand_down()
