"""Mock Go2 robot for local smoke tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from PIL import Image

from deploy.go2.adapters.base import BaseRobotAdapter, RobotObservation


@dataclass
class MockRobot(BaseRobotAdapter):
    mode: int = 3
    gait_type: int = 1
    image_size: tuple = (224, 224)
    state: Dict[str, float] = field(
        default_factory=lambda: {
            "vx": 0.0,
            "vy": 0.0,
            "yaw_speed": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "vx_prev": 0.0,
            "vy_prev": 0.0,
            "wz_prev": 0.0,
            "mode": 3,
            "gait_type": 1,
        }
    )
    command_history: List[Dict[str, float]] = field(default_factory=list)

    def _make_image(self):
        array = np.full((self.image_size[1], self.image_size[0], 3), 180, dtype=np.uint8)
        return Image.fromarray(array)

    def read_observation(self) -> RobotObservation:
        self.state["mode"] = self.mode
        self.state["gait_type"] = self.gait_type
        return RobotObservation(state=dict(self.state), image=self._make_image(), timestamp=time.time())

    def send_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        self.state["vx_prev"] = vx
        self.state["vy_prev"] = vy
        self.state["wz_prev"] = wz
        self.state["vx"] = vx
        self.state["vy"] = vy
        self.state["yaw_speed"] = wz
        self.command_history.append({"vx": vx, "vy": vy, "wz": wz, "timestamp": time.time()})
