"""Helpers for reading state from a Go2 adapter."""

from __future__ import annotations

from typing import Dict

from deploy.go2.adapters.base import BaseRobotAdapter, RobotObservation


class Go2StateReader:
    def __init__(self, adapter: BaseRobotAdapter):
        self.adapter = adapter

    def read(self) -> RobotObservation:
        return self.adapter.read_observation()

    @staticmethod
    def extract_core_state(observation: RobotObservation) -> Dict[str, object]:
        return dict(observation.state)
