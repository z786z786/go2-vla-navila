"""Safety guardrails for Go2 high-level velocity commands."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from robotics.config import VelocityControlConfig


@dataclass
class SafetyFilterResult:
    command: torch.Tensor
    mode: str
    reason: str


class VelocitySafetyFilter:
    """Apply clipping, smoothing, and fallback gating to predicted commands."""

    def __init__(self, config: Optional[VelocityControlConfig] = None):
        self.config = config or VelocityControlConfig()
        self.last_command = torch.zeros(3, dtype=torch.float32)
        self.last_valid_command = None
        self.last_image_timestamp = None

    def _zero(self, reason: str, mode: str = "zero-command") -> SafetyFilterResult:
        zero = torch.zeros(3, dtype=torch.float32)
        self.last_command = zero
        return SafetyFilterResult(command=zero, mode=mode, reason=reason)

    def _state_only_allowed(self, state: Dict[str, object], image_missing: bool, image_timestamp: Optional[float]) -> bool:
        if not self.config.allow_state_only_fallback or not image_missing:
            return False
        if image_timestamp is None:
            return False
        if time.time() - image_timestamp > self.config.fallback_image_timeout_s:
            return False
        required = ("yaw_speed", "vx_prev", "vy_prev", "wz_prev", "mode")
        if not (("vx" in state and "vy" in state) or "body_velocity" in state):
            return False
        for key in required:
            if key not in state:
                return False
            value = state[key]
            if value is None or not math.isfinite(float(value)):
                return False
        if not self.config.ensure_mode_allowed(state.get("mode")):
            return False
        if self.last_valid_command is None:
            return False
        return True

    def _clip(self, command: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [
                float(command[0].clamp(*self.config.clip_vx)),
                float(command[1].clamp(*self.config.clip_vy)),
                float(command[2].clamp(*self.config.clip_wz)),
            ],
            dtype=torch.float32,
        )

    def _rate_limit(self, command: torch.Tensor) -> torch.Tensor:
        delta = command - self.last_command
        delta[0] = delta[0].clamp(-self.config.max_delta_vx, self.config.max_delta_vx)
        delta[1] = delta[1].clamp(-self.config.max_delta_vy, self.config.max_delta_vy)
        delta[2] = delta[2].clamp(-self.config.max_delta_wz, self.config.max_delta_wz)
        return self.last_command + delta

    def _ema(self, command: torch.Tensor) -> torch.Tensor:
        return self.config.ema_alpha * command + (1.0 - self.config.ema_alpha) * self.last_command

    def apply(
        self,
        predicted_command: Optional[torch.Tensor],
        state: Dict[str, object],
        image_missing: bool = False,
        image_timestamp: Optional[float] = None,
    ) -> SafetyFilterResult:
        if image_timestamp is not None:
            self.last_image_timestamp = image_timestamp
        if predicted_command is None or predicted_command.numel() != 3:
            return self._zero("invalid-shape")
        if not torch.isfinite(predicted_command).all():
            return self._zero("invalid-value")
        mode = state.get("mode")
        if not self.config.ensure_mode_allowed(mode):
            return self._zero("mode-guard")
        if image_missing and not self._state_only_allowed(state, image_missing, image_timestamp or self.last_image_timestamp):
            return self._zero("image-missing")
        command = self._clip(predicted_command.float())
        command = self._rate_limit(command)
        command = self._ema(command)
        self.last_command = command
        self.last_valid_command = command.clone()
        return SafetyFilterResult(command=command, mode="state-only" if image_missing else "normal", reason="ok")
