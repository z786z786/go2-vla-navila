"""Runtime safety hooks for the real DT1 simulator runner.

These helpers do not create Isaac.  Their live installation is intentionally
explicit so contact and pre-reset evidence can be observed and restored on one
environment instance without touching the legacy wrapper implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class SubstepContactLatch:
    """Latch target contact once per physical substep, not just per env step."""

    threshold_n: float = 1.0
    substep_hits: list[bool] = field(default_factory=list)
    max_force_by_target: dict[str, float] = field(default_factory=lambda: {"red": 0.0, "blue": 0.0})

    def __post_init__(self) -> None:
        if isinstance(self.threshold_n, bool) or not isinstance(self.threshold_n, (int, float)) or not math.isfinite(float(self.threshold_n)) or self.threshold_n <= 0.0:
            raise ValueError("contact threshold must be a positive finite number")

    def begin_env_step(self) -> None:
        """Arm exactly one new manager step; stale callbacks are never hidden."""
        if self.substep_hits:
            raise RuntimeError("unconsumed contact substeps would corrupt the next DT1 action record")
        self.max_force_by_target = {"red": 0.0, "blue": 0.0}

    def capture(self, forces: Mapping[str, float]) -> bool:
        if set(forces) != {"red", "blue"}:
            raise ValueError("substep contact force mapping must contain red and blue")
        clean: dict[str, float] = {}
        for color, value in forces.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"contact force {color} must be finite and non-negative")
            clean[color] = float(value)
        for color, value in clean.items():
            self.max_force_by_target[color] = max(self.max_force_by_target[color], value)
        hit = any(value >= self.threshold_n for value in clean.values())
        self.substep_hits.append(hit)
        return hit

    def consume_env_step(self, *, expected_substeps: int = 4) -> tuple[list[bool], dict[str, float]]:
        if len(self.substep_hits) != expected_substeps:
            raise RuntimeError(f"expected {expected_substeps} contact substeps, observed {len(self.substep_hits)}")
        hits, maxima = list(self.substep_hits), dict(self.max_force_by_target)
        self.substep_hits.clear()
        self.max_force_by_target = {"red": 0.0, "blue": 0.0}
        return hits, maxima


class PreResetEvidenceCapture:
    """Capture scene evidence immediately before ManagerBasedRLEnv auto-reset."""

    def __init__(self, base_env: Any, snapshot: Callable[[Any], Mapping[str, Any]], on_substep: Callable[[], None]) -> None:
        self.base_env = base_env
        self.snapshot = snapshot
        self.on_substep = on_substep
        self.records: list[dict[str, Any]] = []
        self._original_reset: Any | None = None
        self._original_scene_update: Any | None = None

    def install(self) -> None:
        if self._original_reset is not None:
            raise RuntimeError("pre-reset evidence capture is already installed")
        self._original_reset = self.base_env._reset_idx
        self._original_scene_update = self.base_env.scene.update

        def wrapped_reset(env_ids: Any) -> Any:
            self.records.append({"env_ids": self._jsonable_ids(env_ids), **dict(self.snapshot(self.base_env))})
            return self._original_reset(env_ids)

        def wrapped_scene_update(*args: Any, **kwargs: Any) -> Any:
            result = self._original_scene_update(*args, **kwargs)
            self.on_substep()
            return result

        self.base_env._reset_idx = wrapped_reset
        self.base_env.scene.update = wrapped_scene_update

    def restore(self) -> None:
        if self._original_reset is not None:
            self.base_env._reset_idx = self._original_reset
            self._original_reset = None
        if self._original_scene_update is not None:
            self.base_env.scene.update = self._original_scene_update
            self._original_scene_update = None

    @staticmethod
    def _jsonable_ids(value: Any) -> list[int]:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [int(item) for item in value]


def camera_audit_fields(camera: Any) -> dict[str, int | float | None]:
    """Record sensor bookkeeping without treating it as a unique-image proof."""
    frame = getattr(camera, "frame", None)
    timestamp = getattr(camera, "_timestamp_last_update", None)
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().tolist()
    if hasattr(timestamp, "detach"):
        timestamp = timestamp.detach().cpu().tolist()
    return {"camera_sensor_frame": int(frame[0]) if isinstance(frame, list) else None,
            "camera_timestamp_s": float(timestamp[0]) if isinstance(timestamp, list) else None}
