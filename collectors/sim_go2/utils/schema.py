from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RAW_SCHEMA_VERSION = "go2_isaac_raw_v1"
PACKED_SCHEMA_VERSION = "go2_isaac_packed_v1"


@dataclass(frozen=True)
class RawStepRecord:
    episode_id: str
    step_id: int
    timestamp: float
    image_path: str
    instruction: str
    task_id: str
    target_class: str
    state: list[float]
    raw_cmd_full: list[float]
    raw_cmd_train: list[float]
    success: bool
    collision: bool
    timeout: bool
    scene_id: str
    seed: int
    split_hint: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PackedSampleRecord:
    image: str
    instruction: str
    state: list[float]
    action_chunk: list[list[float]]
    action_mask: list[int]
    task_id: str
    target_class: str
    target_color: str
    target_shape: str
    episode_id: str
    step_id: int
    control_hz: int
    source: str
    scene_id: str
    split: str
