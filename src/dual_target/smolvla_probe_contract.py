"""CPU-only contract for DT1's non-training Isaac + base-SmolVLA probe.

The resource probe is not a navigation evaluation and cannot execute a model
action.  This explicit artifact lets the Isaac owner write one front RGB and
three body-frame values, then lets an independently started SmolVLA process
prove its *new* 3D interface and identity diagnostics without inheriting an
old dataset normalizer or an old 30-D request schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    ACTION_CHUNK_SIZE,
    ACTION_NAMES,
    ALLOWED_TASKS,
    EXECUTE_ACTION_STEPS,
    IMAGE_KEY,
    STATE_KEY,
    STATE_NAMES,
    TASK_KEY,
)


SMOLVLA_PROBE_SCHEMA = "go2-dual-target-dt1-smolvla-probe-input-v1"
IDENTITY_NORMALIZATION = "fresh_identity_mean0_std1_diagnostic_only"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class SmolVlaProbeContractError(ValueError):
    """A purported 3D, non-training coexistence request is unsafe."""


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SmolVlaProbeContractError(f"{label} must contain exactly {length} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise SmolVlaProbeContractError(f"{label}[{index}] must be finite")
        result.append(float(item))
    return tuple(result)


def fresh_identity_dataset_stats() -> dict[str, dict[str, list[float]]]:
    """Return diagnostic identity statistics for the new 3-state/3-action API."""
    return {
        STATE_KEY: {"mean": [0.0] * len(STATE_NAMES), "std": [1.0] * len(STATE_NAMES)},
        "action": {"mean": [0.0] * len(ACTION_NAMES), "std": [1.0] * len(ACTION_NAMES)},
    }


@dataclass(frozen=True)
class SmolVlaProbeInput:
    """One frozen observation for a resource/interface probe, never control."""

    schema_version: str
    rgb_path: str
    rgb_sha256: str
    state: tuple[float, float, float]
    task: str
    action_dim: int
    action_chunk_size: int
    execute_action_steps: int
    normalization: str
    dataset_stats: dict[str, dict[str, list[float]]]
    inference_only: bool
    execute_model_actions: bool
    training: bool
    navigation_success_approved: bool
    dt1_approved: bool

    def validate(self) -> None:
        if self.schema_version != SMOLVLA_PROBE_SCHEMA:
            raise SmolVlaProbeContractError("unexpected SmolVLA probe schema")
        if not isinstance(self.rgb_path, str) or not self.rgb_path:
            raise SmolVlaProbeContractError("front RGB path is required")
        if not isinstance(self.rgb_sha256, str) or _HEX64.fullmatch(self.rgb_sha256) is None:
            raise SmolVlaProbeContractError("front RGB must have a lowercase SHA-256 digest")
        _finite_vector(self.state, len(STATE_NAMES), "state")
        if self.task not in ALLOWED_TASKS:
            raise SmolVlaProbeContractError("probe task must be one frozen instruction")
        if self.action_dim != len(ACTION_NAMES):
            raise SmolVlaProbeContractError("probe must expose exactly three action dimensions")
        if self.action_chunk_size != ACTION_CHUNK_SIZE or self.execute_action_steps != EXECUTE_ACTION_STEPS:
            raise SmolVlaProbeContractError("probe must preserve the frozen 50-chunk/10-execution interface")
        if self.normalization != IDENTITY_NORMALIZATION or self.dataset_stats != fresh_identity_dataset_stats():
            raise SmolVlaProbeContractError("probe must use fresh 3D identity diagnostics, never a legacy normalizer")
        if (type(self.inference_only) is not bool or not self.inference_only
                or type(self.execute_model_actions) is not bool or self.execute_model_actions
                or type(self.training) is not bool or self.training):
            raise SmolVlaProbeContractError("probe must be inference-only and may not execute model actions or train")
        if (type(self.navigation_success_approved) is not bool or self.navigation_success_approved
                or type(self.dt1_approved) is not bool or self.dt1_approved):
            raise SmolVlaProbeContractError("resource probe cannot approve navigation or DT1")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def probe_input_from_dict(payload: Mapping[str, Any]) -> SmolVlaProbeInput:
    """Parse an exact JSON request before a model process can allocate CUDA."""
    expected = set(SmolVlaProbeInput.__dataclass_fields__)
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise SmolVlaProbeContractError("SmolVLA probe input keys do not match the frozen schema")
    raw = dict(payload)
    raw["state"] = _finite_vector(raw["state"], len(STATE_NAMES), "state")
    item = SmolVlaProbeInput(**raw)
    item.validate()
    return item
