"""Exact Go2 low-level observation/history handling for the dual-target adapter.

The upstream history wrapper flattens a copy of its proprioception buffer into
``low_level_obs``.  Updating the buffer alone therefore leaves the policy's
already-flattened tail stale.  This module is intentionally small and makes the
rebuild a mandatory operation after every high-level velocity update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


PROPRIO_DIM = 45
HISTORY_LENGTH = 9
COMMAND_SLICE = slice(6, 9)
PROPRIO_BLOCKS = (
    ("base_ang_vel", 3),
    ("base_rpy", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("last_action", 12),
)


class LowLevelContractError(ValueError):
    """The current runtime shape/order is not the frozen old locomotion contract."""


@dataclass(frozen=True)
class LowLevelLayout:
    proprio_dim: int = PROPRIO_DIM
    history_length: int = HISTORY_LENGTH
    command_start: int = COMMAND_SLICE.start
    command_end: int = COMMAND_SLICE.stop

    def validate(self) -> None:
        if self.proprio_dim != sum(width for _, width in PROPRIO_BLOCKS):
            raise LowLevelContractError("proprio dimension no longer matches the frozen 45-D order")
        if self.history_length != HISTORY_LENGTH:
            raise LowLevelContractError("DT1 requires exactly nine proprioception history frames")
        if (self.command_start, self.command_end) != (6, 9):
            raise LowLevelContractError("velocity command must occupy proprio indices 6:9")

    @property
    def history_flat_dim(self) -> int:
        return self.proprio_dim * self.history_length

    def base_policy_dim(self, flattened_width: int) -> int:
        self.validate()
        base = flattened_width - self.history_flat_dim
        if base < self.command_end:
            raise LowLevelContractError(
                f"flattened low-level observation width={flattened_width} cannot contain base command slice 6:9"
            )
        return base

    def latest_history_command_slice(self, flattened_width: int) -> slice:
        base = self.base_policy_dim(flattened_width)
        start = base + (self.history_length - 1) * self.proprio_dim + self.command_start
        return slice(start, start + (self.command_end - self.command_start))

    def command_write_plan(self, flattened_width: int) -> dict[str, tuple[int, int]]:
        """Expose both required write locations for audits/tests without torch."""
        tail = self.latest_history_command_slice(flattened_width)
        return {
            "base_policy": (self.command_start, self.command_end),
            "latest_flattened_history": (tail.start, tail.stop),
            "latest_history_buffer": (self.command_start, self.command_end),
        }


def validate_velocity_command(command: Sequence[object]) -> tuple[float, float, float]:
    if isinstance(command, (str, bytes)) or len(command) != 3:
        raise LowLevelContractError("velocity command must be [vx, vy, wz]")
    result: list[float] = []
    for index, value in enumerate(command):
        if isinstance(value, bool):
            raise LowLevelContractError(f"velocity command[{index}] must be finite")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise LowLevelContractError(f"velocity command[{index}] must be finite") from exc
        if not math.isfinite(number):
            raise LowLevelContractError(f"velocity command[{index}] must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


def synchronized_command_observation(
    low_level_obs: Any,
    proprio_obs_buf: Any,
    command: Sequence[object],
    layout: LowLevelLayout = LowLevelLayout(),
) -> Any:
    """Update command locations and return a freshly rebuilt flattened tensor.

    ``low_level_obs`` is not mutated in place as the source of truth after the
    write: it is rebuilt from its base-policy portion and the updated nine-frame
    buffer.  This fixes the upstream copy/staleness behaviour explicitly.
    """
    command_values = validate_velocity_command(command)
    layout.validate()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - actual path is Isaac-only
        raise RuntimeError("synchronizing the live low-level observation requires torch") from exc
    if low_level_obs.ndim != 2 or proprio_obs_buf.ndim != 3:
        raise LowLevelContractError("expected [N,W] flattened obs and [N,9,45] proprio history")
    if tuple(proprio_obs_buf.shape[1:]) != (layout.history_length, layout.proprio_dim):
        raise LowLevelContractError(
            f"expected proprio history [N,{layout.history_length},{layout.proprio_dim}], got {tuple(proprio_obs_buf.shape)}"
        )
    if low_level_obs.shape[0] != proprio_obs_buf.shape[0]:
        raise LowLevelContractError("low-level observation and history batch sizes differ")
    base_width = layout.base_policy_dim(int(low_level_obs.shape[1]))
    if low_level_obs.shape[1] != base_width + layout.history_flat_dim:
        raise LowLevelContractError("flattened low-level observation does not end in nine full proprio frames")
    command_tensor = torch.as_tensor(command_values, dtype=low_level_obs.dtype, device=low_level_obs.device)
    low_level_obs[:, layout.command_start : layout.command_end] = command_tensor
    proprio_obs_buf[:, -1, layout.command_start : layout.command_end] = command_tensor
    rebuilt = torch.cat((low_level_obs[:, :base_width], proprio_obs_buf.reshape(low_level_obs.shape[0], -1)), dim=1)
    tail = layout.latest_history_command_slice(int(rebuilt.shape[1]))
    if not bool(torch.allclose(rebuilt[:, tail], command_tensor.expand(rebuilt.shape[0], -1), atol=0.0, rtol=0.0)):
        raise LowLevelContractError("rebuilt flattened history does not contain the current velocity command")
    return rebuilt


def command_consistency_report(flattened_width: int, layout: LowLevelLayout = LowLevelLayout()) -> dict[str, object]:
    """Small audit record that DT1 writes beside the runtime shape observation."""
    layout.validate()
    return {
        "proprio_dim": layout.proprio_dim,
        "history_length": layout.history_length,
        "blocks": [{"name": name, "width": width} for name, width in PROPRIO_BLOCKS],
        "write_plan": layout.command_write_plan(flattened_width),
        "rebuild_required": True,
    }
