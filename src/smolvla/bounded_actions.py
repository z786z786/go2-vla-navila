"""Invertible bounded codec between physical Go2 actions and SmolVLA latents."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


CODEC_VERSION = "go2-bounded-action-codec-v1"
CODEC_FILENAME = "bounded_action_codec.json"
DEFAULT_EPSILON = 1e-4
PHYSICAL_BOUNDS = {
    "vx": [0.0, 0.5],
    "vy": [0.0, 0.0],
    "wz": [-0.5, 0.5],
}


def _validate_tensor(actions: torch.Tensor, *, label: str) -> None:
    if not isinstance(actions, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if not actions.is_floating_point() or actions.ndim < 1 or actions.shape[-1] != 3:
        raise ValueError(f"{label} must be a floating tensor with final dimension 3")
    if not bool(torch.all(torch.isfinite(actions))):
        raise ValueError(f"{label} must contain only finite values")


def encode_physical_actions(actions: torch.Tensor, epsilon: float = DEFAULT_EPSILON) -> torch.Tensor:
    """Map finite physical `[vx, 0, wz]` actions to finite, invertible latents."""
    _validate_tensor(actions, label="physical actions")
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be finite and in (0, 1)")
    vx, vy, wz = actions.unbind(dim=-1)
    if not bool(torch.all((vx >= 0.0) & (vx <= 0.5))):
        raise ValueError("physical vx must be in [0, 0.5]")
    if not bool(torch.all(vy == 0.0)):
        raise ValueError("physical vy must be exactly zero")
    if not bool(torch.all((wz >= -0.5) & (wz <= 0.5))):
        raise ValueError("physical wz must be in [-0.5, 0.5]")
    lo = -1.0 + epsilon
    hi = 1.0 - epsilon
    latent_vx = torch.atanh(torch.clamp(4.0 * vx - 1.0, min=lo, max=hi))
    latent_wz = torch.atanh(torch.clamp(2.0 * wz, min=lo, max=hi))
    return torch.stack((latent_vx, torch.zeros_like(latent_vx), latent_wz), dim=-1)


def decode_latent_actions(latent: torch.Tensor) -> torch.Tensor:
    """Decode finite latents to physical actions inside immutable Go2 bounds."""
    _validate_tensor(latent, label="latent actions")
    zx, _zy, zw = latent.unbind(dim=-1)
    vx = torch.clamp(0.25 * (torch.tanh(zx) + 1.0), min=0.0, max=0.5)
    wz = torch.clamp(0.5 * torch.tanh(zw), min=-0.5, max=0.5)
    return torch.stack((vx, torch.zeros_like(vx), wz), dim=-1)


def codec_metadata(*, source_checkpoint_sha256: str, epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    if not isinstance(source_checkpoint_sha256, str) or len(source_checkpoint_sha256) != 64:
        raise ValueError("source_checkpoint_sha256 must be a SHA-256 hex string")
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be finite and in (0, 1)")
    return {
        "codec_version": CODEC_VERSION,
        "epsilon": epsilon,
        "physical_bounds": PHYSICAL_BOUNDS,
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }


def save_codec_metadata(
    checkpoint: Path, *, source_checkpoint_sha256: str, epsilon: float = DEFAULT_EPSILON
) -> Path:
    path = checkpoint / CODEC_FILENAME
    path.write_text(
        json.dumps(codec_metadata(source_checkpoint_sha256=source_checkpoint_sha256, epsilon=epsilon), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def load_codec_metadata(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / CODEC_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"M6.1 checkpoint is missing {CODEC_FILENAME}: {checkpoint}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value != codec_metadata(
        source_checkpoint_sha256=value.get("source_checkpoint_sha256"),
        epsilon=value.get("epsilon"),
    ):
        raise ValueError(f"invalid bounded action codec metadata: {path}")
    return value
