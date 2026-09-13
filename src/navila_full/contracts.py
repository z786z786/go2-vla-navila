"""Fail-closed contracts for the full official-episode NaVILA experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DATASET_SCHEMA_VERSION = "navila_full_episode_go2_v1"
SELECTION_MANIFEST_FORMAT = "navila-full-episode-selection-v1"
CONVERSION_MANIFEST_FORMAT = "navila-full-episode-lerobot-v1"
POLICY_PROTOCOL_VERSION = "go2-smolvla-full-v1"

IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
POLICY_INPUTS = (IMAGE_KEY, STATE_KEY, "task")
STATE_NAMES = ("body_vx", "body_vy", "body_yaw_rate")
ACTION_NAMES = ("vx", "vy", "wz")
AUDIT_ONLY_PATH_MARKERS = ("/outputs/m6", "/outputs/m7", "/data/lerobot/short_vln_v1")
PROHIBITED_POLICY_SIGNALS = frozenset(
    {
        "rpy", "roll", "pitch", "yaw", "projected_gravity", "joint_state",
        "joint_position", "joint_velocity", "path", "waypoint", "goal",
        "goal_direction", "goal_distance", "pd_state", "pd_command",
        "semantic_segmentation", "reference_path", "gt_locations",
    }
)

POLICY_CONTRACT: dict[str, Any] = {
    "n_obs_steps": 1,
    "chunk_size": 50,
    "n_action_steps": 10,
    "dataset_rate_hz": 50,
    "policy_rate_hz": 50,
    "replan_every_frames": 10,
    "prediction_horizon_seconds": 1.0,
    "action": {"shape": [50, 3], "names": list(ACTION_NAMES)},
    "state": {"shape": [3], "names": list(STATE_NAMES)},
    "inputs": list(POLICY_INPUTS),
    "visual": "IDENTITY",
    "image_preparation": "SmolVLA.prepare_images: aspect_ratio_resize_pad_512_and_[0,1]_to_[-1,1]",
    "train_expert_only": True,
    "freeze_vlm": True,
    "freeze_vision_encoder": True,
    "trainable_modules": ["action_expert", "state_projection", "action_projections"],
    "normalizers": {"state": "recompute_from_new_train", "action": "recompute_from_new_train"},
    "pd_usage": "expert_collection_and_oracle_only",
    "latency_p95_seconds_max": 0.6,
}


def required_training_steps(n_train_samples: int, effective_batch_size: int) -> int:
    """Use the M7-N fixed budget, derived only from new full-episode samples."""
    if n_train_samples <= 0:
        raise ValueError("n_train_samples must be positive")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    return min(30_000, math.ceil(20 * n_train_samples / effective_batch_size))


def _path_is_legacy(value: str | Path) -> bool:
    normalized = str(value).replace("\\", "/").rstrip("/")
    return any(marker in normalized for marker in AUDIT_ONLY_PATH_MARKERS)


def validate_training_plan(plan: Mapping[str, Any]) -> list[str]:
    """Reject non-3-D, legacy, or privileged-signal training configuration."""
    errors: list[str] = []
    for key, expected in POLICY_CONTRACT.items():
        if key in {"action", "state", "inputs", "normalizers", "trainable_modules"}:
            continue
        if plan.get(key) != expected:
            errors.append(f"{key} must equal {expected!r}")
    if plan.get("action") != POLICY_CONTRACT["action"]:
        errors.append("action must be exactly 50x3 [vx, vy, wz]")
    if plan.get("state") != POLICY_CONTRACT["state"]:
        errors.append("state must be exactly [body_vx, body_vy, body_yaw_rate]")
    if tuple(plan.get("inputs", ())) != POLICY_INPUTS:
        errors.append("policy inputs must be current RGB, 3-D state, and task only")
    if sorted(plan.get("trainable_modules", ())) != sorted(POLICY_CONTRACT["trainable_modules"]):
        errors.append("only action expert, state projection, and action projections may train")
    if plan.get("normalizers") != POLICY_CONTRACT["normalizers"]:
        errors.append("state/action normalizers must be recomputed from new train data")
    rejected = sorted({str(value).lower() for value in plan.get("policy_signals", ())} & PROHIBITED_POLICY_SIGNALS)
    if rejected:
        errors.append(f"prohibited policy signals: {rejected}")
    if plan.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("dataset must use the full official-episode NaVILA schema")
    if plan.get("source_checkpoint"):
        errors.append("a legacy policy checkpoint may not initialize full-episode training")
    for field in ("dataset_root", "source_checkpoint", "selection_manifest", "acceptance_reference"):
        if plan.get(field) and _path_is_legacy(plan[field]):
            errors.append(f"{field} points at an audit-only M6/M7/D5 artifact")
    try:
        samples = int(plan.get("n_train_samples", 0))
        batch = int(plan.get("effective_batch_size", 0))
    except (TypeError, ValueError):
        samples, batch = 0, 0
    if samples <= 0 or batch <= 0:
        errors.append("n_train_samples and effective_batch_size must be positive")
    elif plan.get("training_steps") != required_training_steps(samples, batch):
        errors.append("training_steps does not follow the fixed full-episode formula")
    return errors
