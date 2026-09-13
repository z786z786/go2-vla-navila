#!/usr/bin/env python3
"""Pure, auditable helpers for the M6.2 D5 data-expansion gate.

This module deliberately has no Isaac, LeRobot, NumPy, or torch dependency so
that coverage definitions and decision criteria can be tested on the local host
as well as the server.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


ACTION_REGIMES = ("STRAIGHT", "LEFT", "RIGHT", "TRANSITION")
NEAR_GOAL_RADIUS_M = 0.5
DISTANCE_BIN_NAMES = ("short", "medium", "long")


def action_regime(wz: float) -> str:
    """Classify a physical expert yaw command using the declared D5 bounds."""
    value = float(wz)
    if not math.isfinite(value):
        raise ValueError("wz must be finite")
    if abs(value) < 0.05:
        return "STRAIGHT"
    if value > 0.10:
        return "LEFT"
    if value < -0.10:
        return "RIGHT"
    return "TRANSITION"


def _identity(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record requires non-empty {key}")
    return value


def _coverage(values: Iterable[str], total: int) -> dict[str, Any]:
    unique = sorted(set(values))
    return {"count": len(unique), "values": unique, "percentage": 0.0 if total == 0 else 100.0 * len(unique) / total}


def summarize_regimes(records: Sequence[Mapping[str, Any]], *, near_goal_radius_m: float = NEAR_GOAL_RADIUS_M) -> dict[str, Any]:
    """Summarize mutually-exclusive yaw regimes plus overlapping near-goal data."""
    if not math.isfinite(near_goal_radius_m) or near_goal_radius_m <= 0.0:
        raise ValueError("near_goal_radius_m must be positive and finite")
    if not records:
        raise ValueError("regime summary requires one or more records")
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in ACTION_REGIMES}
    near_goal: list[Mapping[str, Any]] = []
    near_goal_by_regime: dict[str, list[Mapping[str, Any]]] = {name: [] for name in ACTION_REGIMES}
    for record in records:
        try:
            regime = action_regime(float(record["expert_wz"]))
            distance = float(record["distance_to_goal_xy_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("record requires finite expert_wz and distance_to_goal_xy_m") from error
        if not math.isfinite(distance):
            raise ValueError("distance_to_goal_xy_m must be finite")
        _identity(record, "episode_id")
        _identity(record, "scene_id")
        grouped[regime].append(record)
        if distance < near_goal_radius_m:
            near_goal.append(record)
            near_goal_by_regime[regime].append(record)

    total = len(records)

    def one(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "frame_count": len(items),
            "percentage": 100.0 * len(items) / total,
            "episode_coverage": _coverage((_identity(item, "episode_id") for item in items), total),
            "scene_coverage": _coverage((_identity(item, "scene_id") for item in items), total),
        }

    result = {
        "frame_count": total,
        "near_goal_radius_m": near_goal_radius_m,
        "wz_regimes": {name: one(grouped[name]) for name in ACTION_REGIMES},
        "near_goal": one(near_goal),
        "near_goal_by_wz_regime": {name: one(near_goal_by_regime[name]) for name in ACTION_REGIMES},
    }
    if sum(row["frame_count"] for row in result["wz_regimes"].values()) != total:
        raise AssertionError("yaw regimes must partition the records")
    return result


def _stats_required(value: Mapping[str, Any], name: str) -> dict[str, float]:
    required = ("mean", "std", "p5", "p50", "p95")
    result: dict[str, float] = {}
    for key in required:
        number = float(value[key])
        if not math.isfinite(number):
            raise ValueError(f"{name}.{key} must be finite")
        result[key] = number
    if result["std"] < 0.0:
        raise ValueError(f"{name}.std must be non-negative")
    return result


def classify_normalizer_shift(
    baseline: Mapping[str, Mapping[str, Any]], expanded: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Classify D5 normalizer drift without treating a zero-variance vy as a ratio."""
    result: dict[str, Any] = {"substantial": False, "criteria": {"mean_or_quantile_std_fraction": 0.5, "std_ratio_range": [0.8, 1.25]}, "per_dimension": {}}
    for name in ("vx", "vy", "wz"):
        old, new = _stats_required(baseline[name], f"baseline.{name}"), _stats_required(expanded[name], f"expanded.{name}")
        if old["std"] == 0.0:
            result["per_dimension"][name] = {
                "status": "degenerate" if new["std"] == 0.0 and new["mean"] == old["mean"] else "changed_from_degenerate",
                "baseline": old,
                "expanded": new,
                "substantial": not (new["std"] == 0.0 and new["mean"] == old["mean"]),
            }
            result["substantial"] |= bool(result["per_dimension"][name]["substantial"])
            continue
        scale = old["std"]
        mean_shift = abs(new["mean"] - old["mean"]) > 0.5 * scale
        quantile_shift = any(abs(new[key] - old[key]) > 0.5 * scale for key in ("p5", "p50", "p95"))
        std_ratio = new["std"] / old["std"]
        std_shift = not 0.8 <= std_ratio <= 1.25
        substantial = mean_shift or quantile_shift or std_shift
        result["per_dimension"][name] = {
            "status": "nondegenerate",
            "baseline": old,
            "expanded": new,
            "std_ratio": std_ratio,
            "mean_shift_substantial": mean_shift,
            "quantile_shift_substantial": quantile_shift,
            "std_shift_substantial": std_shift,
            "substantial": substantial,
        }
        result["substantial"] |= substantial
    return result


def exposure_steps(
    baseline_steps: int, baseline_trainable_samples: int, expanded_trainable_samples: int, *, save_multiple: int = 500
) -> int:
    """Compute equal-exposure steps from audited trainable anchors, rounding up."""
    if min(baseline_steps, baseline_trainable_samples, expanded_trainable_samples, save_multiple) <= 0:
        raise ValueError("exposure inputs must be positive")
    raw = baseline_steps * expanded_trainable_samples / baseline_trainable_samples
    return int(math.ceil(raw / save_multiple) * save_multiple)


def diagnostic_labels() -> dict[str, str]:
    return {
        "short_vln_v1_0000": "TRAIN-ROUTE DIAGNOSTIC",
        "short_vln_v1_0004": "TRAIN-ROUTE DIAGNOSTIC",
        "short_vln_v1_0001": "SEEN-VAL GENERALIZATION",
        "short_vln_v1_0003": "SEEN-VAL GENERALIZATION",
        "short_vln_v1_0006": "UNSEEN-SCENE DIAGNOSTIC",
    }


def distance_bin(path_length_m: float) -> str:
    value = float(path_length_m)
    if not math.isfinite(value) or not 1.5 <= value <= 4.0:
        raise ValueError("path_length must be finite and within short-VLN range [1.5, 4.0]")
    return "short" if value < 2.25 else "medium" if value < 3.0 else "long"


def heading_bin(rotation_wxyz: Sequence[float]) -> int:
    if len(rotation_wxyz) != 4:
        raise ValueError("rotation_wxyz must contain four values")
    w, x, y, z = (float(value) for value in rotation_wxyz)
    if not all(math.isfinite(value) for value in (w, x, y, z)):
        raise ValueError("rotation_wxyz must be finite")
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return int(math.floor(((yaw + math.pi) % (2.0 * math.pi)) / (math.pi / 4.0))) % 8


def episode_category(episode: Mapping[str, Any]) -> str:
    try:
        category = episode["instruction_metadata"]["category"]
    except (KeyError, TypeError) as error:
        raise ValueError("episode lacks instruction_metadata.category") from error
    if category not in {"straight", "left_turn", "right_turn"}:
        raise ValueError(f"unsupported category {category!r}")
    return str(category)


def episode_features(episode: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "short_episode_id": str(episode["short_episode_id"]),
            "split": str(episode["split"]),
            "category": episode_category(episode),
            "scene_id": str(episode["scene_id"]),
            "heading_bin": heading_bin(episode["start_pose"]["rotation_wxyz"]),
            "distance_bin": distance_bin(float(episode["path_length"])),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid short episode {episode.get('short_episode_id')!r}") from error


def selection_hash(episodes: Sequence[Mapping[str, Any]]) -> str:
    values = "\n".join(str(episode["short_episode_id"]) for episode in episodes).encode("utf-8")
    return hashlib.sha256(values).hexdigest()


def select_stratified_episodes(
    episodes: Sequence[Mapping[str, Any]], *, split: str, per_category: int,
    required_ids: Sequence[str], min_scenes: int, min_heading_bins: int,
    min_distance_per_bin: int, fixed_ids: Sequence[str] = (),
    excluded_ids: Sequence[str] = (), seed: int = 20260831,
    attempts: int = 50_000,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced subset using category quotas and hard coverage gates."""
    required = {str(value) for value in required_ids}
    fixed = {str(value) for value in fixed_ids}
    excluded = {str(value) for value in excluded_ids}
    if fixed & excluded:
        raise ValueError(f"fixed and excluded IDs overlap: {sorted(fixed & excluded)}")
    if required & excluded:
        raise ValueError(f"required IDs cannot be excluded: {sorted(required & excluded)}")
    required |= fixed
    candidates = [episode_features(item) for item in episodes if str(item.get("split")) == split]
    categories = ("straight", "left_turn", "right_turn")
    by_id = {item["short_episode_id"]: item for item in candidates}
    if not required <= set(by_id):
        raise ValueError(f"required IDs are not in split {split}: {sorted(required - set(by_id))}")
    pools = {
        category: [
            item for item in candidates
            if item["category"] == category and item["short_episode_id"] not in excluded
        ]
        for category in categories
    }
    required_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ident in sorted(required):
        required_by_category[by_id[ident]["category"]].append(by_id[ident])
    for category in categories:
        if len(required_by_category[category]) > per_category or len(pools[category]) < per_category:
            raise ValueError(f"category {category} cannot meet selection quota")
    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        selected: list[dict[str, Any]] = []
        for category in categories:
            forced = required_by_category[category]
            forced_ids = {item["short_episode_id"] for item in forced}
            choices = [item for item in pools[category] if item["short_episode_id"] not in forced_ids]
            selected.extend(forced + rng.sample(choices, per_category - len(forced)))
        scenes = {item["scene_id"] for item in selected}
        headings = {item["heading_bin"] for item in selected}
        bins = Counter(item["distance_bin"] for item in selected)
        if len(scenes) >= min_scenes and len(headings) >= min_heading_bins and all(bins[name] >= min_distance_per_bin for name in DISTANCE_BIN_NAMES):
            return sorted(selected, key=lambda item: item["short_episode_id"])
    raise RuntimeError("no deterministic selection satisfied the declared D5 coverage constraints")
