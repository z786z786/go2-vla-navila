from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from collectors.sim_go2.utils.io import stable_hash
from collectors.sim_go2.utils.state_utils import yaw_to_quat_wxyz


def sample_uniform(rng: random.Random, value_range: Sequence[float]) -> float:
    return rng.uniform(float(value_range[0]), float(value_range[1]))


def sample_weighted_label(rng: random.Random, label_to_weight: Mapping[str, float]) -> str:
    labels = list(label_to_weight.keys())
    weights = [max(0.0, float(label_to_weight[label])) for label in labels]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("At least one sampling weight must be positive.")
    return rng.choices(labels, [value / total for value in weights], k=1)[0]


def sample_turn_bucket_and_angle(rng: random.Random, turning_cfg: Mapping[str, Any]) -> tuple[str, float]:
    label = sample_weighted_label(
        rng,
        {
            "straight": float(turning_cfg.get("straight_ratio", 0.0)),
            "left_small": float(turning_cfg.get("left_small_ratio", turning_cfg.get("left_ratio", 0.0) * 0.5)),
            "left_large": float(turning_cfg.get("left_large_ratio", turning_cfg.get("left_ratio", 0.0) * 0.5)),
            "right_small": float(turning_cfg.get("right_small_ratio", turning_cfg.get("right_ratio", 0.0) * 0.5)),
            "right_large": float(turning_cfg.get("right_large_ratio", turning_cfg.get("right_ratio", 0.0) * 0.5)),
            "reverse_reorient": float(turning_cfg.get("reverse_reorient_ratio", 0.0)),
        },
    )
    angle_deg_ranges = {
        "straight": turning_cfg.get("straight_angle_deg", (-15.0, 15.0)),
        "left_small": turning_cfg.get("left_small_angle_deg", (20.0, 65.0)),
        "left_large": turning_cfg.get("left_large_angle_deg", (70.0, 160.0)),
        "right_small": turning_cfg.get("right_small_angle_deg", (-65.0, -20.0)),
        "right_large": turning_cfg.get("right_large_angle_deg", (-160.0, -70.0)),
        "reverse_reorient": turning_cfg.get("reverse_reorient_angle_deg", (140.0, 180.0)),
    }
    angle_deg = sample_uniform(rng, angle_deg_ranges[label])
    if label == "reverse_reorient" and rng.random() < 0.5:
        angle_deg = -angle_deg
    return label, math.radians(angle_deg)


def sample_point_in_region(rng: random.Random, region_cfg: Mapping[str, Any]) -> tuple[float, float]:
    return sample_uniform(rng, region_cfg.get("x", (-1.0, 1.0))), sample_uniform(rng, region_cfg.get("y", (-1.0, 1.0)))


def point_in_region(x: float, y: float, region_cfg: Mapping[str, Any], margin: float = 0.0) -> bool:
    x_min, x_max = [float(value) for value in region_cfg.get("x", (-math.inf, math.inf))]
    y_min, y_max = [float(value) for value in region_cfg.get("y", (-math.inf, math.inf))]
    return (x_min + margin) <= float(x) <= (x_max - margin) and (y_min + margin) <= float(y) <= (y_max - margin)


def quat_multiply(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def sample_camera_offset(base_offset: Mapping[str, Any], randomization_cfg: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    offset = {
        "pos": list(base_offset.get("pos", (0.0, 0.0, 0.0))),
        "rot": tuple(float(value) for value in base_offset.get("rot", (1.0, 0.0, 0.0, 0.0))),
        "convention": str(base_offset.get("convention", "ros")),
    }
    if not randomization_cfg.get("enabled", False):
        return offset
    pos_jitter = list(randomization_cfg.get("pos_jitter", (0.0, 0.0, 0.0)))
    offset["pos"] = [float(offset["pos"][index]) + rng.uniform(-float(pos_jitter[index]), float(pos_jitter[index])) for index in range(3)]
    yaw_jitter_deg = float(randomization_cfg.get("yaw_jitter_deg", 0.0))
    if yaw_jitter_deg > 0.0:
        jitter = yaw_to_quat_wxyz(math.radians(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)))
        offset["rot"] = quat_multiply(offset["rot"], jitter)
    return offset


def sample_light_intensity(base_intensity: float, randomization_cfg: Mapping[str, Any], rng: random.Random) -> float:
    intensity_range = randomization_cfg.get("intensity_jitter")
    if not intensity_range:
        return float(base_intensity)
    return float(base_intensity) * sample_uniform(rng, intensity_range)


def assign_split_by_layout(layout_key: str, train_ratio: float, val_ratio: float) -> str:
    value = int(stable_hash(layout_key, digits=8), 16) / float(16**8)
    if value < float(train_ratio):
        return "train"
    if value < float(train_ratio) + float(val_ratio):
        return "val"
    return "test"
