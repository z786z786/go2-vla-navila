#!/usr/bin/env python3
"""Offline, auditable action-replay evaluation for the bounded M6.1 SmolVLA.

This module deliberately uses only the saved LeRobot training observations:
front RGB, 30-D state, and instruction.  It neither starts Isaac nor uses the
M7 socket/JPEG protocol, so its result isolates training-set action prediction
from closed-loop distribution shift and rollout integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.smolvla.bounded_actions import CODEC_FILENAME, PHYSICAL_BOUNDS, decode_latent_actions, load_codec_metadata


IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_NAMES = ("vx", "vy", "wz")
DEFAULT_DATASET_ROOT = Path("<external-data-root>")
DEFAULT_CHECKPOINT = Path(
    "<external-data-root>+0800/checkpoints/step_002000"
)
DEFAULT_SEEDS = (20260831, 20260832, 20260833)
FPS = 50
WZ_SIGN_DEADBAND = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "seen_val"), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def derive_inference_seed(base_seed: int, episode_id: str, frame_index: int) -> int:
    """Return an order-independent CUDA noise seed for one saved observation."""
    if not isinstance(base_seed, int) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("base_seed and non-negative frame_index must be integers")
    checksum = zlib.crc32(f"{episode_id}:{frame_index}".encode("utf-8"))
    return (base_seed + checksum) % (2**31)


def _finite_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite [N, 3], got {array.shape}")
    return array


def _scalar(value: float | np.floating[Any] | None) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


def scalar_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("distribution requires one or more finite values")
    quantiles = np.quantile(values, (0.05, 0.5, 0.95))
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p5": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def pearson_correlation(predicted: np.ndarray, expected: np.ndarray) -> float | None:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if len(predicted) != len(expected) or not len(predicted):
        raise ValueError("correlation inputs must have the same non-zero length")
    if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("correlation inputs must be finite")
    if predicted.std() == 0.0 or expected.std() == 0.0:
        return None
    return _scalar(np.corrcoef(predicted, expected)[0, 1])


def coefficient_of_determination(predicted: np.ndarray, expected: np.ndarray) -> float | None:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    total = float(np.square(expected - expected.mean()).sum())
    if total == 0.0:
        return None
    return float(1.0 - np.square(predicted - expected).sum() / total)


def saturation_ratios(actions: np.ndarray) -> dict[str, float]:
    actions = _finite_array(actions, "actions")
    vx, vy, wz = actions.T
    hit_vx_low = vx <= 0.01
    hit_vx_high = vx >= 0.49
    hit_wz = np.abs(wz) >= 0.49
    hit_vy = np.abs(vy) > 1e-6
    return {
        "vx_low_le_0_01": float(hit_vx_low.mean()),
        "vx_high_ge_0_49": float(hit_vx_high.mean()),
        "wz_abs_ge_0_49": float(hit_wz.mean()),
        "vy_nonzero_gt_1e_6": float(hit_vy.mean()),
        "any": float((hit_vx_low | hit_vx_high | hit_wz | hit_vy).mean()),
    }


def wz_sign_metrics(predicted: np.ndarray, expected: np.ndarray, deadband: float = WZ_SIGN_DEADBAND) -> dict[str, Any]:
    predicted = _finite_array(predicted, "predicted")[:, 2]
    expected = _finite_array(expected, "expected")[:, 2]
    if deadband < 0.0:
        raise ValueError("deadband must be non-negative")
    primary_mask = np.abs(expected) >= deadband
    if primary_mask.any():
        primary_pred_sign = np.where(predicted[primary_mask] >= deadband, 1, np.where(predicted[primary_mask] <= -deadband, -1, 0))
        primary_expected_sign = np.sign(expected[primary_mask]).astype(int)
        primary_accuracy: float | None = float((primary_pred_sign == primary_expected_sign).mean())
    else:
        primary_accuracy = None
    raw_mask = expected != 0.0
    raw_accuracy = None if not raw_mask.any() else float((np.sign(predicted[raw_mask]) == np.sign(expected[raw_mask])).mean())
    return {
        "deadband": deadband,
        "eligible_count": int(primary_mask.sum()),
        "eligible_ratio": float(primary_mask.mean()),
        "accuracy": primary_accuracy,
        "raw_nonzero_gt_accuracy": raw_accuracy,
    }


def action_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    predicted = _finite_array(predicted, "predicted")
    expected = _finite_array(expected, "expected")
    if predicted.shape != expected.shape or not len(predicted):
        raise ValueError("predicted and expected must have equal non-empty [N, 3] shapes")
    difference = predicted - expected
    result: dict[str, Any] = {"count": int(len(predicted)), "per_action": {}}
    for index, name in enumerate(ACTION_NAMES):
        result["per_action"][name] = {
            "mse": float(np.square(difference[:, index]).mean()),
            "rmse": float(np.sqrt(np.square(difference[:, index]).mean())),
            "mae": float(np.abs(difference[:, index]).mean()),
            "bias": float(difference[:, index].mean()),
            "correlation_pearson": pearson_correlation(predicted[:, index], expected[:, index]),
            "r2_vs_target_mean": coefficient_of_determination(predicted[:, index], expected[:, index]),
            "model": scalar_distribution(predicted[:, index]),
            "expert": scalar_distribution(expected[:, index]),
        }
    action_l2_predicted = np.linalg.norm(predicted, axis=1)
    action_l2_expected = np.linalg.norm(expected, axis=1)
    translational_predicted = np.linalg.norm(predicted[:, :2], axis=1)
    translational_expected = np.linalg.norm(expected[:, :2], axis=1)
    abs_wz_predicted = np.abs(predicted[:, 2])
    abs_wz_expected = np.abs(expected[:, 2])
    result["derived_magnitudes"] = {
        "action_l2": {
            "model": scalar_distribution(action_l2_predicted),
            "expert": scalar_distribution(action_l2_expected),
            "mae": float(np.abs(action_l2_predicted - action_l2_expected).mean()),
            "correlation_pearson": pearson_correlation(action_l2_predicted, action_l2_expected),
        },
        "translational_speed": {
            "model": scalar_distribution(translational_predicted),
            "expert": scalar_distribution(translational_expected),
            "mae": float(np.abs(translational_predicted - translational_expected).mean()),
            "correlation_pearson": pearson_correlation(translational_predicted, translational_expected),
        },
        "abs_wz": {
            "model": scalar_distribution(abs_wz_predicted),
            "expert": scalar_distribution(abs_wz_expected),
            "mae": float(np.abs(abs_wz_predicted - abs_wz_expected).mean()),
            "correlation_pearson": pearson_correlation(abs_wz_predicted, abs_wz_expected),
        },
    }
    result["wz_sign"] = wz_sign_metrics(predicted, expected)
    result["saturation_ratio"] = {"model": saturation_ratios(predicted), "expert": saturation_ratios(expected)}
    return result


def baseline_metrics(expected: np.ndarray, global_mean: np.ndarray) -> dict[str, Any]:
    expected = _finite_array(expected, "expected")
    global_mean = np.asarray(global_mean, dtype=np.float64)
    if global_mean.shape != (3,):
        raise ValueError("global_mean must have shape [3]")
    zero = np.zeros_like(expected)
    mean = np.broadcast_to(global_mean, expected.shape).copy()
    primary_mask = np.abs(expected[:, 2]) >= WZ_SIGN_DEADBAND
    signs = np.sign(expected[primary_mask, 2]).astype(int)
    majority_sign = 0 if not len(signs) else int(np.unique(signs, return_counts=True)[0][np.argmax(np.unique(signs, return_counts=True)[1])])
    return {
        "zero": action_metrics(zero, expected),
        "global_train_mean": action_metrics(mean, expected),
        "majority_wz_sign": {
            "sign": majority_sign,
            "accuracy": None if not len(signs) else float((signs == majority_sign).mean()),
            "eligible_count": int(len(signs)),
        },
    }


def flatten_valid(predicted: np.ndarray, expected: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if predicted.shape != expected.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("predicted and expected must be matching [frames, horizon, 3] arrays")
    if valid.shape != predicted.shape[:2]:
        raise ValueError("valid mask must have shape [frames, horizon]")
    return predicted[valid], expected[valid]


def alignment_metrics(
    predictions: np.ndarray, expected: np.ndarray, valid: np.ndarray, global_mean: np.ndarray
) -> dict[str, Any]:
    """Compute a seed-wise and ensemble summary for a selected alignment mask."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 4 or predictions.shape[1:] != expected.shape:
        raise ValueError("predictions must have shape [seeds, frames, horizon, 3]")
    per_seed = []
    for prediction in predictions:
        pred_flat, expected_flat = flatten_valid(prediction, expected, valid)
        per_seed.append(action_metrics(pred_flat, expected_flat))
    ensemble_flat, expected_flat = flatten_valid(predictions.mean(axis=0), expected, valid)
    return {
        "valid_pair_count_per_seed": int(valid.sum()),
        "expert": action_metrics(expected_flat, expected_flat),
        "per_seed": per_seed,
        "ensemble_mean": action_metrics(ensemble_flat, expected_flat),
        "baselines": baseline_metrics(expected_flat, global_mean),
    }


def horizon_metrics(predictions: np.ndarray, expected: np.ndarray, valid: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in range(expected.shape[1]):
        horizon_valid = valid[:, horizon]
        ensemble = predictions.mean(axis=0)[:, horizon]
        pred_flat = ensemble[horizon_valid]
        expected_flat = expected[:, horizon][horizon_valid]
        metrics = action_metrics(pred_flat, expected_flat)
        rows.append(
            {
                "offset": horizon,
                "valid_pair_count_per_seed": int(horizon_valid.sum()),
                "ensemble_mean": metrics,
            }
        )
    return rows


def ensure_physical_actions(actions: np.ndarray) -> None:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape != (50, 3) or not np.isfinite(actions).all():
        raise ValueError(f"expected finite physical [50, 3] action block, got {actions.shape}")
    if not bool(np.all((actions[:, 0] >= PHYSICAL_BOUNDS["vx"][0]) & (actions[:, 0] <= PHYSICAL_BOUNDS["vx"][1]))):
        raise ValueError("decoded vx outside physical bounds")
    if not bool(np.all(actions[:, 1] == 0.0)):
        raise ValueError("decoded vy must be exactly zero")
    if not bool(np.all((actions[:, 2] >= PHYSICAL_BOUNDS["wz"][0]) & (actions[:, 2] <= PHYSICAL_BOUNDS["wz"][1]))):
        raise ValueError("decoded wz outside physical bounds")


def load_runtime(checkpoint: Path) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for offline action evaluation")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"expected SmolVLAConfig, got {type(config).__name__}")
    if set(config.input_features) != {IMAGE_KEY, STATE_KEY}:
        raise ValueError(f"unexpected checkpoint inputs: {sorted(config.input_features)}")
    if config.input_features[STATE_KEY].shape != (30,):
        raise ValueError("checkpoint must have a 30-D state input")
    if config.output_features[ACTION_KEY].shape != (3,) or config.chunk_size != 50 or config.n_obs_steps != 1:
        raise ValueError("checkpoint must emit a current-observation 50x3 action chunk")
    codec = load_codec_metadata(checkpoint)
    config.device = "cuda"
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return torch, policy, preprocessor, postprocessor, codec


def load_split_dataset(dataset_root: Path, split: str, config: Any, video_backend: str) -> tuple[Any, dict[str, Any]]:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest_path = dataset_root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_info = manifest["splits"][split]
    base = LeRobotDataset(split_info["repo_id"], root=dataset_root / split, video_backend=video_backend, return_uint8=True)
    delta_timestamps = resolve_delta_timestamps(config, base.meta)
    train = LeRobotDataset(
        split_info["repo_id"],
        root=dataset_root / split,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        return_uint8=True,
    )
    return train, manifest


def episode_spans(dataset: Any, manifest: dict[str, Any], split: str = "train") -> list[dict[str, Any]]:
    episodes = dataset.meta.episodes
    source = manifest["splits"][split]["episodes"]
    if len(source) != len(episodes["dataset_from_index"]):
        raise ValueError("manifest/dataset training episode count differs")
    spans: list[dict[str, Any]] = []
    for index, (start, stop) in enumerate(zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True)):
        row = source[index]
        span = {
            "episode_id": row["source_short_episode_id"],
            "dataset_start": int(start),
            "dataset_stop": int(stop),
            "frame_count": int(stop) - int(start),
            "instruction": row["task"],
            "source_steps_sha256": row["steps_sha256"],
        }
        if span["frame_count"] != int(row["frame_count"]):
            raise ValueError(f"manifest frame count differs for {span['episode_id']}")
        spans.append(span)
    return spans


def _batch_to_policy_input(batch: dict[str, Any], torch: Any) -> dict[str, Any]:
    image = batch[IMAGE_KEY]
    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    return {IMAGE_KEY: image, STATE_KEY: batch[STATE_KEY], "task": batch["task"]}


def infer_all(
    dataset: Any,
    spans: list[dict[str, Any],],
    torch: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    base_seeds: list[int],
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    from torch.utils.data import DataLoader

    total_frames = len(dataset)
    seed_count = len(base_seeds)
    physical = np.empty((seed_count, total_frames, 50, 3), dtype=np.float32)
    latent = np.empty_like(physical)
    expected = np.empty((total_frames, 50, 3), dtype=np.float32)
    valid = np.empty((total_frames, 50), dtype=bool)
    inference_ms = np.empty((seed_count, total_frames), dtype=np.float64)
    episode_ids: list[str] = []
    local_frame_indices = np.empty(total_frames, dtype=np.int32)
    span_index = 0
    with torch.no_grad():
        for global_index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
            while global_index >= spans[span_index]["dataset_stop"]:
                span_index += 1
            span = spans[span_index]
            episode_id = span["episode_id"]
            local_index = global_index - span["dataset_start"]
            target = batch[ACTION_KEY].squeeze(0).to(torch.float32).cpu().numpy()
            pad = batch.get(f"{ACTION_KEY}_is_pad")
            target_valid = np.ones(50, dtype=bool) if pad is None else ~pad.squeeze(0).bool().cpu().numpy()
            if target.shape != (50, 3):
                raise ValueError(f"expected GT action chunk [50,3], got {target.shape}")
            expected[global_index] = target
            valid[global_index] = target_valid
            episode_ids.append(episode_id)
            local_frame_indices[global_index] = local_index
            processed = preprocessor(_batch_to_policy_input(batch, torch))
            for seed_index, base_seed in enumerate(base_seeds):
                seed = derive_inference_seed(base_seed, episode_id, local_index)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                noise = torch.randn(
                    (1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator, device="cuda"
                )
                policy.reset()
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = policy.predict_action_chunk(processed, noise=noise)
                torch.cuda.synchronize()
                inference_ms[seed_index, global_index] = (time.perf_counter() - started) * 1000.0
                latent_block = postprocessor(prediction).squeeze(0).to(torch.float32)
                physical_block = decode_latent_actions(latent_block)
                latent_np = latent_block.cpu().numpy()
                physical_np = physical_block.cpu().numpy()
                ensure_physical_actions(physical_np)
                latent[seed_index, global_index] = latent_np
                physical[seed_index, global_index] = physical_np
            if progress_every > 0 and ((global_index + 1) % progress_every == 0 or global_index + 1 == total_frames):
                print(f"evaluated {global_index + 1}/{total_frames} GT observations", flush=True)
    return physical, latent, expected, valid, inference_ms, episode_ids, local_frame_indices


def _episode_indices(episode_ids: Iterable[str]) -> dict[str, np.ndarray]:
    values = np.asarray(list(episode_ids))
    return {episode_id: np.flatnonzero(values == episode_id) for episode_id in dict.fromkeys(values.tolist())}


def plot_timeseries(output_dir: Path, predictions: np.ndarray, expected: np.ndarray, episode_ids: list[str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts: list[str] = []
    for episode_id, indices in _episode_indices(episode_ids).items():
        seconds = indices / FPS
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
        for axis, action_index, label, unit in ((axes[0], 0, "vx", "m/s"), (axes[1], 2, "wz", "rad/s")):
            axis.plot(seconds, expected[indices, 0, action_index], color="#1f4e79", linewidth=2.0, label=f"Expert {label}")
            for seed_index in range(predictions.shape[0]):
                axis.plot(seconds, predictions[seed_index, indices, 0, action_index], color="#cf6d17", linewidth=0.7, alpha=0.35, linestyle="--", label="Model seed" if seed_index == 0 else None)
            mean = predictions[:, indices, 0, action_index].mean(axis=0)
            std = predictions[:, indices, 0, action_index].std(axis=0)
            axis.fill_between(seconds, mean - std, mean + std, color="#cf6d17", alpha=0.15, label="Model seed std")
            axis.plot(seconds, mean, color="#cf6d17", linewidth=1.8, linestyle="--", label=f"Model {label} mean")
            axis.set_ylabel(f"{label} ({unit})")
            axis.grid(alpha=0.25)
            axis.legend(loc="best", ncol=2)
        axes[1].set_xlabel("GT episode time (s)")
        path = output_dir / f"{episode_id}_vx_wz_timeseries.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        artifacts.append(str(path))
    return artifacts


def plot_scatter_and_distributions(output_dir: Path, predictions: np.ndarray, expected: np.ndarray) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts: list[str] = []
    mean = predictions.mean(axis=0)[:, 0]
    target = expected[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, index, name in zip(axes, (0, 2), ("vx", "wz"), strict=True):
        axis.scatter(target[:, index], mean[:, index], s=9, alpha=0.45)
        lo = min(float(target[:, index].min()), float(mean[:, index].min()))
        hi = max(float(target[:, index].max()), float(mean[:, index].max()))
        axis.plot((lo, hi), (lo, hi), color="black", linestyle=":", label="y=x")
        axis.set_xlabel(f"Expert {name}")
        axis.set_ylabel(f"Model {name}")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    scatter = output_dir / "first_step_scatter.png"
    fig.savefig(scatter, dpi=170)
    plt.close(fig)
    artifacts.append(str(scatter))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, index, name in zip(axes, range(3), ACTION_NAMES, strict=True):
        axis.hist(target[:, index], bins=40, density=True, histtype="step", linewidth=1.7, label="Expert")
        axis.hist(mean[:, index], bins=40, density=True, histtype="step", linewidth=1.7, linestyle="--", label="Model mean")
        axis.set_title(name)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    distribution = output_dir / "first_step_distributions.png"
    fig.savefig(distribution, dpi=170)
    plt.close(fig)
    artifacts.append(str(distribution))
    return artifacts


def plot_horizon(output_dir: Path, horizons: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offsets = [row["offset"] for row in horizons]
    vx_mse = [row["ensemble_mean"]["per_action"]["vx"]["mse"] for row in horizons]
    wz_mse = [row["ensemble_mean"]["per_action"]["wz"]["mse"] for row in horizons]
    vx_corr = [row["ensemble_mean"]["per_action"]["vx"]["correlation_pearson"] for row in horizons]
    wz_corr = [row["ensemble_mean"]["per_action"]["wz"]["correlation_pearson"] for row in horizons]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(offsets, vx_mse, label="vx MSE")
    axes[0].plot(offsets, wz_mse, label="wz MSE")
    axes[0].set_xlabel("Action chunk offset")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(offsets, [np.nan if value is None else value for value in vx_corr], label="vx Pearson r")
    axes[1].plot(offsets, [np.nan if value is None else value for value in wz_corr], label="wz Pearson r")
    axes[1].set_xlabel("Action chunk offset")
    axes[1].set_ylabel("Correlation")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    path = output_dir / "full_chunk_horizon_metrics.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def write_first_step_csv(
    path: Path,
    predictions: np.ndarray,
    expected: np.ndarray,
    episode_ids: list[str],
    local_frame_indices: np.ndarray,
    inference_ms: np.ndarray,
) -> None:
    rows = [
        "episode_id,frame_index,time_s,vx_gt,vy_gt,wz_gt,"
        "vx_seed0,vy_seed0,wz_seed0,vx_seed1,vy_seed1,wz_seed1,vx_seed2,vy_seed2,wz_seed2,"
        "vx_model_mean,vy_model_mean,wz_model_mean,inference_ms_seed0,inference_ms_seed1,inference_ms_seed2"
    ]
    model = predictions[:, :, 0]
    if model.shape[0] != 3:
        raise ValueError("CSV contract requires exactly three inference seeds")
    for index, episode_id in enumerate(episode_ids):
        values = [
            episode_id,
            str(int(local_frame_indices[index])),
            f"{local_frame_indices[index] / FPS:.6f}",
            *[f"{value:.9g}" for value in expected[index, 0]],
            *[f"{value:.9g}" for value in model[0, index]],
            *[f"{value:.9g}" for value in model[1, index]],
            *[f"{value:.9g}" for value in model[2, index]],
            *[f"{value:.9g}" for value in model[:, index].mean(axis=0)],
            *[f"{value:.6f}" for value in inference_ms[:, index]],
        ]
        rows.append(",".join(values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def render_report(metrics: dict[str, Any], manifest: dict[str, Any]) -> str:
    first = metrics["first_step"]["ensemble_mean"]
    full = metrics["full_chunk"]["ensemble_mean"]
    lines = [
        "# Offline SmolVLA Action Replay",
        "",
        f"Checkpoint: `{manifest['checkpoint']['sha256']}`",
        f"Split `{manifest['split']}` observations: {manifest['counts']['frames']} across {manifest['counts']['episodes']} episodes; seeds: {manifest['base_seeds']}",
        "",
        "## Ensemble-mean action metrics",
        "",
        "| Alignment | Action | MSE | MAE | Pearson r | R² vs target mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for alignment, result in (("first step", first), ("full chunk", full)):
        for name in ACTION_NAMES:
            row = result["per_action"][name]
            correlation = "N/A" if row["correlation_pearson"] is None else f"{row['correlation_pearson']:.6f}"
            r2 = "N/A" if row["r2_vs_target_mean"] is None else f"{row['r2_vs_target_mean']:.6f}"
            lines.append(f"| {alignment} | {name} | {row['mse']:.8f} | {row['mae']:.8f} | {correlation} | {r2} |")
    sign = first["wz_sign"]
    baseline = first["baselines"] if "baselines" in first else metrics["first_step"]["baselines"]
    lines.extend(
        [
            "",
            "## Direction and saturation",
            "",
            f"- First-step `wz` sign accuracy (`|wz_gt| >= {sign['deadband']}`): {sign['accuracy']:.6f}; eligible ratio {sign['eligible_ratio']:.6f}.",
            f"- Majority-sign baseline: {metrics['first_step']['baselines']['majority_wz_sign']['accuracy']:.6f}.",
            f"- First-step model saturation: {json.dumps(first['saturation_ratio']['model'], sort_keys=True)}.",
            "",
            "## Interpretation rule",
            "",
            "Time-aligned predictive signal requires the model to beat the split-mean MSE baseline (R² > 0) for both `vx` and `wz`, and to beat the majority-sign baseline for `wz`. Distribution matching alone is not treated as trajectory prediction.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if len(args.base_seeds) != 3 or len(set(args.base_seeds)) != 3:
        raise ValueError("exactly three distinct --base-seeds are required")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file() or not (checkpoint / CODEC_FILENAME).is_file():
        raise FileNotFoundError("checkpoint must contain model.safetensors and the M6.1 bounded-action codec")
    output_dir.mkdir(parents=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch, policy, preprocessor, postprocessor, codec = load_runtime(checkpoint)
    dataset, manifest_source = load_split_dataset(dataset_root, args.split, policy.config, args.video_backend)
    spans = episode_spans(dataset, manifest_source, args.split)
    started = time.perf_counter()
    physical, latent, expected, valid, inference_ms, episode_ids, local_frame_indices = infer_all(
        dataset, spans, torch, policy, preprocessor, postprocessor, list(args.base_seeds), args.progress_every
    )
    elapsed_s = time.perf_counter() - started
    first_valid = valid[:, :1]
    global_mean = expected[:, 0].mean(axis=0)
    first_step = alignment_metrics(physical[:, :, :1], expected[:, :1], first_valid, global_mean)
    full_chunk = alignment_metrics(physical, expected, valid, global_mean)
    horizons = horizon_metrics(physical, expected, valid)
    per_episode: dict[str, Any] = {}
    for episode_id, indices in _episode_indices(episode_ids).items():
        per_episode[episode_id] = {
            "first_step": alignment_metrics(physical[:, indices, :1], expected[indices, :1], valid[indices, :1], global_mean),
            "full_chunk": alignment_metrics(physical[:, indices], expected[indices], valid[indices], global_mean),
        }
    manifest = {
        "format": "go2-short-vln-offline-action-eval-v2",
        "created_at_utc": now_utc(),
        "command": sys.argv,
        "dataset_root": str(dataset_root),
        "split": args.split,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint / "model.safetensors"),
            "codec_sha256": sha256(checkpoint / CODEC_FILENAME),
            "codec": codec,
        },
        "base_seeds": list(args.base_seeds),
        "input_allowlist": [IMAGE_KEY, STATE_KEY, "task"],
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "counts": {
            "episodes": len(spans),
            "frames": len(dataset),
            "first_step_pairs_per_seed": int(first_valid.sum()),
            "full_chunk_pairs_per_seed": int(valid.sum()),
            "seeds": len(args.base_seeds),
        },
        "episodes": spans,
        "wall_seconds": elapsed_s,
    }
    metrics = {"first_step": first_step, "full_chunk": full_chunk, "horizons": horizons, "per_episode": per_episode}
    np.savez_compressed(
        output_dir / "predictions.npz",
        physical_predictions=physical,
        latent_predictions=latent,
        expert_action_chunks=expected,
        action_valid_mask=valid,
        episode_ids=np.asarray(episode_ids),
        local_frame_indices=local_frame_indices,
        inference_ms=inference_ms,
        base_seeds=np.asarray(args.base_seeds, dtype=np.int64),
    )
    write_first_step_csv(output_dir / "first_step_predictions.csv", physical, expected, episode_ids, local_frame_indices, inference_ms)
    figures = plot_timeseries(output_dir, physical, expected, episode_ids)
    figures.extend(plot_scatter_and_distributions(output_dir, physical, expected))
    figures.append(plot_horizon(output_dir, horizons))
    manifest["artifacts"] = {"figures": figures, "predictions": "predictions.npz", "first_step_csv": "first_step_predictions.csv"}
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "report.md").write_text(render_report(metrics, manifest), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "wall_seconds": elapsed_s, "full_chunk_pairs_per_seed": int(valid.sum())}, indent=2))


if __name__ == "__main__":
    main()
