#!/usr/bin/env python3
"""Audit the M5 LeRobot conversion and exercise SmolVLA input preprocessing."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import DEFAULT_FEATURES
from lerobot.utils.feature_utils import dataset_to_policy_features


IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
SPLITS = ("train", "seen_val", "unseen_test")
EXPECTED_USER_FEATURES = {IMAGE_KEY, STATE_KEY, ACTION_KEY}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def tensor_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    result: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.numel() and (value.is_floating_point() or value.dtype == torch.uint8):
        result.update({"min": float(value.min()), "max": float(value.max())})
    return result


def finite_tensors(mapping: dict[str, Any]) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in mapping.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )


def load_split(root: Path, split_info: dict[str, Any], video_backend: str, **kwargs: Any) -> LeRobotDataset:
    return LeRobotDataset(
        split_info["repo_id"],
        root=root,
        video_backend=video_backend,
        return_uint8=True,
        **kwargs,
    )


def save_random_samples(
    datasets: dict[str, LeRobotDataset], report_dir: Path, seed: int, count: int = 10
) -> tuple[Path, list[dict[str, Any]]]:
    universe = [(split, index) for split, dataset in datasets.items() for index in range(len(dataset))]
    rng = random.Random(seed)
    selected = rng.sample(universe, min(count, len(universe)))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes_flat = axes.ravel()
    records: list[dict[str, Any]] = []
    for axis, (split, index) in zip(axes_flat, selected, strict=False):
        sample = datasets[split][index]
        image = sample[IMAGE_KEY].detach().cpu()
        if image.ndim == 4:
            image = image[-1]
        image_np = image.permute(1, 2, 0).numpy()
        action = sample[ACTION_KEY].detach().cpu().reshape(-1).tolist()
        episode_index = int(sample["episode_index"])
        frame_index = int(sample["frame_index"])
        axis.imshow(image_np)
        axis.set_title(
            f"{split} e{episode_index} f{frame_index}\n"
            f"a=[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]",
            fontsize=9,
        )
        axis.axis("off")
        records.append(
            {
                "split": split,
                "dataset_index": index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": sample["task"],
                "action": action,
            }
        )
    for axis in axes_flat[len(selected) :]:
        axis.axis("off")
    fig.tight_layout()
    path = report_dir / "random_samples_10.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, records


def save_histograms(
    actions: np.ndarray, lengths: list[int], report_dir: Path
) -> tuple[Path, Path]:
    action_path = report_dir / "action_histogram.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for dim, (axis, name) in enumerate(zip(axes, ("vx", "vy", "wz"), strict=True)):
        axis.hist(actions[:, dim], bins=30, color="#3976af", edgecolor="white")
        axis.set_title(name)
        axis.set_xlabel("command")
        axis.set_ylabel("frames")
        axis.grid(alpha=0.2)
    fig.suptitle("Go2 expert action distribution")
    fig.tight_layout()
    fig.savefig(action_path, dpi=150)
    plt.close(fig)

    length_path = report_dir / "episode_length_histogram.png"
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(lengths, bins=min(12, max(3, len(set(lengths)))), color="#d4772c", edgecolor="white")
    axis.set_title("Episode lengths")
    axis.set_xlabel("frames")
    axis.set_ylabel("episodes")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(length_path, dpi=150)
    plt.close(fig)
    return action_path, length_path


def write_scene_report(manifest: dict[str, Any], report_dir: Path, splits: tuple[str, ...]) -> tuple[Path, dict[str, Any]]:
    scenes = {
        split: sorted({episode["scene_id"] for episode in manifest["splits"][split]["episodes"]})
        for split in splits
    }
    overlaps = {
        "train_seen_val": sorted(set(scenes.get("train", [])) & set(scenes.get("seen_val", []))),
        "train_unseen_test": sorted(set(scenes.get("train", [])) & set(scenes.get("unseen_test", []))),
        "seen_val_unseen_test": sorted(set(scenes.get("seen_val", [])) & set(scenes.get("unseen_test", []))),
    }
    result = {
        "scenes": scenes,
        "overlaps": overlaps,
        "unseen_test_isolated": not overlaps["train_unseen_test"]
        and not overlaps["seen_val_unseen_test"],
    }
    path = report_dir / "scene_split_report.md"
    lines = ["# M5 Scene Split Report", ""]
    for split in splits:
        lines.append(f"- `{split}`: {', '.join(scenes[split])}")
    lines.extend(
        [
            "",
            f"Train / seen-val overlap: {', '.join(overlaps['train_seen_val']) or 'none'}",
            f"Train / unseen-test overlap: {', '.join(overlaps['train_unseen_test']) or 'none'}",
            f"Seen-val / unseen-test overlap: {', '.join(overlaps['seen_val_unseen_test']) or 'none'}",
            "",
            f"Unseen-test scene isolation: **{'PASS' if result['unseen_test_isolated'] else 'FAIL'}**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, result


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_splits = tuple(manifest["splits"])
    if "train" not in selected_splits:
        raise ValueError("dataset manifest requires train split")

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["manifest_policy_fields_exact"] = manifest["policy_fields"] == [
        IMAGE_KEY,
        STATE_KEY,
        ACTION_KEY,
        "task",
    ]
    checks["single_step_action_storage_declared"] = manifest["temporal_storage"].startswith(
        "one action per source frame"
    )
    checks["state_mapping_30d"] = (
        manifest["state_mapping"]["output_dimension"] == 30
        and len(manifest["state_mapping"]["names"]) == 30
    )
    checks["action_mapping_3d"] = manifest["action_mapping"]["names"] == ["vx", "vy", "wz"]

    datasets: dict[str, LeRobotDataset] = {}
    all_actions: list[np.ndarray] = []
    all_lengths: list[int] = []
    split_details: dict[str, Any] = {}
    for split in selected_splits:
        split_info = manifest["splits"][split]
        dataset = load_split(dataset_root / split, split_info, args.video_backend)
        datasets[split] = dataset
        user_features = set(dataset.meta.features) - set(DEFAULT_FEATURES)
        split_checks: dict[str, bool] = {
            "lerobot_load": len(dataset) > 0,
            "feature_allowlist": user_features == EXPECTED_USER_FEATURES,
            "fps_50": dataset.meta.fps == 50,
            "metadata_episode_count": dataset.meta.total_episodes == split_info["episode_count"],
            "metadata_frame_count": dataset.meta.total_frames == split_info["frame_count"],
            "action_shape_3": tuple(dataset.meta.features[ACTION_KEY]["shape"]) == (3,),
            "state_shape_30": tuple(dataset.meta.features[STATE_KEY]["shape"]) == (30,),
            "image_shape_hwc": tuple(dataset.meta.features[IMAGE_KEY]["shape"]) == (512, 512, 3),
        }

        hf = dataset.hf_dataset
        indices = [int(hf[i]["index"]) for i in range(len(hf))]
        timestamps = [float(hf[i]["timestamp"]) for i in range(len(hf))]
        states = np.stack([hf[i][STATE_KEY].numpy() for i in range(len(hf))])
        actions = np.stack([hf[i][ACTION_KEY].numpy() for i in range(len(hf))])
        episode_indices = [int(hf[i]["episode_index"]) for i in range(len(hf))]
        frame_indices = [int(hf[i]["frame_index"]) for i in range(len(hf))]
        split_checks.update(
            {
                "global_indices_contiguous": indices == list(range(len(hf))),
                "state_finite": bool(np.isfinite(states).all()),
                "action_finite": bool(np.isfinite(actions).all()),
                "state_values_shape": states.shape == (len(hf), 30),
                "action_values_shape": actions.shape == (len(hf), 3),
            }
        )

        boundaries_ok = True
        timestamp_ok = True
        observed_lengths: list[int] = []
        cursor = 0
        for episode_index, source_episode in enumerate(split_info["episodes"]):
            length = int(source_episode["frame_count"])
            observed_lengths.append(length)
            stop = cursor + length
            boundaries_ok &= episode_indices[cursor:stop] == [episode_index] * length
            boundaries_ok &= frame_indices[cursor:stop] == list(range(length))
            expected_ts = np.arange(length, dtype=np.float64) / 50.0
            timestamp_ok &= bool(np.allclose(timestamps[cursor:stop], expected_ts, atol=1e-5))
            cursor = stop
        boundaries_ok &= cursor == len(hf)
        split_checks["episode_boundaries"] = boundaries_ok
        split_checks["timestamps_match_50hz"] = timestamp_ok

        sample = dataset[0]
        image = sample[IMAGE_KEY]
        split_checks.update(
            {
                "decoded_image_shape_chw": tuple(image.shape) == (3, 512, 512),
                "decoded_image_dtype_uint8": image.dtype == torch.uint8,
                "decoded_image_range": int(image.min()) >= 0 and int(image.max()) <= 255,
                "task_nonempty": isinstance(sample["task"], str) and bool(sample["task"].strip()),
            }
        )

        loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, num_workers=0)
        batch = next(iter(loader))
        split_checks.update(
            {
                "dataloader_batch": batch[ACTION_KEY].ndim == 2
                and batch[ACTION_KEY].shape[-1] == 3
                and batch[STATE_KEY].shape[-1] == 30,
                "dataloader_finite": finite_tensors(batch),
            }
        )
        split_details[split] = {
            "checks": split_checks,
            "frames": len(dataset),
            "episodes": dataset.meta.total_episodes,
            "sample": {key: tensor_summary(sample[key]) for key in (IMAGE_KEY, STATE_KEY, ACTION_KEY)},
            "batch": {key: tensor_summary(batch[key]) for key in (IMAGE_KEY, STATE_KEY, ACTION_KEY)},
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
            "action_mean": actions.mean(axis=0).tolist(),
        }
        checks[f"{split}_all_checks"] = all(split_checks.values())
        all_actions.append(actions)
        all_lengths.extend(observed_lengths)

    scene_report_path, scene_result = write_scene_report(manifest, report_dir, selected_splits)
    if "unseen_test" in selected_splits:
        checks["unseen_test_scene_isolation"] = scene_result["unseen_test_isolated"]

    # Confirm SmolVLA temporal sampling is produced by the loader from single-step
    # stored actions. This is deliberately tested only on train.
    train_info = manifest["splits"]["train"]
    policy_features = dataset_to_policy_features(datasets["train"].meta.features)
    smolvla_config = SmolVLAConfig(
        input_features={key: value for key, value in policy_features.items() if key.startswith("observation")},
        output_features={ACTION_KEY: policy_features[ACTION_KEY]},
        device="cpu",
        chunk_size=50,
        n_action_steps=50,
    )
    delta_timestamps = resolve_delta_timestamps(smolvla_config, datasets["train"].meta)
    delta_dataset = load_split(
        dataset_root / "train",
        train_info,
        args.video_backend,
        delta_timestamps=delta_timestamps,
    )
    delta_loader = DataLoader(delta_dataset, batch_size=2, shuffle=False, num_workers=0)
    delta_batch = next(iter(delta_loader))
    checks["smolvla_temporal_sampling_action_50x3"] = tuple(delta_batch[ACTION_KEY].shape) == (2, 50, 3)
    checks["smolvla_temporal_sampling_observation_one_step"] = (
        tuple(delta_batch[STATE_KEY].shape) == (2, 1, 30)
        # LeRobot keeps the singleton temporal axis for vector observations,
        # while a single requested video timestamp is returned directly as BCHW.
        and tuple(delta_batch[IMAGE_KEY].shape) == (2, 3, 512, 512)
    )
    checks["smolvla_padding_mask_present"] = (
        f"{ACTION_KEY}_is_pad" in delta_batch
        and tuple(delta_batch[f"{ACTION_KEY}_is_pad"].shape) == (2, 50)
    )
    checks["smolvla_delta_batch_finite"] = finite_tensors(delta_batch)

    # Match the official training loop's uint8 DataLoader bridge before invoking
    # the policy preprocessor (lerobot_train.py converts cameras to float [0, 1]).
    preprocess_batch = dict(delta_batch)
    if preprocess_batch[IMAGE_KEY].dtype == torch.uint8:
        preprocess_batch[IMAGE_KEY] = preprocess_batch[IMAGE_KEY].to(torch.float32) / 255.0
    checks["smolvla_image_bridge_float_0_1"] = (
        preprocess_batch[IMAGE_KEY].dtype == torch.float32
        and float(preprocess_batch[IMAGE_KEY].min()) >= 0.0
        and float(preprocess_batch[IMAGE_KEY].max()) <= 1.0
    )
    preprocessor, _ = make_smolvla_pre_post_processors(
        smolvla_config, dataset_stats=datasets["train"].meta.stats
    )
    processed = preprocessor(preprocess_batch)
    checks["smolvla_preprocessor_runs"] = isinstance(processed, dict)
    checks["smolvla_preprocessor_finite"] = finite_tensors(processed)
    checks["smolvla_language_tokens_present"] = any("language" in key for key in processed)
    details["smolvla"] = {
        "chunk_size": smolvla_config.chunk_size,
        "action_delta_indices": smolvla_config.action_delta_indices,
        "observation_delta_indices": smolvla_config.observation_delta_indices,
        "delta_timestamps": delta_timestamps,
        "delta_batch": {
            key: tensor_summary(value)
            for key, value in delta_batch.items()
            if key in {IMAGE_KEY, STATE_KEY, ACTION_KEY, f"{ACTION_KEY}_is_pad"}
        },
        "processed": {key: tensor_summary(value) for key, value in processed.items()},
    }

    actions_np = np.concatenate(all_actions, axis=0)
    action_histogram, length_histogram = save_histograms(actions_np, all_lengths, report_dir)
    random_samples_path, random_sample_records = save_random_samples(datasets, report_dir, args.seed)
    checks["ten_random_samples_emitted"] = len(random_sample_records) == 10 and random_samples_path.is_file()
    checks["histograms_emitted"] = action_histogram.is_file() and length_histogram.is_file()

    details.update(
        {
            "splits": split_details,
            "scene_split": scene_result,
            "random_samples": random_sample_records,
            "action_distribution": {
                "count": int(actions_np.shape[0]),
                "min": actions_np.min(axis=0).tolist(),
                "max": actions_np.max(axis=0).tolist(),
                "mean": actions_np.mean(axis=0).tolist(),
                "std": actions_np.std(axis=0).tolist(),
            },
            "episode_lengths": {
                "count": len(all_lengths),
                "min": min(all_lengths),
                "max": max(all_lengths),
                "mean": float(np.mean(all_lengths)),
                "values": all_lengths,
            },
        }
    )
    passed = all(checks.values())
    report = {
        "format": "go2-short-vln-m5-check-v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "checks": checks,
        "details": details,
        "artifacts": {
            "random_samples": str(random_samples_path),
            "action_histogram": str(action_histogram),
            "episode_length_histogram": str(length_histogram),
            "scene_split_report": str(scene_report_path),
        },
    }
    json_path = report_dir / "m5_dataset_check.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown_path = report_dir / "m5_dataset_check.md"
    lines = [
        "# M5 LeRobot Dataset Check",
        "",
        f"Status: **{'PASS' if passed else 'FAIL'}** ({report['passed_count']}/{report['check_count']})",
        "",
        f"Frames: {actions_np.shape[0]}; episodes: {len(all_lengths)}; FPS: 50; action dimension: 3; state dimension: 30.",
        "",
        "## Checks",
        "",
        *[f"- [{'x' if value else ' '}] `{name}`" for name, value in checks.items()],
        "",
        "## Artifacts",
        "",
        f"- Random samples: `{random_samples_path}`",
        f"- Action histogram: `{action_histogram}`",
        f"- Episode-length histogram: `{length_histogram}`",
        f"- Scene split report: `{scene_report_path}`",
        "",
        "No policy training was run in M5.",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": checks, "report": str(json_path)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
