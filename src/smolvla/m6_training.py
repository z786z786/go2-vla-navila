#!/usr/bin/env python3
"""M6 SmolVLA training bring-up for the Go2 short-VLN LeRobot dataset.

The script keeps the pinned upstream SmolVLA architecture and weights while
replacing the checkpoint's robot-specific feature declaration with the M5
policy interface: one front RGB image, a 30-D state, and a 3-D action chunk.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features

from src.smolvla.bounded_actions import (
    CODEC_FILENAME,
    DEFAULT_EPSILON,
    codec_metadata,
    decode_latent_actions,
    encode_physical_actions,
    load_codec_metadata,
    save_codec_metadata,
)
from src.smolvla.training_acceptance import smoke_acceptance_checks


IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_MODEL_ID = "lerobot/smolvla_base"
DEFAULT_MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DEFAULT_DATASET_ROOT = Path("<external-data-root>")
DEFAULT_SEED = 20260831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("single-batch", "tiny-overfit", "smoke"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--bounded-actions", action="store_true")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--codec-epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        help="Optional exact positive training steps to checkpoint; must include the final step.",
    )
    parser.add_argument(
        "--action-normalizer-checkpoint",
        type=Path,
        help="Reuse only the saved action normalizer statistics from this bounded checkpoint.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def gpu_snapshot() -> dict[str, Any]:
    return {
        "name": torch.cuda.get_device_name(0),
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def model_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_id": DEFAULT_MODEL_ID,
        "revision": args.model_revision,
        "load_path": args.model_id,
    }


def load_manifest(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "conversion_manifest.json").read_text(encoding="utf-8"))


def load_base_dataset(
    dataset_root: Path, split: str, manifest: dict[str, Any], video_backend: str
) -> LeRobotDataset:
    info = manifest["splits"][split]
    return LeRobotDataset(
        info["repo_id"],
        root=dataset_root / split,
        video_backend=video_backend,
        return_uint8=True,
    )


def encoded_action_statistics(dataset: LeRobotDataset, *, epsilon: float) -> dict[str, Any]:
    """Replace only the action statistics using every physical train frame once."""
    chunks: list[torch.Tensor] = []
    for batch in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
        chunks.append(encode_physical_actions(batch[ACTION_KEY].to(torch.float64), epsilon))
    actions = torch.cat(chunks, dim=0)
    if actions.ndim != 2 or actions.shape != (len(dataset), 3):
        raise ValueError(f"expected all train frames as ({len(dataset)}, 3), got {tuple(actions.shape)}")
    return {
        "min": actions.min(dim=0).values.tolist(),
        "max": actions.max(dim=0).values.tolist(),
        "mean": actions.mean(dim=0).tolist(),
        "std": actions.std(dim=0, unbiased=False).tolist(),
        "count": [int(actions.shape[0])],
    }


def saved_action_normalizer_statistics(checkpoint: Path) -> dict[str, Any]:
    """Load the persisted action normalizer, for the D5 normalization ablation only."""
    from safetensors.torch import load_file

    state_path = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    tensors = load_file(str(state_path))
    keys = ("count", "min", "max", "mean", "std")
    missing = [f"action.{key}" for key in keys if f"action.{key}" not in tensors]
    if missing:
        raise ValueError(f"action normalizer is missing {missing}")
    return {key: tensors[f"action.{key}"].detach().clone() for key in keys}


def adapt_config_to_dataset(config: SmolVLAConfig, dataset: LeRobotDataset) -> SmolVLAConfig:
    features = dataset_to_policy_features(dataset.meta.features)
    config.input_features = {
        key: value for key, value in features.items() if key.startswith("observation.")
    }
    config.output_features = {ACTION_KEY: features[ACTION_KEY]}
    config.device = "cuda"
    config.push_to_hub = False
    config.repo_id = None
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    if set(config.input_features) != {IMAGE_KEY, STATE_KEY}:
        raise ValueError(f"Unexpected policy inputs: {sorted(config.input_features)}")
    if config.input_features[STATE_KEY].shape != (30,):
        raise ValueError(f"Expected 30-D state, got {config.input_features[STATE_KEY].shape}")
    if config.output_features[ACTION_KEY].shape != (3,):
        raise ValueError(f"Expected 3-D action, got {config.output_features[ACTION_KEY].shape}")
    return config


def load_policy_and_data(
    args: argparse.Namespace,
) -> tuple[
    SmolVLAPolicy, Any, Any, LeRobotDataset, LeRobotDataset, dict[str, Any], dict[str, Any] | None
]:
    if not torch.cuda.is_available():
        raise RuntimeError("M6 requires CUDA")
    manifest = load_manifest(args.dataset_root)
    base_train = load_base_dataset(args.dataset_root, "train", manifest, args.video_backend)
    config = PreTrainedConfig.from_pretrained(args.model_id, revision=args.model_revision)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config = adapt_config_to_dataset(config, base_train)
    delta_timestamps = resolve_delta_timestamps(config, base_train.meta)
    train = LeRobotDataset(
        manifest["splits"]["train"]["repo_id"],
        root=args.dataset_root / "train",
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    seen_val = LeRobotDataset(
        manifest["splits"]["seen_val"]["repo_id"],
        root=args.dataset_root / "seen_val",
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    policy = SmolVLAPolicy.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        config=config,
    ).to("cuda")
    stats = copy.deepcopy(train.meta.stats)
    codec = None
    if args.bounded_actions:
        if args.source_checkpoint is None:
            raise ValueError("--bounded-actions requires --source-checkpoint")
        source_checkpoint = args.source_checkpoint.resolve()
        source_model = source_checkpoint / "model.safetensors"
        if not source_model.is_file():
            raise FileNotFoundError(source_model)
        stats[ACTION_KEY] = (
            saved_action_normalizer_statistics(args.action_normalizer_checkpoint.resolve())
            if args.action_normalizer_checkpoint is not None
            else encoded_action_statistics(base_train, epsilon=args.codec_epsilon)
        )
        codec = codec_metadata(
            source_checkpoint_sha256=sha256(source_model), epsilon=args.codec_epsilon
        )
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=stats)
    return policy, preprocessor, postprocessor, train, seen_val, manifest, codec


def camera_to_float(batch: dict[str, Any]) -> dict[str, Any]:
    batch = dict(batch)
    image = batch[IMAGE_KEY]
    if image.dtype == torch.uint8:
        batch[IMAGE_KEY] = image.to(torch.float32) / 255.0
    return batch


def policy_batch(
    batch: dict[str, Any], preprocessor: Any, *, codec_epsilon: float | None = None
) -> dict[str, Any]:
    policy_input = camera_to_float(batch)
    if codec_epsilon is not None:
        policy_input[ACTION_KEY] = encode_physical_actions(
            policy_input[ACTION_KEY], epsilon=codec_epsilon
        )
    return preprocessor(policy_input)


def tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def valid_probe_indices(dataset: LeRobotDataset, count: int) -> list[int]:
    candidates: list[int] = []
    episodes = dataset.meta.episodes
    for start, stop in zip(
        episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True
    ):
        first, last = int(start), max(int(start), int(stop) - 50)
        n = max(1, count // max(1, dataset.meta.total_episodes))
        candidates.extend(np.linspace(first, last, n, dtype=int).tolist())
    candidates = sorted(dict.fromkeys(candidates))
    if len(candidates) < count:
        all_safe = []
        for start, stop in zip(
            episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True
        ):
            all_safe.extend(range(int(start), max(int(start) + 1, int(stop) - 49)))
        for index in all_safe:
            if index not in candidates:
                candidates.append(index)
            if len(candidates) >= count:
                break
    return candidates[:count]


@torch.no_grad()
def action_mae(
    policy: SmolVLAPolicy,
    preprocessor: Any,
    postprocessor: Any,
    dataset: LeRobotDataset,
    indices: list[int],
    seed: int,
    codec_epsilon: float | None = None,
) -> dict[str, Any]:
    policy.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    errors: list[torch.Tensor] = []
    predicted: list[torch.Tensor] = []
    expected: list[torch.Tensor] = []
    for batch in DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0):
        target = batch[ACTION_KEY].clone().to(torch.float32)
        pad = batch.get(f"{ACTION_KEY}_is_pad")
        processed = policy_batch(batch, preprocessor, codec_epsilon=codec_epsilon)
        noise = torch.randn(
            (1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator
        ).to("cuda")
        policy.reset()
        pred = policy.predict_action_chunk(processed, noise=noise)
        pred = postprocessor(pred).to(torch.float32)
        if codec_epsilon is not None:
            pred = decode_latent_actions(pred)
        valid = torch.ones(target.shape[:2], dtype=torch.bool) if pad is None else ~pad.bool()
        diff = torch.abs(pred - target)
        errors.append(diff[valid])
        predicted.append(pred[valid])
        expected.append(target[valid])
    errors_cat = torch.cat(errors)
    pred_cat, target_cat = torch.cat(predicted), torch.cat(expected)
    return {
        "indices": indices,
        "mae": float(errors_cat.mean()),
        "per_dimension_mae": errors_cat.mean(dim=0).tolist(),
        "prediction_min": pred_cat.min(dim=0).values.tolist(),
        "prediction_max": pred_cat.max(dim=0).values.tolist(),
        "target_min": target_cat.min(dim=0).values.tolist(),
        "target_max": target_cat.max(dim=0).values.tolist(),
        "output_shape": [1, policy.config.chunk_size, policy.config.action_feature.shape[0]],
    }


@torch.no_grad()
def fixed_loss(
    policy: SmolVLAPolicy,
    preprocessor: Any,
    dataset: LeRobotDataset,
    indices: list[int],
    seed: int,
    codec_epsilon: float | None = None,
) -> float:
    policy.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for batch in DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0):
        processed = policy_batch(batch, preprocessor, codec_epsilon=codec_epsilon)
        noise = torch.randn(
            (1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator
        ).to("cuda")
        sample_time = torch.rand((1,), generator=generator).to("cuda")
        loss, _ = policy.forward(processed, noise=noise, time=sample_time)
        losses.append(float(loss))
    policy.train()
    return float(np.mean(losses))


def optimizer_for(policy: SmolVLAPolicy) -> torch.optim.AdamW:
    config = policy.config
    return torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.optimizer_lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )


def lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, decay_steps: int, final_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_ratio + (1.0 - final_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def checkpoint_size_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def save_checkpoint(
    root: Path,
    step: int,
    policy: SmolVLAPolicy,
    preprocessor: Any,
    postprocessor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    codec: dict[str, Any] | None = None,
) -> Path:
    checkpoint = root / "checkpoints" / f"step_{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    if codec is not None:
        save_codec_metadata(
            checkpoint,
            source_checkpoint_sha256=codec["source_checkpoint_sha256"],
            epsilon=float(codec["epsilon"]),
        )
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint / "training_state.pt",
    )
    write_json(
        checkpoint / "m6_checkpoint.json",
        {
            "step": step,
            "saved_at_utc": now_utc(),
            "size_bytes": checkpoint_size_bytes(checkpoint),
            "bounded_action_codec": CODEC_FILENAME if codec is not None else None,
        },
    )
    return checkpoint


def reload_and_infer(
    checkpoint: Path,
    train: LeRobotDataset,
    probe_indices: list[int],
    seed: int,
    bounded_actions: bool = False,
) -> dict[str, Any]:
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    codec = load_codec_metadata(checkpoint) if bounded_actions else None
    metric = action_mae(
        policy,
        preprocessor,
        postprocessor,
        train,
        probe_indices,
        seed,
        codec_epsilon=float(codec["epsilon"]) if codec is not None else None,
    )
    metric["policy_config_inputs"] = sorted(config.input_features)
    metric["policy_config_action_shape"] = list(config.output_features[ACTION_KEY].shape)
    metric["finite"] = bool(math.isfinite(metric["mae"]))
    metric["bounded_action_codec"] = codec
    return metric


def run_single_batch(args: argparse.Namespace) -> dict[str, Any]:
    policy, preprocessor, _, train, _, manifest, codec = load_policy_and_data(args)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    raw = next(iter(loader))
    processed = policy_batch(raw, preprocessor, codec_epsilon=codec["epsilon"] if codec else None)
    optimizer = optimizer_for(policy)
    policy.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss, details = policy.forward(processed)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite single-batch loss: {loss}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        policy.config.optimizer_grad_clip_norm,
    )
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "format": "go2-short-vln-m6-single-batch-v1",
        "stage": "single-batch",
        "passed": bool(math.isfinite(float(loss)) and math.isfinite(float(grad_norm))),
        "checked_at_utc": now_utc(),
        "model": model_provenance(args),
        "dataset_manifest_sha256": sha256(args.dataset_root / "conversion_manifest.json"),
        "batch_size": args.batch_size,
        "raw_shapes": tensor_shapes(raw),
        "processed_shapes": tensor_shapes(processed),
        "loss": float(loss),
        "loss_details": details,
        "gradient_norm": float(grad_norm),
        "step_time_s": elapsed,
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in policy.parameters()),
        "gpu": gpu_snapshot(),
        "split_frames": {
            split: manifest["splits"][split]["frame_count"] for split in manifest["splits"]
        },
    }
    return result


def plot_metrics(history: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "training_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    steps = [row["step"] for row in history]
    axes[0].plot(steps, [row["loss"] for row in history], color="#286090", linewidth=1)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("flow-matching loss")
    axes[0].grid(alpha=0.25)
    eval_rows = [row for row in history if row.get("val_loss") is not None]
    if eval_rows:
        axes[1].plot(
            [row["step"] for row in eval_rows],
            [row["val_loss"] for row in eval_rows],
            marker="o",
            color="#c65f22",
        )
    axes[1].set_title("Fixed seen-val loss")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage
    default_steps = 500 if stage == "tiny-overfit" else 2000
    steps = args.steps or default_steps
    default_log_freq = 10 if stage == "tiny-overfit" else 50
    log_freq = args.log_freq or default_log_freq
    default_save_freq = steps if stage == "tiny-overfit" else 500
    save_freq = args.save_freq or default_save_freq
    warmup_steps = 50 if stage == "tiny-overfit" else 1000
    decay_steps = steps if stage == "tiny-overfit" else 30000
    final_ratio = 0.1 if stage == "tiny-overfit" else 0.025
    checkpoint_steps = None if args.checkpoint_steps is None else set(args.checkpoint_steps)
    if checkpoint_steps is not None:
        if any(step <= 0 or step > steps for step in checkpoint_steps) or steps not in checkpoint_steps:
            raise ValueError("--checkpoint-steps must be positive, within --steps, and include the final step")

    policy, preprocessor, postprocessor, train, seen_val, manifest, codec = load_policy_and_data(args)
    codec_epsilon = float(codec["epsilon"]) if codec is not None else None
    probe_indices = valid_probe_indices(train, args.probe_count)
    val_indices = valid_probe_indices(seen_val, args.val_count)
    pretrain_probe = action_mae(
        policy, preprocessor, postprocessor, train, probe_indices, args.seed + 11, codec_epsilon
    )
    policy.train()
    optimizer = optimizer_for(policy)
    scheduler = lr_scheduler(optimizer, warmup_steps, decay_steps, final_ratio)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    iterator = iter(loader)
    history: list[dict[str, Any]] = []
    checkpoints: list[Path] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for step in range(1, steps + 1):
        step_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        processed = policy_batch(batch, preprocessor, codec_epsilon=codec_epsilon)
        optimizer.zero_grad(set_to_none=True)
        loss, details = policy.forward(processed)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            policy.config.optimizer_grad_clip_norm,
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start
        row: dict[str, Any] = {
            "step": step,
            "loss": float(loss),
            "gradient_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
            "step_time_s": elapsed,
            "samples_per_s": args.batch_size / elapsed,
            "val_loss": None,
            "loss_details": details,
        }
        if stage == "smoke" and step % save_freq == 0:
            row["val_loss"] = fixed_loss(
                policy, preprocessor, seen_val, val_indices, args.seed + 23, codec_epsilon
            )
        history.append(row)
        if step % log_freq == 0 or step == 1:
            print(json.dumps({key: value for key, value in row.items() if key != "loss_details"}), flush=True)
        if (checkpoint_steps is None and (step % save_freq == 0 or step == steps)) or (
            checkpoint_steps is not None and step in checkpoint_steps
        ):
            checkpoints.append(
                save_checkpoint(
                    args.output_dir,
                    step,
                    policy,
                    preprocessor,
                    postprocessor,
                    optimizer,
                    scheduler,
                    codec,
                )
            )

    wall_time = time.perf_counter() - started
    first_window = [row["loss"] for row in history[: min(25, len(history))]]
    last_window = [row["loss"] for row in history[-min(25, len(history)) :]]
    final_checkpoint = checkpoints[-1]
    del policy, optimizer, scheduler, iterator, loader
    torch.cuda.empty_cache()
    reloaded = reload_and_infer(
        final_checkpoint,
        train,
        probe_indices,
        args.seed + 11,
        bounded_actions=args.bounded_actions,
    )
    loss_ratio = float(np.mean(last_window) / np.mean(first_window))
    mae_ratio = float(reloaded["mae"] / pretrain_probe["mae"])
    tiny_passed = loss_ratio <= 0.60 and mae_ratio <= 0.80
    smoke_checks = smoke_acceptance_checks(steps, history, reloaded)
    smoke_passed = all(smoke_checks.values())
    curve_path = plot_metrics(history, args.output_dir)
    write_json(args.output_dir / "training_history.json", history)
    result = {
        "format": "go2-short-vln-m6_1-training-v1" if args.bounded_actions else "go2-short-vln-m6-training-v1",
        "stage": stage,
        "passed": tiny_passed if stage == "tiny-overfit" else smoke_passed,
        "completed_at_utc": now_utc(),
        "model": model_provenance(args),
        "dataset_manifest_sha256": sha256(args.dataset_root / "conversion_manifest.json"),
        "dataset": {
            "train_episodes": manifest["splits"]["train"]["episode_count"],
            "train_frames": manifest["splits"]["train"]["frame_count"],
            "seen_val_episodes": manifest["splits"]["seen_val"]["episode_count"],
            "seen_val_frames": manifest["splits"]["seen_val"]["frame_count"],
        },
        "bounded_action_codec": codec,
        "training": {
            "steps": steps,
            "batch_size": args.batch_size,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "optimizer_lr": 1e-4,
            "first_25_loss_mean": float(np.mean(first_window)),
            "last_25_loss_mean": float(np.mean(last_window)),
            "loss_ratio": loss_ratio,
            "mean_step_time_s": float(np.mean([row["step_time_s"] for row in history])),
            "mean_samples_per_s": float(np.mean([row["samples_per_s"] for row in history])),
            "wall_time_s": wall_time,
        },
        "pretrain_probe": pretrain_probe,
        "reloaded_checkpoint_probe": reloaded,
        "probe_mae_ratio": mae_ratio,
        "validation": [
            {"step": row["step"], "loss": row["val_loss"]}
            for row in history
            if row["val_loss"] is not None
        ],
        "checkpoints": [
            {"path": str(path), "size_bytes": checkpoint_size_bytes(path)} for path in checkpoints
        ],
        "gpu": gpu_snapshot(),
        "artifacts": {
            "history": str(args.output_dir / "training_history.json"),
            "curves": str(curve_path),
        },
        "acceptance": {
            "tiny_loss_ratio_max": 0.60,
            "tiny_mae_ratio_max": 0.80,
            "checkpoint_reloaded": reloaded["finite"],
            "output_action_shape": reloaded["output_shape"],
            "smoke_checks": smoke_checks,
        },
    }
    return result


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = now_utc()
    try:
        result = run_single_batch(args) if args.stage == "single-batch" else run_training(args)
    except Exception as error:
        failure = {
            "format": "go2-short-vln-m6-failure-v1",
            "stage": args.stage,
            "started_at_utc": started,
            "failed_at_utc": now_utc(),
            "error_type": type(error).__name__,
            "error": str(error),
            "gpu": gpu_snapshot() if torch.cuda.is_available() else None,
        }
        write_json(args.output_dir / "m6_failure.json", failure)
        raise
    result["started_at_utc"] = started
    report_path = args.output_dir / "m6_report.json"
    write_json(report_path, result)
    print(json.dumps({"passed": result["passed"], "stage": args.stage, "report": str(report_path)}, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
