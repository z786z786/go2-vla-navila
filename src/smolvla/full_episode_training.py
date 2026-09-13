#!/usr/bin/env python3
"""Fresh SmolVLA training for the full-episode 3-D Go2 policy interface.

Foundation SmolVLA weights are loaded, but no M3/M6/M7 robot checkpoint,
normalizer, action codec, or dataset is accepted as an input.
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

from src.navila_full.contracts import ACTION_KEY, DATASET_SCHEMA_VERSION, IMAGE_KEY, POLICY_CONTRACT, STATE_KEY, required_training_steps, validate_training_plan


DEFAULT_MODEL_ID = "lerobot/smolvla_base"
DEFAULT_MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DEFAULT_SEED = 20260905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("single-batch", "smoke", "train"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--steps", type=int, help="Required for smoke; train derives this from its fresh train anchors.")
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--save-freq", type=int, default=500)
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_conversion_manifest(manifest: dict[str, Any], *, selection_manifest: Path) -> None:
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("conversion manifest is not full-episode 3-D data")
    if Path(manifest.get("final_selection_manifest", "")).resolve() != selection_manifest.resolve():
        raise ValueError("conversion manifest is bound to a different final selection manifest")
    if manifest.get("final_selection_manifest_sha256") != sha256(selection_manifest):
        raise ValueError("final selection manifest hash mismatch")
    if manifest.get("state_mapping", {}).get("output_dimension") != 3:
        raise ValueError("conversion state is not 3-D")
    if manifest.get("action_mapping", {}).get("names") != ["vx", "vy", "wz"]:
        raise ValueError("conversion action is not [vx, vy, wz]")
    terminal = manifest.get("terminal_sampler", {})
    if not terminal.get("passed") or terminal.get("terminal_or_near_zero_selected_ratio", 1.0) > 0.10:
        raise ValueError("terminal-anchor cap is absent or exceeds 10%")
    forbidden = set(manifest.get("forbidden_source_fields_not_copied", ()))
    if not {"reference_path", "gt_locations", "planner_command"} <= forbidden:
        raise ValueError("conversion manifest does not exclude privileged source fields")


def adapt_config_to_dataset(config: SmolVLAConfig, dataset: LeRobotDataset) -> SmolVLAConfig:
    features = dataset_to_policy_features(dataset.meta.features)
    config.input_features = {key: value for key, value in features.items() if key.startswith("observation.")}
    config.output_features = {ACTION_KEY: features[ACTION_KEY]}
    config.device = "cuda"
    config.push_to_hub = False
    config.repo_id = None
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    config.n_obs_steps = 1
    config.chunk_size = 50
    config.n_action_steps = 10
    if set(config.input_features) != {IMAGE_KEY, STATE_KEY}:
        raise ValueError(f"unexpected policy inputs: {sorted(config.input_features)}")
    if config.input_features[STATE_KEY].shape != (3,):
        raise ValueError(f"expected 3-D state, got {config.input_features[STATE_KEY].shape}")
    if config.output_features[ACTION_KEY].shape != (3,):
        raise ValueError(f"expected 3-D action, got {config.output_features[ACTION_KEY].shape}")
    return config


def load_policy_and_data(args: argparse.Namespace) -> tuple[SmolVLAPolicy, Any, Any, LeRobotDataset, LeRobotDataset, list[int], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("full-episode SmolVLA training requires CUDA")
    manifest = json.loads((args.dataset_root / "conversion_manifest.json").read_text(encoding="utf-8"))
    validate_conversion_manifest(manifest, selection_manifest=args.selection_manifest)
    train_base = LeRobotDataset(manifest["splits"]["train"]["repo_id"], root=args.dataset_root / "train", video_backend=args.video_backend, return_uint8=True)
    config = PreTrainedConfig.from_pretrained(args.model_id, revision=args.model_revision)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"expected SmolVLAConfig, got {type(config).__name__}")
    config = adapt_config_to_dataset(config, train_base)
    delta_timestamps = resolve_delta_timestamps(config, train_base.meta)
    train = LeRobotDataset(manifest["splits"]["train"]["repo_id"], root=args.dataset_root / "train", delta_timestamps=delta_timestamps, video_backend=args.video_backend, return_uint8=True)
    seen_val = LeRobotDataset(manifest["splits"]["seen-val"]["repo_id"], root=args.dataset_root / "seen_val", delta_timestamps=delta_timestamps, video_backend=args.video_backend, return_uint8=True)
    selected = [int(value) for value in manifest["terminal_sampler"]["selected_train_indices"]]
    if not selected or min(selected) < 0 or max(selected) >= len(train) or len(selected) != len(set(selected)):
        raise ValueError("invalid capped full-episode train anchor indices")
    policy = SmolVLAPolicy.from_pretrained(args.model_id, revision=args.model_revision, config=config).to("cuda")
    # Stats are copied from the *fresh* full-episode train dataset only.
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=copy.deepcopy(train_base.meta.stats))
    return policy, preprocessor, postprocessor, train, seen_val, selected, manifest


def camera_to_float(batch: dict[str, Any]) -> dict[str, Any]:
    output = dict(batch)
    if output[IMAGE_KEY].dtype == torch.uint8:
        output[IMAGE_KEY] = output[IMAGE_KEY].to(torch.float32) / 255.0
    return output


def optimizer_for(policy: SmolVLAPolicy) -> torch.optim.AdamW:
    config = policy.config
    return torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad], lr=config.optimizer_lr, betas=config.optimizer_betas, eps=config.optimizer_eps, weight_decay=config.optimizer_weight_decay)


def save_checkpoint(root: Path, step: int, policy: SmolVLAPolicy, preprocessor: Any, postprocessor: Any, optimizer: torch.optim.Optimizer) -> Path:
    checkpoint = root / "checkpoints" / f"step_{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, checkpoint / "training_state.pt")
    write_json(checkpoint / "full_episode_checkpoint.json", {"format": "navila-full-episode-smolvla-checkpoint-v1", "step": step, "created_at_utc": now_utc()})
    return checkpoint


def train(args: argparse.Namespace) -> dict[str, Any]:
    policy, preprocessor, postprocessor, train_data, seen_val, selected, manifest = load_policy_and_data(args)
    effective_batch = args.batch_size
    fixed_steps = required_training_steps(len(selected), effective_batch)
    plan = {
        **POLICY_CONTRACT, "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_root": str(args.dataset_root), "selection_manifest": str(args.selection_manifest),
        "acceptance_reference": str(args.output_dir), "source_checkpoint": None,
        "policy_signals": ["body_vx", "body_vy", "body_yaw_rate"],
        "n_train_samples": len(selected), "effective_batch_size": effective_batch,
        "training_steps": fixed_steps,
    }
    errors = validate_training_plan(plan)
    if errors:
        raise RuntimeError("full training contract rejected: " + "; ".join(errors))
    if args.stage == "single-batch":
        steps = 1
    elif args.stage == "smoke":
        if args.steps is None or args.steps <= 0:
            raise ValueError("smoke requires a positive --steps")
        steps = args.steps
    else:
        if args.steps is not None and args.steps != fixed_steps:
            raise ValueError(f"train fixes --steps={fixed_steps} from fresh capped anchors")
        steps = fixed_steps
    subset = Subset(train_data, selected)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, generator=torch.Generator().manual_seed(args.seed))
    optimizer = optimizer_for(policy)
    iterator = iter(loader)
    history: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    started = time.perf_counter()
    policy.train()
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        step_started = time.perf_counter()
        processed = preprocessor(camera_to_float(batch))
        optimizer.zero_grad(set_to_none=True)
        loss, details = policy.forward(processed)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in policy.parameters() if parameter.requires_grad], policy.config.optimizer_grad_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        torch.cuda.synchronize()
        row = {"step": step, "loss": float(loss), "gradient_norm": float(norm), "step_time_s": time.perf_counter() - step_started, "loss_details": details}
        history.append(row)
        if step == 1 or step % args.log_freq == 0 or step == steps:
            print(json.dumps({key: value for key, value in row.items() if key != "loss_details"}), flush=True)
        if step == steps or (args.save_freq > 0 and step % args.save_freq == 0):
            checkpoints.append(str(save_checkpoint(args.output_dir, step, policy, preprocessor, postprocessor, optimizer)))
    result = {
        "format": "navila-full-episode-smolvla-training-v1", "stage": args.stage,
        "completed_at_utc": now_utc(), "passed": all(math.isfinite(row["loss"]) for row in history),
        "model": {"source_id": args.model_id, "revision": args.model_revision, "robot_checkpoint_used": None},
        "dataset_manifest_sha256": sha256(args.dataset_root / "conversion_manifest.json"),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "training_contract": plan,
        "dataset": {"train_episodes": manifest["splits"]["train"]["episode_count"], "train_frames": manifest["splits"]["train"]["frame_count"], "capped_train_anchors": len(selected), "seen_val_episodes": manifest["splits"]["seen-val"]["episode_count"]},
        "history": history, "checkpoints": checkpoints, "wall_time_s": time.perf_counter() - started,
    }
    write_json(args.output_dir / "full_episode_training_report.json", result)
    return result


def main() -> None:
    args = parse_args()
    args.dataset_root, args.selection_manifest, args.output_dir = args.dataset_root.resolve(), args.selection_manifest.resolve(), args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    set_determinism(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        result = train(args)
    except Exception as error:
        write_json(args.output_dir / "full_episode_training_failure.json", {"format": "navila-full-episode-training-failure-v1", "error_type": type(error).__name__, "error": str(error)})
        raise
    print(json.dumps({"passed": result["passed"], "report": str(args.output_dir / "full_episode_training_report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
