#!/usr/bin/env python3
"""Audit D5 expert coverage, LeRobot sampling, and action normalizers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.smolvla.d5_data_expansion import classify_normalizer_shift, summarize_regimes


ACTION_KEY = "action"
SPLIT_DIRS = {"train": "train", "seen-val": "seen_val"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("statistics require finite non-empty values")
    p5, p50, p95 = np.quantile(values, (0.05, 0.5, 0.95))
    return {
        "mean": float(values.mean()), "std": float(values.std()), "p5": float(p5), "p50": float(p50), "p95": float(p95),
        "min": float(values.min()), "max": float(values.max()),
    }


def _action_statistics(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3 or not np.isfinite(actions).all():
        raise ValueError("actions must be finite [N,3]")
    return {name: _distribution(actions[:, index]) for index, name in enumerate(("vx", "vy", "wz"))}


def _load_expert_records(root: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(root.rglob("summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        split = str(summary.get("split"))
        if split not in SPLIT_DIRS:
            continue
        for line in (summary_path.parent / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["episode_id"] = str(summary["short_episode_id"])
            row["scene_id"] = str(summary["scene_id"])
            records[split].append(row)
    if not records:
        raise ValueError(f"no M4 expert records under {root}")
    return records


def _load_base_dataset(dataset_root: Path, split: str, video_backend: str) -> tuple[Any, dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads((dataset_root / "conversion_manifest.json").read_text(encoding="utf-8"))
    info = manifest["splits"][split]
    return LeRobotDataset(info["repo_id"], root=dataset_root / split, video_backend=video_backend, return_uint8=True), manifest


def _physical_actions(dataset: Any) -> np.ndarray:
    values = [dataset.hf_dataset[index][ACTION_KEY].numpy() for index in range(len(dataset))]
    return np.stack(values).astype(np.float32)


def _latent_actions(actions: np.ndarray, epsilon: float) -> np.ndarray:
    import torch

    from src.smolvla.bounded_actions import encode_physical_actions

    return encode_physical_actions(torch.from_numpy(actions), epsilon=epsilon).numpy()


def _saved_action_normalizer(checkpoint: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    state = load_file(str(checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    output: dict[str, Any] = {}
    for field in ("count", "min", "max", "mean", "std"):
        key = f"action.{field}"
        if key not in state:
            raise ValueError(f"checkpoint normalizer lacks {key}")
        output[field] = state[key].cpu().tolist()
    return output


def _sampler_audit(dataset_root: Path, split: str, checkpoint: Path, video_backend: str) -> dict[str, Any]:
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    base, manifest = _load_base_dataset(dataset_root, split, video_backend)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError("checkpoint is not SmolVLA")
    deltas = resolve_delta_timestamps(config, base.meta)
    info = manifest["splits"][split]
    sampled = LeRobotDataset(info["repo_id"], root=dataset_root / split, delta_timestamps=deltas, video_backend=video_backend, return_uint8=True)
    spans = []
    total_valid_targets = 0
    pad_observations = []
    for start, stop in zip(sampled.meta.episodes["dataset_from_index"], sampled.meta.episodes["dataset_to_index"], strict=True):
        length = int(stop) - int(start)
        total_valid_targets += sum(min(config.chunk_size, length - index) for index in range(length))
        terminal = sampled[int(stop) - 1]
        mask = terminal[f"{ACTION_KEY}_is_pad"].bool().cpu().numpy()
        pad_observations.append({"frame_index": length - 1, "valid_targets": int((~mask).sum()), "padded_targets": int(mask.sum())})
        spans.append({"start": int(start), "stop": int(stop), "frames": length})
    return {
        "sampling_unit": "frame_anchor",
        "base_dataset_length": len(base),
        "sampled_dataset_length": len(sampled),
        "trainable_anchor_count": len(sampled),
        "action_chunk_size": config.chunk_size,
        "action_delta_indices": config.action_delta_indices,
        "observation_delta_indices": config.observation_delta_indices,
        "episode_spans": spans,
        "terminal_padding_observations": pad_observations,
        "total_valid_action_targets": total_valid_targets,
        "mean_valid_action_targets_per_anchor": total_valid_targets / len(sampled),
        "dataloader_shuffle_unit": "dataset index / frame anchor (PyTorch RandomSampler when training uses shuffle=True)",
    }


def _normalizer_report(actions: np.ndarray, epsilon: float) -> dict[str, Any]:
    return {"physical": _action_statistics(actions), "latent": _action_statistics(_latent_actions(actions, epsilon))}


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# D5 Dataset / Sampler / Normalizer Audit", ""]
    for split, summary in report["frame_regimes"].items():
        lines.extend([f"## {split} frame-level action regimes", "", "| Regime | Frames | % | Episodes | Scenes |", "|---|---:|---:|---:|---:|"])
        for name, row in summary["wz_regimes"].items():
            lines.append(f"| {name} | {row['frame_count']} | {row['percentage']:.2f} | {row['episode_coverage']['count']} | {row['scene_coverage']['count']} |")
        near = summary["near_goal"]
        lines.append(f"| NEAR_GOAL (<{summary['near_goal_radius_m']}m, overlapping) | {near['frame_count']} | {near['percentage']:.2f} | {near['episode_coverage']['count']} | {near['scene_coverage']['count']} |")
        lines.append("")
    sampler = report["sampler_audit"]
    lines.extend(["## Sampler", "", f"- Unit: `{sampler['sampling_unit']}`", f"- D5 trainable anchors: `{sampler['trainable_anchor_count']}`", f"- D5 valid action targets: `{sampler['total_valid_action_targets']}`", "", "## Normalizer", "", f"- Substantial latent action-stat shift: **{report['normalizer_shift']['substantial']}**", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expert = _load_expert_records(args.expert_root.resolve())
    frame_regimes = {split: summarize_regimes(records) for split, records in expert.items()}
    baseline, _ = _load_base_dataset(args.baseline_dataset_root.resolve(), "train", args.video_backend)
    expanded, _ = _load_base_dataset(args.dataset_root.resolve(), "train", args.video_backend)
    epsilon = float(json.loads((args.baseline_checkpoint / "bounded_action_codec.json").read_text(encoding="utf-8"))["epsilon"])
    baseline_norm = _normalizer_report(_physical_actions(baseline), epsilon)
    expanded_norm = _normalizer_report(_physical_actions(expanded), epsilon)
    saved = _saved_action_normalizer(args.baseline_checkpoint.resolve())
    report = {
        "format": "go2-short-vln-m6_2-d5-audit-v1",
        "inputs": {"expert_root": str(args.expert_root.resolve()), "dataset_root": str(args.dataset_root.resolve()), "baseline_dataset_root": str(args.baseline_dataset_root.resolve()), "baseline_checkpoint": str(args.baseline_checkpoint.resolve())},
        "frame_regimes": frame_regimes,
        "sampler_audit": _sampler_audit(args.dataset_root.resolve(), "train", args.baseline_checkpoint.resolve(), args.video_backend),
        "baseline_sampler_audit": _sampler_audit(args.baseline_dataset_root.resolve(), "train", args.baseline_checkpoint.resolve(), args.video_backend),
        "normalizer_statistics": {"baseline": baseline_norm, "expanded": expanded_norm, "baseline_saved_preprocessor": saved},
        "normalizer_shift": classify_normalizer_shift(baseline_norm["latent"], expanded_norm["latent"]),
    }
    write_json(output / "d5_dataset_audit.json", report)
    (output / "d5_dataset_audit.md").write_text(_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "normalizer_shift": report["normalizer_shift"]["substantial"], "sampler": report["sampler_audit"]["sampling_unit"]}, indent=2))


if __name__ == "__main__":
    main()
