#!/usr/bin/env python3
"""D2: derive auditable training-action diagnostics from frozen M6.1 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.smolvla.evaluate_train_actions import ACTION_NAMES, action_metrics, baseline_metrics


DEFAULT_DATASET_ROOT = Path("<external-data-root>")
DEFAULT_CHECKPOINT = Path(
    "<external-data-root>+0800/checkpoints/step_002000"
)
DEFAULT_SOURCE_EVIDENCE = Path(
    "<external-data-root>+0800"
)
FROZEN_MODEL_SHA256 = "facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7"
FROZEN_CODEC_SHA256 = "143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27"
FROZEN_MANIFEST_SHA256 = "d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402"
FROZEN_SOURCE_HASHES = {
    "manifest.json": "8316a6f802b0f747d7c5c68d6fdbf7eba4683187655dbee1ba1a5d219ed234d9",
    "metrics.json": "ccca94d1302ec57cea55a65729c74e4a4bffea5657741b10900b40e2847b153b",
    "predictions.npz": "834cb049f2dda4bd74498ebd68129f3c3c160cbec98b45f091d9ade775d0292e",
    "report.md": "a98245cfd2c2c8e1bebbdf5dc8d8ba4385336528fe566b8a92f639faa026bc73",
}
FROZEN_EVALUATOR_HASHES = {
    "src/smolvla/evaluate_train_actions.py": "7cda15a12b35dbc2b2bde805b9f3eaf3f48a0c40491d047e68adce99f30affaa",
    "tests/smolvla/test_train_action_eval.py": "581b6917b0381325e024f8d01c667d5abc2cd1f77d0bcc38fbf0db490637d1c6",
    "scripts/m7_run_offline_train_eval.sh": "6eef8e7a4c6f3d959beb5ddfdb2b33fe1571d3b1306e9b82b117af59628e29d2",
}
EXPECTED_EPISODE_COUNTS = {"short_vln_v1_0000": 427, "short_vln_v1_0004": 420}
EXPECTED_SEEDS = (20260831, 20260832, 20260833)
INPUT_ALLOWLIST = ["observation.images.front", "observation.state", "task"]
FPS = 50
DT_SECONDS = 1.0 / FPS
WZ_SIGN_DEADBAND = 0.05
MAGNITUDE_RATIO_LIMITS = (0.9, 1.1)
PEARSON_MIN = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(array: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _validate_physical(array: np.ndarray, label: str) -> None:
    _finite(array, label)
    if not bool(np.all((array[..., 0] >= 0.0) & (array[..., 0] <= 0.5))):
        raise ValueError(f"{label} vx is outside [0, 0.5]")
    if not bool(np.all(array[..., 1] == 0.0)):
        raise ValueError(f"{label} vy must be exactly zero")
    if not bool(np.all((array[..., 2] >= -0.5) & (array[..., 2] <= 0.5))):
        raise ValueError(f"{label} wz is outside [-0.5, 0.5]")


def validate_prediction_arrays(
    physical_predictions: np.ndarray,
    latent_predictions: np.ndarray,
    expert_action_chunks: np.ndarray,
    action_valid_mask: np.ndarray,
    episode_ids: list[str] | np.ndarray,
    local_frame_indices: np.ndarray,
    base_seeds: np.ndarray,
    *,
    expected_episode_counts: dict[str, int] | None = None,
) -> None:
    expected_episode_counts = EXPECTED_EPISODE_COUNTS if expected_episode_counts is None else expected_episode_counts
    frame_count = sum(expected_episode_counts.values())
    physical_predictions = np.asarray(physical_predictions)
    latent_predictions = np.asarray(latent_predictions)
    expert_action_chunks = np.asarray(expert_action_chunks)
    action_valid_mask = np.asarray(action_valid_mask)
    local_frame_indices = np.asarray(local_frame_indices)
    seeds = np.asarray(base_seeds)
    if physical_predictions.shape != (3, frame_count, 50, 3):
        raise ValueError(f"physical_predictions must have shape [3,{frame_count},50,3]")
    if latent_predictions.shape != physical_predictions.shape:
        raise ValueError("latent_predictions must match physical_predictions")
    if expert_action_chunks.shape != (frame_count, 50, 3):
        raise ValueError(f"expert_action_chunks must have shape [{frame_count},50,3]")
    if action_valid_mask.shape != (frame_count, 50) or action_valid_mask.dtype != bool:
        raise ValueError(f"action_valid_mask must have bool shape [{frame_count},50]")
    if local_frame_indices.shape != (frame_count,):
        raise ValueError(f"local_frame_indices must have shape [{frame_count}]")
    if seeds.shape != (3,) or len(set(int(seed) for seed in seeds)) != 3:
        raise ValueError("base_seeds must contain exactly three distinct seeds")
    _validate_physical(physical_predictions, "physical_predictions")
    _finite(latent_predictions, "latent_predictions")
    _validate_physical(expert_action_chunks, "expert_action_chunks")
    values = [str(value) for value in episode_ids]
    expected_ids = list(expected_episode_counts)
    if len(values) != frame_count or set(values) != set(expected_ids):
        raise ValueError("episode IDs do not match the frozen train split")
    start = 0
    for episode_id, count in expected_episode_counts.items():
        stop = start + count
        if values[start:stop] != [episode_id] * count:
            raise ValueError("episode IDs are not contiguous in frozen train order")
        if not np.array_equal(local_frame_indices[start:stop], np.arange(count, dtype=local_frame_indices.dtype)):
            raise ValueError("local_frame_indices are not contiguous per episode")
        start = stop
    if not bool(np.all(action_valid_mask[:, 0])):
        raise ValueError("first action offset must be valid for every training observation")


def _with_distribution_ratios(metrics: dict[str, Any]) -> dict[str, Any]:
    for name in ACTION_NAMES:
        row = metrics["per_action"][name]
        model = row["model"]
        expert = row["expert"]
        model_abs_mean = row.pop("_model_abs_mean", None)
        expert_abs_mean = row.pop("_expert_abs_mean", None)
        if model_abs_mean is None or expert_abs_mean is None:
            raise ValueError("internal D2 metric construction error")
        row["mean_abs_model"] = model_abs_mean
        row["mean_abs_expert"] = expert_abs_mean
        row["mean_abs_magnitude_ratio"] = None if expert_abs_mean == 0.0 else float(model_abs_mean / expert_abs_mean)
        row["std_ratio"] = None if expert["std"] == 0.0 else float(model["std"] / expert["std"])
    return metrics


def _first_step_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    metrics = action_metrics(predicted, expected)
    for index, name in enumerate(ACTION_NAMES):
        metrics["per_action"][name]["_model_abs_mean"] = float(np.abs(predicted[:, index]).mean())
        metrics["per_action"][name]["_expert_abs_mean"] = float(np.abs(expected[:, index]).mean())
    return _with_distribution_ratios(metrics)


def _cumulative_controls(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    return {
        "dt_seconds": DT_SECONDS,
        "model_sum_vx_dt": float(predicted[:, 0].sum() * DT_SECONDS),
        "expert_sum_vx_dt": float(expected[:, 0].sum() * DT_SECONDS),
        "model_sum_abs_wz_dt": float(np.abs(predicted[:, 2]).sum() * DT_SECONDS),
        "expert_sum_abs_wz_dt": float(np.abs(expected[:, 2]).sum() * DT_SECONDS),
    }


def compute_d2_metrics(
    physical_predictions: np.ndarray,
    expert_action_chunks: np.ndarray,
    action_valid_mask: np.ndarray,
    episode_ids: list[str] | np.ndarray,
    local_frame_indices: np.ndarray,
) -> dict[str, Any]:
    physical_predictions = np.asarray(physical_predictions, dtype=np.float64)
    expert_action_chunks = np.asarray(expert_action_chunks, dtype=np.float64)
    action_valid_mask = np.asarray(action_valid_mask, dtype=bool)
    if physical_predictions.ndim != 4 or physical_predictions.shape[0] != 3 or physical_predictions.shape[-2:] != (50, 3):
        raise ValueError("physical_predictions must have shape [3,frames,50,3]")
    if expert_action_chunks.shape != physical_predictions.shape[1:]:
        raise ValueError("expert_action_chunks must match one prediction seed")
    if action_valid_mask.shape != physical_predictions.shape[1:3]:
        raise ValueError("action_valid_mask must match prediction frame and horizon dimensions")
    values = np.asarray([str(value) for value in episode_ids])
    local_frame_indices = np.asarray(local_frame_indices)
    if values.shape != (physical_predictions.shape[1],) or local_frame_indices.shape != values.shape:
        raise ValueError("episode IDs and local frame indices must match prediction frames")
    if not bool(np.all(action_valid_mask[:, 0])):
        raise ValueError("first action offset must be valid")
    ensemble = physical_predictions.mean(axis=0)
    first_predicted = ensemble[:, 0]
    first_expected = expert_action_chunks[:, 0]
    global_metrics = _first_step_metrics(first_predicted, first_expected)
    global_metrics["baselines"] = baseline_metrics(first_expected, first_expected.mean(axis=0))
    seed_metrics = [_first_step_metrics(seed[:, 0], first_expected) for seed in physical_predictions]
    valid_predicted = ensemble[action_valid_mask]
    valid_expected = expert_action_chunks[action_valid_mask]
    full_chunk = _first_step_metrics(valid_predicted, valid_expected)
    per_episode: dict[str, Any] = {}
    for episode_id in dict.fromkeys(values.tolist()):
        mask = values == episode_id
        predicted = first_predicted[mask]
        expected = first_expected[mask]
        per_episode[episode_id] = {
            "frames": int(mask.sum()),
            "duration_seconds": float(mask.sum() * DT_SECONDS),
            "first_step": _first_step_metrics(predicted, expected),
            "per_seed_first_step": [_first_step_metrics(seed[mask, 0], expected) for seed in physical_predictions],
            "cumulative": _cumulative_controls(predicted, expected),
        }
    return {
        "primary_alignment": "ensemble mean of three fixed-seed first action offsets",
        "global": global_metrics,
        "global_per_seed": seed_metrics,
        "full_valid_chunk": full_chunk,
        "per_episode": per_episode,
        "source_summary": {
            "frames": int(len(values)),
            "valid_full_chunk_pairs": int(action_valid_mask.sum()),
            "episode_ids": list(dict.fromkeys(values.tolist())),
            "local_frame_indices": local_frame_indices.tolist(),
        },
    }


def classify_d2(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = metrics["global"]["per_action"]
    lower, upper = MAGNITUDE_RATIO_LIMITS
    def magnitude(name: str) -> float | None:
        return rows[name]["mean_abs_magnitude_ratio"]
    def std(name: str) -> float | None:
        return rows[name]["std_ratio"]
    vx = magnitude("vx")
    wz = magnitude("wz")
    return {
        "magnitude_ratio_limits": [lower, upper],
        "vx_magnitude_shrinkage": vx is not None and vx < lower,
        "wz_magnitude_shrinkage": wz is not None and wz < lower,
        "vx_magnitude_amplification": vx is not None and vx > upper,
        "wz_magnitude_amplification": wz is not None and wz > upper,
        "vx_variance_shrinkage": std("vx") is not None and std("vx") < lower,
        "wz_variance_shrinkage": std("wz") is not None and std("wz") < lower,
        "vx_variance_amplification": std("vx") is not None and std("vx") > upper,
        "wz_variance_amplification": std("wz") is not None and std("wz") > upper,
    }


def acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = metrics["global"]["per_action"]
    sign = metrics["global"]["wz_sign"]
    baseline = metrics["global"]["baselines"]["majority_wz_sign"]
    lower, upper = MAGNITUDE_RATIO_LIMITS
    checks = {
        "both_training_episodes_covered": set(metrics["per_episode"]) == set(EXPECTED_EPISODE_COUNTS),
        "vx_pearson_gt_0_5": rows["vx"]["correlation_pearson"] is not None and rows["vx"]["correlation_pearson"] > PEARSON_MIN,
        "wz_pearson_gt_0_5": rows["wz"]["correlation_pearson"] is not None and rows["wz"]["correlation_pearson"] > PEARSON_MIN,
        "vx_r2_gt_0": rows["vx"]["r2_vs_target_mean"] is not None and rows["vx"]["r2_vs_target_mean"] > 0.0,
        "wz_r2_gt_0": rows["wz"]["r2_vs_target_mean"] is not None and rows["wz"]["r2_vs_target_mean"] > 0.0,
        "wz_sign_beats_majority": sign["accuracy"] is not None and sign["accuracy"] > baseline["accuracy"],
        "vx_magnitude_ratio_in_range": lower <= rows["vx"]["mean_abs_magnitude_ratio"] <= upper,
        "wz_magnitude_ratio_in_range": lower <= rows["wz"]["mean_abs_magnitude_ratio"] <= upper,
        "vy_exact_zero": rows["vy"]["mae"] == 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _source_hashes(source_evidence: Path) -> dict[str, str]:
    hashes = {name: sha256(source_evidence / name) for name in FROZEN_SOURCE_HASHES}
    mismatches = {name: {"actual": hashes[name], "expected": FROZEN_SOURCE_HASHES[name]} for name in hashes if hashes[name] != FROZEN_SOURCE_HASHES[name]}
    if mismatches:
        raise RuntimeError(f"frozen source evidence hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return hashes


def load_frozen_source(source_evidence: Path, dataset_root: Path, checkpoint: Path, project_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    source_evidence = source_evidence.resolve()
    dataset_root = dataset_root.resolve()
    checkpoint = checkpoint.resolve()
    for path in (source_evidence, dataset_root, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    source_hashes = _source_hashes(source_evidence)
    input_hashes = {
        "model": sha256(checkpoint / "model.safetensors"),
        "codec": sha256(checkpoint / "bounded_action_codec.json"),
        "dataset_manifest": sha256(dataset_root / "conversion_manifest.json"),
    }
    expected_inputs = {"model": FROZEN_MODEL_SHA256, "codec": FROZEN_CODEC_SHA256, "dataset_manifest": FROZEN_MANIFEST_SHA256}
    mismatches = {name: {"actual": input_hashes[name], "expected": expected_inputs[name]} for name in input_hashes if input_hashes[name] != expected_inputs[name]}
    if mismatches:
        raise RuntimeError(f"D0 frozen input hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    evaluator_hashes = {name: sha256(project_root / name) for name in FROZEN_EVALUATOR_HASHES}
    evaluator_mismatches = {name: {"actual": evaluator_hashes[name], "expected": FROZEN_EVALUATOR_HASHES[name]} for name in evaluator_hashes if evaluator_hashes[name] != FROZEN_EVALUATOR_HASHES[name]}
    if evaluator_mismatches:
        raise RuntimeError(f"frozen offline evaluator hash mismatch: {json.dumps(evaluator_mismatches, sort_keys=True)}")
    source_manifest = json.loads((source_evidence / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("checkpoint", {}).get("sha256") != FROZEN_MODEL_SHA256 or source_manifest.get("checkpoint", {}).get("codec_sha256") != FROZEN_CODEC_SHA256:
        raise RuntimeError("source evidence checkpoint provenance mismatch")
    if source_manifest.get("input_allowlist") != INPUT_ALLOWLIST:
        raise RuntimeError("source evidence input allowlist mismatch")
    if tuple(source_manifest.get("base_seeds", [])) != EXPECTED_SEEDS:
        raise RuntimeError("source evidence seed provenance mismatch")
    if source_manifest.get("counts", {}).get("frames") != 847 or source_manifest.get("counts", {}).get("episodes") != 2:
        raise RuntimeError("source evidence frame/episode count mismatch")
    with np.load(source_evidence / "predictions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {"physical_predictions", "latent_predictions", "expert_action_chunks", "action_valid_mask", "episode_ids", "local_frame_indices", "inference_ms", "base_seeds"}
    if set(arrays) != required:
        raise RuntimeError(f"source archive fields mismatch: {sorted(arrays)}")
    validate_prediction_arrays(
        arrays["physical_predictions"],
        arrays["latent_predictions"],
        arrays["expert_action_chunks"],
        arrays["action_valid_mask"],
        arrays["episode_ids"],
        arrays["local_frame_indices"],
        arrays["base_seeds"],
    )
    provenance = {"source_hashes": source_hashes, "input_hashes": input_hashes, "evaluator_hashes": evaluator_hashes}
    return arrays, source_manifest, provenance


def _plot_timeseries(output_dir: Path, physical: np.ndarray, expert: np.ndarray, episode_ids: np.ndarray, frame_indices: np.ndarray) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensemble = physical.mean(axis=0)[:, 0]
    figures: list[str] = []
    for episode_id in EXPECTED_EPISODE_COUNTS:
        mask = episode_ids.astype(str) == episode_id
        seconds = frame_indices[mask] * DT_SECONDS
        figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
        for axis, index, name, unit in zip(axes, range(3), ACTION_NAMES, ("m/s", "m/s", "rad/s"), strict=True):
            axis.plot(seconds, expert[mask, 0, index], color="#1f4e79", linewidth=2.0, label=f"Expert {name}")
            for seed_index in range(physical.shape[0]):
                axis.plot(seconds, physical[seed_index, mask, 0, index], color="#cf6d17", linewidth=0.65, alpha=0.3, linestyle="--", label="Model seed" if seed_index == 0 else None)
            axis.plot(seconds, ensemble[mask, index], color="#cf6d17", linewidth=1.8, linestyle="--", label=f"Model {name} mean")
            axis.set_ylabel(f"{name} ({unit})")
            axis.grid(alpha=0.25)
            axis.legend(loc="best", ncol=2)
        axes[-1].set_xlabel("GT episode time (s)")
        path = output_dir / f"{episode_id}_actions_timeseries.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        figures.append(path.name)
    return figures


def _plot_distributions(output_dir: Path, physical: np.ndarray, expert: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = physical.mean(axis=0)[:, 0]
    target = expert[:, 0]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, index, name in zip(axes, range(3), ACTION_NAMES, strict=True):
        axis.hist(target[:, index], bins=40, density=True, histtype="step", linewidth=1.8, label="Expert")
        axis.hist(predicted[:, index], bins=40, density=True, histtype="step", linestyle="--", linewidth=1.8, label="Model mean")
        axis.set_title(name)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    path = output_dir / "first_step_distributions.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path.name


def _plot_magnitude_variance(output_dir: Path, metrics: dict[str, Any]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = metrics["global"]["per_action"]
    names = ("vx", "wz")
    figure, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    positions = np.arange(len(names))
    width = 0.34
    axis.bar(positions - width / 2, [rows[name]["mean_abs_magnitude_ratio"] for name in names], width, label="mean |action| ratio")
    axis.bar(positions + width / 2, [rows[name]["std_ratio"] for name in names], width, label="std ratio")
    axis.axhspan(*MAGNITUDE_RATIO_LIMITS, color="#77aa77", alpha=0.15, label="no material mismatch")
    axis.axhline(1.0, color="black", linestyle=":")
    axis.set_xticks(positions, names)
    axis.set_ylabel("model / expert")
    axis.set_ylim(0.0, max(1.2, max(rows[name]["mean_abs_magnitude_ratio"] for name in names) + 0.1))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    path = output_dir / "magnitude_variance_ratios.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path.name


def _plot_cumulative(output_dir: Path, metrics: dict[str, Any]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = list(metrics["per_episode"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, suffix, title in zip(axes, ("sum_vx_dt", "sum_abs_wz_dt"), ("Σ vx · dt", "Σ |wz| · dt"), strict=True):
        model = [metrics["per_episode"][episode]["cumulative"][f"model_{suffix}"] for episode in episodes]
        expert = [metrics["per_episode"][episode]["cumulative"][f"expert_{suffix}"] for episode in episodes]
        positions = np.arange(len(episodes))
        axis.bar(positions - 0.18, expert, 0.36, label="Expert")
        axis.bar(positions + 0.18, model, 0.36, label="Model")
        axis.set_xticks(positions, [episode.rsplit("_", 1)[-1] for episode in episodes])
        axis.set_xlabel("training episode")
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
    path = output_dir / "cumulative_commands.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path.name


def _write_first_step_csv(path: Path, physical: np.ndarray, expert: np.ndarray, episode_ids: np.ndarray, frame_indices: np.ndarray) -> None:
    prediction = physical.mean(axis=0)[:, 0]
    rows = ["episode_id,frame_index,time_s,vx_gt,vy_gt,wz_gt,vx_model,vy_model,wz_model"]
    for index, episode_id in enumerate(episode_ids.astype(str)):
        rows.append(
            ",".join(
                [episode_id, str(int(frame_indices[index])), f"{frame_indices[index] * DT_SECONDS:.6f}"]
                + [f"{value:.9g}" for value in expert[index, 0]]
                + [f"{value:.9g}" for value in prediction[index]]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def render_report(metrics: dict[str, Any], verdict: dict[str, Any], classification: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# D2 Offline Expert vs Model Action Analysis",
        "",
        f"Status: **{'PASS' if verdict['passed'] else 'FAIL'}**",
        f"Primary alignment: {metrics['primary_alignment']}; {metrics['source_summary']['frames']} GT observations.",
        "",
        "| Action | MAE | RMSE | Pearson r | R² | |model| / |expert| | std ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ACTION_NAMES:
        row = metrics["global"]["per_action"][name]
        ratio = "N/A" if row["mean_abs_magnitude_ratio"] is None else f"{row['mean_abs_magnitude_ratio']:.6f}"
        std_ratio = "N/A" if row["std_ratio"] is None else f"{row['std_ratio']:.6f}"
        correlation = "N/A" if row["correlation_pearson"] is None else f"{row['correlation_pearson']:.6f}"
        r2 = "N/A" if row["r2_vs_target_mean"] is None else f"{row['r2_vs_target_mean']:.6f}"
        lines.append(f"| {name} | {row['mae']:.6f} | {row['rmse']:.6f} | {correlation} | {r2} | {ratio} | {std_ratio} |")
    sign = metrics["global"]["wz_sign"]
    baseline = metrics["global"]["baselines"]["majority_wz_sign"]
    lines.extend(
        [
            "",
            f"- `wz` sign accuracy at `|GT| >= {sign['deadband']}`: {sign['accuracy']:.6f}; majority baseline: {baseline['accuracy']:.6f}.",
            f"- Magnitude/variance classification: {json.dumps(classification, sort_keys=True)}.",
            "",
            "## Cumulative first-step controls",
            "",
            "| Episode | Model Σvx·dt | Expert Σvx·dt | Model Σ|wz|·dt | Expert Σ|wz|·dt |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for episode_id, row in metrics["per_episode"].items():
        cumulative = row["cumulative"]
        lines.append(
            f"| {episode_id} | {cumulative['model_sum_vx_dt']:.6f} | {cumulative['expert_sum_vx_dt']:.6f} | {cumulative['model_sum_abs_wz_dt']:.6f} | {cumulative['expert_sum_abs_wz_dt']:.6f} |"
        )
    lines.extend(["", f"Frozen source prediction SHA-256: `{manifest['source_evidence']['hashes']['predictions.npz']}`."])
    return "\n".join(lines) + "\n"


def append_d2_diagnosis(path: Path, output_dir: Path, metrics: dict[str, Any], verdict: dict[str, Any], classification: dict[str, Any], manifest: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    if "## D2 — Offline Expert vs Model Action Analysis" in text:
        raise FileExistsError(f"D2 is already recorded in {path}; refusing duplicate diagnosis entry")
    rows = metrics["global"]["per_action"]
    sign = metrics["global"]["wz_sign"]
    baseline = metrics["global"]["baselines"]["majority_wz_sign"]
    result = "The model has meaningful time-aligned training-set action signal without global action-magnitude shrinkage; D3 should isolate closed-loop execution and drift." if verdict["passed"] else "The offline action gate failed; stop before D3 and investigate fitting, normalization, training configuration, or data coverage."
    section = f"""

## D2 — Offline Expert vs Model Action Analysis

**Stage:** D2  
**Status:** {'PASS' if verdict['passed'] else 'FAIL'}  
**Input:** the hash-verified three-seed offline prediction archive; no Isaac, socket, planner, model inference, training, checkpoint, dataset, or codec write.

The primary comparison is the three-seed ensemble mean at the first action offset for all 847 GT training observations. Global `vx` MAE/Pearson/R²/magnitude ratio were `{rows['vx']['mae']:.6f}` / `{rows['vx']['correlation_pearson']:.6f}` / `{rows['vx']['r2_vs_target_mean']:.6f}` / `{rows['vx']['mean_abs_magnitude_ratio']:.6f}`; `wz` values were `{rows['wz']['mae']:.6f}` / `{rows['wz']['correlation_pearson']:.6f}` / `{rows['wz']['r2_vs_target_mean']:.6f}` / `{rows['wz']['mean_abs_magnitude_ratio']:.6f}`. `vy` remains exactly zero in both targets and predictions. `wz` sign accuracy at `|GT|>=0.05` was `{sign['accuracy']:.6f}`, above the majority-sign baseline `{baseline['accuracy']:.6f}`. The global `vx`/`wz` standard-deviation ratios were `{rows['vx']['std_ratio']:.6f}` and `{rows['wz']['std_ratio']:.6f}`; classification flags were `{json.dumps(classification, sort_keys=True)}`.

Artifacts: `{output_dir}`. The frozen prediction archive SHA-256 remained `{manifest['source_evidence']['hashes']['predictions.npz']}` and frozen model/codec/dataset SHA-256 remained `{manifest['input_hashes']['model']}`, `{manifest['input_hashes']['codec']}`, and `{manifest['input_hashes']['dataset_manifest']}`. {result}

**Next recommended stage:** {'D3 — Exact Train-Episode Closed-Loop Test; wait for `CONTINUE D3`.' if verdict['passed'] else 'Stop; repair the failed D2 cause before any closed-loop stage.'}
"""
    path.write_text(text.rstrip() + section + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    diagnosis_report = args.diagnosis_report.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if not diagnosis_report.is_file():
        raise FileNotFoundError(diagnosis_report)
    project_root = Path.cwd()
    arrays, source_manifest, provenance = load_frozen_source(args.source_evidence, args.dataset_root, args.checkpoint, project_root)
    metrics = compute_d2_metrics(
        arrays["physical_predictions"],
        arrays["expert_action_chunks"],
        arrays["action_valid_mask"],
        arrays["episode_ids"],
        arrays["local_frame_indices"],
    )
    classification = classify_d2(metrics)
    verdict = acceptance(metrics)
    manifest = {
        "format": "go2-short-vln-m6-2-d2-offline-analysis-v1",
        "created_at_utc": now_utc(),
        "command": sys.argv,
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "source_evidence": {"path": str(args.source_evidence.expanduser().resolve()), "hashes": provenance["source_hashes"], "manifest_format": source_manifest["format"]},
        "input_hashes": provenance["input_hashes"],
        "evaluator_hashes": provenance["evaluator_hashes"],
        "thresholds": {"pearson_min": PEARSON_MIN, "magnitude_and_std_ratio_limits": list(MAGNITUDE_RATIO_LIMITS), "wz_sign_deadband": WZ_SIGN_DEADBAND, "dt_seconds": DT_SECONDS},
    }
    output_dir.mkdir(parents=True)
    _write_first_step_csv(output_dir / "first_step_ensemble.csv", arrays["physical_predictions"], arrays["expert_action_chunks"], arrays["episode_ids"], arrays["local_frame_indices"])
    figures = _plot_timeseries(output_dir, arrays["physical_predictions"], arrays["expert_action_chunks"], arrays["episode_ids"], arrays["local_frame_indices"])
    figures.append(_plot_distributions(output_dir, arrays["physical_predictions"], arrays["expert_action_chunks"]))
    figures.append(_plot_magnitude_variance(output_dir, metrics))
    figures.append(_plot_cumulative(output_dir, metrics))
    manifest["artifacts"] = {"figures": figures, "first_step_ensemble_csv": "first_step_ensemble.csv"}
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "metrics.json", {"metrics": metrics, "classification": classification, "acceptance": verdict})
    (output_dir / "report.md").write_text(render_report(metrics, verdict, classification, manifest), encoding="utf-8")
    append_d2_diagnosis(diagnosis_report, output_dir, metrics, verdict, classification, manifest)
    print(json.dumps({"output_dir": str(output_dir), "passed": verdict["passed"], "checks": verdict["checks"]}, indent=2))
    if not verdict["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
