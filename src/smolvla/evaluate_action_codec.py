#!/usr/bin/env python3
"""D1: audit the M6.1 bounded-action codec on every stored training action."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.smolvla.bounded_actions import (
    CODEC_FILENAME,
    DEFAULT_EPSILON,
    decode_latent_actions,
    encode_physical_actions,
    load_codec_metadata,
)


ACTION_KEY = "action"
ACTION_NAMES = ("vx", "vy", "wz")
DEFAULT_DATASET_ROOT = Path("<external-data-root>")
DEFAULT_CHECKPOINT = Path(
    "<external-data-root>+0800/checkpoints/step_002000"
)
FROZEN_MODEL_SHA256 = "facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7"
FROZEN_CODEC_SHA256 = "143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27"
FROZEN_MANIFEST_SHA256 = "d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402"
EXPECTED_TRAIN_FRAMES = 847
BOUNDARY_TOLERANCE = 1e-6
INTERIOR_TOLERANCE = 1e-6
DISTRIBUTION_TOLERANCE = 5.1e-5
MAGNITUDE_RATIO_MIN = 0.9995
MAGNITUDE_RATIO_MAX = 1.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("distribution requires a non-empty finite array")
    p5, p50, p95 = np.quantile(values, (0.05, 0.5, 0.95))
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def frozen_inputs(dataset_root: Path, checkpoint: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = dataset_root / "conversion_manifest.json"
    model_path = checkpoint / "model.safetensors"
    codec_path = checkpoint / CODEC_FILENAME
    for path in (manifest_path, model_path, codec_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {
        "dataset_manifest": sha256(manifest_path),
        "model": sha256(model_path),
        "codec": sha256(codec_path),
    }
    expected = {
        "dataset_manifest": FROZEN_MANIFEST_SHA256,
        "model": FROZEN_MODEL_SHA256,
        "codec": FROZEN_CODEC_SHA256,
    }
    mismatches = {name: {"actual": hashes[name], "expected": expected[name]} for name in hashes if hashes[name] != expected[name]}
    if mismatches:
        raise RuntimeError(f"D0 baseline hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("splits", {}).get("train", {}).get("frame_count") != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("D0 baseline train frame count mismatch")
    return manifest, hashes


def load_train_actions(dataset_root: Path, manifest: dict[str, Any], video_backend: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    train = manifest["splits"]["train"]
    dataset = LeRobotDataset(train["repo_id"], root=dataset_root / "train", video_backend=video_backend, return_uint8=True)
    hf = dataset.hf_dataset
    actions = np.stack([_tensor_to_numpy(hf[index][ACTION_KEY]) for index in range(len(hf))]).astype(np.float32)
    episode_indices = np.asarray([int(_tensor_to_numpy(hf[index]["episode_index"])) for index in range(len(hf))], dtype=np.int32)
    frame_indices = np.asarray([int(_tensor_to_numpy(hf[index]["frame_index"])) for index in range(len(hf))], dtype=np.int32)
    if actions.shape != (EXPECTED_TRAIN_FRAMES, 3) or not np.isfinite(actions).all():
        raise ValueError(f"expected finite [{EXPECTED_TRAIN_FRAMES},3] train actions, got {actions.shape}")
    if not np.all(actions[:, 0] >= 0.0) or not np.all(actions[:, 0] <= 0.5):
        raise ValueError("train vx is outside frozen physical bounds")
    if not np.all(actions[:, 1] == 0.0):
        raise ValueError("train vy is not exactly zero")
    if not np.all(actions[:, 2] >= -0.5) or not np.all(actions[:, 2] <= 0.5):
        raise ValueError("train wz is outside frozen physical bounds")
    return actions, episode_indices, frame_indices


def boundary_limits(epsilon: float) -> dict[str, float]:
    return {
        "vx": 0.25 * epsilon + BOUNDARY_TOLERANCE,
        "vy": 0.0,
        "wz": 0.5 * epsilon + BOUNDARY_TOLERANCE,
    }


def _near_bound(values: np.ndarray, bound: float) -> np.ndarray:
    return np.isclose(values, bound, rtol=0.0, atol=1e-7)


def per_dimension_metrics(raw: np.ndarray, decoded: np.ndarray, epsilon: float) -> dict[str, Any]:
    raw = np.asarray(raw, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    if raw.shape != decoded.shape or raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw and decoded must be matching [N,3] arrays")
    errors = decoded - raw
    limits = boundary_limits(epsilon)
    result: dict[str, Any] = {}
    for dim, name in enumerate(ACTION_NAMES):
        raw_values, decoded_values, error = raw[:, dim], decoded[:, dim], errors[:, dim]
        if name == "vx":
            boundary = _near_bound(raw_values, 0.0) | _near_bound(raw_values, 0.5)
        elif name == "wz":
            boundary = _near_bound(raw_values, -0.5) | _near_bound(raw_values, 0.5)
        else:
            boundary = np.zeros(len(raw_values), dtype=bool)
        interior = ~boundary
        raw_abs_mean = float(np.abs(raw_values).mean())
        magnitude_ratio = None if raw_abs_mean == 0.0 else float(np.abs(decoded_values).mean() / raw_abs_mean)
        result[name] = {
            "count": int(len(raw_values)),
            "raw": distribution(raw_values),
            "decoded": distribution(decoded_values),
            "error": {
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "max_abs": float(np.abs(error).max()),
                "signed": distribution(error),
            },
            "mean_abs_magnitude_ratio": magnitude_ratio,
            "boundary_count": int(boundary.sum()),
            "interior_count": int(interior.sum()),
            "boundary_max_abs_error": None if not boundary.any() else float(np.abs(error[boundary]).max()),
            "interior_max_abs_error": None if not interior.any() else float(np.abs(error[interior]).max()),
            "allowed_boundary_max_abs_error": limits[name],
            "distribution_max_abs_drift": max(abs(raw_dist - decoded_dist) for raw_dist, decoded_dist in zip(
                (distribution(raw_values)[key] for key in ("mean", "std", "p5", "p50", "p95", "min", "max")),
                (distribution(decoded_values)[key] for key in ("mean", "std", "p5", "p50", "p95", "min", "max")),
                strict=True,
            )),
        }
    wz_raw, wz_decoded = raw[:, 2], decoded[:, 2]
    nonzero = wz_raw != 0.0
    negative_bound = _near_bound(wz_raw, -0.5)
    positive_bound = _near_bound(wz_raw, 0.5)
    abs_error = np.abs(errors[:, 2])
    result["wz"]["sign"] = {
        "nonzero_count": int(nonzero.sum()),
        "inversion_count": int((np.sign(wz_raw[nonzero]) != np.sign(wz_decoded[nonzero])).sum()),
    }
    result["wz"]["saturation_symmetry"] = {
        "negative_bound_count": int(negative_bound.sum()),
        "positive_bound_count": int(positive_bound.sum()),
        "negative_bound_mean_abs_error": None if not negative_bound.any() else float(abs_error[negative_bound].mean()),
        "positive_bound_mean_abs_error": None if not positive_bound.any() else float(abs_error[positive_bound].mean()),
        "abs_error_difference": None if not (negative_bound.any() and positive_bound.any()) else float(abs(abs_error[negative_bound].mean() - abs_error[positive_bound].mean())),
    }
    return result


def acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for name in ACTION_NAMES:
        row = metrics[name]
        checks[f"{name}_finite"] = math.isfinite(row["error"]["max_abs"])
        checks[f"{name}_distribution_preserved"] = row["distribution_max_abs_drift"] <= DISTRIBUTION_TOLERANCE
        if name == "vy":
            checks["vy_exact_zero"] = row["error"]["max_abs"] == 0.0
        else:
            checks[f"{name}_magnitude_preserved"] = MAGNITUDE_RATIO_MIN <= row["mean_abs_magnitude_ratio"] <= MAGNITUDE_RATIO_MAX
            if row["boundary_max_abs_error"] is not None:
                checks[f"{name}_boundary_error_within_epsilon_contract"] = row["boundary_max_abs_error"] <= row["allowed_boundary_max_abs_error"]
            checks[f"{name}_interior_error_float32_scale"] = row["interior_max_abs_error"] <= INTERIOR_TOLERANCE
    checks["wz_no_sign_inversion"] = metrics["wz"]["sign"]["inversion_count"] == 0
    symmetry = metrics["wz"]["saturation_symmetry"]["abs_error_difference"]
    if symmetry is not None:
        checks["wz_saturation_symmetric"] = symmetry <= BOUNDARY_TOLERANCE
    return {"passed": all(checks.values()), "checks": checks}


def make_dimension_plot(path: Path, name: str, raw: np.ndarray, decoded: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    if name == "vy":
        axes[0].plot(raw, label="raw", linewidth=2)
        axes[0].plot(decoded, label="decoded", linestyle="--")
        axes[0].set_ylim(-0.001, 0.001)
        axes[0].text(0.5, 0.5, "Both raw and decoded vy are exactly zero", ha="center", va="center", transform=axes[0].transAxes)
    else:
        lower = min(float(raw.min()), float(decoded.min()))
        upper = max(float(raw.max()), float(decoded.max()))
        axes[0].scatter(raw, decoded, s=11, alpha=0.6)
        axes[0].plot((lower, upper), (lower, upper), color="black", linestyle=":", label="identity")
        axes[0].set_xlabel(f"raw {name}")
        axes[0].set_ylabel(f"decoded {name}")
        axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].hist(raw, bins=40, density=True, histtype="step", linewidth=1.8, label="raw")
    axes[1].hist(decoded, bins=40, density=True, histtype="step", linestyle="--", linewidth=1.8, label="decoded")
    axes[1].set_xlabel(name)
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_error_plot(path: Path, errors: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes, ACTION_NAMES, strict=True)):
        axis.hist(errors[:, index], bins=40, histtype="stepfilled", alpha=0.65)
        axis.axvline(0.0, color="black", linestyle=":")
        axis.set_title(f"{name} decoded - raw")
        axis.set_xlabel("error")
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_report(manifest: dict[str, Any], metrics: dict[str, Any], verdict: dict[str, Any]) -> str:
    lines = [
        "# D1 Action Codec Round-Trip",
        "",
        f"Status: **{'PASS' if verdict['passed'] else 'FAIL'}**",
        f"Input: {manifest['counts']['frames']} raw train actions; codec epsilon `{manifest['codec']['epsilon']}`.",
        "",
        "| Dim | MAE | RMSE | Max abs | Magnitude ratio | Distribution max drift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ACTION_NAMES:
        row = metrics[name]
        ratio = "N/A" if row["mean_abs_magnitude_ratio"] is None else f"{row['mean_abs_magnitude_ratio']:.9f}"
        lines.append(
            f"| {name} | {row['error']['mae']:.9g} | {row['error']['rmse']:.9g} | {row['error']['max_abs']:.9g} | {ratio} | {row['distribution_max_abs_drift']:.9g} |"
        )
    sign = metrics["wz"]["sign"]
    symmetry = metrics["wz"]["saturation_symmetry"]
    lines.extend(
        [
            "",
            f"- `wz` sign inversions: {sign['inversion_count']} / {sign['nonzero_count']}",
            f"- `wz` positive/negative boundary absolute-error difference: {symmetry['abs_error_difference']}",
            "",
            "Exact physical bounds are intentionally reconstructed just inside the bounds because epsilon keeps `atanh` finite. The permitted errors are `0.25*epsilon + 1e-6` for `vx` and `0.5*epsilon + 1e-6` for `wz`; all non-boundary values must round-trip at float32 scale.",
        ]
    )
    return "\n".join(lines) + "\n"


def append_d1_diagnosis(path: Path, output_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any], verdict: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    if "## D1 — Action Codec Round-Trip Test" in text:
        raise FileExistsError(f"D1 is already recorded in {path}; refusing duplicate diagnosis entry")
    vx, vy, wz = (metrics[name] for name in ACTION_NAMES)
    conclusion = (
        "The codec preserves the training action distribution within the configured epsilon boundary contract; it is not the source of the M7 navigation failure."
        if verdict["passed"]
        else "The codec violates its round-trip acceptance contract; stop and repair the codec before D2."
    )
    section = f"""

## D1 — Action Codec Round-Trip Test

**Stage:** D1  
**Status:** {'PASS' if verdict['passed'] else 'FAIL'}  
**Input:** all {manifest['counts']['frames']} frozen train actions; no model, Isaac, GPU, or checkpoint write.

The production `float32` codec executed `raw -> encode -> decode` with epsilon `{manifest['codec']['epsilon']}`. `vx` MAE/RMSE/max-absolute-error were `{vx['error']['mae']:.9g}` / `{vx['error']['rmse']:.9g}` / `{vx['error']['max_abs']:.9g}`; `wz` values were `{wz['error']['mae']:.9g}` / `{wz['error']['rmse']:.9g}` / `{wz['error']['max_abs']:.9g}`. `vy` max error was `{vy['error']['max_abs']:.9g}`. The `vx`/`wz` absolute-magnitude ratios were `{vx['mean_abs_magnitude_ratio']:.9f}` and `{wz['mean_abs_magnitude_ratio']:.9f}`; `wz` sign inversions were `{wz['sign']['inversion_count']}/{wz['sign']['nonzero_count']}`. Exact physical boundary actions are intentionally reconstructed inward by the finite-`atanh` epsilon contract, with permitted maxima `vx={vx['allowed_boundary_max_abs_error']:.9g}` and `wz={wz['allowed_boundary_max_abs_error']:.9g}`.

Artifacts: `{output_dir}`; model/codec/dataset hashes remained `{manifest['input_hashes']['model']}`, `{manifest['input_hashes']['codec']}`, and `{manifest['input_hashes']['dataset_manifest']}`. {conclusion}

**Next recommended stage:** {'D2 — Offline Expert vs Model Action Analysis; wait for `CONTINUE D2`.' if verdict['passed'] else 'Stop; repair codec and rerun D1.'}
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
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_source, input_hashes = frozen_inputs(dataset_root, checkpoint)
    codec = load_codec_metadata(checkpoint)
    if codec["epsilon"] != DEFAULT_EPSILON:
        raise RuntimeError("D0 baseline codec epsilon mismatch")
    output_dir.mkdir(parents=True)
    raw, episode_indices, frame_indices = load_train_actions(dataset_root, manifest_source, args.video_backend)
    with torch.no_grad():
        latent = encode_physical_actions(torch.from_numpy(raw), epsilon=codec["epsilon"]).cpu().numpy()
        decoded = decode_latent_actions(torch.from_numpy(latent)).cpu().numpy()
    if not np.isfinite(latent).all() or not np.isfinite(decoded).all():
        raise RuntimeError("codec emitted non-finite values")
    errors = decoded - raw
    metrics = per_dimension_metrics(raw, decoded, codec["epsilon"])
    verdict = acceptance(metrics)
    manifest = {
        "format": "go2-short-vln-m6_2-d1-codec-v1",
        "created_at_utc": now_utc(),
        "command": sys.argv,
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "codec": codec,
        "input_hashes": input_hashes,
        "counts": {"frames": int(len(raw)), "episodes": int(len(np.unique(episode_indices)))},
        "thresholds": {
            "interior_max_abs_error": INTERIOR_TOLERANCE,
            "distribution_max_abs_drift": DISTRIBUTION_TOLERANCE,
            "magnitude_ratio": [MAGNITUDE_RATIO_MIN, MAGNITUDE_RATIO_MAX],
            "boundary_limits": boundary_limits(codec["epsilon"]),
        },
    }
    np.savez_compressed(
        output_dir / "roundtrip_actions.npz",
        raw_actions=raw,
        latent_actions=latent,
        decoded_actions=decoded,
        signed_errors=errors,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
    )
    for index, name in enumerate(ACTION_NAMES):
        make_dimension_plot(output_dir / f"{name}_raw_vs_decoded.png", name, raw[:, index], decoded[:, index])
    make_error_plot(output_dir / "action_error_hist.png", errors)
    write_json(output_dir / "metrics.json", {"per_dimension": metrics, "acceptance": verdict})
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "report.md").write_text(render_report(manifest, metrics, verdict), encoding="utf-8")
    append_d1_diagnosis(diagnosis_report, output_dir, manifest, metrics, verdict)
    print(json.dumps({"output_dir": str(output_dir), "passed": verdict["passed"], "checks": verdict["checks"]}, indent=2))
    if not verdict["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
