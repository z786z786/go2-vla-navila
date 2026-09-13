#!/usr/bin/env python3
"""D4: audit the frozen M4 -> M5 -> SmolVLA temporal data path offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.inference.state import build_policy_state_from_m4_record


FPS = 50
DT_S = 1.0 / FPS
EPISODES = ("short_vln_v1_0000", "short_vln_v1_0004")
LAGS = tuple(range(-5, 6))
TURN_WZ_THRESHOLD = 0.1
MODEL_SHA256 = "facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7"
CODEC_SHA256 = "143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27"
DATASET_SHA256 = "d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402"
D2_PREDICTIONS_SHA256 = "834cb049f2dda4bd74498ebd68129f3c3c160cbec98b45f091d9ade775d0292e"
MIN_LAG_CORRELATION_GAIN = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    if len(quaternion) != 4:
        raise ValueError("WXYZ quaternion must contain four entries")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pearson(source: np.ndarray, target: np.ndarray) -> float | None:
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 1 or len(source) < 3:
        return None
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("lag inputs must be finite")
    if float(source.std()) < 1e-12 or float(target.std()) < 1e-12:
        return None
    return float(np.corrcoef(source, target)[0, 1])


def lag_curve(source: Sequence[float], target: Sequence[float], lags: Sequence[int] = LAGS) -> dict[int, float | None]:
    """Correlation source[t] vs target[t + lag]; positive lag means source leads."""
    source_array, target_array = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source_array.shape != target_array.shape or source_array.ndim != 1:
        raise ValueError("lag series must be equal-length vectors")
    values: dict[int, float | None] = {}
    for lag in lags:
        if lag >= 0:
            left, right = source_array[: len(source_array) - lag], target_array[lag:]
        else:
            left, right = source_array[-lag:], target_array[: len(target_array) + lag]
        values[int(lag)] = pearson(left, right)
    return values


def best_lag(curve: dict[int, float | None]) -> tuple[int | None, float | None]:
    valid = [(lag, value) for lag, value in curve.items() if value is not None]
    return max(valid, key=lambda item: item[1]) if valid else (None, None)


def clear_nonzero_peak(curve: dict[int, float | None], *, threshold: float = MIN_LAG_CORRELATION_GAIN) -> dict[str, Any]:
    peak_lag, peak_value = best_lag(curve)
    zero_value = curve.get(0)
    gain = None if peak_value is None or zero_value is None else peak_value - zero_value
    return {
        "best_lag": peak_lag,
        "best_correlation": peak_value,
        "zero_lag_correlation": zero_value,
        "gain_vs_zero": gain,
        "clear_nonzero_peak": bool(
            peak_lag is not None and peak_lag != 0 and gain is not None and gain >= threshold
        ),
    }


def signal_difference(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 4:
        raise ValueError("need at least four samples for first-difference lag analysis")
    return np.diff(array)


def source_action(record: dict[str, Any]) -> np.ndarray:
    return np.asarray([record["expert_vx"], record["expert_vy"], record["expert_wz"]], dtype=np.float32)


def image_fingerprint_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    array = np.asarray(image)
    if array.shape != (512, 512, 3):
        raise ValueError(f"expected RGB image shape [512,512,3], got {array.shape}")
    gray = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def image_lag_error(source: Sequence[np.ndarray], target: Sequence[np.ndarray]) -> dict[int, float]:
    if len(source) != len(target) or not source:
        raise ValueError("image sequences must be non-empty and equal length")
    result: dict[int, float] = {}
    for lag in LAGS:
        if lag >= 0:
            left, right = source[: len(source) - lag], target[lag:]
        else:
            left, right = source[-lag:], target[: len(target) + lag]
        result[lag] = float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(left, right)]))
    return result


def zero_lag_image_pass(errors: dict[int, float]) -> dict[str, Any]:
    best = min(errors, key=errors.get)
    minimum, zero = errors[best], errors[0]
    return {
        "best_lag": best,
        "best_error": minimum,
        "zero_lag_error": zero,
        "zero_within_one_percent_of_best": zero <= minimum * 1.01 + 1e-12,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str, *, atol: float = 1e-6) -> None:
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=atol):
        difference = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.shape == expected.shape else float("inf")
        raise ValueError(f"{label} mismatch shape={actual.shape}/{expected.shape} max_abs={difference}")


def _load_train_datasets(dataset_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features

    manifest = json.loads((dataset_root / "conversion_manifest.json").read_text(encoding="utf-8"))
    info = manifest["splits"]["train"]
    base = LeRobotDataset(info["repo_id"], root=dataset_root / "train", video_backend="pyav", return_uint8=True)
    policy_features = dataset_to_policy_features(base.meta.features)
    config = SmolVLAConfig(
        input_features={key: value for key, value in policy_features.items() if key.startswith("observation")},
        output_features={"action": policy_features["action"]},
        device="cpu", chunk_size=50, n_action_steps=50,
    )
    delta = LeRobotDataset(
        info["repo_id"], root=dataset_root / "train", video_backend="pyav", return_uint8=True,
        delta_timestamps=resolve_delta_timestamps(config, base.meta),
    )
    return base, delta, manifest


def audit_mapping(m4_root: Path, dataset_root: Path, d2_predictions: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base, delta, manifest = _load_train_datasets(dataset_root)
    if len(base) != 847 or len(delta) != 847:
        raise ValueError("frozen train loader must contain 847 anchors")
    episodes = manifest["splits"]["train"]["episodes"]
    if [row["source_short_episode_id"] for row in episodes] != list(EPISODES):
        raise ValueError("train manifest episode order differs from frozen D4 scope")
    source_by_episode = {episode: load_jsonl(m4_root / episode / "steps.jsonl") for episode in EPISODES}
    for row in episodes:
        episode_id = row["source_short_episode_id"]
        if sha256(m4_root / episode_id / "steps.jsonl") != row["steps_sha256"]:
            raise ValueError(f"M4 source steps hash mismatch for {episode_id}")
        if len(source_by_episode[episode_id]) != int(row["frame_count"]):
            raise ValueError(f"M4 source frame count mismatch for {episode_id}")

    image_source: dict[str, list[np.ndarray]] = defaultdict(list)
    image_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    state_equal = action_equal = timestamps_equal = index_equal = chunk_equal = padding_equal = d2_chunk_equal = True
    per_episode: dict[str, list[dict[str, Any]]] = {episode: [] for episode in EPISODES}
    global_index = 0
    d2 = np.load(d2_predictions, allow_pickle=False)
    required = {"expert_action_chunks", "action_valid_mask", "episode_ids", "local_frame_indices"}
    if not required <= set(d2.files):
        raise ValueError("D2 archive is missing temporal audit arrays")
    for manifest_episode in episodes:
        episode_id = manifest_episode["source_short_episode_id"]
        records = source_by_episode[episode_id]
        for local_index, record in enumerate(records):
            raw = base.get_raw_item(global_index)
            sample = base[global_index]
            expected_state = np.asarray(build_policy_state_from_m4_record(record), dtype=np.float32)
            expected_action = source_action(record)
            state = _as_numpy(raw["observation.state"]).astype(np.float32)
            action = _as_numpy(raw["action"]).astype(np.float32)
            state_equal &= state.shape == (30,) and np.allclose(state, expected_state, atol=1e-6, rtol=0.0)
            action_equal &= action.shape == (3,) and np.allclose(action, expected_action, atol=1e-6, rtol=0.0)
            timestamps_equal &= math.isclose(float(_as_numpy(raw["timestamp"])), local_index / FPS, abs_tol=1e-6)
            index_equal &= int(_as_numpy(raw["frame_index"])) == local_index and int(_as_numpy(raw["episode_index"])) == int(manifest_episode["lerobot_episode_index"])
            source_path = m4_root / episode_id / record["front_rgb"]
            import cv2
            source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if source_bgr is None:
                raise ValueError(f"unreadable source RGB {source_path}")
            image_source[episode_id].append(image_fingerprint_rgb(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)))
            decoded = _as_numpy(sample["observation.images.front"])
            image_dataset[episode_id].append(image_fingerprint_rgb(np.transpose(decoded, (1, 2, 0))))
            delta_sample = delta[global_index]
            chunk = _as_numpy(delta_sample["action"]).astype(np.float32)
            padding = _as_numpy(delta_sample["action_is_pad"]).astype(bool)
            expected_chunk = np.stack([source_action(records[min(local_index + offset, len(records) - 1)]) for offset in range(50)])
            expected_padding = np.asarray([local_index + offset >= len(records) for offset in range(50)], dtype=bool)
            chunk_equal &= chunk.shape == (50, 3) and np.allclose(chunk, expected_chunk, atol=1e-6, rtol=0.0)
            padding_equal &= padding.shape == (50,) and np.array_equal(padding, expected_padding)
            d2_chunk_equal &= (
                str(d2["episode_ids"][global_index]) == episode_id
                and int(d2["local_frame_indices"][global_index]) == local_index
                and np.allclose(d2["expert_action_chunks"][global_index], chunk, atol=1e-6, rtol=0.0)
                and np.array_equal(d2["action_valid_mask"][global_index], ~padding)
            )
            per_episode[episode_id].append(record)
            global_index += 1
    image_checks = {}
    for episode in EPISODES:
        errors = image_lag_error(image_source[episode], image_dataset[episode])
        image_checks[episode] = {
            **zero_lag_image_pass(errors),
            "lag_errors": {str(lag): value for lag, value in errors.items()},
        }
    checks = {
        "anchor_count_847": global_index == 847,
        "m4_to_m5_state_exact": state_equal,
        "m4_to_m5_action_exact": action_equal,
        "m4_to_m5_timestamp_exact": timestamps_equal,
        "m4_to_m5_indices_exact": index_equal,
        "loader_action_chunk_t_to_t_plus_49_exact": chunk_equal,
        "loader_padding_mask_exact": padding_equal,
        "d2_gt_chunk_and_mask_exact": d2_chunk_equal,
        "image_zero_lag_alignment": all(item["zero_within_one_percent_of_best"] for item in image_checks.values()),
    }
    return {"checks": checks, "images": image_checks, "manifest": manifest}, per_episode, {key: d2[key] for key in d2.files}


def _episode_lag_metrics(predictions: np.ndarray, targets: np.ndarray, turn_mask: np.ndarray) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for dimension, name in ((0, "vx"), (2, "wz")):
        entries: dict[str, Any] = {}
        for subset, mask in (("all", np.ones(len(targets), dtype=bool)), ("turn", turn_mask)):
            source, target = predictions[mask, dimension], targets[mask, dimension]
            raw, diff = lag_curve(source, target), lag_curve(signal_difference(source), signal_difference(target))
            entries[subset] = {"raw": {str(k): v for k, v in raw.items()}, "raw_peak": clear_nonzero_peak(raw), "difference": {str(k): v for k, v in diff.items()}, "difference_peak": clear_nonzero_peak(diff)}
        results[name] = entries
    return results


def _consistent_shift(metrics: dict[str, Any]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for episode_id in EPISODES:
        per_seed = metrics[episode_id]["per_seed"]
        for seed_index, item in enumerate(per_seed):
            raw = item["wz"]["all"]["raw_peak"]
            diff = item["wz"]["all"]["difference_peak"]
            sign_match = raw["best_lag"] is not None and diff["best_lag"] is not None and raw["best_lag"] * diff["best_lag"] > 0
            if raw["clear_nonzero_peak"] and diff["clear_nonzero_peak"] and sign_match:
                evidence.append({"episode_id": episode_id, "seed_index": seed_index, "lag_sign": 1 if raw["best_lag"] > 0 else -1})
    signs = {row["lag_sign"] for row in evidence}
    per_episode_counts = {episode: sum(row["episode_id"] == episode for row in evidence) for episode in EPISODES}
    detected = bool(len(signs) == 1 and all(count >= 2 for count in per_episode_counts.values()))
    return {"detected": detected, "evidence": evidence, "per_episode_counts": per_episode_counts}


def model_lag_metrics(d2: dict[str, Any]) -> dict[str, Any]:
    physical = np.asarray(d2["physical_predictions"], dtype=np.float64)
    targets = np.asarray(d2["expert_action_chunks"], dtype=np.float64)[:, 0]
    ids = np.asarray(d2["episode_ids"]).astype(str)
    if physical.shape != (3, 847, 50, 3):
        raise ValueError(f"unexpected D2 physical prediction shape {physical.shape}")
    result: dict[str, Any] = {}
    for episode_id in EPISODES:
        mask = ids == episode_id
        turn_mask = np.abs(targets[mask, 2]) >= TURN_WZ_THRESHOLD
        seed_metrics = [_episode_lag_metrics(physical[index, mask, 0], targets[mask], turn_mask) for index in range(3)]
        result[episode_id] = {"frames": int(mask.sum()), "turn_frames": int(turn_mask.sum()), "per_seed": seed_metrics, "ensemble": _episode_lag_metrics(physical[:, mask, 0].mean(axis=0), targets[mask], turn_mask)}
    return {"per_episode": result, "clear_model_wz_shift": _consistent_shift(result)}


def heading_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    command = np.asarray([float(record["expert_wz"]) for record in records], dtype=np.float64)
    yaw = np.asarray([yaw_from_wxyz(record["robot_pose"]["quaternion_wxyz"]) for record in records], dtype=np.float64)
    next_yaw = np.asarray([yaw_from_wxyz(record["next_robot_pose"]["quaternion_wxyz"]) for record in records], dtype=np.float64)
    goal = np.asarray(records[0]["goal_pose"]["position"], dtype=np.float64)
    positions = np.asarray([record["robot_pose"]["position_w"] for record in records], dtype=np.float64)
    bearing = np.arctan2(goal[1] - positions[:, 1], goal[0] - positions[:, 0])
    bearing_error = np.asarray([wrap_angle(value) for value in bearing - yaw])
    yaw_change = np.asarray([wrap_angle(after - before) / DT_S for before, after in zip(yaw, next_yaw)])
    bearing_curve, response_curve = lag_curve(bearing_error, command), lag_curve(command, yaw_change)
    response_peak = clear_nonzero_peak(response_curve)
    return {
        "bearing_error_vs_expert_wz": {str(k): v for k, v in bearing_curve.items()},
        "expert_wz_vs_realized_yaw_change": {str(k): v for k, v in response_curve.items()},
        "response_peak": response_peak,
        "causal_response_lag_0_to_5": response_peak["best_lag"] is not None and 0 <= response_peak["best_lag"] <= 5,
    }


def _plot(output: Path, model: dict[str, Any], heading: dict[str, Any], mapping: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for episode_id in EPISODES:
        ensemble = model["per_episode"][episode_id]["ensemble"]
        axes[0, 0].plot(LAGS, [ensemble["wz"]["all"]["raw"][str(lag)] for lag in LAGS], marker="o", label=episode_id)
        axes[0, 1].plot(LAGS, [ensemble["wz"]["all"]["difference"][str(lag)] for lag in LAGS], marker="o", label=episode_id)
        axes[1, 0].plot(LAGS, [heading[episode_id]["expert_wz_vs_realized_yaw_change"][str(lag)] for lag in LAGS], marker="o", label=episode_id)
        image = mapping["images"][episode_id]
        axes[1, 1].plot(LAGS, [image_lag_error_placeholder(image, lag) for lag in LAGS], marker="o", label=episode_id)
    labels = ("Model wz vs expert wz (raw)", "Model wz vs expert wz (first difference)", "Expert wz vs realized yaw change", "RGB fingerprint error")
    for axis, title in zip(axes.ravel(), labels, strict=True):
        axis.axvline(0, color="black", linewidth=1, alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("lag (frames; positive = source leads)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[1, 1].set_ylabel("mean absolute fingerprint error")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def image_lag_error_placeholder(image_result: dict[str, Any], lag: int) -> float:
    # The curve is injected by audit_mapping below for plotting while the public report stays compact.
    return float(image_result["lag_errors"][str(lag)])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    checks = report["mapping"]["checks"]
    rows = []
    for episode_id in EPISODES:
        ensemble = report["model_lag"]["per_episode"][episode_id]["ensemble"]["wz"]["all"]
        image = report["mapping"]["images"][episode_id]
        response = report["heading"][episode_id]["response_peak"]
        rows.append(f"| {episode_id} | {image['best_lag']} | {ensemble['raw_peak']['best_lag']} | {ensemble['raw_peak']['gain_vs_zero']:.4f} | {ensemble['difference_peak']['best_lag']} | {response['best_lag']} |")
    return (
        "# D4 Observation / Action Temporal Alignment\n\n"
        "- M4 capture: rotated RGB, pre-step state and planner command are recorded at control tick `t`; command executes in that tick.\n"
        "- M5 storage: 50 Hz, frame stride 1; source frame `t` maps directly to LeRobot frame `t`.\n"
        "- SmolVLA target: observation offsets `[0]`, action offsets `[0..49]`, hence `action_t...action_t+49`; end padding is masked from loss.\n"
        "- Closed-loop execution remains 50 Hz control, 5 Hz replanning, first 10 predicted actions executed.\n\n"
        "| Episode | RGB best lag | Model wz raw best lag | Raw gain vs 0 | Model wz diff best lag | Command→yaw best lag |\n|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\nHard mapping checks: **{}**. D4 verdict: **{}**.\n".format("PASS" if all(checks.values()) else "FAIL", report["verdict"])
    )


def append_diagnosis(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8")
    if "## D4 — Observation / Action Temporal Alignment" in existing:
        raise FileExistsError(f"D4 already recorded in {path}")
    result = "No hard M4→M5→loader mapping mismatch or consistent non-zero model lag was found." if report["verdict"] == "PASS" else "Temporal alignment evidence failed; stop before D5 and repair the identified layer."
    text = (
        "\n## D4 — Observation / Action Temporal Alignment\n\n"
        f"**Stage:** D4  \n**Status:** {report['verdict']}  \n**Evidence root:** `{report['output_dir']}`\n\n"
        f"{result} M4/M5 source hashes, every one of 847 anchors, 50-step chunks, padding masks, RGB fingerprint lags and D2 prediction lags are retained in the D4 artifact report.\n\n"
        "**Stop here.** Wait for explicit direction before D5.\n"
    )
    path.write_text(existing.rstrip() + "\n" + text, encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    input_hashes = {
        "model": sha256(args.checkpoint / "model.safetensors"),
        "codec": sha256(args.checkpoint / "bounded_action_codec.json"),
        "dataset_manifest": sha256(args.dataset_root / "conversion_manifest.json"),
        "d2_predictions": sha256(args.d2_predictions),
    }
    expected = {"model": MODEL_SHA256, "codec": CODEC_SHA256, "dataset_manifest": DATASET_SHA256, "d2_predictions": D2_PREDICTIONS_SHA256}
    mismatches = {key: {"expected": expected[key], "actual": input_hashes[key]} for key in expected if input_hashes[key] != expected[key]}
    if mismatches:
        raise RuntimeError(f"frozen D4 input hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    mapping, records, d2 = audit_mapping(args.m4_root, args.dataset_root, args.d2_predictions)
    model = model_lag_metrics(d2)
    heading = {episode: heading_metrics(records[episode]) for episode in EPISODES}
    mapping_passed = all(mapping["checks"].values())
    model_shift = bool(model["clear_model_wz_shift"]["detected"])
    heading_causal = all(heading[episode]["causal_response_lag_0_to_5"] for episode in EPISODES)
    verdict = "PASS" if mapping_passed and not model_shift and heading_causal else "FAIL"
    report = {"format": "go2-short-vln-m6-2-d4-temporal-alignment-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(args.output_dir), "input_hashes": input_hashes, "mapping": mapping, "model_lag": model, "heading": heading, "verdict": verdict, "checks": {"hard_mapping_passed": mapping_passed, "no_consistent_model_wz_shift": not model_shift, "command_to_yaw_causal": heading_causal}}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--d2-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for attribute in ("m4_root", "dataset_root", "d2_predictions", "checkpoint", "output_dir", "diagnosis_report"):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    report = build_report(args)
    for episode in EPISODES:
        if "lag_errors" not in report["mapping"]["images"][episode]:
            raise RuntimeError("image lag curves were not retained")
    plot = args.output_dir / "lag_correlation.png"
    _plot(plot, report["model_lag"], report["heading"], report["mapping"])
    report["artifacts"] = {"lag_correlation": str(plot)}
    write_json(args.output_dir / "d4_report.json", report)
    (args.output_dir / "d4_report.md").write_text(render_markdown(report), encoding="utf-8")
    append_diagnosis(args.diagnosis_report, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
