from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from collectors.common.llada_vla_common import (
    Sample,
    apply_auto_frame_filters,
    assign_splits,
    discover_session_roots,
    dump_json,
    dump_jsonl,
    ensure_session_materialized,
    episode_entries,
    load_quality_review_index,
    load_session_samples,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_image_path(sample: Sample, session_root: Path) -> str | None:
    if sample.source_image_path is None:
        return None
    source = Path(sample.source_image_path)
    try:
        relative = source.relative_to(session_root / "images")
    except ValueError:
        return None
    return str(Path(sample.session_id) / relative)


def _materialize_image_tree(output_root: Path, session_root: Path, session_id: str) -> Path:
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    target = images_dir / session_id
    if target.exists() or target.is_symlink():
        return target
    source = (session_root / "images").resolve()
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        import shutil

        shutil.copytree(source, target)
    return target


def _normalize_state_for_training(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(state or {})
    for key in ("vx_prev", "vy_prev", "wz_prev", "vx_previous", "vy_previous", "wz_previous"):
        normalized.pop(key, None)
    if normalized.get("yaw_speed") is None and normalized.get("wz") is not None:
        normalized["yaw_speed"] = normalized.get("wz")
    return normalized


def _inject_previous_action_fields(record: Dict[str, Any]) -> None:
    previous_action = record.get("previous_action")
    if not isinstance(previous_action, dict):
        previous_action = {}
    record["state"] = _normalize_state_for_training(dict(record.get("state") or {}))
    record["previous_action"] = {
        "vx": _safe_float(previous_action.get("vx")),
        "vy": _safe_float(previous_action.get("vy")),
        "wz": _safe_float(previous_action.get("wz")),
    }
    record["prev_action_valid"] = bool(record.get("prev_action_valid", False))
    record["prev_action_source"] = str(record.get("prev_action_source") or "warmup_zero")
    record["prev_action_reason"] = str(record.get("prev_action_reason") or record["prev_action_source"])


def sample_to_processed_record(
    sample: Sample,
    *,
    session_root: Path,
    output_root: Path,
    horizon_k: int,
    source_dataset: str,
) -> Dict[str, Any]:
    record = sample.to_manifest_record(output_root)
    _inject_previous_action_fields(record)
    record["source_dataset"] = source_dataset
    record["source_type"] = source_dataset
    record["split"] = ""
    record["action_mask"] = [1] * min(len(sample.action_chunk), horizon_k) + [0] * max(0, horizon_k - len(sample.action_chunk))
    image_path = _build_image_path(sample, session_root)
    if image_path:
        record["image_path"] = image_path
        record["image"] = image_path
    return record


def collect_samples_from_root(
    raw_root: Path,
    *,
    source_dataset: str,
    output_root: Path,
    horizon_k: int,
    review_path: Path | None = None,
    drop_initial_frames: int = 0,
    max_success_tail_frames: int | None = None,
) -> List[Dict[str, Any]]:
    session_roots = discover_session_roots(raw_root)
    review_index = load_quality_review_index(raw_root, review_path)
    rows: List[Dict[str, Any]] = []
    for session_root in session_roots:
        session_id, _entries = episode_entries(session_root)
        ensure_session_materialized(output_root, session_root, session_id)
        _materialize_image_tree(output_root, session_root, session_id)
        samples = load_session_samples(
            session_root,
            action_horizon=horizon_k,
            min_trajectory_length=1,
            quality_review_index=review_index,
            include_unreviewed=True,
            drop_initial_frames=drop_initial_frames,
            max_success_tail_frames=max_success_tail_frames,
        )
        for sample in samples:
            rows.append(
                sample_to_processed_record(
                    sample,
                    session_root=session_root,
                    output_root=output_root,
                    horizon_k=horizon_k,
                    source_dataset=source_dataset,
                )
            )
    return rows


def build_processed_dataset(
    *,
    real_root: Path,
    sim_root: Path,
    output_root: Path,
    horizon_k: int,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int | None,
    drop_initial_frames_real: int = 0,
    drop_initial_frames_sim: int = 0,
    max_success_tail_frames_real: int | None = None,
    max_success_tail_frames_sim: int | None = None,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = collect_samples_from_root(
        real_root,
        source_dataset="real",
        output_root=output_root,
        horizon_k=horizon_k,
        drop_initial_frames=drop_initial_frames_real,
        max_success_tail_frames=max_success_tail_frames_real,
    )
    rows.extend(
        collect_samples_from_root(
            sim_root,
            source_dataset="sim",
            output_root=output_root,
            horizon_k=horizon_k,
            drop_initial_frames=drop_initial_frames_sim,
            max_success_tail_frames=max_success_tail_frames_sim,
        )
    )

    row_to_sample = {
        row["sample_id"]: Sample(
            session_id=str(row["session_id"]),
            trajectory_id=str(row["trajectory_id"]),
            trajectory_index=int(row.get("trajectory_index") or 0),
            trajectory_step_index=int(row.get("trajectory_step_index") or 0),
            trajectory_length=int(row.get("trajectory_length") or 0),
            step_id=int(row.get("step_id") or 0),
            timestamp=_safe_float(row.get("timestamp")),
            instruction=str(row.get("instruction") or ""),
            image_path=str(row.get("image_path") or ""),
            source_image_path=str(row.get("source_image_path") or ""),
            state=dict(row.get("state") or {}),
            raw_action=dict(row.get("raw_action") or {}),
            control_action=dict(row.get("control_action") or {}),
            action_chunk=list(row.get("action_chunk") or []),
            raw_record=dict(row),
        )
        for row in rows
    }
    split_samples = assign_splits(
        list(row_to_sample.values()),
        split_mode=split_mode,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
    )

    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name, samples in split_samples.items():
        for sample in samples:
            row = dict(sample.raw_record)
            row["split"] = split_name
            split_rows[split_name].append(row)

    all_rows = split_rows["train"] + split_rows["val"] + split_rows["test"]
    dump_jsonl(output_root / "train.jsonl", split_rows["train"])
    dump_jsonl(output_root / "val.jsonl", split_rows["val"])
    dump_jsonl(output_root / "test.jsonl", split_rows["test"])
    dump_jsonl(output_root / "dataset.jsonl", all_rows)

    source_counts = Counter(str(row.get("source_dataset") or "unknown") for row in all_rows)
    split_counts = {name: len(split_rows[name]) for name in ("train", "val", "test")}
    split_source_counts = {
        split_name: dict(Counter(str(row.get("source_dataset") or "unknown") for row in rows_for_split))
        for split_name, rows_for_split in split_rows.items()
    }
    stats = {
        "sample_count": len(all_rows),
        "split_counts": split_counts,
        "split_source_counts": split_source_counts,
        "source_counts": dict(source_counts),
        "horizon_k": horizon_k,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "real_root": str(real_root),
        "sim_root": str(sim_root),
        "output_root": str(output_root),
        "drop_initial_frames_real": int(drop_initial_frames_real),
        "drop_initial_frames_sim": int(drop_initial_frames_sim),
        "max_success_tail_frames_real": max_success_tail_frames_real,
        "max_success_tail_frames_sim": max_success_tail_frames_sim,
    }
    dump_json(output_root / "stats.json", stats)
    return stats


def validate_processed_dataset(
    *,
    data_paths: Sequence[Path],
    image_root: Path,
    horizon_k: int,
) -> Dict[str, Any]:
    issues: List[str] = []
    counters = Counter()
    for path in data_paths:
        if not path.exists():
            issues.append(f"missing dataset file: {path}")
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counters[str(path.name)] = len(records)
        for index, record in enumerate(records):
            prefix = f"{path.name}:{index}"
            for key in ("instruction", "state", "action_chunk"):
                if key not in record:
                    issues.append(f"{prefix} missing `{key}`")
            state = record.get("state")
            if not isinstance(state, dict):
                issues.append(f"{prefix} state must be dict")
            action_chunk = record.get("action_chunk")
            if not isinstance(action_chunk, list):
                issues.append(f"{prefix} action_chunk must be list")
            action_mask = record.get("action_mask")
            if not isinstance(action_mask, list) or len(action_mask) != horizon_k:
                issues.append(f"{prefix} action_mask must have length {horizon_k}")
            image_path = record.get("image_path") or record.get("image") or record.get("source_image_path")
            if not image_path:
                issues.append(f"{prefix} missing image path")
            else:
                candidate = Path(str(image_path))
                if not candidate.is_absolute():
                    candidate = image_root / candidate
                if not candidate.exists():
                    issues.append(f"{prefix} unresolved image path: {image_path}")
        if issues:
            break
    return {"ok": not issues, "issues": issues[:50], "counts": dict(counters), "image_root": str(image_root), "horizon_k": horizon_k}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate processed datasets for Go2 velocity-control training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--real-root", type=Path, required=True)
    build_parser.add_argument("--sim-root", type=Path, required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument("--horizon-k", type=int, default=6)
    build_parser.add_argument(
        "--split-mode",
        choices=["by_session", "by_trajectory", "by_contrast_group", "auto"],
        default="by_trajectory",
    )
    build_parser.add_argument("--train-ratio", type=float, default=0.9)
    build_parser.add_argument("--val-ratio", type=float, default=0.1)
    build_parser.add_argument("--test-ratio", type=float, default=0.0)
    build_parser.add_argument("--split-seed", type=int, default=42)
    build_parser.add_argument("--drop-initial-frames-real", type=int, default=0)
    build_parser.add_argument("--drop-initial-frames-sim", type=int, default=0)
    build_parser.add_argument("--max-success-tail-frames-real", type=int)
    build_parser.add_argument("--max-success-tail-frames-sim", type=int)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data-path", type=Path, action="append", required=True)
    validate_parser.add_argument("--image-root", type=Path, required=True)
    validate_parser.add_argument("--horizon-k", type=int, default=6)

    args = parser.parse_args()
    if args.command == "build":
        stats = build_processed_dataset(
            real_root=args.real_root.resolve(),
            sim_root=args.sim_root.resolve(),
            output_root=args.output_root.resolve(),
            horizon_k=args.horizon_k,
            split_mode=args.split_mode,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            drop_initial_frames_real=args.drop_initial_frames_real,
            drop_initial_frames_sim=args.drop_initial_frames_sim,
            max_success_tail_frames_real=args.max_success_tail_frames_real,
            max_success_tail_frames_sim=args.max_success_tail_frames_sim,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    report = validate_processed_dataset(
        data_paths=[path.resolve() for path in args.data_path],
        image_root=args.image_root.resolve(),
        horizon_k=args.horizon_k,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
