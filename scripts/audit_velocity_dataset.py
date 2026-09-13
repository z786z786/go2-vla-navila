#!/usr/bin/env python3
"""Audit velocity-control JSONL datasets and optionally export a source-filtered copy."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is expected in the training env.
    Image = None


ACTION_DIMS = ("vx", "vy", "wz")
REQUIRED_KEYS = ("image", "instruction", "state", "action_chunk", "action_mask")
DEFAULT_RANGES = {
    "vx": (-1.0, 1.0),
    "vy": (-0.6, 0.6),
    "wz": (-1.2, 1.2),
}


@dataclass
class SourceStats:
    """Aggregated stats for one data source such as real or sim."""

    samples: int = 0
    steps: int = 0
    bad_samples: int = 0
    bad_action_steps: int = 0
    missing_images: int = 0
    decoded_images: int = 0
    unreadable_images: int = 0
    nonfinite_actions: int = 0
    range_violations: Dict[str, int] = field(default_factory=lambda: {dim: 0 for dim in ACTION_DIMS})
    boundary_clamp_candidates: Dict[str, int] = field(default_factory=lambda: {dim: 0 for dim in ACTION_DIMS})
    mins: Dict[str, float] = field(default_factory=lambda: {dim: math.inf for dim in ACTION_DIMS})
    maxs: Dict[str, float] = field(default_factory=lambda: {dim: -math.inf for dim in ACTION_DIMS})
    sum_abs: Dict[str, float] = field(default_factory=lambda: {dim: 0.0 for dim in ACTION_DIMS})
    nonzero: Dict[str, int] = field(default_factory=lambda: {dim: 0 for dim in ACTION_DIMS})
    instructions: Counter = field(default_factory=Counter)
    targets: Counter = field(default_factory=Counter)
    sessions: Counter = field(default_factory=Counter)
    modes: Counter = field(default_factory=Counter)
    gait_types: Counter = field(default_factory=Counter)
    masks: Counter = field(default_factory=Counter)
    action_lengths: Counter = field(default_factory=Counter)
    prev_action_sources: Counter = field(default_factory=Counter)
    prev_action_valid: Counter = field(default_factory=Counter)
    prev_action_reasons: Counter = field(default_factory=Counter)
    event_counts: Counter = field(default_factory=Counter)
    contrast_groups: Dict[str, set[str]] = field(default_factory=dict)
    contrast_group_episodes: Dict[str, set[str]] = field(default_factory=dict)
    contrast_group_splits: Dict[str, set[str]] = field(default_factory=dict)
    contrast_group_scenes: Dict[str, set[str]] = field(default_factory=dict)

    def observe_contrast(
        self,
        *,
        contrast_group_id: str,
        active_target_id: str,
        episode_id: str,
        split_name: str,
        scene_id: str,
    ) -> None:
        if not contrast_group_id:
            return
        target_key = active_target_id or "unknown"
        episode_key = episode_id or "unknown"
        scene_key = scene_id or "unknown"
        self.contrast_groups.setdefault(contrast_group_id, set()).add(target_key)
        self.contrast_group_episodes.setdefault(contrast_group_id, set()).add(episode_key)
        self.contrast_group_splits.setdefault(contrast_group_id, set()).add(split_name)
        self.contrast_group_scenes.setdefault(contrast_group_id, set()).add(scene_key)

    def observe_action(self, values: Tuple[float, float, float], ranges: Mapping[str, Tuple[float, float]]) -> None:
        self.steps += 1
        for dim, value in zip(ACTION_DIMS, values):
            if not math.isfinite(value):
                self.nonfinite_actions += 1
                continue
            self.mins[dim] = min(self.mins[dim], value)
            self.maxs[dim] = max(self.maxs[dim], value)
            self.sum_abs[dim] += abs(value)
            if abs(value) > 1e-8:
                self.nonzero[dim] += 1
            low, high = ranges[dim]
            if value < low or value > high:
                self.range_violations[dim] += 1
            if value <= low or value >= high:
                self.boundary_clamp_candidates[dim] += 1

    def to_dict(self) -> Dict[str, Any]:
        denom = max(self.steps, 1)
        contrast_group_count = len(self.contrast_groups)
        complete_contrast_group_count = sum(
            1
            for group_id, targets in self.contrast_groups.items()
            if len(targets) >= 2 and len(self.contrast_group_episodes.get(group_id, set())) >= 2
        )
        contrast_split_leakage_count = sum(
            1 for splits in self.contrast_group_splits.values() if len(splits) > 1
        )
        return {
            "samples": self.samples,
            "steps": self.steps,
            "bad_samples": self.bad_samples,
            "bad_action_steps": self.bad_action_steps,
            "missing_images": self.missing_images,
            "decoded_images": self.decoded_images,
            "unreadable_images": self.unreadable_images,
            "nonfinite_actions": self.nonfinite_actions,
            "range_violations": dict(self.range_violations),
            "boundary_clamp_candidates": dict(self.boundary_clamp_candidates),
            "ranges": {
                dim: [
                    None if self.mins[dim] == math.inf else self.mins[dim],
                    None if self.maxs[dim] == -math.inf else self.maxs[dim],
                ]
                for dim in ACTION_DIMS
            },
            "mean_abs": {dim: self.sum_abs[dim] / denom for dim in ACTION_DIMS},
            "nonzero_fraction": {dim: self.nonzero[dim] / denom for dim in ACTION_DIMS},
            "instructions_top": self.instructions.most_common(20),
            "targets_top": self.targets.most_common(20),
            "sessions_top": self.sessions.most_common(20),
            "modes_top": self.modes.most_common(20),
            "gait_types_top": self.gait_types.most_common(20),
            "masks_top": [(list(key), value) for key, value in self.masks.most_common(20)],
            "action_lengths": dict(self.action_lengths),
            "prev_action_sources": dict(self.prev_action_sources),
            "prev_action_valid": dict(self.prev_action_valid),
            "prev_action_reasons": dict(self.prev_action_reasons),
            "event_counts": dict(self.event_counts),
            "event_fraction": {
                key: value / max(self.samples, 1)
                for key, value in sorted(self.event_counts.items())
            },
            "contrast_group_count": contrast_group_count,
            "contrast_complete_group_count": complete_contrast_group_count,
            "contrast_complete_group_ratio": (
                complete_contrast_group_count / max(contrast_group_count, 1)
            ),
            "contrast_split_leakage_count": contrast_split_leakage_count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing train.jsonl, val.jsonl, images/")
    parser.add_argument("--output-dir", required=True, help="Audit output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to audit")
    parser.add_argument("--filter-source", default=None, help="Optional source_dataset/source value to export")
    parser.add_argument("--filtered-output-root", default=None, help="Where to write filtered train/val JSONL and image symlink")
    parser.add_argument("--decode-image-samples", type=int, default=256, help="Max images per source to PIL-decode")
    parser.add_argument("--vx-range", nargs=2, type=float, default=DEFAULT_RANGES["vx"])
    parser.add_argument("--vy-range", nargs=2, type=float, default=DEFAULT_RANGES["vy"])
    parser.add_argument("--wz-range", nargs=2, type=float, default=DEFAULT_RANGES["wz"])
    parser.add_argument("--lane", default=None, help="Metadata lane label, e.g. real_only or sim_only")
    parser.add_argument("--upstream-dataset-root", default=None, help="Original dataset root before filtering")
    parser.add_argument("--config-name", default=None, help="Associated config or data-prep profile name")
    parser.add_argument("--velocity-profile-name", default=None, help="Velocity profile metadata label")
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def source_of(sample: Mapping[str, Any]) -> str:
    return str(sample.get("source_dataset") or sample.get("source") or sample.get("source_type") or "unknown")


def image_path_for(dataset_root: Path, sample: Mapping[str, Any]) -> Path:
    image_value = sample.get("image") or sample.get("image_path") or sample.get("source_image_path")
    if image_value is None:
        return dataset_root / "images" / "__missing_image__"
    path = Path(str(image_value))
    if path.is_absolute():
        return path
    return dataset_root / "images" / path


def action_values(step: Any) -> Optional[Tuple[float, float, float]]:
    try:
        if isinstance(step, Mapping):
            return float(step.get("vx", 0.0)), float(step.get("vy", 0.0)), float(step.get("wz", 0.0))
        values = list(step)
        if len(values) >= 3:
            return float(values[0]), float(values[1]), float(values[2])
        if len(values) == 2:
            return float(values[0]), 0.0, float(values[1])
    except Exception:
        return None
    return None


def previous_action_values(sample: Mapping[str, Any]) -> Tuple[float, float, float]:
    previous = sample.get("previous_action") or sample.get("prev_action")
    if isinstance(previous, Mapping):
        return (
            float(previous.get("vx", 0.0)),
            float(previous.get("vy", 0.0)),
            float(previous.get("wz", 0.0)),
        )
    state = sample.get("state")
    if isinstance(state, Mapping):
        return (
            float(state.get("vx_prev", 0.0)),
            float(state.get("vy_prev", 0.0)),
            float(state.get("wz_prev", 0.0)),
        )
    action = sample.get("control_action") or sample.get("raw_action") or {}
    if isinstance(action, Mapping):
        return (
            float(action.get("vx", 0.0)),
            float(action.get("vy", 0.0)),
            float(action.get("wz", 0.0)),
        )
    return (0.0, 0.0, 0.0)


def event_flags(sample: Mapping[str, Any]) -> Dict[str, bool]:
    chunk = sample.get("action_chunk") or []
    values = [action_values(step) for step in chunk]
    values = [item for item in values if item is not None]
    if not values:
        return {
            "turning": False,
            "first_step_change": False,
            "wz_change": False,
            "vx_change": False,
            "low_vx_or_stop": False,
            "straight_dominant": False,
        }
    prev_vx, prev_vy, prev_wz = previous_action_values(sample)
    turning = any(abs(wz) > 0.05 for _, _, wz in values)
    first_vx, first_vy, first_wz = values[0]
    first_step_change = (
        abs(first_vx - prev_vx) > 0.03 or
        abs(first_vy - prev_vy) > 0.03 or
        abs(first_wz - prev_wz) > 0.05
    )
    wz_change = False
    vx_change = False
    last_vx, last_wy, last_wz = prev_vx, prev_vy, prev_wz
    for vx, vy, wz in values:
        if abs(wz - last_wz) > 0.05:
            wz_change = True
        if abs(vx - last_vx) > 0.03:
            vx_change = True
        last_vx, last_wy, last_wz = vx, vy, wz
    low_vx_or_stop = any(vx < 0.1 for vx, _, _ in values)
    straight_dominant = all(abs(wz) <= 0.05 for _, _, wz in values)
    return {
        "turning": turning,
        "first_step_change": first_step_change,
        "wz_change": wz_change,
        "vx_change": vx_change,
        "low_vx_or_stop": low_vx_or_stop,
        "straight_dominant": straight_dominant,
    }


def validate_sample_schema(sample: Mapping[str, Any]) -> List[str]:
    issues: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in sample:
            issues.append(f"missing_key:{key}")
    if not isinstance(sample.get("state", {}), Mapping):
        issues.append("state_not_dict")
    if not isinstance(sample.get("action_chunk", []), list):
        issues.append("action_chunk_not_list")
    action_mask = sample.get("action_mask")
    if not isinstance(action_mask, list):
        issues.append("action_mask_not_list")
    elif len(action_mask) != 6:
        issues.append(f"action_mask_len:{len(action_mask)}")
    return issues


def maybe_decode_image(path: Path) -> bool:
    if Image is None:
        return False
    with Image.open(path) as image:
        image.verify()
    return True


def write_filtered_dataset(
    dataset_root: Path,
    filtered_output_root: Path,
    split_rows: Mapping[str, List[Dict[str, Any]]],
    summary: Mapping[str, Any],
) -> None:
    filtered_output_root.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        with (filtered_output_root / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (filtered_output_root / "dataset.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for split in ("train", "val", "test")
            for row in split_rows.get(split, [])
        ),
        encoding="utf-8",
    )
    images_link = filtered_output_root / "images"
    if not images_link.exists():
        images_link.symlink_to((dataset_root / "images").resolve(), target_is_directory=True)
    (filtered_output_root / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Velocity Dataset Audit",
        "",
        f"- dataset_root: `{summary['dataset_root']}`",
        f"- splits: `{', '.join(summary['splits'])}`",
        f"- total_samples: `{summary['total_samples']}`",
        f"- status: `{'PASS' if summary['ok'] else 'FAIL'}`",
        "",
        "## Issues",
        "",
    ]
    if summary["issues"]:
        lines.extend(f"- {issue}" for issue in summary["issues"])
    else:
        lines.append("- none")
    lines.extend(["", "## Source Summary", ""])
    for source, stats in sorted(summary["sources"].items()):
        lines.extend(
            [
                f"### {source}",
                "",
                f"- samples: `{stats['samples']}`",
                f"- steps: `{stats['steps']}`",
                f"- missing_images: `{stats['missing_images']}`",
                f"- unreadable_images: `{stats['unreadable_images']}`",
                f"- bad_action_steps: `{stats['bad_action_steps']}`",
                f"- nonfinite_actions: `{stats['nonfinite_actions']}`",
                f"- ranges: `{stats['ranges']}`",
                f"- mean_abs: `{stats['mean_abs']}`",
                f"- nonzero_fraction: `{stats['nonzero_fraction']}`",
                f"- range_violations: `{stats['range_violations']}`",
                f"- prev_action_sources: `{stats['prev_action_sources']}`",
                f"- prev_action_valid: `{stats['prev_action_valid']}`",
                f"- prev_action_reasons: `{stats['prev_action_reasons']}`",
                f"- contrast_groups: `{stats['contrast_group_count']}`",
                f"- contrast_complete_groups: `{stats['contrast_complete_group_count']}`",
                f"- contrast_split_leakage: `{stats['contrast_split_leakage_count']}`",
                f"- event_fraction: `{stats['event_fraction']}`",
                f"- instructions_top: `{stats['instructions_top'][:8]}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    ranges = {
        "vx": tuple(args.vx_range),
        "vy": tuple(args.vy_range),
        "wz": tuple(args.wz_range),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, SourceStats] = defaultdict(SourceStats)
    split_counts: Dict[str, Counter] = {}
    split_image_paths: Dict[str, set[str]] = {}
    schema_issues = Counter()
    filtered_rows: Dict[str, List[Dict[str, Any]]] = {split: [] for split in args.splits}
    filtered_event_counts: Dict[str, Counter] = {split: Counter() for split in args.splits}
    filtered_prev_sources: Dict[str, Counter] = {split: Counter() for split in args.splits}
    filtered_prev_valid: Dict[str, Counter] = {split: Counter() for split in args.splits}
    filtered_prev_reasons: Dict[str, Counter] = {split: Counter() for split in args.splits}
    filtered_contrast_groups: Dict[str, Dict[str, set[str]]] = {split: {} for split in args.splits}
    filtered_contrast_episodes: Dict[str, Dict[str, set[str]]] = {split: {} for split in args.splits}

    for split in args.splits:
        split_path = dataset_root / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        split_counts[split] = Counter()
        split_image_paths[split] = set()
        for sample in load_jsonl(split_path):
            source = source_of(sample)
            source_stats = stats[source]
            source_stats.samples += 1
            split_counts[split][source] += 1

            sample_issues = validate_sample_schema(sample)
            if sample_issues:
                source_stats.bad_samples += 1
                schema_issues.update(sample_issues)

            instruction = str(sample.get("instruction", ""))
            state = sample.get("state") if isinstance(sample.get("state"), Mapping) else {}
            source_stats.instructions[instruction] += 1
            source_stats.targets[str(sample.get("target_type") or sample.get("target_label") or "unknown")] += 1
            source_stats.sessions[str(sample.get("session_id") or "unknown")] += 1
            source_stats.modes[str(state.get("mode", "missing"))] += 1
            source_stats.gait_types[str(state.get("gait_type", "missing"))] += 1
            action_mask = sample.get("action_mask")
            if isinstance(action_mask, list):
                source_stats.masks[tuple(int(bool(value)) for value in action_mask)] += 1
            source_stats.prev_action_sources[str(sample.get("prev_action_source") or "missing")] += 1
            source_stats.prev_action_valid[str(bool(sample.get("prev_action_valid", False))).lower()] += 1
            source_stats.prev_action_reasons[str(sample.get("prev_action_reason") or "missing")] += 1
            contrast_group_id = str(sample.get("contrast_group_id") or "")
            active_target_id = str(sample.get("active_target_id") or sample.get("target_label") or sample.get("target_type") or "")
            source_stats.observe_contrast(
                contrast_group_id=contrast_group_id,
                active_target_id=active_target_id,
                episode_id=f"{sample.get('session_id') or 'unknown'}:{sample.get('episode_id') or sample.get('trajectory_id') or 'unknown'}",
                split_name=split,
                scene_id=str(sample.get("scene_id") or ""),
            )

            image_path = image_path_for(dataset_root, sample)
            split_image_paths[split].add(str(image_path))
            if not image_path.exists():
                source_stats.missing_images += 1
            elif source_stats.decoded_images < args.decode_image_samples:
                try:
                    maybe_decode_image(image_path)
                    source_stats.decoded_images += 1
                except Exception:
                    source_stats.unreadable_images += 1

            action_chunk = sample.get("action_chunk") or []
            source_stats.action_lengths[len(action_chunk)] += 1
            for step in action_chunk:
                values = action_values(step)
                if values is None:
                    source_stats.bad_action_steps += 1
                else:
                    source_stats.observe_action(values, ranges)
            for event_name, flag in event_flags(sample).items():
                if flag:
                    source_stats.event_counts[event_name] += 1

            if args.filter_source is not None and source == args.filter_source:
                filtered_rows[split].append(sample)
                filtered_prev_sources[split][str(sample.get("prev_action_source") or "missing")] += 1
                filtered_prev_valid[split][str(bool(sample.get("prev_action_valid", False))).lower()] += 1
                filtered_prev_reasons[split][str(sample.get("prev_action_reason") or "missing")] += 1
                if contrast_group_id:
                    filtered_contrast_groups[split].setdefault(contrast_group_id, set()).add(active_target_id or "unknown")
                    filtered_contrast_episodes[split].setdefault(contrast_group_id, set()).add(
                        f"{sample.get('session_id') or 'unknown'}:{sample.get('episode_id') or sample.get('trajectory_id') or 'unknown'}"
                    )
                for event_name, flag in event_flags(sample).items():
                    if flag:
                        filtered_event_counts[split][event_name] += 1

    leakage = sorted(split_image_paths.get("train", set()) & split_image_paths.get("val", set()))
    issues: List[str] = []
    if leakage:
        issues.append(f"train_val_image_leakage:{len(leakage)}")
    for issue, count in schema_issues.items():
        issues.append(f"schema:{issue}:{count}")
    for source, source_stats in sorted(stats.items()):
        if source_stats.missing_images:
            issues.append(f"{source}:missing_images:{source_stats.missing_images}")
        if source_stats.unreadable_images:
            issues.append(f"{source}:unreadable_images:{source_stats.unreadable_images}")
        if source_stats.bad_action_steps:
            issues.append(f"{source}:bad_action_steps:{source_stats.bad_action_steps}")
        if source_stats.nonfinite_actions:
            issues.append(f"{source}:nonfinite_actions:{source_stats.nonfinite_actions}")
        if source_stats.to_dict()["contrast_split_leakage_count"]:
            issues.append(f"{source}:contrast_split_leakage:{source_stats.to_dict()['contrast_split_leakage_count']}")

    total_samples = sum(source.samples for source in stats.values())
    filtered_summary = None
    if args.filter_source and args.filtered_output_root:
        filtered_summary = {
            "lane": args.lane,
            "source_filter": args.filter_source,
            "upstream_dataset_root": str(Path(args.upstream_dataset_root) if args.upstream_dataset_root else dataset_root),
            "source_root": str(dataset_root),
            "output_root": str(Path(args.filtered_output_root)),
            "config_name": args.config_name,
            "velocity_profile_name": args.velocity_profile_name,
            "split_counts": {split: len(rows) for split, rows in filtered_rows.items()},
            "sample_count": sum(len(rows) for rows in filtered_rows.values()),
            "horizon_k": 6,
            "prev_action_sources": {split: dict(counter) for split, counter in filtered_prev_sources.items()},
            "prev_action_valid": {split: dict(counter) for split, counter in filtered_prev_valid.items()},
            "prev_action_reasons": {split: dict(counter) for split, counter in filtered_prev_reasons.items()},
            "event_counts": {split: dict(counter) for split, counter in filtered_event_counts.items()},
            "contrast_group_counts": {
                split: len(groups) for split, groups in filtered_contrast_groups.items()
            },
            "contrast_complete_group_counts": {
                split: sum(
                    1
                    for group_id, targets in groups.items()
                    if len(targets) >= 2 and len(filtered_contrast_episodes[split].get(group_id, set())) >= 2
                )
                for split, groups in filtered_contrast_groups.items()
            },
        }
        write_filtered_dataset(dataset_root, Path(args.filtered_output_root), filtered_rows, filtered_summary)

    summary = {
        "ok": not issues,
        "lane": args.lane,
        "dataset_root": str(dataset_root),
        "upstream_dataset_root": str(Path(args.upstream_dataset_root) if args.upstream_dataset_root else dataset_root),
        "splits": args.splits,
        "total_samples": total_samples,
        "config_name": args.config_name,
        "velocity_profile_name": args.velocity_profile_name,
        "split_counts": {split: dict(counter) for split, counter in split_counts.items()},
        "ranges": ranges,
        "issues": issues,
        "sources": {source: source_stats.to_dict() for source, source_stats in sorted(stats.items())},
        "filtered_dataset": filtered_summary,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "audit_report.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "issues": issues, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
