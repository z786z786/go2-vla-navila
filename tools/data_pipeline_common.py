from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from llada_vla_common import (
        DEFAULT_MAX_PROCESSING_QUALITY_LABEL,
        FRAME_SELECTION_MODE_ALL,
        action_bucket,
        control_action_from_frame,
        discover_session_roots,
        dump_json,
        dump_jsonl,
        episode_entries,
        episode_frame_processing,
        episode_task_metadata,
        estimate_nominal_dt,
        frame_processing_flag,
        load_json,
        load_jsonl,
        load_quality_review_index,
        previous_action_info_for_step,
        raw_action_from_frame,
        resolved_episode_quality,
        state_from_frame,
    )
else:
    from .llada_vla_common import (
        DEFAULT_MAX_PROCESSING_QUALITY_LABEL,
        FRAME_SELECTION_MODE_ALL,
        action_bucket,
        control_action_from_frame,
        discover_session_roots,
        dump_json,
        dump_jsonl,
        episode_entries,
        episode_frame_processing,
        episode_task_metadata,
        estimate_nominal_dt,
        frame_processing_flag,
        load_json,
        load_jsonl,
        load_quality_review_index,
        previous_action_info_for_step,
        raw_action_from_frame,
        resolved_episode_quality,
        state_from_frame,
    )

CANONICAL_SCHEMA_VERSION = "go2_canonical_v1"
WINDOW_SCHEMA_VERSION = "go2_training_windows_v1"
EXPORT_SCHEMA_VERSION = "go2_training_export_v1"
LIVE_MONITOR_SCHEMA_VERSION = "go2_live_monitor_v1"

CANONICAL_STATE_FIELDS = ["vx", "vy", "vz"]
CANONICAL_STATE_EXTRA_FIELDS = ["wz", "yaw", "roll", "pitch", "x", "y", "z", "body_height", "error_code", "mode", "gait_type"]
PRIMARY_ACTION_FIELDS = ["vx", "wz"]
FULL_ACTION_FIELDS = ["vx", "vy", "wz"]
PHASES = [
    "IDLE",
    "START",
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "CORRECTION",
    "APPROACH",
    "STOP",
    "ALIGN",
    "RECOVERY",
]
DEFAULT_PHASE_PRIORITY = [
    "RECOVERY",
    "STOP",
    "ALIGN",
    "APPROACH",
    "START",
    "TURN_LEFT",
    "TURN_RIGHT",
    "CORRECTION",
    "FORWARD",
    "IDLE",
]
DEFAULT_HIGH_VALUE_PHASES = ["RECOVERY", "ALIGN", "STOP", "APPROACH", "CORRECTION"]
PHASE_TO_INSTRUCTION = {
    "FORWARD": "move toward the target",
    "TURN_LEFT": "turn left and approach the target",
    "TURN_RIGHT": "turn right and approach the target",
    "CORRECTION": "make a small correction and continue",
    "APPROACH": "move closer to the target",
    "STOP": "stop near the target",
    "ALIGN": "face the target",
    "RECOVERY": "recover and reacquire the target",
    "START": "start moving toward the target",
    "IDLE": "hold position and wait",
}
FIELD_SOURCE_DESCRIPTION = {
    "timestamp": "frame.timestamp",
    "image_path": "frame.image relative to raw_root",
    "raw_instruction": "episode.instruction / frame.instruction",
    "normalized_instruction": "normalized from raw_instruction",
    "state": "frame.state ordered as [vx, vy, vz]",
    "state_dict": "frame.state full dictionary with extra raw fields when available",
    "action": "frame.control_action projected to [vx, wz]",
    "success": "episode.success mapped to bool|null",
    "collision": "not present in current raw data, kept as null unless future raw field exists",
    "goal_visible": "raw field if present, otherwise null",
    "goal_relative_angle": "raw field if present, otherwise null",
    "goal_relative_distance": "raw field if present, otherwise null",
    "proxy_goal_relative_angle": "dead-reckoned endpoint proxy from integrated state velocities",
    "proxy_goal_relative_distance": "dead-reckoned endpoint proxy from integrated state velocities",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _has_leading_article(value: str) -> bool:
    lowered = _safe_str(value).lower()
    prefixes = (
        "the ",
        "a ",
        "an ",
        "this ",
        "that ",
        "these ",
        "those ",
        "my ",
        "your ",
        "our ",
        "their ",
    )
    return any(lowered.startswith(prefix) for prefix in prefixes)


def _with_definite_article(value: str) -> str:
    text = _safe_str(value)
    if not text or _has_leading_article(text):
        return text
    return f"the {text}"


def _resolve_semantic_target_text(record: Dict[str, Any]) -> str:
    return _safe_str(record.get("target_label") or record.get("target_description") or record.get("target_type"))


def _semantic_instruction_candidates(record: Dict[str, Any]) -> List[str]:
    task_family = _safe_str(record.get("task_family"))
    target_text = _with_definite_article(_resolve_semantic_target_text(record))
    if not task_family or not target_text:
        return []
    if task_family == "goal_navigation":
        return [
            f"go to {target_text}",
            f"move to {target_text}",
            f"navigate to {target_text}",
            f"approach {target_text}",
        ]
    if task_family == "visual_following":
        return [
            f"follow {target_text}",
            f"track {target_text}",
            f"stay with {target_text}",
            f"walk with {target_text}",
        ]
    if task_family == "obstacle_aware_navigation":
        return [
            f"go around {target_text}",
            f"move around {target_text}",
            f"navigate around {target_text}",
            f"avoid {target_text}",
        ]
    return []


def _turn_bucket_from_actions(actions: Sequence[Sequence[float]], *, small_turn_threshold: float = 0.1, large_turn_threshold: float = 0.35) -> str:
    if not actions:
        return "unknown"
    signed_peak = 0.0
    peak_abs = 0.0
    for step in actions:
        if len(step) < 2:
            continue
        wz = _safe_float(step[1])
        if abs(wz) > peak_abs:
            peak_abs = abs(wz)
            signed_peak = wz
    if peak_abs < small_turn_threshold:
        return "straight"
    if signed_peak > 0.0:
        return "left_large" if peak_abs >= large_turn_threshold else "left_small"
    return "right_large" if peak_abs >= large_turn_threshold else "right_small"


def _safe_bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _safe_str(value).lower()
    if text in {"true", "1", "yes", "y", "visible", "success", "hit", "collision"}:
        return True
    if text in {"false", "0", "no", "n", "hidden", "fail", "miss", "none"}:
        return False
    if not text:
        return None
    return None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_instruction_text(text: str) -> str:
    lowered = " ".join(_safe_str(text).lower().split())
    return lowered


def load_config_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def merge_config(defaults: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = dict(defaults)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def canonical_defaults() -> Dict[str, Any]:
    return {
        "quality": {
            "max_processing_quality_label": DEFAULT_MAX_PROCESSING_QUALITY_LABEL,
        },
        "proxy_goal": {
            "enabled": True,
            "min_total_distance": 1e-4,
        },
    }


def phase_defaults() -> Dict[str, Any]:
    return {
        "priority": list(DEFAULT_PHASE_PRIORITY),
        "thresholds": {
            "idle_vx": 0.03,
            "idle_wz": 0.06,
            "start_vx": 0.10,
            "forward_vx": 0.10,
            "reverse_vx": -0.04,
            "turn_wz": 0.18,
            "correction_wz": 0.10,
            "correction_vx": 0.05,
            "align_wz": 0.18,
            "align_vx": 0.08,
            "align_angle": 0.22,
            "approach_vx": 0.08,
            "approach_distance": 0.9,
            "approach_distance_delta": 0.005,
            "stop_distance": 0.45,
            "stop_tail_ratio": 0.12,
            "start_frame_count": 6,
            "recovery_sign_flips": 3,
            "recovery_window": 6,
            "recovery_turn": 0.20,
        },
    }


def filter_defaults() -> Dict[str, Any]:
    return {
        "turn_threshold": 0.18,
        "accel_threshold": 0.08,
        "wz_accel_threshold": 0.10,
        "goal_angle_delta_threshold": 0.12,
        "goal_distance_threshold": 0.9,
        "keep_every_n": 6,
        "dense_keep_every_n_near_goal": 2,
        "high_value_phases": list(DEFAULT_HIGH_VALUE_PHASES),
        "force_keep_first_last": True,
    }


def window_defaults() -> Dict[str, Any]:
    return {
        "chunk_len": 4,
        "stride": 1,
        "pad_strategy": "repeat_last",
        "action_fields": list(PRIMARY_ACTION_FIELDS),
    }


def export_defaults() -> Dict[str, Any]:
    return {
        "split": {
            "mode": "by_episode",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": None,
        },
        "instruction_policy": {
            "mode": "semantic_auto",
            "mix_phase_probability": 0.5,
        },
        "routes": ["A", "B"],
    }


def live_monitor_defaults() -> Dict[str, Any]:
    return {
        "turn_threshold": 0.18,
        "forward_threshold": 0.10,
        "stop_vx_threshold": 0.03,
        "stop_wz_threshold": 0.06,
        "correction_wz_threshold": 0.10,
        "recovery_vx_threshold": -0.04,
        "history_size": 120,
        "render_every": 1.0,
        "scarcity_threshold": 0.08,
        "high_value_phases": list(DEFAULT_HIGH_VALUE_PHASES),
    }


def discover_session_roots_deep(raw_root: Path) -> List[Path]:
    direct = discover_session_roots(raw_root)
    if direct:
        nested: List[Path] = []
        for child in sorted(raw_root.rglob("index.json")):
            session_root = child.parent
            if session_root == raw_root:
                continue
            if (session_root / "episodes").is_dir() and session_root not in direct and session_root not in nested:
                nested.append(session_root)
        return sorted(direct + nested)

    roots: List[Path] = []
    for index_path in sorted(raw_root.rglob("index.json")):
        session_root = index_path.parent
        if (session_root / "episodes").is_dir():
            roots.append(session_root)
    deduped: List[Path] = []
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def canonical_episode_relpath(session_id: str, episode_id: str) -> Path:
    return Path("episodes") / session_id / f"{episode_id}.jsonl"


def canonical_episode_path(view_root: Path, session_id: str, episode_id: str) -> Path:
    return view_root / canonical_episode_relpath(session_id, episode_id)


def infer_session_root_from_canonical(view_root: Path, record: Dict[str, Any]) -> Optional[Path]:
    source = record.get("source")
    if not isinstance(source, dict):
        return None
    raw_session_root = source.get("raw_session_root")
    if not raw_session_root:
        return None
    return Path(str(raw_session_root))


def extract_nested_value(sources: Sequence[Any], names: Sequence[str]) -> Any:
    lowered_names = {name.lower() for name in names}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if str(key).lower() in lowered_names:
                return value
    return None


def success_label_to_bool(label: Any) -> Optional[bool]:
    text = _safe_str(label).lower()
    if text == "success":
        return True
    if text == "fail":
        return False
    return None


def _quality_review_map(raw_root: Path, review_path: Optional[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return load_quality_review_index(raw_root, review_path)


def integrate_positions(frames: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = []
    x = 0.0
    y = 0.0
    last_timestamp: Optional[float] = None
    last_state: Optional[Dict[str, float]] = None
    for frame in frames:
        timestamp = _safe_float(frame.get("timestamp"))
        state = state_from_frame(frame)
        if last_timestamp is not None and last_state is not None:
            dt = max(0.0, timestamp - last_timestamp)
            yaw = _safe_float(last_state.get("yaw"))
            vx = _safe_float(last_state.get("vx"))
            vy = _safe_float(last_state.get("vy"))
            world_vx = vx * math.cos(yaw) - vy * math.sin(yaw)
            world_vy = vx * math.sin(yaw) + vy * math.cos(yaw)
            x += world_vx * dt
            y += world_vy * dt
        positions.append((x, y))
        last_timestamp = timestamp
        last_state = state
    return positions


def build_proxy_goal_features(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Optional[float]]]:
    if not frames:
        return []
    positions = integrate_positions(frames)
    final_x, final_y = positions[-1]
    total_distance = math.hypot(final_x - positions[0][0], final_y - positions[0][1])
    proxy_records: List[Dict[str, Optional[float]]] = []
    for index, frame in enumerate(frames):
        x, y = positions[index]
        state = state_from_frame(frame)
        dx = final_x - x
        dy = final_y - y
        remaining = math.hypot(dx, dy)
        yaw = _safe_float(state.get("yaw"))
        angle = wrap_angle(math.atan2(dy, dx) - yaw) if remaining > 1e-8 else 0.0
        progress = 0.0
        if total_distance > 1e-8:
            progress = clamp(1.0 - remaining / total_distance, 0.0, 1.0)
        proxy_records.append(
            {
                "proxy_x": x,
                "proxy_y": y,
                "proxy_goal_relative_distance": remaining,
                "proxy_goal_relative_angle": angle,
                "proxy_goal_progress": progress,
                "proxy_total_displacement": total_distance,
            }
        )
    return proxy_records


def _relative_image_path(raw_root: Path, session_root: Path, image_rel: str) -> str:
    try:
        return str((session_root / image_rel).resolve().relative_to(raw_root.resolve()))
    except Exception:
        return str(Path(session_root.name) / image_rel)


def build_canonical_episode(
    raw_root: Path,
    session_root: Path,
    session_id: str,
    episode_meta: Dict[str, Any],
    canonical_config: Dict[str, Any],
    quality_review_index: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    episode_id = _safe_str(episode_meta.get("episode_id"))
    if not episode_id:
        return [], {}
    episode_path = session_root / "episodes" / f"{episode_id}.json"
    payload = load_json(episode_path)
    task_metadata = episode_task_metadata(payload, episode_meta)
    quality = resolved_episode_quality(payload, episode_meta, quality_review_index, session_id, episode_id)
    frame_processing = episode_frame_processing(payload)
    frames = [frame for frame in payload.get("frames") or [] if isinstance(frame, dict)]
    proxy_features = build_proxy_goal_features(frames) if canonical_config.get("proxy_goal", {}).get("enabled", True) else [{} for _ in frames]
    nominal_dt = estimate_nominal_dt(frames)

    selected_keep_map = frame_processing.get("frame_keep_map") or {}
    canonical_frames: List[Dict[str, Any]] = []
    instruction = _safe_str(payload.get("instruction") or episode_meta.get("instruction"))
    normalized_instruction = normalize_instruction_text(instruction)
    success_label = _safe_str(payload.get("success") or episode_meta.get("success"))
    success_value = success_label_to_bool(success_label)
    termination_reason = _safe_str(payload.get("termination_reason") or episode_meta.get("termination_reason"))

    for frame_id, frame in enumerate(frames):
        state_dict = state_from_frame(frame)
        raw_action = raw_action_from_frame(frame)
        control_action = frame.get("control_action") or frame.get("action") or frame.get("raw_action") or {}
        previous_action, prev_action_valid, prev_action_source, prev_action_reason = previous_action_info_for_step(
            frames=frames,
            selected_indices=list(range(len(frames))),
            step_pos=frame_id,
            nominal_dt=nominal_dt,
        )
        image_rel = _safe_str(frame.get("image"))
        image_path = _relative_image_path(raw_root, session_root, image_rel) if image_rel else ""
        frame_meta = frame.get("meta") if isinstance(frame.get("meta"), dict) else {}
        sources = [frame, frame_meta, payload, episode_meta]
        actual_goal_visible = _safe_bool_or_none(extract_nested_value(sources, ["goal_visible", "target_visible"]))
        actual_goal_angle = extract_nested_value(sources, ["goal_relative_angle", "target_relative_angle", "goal_angle"])
        actual_goal_distance = extract_nested_value(sources, ["goal_relative_distance", "target_relative_distance", "goal_distance"])
        collision_value = _safe_bool_or_none(extract_nested_value(sources, ["collision", "has_collision", "collided"]))
        frame_instruction = _safe_str(frame.get("instruction") or instruction)
        retain_flag = frame_processing_flag(frame)
        if retain_flag is None:
            retain_flag = frame_id in set(frame_processing.get("selected_frame_indices") or []) if frame_processing.get("frame_selection_mode") == FRAME_SELECTION_MODE_ALL else selected_keep_map.get(frame_id, False)
        canonical_frame = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "level": "canonical",
            "session_id": session_id,
            "episode_id": episode_id,
            "frame_id": frame_id,
            "timestamp": _safe_float(frame.get("timestamp")),
            "image_path": image_path,
            "source_image_path": str(session_root / image_rel) if image_rel else "",
            "raw_instruction": frame_instruction or instruction,
            "instruction": frame_instruction or instruction,
            "normalized_instruction": normalize_instruction_text(frame_instruction or instruction),
            "phase_derived_instruction": None,
            "instruction_for_training": frame_instruction or instruction,
            "state": [state_dict.get(field, 0.0) for field in CANONICAL_STATE_FIELDS],
            "state_dict": state_dict,
            "action": [_safe_float(control_action.get("vx")), _safe_float(control_action.get("wz"))],
            "action_dict": {"vx": _safe_float(control_action.get("vx")), "wz": _safe_float(control_action.get("wz"))},
            "action_full": [_safe_float(control_action.get(field)) for field in FULL_ACTION_FIELDS],
            "action_full_dict": {field: _safe_float(control_action.get(field)) for field in FULL_ACTION_FIELDS},
            "raw_action": raw_action,
            "previous_action": previous_action,
            "prev_action_valid": prev_action_valid,
            "prev_action_source": prev_action_source,
            "prev_action_reason": prev_action_reason,
            "success": success_value,
            "success_label": success_label or None,
            "collision": collision_value,
            "goal_visible": actual_goal_visible,
            "goal_relative_angle": None if actual_goal_angle is None else _safe_float(actual_goal_angle),
            "goal_relative_distance": None if actual_goal_distance is None else _safe_float(actual_goal_distance),
            "proxy_goal_relative_angle": proxy_features[frame_id].get("proxy_goal_relative_angle"),
            "proxy_goal_relative_distance": proxy_features[frame_id].get("proxy_goal_relative_distance"),
            "proxy_goal_progress": proxy_features[frame_id].get("proxy_goal_progress"),
            "proxy_position": [
                _safe_float(proxy_features[frame_id].get("proxy_x")),
                _safe_float(proxy_features[frame_id].get("proxy_y")),
            ],
            "phase": None,
            "phase_reason": None,
            "phase_priority": None,
            "quality_label": quality.get("quality_label"),
            "original_quality_label": quality.get("original_quality_label"),
            "quality_overridden": bool(quality.get("quality_overridden")),
            "quality_notes": quality.get("quality_notes") or "",
            "exclude_from_processing": bool(quality.get("exclude_from_processing")),
            "retain_for_processing": bool(retain_flag),
            "frame_selection_mode": frame_processing.get("frame_selection_mode") or FRAME_SELECTION_MODE_ALL,
            "selected_frame_count": frame_processing.get("selected_frame_count") or 0,
            "total_frame_count": frame_processing.get("total_frame_count") or len(frames),
            "capture_mode": task_metadata.get("capture_mode") or payload.get("capture_mode") or "",
            "task_family": task_metadata.get("task_family") or "",
            "target_type": task_metadata.get("target_type") or "",
            "target_label": task_metadata.get("target_label") or "",
            "target_description": task_metadata.get("target_description") or "",
            "collector_notes": task_metadata.get("collector_notes") or "",
            "instruction_source": task_metadata.get("instruction_source") or "",
            "segment_status": task_metadata.get("segment_status") or payload.get("segment_status") or "",
            "termination_reason": task_metadata.get("termination_reason") or termination_reason,
            "scene_id": _safe_str(payload.get("scene_id") or episode_meta.get("scene_id")),
            "operator_id": _safe_str(payload.get("operator_id") or episode_meta.get("operator_id")),
            "source": {
                "raw_root": str(raw_root),
                "raw_session_root": str(session_root),
                "raw_episode_path": str(episode_path),
                "source_frame_instruction": frame_instruction,
                "source_frame_meta": frame_meta,
            },
        }
        canonical_frames.append(canonical_frame)

    episode_summary = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "level": "canonical",
        "session_id": session_id,
        "episode_id": episode_id,
        "episode_path": str(canonical_episode_relpath(session_id, episode_id)),
        "frame_count": len(canonical_frames),
        "instruction": instruction,
        "normalized_instruction": normalized_instruction,
        "scene_id": _safe_str(payload.get("scene_id") or episode_meta.get("scene_id")),
        "operator_id": _safe_str(payload.get("operator_id") or episode_meta.get("operator_id")),
        "task_family": task_metadata.get("task_family") or "",
        "target_type": task_metadata.get("target_type") or "",
        "target_label": task_metadata.get("target_label") or "",
        "target_description": task_metadata.get("target_description") or "",
        "success": success_value,
        "success_label": success_label or None,
        "termination_reason": termination_reason or None,
        "quality_label": quality.get("quality_label"),
        "selected_frame_count": frame_processing.get("selected_frame_count") or 0,
        "frame_selection_mode": frame_processing.get("frame_selection_mode") or FRAME_SELECTION_MODE_ALL,
    }
    return canonical_frames, episode_summary


def write_canonical_view(
    raw_root: Path,
    output_root: Path,
    session_roots: Sequence[Path],
    canonical_config: Dict[str, Any],
    quality_review_index: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    episode_summaries: List[Dict[str, Any]] = []
    total_frames = 0
    total_selected_frames = 0
    session_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for session_root in session_roots:
        session_id, entries = episode_entries(session_root)
        for episode_meta in entries:
            frames, summary = build_canonical_episode(
                raw_root=raw_root,
                session_root=session_root,
                session_id=session_id,
                episode_meta=episode_meta,
                canonical_config=canonical_config,
                quality_review_index=quality_review_index,
            )
            if not frames:
                continue
            path = canonical_episode_path(output_root, session_id, summary["episode_id"])
            dump_jsonl(path, frames)
            episode_summaries.append(summary)
            total_frames += len(frames)
            total_selected_frames += _safe_int(summary.get("selected_frame_count"), len(frames))
            session_counts[session_id] += 1
            instruction_counts[_safe_str(summary.get("instruction"))] += 1
            quality_counts[str(summary.get("quality_label"))] += 1

    episode_summaries.sort(key=lambda item: (item.get("session_id"), item.get("episode_id")))
    dump_jsonl(output_root / "episode_manifest.jsonl", episode_summaries)
    index_payload = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "level": "canonical",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "episode_count": len(episode_summaries),
        "frame_count": total_frames,
        "selected_frame_count": total_selected_frames,
        "session_counts": dict(sorted(session_counts.items())),
        "instruction_counts": dict(sorted(instruction_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_counts": dict(sorted(quality_counts.items())),
        "state_fields": list(CANONICAL_STATE_FIELDS),
        "state_extra_fields": list(CANONICAL_STATE_EXTRA_FIELDS),
        "action_fields": list(PRIMARY_ACTION_FIELDS),
        "action_full_fields": list(FULL_ACTION_FIELDS),
        "field_sources": dict(FIELD_SOURCE_DESCRIPTION),
    }
    dump_json(output_root / "index.json", index_payload)
    return index_payload


def load_canonical_manifest(view_root: Path) -> List[Dict[str, Any]]:
    manifest_path = view_root / "episode_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"canonical manifest not found: {manifest_path}")
    return load_jsonl(manifest_path)


def load_canonical_episode(view_root: Path, session_id: str, episode_id: str) -> List[Dict[str, Any]]:
    return load_jsonl(canonical_episode_path(view_root, session_id, episode_id))


def write_canonical_episode(view_root: Path, session_id: str, episode_id: str, frames: Sequence[Dict[str, Any]]) -> None:
    dump_jsonl(canonical_episode_path(view_root, session_id, episode_id), frames)


def resolved_goal_distance(record: Dict[str, Any]) -> Optional[float]:
    if record.get("goal_relative_distance") is not None:
        return _safe_float(record.get("goal_relative_distance"))
    if record.get("proxy_goal_relative_distance") is not None:
        return _safe_float(record.get("proxy_goal_relative_distance"))
    return None


def resolved_goal_angle(record: Dict[str, Any]) -> Optional[float]:
    if record.get("goal_relative_angle") is not None:
        return _safe_float(record.get("goal_relative_angle"))
    if record.get("proxy_goal_relative_angle") is not None:
        return _safe_float(record.get("proxy_goal_relative_angle"))
    return None


def _recent_sign_flips(values: Sequence[float], threshold: float) -> int:
    filtered = [0 if abs(value) < threshold else (1 if value > 0 else -1) for value in values]
    filtered = [value for value in filtered if value != 0]
    flips = 0
    for left, right in zip(filtered, filtered[1:]):
        if left != right:
            flips += 1
    return flips


def classify_phase(
    records: Sequence[Dict[str, Any]],
    index: int,
    phase_config: Dict[str, Any],
    *,
    live_mode: bool = False,
) -> Tuple[str, str]:
    thresholds = merge_config(phase_defaults().get("thresholds", {}), phase_config.get("thresholds", {}))
    record = records[index]
    vx = _safe_float((record.get("action_dict") or {}).get("vx", record.get("action", [0.0, 0.0])[0] if record.get("action") else 0.0))
    wz = _safe_float((record.get("action_dict") or {}).get("wz", record.get("action", [0.0, 0.0])[1] if record.get("action") else 0.0))
    goal_distance = resolved_goal_distance(record)
    goal_angle = resolved_goal_angle(record)
    prev_goal_distance = resolved_goal_distance(records[index - 1]) if index > 0 else goal_distance
    prev_goal_angle = resolved_goal_angle(records[index - 1]) if index > 0 else goal_angle
    distance_delta = None
    angle_delta = None
    if goal_distance is not None and prev_goal_distance is not None:
        distance_delta = prev_goal_distance - goal_distance
    if goal_angle is not None and prev_goal_angle is not None:
        angle_delta = abs(prev_goal_angle) - abs(goal_angle)

    total = len(records)
    tail_count = max(1, int(math.ceil(total * _safe_float(thresholds.get("stop_tail_ratio"), 0.12))))
    near_tail = index >= max(0, total - tail_count)
    moving = abs(vx) >= _safe_float(thresholds.get("forward_vx"), 0.10) or abs(wz) >= _safe_float(thresholds.get("turn_wz"), 0.18)
    stop_like = abs(vx) <= _safe_float(thresholds.get("idle_vx"), 0.03) and abs(wz) <= _safe_float(thresholds.get("idle_wz"), 0.06)
    near_goal = goal_distance is not None and goal_distance <= _safe_float(thresholds.get("approach_distance"), 0.9)
    stop_goal = goal_distance is not None and goal_distance <= _safe_float(thresholds.get("stop_distance"), 0.45)
    start_frames = max(1, _safe_int(thresholds.get("start_frame_count"), 6))
    recent_wz = [
        _safe_float((item.get("action_dict") or {}).get("wz", item.get("action", [0.0, 0.0])[1] if item.get("action") else 0.0))
        for item in records[max(0, index - _safe_int(thresholds.get("recovery_window"), 6) + 1): index + 1]
    ]
    sign_flips = _recent_sign_flips(recent_wz, _safe_float(thresholds.get("correction_wz"), 0.10))

    failure_like = record.get("collision") is True or record.get("success") is False
    if failure_like and near_tail:
        return "RECOVERY", "failure_tail"
    if vx <= _safe_float(thresholds.get("reverse_vx"), -0.04):
        return "RECOVERY", "reverse_motion"
    if sign_flips >= _safe_int(thresholds.get("recovery_sign_flips"), 3) and abs(wz) >= _safe_float(thresholds.get("recovery_turn"), 0.20):
        return "RECOVERY", "turn_oscillation"
    if stop_like and (stop_goal or near_tail):
        return "STOP", "slow_near_goal_or_tail"
    if near_goal and abs(wz) >= _safe_float(thresholds.get("align_wz"), 0.18) and abs(vx) <= _safe_float(thresholds.get("align_vx"), 0.08):
        if goal_angle is None or abs(goal_angle) >= _safe_float(thresholds.get("align_angle"), 0.22):
            return "ALIGN", "turning_near_goal"
    if near_goal and vx >= _safe_float(thresholds.get("approach_vx"), 0.08):
        if distance_delta is None or distance_delta >= _safe_float(thresholds.get("approach_distance_delta"), 0.005):
            return "APPROACH", "forward_near_goal"
    if moving and index < start_frames:
        return "START", "early_motion"
    if abs(wz) >= _safe_float(thresholds.get("turn_wz"), 0.18) and abs(vx) <= _safe_float(thresholds.get("correction_vx"), 0.05):
        return ("TURN_LEFT", "turn_in_place") if wz > 0 else ("TURN_RIGHT", "turn_in_place")
    if abs(wz) >= _safe_float(thresholds.get("correction_wz"), 0.10) and abs(vx) >= _safe_float(thresholds.get("correction_vx"), 0.05):
        return "CORRECTION", "move_and_turn"
    if vx >= _safe_float(thresholds.get("forward_vx"), 0.10):
        if near_goal and (distance_delta is None or distance_delta >= 0.0):
            return "APPROACH", "forward_progress"
        return "FORWARD", "forward_motion"
    if stop_like:
        return ("STOP", "low_motion") if live_mode else (("STOP", "low_motion_tail") if near_tail else ("IDLE", "low_motion"))
    return "CORRECTION", "fallback"


def annotate_episode_records(records: Sequence[Dict[str, Any]], phase_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    phase_counter: Counter[str] = Counter()
    segments: List[Dict[str, Any]] = []
    last_phase = None
    segment_start = 0
    for index in range(len(records)):
        phase, reason = classify_phase(records, index, phase_config)
        updated = dict(records[index])
        updated["phase"] = phase
        updated["phase_reason"] = reason
        updated["phase_priority"] = (phase_config.get("priority") or DEFAULT_PHASE_PRIORITY).index(phase) if phase in (phase_config.get("priority") or DEFAULT_PHASE_PRIORITY) else len(DEFAULT_PHASE_PRIORITY)
        annotated.append(updated)
        phase_counter[phase] += 1
        if phase != last_phase:
            if last_phase is not None:
                segments.append(
                    {
                        "phase": last_phase,
                        "start_frame": segment_start,
                        "end_frame": index - 1,
                        "length": index - segment_start,
                    }
                )
            last_phase = phase
            segment_start = index
    if last_phase is not None:
        segments.append(
            {
                "phase": last_phase,
                "start_frame": segment_start,
                "end_frame": len(records) - 1,
                "length": len(records) - segment_start,
            }
        )
    stats = {
        "phase_counts": dict(sorted(phase_counter.items(), key=lambda item: (-item[1], item[0]))),
        "phase_segments": segments,
        "segment_count": len(segments),
    }
    return annotated, stats


def derive_instruction_from_phase(phase: Optional[str]) -> Optional[str]:
    if phase is None:
        return None
    return PHASE_TO_INSTRUCTION.get(phase)


def apply_phase_instruction(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    derived_counter: Counter[str] = Counter()
    updated_records: List[Dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        derived_instruction = derive_instruction_from_phase(record.get("phase"))
        updated["phase_derived_instruction"] = derived_instruction
        updated_records.append(updated)
        if derived_instruction:
            derived_counter[derived_instruction] += 1
    return updated_records, {
        "phase_derived_instruction_counts": dict(sorted(derived_counter.items(), key=lambda item: (-item[1], item[0]))),
        "non_empty_phase_derived_instruction_count": sum(derived_counter.values()),
    }


def summarize_contiguous_segments(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"segment_count": 0, "segments": [], "phase_segment_counts": {}}
    segments: List[Dict[str, Any]] = []
    phase_segment_counts: Counter[str] = Counter()
    start = 0
    current = records[0].get("phase")
    for index, record in enumerate(records[1:], start=1):
        phase = record.get("phase")
        if phase == current:
            continue
        segments.append({"phase": current, "start_frame": start, "end_frame": index - 1, "length": index - start})
        phase_segment_counts[_safe_str(current)] += 1
        start = index
        current = phase
    segments.append({"phase": current, "start_frame": start, "end_frame": len(records) - 1, "length": len(records) - start})
    phase_segment_counts[_safe_str(current)] += 1
    return {
        "segment_count": len(segments),
        "segments": segments,
        "phase_segment_counts": dict(sorted(phase_segment_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def filter_episode_records(records: Sequence[Dict[str, Any]], filter_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = merge_config(filter_defaults(), filter_config)
    keep_every_n = max(1, _safe_int(config.get("keep_every_n"), 6))
    dense_every_n = max(1, _safe_int(config.get("dense_keep_every_n_near_goal"), 2))
    high_value_phases = set(config.get("high_value_phases") or DEFAULT_HIGH_VALUE_PHASES)
    kept: List[Dict[str, Any]] = []
    phase_before: Counter[str] = Counter(_safe_str(record.get("phase")) or "UNLABELED" for record in records)
    phase_after: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    last_kept_index = -10**9

    for index, record in enumerate(records):
        reasons: List[str] = []
        vx = _safe_float((record.get("action_dict") or {}).get("vx", record.get("action", [0.0, 0.0])[0] if record.get("action") else 0.0))
        wz = _safe_float((record.get("action_dict") or {}).get("wz", record.get("action", [0.0, 0.0])[1] if record.get("action") else 0.0))
        prev = records[index - 1] if index > 0 else None
        prev_vx = _safe_float((prev.get("action_dict") or {}).get("vx")) if prev else vx
        prev_wz = _safe_float((prev.get("action_dict") or {}).get("wz")) if prev else wz
        goal_angle = resolved_goal_angle(record)
        prev_goal_angle = resolved_goal_angle(prev) if prev else goal_angle
        goal_distance = resolved_goal_distance(record)
        phase = _safe_str(record.get("phase")) or "UNLABELED"

        if config.get("force_keep_first_last", True) and index in {0, len(records) - 1}:
            reasons.append("endpoint")
        if phase in high_value_phases:
            reasons.append(f"phase:{phase}")
        if abs(wz) >= _safe_float(config.get("turn_threshold"), 0.18):
            reasons.append("turn")
        if abs(vx - prev_vx) >= _safe_float(config.get("accel_threshold"), 0.08) or abs(wz - prev_wz) >= _safe_float(config.get("wz_accel_threshold"), 0.10):
            reasons.append("accel_change")
        if goal_angle is not None and prev_goal_angle is not None and abs(goal_angle - prev_goal_angle) >= _safe_float(config.get("goal_angle_delta_threshold"), 0.12):
            reasons.append("goal_angle_change")
        if goal_distance is not None and goal_distance <= _safe_float(config.get("goal_distance_threshold"), 0.9):
            reasons.append("near_goal")
        periodic_gap = dense_every_n if goal_distance is not None and goal_distance <= _safe_float(config.get("goal_distance_threshold"), 0.9) else keep_every_n
        if index - last_kept_index >= periodic_gap:
            reasons.append("periodic")

        if not reasons:
            continue
        kept_record = dict(record)
        kept_record["filter_keep_reasons"] = sorted(set(reasons))
        kept_record["filter_score"] = float(len(set(reasons)))
        kept.append(kept_record)
        last_kept_index = index
        phase_after[phase] += 1
        for reason in set(reasons):
            reason_counts[reason] += 1

    stats = {
        "input_frame_count": len(records),
        "output_frame_count": len(kept),
        "retention_ratio": (len(kept) / len(records)) if records else 0.0,
        "phase_before": dict(sorted(phase_before.items(), key=lambda item: (-item[1], item[0]))),
        "phase_after": dict(sorted(phase_after.items(), key=lambda item: (-item[1], item[0]))),
        "phase_retention_ratio": {
            phase: (phase_after.get(phase, 0) / count if count else 0.0)
            for phase, count in phase_before.items()
        },
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
    }
    return kept, stats


def dominant_phase(phase_seq: Sequence[Optional[str]]) -> Optional[str]:
    counter: Counter[str] = Counter(_safe_str(phase) for phase in phase_seq if _safe_str(phase))
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _pad_action_chunk(actions: List[List[float]], mask: List[int], chunk_len: int, pad_strategy: str) -> Tuple[List[List[float]], List[int]]:
    if not actions:
        actions = [[0.0, 0.0]]
        mask = [0]
    while len(actions) < chunk_len:
        if pad_strategy == "zero":
            actions.append([0.0 for _ in actions[0]])
        else:
            actions.append(list(actions[-1]))
        mask.append(0)
    return actions[:chunk_len], mask[:chunk_len]


def slice_episode_windows(records: Sequence[Dict[str, Any]], window_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = merge_config(window_defaults(), window_config)
    chunk_len = max(1, _safe_int(config.get("chunk_len"), 4))
    stride = max(1, _safe_int(config.get("stride"), 1))
    action_fields = list(config.get("action_fields") or PRIMARY_ACTION_FIELDS)
    pad_strategy = _safe_str(config.get("pad_strategy") or "repeat_last") or "repeat_last"
    windows: List[Dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()

    for start in range(0, len(records), stride):
        slice_records = list(records[start:start + chunk_len])
        if not slice_records:
            continue
        actions: List[List[float]] = []
        action_mask: List[int] = []
        phase_seq: List[Optional[str]] = []
        frame_ids: List[int] = []
        timestamps: List[float] = []
        full_actions: List[List[float]] = []
        for record in slice_records:
            action_dict = record.get("action_full_dict") if isinstance(record.get("action_full_dict"), dict) else record.get("action_dict") or {}
            actions.append([_safe_float(action_dict.get(field)) for field in action_fields])
            full_actions.append([_safe_float((record.get("action_full_dict") or {}).get(field)) for field in FULL_ACTION_FIELDS])
            action_mask.append(1)
            phase_seq.append(record.get("phase"))
            frame_ids.append(_safe_int(record.get("frame_id")))
            timestamps.append(_safe_float(record.get("timestamp")))
        actions, action_mask = _pad_action_chunk(actions, action_mask, chunk_len, pad_strategy)
        while len(phase_seq) < chunk_len:
            phase_seq.append(phase_seq[-1] if phase_seq else None)
            frame_ids.append(frame_ids[-1] if frame_ids else 0)
            timestamps.append(timestamps[-1] if timestamps else 0.0)
            full_actions.append(list(full_actions[-1]) if full_actions else [0.0, 0.0, 0.0])
        dominant = dominant_phase(phase_seq)
        if dominant:
            phase_counts[dominant] += 1
        first = slice_records[0]
        dominant_phase_instruction = derive_instruction_from_phase(dominant)
        sample_id = f"{_safe_str(first.get('session_id'))}:{_safe_str(first.get('episode_id'))}:{_safe_int(first.get('frame_id')):06d}"
        window = {
            "schema_version": WINDOW_SCHEMA_VERSION,
            "level": "training_windows",
            "sample_id": sample_id,
            "session_id": first.get("session_id"),
            "episode_id": first.get("episode_id"),
            "start_frame": _safe_int(first.get("frame_id")),
            "frame_ids": frame_ids[:chunk_len],
            "timestamps": timestamps[:chunk_len],
            "image_path": first.get("image_path") or "",
            "source_image_path": first.get("source_image_path") or "",
            "raw_instruction": first.get("raw_instruction") or first.get("instruction") or "",
            "instruction": first.get("instruction") or "",
            "normalized_instruction": first.get("normalized_instruction") or "",
            "phase_derived_instruction": dominant_phase_instruction or first.get("phase_derived_instruction"),
            "state": list(first.get("state") or []),
            "state_dict": dict(first.get("state_dict") or {}),
            "previous_action": dict(first.get("previous_action") or {}),
            "prev_action_valid": bool(first.get("prev_action_valid", False)),
            "prev_action_source": first.get("prev_action_source") or "",
            "prev_action_reason": first.get("prev_action_reason") or "",
            "actions_continuous": actions,
            "actions_continuous_full": full_actions[:chunk_len],
            "action_fields": action_fields,
            "action_full_fields": list(FULL_ACTION_FIELDS),
            "action_mask": action_mask,
            "phase": dominant,
            "phase_seq": phase_seq[:chunk_len],
            "success": first.get("success"),
            "success_label": first.get("success_label"),
            "task_family": first.get("task_family") or "",
            "target_type": first.get("target_type") or "",
            "target_label": first.get("target_label") or "",
            "target_description": first.get("target_description") or "",
            "turn_bucket": first.get("turn_bucket") or _turn_bucket_from_actions(actions),
            "visibility_bucket": first.get("visibility_bucket") or "unknown",
            "target_visible_first_frame": first.get("target_visible_first_frame"),
            "target_visible_within_3f": first.get("target_visible_within_3f"),
            "target_visible_within_10f": first.get("target_visible_within_10f"),
            "target_pixel_ratio_first": first.get("target_pixel_ratio_first"),
            "scene_kind": first.get("scene_kind") or "",
            "scene_mode": first.get("scene_mode") or "",
            "scene_id": first.get("scene_id") or "",
            "operator_id": first.get("operator_id") or "",
            "source": dict(first.get("source") or {}) if isinstance(first.get("source"), dict) else {},
            "source_dataset": first.get("source_dataset") or "",
        }
        windows.append(window)

    stats = {
        "sample_count": len(windows),
        "phase_counts": dict(sorted(phase_counts.items(), key=lambda item: (-item[1], item[0]))),
        "chunk_len": chunk_len,
        "stride": stride,
    }
    return windows, stats


def write_windows_view(output_root: Path, windows: Sequence[Dict[str, Any]], stats: Dict[str, Any], extra_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_root / "windows.jsonl", windows)
    final_stats = dict(stats)
    if extra_stats:
        final_stats.update(extra_stats)
    dump_json(output_root / "stats.json", final_stats)
    return final_stats


def _ratio_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    if total <= 0:
        return {"train": 0, "val": 0, "test": 0}
    ratios = [max(0.0, train_ratio), max(0.0, val_ratio), max(0.0, test_ratio)]
    if sum(ratios) <= 0.0:
        ratios = [1.0, 0.0, 0.0]
    ratio_sum = sum(ratios)
    ratios = [value / ratio_sum for value in ratios]
    counts = {
        "train": int(total * ratios[0]),
        "val": int(total * ratios[1]),
        "test": total - int(total * ratios[0]) - int(total * ratios[1]),
    }
    if total >= 3:
        if ratios[0] > 0.0 and counts["train"] == 0:
            counts["train"] = 1
        if ratios[1] > 0.0 and counts["val"] == 0:
            counts["val"] = 1
        if ratios[2] > 0.0 and counts["test"] == 0:
            counts["test"] = 1
    while sum(counts.values()) > total:
        for name in ("train", "val", "test"):
            if counts[name] > 0 and sum(counts.values()) > total:
                counts[name] -= 1
    while sum(counts.values()) < total:
        counts["train"] += 1
    return counts


def assign_window_splits(records: Sequence[Dict[str, Any]], split_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    config = merge_config(export_defaults().get("split", {}), split_config)
    mode = _safe_str(config.get("mode") or "by_episode")
    train_ratio = _safe_float(config.get("train_ratio"), 0.8)
    val_ratio = _safe_float(config.get("val_ratio"), 0.1)
    test_ratio = _safe_float(config.get("test_ratio"), 0.1)
    seed = config.get("seed")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if mode == "by_session":
            key = _safe_str(record.get("session_id"))
        else:
            key = f"{_safe_str(record.get('session_id'))}:{_safe_str(record.get('episode_id'))}"
        groups.setdefault(key, []).append(record)

    grouped_items = [groups[key] for key in sorted(groups)]
    if seed is not None and len(grouped_items) > 1:
        rng = random.Random(_safe_int(seed))
        rng.shuffle(grouped_items)
    counts = _ratio_counts(len(grouped_items), train_ratio, val_ratio, test_ratio)
    split_names: List[str] = []
    for name in ("train", "val", "test"):
        split_names.extend([name] * counts[name])
    while len(split_names) < len(grouped_items):
        split_names.append("train")
    split_records = {"train": [], "val": [], "test": []}
    for group, split_name in zip(grouped_items, split_names):
        split_records[split_name].extend(group)
    return split_records


def select_instruction(record: Dict[str, Any], instruction_config: Dict[str, Any]) -> str:
    config = merge_config(export_defaults().get("instruction_policy", {}), instruction_config)
    raw_instruction = _safe_str(record.get("raw_instruction") or record.get("instruction"))
    normalized_instruction = _safe_str(record.get("normalized_instruction") or raw_instruction)
    phase_instruction = _safe_str(record.get("phase_derived_instruction"))
    mode = _safe_str(config.get("mode") or "raw")
    if mode == "semantic_auto":
        candidates = _semantic_instruction_candidates(record)
        if candidates:
            sample_key = _safe_str(record.get("sample_id") or record.get("episode_id"))
            digest = hashlib.blake2b(sample_key.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % len(candidates)
            return candidates[index]
        return raw_instruction or normalized_instruction or phase_instruction
    if mode == "phase":
        return phase_instruction or raw_instruction or normalized_instruction
    if mode == "concat":
        if raw_instruction and phase_instruction:
            return f"{raw_instruction} | {phase_instruction}"
        return raw_instruction or phase_instruction or normalized_instruction
    if mode == "mixed":
        if phase_instruction:
            sample_key = _safe_str(record.get("sample_id") or record.get("episode_id"))
            digest = hashlib.blake2b(sample_key.encode("utf-8"), digest_size=8).digest()
            value = (int.from_bytes(digest, "little") % 1000) / 1000.0
            if value < _safe_float(config.get("mix_phase_probability"), 0.5):
                return phase_instruction
        return raw_instruction or normalized_instruction
    return raw_instruction or normalized_instruction or phase_instruction


def export_route_record(record: Dict[str, Any], route_name: str, instruction_config: Dict[str, Any]) -> Dict[str, Any]:
    selected_instruction = select_instruction(record, instruction_config)
    base = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "route": route_name,
        "sample_id": record.get("sample_id"),
        "session_id": record.get("session_id"),
        "episode_id": record.get("episode_id"),
        "start_frame": record.get("start_frame"),
        "image_path": record.get("image_path"),
        "source_image_path": record.get("source_image_path"),
        "instruction": selected_instruction,
        "raw_instruction": record.get("raw_instruction"),
        "normalized_instruction": record.get("normalized_instruction"),
        "phase_derived_instruction": record.get("phase_derived_instruction"),
        "state": record.get("state"),
        "state_dict": record.get("state_dict"),
        "previous_action": record.get("previous_action"),
        "prev_action_valid": record.get("prev_action_valid"),
        "prev_action_source": record.get("prev_action_source"),
        "prev_action_reason": record.get("prev_action_reason"),
        "actions_continuous": record.get("actions_continuous"),
        "action_mask": record.get("action_mask"),
        "phase": record.get("phase"),
        "phase_seq": record.get("phase_seq"),
        "action_fields": record.get("action_fields") or PRIMARY_ACTION_FIELDS,
        "task_family": record.get("task_family") or "",
        "target_type": record.get("target_type") or "",
        "target_label": record.get("target_label") or "",
        "target_description": record.get("target_description") or "",
        "turn_bucket": record.get("turn_bucket") or "",
        "visibility_bucket": record.get("visibility_bucket") or "",
        "target_visible_first_frame": record.get("target_visible_first_frame"),
        "target_visible_within_3f": record.get("target_visible_within_3f"),
        "target_visible_within_10f": record.get("target_visible_within_10f"),
        "target_pixel_ratio_first": record.get("target_pixel_ratio_first"),
        "scene_kind": record.get("scene_kind") or "",
        "scene_mode": record.get("scene_mode") or "",
        "scene_id": record.get("scene_id") or "",
        "operator_id": record.get("operator_id") or "",
        "source_dataset": record.get("source_dataset") or "",
        "source": dict(((record.get("source") or {}) if isinstance(record.get("source"), dict) else {}), source_type=_safe_str(((record.get("source") or {}) if isinstance(record.get("source"), dict) else {}).get("source_type"))),
        "success": record.get("success"),
        "success_label": record.get("success_label"),
    }
    if route_name == "A":
        base["export_view"] = "continuous_control"
    else:
        base["export_view"] = "tokenizer_ready_vla"
        base["actions_continuous_full"] = record.get("actions_continuous_full")
        base["action_full_fields"] = record.get("action_full_fields") or FULL_ACTION_FIELDS
        base["tokenizer_ready"] = {
            "chunk_len": len(record.get("actions_continuous") or []),
            "action_fields": record.get("action_fields") or PRIMARY_ACTION_FIELDS,
            "action_full_fields": record.get("action_full_fields") or FULL_ACTION_FIELDS,
        }
    return base


def export_training_views(
    windows: Sequence[Dict[str, Any]],
    output_root: Path,
    export_config: Dict[str, Any],
) -> Dict[str, Any]:
    config = merge_config(export_defaults(), export_config)
    split_records = assign_window_splits(windows, config.get("split") or {})
    output_root.mkdir(parents=True, exist_ok=True)
    overall_stats: Dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "routes": {},
        "split_config": config.get("split") or {},
        "instruction_policy": config.get("instruction_policy") or {},
    }
    for route_name in config.get("routes") or ["A", "B"]:
        route_dir = output_root / f"route_{str(route_name).lower()}"
        route_dir.mkdir(parents=True, exist_ok=True)
        route_stats = {
            "sample_count": 0,
            "split_counts": {},
            "phase_counts": Counter(),
            "instruction_counts": Counter(),
        }
        all_exported: List[Dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            exported = [export_route_record(record, str(route_name), config.get("instruction_policy") or {}) for record in split_records[split_name]]
            dump_jsonl(route_dir / f"{split_name}.jsonl", exported)
            route_stats["split_counts"][split_name] = len(exported)
            route_stats["sample_count"] += len(exported)
            for item in exported:
                route_stats["phase_counts"][str(item.get("phase") or "UNLABELED")] += 1
                route_stats["instruction_counts"][str(item.get("instruction") or "")] += 1
            all_exported.extend(exported)
        dump_jsonl(route_dir / "dataset.jsonl", all_exported)
        route_stats["phase_counts"] = dict(sorted(route_stats["phase_counts"].items(), key=lambda item: (-item[1], item[0])))
        route_stats["instruction_counts"] = dict(sorted(route_stats["instruction_counts"].items(), key=lambda item: (-item[1], item[0])))
        dump_json(route_dir / "stats.json", route_stats)
        overall_stats["routes"][str(route_name)] = route_stats
    dump_json(output_root / "stats.json", overall_stats)
    return overall_stats


def summarize_instruction_distribution(records: Sequence[Dict[str, Any]], field: str = "instruction") -> Dict[str, int]:
    counter = Counter(_safe_str(record.get(field)) for record in records if _safe_str(record.get(field)))
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def summarize_phase_distribution(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(_safe_str(record.get("phase")) or "UNLABELED" for record in records)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def summarize_window_integrity(windows: Sequence[Dict[str, Any]], expected_chunk_len: int = 4) -> Dict[str, Any]:
    bad_chunk_len = 0
    bad_mask_len = 0
    bad_state = 0
    for record in windows:
        actions = list(record.get("actions_continuous") or [])
        mask = list(record.get("action_mask") or [])
        state = list(record.get("state") or [])
        if len(actions) != expected_chunk_len:
            bad_chunk_len += 1
        if len(mask) != expected_chunk_len:
            bad_mask_len += 1
        if len(state) != len(CANONICAL_STATE_FIELDS):
            bad_state += 1
    return {
        "window_count": len(windows),
        "bad_chunk_len": bad_chunk_len,
        "bad_mask_len": bad_mask_len,
        "bad_state_len": bad_state,
    }


def record_motion_summary(vx_values: Sequence[float], wz_values: Sequence[float]) -> Dict[str, float]:
    if not vx_values or not wz_values:
        return {
            "vx_mean": 0.0,
            "vx_min": 0.0,
            "vx_max": 0.0,
            "wz_mean": 0.0,
            "wz_min": 0.0,
            "wz_max": 0.0,
        }
    return {
        "vx_mean": float(statistics.mean(vx_values)),
        "vx_min": float(min(vx_values)),
        "vx_max": float(max(vx_values)),
        "wz_mean": float(statistics.mean(wz_values)),
        "wz_min": float(min(wz_values)),
        "wz_max": float(max(wz_values)),
    }


def classify_live_status(status: Dict[str, Any], history: Sequence[Dict[str, Any]], live_config: Dict[str, Any]) -> Tuple[str, str, bool]:
    config = merge_config(live_monitor_defaults(), live_config)
    command = status.get("command") if isinstance(status.get("command"), dict) else {}
    collector = status.get("collector") if isinstance(status.get("collector"), dict) else {}
    vx = _safe_float(command.get("vx"))
    wz = _safe_float(command.get("wz"))
    recording = bool(collector.get("recording"))
    synthetic = {
        "action_dict": {"vx": vx, "wz": wz},
        "action": [vx, wz],
        "success": None,
        "collision": collector.get("fault_reason") not in {None, ""},
    }
    synthetic_history = list(history) + [synthetic]
    phase, reason = classify_phase(
        synthetic_history,
        len(synthetic_history) - 1,
        {
            "thresholds": {
                "forward_vx": config.get("forward_threshold"),
                "turn_wz": config.get("turn_threshold"),
                "idle_vx": config.get("stop_vx_threshold"),
                "idle_wz": config.get("stop_wz_threshold"),
                "correction_wz": config.get("correction_wz_threshold"),
                "reverse_vx": config.get("recovery_vx_threshold"),
            }
        },
        live_mode=True,
    )
    high_value = phase in set(config.get("high_value_phases") or DEFAULT_HIGH_VALUE_PHASES)
    if not recording and phase == "STOP":
        phase = "IDLE"
    return phase, reason, high_value
