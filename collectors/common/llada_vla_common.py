from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "go2_local_dataset_v1"
QUALITY_REVIEW_SCHEMA_VERSION = "go2_quality_review_v1"
QUALITY_REVIEW_FILENAME = "quality_review.json"
QUALITY_LABEL_EXCLUDED = 3
DEFAULT_MAX_PROCESSING_QUALITY_LABEL = 2
DERIVED_LABELS_DIRNAME = "derived_labels"
FALLBACK_DERIVED_LABELS_DIRNAMES = ["distribution_labels"]
FRAME_SELECTION_MODE_ALL = "all"
FRAME_SELECTION_MODE_RETAIN_ONLY = "retain_only"
STATE_FIELDS = ["vx", "vy", "wz", "yaw"]
STATE_EXTRA_FIELDS = ["vz", "roll", "pitch", "x", "y", "z", "body_height", "error_code", "mode", "gait_type"]
ACTION_FIELDS = ["vx", "vy", "wz"]
RAW_ACTION_FIELDS = ["vx", "vy", "wz", "camera_pitch", "keys"]
CONTROLLED_INSTRUCTIONS = [
    "go forward",
    "move backward",
    "strafe left",
    "strafe right",
    "stand up",
    "lie down",
    "turn left",
    "turn right",
    "stay still",
]
KNOWN_TASK_FAMILIES = [
    "legacy_motion",
    "goal_navigation",
    "visual_following",
    "obstacle_aware_navigation",
]
VISUAL_TASK_FAMILIES = [
    "goal_navigation",
    "visual_following",
    "obstacle_aware_navigation",
]
KNOWN_TARGET_TYPES = [
    "door",
    "apple",
    "red_object",
    "box",
    "person",
    "obstacle",
    "landmark",
    "waypoint",
]
TASK_METADATA_FIELDS = [
    "capture_mode",
    "task_family",
    "target_type",
    "target_label",
    "target_description",
    "collector_notes",
    "instruction_source",
    "segment_status",
    "success",
    "termination_reason",
]
QUALITY_REVIEW_FIELDS = [
    "quality_label",
    "original_quality_label",
    "quality_overridden",
    "quality_notes",
    "exclude_from_processing",
]
FRAME_REVIEW_FIELDS = [
    "frame_selection_mode",
    "selected_frame_count",
    "total_frame_count",
]
CONTRAST_METADATA_FIELDS = [
    "contrast_group_id",
    "contrast_variant",
    "active_target_id",
    "scene_targets",
]
PROCESSED_DATASET_OPTIONAL_FIELDS = [
    "previous_action",
    "prev_action_valid",
    "prev_action_source",
    "prev_action_reason",
] + CONTRAST_METADATA_FIELDS
ACTION_ACTIVITY_EPS = 0.05
PREVIOUS_ACTION_MAX_GAP_FACTOR = 2.5
PREVIOUS_ACTION_MAX_GAP_SECONDS = 0.2
TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")
INSTRUCTION_TEMPLATE_PATTERNS = {
    "goal_navigation": [
        re.compile(r"^go to the [a-z0-9 ]+$"),
        re.compile(r"^go to [a-z0-9 ]+$"),
        re.compile(r"^approach the [a-z0-9 ]+$"),
        re.compile(r"^approach [a-z0-9 ]+$"),
    ],
    "visual_following": [
        re.compile(r"^follow the [a-z0-9 ]+$"),
        re.compile(r"^follow [a-z0-9 ]+$"),
    ],
    "obstacle_aware_navigation": [
        re.compile(r"^go around the [a-z0-9 ]+$"),
        re.compile(r"^go around [a-z0-9 ]+$"),
        re.compile(r"^avoid the [a-z0-9 ]+$"),
        re.compile(r"^avoid [a-z0-9 ]+$"),
    ],
}

QualityReviewIndex = Dict[Tuple[str, str], Dict[str, Any]]
INSTRUCTION_TARGET_PATTERNS = {
    "goal_navigation": [
        re.compile(r"^(?:go to|approach)(?: the)? (?P<label>[a-z0-9 ]+)$"),
    ],
    "visual_following": [
        re.compile(r"^follow(?: the)? (?P<label>[a-z0-9 ]+)$"),
    ],
    "obstacle_aware_navigation": [
        re.compile(r"^(?:go around|avoid)(?: the)? (?P<label>[a-z0-9 ]+?)(?: and [a-z0-9 ]+)?$"),
    ],
}


@dataclass(frozen=True)
class Sample:
    session_id: str
    trajectory_id: str
    trajectory_index: int
    trajectory_step_index: int
    trajectory_length: int
    step_id: int
    timestamp: float
    instruction: str
    image_path: Optional[str]
    source_image_path: Optional[str]
    state: Dict[str, Any]
    raw_action: Dict[str, Any]
    control_action: Dict[str, Any]
    action_chunk: List[Dict[str, Any]]
    raw_record: Dict[str, Any]

    def to_manifest_record(self, dataset_root: Path) -> Dict[str, Any]:
        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"{self.session_id}:{self.trajectory_id}:{self.step_id:06d}",
            "session_id": self.session_id,
            "episode_id": self.trajectory_id,
            "trajectory_id": self.trajectory_id,
            "trajectory_index": self.trajectory_index,
            "trajectory_step_index": self.trajectory_step_index,
            "trajectory_length": self.trajectory_length,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "instruction": self.instruction,
            "state": self.state,
            "raw_action": self.raw_action,
            "control_action": self.control_action,
            "action_chunk": self.action_chunk,
            "chunk_length": len(self.action_chunk),
            "scene_id": str(self.raw_record.get("scene_id") or ""),
            "operator_id": str(self.raw_record.get("operator_id") or ""),
            "dataset_root": str(dataset_root),
        }
        for field in TASK_METADATA_FIELDS:
            value = self.raw_record.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            if isinstance(value, list) and not value:
                continue
            record[field] = value
        for field in QUALITY_REVIEW_FIELDS:
            value = self.raw_record.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            record[field] = value
        for field in FRAME_REVIEW_FIELDS:
            value = self.raw_record.get(field)
            if value is None:
                continue
            record[field] = value
        if self.image_path is not None:
            record["image_path"] = self.image_path
        if self.source_image_path is not None:
            record["source_image_path"] = self.source_image_path
        derived_labels = self.raw_record.get("derived_labels")
        if isinstance(derived_labels, dict) and derived_labels:
            record["derived_labels"] = derived_labels
        frame_processing = self.raw_record.get("frame_processing")
        if isinstance(frame_processing, dict) and frame_processing:
            record["frame_processing"] = frame_processing
        for field in PROCESSED_DATASET_OPTIONAL_FIELDS:
            value = self.raw_record.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            record[field] = value
        return record


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dump_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def quality_review_path(dataset_root: Path) -> Path:
    return dataset_root / QUALITY_REVIEW_FILENAME


def episode_review_key(session_id: str, episode_id: str) -> Tuple[str, str]:
    return (_safe_str(session_id), _safe_str(episode_id))


def load_quality_review_index(dataset_root: Path, review_path: Optional[Path] = None) -> QualityReviewIndex:
    if review_path is None:
        return {}

    candidate_paths: List[Path] = [review_path.resolve()]

    for path in candidate_paths:
        if not path.exists():
            continue
        payload = load_json(path)
        episodes = payload.get("episodes") if isinstance(payload, dict) else None
        if not isinstance(episodes, list):
            return {}
        review_index: QualityReviewIndex = {}
        for item in episodes:
            if not isinstance(item, dict):
                continue
            session_id, episode_id = episode_review_key(item.get("session_id"), item.get("episode_id"))
            if not session_id or not episode_id:
                continue
            quality_label = max(0, min(3, _safe_int(item.get("quality_label"), 0)))
            review_payload: Dict[str, Any] = {
                "quality_label": quality_label,
                "exclude_from_processing": bool(item.get("exclude_from_processing")) or quality_label == QUALITY_LABEL_EXCLUDED,
            }
            quality_notes = _safe_str(item.get("quality_notes") or item.get("notes"))
            if quality_notes:
                review_payload["quality_notes"] = quality_notes
            review_index[(session_id, episode_id)] = review_payload
        return review_index
    return {}


def quality_review_for_episode(
    review_index: Optional[QualityReviewIndex],
    session_id: str,
    episode_id: str,
) -> Dict[str, Any]:
    if not review_index:
        return {}
    return dict(review_index.get(episode_review_key(session_id, episode_id)) or {})


def should_exclude_episode_from_processing(
    review_index: Optional[QualityReviewIndex],
    session_id: str,
    episode_id: str,
) -> bool:
    review = quality_review_for_episode(review_index, session_id, episode_id)
    return bool(review.get("exclude_from_processing"))


def infer_episode_quality_label(payload: Dict[str, Any], episode_meta: Dict[str, Any]) -> int:
    task_block = payload.get("task")
    if not isinstance(task_block, dict):
        task_block = {}
    segment_status = _safe_str(
        payload.get("segment_status") or
        episode_meta.get("segment_status") or
        task_block.get("segment_status")
    ).lower()
    success = _safe_str(
        payload.get("success") or
        episode_meta.get("success") or
        task_block.get("success")
    ).lower()
    termination_reason = _safe_str(
        payload.get("termination_reason") or
        episode_meta.get("termination_reason") or
        task_block.get("termination_reason")
    ).lower()

    if segment_status == "discard" or termination_reason == "discard":
        return 4
    if segment_status == "clean" and success == "success":
        return 1
    if segment_status == "usable" and success == "partial":
        return 2
    if success == "fail":
        return 3
    if segment_status == "usable" and termination_reason in {"operator_stop", "timeout", "failed"}:
        return 3
    return 0


def resolved_episode_quality(
    payload: Dict[str, Any],
    episode_meta: Dict[str, Any],
    review_index: Optional[QualityReviewIndex],
    session_id: str,
    episode_id: str,
) -> Dict[str, Any]:
    original_quality_label = infer_episode_quality_label(payload, episode_meta)
    review = quality_review_for_episode(review_index, session_id, episode_id)
    override_quality_label = max(0, min(3, _safe_int(review.get("quality_label"), 0)))
    quality_label = override_quality_label if override_quality_label > 0 else original_quality_label
    quality_notes = _safe_str(review.get("quality_notes"))
    quality_overridden = override_quality_label > 0 and override_quality_label != original_quality_label
    exclude_from_processing = (
        bool(review.get("exclude_from_processing")) or quality_label == QUALITY_LABEL_EXCLUDED
    )
    return {
        "original_quality_label": original_quality_label,
        "quality_label": quality_label,
        "quality_overridden": quality_overridden,
        "quality_notes": quality_notes,
        "exclude_from_processing": exclude_from_processing,
    }


def derived_labels_path(session_root: Path, episode_id: str) -> Path:
    return session_root / DERIVED_LABELS_DIRNAME / f"{episode_id}.json"


def load_episode_derived_labels(session_root: Path, episode_id: str) -> Dict[str, Any]:
    if not episode_id:
        return {}
    candidate_paths = [derived_labels_path(session_root, episode_id)]
    for dirname in FALLBACK_DERIVED_LABELS_DIRNAMES:
        candidate_paths.append(session_root / dirname / f"{episode_id}.json")
    for path in candidate_paths:
        if not path.exists():
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            return payload
    return {}


def discover_session_roots(raw_root: Path) -> List[Path]:
    if raw_root.exists() and (raw_root / "index.json").exists() and (raw_root / "episodes").is_dir():
        return [raw_root]

    roots: List[Path] = []
    for child in sorted(raw_root.iterdir() if raw_root.exists() else []):
        if child.is_dir() and (child / "index.json").exists() and (child / "episodes").is_dir():
            roots.append(child)
    return roots


def ensure_session_materialized(output_root: Path, session_root: Path, session_id: str) -> Path:
    sessions_dir = output_root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target = sessions_dir / session_id
    if target.exists() or target.is_symlink():
        return target

    source = session_root.resolve()
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)
    return target


def has_converted_manifests(dataset_root: Path) -> bool:
    return (dataset_root / "dataset.jsonl").exists()


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


def normalize_tags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _normalize_target_label(value: str) -> str:
    return " ".join(TEXT_TOKEN_RE.findall(_safe_str(value).lower())).strip()


def is_controlled_instruction(instruction: str) -> bool:
    return instruction in CONTROLLED_INSTRUCTIONS


def infer_task_family(instruction: str, explicit_task_family: str = "") -> str:
    task_family = _safe_str(explicit_task_family)
    if task_family:
        return task_family
    normalized = _safe_str(instruction).lower()
    if normalized in CONTROLLED_INSTRUCTIONS:
        return "legacy_motion"
    for family, patterns in INSTRUCTION_TEMPLATE_PATTERNS.items():
        if any(pattern.fullmatch(normalized) for pattern in patterns):
            return family
    return ""


def infer_target_label(
    instruction: str,
    explicit_task_family: str = "",
    explicit_target_type: str = "",
    explicit_target_description: str = "",
) -> str:
    normalized_instruction = _safe_str(instruction).lower()
    task_family = infer_task_family(normalized_instruction, explicit_task_family)
    patterns = INSTRUCTION_TARGET_PATTERNS.get(task_family, [])
    for pattern in patterns:
        match = pattern.fullmatch(normalized_instruction)
        if not match:
            continue
        label = _normalize_target_label(match.group("label"))
        if label:
            return label

    target_type_label = _normalize_target_label(explicit_target_type.replace("_", " "))
    if target_type_label:
        return target_type_label
    return _normalize_target_label(explicit_target_description)


def instruction_matches_known_template(instruction: str, task_family: str = "") -> bool:
    normalized = _safe_str(instruction).lower()
    family = infer_task_family(normalized, task_family)
    if family == "legacy_motion":
        return normalized in CONTROLLED_INSTRUCTIONS
    patterns = INSTRUCTION_TEMPLATE_PATTERNS.get(family, [])
    return any(pattern.fullmatch(normalized) for pattern in patterns)


def episode_task_metadata(payload: Dict[str, Any], episode_meta: Dict[str, Any]) -> Dict[str, Any]:
    task_block = payload.get("task")
    if not isinstance(task_block, dict):
        task_block = {}

    instruction = _safe_str(payload.get("instruction"))
    explicit_task_family = (
        _safe_str(payload.get("task_family")) or
        _safe_str(episode_meta.get("task_family")) or
        _safe_str(task_block.get("task_family"))
    )
    task_family = infer_task_family(instruction, explicit_task_family)
    target_type = (
        _safe_str(payload.get("target_type")) or
        _safe_str(episode_meta.get("target_type")) or
        _safe_str(task_block.get("target_type"))
    )
    target_description = (
        _safe_str(payload.get("target_description")) or
        _safe_str(episode_meta.get("target_description")) or
        _safe_str(task_block.get("target_description"))
    )
    explicit_target_label = (
        _safe_str(payload.get("target_label")) or
        _safe_str(episode_meta.get("target_label")) or
        _safe_str(task_block.get("target_label"))
    )
    return {
        "capture_mode": (
            _safe_str(payload.get("capture_mode")) or
            _safe_str(episode_meta.get("capture_mode")) or
            _safe_str(task_block.get("capture_mode"))
        ),
        "task_family": task_family,
        "target_type": target_type,
        "target_label": explicit_target_label or infer_target_label(
            instruction,
            explicit_task_family=task_family,
            explicit_target_type=target_type,
            explicit_target_description=target_description,
        ),
        "target_description": target_description,
        "collector_notes": (
            _safe_str(payload.get("collector_notes")) or
            _safe_str(episode_meta.get("collector_notes")) or
            _safe_str(task_block.get("collector_notes")) or
            _safe_str(payload.get("notes")) or
            _safe_str(episode_meta.get("notes"))
        ),
        "instruction_source": (
            _safe_str(payload.get("instruction_source")) or
            _safe_str(task_block.get("instruction_source")) or
            ("semantic_text" if task_family and task_family != "legacy_motion" else "motion_label")
        ),
        "segment_status": (
            _safe_str(payload.get("segment_status")) or
            _safe_str(episode_meta.get("segment_status")) or
            _safe_str(task_block.get("segment_status"))
        ),
        "success": (
            _safe_str(payload.get("success")) or
            _safe_str(episode_meta.get("success")) or
            _safe_str(task_block.get("success"))
        ),
        "termination_reason": (
            _safe_str(payload.get("termination_reason")) or
            _safe_str(episode_meta.get("termination_reason")) or
            _safe_str(task_block.get("termination_reason"))
        ),
    }


def action_bucket(
    action: Dict[str, Any],
    epsilon: float = ACTION_ACTIVITY_EPS,
) -> str:
    vx = _safe_float(action.get("vx"))
    vy = _safe_float(action.get("vy"))
    wz = _safe_float(action.get("wz"))
    active: List[str] = []
    if vx >= epsilon:
        active.append("forward")
    elif vx <= -epsilon:
        active.append("backward")
    if vy >= epsilon:
        active.append("strafe_left")
    elif vy <= -epsilon:
        active.append("strafe_right")
    if wz >= epsilon:
        active.append("turn_left")
    elif wz <= -epsilon:
        active.append("turn_right")
    if not active:
        return "stop"
    if len(active) == 1:
        return active[0]
    return "mixed:" + "+".join(active)


def summarize_trajectory_actions(
    actions: Sequence[Dict[str, Any]],
    timestamps: Optional[Sequence[float]] = None,
    epsilon: float = ACTION_ACTIVITY_EPS,
) -> Dict[str, Any]:
    frame_count = len(actions)
    if frame_count == 0:
        return {
            "frame_count": 0,
            "duration_seconds": 0.0,
            "action_change_count": 0,
            "stop_ratio": 0.0,
            "turn_ratio": 0.0,
            "move_ratio": 0.0,
            "distinct_action_bucket_count": 0,
            "action_buckets": [],
        }

    buckets = [action_bucket(action, epsilon=epsilon) for action in actions]
    compressed: List[str] = []
    for bucket in buckets:
        if not compressed or compressed[-1] != bucket:
            compressed.append(bucket)
    change_count = max(0, len(compressed) - 1)
    stop_frames = sum(1 for bucket in buckets if bucket == "stop")
    turn_frames = sum(1 for action in actions if abs(_safe_float(action.get("wz"))) >= epsilon)
    move_frames = sum(1 for bucket in buckets if bucket != "stop")

    duration_seconds = 0.0
    if timestamps and len(timestamps) >= 2:
        ordered = [_safe_float(value) for value in timestamps]
        duration_seconds = max(0.0, ordered[-1] - ordered[0])

    return {
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "action_change_count": change_count,
        "stop_ratio": stop_frames / frame_count,
        "turn_ratio": turn_frames / frame_count,
        "move_ratio": move_frames / frame_count,
        "distinct_action_bucket_count": len(set(buckets)),
        "action_buckets": compressed,
    }


def summarize_trajectory_metric_series(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(statistics.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def action_from_payload(payload: Any, raw: bool = False) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    record: Dict[str, Any] = {field: _safe_float(payload.get(field)) for field in ACTION_FIELDS}
    if raw:
        record["camera_pitch"] = _safe_float(payload.get("camera_pitch"))
        record["keys"] = _safe_int(payload.get("keys"))
    return record


def raw_action_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    payload = frame.get("raw_action")
    if payload is None:
        payload = frame.get("action")
    if payload is None:
        payload = frame.get("control_action")
    return action_from_payload(payload, raw=True)


def control_action_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    payload = frame.get("control_action")
    if payload is None:
        payload = frame.get("action")
    if payload is None:
        payload = frame.get("raw_action")
    return action_from_payload(payload, raw=False)


def previous_action_from_frame(frame: Dict[str, Any]) -> Optional[Dict[str, float]]:
    payload = frame.get("previous_action")
    if payload is None:
        payload = frame.get("prev_action")
    if isinstance(payload, dict) and any(key in payload for key in ACTION_FIELDS):
        return action_from_payload(payload, raw=False)

    state = frame.get("state")
    if isinstance(state, dict) and any(key in state for key in ("vx_prev", "vy_prev", "wz_prev")):
        return {
            "vx": _safe_float(state.get("vx_prev")),
            "vy": _safe_float(state.get("vy_prev")),
            "wz": _safe_float(state.get("wz_prev")),
        }
    return None


def state_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    payload = frame.get("state") or {}
    record = {field: _safe_float(payload.get(field)) for field in STATE_FIELDS}
    for field in STATE_EXTRA_FIELDS:
        record[field] = _safe_float(payload.get(field))
    return record


def control_timestamp_from_frame(frame: Dict[str, Any]) -> float:
    meta = frame.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    for key in (
        "control_action_timestamp",
        "action_timestamp",
        "raw_action_timestamp",
        "state_timestamp",
        "image_timestamp",
    ):
        value = meta.get(key)
        if value is None:
            continue
        timestamp = _safe_float(value, default=math.nan)
        if math.isfinite(timestamp):
            return timestamp
    return _safe_float(frame.get("timestamp"), default=math.nan)


def estimate_nominal_dt(frames: Sequence[Dict[str, Any]]) -> Optional[float]:
    timestamps = [control_timestamp_from_frame(frame) for frame in frames]
    diffs = [
        current - previous
        for previous, current in zip(timestamps[:-1], timestamps[1:])
        if math.isfinite(previous) and math.isfinite(current) and current > previous
    ]
    if not diffs:
        return None
    ordered = sorted(diffs)
    conservative_window = ordered[: max(1, len(ordered) // 2)]
    return float(statistics.median(conservative_window))


def previous_action_info_for_step(
    *,
    frames: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    step_pos: int,
    nominal_dt: Optional[float],
) -> Tuple[Dict[str, float], bool, str, str]:
    zero = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    current_frame = frames[step_pos]

    logged_previous = previous_action_from_frame(current_frame)
    if logged_previous is not None:
        return logged_previous, True, "logged", "logged"

    if step_pos <= 0:
        return zero, False, "warmup_zero", "episode_boundary"

    prev_source_index = int(selected_indices[step_pos - 1])
    current_source_index = int(selected_indices[step_pos])
    if current_source_index != prev_source_index + 1:
        return zero, False, "warmup_zero", "index_gap"

    previous_timestamp = control_timestamp_from_frame(frames[step_pos - 1])
    current_timestamp = control_timestamp_from_frame(current_frame)
    if math.isfinite(previous_timestamp) and math.isfinite(current_timestamp):
        dt = current_timestamp - previous_timestamp
        max_gap = PREVIOUS_ACTION_MAX_GAP_SECONDS
        if nominal_dt is not None and nominal_dt > 0.0:
            max_gap = max(max_gap, PREVIOUS_ACTION_MAX_GAP_FACTOR * nominal_dt)
        if dt <= 0.0 or dt > max_gap:
            return zero, False, "warmup_zero", "timestamp_gap"

    return control_action_from_frame(frames[step_pos - 1]), True, "reconstructed", "reconstructed"


def normalize_frame_selection_mode(value: Any) -> str:
    text = _safe_str(value).lower()
    if text in {"retain_only", "keep_only", "selected_only"}:
        return FRAME_SELECTION_MODE_RETAIN_ONLY
    return FRAME_SELECTION_MODE_ALL


def frame_processing_flag(frame: Dict[str, Any]) -> Optional[bool]:
    meta = frame.get("meta")
    if not isinstance(meta, dict):
        return None
    if "retain_for_processing" in meta:
        return bool(meta.get("retain_for_processing"))
    if "selected_for_processing" in meta:
        return bool(meta.get("selected_for_processing"))
    return None


def episode_frame_processing(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = normalize_frame_selection_mode(
        payload.get("frame_selection_mode") or payload.get("frame_processing_mode")
    )
    keep_map: Dict[int, bool] = {}
    frames = list(payload.get("frames") or [])
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        flag = frame_processing_flag(frame)
        if flag is None:
            continue
        keep_map[frame_index] = flag
    selected_indices = selected_frame_indices(payload, frames=frames, mode=mode, keep_map=keep_map)
    return {
        "frame_selection_mode": mode,
        "frame_keep_map": keep_map,
        "selected_frame_indices": selected_indices,
        "selected_frame_count": len(selected_indices),
        "total_frame_count": len(frames),
    }


def apply_auto_frame_filters(
    payload: Dict[str, Any],
    *,
    selected_indices: Sequence[int],
    drop_initial_frames: int = 0,
    max_success_tail_frames: Optional[int] = None,
) -> List[int]:
    filtered = list(selected_indices)
    if drop_initial_frames > 0:
        filtered = filtered[int(drop_initial_frames):]

    if max_success_tail_frames is not None and max_success_tail_frames >= 0 and filtered:
        frames = list(payload.get("frames") or [])
        first_success_pos: Optional[int] = None
        for position, frame_index in enumerate(filtered):
            if frame_index < 0 or frame_index >= len(frames):
                continue
            frame = frames[frame_index]
            meta = frame.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            success = _safe_str(meta.get("success")).lower()
            termination_reason = _safe_str(meta.get("termination_reason")).lower()
            if success == "success" or termination_reason == "goal_reached":
                first_success_pos = position
                break
        if first_success_pos is not None:
            keep_count = max(1, int(max_success_tail_frames))
            filtered = filtered[: first_success_pos + keep_count]

    return filtered


def selected_frame_indices(
    payload: Dict[str, Any],
    *,
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    mode: Optional[str] = None,
    keep_map: Optional[Dict[int, bool]] = None,
) -> List[int]:
    resolved_frames = list(frames) if frames is not None else list(payload.get("frames") or [])
    resolved_mode = normalize_frame_selection_mode(
        mode if mode is not None else (payload.get("frame_selection_mode") or payload.get("frame_processing_mode"))
    )
    resolved_keep_map = keep_map if keep_map is not None else {
        idx: flag
        for idx, frame in enumerate(resolved_frames)
        for flag in [frame_processing_flag(frame)]
        if flag is not None
    }

    indices: List[int] = []
    for frame_index, _frame in enumerate(resolved_frames):
        flag = resolved_keep_map.get(frame_index)
        if resolved_mode == FRAME_SELECTION_MODE_RETAIN_ONLY:
            if flag is True:
                indices.append(frame_index)
            continue
        if flag is False:
            continue
        indices.append(frame_index)
    return indices


def processing_quality_ok(quality_label: Any, max_quality_label: int = DEFAULT_MAX_PROCESSING_QUALITY_LABEL) -> bool:
    value = _safe_int(quality_label, 0)
    threshold = max(1, min(3, _safe_int(max_quality_label, DEFAULT_MAX_PROCESSING_QUALITY_LABEL)))
    return value > 0 and value <= threshold


def episode_entries(session_root: Path) -> Tuple[str, List[Dict[str, Any]]]:
    index_payload = load_json(session_root / "index.json")
    session_id = str(index_payload.get("session_id") or session_root.name)
    episodes = list(index_payload.get("episodes") or [])
    return session_id, episodes


def load_session_samples(
    session_root: Path,
    action_horizon: int = 1,
    min_trajectory_length: int = 1,
    quality_review_index: Optional[QualityReviewIndex] = None,
    max_processing_quality_label: int = DEFAULT_MAX_PROCESSING_QUALITY_LABEL,
    include_unreviewed: bool = False,
    drop_initial_frames: int = 0,
    max_success_tail_frames: Optional[int] = None,
) -> List[Sample]:
    session_id, entries = episode_entries(session_root)
    samples: List[Sample] = []

    for trajectory_index, episode_meta in enumerate(entries):
        episode_id = str(episode_meta.get("episode_id") or "")
        if not episode_id:
            continue
        episode_path = session_root / "episodes" / f"{episode_id}.json"
        payload = load_json(episode_path)
        quality_review = resolved_episode_quality(payload, episode_meta, quality_review_index, session_id, episode_id)
        if quality_review.get("exclude_from_processing"):
            continue
        quality_label = quality_review.get("quality_label")
        if not processing_quality_ok(quality_label, max_processing_quality_label):
            if include_unreviewed and _safe_int(quality_label, 0) == 0:
                pass
            else:
                continue
        instruction = str(payload.get("instruction") or "")
        scene_id = str(payload.get("scene_id") or episode_meta.get("scene_id") or "")
        operator_id = str(payload.get("operator_id") or episode_meta.get("operator_id") or "")
        task_metadata = episode_task_metadata(payload, episode_meta)
        derived_labels = load_episode_derived_labels(session_root, episode_id)
        schema_version = str(payload.get("schema_version") or payload.get("meta", {}).get("schema_version") or "")
        frames = list(payload.get("frames") or [])
        frame_processing = episode_frame_processing(payload)
        selected_indices = apply_auto_frame_filters(
            payload,
            selected_indices=frame_processing["selected_frame_indices"],
            drop_initial_frames=drop_initial_frames,
            max_success_tail_frames=max_success_tail_frames,
        )
        selected_frames = [frames[index] for index in selected_indices if 0 <= index < len(frames)]
        nominal_dt = estimate_nominal_dt(selected_frames)
        trajectory_length = len(selected_frames)
        if trajectory_length < max(1, int(min_trajectory_length)):
            continue

        for step_pos, frame in enumerate(selected_frames):
            image_rel = frame.get("image")
            source_image_path = None if not image_rel else str(session_root / str(image_rel))
            image_path = None if not image_rel else str(Path("sessions") / session_id / str(image_rel))
            frame_meta = frame.get("meta")
            if not isinstance(frame_meta, dict):
                frame_meta = {}

            action_chunk: List[Dict[str, Any]] = []
            for future_pos in range(step_pos, min(step_pos + action_horizon, trajectory_length)):
                action_chunk.append(control_action_from_frame(selected_frames[future_pos]))

            source_frame_index = selected_indices[step_pos] if step_pos < len(selected_indices) else step_pos
            previous_action, prev_action_valid, prev_action_source, prev_action_reason = previous_action_info_for_step(
                frames=selected_frames,
                selected_indices=selected_indices,
                step_pos=step_pos,
                nominal_dt=nominal_dt,
            )
            raw_record = {
                "schema_version": schema_version,
                "episode_id": episode_id,
                "trajectory_id": episode_id,
                "session_id": session_id,
                "instruction": instruction,
                "scene_id": scene_id,
                "operator_id": operator_id,
                "frame_selection_mode": frame_processing["frame_selection_mode"],
                "selected_frame_count": len(selected_indices),
                "total_frame_count": frame_processing["total_frame_count"],
                "frame_processing": {
                    "source_frame_index": source_frame_index,
                    "selected_for_processing": True,
                    "auto_drop_initial_frames": int(drop_initial_frames),
                    "auto_max_success_tail_frames": max_success_tail_frames,
                },
                "previous_action": previous_action,
                "prev_action_valid": prev_action_valid,
                "prev_action_source": prev_action_source,
                "prev_action_reason": prev_action_reason,
            }
            raw_record.update(task_metadata)
            for field in ("contrast_group_id", "contrast_variant", "active_target_id"):
                value = (
                    _safe_str(frame_meta.get(field)) or
                    _safe_str(payload.get(field)) or
                    _safe_str(episode_meta.get(field))
                )
                if value:
                    raw_record[field] = value
            scene_targets = payload.get("scene_targets")
            if not isinstance(scene_targets, list):
                scene_targets = episode_meta.get("scene_targets")
            if isinstance(scene_targets, list) and scene_targets:
                raw_record["scene_targets"] = scene_targets
            for field in QUALITY_REVIEW_FIELDS:
                value = quality_review.get(field)
                if value is None:
                    continue
                if isinstance(value, str) and not value:
                    continue
                raw_record[field] = value
            if derived_labels:
                raw_record["derived_labels"] = derived_labels

            samples.append(
                Sample(
                    session_id=session_id,
                    trajectory_id=episode_id,
                    trajectory_index=trajectory_index,
                    trajectory_step_index=step_pos,
                    trajectory_length=trajectory_length,
                    step_id=step_pos + 1,
                    timestamp=_safe_float(frame.get("timestamp")),
                    instruction=instruction,
                    image_path=image_path,
                    source_image_path=source_image_path,
                    state=state_from_frame(frame),
                    raw_action=raw_action_from_frame(frame),
                    control_action=control_action_from_frame(frame),
                    action_chunk=action_chunk,
                    raw_record=raw_record,
                )
            )

    return samples


def _ratio_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    if total <= 0:
        return {"train": 0, "val": 0, "test": 0}

    ratios = [max(0.0, train_ratio), max(0.0, val_ratio), max(0.0, test_ratio)]
    if sum(ratios) <= 0.0:
        ratios = [1.0, 0.0, 0.0]
    ratio_sum = sum(ratios)
    ratios = [value / ratio_sum for value in ratios]

    raw_counts = [total * ratios[0], total * ratios[1], total * ratios[2]]
    floored = [int(value) for value in raw_counts]
    remainder = total - sum(floored)
    fractional = sorted(
        [
            (raw_counts[index] - floored[index], index)
            for index in range(3)
            if ratios[index] > 0.0
        ],
        reverse=True,
    )
    for step in range(remainder):
        target_index = fractional[step % len(fractional)][1] if fractional else 0
        floored[target_index] += 1

    counts = {
        "train": floored[0],
        "val": floored[1],
        "test": floored[2],
    }

    return counts


def assign_splits(
    samples: Sequence[Sample],
    split_mode: str = "auto",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: Optional[int] = None,
) -> Dict[str, List[Sample]]:
    if not samples:
        return {"train": [], "val": [], "test": []}

    session_groups: Dict[str, List[Sample]] = {}
    trajectory_groups: Dict[Tuple[str, str], List[Sample]] = {}
    contrast_groups: Dict[str, List[Sample]] = {}
    for sample in sorted(samples, key=lambda item: (item.session_id, item.trajectory_id, item.step_id)):
        session_groups.setdefault(sample.session_id, []).append(sample)
        split_group_id = _safe_str(sample.raw_record.get("contrast_group_id")) or sample.trajectory_id
        trajectory_groups.setdefault((sample.session_id, split_group_id), []).append(sample)
        contrast_groups.setdefault(split_group_id, []).append(sample)

    if split_mode == "auto":
        split_mode = "by_session" if len(session_groups) >= 2 else "by_trajectory"
    if split_mode not in {"by_session", "by_trajectory", "by_contrast_group"}:
        raise ValueError(f"unsupported split mode: {split_mode}")

    grouped_items: List[List[Sample]]
    if split_mode == "by_session":
        def _session_sort_key(key: str) -> Tuple[int, str]:
            episode_count = len({sample.trajectory_id for sample in session_groups[key]})
            return (-episode_count, key)

        grouped_items = [session_groups[key] for key in sorted(session_groups, key=_session_sort_key)]
    elif split_mode == "by_contrast_group":
        grouped_items = [
            contrast_groups[key]
            for key in sorted(
                contrast_groups,
                key=lambda item: (
                    -len({sample.session_id for sample in contrast_groups[item]}),
                    item,
                ),
            )
        ]
    else:
        grouped_items = [
            trajectory_groups[key]
            for key in sorted(
                trajectory_groups,
                key=lambda item: (
                    item[0],
                    item[1],
                ),
            )
        ]

    if split_seed is not None and len(grouped_items) > 1:
        rng = random.Random(int(split_seed))
        rng.shuffle(grouped_items)

    counts = _ratio_counts(len(grouped_items), train_ratio, val_ratio, test_ratio)
    split_names: List[str] = []
    for name in ("train", "val", "test"):
        split_names.extend([name] * counts[name])
    while len(split_names) < len(grouped_items):
        split_names.append("train")

    split_samples = {"train": [], "val": [], "test": []}
    for group, split_name in zip(grouped_items, split_names):
        split_samples[split_name].extend(group)
    return split_samples


def hash_token(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def tokenize_instruction(text: str) -> List[str]:
    return TEXT_TOKEN_RE.findall(text.lower())


def instruction_feature_vector(text: str, dim: int = 64) -> List[float]:
    tokens = tokenize_instruction(text)
    vector = [0.0] * dim
    if not tokens:
        return vector
    scale = 1.0 / len(tokens)
    for token in tokens:
        vector[hash_token(token, dim)] += scale
    return vector


def state_feature_vector(state: Dict[str, Any]) -> List[float]:
    state_dict = state_dict_from_any(state)
    return [_safe_float(state_dict.get(field)) for field in STATE_FIELDS]


def state_dict_from_any(state: Any) -> Dict[str, float]:
    if isinstance(state, dict):
        return {field: _safe_float(state.get(field)) for field in STATE_FIELDS}
    if isinstance(state, (list, tuple)):
        values = list(state)
        return {
            field: _safe_float(values[index]) if index < len(values) else 0.0
            for index, field in enumerate(STATE_FIELDS)
        }
    return {field: 0.0 for field in STATE_FIELDS}


def state_values_from_record(record: Dict[str, Any]) -> Dict[str, float]:
    if isinstance(record.get("state_dict"), dict):
        return state_dict_from_any(record.get("state_dict"))
    return state_dict_from_any(record.get("state"))


def infer_target_fields_from_record(record: Dict[str, Any]) -> List[str]:
    actions_continuous = record.get("actions_continuous")
    if isinstance(actions_continuous, list) and actions_continuous:
        action_fields = record.get("action_fields")
        if not isinstance(action_fields, list) or not action_fields:
            action_fields = ["vx", "wz"]
        target_fields: List[str] = []
        for step_index in range(len(actions_continuous)):
            for field in action_fields:
                target_fields.append(f"{field}_t{step_index}")
        return target_fields

    if isinstance(record.get("control_action"), dict):
        action = dict(record.get("control_action") or {})
        return [field for field in ACTION_FIELDS if field in action]

    if isinstance(record.get("action_dict"), dict):
        action_dict = dict(record.get("action_dict") or {})
        ordered = [field for field in ("vx", "wz") if field in action_dict]
        return ordered or list(action_dict.keys())

    action = record.get("action")
    if isinstance(action, (list, tuple)):
        action_fields = record.get("action_fields")
        if isinstance(action_fields, list) and len(action_fields) == len(action):
            return [str(field) for field in action_fields]
        default_fields = ["vx", "wz"]
        return default_fields[: len(action)]

    return list(ACTION_FIELDS)


def target_vector_from_record(record: Dict[str, Any], target_fields: Optional[Sequence[str]] = None) -> List[float]:
    resolved_target_fields = list(target_fields) if target_fields is not None else infer_target_fields_from_record(record)
    actions_continuous = record.get("actions_continuous")
    if isinstance(actions_continuous, list) and actions_continuous:
        flat: List[float] = []
        for action in actions_continuous:
            if not isinstance(action, (list, tuple)):
                continue
            flat.extend(_safe_float(value) for value in action)
        return flat[: len(resolved_target_fields)]

    if isinstance(record.get("control_action"), dict):
        action = dict(record.get("control_action") or {})
        return [_safe_float(action.get(field)) for field in resolved_target_fields]

    if isinstance(record.get("action_dict"), dict):
        action = dict(record.get("action_dict") or {})
        return [_safe_float(action.get(field)) for field in resolved_target_fields]

    action = record.get("action")
    if isinstance(action, (list, tuple)):
        values = list(action)
        return [_safe_float(values[index]) if index < len(values) else 0.0 for index in range(len(resolved_target_fields))]

    return [0.0] * len(resolved_target_fields)


def resolve_image_path(record: Dict[str, Any], dataset_root: Path) -> Optional[Path]:
    source = record.get("source_image_path")
    if source:
        path = Path(str(source))
        if path.exists():
            return path
    image_path = record.get("image_path")
    if image_path:
        path = dataset_root / str(image_path)
        if path.exists():
            return path
    return None


def image_feature_vector(image_path: Optional[Path]) -> List[float]:
    if image_path is None or not image_path.exists():
        return [0.0] * 20

    size = image_path.stat().st_size
    if size <= 0:
        return [0.0] * 20

    with image_path.open("rb") as handle:
        data = handle.read(4096)
    if not data:
        return [0.0] * 20

    hist = [0] * 16
    total = 0
    total_sq = 0
    for byte in data:
        hist[byte // 16] += 1
        total += byte
        total_sq += byte * byte

    mean = total / size
    variance = max(0.0, (total_sq / size) - (mean * mean))
    std = math.sqrt(variance)
    entropy = 0.0
    for count in hist:
        if count == 0:
            continue
        probability = count / size
        entropy -= probability * math.log2(probability)

    vector = [count / size for count in hist]
    vector.extend([math.log1p(size), mean / 255.0, std / 128.0, entropy / 8.0])
    return vector


def feature_vector_from_record(record: Dict[str, Any], dataset_root: Path, text_dim: int = 64) -> List[float]:
    vector: List[float] = []
    vector.extend(instruction_feature_vector(str(record.get("instruction") or ""), text_dim))
    vector.extend(state_feature_vector(state_values_from_record(record)))
    vector.extend(image_feature_vector(resolve_image_path(record, dataset_root)))
    return vector


def fit_standardizer(vectors: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    if not vectors:
        return [], []

    dim = len(vectors[0])
    means = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            means[index] += float(value)
    means = [value / len(vectors) for value in means]

    variances = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            diff = float(value) - means[index]
            variances[index] += diff * diff
    stds = [math.sqrt(value / max(1, len(vectors))) for value in variances]
    stds = [std if std > 1e-12 else 1.0 for std in stds]
    return means, stds


def standardize_vector(vector: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> List[float]:
    if not means or not stds:
        return list(vector)
    return [(float(value) - means[index]) / stds[index] for index, value in enumerate(vector)]
