"""Pure D5 collection decisions, kept independent of Isaac runtime imports."""

from __future__ import annotations

import os
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from collections import Counter
from typing import Any, Mapping, Sequence

from src.smolvla.d5_data_expansion import (
    DISTANCE_BIN_NAMES,
    episode_features,
    select_stratified_episodes,
)


INFRASTRUCTURE_CHECKS = (
    "record_count_positive",
    "record_summary_count_match",
    "frame_indices_contiguous",
    "no_missing_frames",
    "rgb_sequence_normal",
    "timestamps_strictly_monotonic",
    "timestamp_period_matches_metadata",
    "actions_finite",
    "actions_in_expected_range",
    "planner_equals_locomotion_command",
    "robot_state_finite",
    "robot_state_dimension_constant",
    "normal_termination",
)

D5_SPLIT_RULES = {
    "train": {
        "per_category": 10,
        "required_ids": ("short_vln_v1_0000", "short_vln_v1_0004"),
        "min_scenes": 8,
        "min_heading_bins": 8,
        "min_distance_per_bin": 6,
    },
    "seen-val": {
        "per_category": 4,
        "required_ids": ("short_vln_v1_0001", "short_vln_v1_0003"),
        "min_scenes": 7,
        "min_heading_bins": 6,
        "min_distance_per_bin": 2,
    },
}
D5_SPLIT_ACCEPTED_CAPS = {"train": 34, "seen-val": 16}
D5_TOTAL_ACCEPTED_CAP = 50
D5_ATTEMPT_MULTIPLIER = 3
D5_CATEGORIES = ("straight", "left_turn", "right_turn")
D5_ASSIGNMENT_SIDECAR = "d5_assignment.json"

OFFICIAL_LARGE_ORIENTATION_RAD = 0.6
IMMUTABLE_OFFICIAL_PLANNER_SHA256 = "20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432"
REPLACEABLE_REJECT_KINDS = frozenset(
    {"planner_reject", "expert_rollout_reject", "expert_reset_reject"}
)
_OFFICIAL_LARGE_ORIENTATION_LOG = re.compile(
    r"^Large orientation:\s*tensor\(\[([^\]]+)\]\)\s+tensor\(\[([^\]]+)\]\)\s*$",
    re.MULTILINE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_safe(value: Any) -> Any:
    """Convert D5 audit values to deterministic JSON before an atomic write."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=lambda item: _canonical_json(_json_safe(item)))]
    raise TypeError(f"D5 manifest contains unsupported value type: {type(value).__name__}")


def collection_ruleset() -> dict[str, Any]:
    """Return the immutable-on-resume D5 acceptance rules recorded in the manifest.

    The train scene threshold was explicitly revised from nine to eight after
    D5-R4 proved that eight is the maximum reachable accepted-scene coverage.
    Recording both values prevents a resumed collection from silently changing
    the meaning of the accepted cohort.
    """
    return _json_safe({
        "format": "go2-short-vln-m6_2-d5-rules-v2",
        "train_min_scenes_change": {
            "previous": 9,
            "current": 8,
            "authority": "user_direction_2026-09-02",
        },
        "split_rules": D5_SPLIT_RULES,
        "accepted_caps": D5_SPLIT_ACCEPTED_CAPS,
        "total_accepted_cap": D5_TOTAL_ACCEPTED_CAP,
    })


def _assignment_policy(
    path: Path | None, *, dataset: Path, episodes: Sequence[Mapping[str, Any]],
    base_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Load a narrowly-scoped, additive D5 collection split override."""
    if path is None:
        return {"path": None, "sha256": None, "assignments": {}}
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("format") != "go2-short-vln-m6_2-d5-assignment-overrides-v1":
        raise ValueError("invalid D5 assignment override format")
    if payload.get("source_dataset_sha256") != sha256(dataset):
        raise ValueError("D5 assignment override dataset hash does not match")
    rows = payload.get("assignments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("D5 assignment override lacks assignments")
    raw_by_id = {str(item.get("short_episode_id")): item for item in episodes if isinstance(item, Mapping)}
    assignments: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("D5 assignment override contains a non-object")
        identifier = row.get("short_episode_id")
        source_split, collection_split = row.get("source_split"), row.get("collection_split")
        reason = row.get("reason")
        if not isinstance(identifier, str) or identifier in assignments:
            raise ValueError("D5 assignment override has a missing or duplicate episode ID")
        source = raw_by_id.get(identifier)
        if not isinstance(source, Mapping) or source.get("split") != source_split:
            raise ValueError(f"D5 assignment override source split does not match {identifier}")
        if source_split not in D5_SPLIT_RULES or collection_split not in D5_SPLIT_RULES or source_split == collection_split:
            raise ValueError(f"D5 assignment override has invalid split assignment for {identifier}")
        if collection_split != "train" or source_split != "seen-val" or reason != "recover_required_train_scene_coverage":
            raise ValueError(f"D5 assignment override is outside the approved D5-R4 scope: {identifier}")
        if row.get("exclude_from_seen_val_evaluation") is not True:
            raise ValueError(f"D5 assignment override must exclude {identifier} from seen-val evaluation")
        if identifier in base_ids[source_split] or identifier in base_ids[collection_split]:
            raise ValueError(f"D5 assignment override may not alter an initial selected route: {identifier}")
        assignments[identifier] = {
            "short_episode_id": identifier,
            "source_split": source_split,
            "collection_split": collection_split,
            "reason": reason,
            "exclude_from_seen_val_evaluation": True,
        }
    return {
        "path": str(resolved), "sha256": sha256(resolved),
        "format": payload["format"], "source_dataset_sha256": payload["source_dataset_sha256"],
        "assignments": assignments,
    }


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    return False


def _rpy_from_quaternion_wxyz(quaternion: Sequence[Any]) -> tuple[float, float, float]:
    if len(quaternion) != 4 or not _finite_numbers(quaternion):
        raise ValueError("quaternion_wxyz must contain four finite values")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("quaternion_wxyz must have nonzero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def official_large_orientation(quaternion_wxyz: Sequence[Any]) -> bool:
    """Match the immutable demo planner's strict roll/pitch threshold."""
    roll, pitch, _ = _rpy_from_quaternion_wxyz(quaternion_wxyz)
    return abs(roll) > OFFICIAL_LARGE_ORIENTATION_RAD or abs(pitch) > OFFICIAL_LARGE_ORIENTATION_RAD


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _rounded_orientation_lower_bound(value_text: str) -> tuple[float, float]:
    """Return a printed orientation value and a conservative absolute lower bound."""
    value = float(value_text)
    if not math.isfinite(value):
        raise ValueError("official large-orientation log is non-finite")
    fractional = value_text.strip().lower().partition("e")[0].partition(".")[2]
    uncertainty = 0.5 * (10.0 ** (-len(fractional))) if fractional else 0.5
    return value, max(0.0, abs(value) - uncertainty)


def _zero_frame_orientation_recovery_evidence(
    attempt_dir: Path, *, m4_log_root: Path
) -> dict[str, Any]:
    """Prove a pre-action official orientation exit from immutable runner evidence."""
    steps_path = attempt_dir / "steps.jsonl"
    config_path = attempt_dir / "collector_config.json"
    if not steps_path.is_file() or steps_path.stat().st_size != 0:
        raise ValueError("zero-frame recovery requires an empty steps.jsonl")
    if not config_path.is_file():
        raise ValueError("zero-frame recovery lacks collector_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("zero-frame recovery collector config is invalid")
    identifier = config.get("short_episode_id")
    if identifier != attempt_dir.name:
        raise ValueError("zero-frame recovery episode ID does not match attempt directory")
    if config.get("official_planner_sha256") != IMMUTABLE_OFFICIAL_PLANNER_SHA256:
        raise ValueError("zero-frame recovery official planner hash does not match D5 provenance")
    rgb_dir = attempt_dir / "front_rgb"
    if not rgb_dir.is_dir() or any(rgb_dir.iterdir()):
        raise ValueError("zero-frame recovery unexpectedly contains RGB frames")
    if not m4_log_root.is_dir():
        raise ValueError("zero-frame recovery M4 log root is unavailable")
    candidates = []
    expected_marker = f"m4_episode_dir={attempt_dir}"
    for log_path in sorted(m4_log_root.glob(f"*-{identifier}.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if expected_marker in text:
            candidates.append((log_path, text))
    if len(candidates) != 1:
        raise ValueError("zero-frame recovery requires exactly one matching M4 runner log")
    log_path, log_text = candidates[0]
    matches = list(_OFFICIAL_LARGE_ORIENTATION_LOG.finditer(log_text))
    if len(matches) != 1:
        raise ValueError("zero-frame recovery requires exactly one official orientation log event")
    match = matches[0]
    roll, roll_lower_bound = _rounded_orientation_lower_bound(match.group(1))
    pitch, pitch_lower_bound = _rounded_orientation_lower_bound(match.group(2))
    if max(roll_lower_bound, pitch_lower_bound) <= OFFICIAL_LARGE_ORIENTATION_RAD:
        raise ValueError("rounded official orientation evidence does not strictly exceed the threshold")
    after_event = log_text[match.end() :]
    if "M4 timeline STOP" not in after_event:
        raise ValueError("zero-frame recovery log lacks the expected timeline stop")
    for marker in ("Traceback (most recent call last):", "CUDA out of memory", "Segmentation fault"):
        if marker in after_event:
            raise ValueError(f"zero-frame recovery log contains infrastructure error: {marker}")
    event_line = log_text.count("\n", 0, match.start()) + 1
    return {
        "kind": "expert_reset_reject",
        "reason": "official_large_orientation_before_first_action",
        "official_threshold_rad": OFFICIAL_LARGE_ORIENTATION_RAD,
        "record_count": 0,
        "steps_sha256": sha256(steps_path),
        "collector_config_sha256": sha256(config_path),
        "m4_log": str(log_path),
        "m4_log_sha256": sha256(log_path),
        "m4_log_line": event_line,
        "printed_rpy_rad": {"roll": roll, "pitch": pitch},
        "conservative_abs_lower_bound_rad": {"roll": roll_lower_bound, "pitch": pitch_lower_bound},
    }


def _large_orientation_recovery_evidence(attempt_dir: Path) -> dict[str, Any]:
    """Validate an interrupted attempt before excluding it as a route failure."""
    steps_path = attempt_dir / "steps.jsonl"
    if not steps_path.is_file():
        raise ValueError("interrupted attempt lacks steps.jsonl")
    records = _read_jsonl(steps_path)
    if len(records) < 2:
        raise ValueError("interrupted attempt needs at least two records")
    for index, record in enumerate(records):
        if record.get("frame_index") != index:
            raise ValueError("interrupted attempt has non-contiguous frame indices")
        if not _finite_numbers(record.get("timestamp")) or not _finite_numbers(record.get("control_dt_s")):
            raise ValueError("interrupted attempt has non-finite timing")
        relative = record.get("front_rgb")
        if not isinstance(relative, str) or not relative or not (attempt_dir / relative).is_file():
            raise ValueError("interrupted attempt has missing RGB evidence")
        planner, locomotion = record.get("planner_command"), record.get("locomotion_command")
        if not _finite_numbers(planner) or not _finite_numbers(locomotion) or len(planner) != 3 or len(locomotion) != 3:
            raise ValueError("interrupted attempt has invalid commands")
        if any(abs(float(left) - float(right)) > 1e-9 for left, right in zip(planner, locomotion)):
            raise ValueError("interrupted attempt planner and locomotion commands differ")
        if not _finite_numbers(record.get("robot_state")):
            raise ValueError("interrupted attempt has non-finite robot state")
    last = records[-1]
    if bool(last.get("termination")):
        raise ValueError("interrupted attempt already has normal termination evidence")
    pose = last.get("next_robot_pose")
    if not isinstance(pose, Mapping):
        raise ValueError("interrupted attempt lacks final pose")
    quaternion = pose.get("quaternion_wxyz")
    roll, pitch, yaw = _rpy_from_quaternion_wxyz(quaternion)
    if not official_large_orientation(quaternion):
        raise ValueError("interrupted attempt does not meet official large-orientation threshold")
    return {
        "kind": "expert_rollout_reject",
        "reason": "official_large_orientation",
        "official_threshold_rad": OFFICIAL_LARGE_ORIENTATION_RAD,
        "record_count": len(records),
        "steps_sha256": sha256(steps_path),
        "last_frame_index": int(last["frame_index"]),
        "last_rpy_rad": {"roll": roll, "pitch": pitch, "yaw": yaw},
        "last_position_w": pose.get("position_w"),
    }


def apply_large_orientation_recovery(
    manifest: dict[str, Any], attempt_dir: Path, *, m4_log_root: Path | None = None
) -> dict[str, Any]:
    """Resolve exactly one official-orientation blocked attempt without altering raw evidence."""
    resolved_attempt = attempt_dir.resolve()
    if manifest.get("status") != "blocked_infrastructure_failure":
        raise ValueError("only a blocked infrastructure manifest may be recovered")
    attempts = manifest.get("attempts")
    rejected = manifest.get("rejected_ids")
    if not isinstance(attempts, list) or not attempts or not isinstance(rejected, dict):
        raise ValueError("invalid blocked D5 collection manifest")
    attempt = attempts[-1]
    if not isinstance(attempt, dict) or Path(str(attempt.get("attempt_dir", ""))).resolve() != resolved_attempt:
        raise ValueError("recovery target must be the final blocked attempt")
    original = attempt.get("decision")
    if not isinstance(original, dict) or original.get("kind") != "infrastructure_failure":
        raise ValueError("final attempt is not an infrastructure failure")
    split = attempt.get("split")
    identifier = attempt.get("short_episode_id")
    if split not in D5_SPLIT_RULES or not isinstance(identifier, str) or not identifier:
        raise ValueError("blocked attempt lacks a valid split or episode ID")
    evidence_path = resolved_attempt / "recovery_evidence.json"
    if evidence_path.exists() or "initial_decision" in attempt:
        raise ValueError("blocked attempt already has recovery evidence")
    steps_path = resolved_attempt / "steps.jsonl"
    if not steps_path.is_file():
        raise ValueError("interrupted attempt lacks steps.jsonl")
    if steps_path.stat().st_size == 0:
        if m4_log_root is None:
            raise ValueError("zero-frame recovery requires an explicit M4 log root")
        evidence = _zero_frame_orientation_recovery_evidence(
            resolved_attempt, m4_log_root=m4_log_root.resolve()
        )
    else:
        evidence = _large_orientation_recovery_evidence(resolved_attempt)
    evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    attempt["initial_decision"] = dict(original)
    attempt["decision"] = {"kind": evidence["kind"], "reason": evidence["reason"]}
    attempt["recovery_evidence"] = str(evidence_path)
    values = rejected.get(split)
    if not isinstance(values, list):
        raise ValueError(f"manifest lacks rejected IDs for {split}")
    if identifier not in values:
        values.append(identifier)
        values.sort()
    events = manifest.setdefault("recovery_events", [])
    if not isinstance(events, list):
        raise ValueError("manifest recovery_events must be a list")
    events.append({"short_episode_id": identifier, "split": split, "evidence": evidence})
    manifest["status"] = "collecting"
    return evidence


def classify_collection_attempt(
    summary: Mapping[str, Any], sanity: Mapping[str, Any]
) -> dict[str, str]:
    """Classify one completed M4 attempt without treating PD failure as infra failure."""
    checks = sanity.get("checks")
    if not isinstance(checks, Mapping):
        return {"kind": "infrastructure_failure", "reason": "missing_sanity_checks"}
    zero_frame_orientation = (
        summary.get("status") == "terminated_without_success"
        and summary.get("termination_reason") == "official_large_orientation_before_first_action"
        and not bool(summary.get("success"))
        and summary.get("record_count") == 0
    )
    if zero_frame_orientation:
        if checks.get("zero_frame_official_orientation_evidence") is not True:
            return {
                "kind": "infrastructure_failure",
                "reason": "failed_infrastructure_checks:zero_frame_official_orientation_evidence",
            }
        return {
            "kind": "expert_reset_reject",
            "reason": "official_large_orientation_before_first_action",
        }
    large_orientation = (
        summary.get("status") == "terminated_without_success"
        and summary.get("termination_reason") == "official_large_orientation"
        and not bool(summary.get("success"))
    )
    required_checks = tuple(name for name in INFRASTRUCTURE_CHECKS if not (large_orientation and name == "normal_termination"))
    missing_or_failed = [name for name in required_checks if checks.get(name) is not True]
    if missing_or_failed:
        return {
            "kind": "infrastructure_failure",
            "reason": "failed_infrastructure_checks:" + ",".join(missing_or_failed),
        }
    if bool(summary.get("success")) and summary.get("status") == "complete" and bool(sanity.get("passed")):
        return {"kind": "accepted", "reason": "success"}
    if large_orientation:
        return {"kind": "expert_rollout_reject", "reason": "official_large_orientation"}
    if (
        summary.get("status") == "terminated_without_success"
        and summary.get("termination_reason") == "environment_done"
        and not bool(summary.get("success"))
    ):
        if checks.get("final_distance_within_success_radius") is False:
            return {"kind": "planner_reject", "reason": "goal_radius_not_reached"}
        return {"kind": "planner_reject", "reason": "normal_pd_termination_without_success"}
    return {"kind": "infrastructure_failure", "reason": "unexpected_attempt_terminal_state"}


def promote_accepted_attempt(attempt_dir: Path, accepted_root: Path) -> Path:
    """Atomically promote one successful attempt without copying its RGB evidence."""
    source = attempt_dir.resolve()
    destination_root = accepted_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(source, destination)
    return accepted_root / source.name


def write_assignment_sidecar(
    attempt_dir: Path, feature: Mapping[str, Any], assignment_policy: Mapping[str, Any]
) -> Path | None:
    """Attach an additive D5 split assignment without modifying M4's summary."""
    assignment = feature.get("assignment_override")
    if not isinstance(assignment, Mapping):
        return None
    path = attempt_dir / D5_ASSIGNMENT_SIDECAR
    if path.exists():
        raise FileExistsError(f"refusing to overwrite D5 assignment evidence: {path}")
    payload = {
        "format": "go2-short-vln-m6_2-d5-assignment-sidecar-v1",
        "short_episode_id": feature["short_episode_id"],
        "source_split": assignment["source_split"],
        "collection_split": assignment["collection_split"],
        "reason": assignment["reason"],
        "exclude_from_seen_val_evaluation": assignment["exclude_from_seen_val_evaluation"],
        "assignment_policy_sha256": assignment_policy["sha256"],
        "source_dataset_sha256": assignment_policy["source_dataset_sha256"],
    }
    _write_json_atomically(path, payload)
    return path


def reselect_d5_splits(
    episodes: Sequence[Mapping[str, Any]], *,
    base_ids: Mapping[str, set[str]] | None = None,
    accepted_ids: Mapping[str, set[str]], rejected_ids: Mapping[str, set[str]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Legacy full-cohort view for tests and initial-manifest validation.

    Collection itself uses :func:`next_d5_selection`: a bounded, one-route
    decision that cannot confuse a combinatorial search limit with infeasible
    coverage.  This compatibility view intentionally retains the original
    deterministic selector where callers need a complete cohort.
    """
    selected: dict[str, list[dict[str, Any]]] = {}
    for split, rules in D5_SPLIT_RULES.items():
        accepted = set(accepted_ids.get(split, set()))
        rejected = set(rejected_ids.get(split, set()))
        required = set(rules["required_ids"])
        if accepted & rejected:
            raise ValueError(f"accepted and rejected D5 routes overlap in {split}")
        if required & rejected:
            raise ValueError(f"required D5 route rejected in {split}: {sorted(required & rejected)}")
        if base_ids is not None:
            original = set(base_ids.get(split, set()))
            if required - original:
                raise ValueError(f"D5 base selection lacks required routes in {split}")
        selected[split] = select_stratified_episodes(
            episodes,
            split=split,
            per_category=int(rules["per_category"]),
            required_ids=tuple(rules["required_ids"]),
            fixed_ids=tuple(sorted(accepted)),
            excluded_ids=tuple(sorted(rejected)),
            min_scenes=int(rules["min_scenes"]),
            min_heading_bins=int(rules["min_heading_bins"]),
            min_distance_per_bin=int(rules["min_distance_per_bin"]),
            seed=seed,
        )
    return selected


def _features_by_split(
    episodes: Sequence[Mapping[str, Any]], assignment_policy: Mapping[str, Any] | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    result = {split: {} for split in D5_SPLIT_RULES}
    assignments = (assignment_policy or {}).get("assignments", {})
    if not isinstance(assignments, Mapping):
        raise ValueError("D5 assignment policy assignments must be a mapping")
    for raw in episodes:
        feature = episode_features(raw)
        identifier = str(feature["short_episode_id"])
        assignment = assignments.get(identifier)
        source_split = str(feature["split"])
        split = str(assignment["collection_split"]) if isinstance(assignment, Mapping) else source_split
        if split in result:
            if identifier in result[split]:
                raise ValueError(f"duplicate D5 episode ID in {split}: {identifier}")
            result[split][identifier] = {
                **feature,
                "split": split,
                "source_split": source_split,
                "assignment_override": dict(assignment) if isinstance(assignment, Mapping) else None,
            }
    return result


def accepted_coverage(items: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Coverage state derived exclusively from already accepted evidence."""
    category_counts = Counter(str(item["category"]) for item in items)
    scenes = {str(item["scene_id"]) for item in items}
    headings = {str(item["heading_bin"]) for item in items}
    distance_counts = Counter(str(item["distance_bin"]) for item in items)
    category_deficits = {
        category: max(0, int(rules["per_category"]) - category_counts[category])
        for category in D5_CATEGORIES
    }
    distance_deficits = {
        name: max(0, int(rules["min_distance_per_bin"]) - distance_counts[name])
        for name in DISTANCE_BIN_NAMES
    }
    coverage_met = (
        len(scenes) >= int(rules["min_scenes"])
        and len(headings) >= int(rules["min_heading_bins"])
        and not any(distance_deficits.values())
    )
    return {
        "accepted_count": len(items),
        "category_counts": {category: category_counts[category] for category in D5_CATEGORIES},
        "category_deficits": category_deficits,
        "scene_count": len(scenes),
        "scene_deficit": max(0, int(rules["min_scenes"]) - len(scenes)),
        "heading_bin_count": len(headings),
        "heading_bin_deficit": max(0, int(rules["min_heading_bins"]) - len(headings)),
        "distance_bin_counts": {name: distance_counts[name] for name in DISTANCE_BIN_NAMES},
        "distance_bin_deficits": distance_deficits,
        "category_minimum_met": not any(category_deficits.values()),
        "coverage_met": coverage_met,
    }


def _coverage_gain(summary: Mapping[str, Any], feature: Mapping[str, Any], rules: Mapping[str, Any]) -> int:
    """Count the unmet coverage constraints this candidate can improve by one."""
    gain = 0
    if summary["scene_deficit"] and str(feature["scene_id"]) not in summary.get("accepted_scenes", set()):
        gain += 1
    if summary["heading_bin_deficit"] and str(feature["heading_bin"]) not in summary.get("accepted_headings", set()):
        gain += 1
    distance = str(feature["distance_bin"])
    if summary["distance_bin_deficits"].get(distance, 0) > 0:
        gain += 1
    return gain


def _coverage_with_sets(items: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]) -> dict[str, Any]:
    summary = accepted_coverage(items, rules)
    summary["accepted_scenes"] = {str(item["scene_id"]) for item in items}
    summary["accepted_headings"] = {str(item["heading_bin"]) for item in items}
    return summary


def next_d5_selection(
    episodes: Sequence[Mapping[str, Any]], *, split: str, base_ids: set[str],
    accepted_ids: set[str], rejected_ids: set[str], seed: int,
    assignment_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose one deterministic route or a bounded, auditable terminal state."""
    if split not in D5_SPLIT_RULES:
        raise ValueError(f"unknown D5 split: {split}")
    rules = D5_SPLIT_RULES[split]
    features = _features_by_split(episodes, assignment_policy)[split]
    unknown = (base_ids | accepted_ids | rejected_ids) - set(features)
    if unknown:
        raise ValueError(f"D5 routes are not in split {split}: {sorted(unknown)}")
    required = set(rules["required_ids"])
    if required & rejected_ids:
        return {"kind": "blocked", "reason": "required_route_rejected", "split": split,
                "required_rejected": sorted(required & rejected_ids)}
    if accepted_ids & rejected_ids:
        raise ValueError(f"accepted and rejected D5 routes overlap in {split}")
    accepted_items = [features[identifier] for identifier in sorted(accepted_ids)]
    summary = _coverage_with_sets(accepted_items, rules)
    if len(accepted_ids) > D5_SPLIT_ACCEPTED_CAPS[split]:
        return {"kind": "blocked", "reason": "accepted_cap_exceeded", "split": split, "summary": summary}
    if summary["category_minimum_met"] and summary["coverage_met"]:
        return {"kind": "complete", "split": split, "summary": summary}

    untried = [item for identifier, item in features.items() if identifier not in accepted_ids | rejected_ids]
    phase = "category_minimum" if not summary["category_minimum_met"] else "coverage_supplement"
    if phase == "category_minimum":
        deficits = {category for category, value in summary["category_deficits"].items() if value > 0}
        candidates = [item for item in untried if item["category"] in deficits]
        # On a fresh collection retain the frozen, audited base selection first.
        base_candidates = [item for item in candidates if item["short_episode_id"] in base_ids]
        if base_candidates:
            candidates = base_candidates
    else:
        candidates = [item for item in untried if _coverage_gain(summary, item, rules) > 0]
        if len(accepted_ids) >= D5_SPLIT_ACCEPTED_CAPS[split]:
            return {"kind": "blocked", "reason": "coverage_budget_exhausted", "split": split, "summary": summary}
    if not candidates:
        return {"kind": "blocked", "reason": "selection_infeasible", "split": split, "phase": phase, "summary": summary}

    def score(item: Mapping[str, Any]) -> tuple[int, int, str, str]:
        identifier = str(item["short_episode_id"])
        # Stable digest makes route choice reproducible without depending on dict order.
        stable = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()
        return (-_coverage_gain(summary, item, rules), 0 if identifier in base_ids else 1, stable, identifier)

    feature = min(candidates, key=score)
    identifier = str(feature["short_episode_id"])
    return {
        "kind": "candidate", "split": split, "phase": phase, "feature": feature,
        "summary": summary, "coverage_gain": _coverage_gain(summary, feature, rules),
        "base_candidate": identifier in base_ids, "score": list(score(feature)),
    }


def collection_accepted_summary(
    episodes: Sequence[Mapping[str, Any]], accepted: Mapping[str, set[str]],
    assignment_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    features = _features_by_split(episodes, assignment_policy)
    splits: dict[str, Any] = {}
    for split, rules in D5_SPLIT_RULES.items():
        items = [features[split][identifier] for identifier in sorted(accepted[split])]
        summary = accepted_coverage(items, rules)
        splits[split] = {
            **summary,
            "accepted_cap": D5_SPLIT_ACCEPTED_CAPS[split],
            "supplemental_count": max(0, len(items) - _target_count(split)),
        }
    total = sum(len(values) for values in accepted.values())
    return {"splits": splits, "total_accepted": total, "total_accepted_cap": D5_TOTAL_ACCEPTED_CAP}


def next_collection_decision(
    episodes: Sequence[Mapping[str, Any]], *, base_ids: Mapping[str, set[str]],
    accepted: Mapping[str, set[str]], rejected: Mapping[str, set[str]], seed: int,
    assignment_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the next split in fixed order, or return a precise terminal reason."""
    summary = collection_accepted_summary(episodes, accepted, assignment_policy)
    if summary["total_accepted"] > D5_TOTAL_ACCEPTED_CAP:
        return {"kind": "blocked", "reason": "total_accepted_cap_exceeded", "summary": summary}
    for split in D5_SPLIT_RULES:
        decision = next_d5_selection(
            episodes, split=split, base_ids=base_ids[split], accepted_ids=accepted[split],
            rejected_ids=rejected[split], seed=seed, assignment_policy=assignment_policy,
        )
        if decision["kind"] == "blocked":
            features = _features_by_split(episodes, assignment_policy)[split]
            accepted_items = [features[identifier] for identifier in accepted[split]]
            possible_items = accepted_items + [
                item for identifier, item in features.items()
                if identifier not in accepted[split] | rejected[split]
            ]
            maximum = accepted_coverage(possible_items, D5_SPLIT_RULES[split])
            decision["feasibility"] = {
                "untried_candidate_count": len(possible_items) - len(accepted_items),
                "maximum_possible": maximum,
            }
        if decision["kind"] != "complete":
            decision["collection_summary"] = summary
            return decision
    return {"kind": "complete", "summary": summary}


def selection_event(decision: Mapping[str, Any], *, event_id: int) -> dict[str, Any]:
    feature = decision["feature"]
    return {
        "event_id": event_id,
        "phase": decision["phase"],
        "split": decision["split"],
        "short_episode_id": feature["short_episode_id"],
        "category": feature["category"],
        "scene_id": feature["scene_id"],
        "source_split": feature.get("source_split", decision["split"]),
        "assignment_override": feature.get("assignment_override"),
        "heading_bin": feature["heading_bin"],
        "distance_bin": feature["distance_bin"],
        "base_candidate": decision["base_candidate"],
        "coverage_gain": decision["coverage_gain"],
        "score": decision["score"],
        "coverage_before": {
            key: value for key, value in decision["summary"].items()
            if key not in {"accepted_scenes", "accepted_headings"}
        },
    }


def _acquire_collection_lock(manifest_path: Path):
    """Acquire an exclusive process lock next to the manifest without deleting evidence."""
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError(f"D5 collection is already running (lock: {lock_path})") from error
    return stream


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomically(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def next_blocked_snapshot_path(manifest_path: Path) -> Path:
    """Choose a new immutable recovery snapshot, including after legacy R1 snapshots."""
    legacy = manifest_path.with_name(f"{manifest_path.stem}.blocked_before_resume.json")
    index = 1 if not legacy.exists() else 2
    while True:
        candidate = manifest_path.with_name(
            f"{manifest_path.stem}.blocked_before_resume.{index:04d}.json"
        )
        if not candidate.exists():
            return candidate
        index += 1


def _load_attempt_artifacts(attempt_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    summary_path, sanity_path = attempt_dir / "summary.json", attempt_dir / "sanity.json"
    if not summary_path.is_file() or not sanity_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(summary, dict) or not isinstance(sanity, dict):
        return None
    return summary, sanity


def _ordered_pending(selected: Mapping[str, Sequence[Mapping[str, Any]]], accepted: Mapping[str, set[str]], rejected: Mapping[str, set[str]]) -> tuple[str, dict[str, Any]] | None:
    for split, rules in D5_SPLIT_RULES.items():
        by_id = {str(item["short_episode_id"]): dict(item) for item in selected[split]}
        ordered = [*rules["required_ids"], *sorted(identifier for identifier in by_id if identifier not in rules["required_ids"])]
        for identifier in ordered:
            if identifier not in accepted[split] and identifier not in rejected[split]:
                return split, by_id[identifier]
    return None


def _target_count(split: str) -> int:
    return int(D5_SPLIT_RULES[split]["per_category"]) * 3


def _ensure_initial_selection_matches(selection: Mapping[str, Any], selected: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    splits = selection.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("selection manifest lacks splits")
    for split, manifest_name in (("train", "train"), ("seen-val", "seen_val")):
        source = splits.get(manifest_name)
        if not isinstance(source, Mapping) or not isinstance(source.get("primary"), list):
            raise ValueError(f"selection manifest lacks {manifest_name}.primary")
        expected = {str(item["short_episode_id"]) for item in selected[split]}
        observed = {str(item["short_episode_id"]) for item in source["primary"]}
        if expected != observed:
            raise ValueError(f"selection manifest does not match deterministic D5 {split} selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--assignment-overrides", type=Path,
        help="optional audited D5 collection split overrides; original short-dataset splits remain immutable",
    )
    parser.add_argument(
        "--m4-log-root",
        type=Path,
        default=Path("/mnt/wxh/go2_short_vln/outputs/m4/logs"),
        help="immutable per-episode M4 runner logs, used only for zero-frame recovery",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--resume", action="store_true", help="resume an interrupted D5 collection manifest")
    parser.add_argument(
        "--selection-dry-run", action="store_true",
        help="print the next bounded D5 selection decision without writing or running Isaac",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    selection_path = args.selection.resolve()
    attempt_root, accepted_root, manifest_path = (
        args.attempt_root.resolve(), args.accepted_root.resolve(), args.manifest.resolve()
    )
    runner = args.runner.resolve()
    m4_log_root = args.m4_log_root.resolve()
    if not dataset.is_file() or not selection_path.is_file() or not runner.is_file():
        raise FileNotFoundError("D5 dataset, selection, or M4 runner does not exist")
    if not args.resume and (manifest_path.exists() or attempt_root.exists() or accepted_root.exists()):
        raise FileExistsError("D5 collection roots and manifest must be new and empty")
    # The dry run is intentionally read-only; every real collection process is
    # locked before it can recover, classify, or promote any attempt evidence.
    lock_stream = None if args.selection_dry_run else _acquire_collection_lock(manifest_path)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(episodes, list) or not isinstance(selection, Mapping):
        raise ValueError("invalid D5 dataset or selection manifest")
    splits = selection.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("selection manifest lacks splits")
    base_ids: dict[str, set[str]] = {}
    for split, manifest_name in (("train", "train"), ("seen-val", "seen_val")):
        source = splits.get(manifest_name)
        primary = source.get("primary") if isinstance(source, Mapping) else None
        if not isinstance(primary, list):
            raise ValueError(f"selection manifest lacks {manifest_name}.primary")
        base_ids[split] = {str(item["short_episode_id"]) for item in primary}
    assignment_policy = _assignment_policy(
        args.assignment_overrides, dataset=dataset, episodes=episodes, base_ids=base_ids
    )
    ruleset = collection_ruleset()
    recovery_events: list[dict[str, Any]] = []
    provenance_migrations: list[dict[str, Any]] = []
    selection_events: list[dict[str, Any]] = []
    selection_block: dict[str, Any] | None = None
    resume_count = 0
    if args.resume:
        if not manifest_path.is_file() or not attempt_root.is_dir() or not accepted_root.is_dir():
            raise FileNotFoundError("D5 resume requires the existing manifest and collection roots")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("invalid D5 resume manifest")
        expected = {
            "dataset": str(dataset), "dataset_sha256": sha256(dataset),
            "initial_selection": str(selection_path), "initial_selection_sha256": sha256(selection_path),
            "runner": str(runner), "runner_sha256": sha256(runner),
            "attempt_root": str(attempt_root), "accepted_root": str(accepted_root), "seed": args.seed,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("D5 resume manifest provenance does not match current inputs")
        pending_migrations: list[dict[str, Any]] = []
        existing_policy = manifest.get("assignment_overrides")
        if existing_policy is None and assignment_policy["assignments"]:
            pending_migrations.append({
                "kind": "additive_assignment_override",
                "previous_assignment_overrides": None,
                "assignment_overrides": assignment_policy,
            })
        elif existing_policy != assignment_policy:
            raise ValueError("D5 resume assignment override provenance does not match current inputs")
        existing_ruleset = manifest.get("collection_ruleset")
        if existing_ruleset is None:
            pending_migrations.append({
                "kind": "approved_train_scene_threshold_change",
                "train_min_scenes": {"previous": 9, "current": 8},
                "authority": "user_direction_2026-09-02",
            })
        elif existing_ruleset != ruleset:
            raise ValueError("D5 resume collection ruleset does not match the recorded acceptance criteria")
        if pending_migrations:
            snapshot = None
            if not args.selection_dry_run:
                snapshot = next_blocked_snapshot_path(manifest_path)
                _write_json_atomically(snapshot, manifest)
            for migration in pending_migrations:
                migration["resume_snapshot"] = str(snapshot) if snapshot is not None else None
            provenance_migrations.extend(pending_migrations)
        if manifest.get("status") == "blocked_infrastructure_failure":
            blocked_snapshot = next_blocked_snapshot_path(manifest_path)
            _write_json_atomically(blocked_snapshot, manifest)
            final_attempt = manifest.get("attempts", [])[-1] if isinstance(manifest.get("attempts"), list) and manifest.get("attempts") else None
            if not isinstance(final_attempt, dict):
                raise ValueError("blocked D5 manifest has no final attempt")
            apply_large_orientation_recovery(
                manifest, Path(str(final_attempt["attempt_dir"])), m4_log_root=m4_log_root
            )
        elif manifest.get("status") not in {"collecting", "blocked_selection_infeasible", "blocked_attempt_budget", "blocked_coverage_budget"}:
            raise ValueError("D5 resume only accepts collecting or recoverable D5 collection manifests")
        accepted_payload, rejected_payload, attempts_payload = (
            manifest.get("accepted_ids"), manifest.get("rejected_ids"), manifest.get("attempts")
        )
        if not isinstance(accepted_payload, dict) or not isinstance(rejected_payload, dict) or not isinstance(attempts_payload, list):
            raise ValueError("D5 resume manifest lacks collection state")
        accepted = {split: {str(value) for value in accepted_payload.get(split, [])} for split in D5_SPLIT_RULES}
        rejected = {split: {str(value) for value in rejected_payload.get(split, [])} for split in D5_SPLIT_RULES}
        attempts = [dict(value) for value in attempts_payload if isinstance(value, dict)]
        if len(attempts) != len(attempts_payload) or any(accepted[split] & rejected[split] for split in D5_SPLIT_RULES):
            raise ValueError("D5 resume manifest has invalid accepted/rejected state")
        for split, identifiers in accepted.items():
            for identifier in identifiers:
                episode_dir = accepted_root / identifier
                if not (episode_dir / "summary.json").is_file() or not (episode_dir / "sanity.json").is_file():
                    raise ValueError(f"accepted D5 evidence is incomplete: {split}/{identifier}")
        attempt_counts = {split: {category: 0 for category in ("straight", "left_turn", "right_turn")} for split in D5_SPLIT_RULES}
        for attempt in attempts:
            split, category = attempt.get("split"), attempt.get("category")
            if split not in D5_SPLIT_RULES or category not in attempt_counts[split]:
                raise ValueError("D5 resume manifest has invalid attempt category")
            attempt_counts[split][category] += 1
        if manifest.get("attempt_counts") != attempt_counts:
            raise ValueError("D5 resume attempt counts do not match attempt history")
        events = manifest.get("recovery_events", [])
        if not isinstance(events, list):
            raise ValueError("D5 resume recovery_events must be a list")
        recovery_events = [dict(event) for event in events if isinstance(event, dict)]
        raw_migrations = manifest.get("provenance_migrations", [])
        if not isinstance(raw_migrations, list) or any(not isinstance(event, dict) for event in raw_migrations):
            raise ValueError("D5 resume provenance_migrations must be a list of objects")
        provenance_migrations = [dict(event) for event in raw_migrations] + provenance_migrations
        raw_selection_events = manifest.get("selection_events", [])
        if not isinstance(raw_selection_events, list) or any(not isinstance(event, dict) for event in raw_selection_events):
            raise ValueError("D5 resume selection_events must be a list of objects")
        selection_events = [dict(event) for event in raw_selection_events]
        raw_selection_block = manifest.get("selection_block")
        if raw_selection_block is not None and not isinstance(raw_selection_block, dict):
            raise ValueError("D5 resume selection_block must be an object or null")
        selection_block = dict(raw_selection_block) if raw_selection_block is not None else None
        resume_count = int(manifest.get("resume_count", 0)) + 1
    else:
        accepted = {split: set() for split in D5_SPLIT_RULES}
        rejected = {split: set() for split in D5_SPLIT_RULES}
        attempt_counts = {split: {category: 0 for category in ("straight", "left_turn", "right_turn")} for split in D5_SPLIT_RULES}
        attempts = []
        selected = reselect_d5_splits(episodes, accepted_ids=accepted, rejected_ids=rejected, seed=args.seed)
        _ensure_initial_selection_matches(selection, selected)
        attempt_root.mkdir(parents=True)
        accepted_root.mkdir(parents=True)

    def persist(status: str) -> None:
        _write_json_atomically(manifest_path, {
            "format": "go2-short-vln-m6_2-d5-collection-v1",
            "status": status,
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "initial_selection": str(selection_path),
            "initial_selection_sha256": sha256(selection_path),
            "seed": args.seed,
            "runner": str(runner),
            "runner_sha256": sha256(runner),
            "attempt_root": str(attempt_root),
            "accepted_root": str(accepted_root),
            "assignment_overrides": assignment_policy,
            "collection_ruleset": ruleset,
            "accepted_ids": {split: sorted(values) for split, values in accepted.items()},
            "rejected_ids": {split: sorted(values) for split, values in rejected.items()},
            "attempt_counts": attempt_counts,
            "attempts": attempts,
            "active_selection": {split: sorted(values) for split, values in base_ids.items()},
            "accepted_summary": collection_accepted_summary(episodes, accepted, assignment_policy),
            "resume_count": resume_count,
            "recovery_events": recovery_events,
            "provenance_migrations": provenance_migrations,
            "selection_events": selection_events,
            "selection_block": selection_block,
        })

    if args.selection_dry_run:
        print(json.dumps(next_collection_decision(
            episodes, base_ids=base_ids, accepted=accepted, rejected=rejected, seed=args.seed,
            assignment_policy=assignment_policy,
        ), indent=2, allow_nan=False, default=sorted))
        return

    try:
        if args.resume:
            persist("collecting")

        while True:
            decision = next_collection_decision(
                episodes, base_ids=base_ids, accepted=accepted, rejected=rejected, seed=args.seed,
                assignment_policy=assignment_policy,
            )
            if decision["kind"] == "complete":
                selection_block = None
                persist("complete")
                print(json.dumps({"status": "complete", "accepted": {split: len(values) for split, values in accepted.items()}, "manifest": str(manifest_path)}, indent=2))
                return
            if decision["kind"] == "blocked":
                selection_block = {key: value for key, value in decision.items() if key != "feature"}
                reason = str(decision["reason"])
                if reason in {"coverage_budget_exhausted", "accepted_cap_exceeded", "total_accepted_cap_exceeded"}:
                    status = "blocked_coverage_budget"
                else:
                    status = "blocked_selection_infeasible"
                persist(status)
                raise RuntimeError(f"D5 {status}: {reason}")

            split = str(decision["split"])
            feature = dict(decision["feature"])
            identifier, category = str(feature["short_episode_id"]), str(feature["category"])
            limit = D5_ATTEMPT_MULTIPLIER * int(D5_SPLIT_RULES[split]["per_category"])
            if attempt_counts[split][category] >= limit:
                selection_block = {"split": split, "category": category, "attempt_limit": limit}
                persist("blocked_attempt_budget")
                raise RuntimeError(f"D5 attempt budget reached for {split}/{category}: {limit}")
            selection_events.append(selection_event(decision, event_id=len(selection_events) + 1))
            selection_block = None
            persist("collecting")
            attempt_dir = attempt_root / identifier
            if attempt_dir.exists():
                raise FileExistsError(f"refusing to reuse attempted D5 episode directory: {attempt_dir}")
            completed = subprocess.run([str(runner), identifier, str(attempt_root)], check=False).returncode
            attempt_counts[split][category] += 1
            artifacts = _load_attempt_artifacts(attempt_dir)
            if artifacts is None:
                attempt_decision = {"kind": "infrastructure_failure", "reason": "missing_or_invalid_attempt_artifacts"}
                summary, sanity = {}, {}
            else:
                summary, sanity = artifacts
                attempt_decision = classify_collection_attempt(summary, sanity)
            attempt = {
                "short_episode_id": identifier, "split": split, "category": category,
                "source_split": feature.get("source_split", split),
                "assignment_override": feature.get("assignment_override"),
                "scene_id": feature["scene_id"], "heading_bin": feature["heading_bin"],
                "distance_bin": feature["distance_bin"], "runner_returncode": completed,
                "decision": attempt_decision, "attempt_dir": str(attempt_dir), "summary": summary,
            }
            if attempt_decision["kind"] == "accepted":
                sidecar = write_assignment_sidecar(attempt_dir, feature, assignment_policy)
                if sidecar is not None:
                    attempt["assignment_sidecar"] = str(sidecar)
                accepted_dir = promote_accepted_attempt(attempt_dir, accepted_root)
                accepted[split].add(identifier)
                attempt["accepted_dir"] = str(accepted_dir)
            elif attempt_decision["kind"] in REPLACEABLE_REJECT_KINDS:
                rejected[split].add(identifier)
            else:
                attempts.append(attempt)
                persist("blocked_infrastructure_failure")
                raise RuntimeError(f"D5 infrastructure failure for {identifier}: {attempt_decision['reason']}")
            attempts.append(attempt)
            persist("collecting")
    finally:
        if lock_stream is not None:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()


if __name__ == "__main__":
    main()
