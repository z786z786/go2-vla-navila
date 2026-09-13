#!/usr/bin/env python3
"""Build the P4-T2 NaVILA R2R record index and whole-scene holdout split.

The script intentionally has no command-line arguments.  It verifies the two
dispatch-prescribed inputs, parses the large annotations array with
``JSONDecoder.raw_decode`` in bounded chunks, and writes all requested outputs.
It never opens the extracted frame files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


ROOT = Path("/home/wxh/go2_short_vln")
PROBE_ROOT = Path("/mnt/wxh/go2_short_vln")
ANN_PATH = Path("/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_annotations.json")
TRAIN_PATH = Path("/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz")
INDEX_DIR = Path("/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index")
INDEX_PATH = INDEX_DIR / "t2_records.jsonl"
INDEX_SHA_PATH = INDEX_DIR / "t2_records.sha256"
SUMMARY_PATH = ROOT / "reports/p4/t2_index_summary.json"
SPLIT_PATH = ROOT / "reports/p4/t2_scene_split.json"
MARKDOWN_PATH = ROOT / "reports/p4/T2_INDEX_SPLIT.md"

EXPECTED_ANN_SHA = "3587c020cc03807cfc03c454bcbb37a3d409139e125e1ea047fa1fbcf8c4cc1a"
EXPECTED_TRAIN_SHA = "f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34"
EXPECTED_ANN_BYTES = 317267756
SEED = 20260913
SPARSE_THRESHOLD = 25

# This is deliberately explicit and stable.  The IDs follow the lexical
# vocabulary order used by the verified P3 full-dataset action distribution.
ACTION_ID = {
    "I think I should stop because I have finished the instruction.": 0,
    "The next action is move forward 25 cm.": 1,
    "The next action is move forward 50 cm.": 2,
    "The next action is move forward 75 cm.": 3,
    "The next action is turn left 15 degree.": 4,
    "The next action is turn left 30 degree.": 5,
    "The next action is turn left 45 degree.": 6,
    "The next action is turn right 15 degree.": 7,
    "The next action is turn right 30 degree.": 8,
    "The next action is turn right 45 degree.": 9,
}
ACTION_LABELS = tuple(ACTION_ID)


def normalized_instruction(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def verify_inputs() -> dict[str, dict[str, Any]]:
    measured: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for label, path, expected_sha, expected_bytes in (
        ("navila_r2r_annotations_full", ANN_PATH, EXPECTED_ANN_SHA, EXPECTED_ANN_BYTES),
        ("r2r_train", TRAIN_PATH, EXPECTED_TRAIN_SHA, None),
    ):
        actual_sha, actual_bytes = sha256_and_size(path)
        measured[str(path)] = {
            "label": label,
            "sha256": actual_sha,
            "bytes": actual_bytes,
        }
        if actual_sha != expected_sha:
            failures.append(f"{path}: sha256 expected {expected_sha}, measured {actual_sha}")
        if expected_bytes is not None and actual_bytes != expected_bytes:
            failures.append(f"{path}: bytes expected {expected_bytes}, measured {actual_bytes}")
    if failures:
        raise RuntimeError("Input verification failed; no outputs were written:\n" + "\n".join(failures))
    return measured


def load_train(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    # The gzipped R2R split is small; the dispatch only prohibits json.load on
    # the 317 MB annotations array.
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError(f"{path} does not have a top-level episodes list")
    episode_to_scene: dict[str, str] = {}
    episode_to_instruction: dict[str, str] = {}
    for index, episode in enumerate(payload["episodes"], start=1):
        try:
            episode_id = str(episode["episode_id"])
            scene_id = episode["scene_id"]
            instruction = episode["instruction"]["instruction_text"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"train episode {index} lacks episode_id/scene_id/instruction_text") from exc
        if not isinstance(scene_id, str) or not isinstance(instruction, str):
            raise ValueError(f"train episode {index} has non-string scene_id or instruction_text")
        if episode_id in episode_to_scene:
            raise ValueError(f"duplicate train episode_id {episode_id!r}")
        episode_to_scene[episode_id] = scene_id
        episode_to_instruction[episode_id] = instruction
    return episode_to_scene, episode_to_instruction


def iter_annotation_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield a single-line/large JSON array incrementally with bounded memory."""
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    buffer = ""
    position = 0
    started = False
    expect_value = False
    completed = False
    eof = False
    with path.open("rt", encoding="utf-8") as source:
        while True:
            if not eof:
                chunk = source.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while not completed:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position == len(buffer):
                    break
                if not started:
                    if buffer[position] != "[":
                        raise ValueError("annotations JSON must begin with an array")
                    started = True
                    expect_value = True
                    position += 1
                    continue
                if expect_value:
                    if buffer[position] == "]":
                        completed = True
                        position += 1
                        continue
                    try:
                        record, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError:
                        if eof:
                            raise ValueError("truncated JSON while decoding annotations")
                        break
                    if not isinstance(record, dict):
                        raise ValueError("annotations array contains a non-object record")
                    yield record
                    position = end
                    expect_value = False
                    continue
                if buffer[position] == ",":
                    position += 1
                    expect_value = True
                    continue
                if buffer[position] == "]":
                    completed = True
                    position += 1
                    continue
                raise ValueError(f"expected ',' or ']' after annotation at character {position}")

            # Discard processed text.  Keep only an incomplete record prefix,
            # which is bounded by the chunk size plus one record.
            if position:
                buffer = buffer[position:]
                position = 0
            if eof:
                if not completed:
                    raise ValueError("annotations JSON ended before its closing array bracket")
                if buffer.strip():
                    raise ValueError("non-whitespace data appears after annotations array")
                return


def parse_video_id(video_id: Any, record_number: int) -> tuple[str, int]:
    if not isinstance(video_id, str):
        raise ValueError(f"annotation record {record_number} has non-string video_id")
    video, separator, step_text = video_id.rpartition("-")
    if not separator or not video:
        raise ValueError(f"annotation record {record_number} has invalid video_id {video_id!r}")
    try:
        step = int(step_text)
    except ValueError as exc:
        raise ValueError(f"annotation record {record_number} has non-integer step in {video_id!r}") from exc
    if step < 0:
        raise ValueError(f"annotation record {record_number} has negative step in {video_id!r}")
    return video, step


def validate_record(
    record: dict[str, Any],
    record_number: int,
    episode_to_scene: dict[str, str],
    episode_to_instruction: dict[str, str],
) -> tuple[str, int, str, str, list[str]]:
    try:
        video_id = record["video_id"]
        instruction = record["q"]
        action = record["a"]
        frames = record["frames"]
    except KeyError as exc:
        raise ValueError(f"annotation record {record_number} lacks required field {exc}") from exc
    video, step = parse_video_id(video_id, record_number)
    if not isinstance(instruction, str) or not isinstance(action, str):
        raise ValueError(f"annotation record {record_number} has non-string q/a")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"annotation record {record_number} has empty/non-list frames")
    if video not in episode_to_scene:
        raise ValueError(
            f"annotation record {record_number} video {video!r} has no matching train episode_id; aborting"
        )
    expected_instruction = episode_to_instruction[video]
    if instruction != expected_instruction:
        raise ValueError(
            f"annotation record {record_number} instruction mismatch for video {video!r}: "
            "q is not byte-for-byte equal to train instruction_text"
        )
    for frame_index, frame_path in enumerate(frames):
        expected_path = f"{video}/frame_{frame_index}.jpg"
        if frame_path != expected_path:
            raise ValueError(
                f"annotation record {record_number} frame path mismatch at index {frame_index}: "
                f"expected {expected_path!r}, got {frame_path!r}"
            )
    if action not in ACTION_ID:
        raise ValueError(f"annotation record {record_number} has unknown action string {action!r}")
    return video, step, instruction, action, frames


def first_pass(
    episode_to_scene: dict[str, str],
    episode_to_instruction: dict[str, str],
) -> tuple[int, Counter[str], Counter[str], Counter[str], dict[str, str]]:
    record_count = 0
    action_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    video_id_counts: Counter[str] = Counter()
    video_scene: dict[str, str] = {}
    for record_count, record in enumerate(iter_annotation_records(ANN_PATH), start=1):
        video, _step, _instruction, action, _frames = validate_record(
            record, record_count, episode_to_scene, episode_to_instruction
        )
        video_id_counts[record["video_id"]] += 1
        video_counts[video] += 1
        action_counts[action] += 1
        scene = episode_to_scene[video]
        prior_scene = video_scene.setdefault(video, scene)
        if prior_scene != scene:
            raise ValueError(f"video {video!r} maps to multiple scenes")
    annotation_videos = set(video_counts)
    train_videos = set(episode_to_scene)
    if annotation_videos != train_videos:
        missing_from_annotations = sorted(train_videos - annotation_videos)
        missing_from_train = sorted(annotation_videos - train_videos)
        raise ValueError(
            "video↔episode mapping is not a complete bijection: "
            f"missing annotation videos={missing_from_annotations[:5]} (count {len(missing_from_annotations)}), "
            f"missing train episodes={missing_from_train[:5]} (count {len(missing_from_train)})"
        )
    if record_count != 353894 or len(video_counts) != 10819:
        raise ValueError(
            f"unexpected verified counts: records={record_count}, videos={len(video_counts)}; "
            "dispatch requires 353894 records and 10819 videos"
        )
    if set(video_id_counts.values()) - {1, 2, 3}:
        raise ValueError("unexpected annotation video_id multiplicity outside {1,2,3}")
    if Counter(video_id_counts.values()) != Counter({1: 234113, 2: 43662, 3: 10819}):
        raise ValueError(
            "unexpected video_id multiplicity distribution: "
            f"{dict(sorted(Counter(video_id_counts.values()).items()))}"
        )
    return record_count, action_counts, video_counts, video_id_counts, video_scene


def stable_scene_rank(scene_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{scene_id}".encode("utf-8")).hexdigest()


def choose_holdout_scenes(scene_to_videos: dict[str, set[str]], total_videos: int) -> tuple[set[str], dict[str, Any]]:
    sparse = {scene for scene, videos in scene_to_videos.items() if len(videos) < SPARSE_THRESHOLD}
    eligible = sorted(set(scene_to_videos) - sparse, key=lambda scene: (stable_scene_rank(scene), scene))
    target = int(round(total_videos * 0.10))

    # Subset-sum over whole scenes.  A seeded scene order supplies a stable
    # tie-break while keeping the achieved video count as close as possible to
    # the nearest-integer 10% target.
    states: dict[int, tuple[str, ...]] = {0: ()}
    for scene in eligible:
        count = len(scene_to_videos[scene])
        prior = list(states.items())
        for current, chosen in prior:
            candidate_total = current + count
            if candidate_total not in states:
                states[candidate_total] = chosen + (scene,)

    def choice_key(item: tuple[int, tuple[str, ...]]) -> tuple[Any, ...]:
        amount, chosen = item
        return (abs(amount - target), amount > target, stable_scene_rank("|".join(sorted(chosen))), tuple(sorted(chosen)))

    achieved, chosen_tuple = min(states.items(), key=choice_key)
    holdout = set(chosen_tuple)
    if sparse & holdout:
        raise AssertionError("sparse scenes must never enter holdout")
    decision = {
        "seed": SEED,
        "target_holdout_fraction": 0.10,
        "target_holdout_video_count": target,
        "achieved_holdout_video_count": achieved,
        "achieved_holdout_fraction": achieved / total_videos,
        "sparse_scene_threshold_exclusive": SPARSE_THRESHOLD,
        "sparse_scene_count": len(sparse),
        "eligible_scene_count": len(eligible),
        "algorithm": "seeded whole-scene subset-sum; minimize absolute video-count error, then prefer under-target and seeded digest tie-break",
    }
    return holdout, decision


def action_distribution(counts: Counter[str], total: int) -> dict[str, Any]:
    ordered_counts = {label: counts.get(label, 0) for label in ACTION_LABELS}
    if total <= 0:
        majority_label = None
        majority_count = 0
    else:
        majority_label, majority_count = min(
            ((label, -count) for label, count in ordered_counts.items()),
            key=lambda item: (item[1], item[0]),
        )
        majority_count = -majority_count
    return {
        "total_record_count": total,
        "action_counts": ordered_counts,
        "majority_class": {
            "action": majority_label,
            "count": majority_count,
            "share_fraction": (majority_count / total) if total else 0.0,
            "share_percent": (100.0 * majority_count / total) if total else 0.0,
        },
    }


def scene_stem(scene_id: str) -> str:
    return PurePosixPath(scene_id).stem


def build_index_and_split(
    episode_to_scene: dict[str, str],
    episode_to_instruction: dict[str, str],
    video_counts: Counter[str],
    video_id_counts: Counter[str],
    video_scene: dict[str, str],
    full_record_count: int,
    full_action_counts: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_to_videos: dict[str, set[str]] = defaultdict(set)
    for video, scene in video_scene.items():
        scene_to_videos[scene].add(video)
    holdout_scenes, selection = choose_holdout_scenes(scene_to_videos, len(video_counts))
    holdout_videos = {video for scene in holdout_scenes for video in scene_to_videos[scene]}
    train_videos = set(video_counts) - holdout_videos
    if holdout_scenes & (set(scene_to_videos) - holdout_scenes):
        raise AssertionError("scene split construction failed")
    if holdout_videos & train_videos:
        raise AssertionError("video split intersection must be empty")

    side_actions: dict[str, Counter[str]] = {"train": Counter(), "holdout": Counter()}
    side_records: Counter[str] = Counter()
    side_scene_ids: dict[str, set[str]] = {"train": set(), "holdout": set()}
    # The requested artifact belongs on /mnt.  Fail explicitly if the formal
    # destination is unavailable; never silently stage a replacement on /tmp.
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        probe = PROBE_ROOT / f".probe_t2_{os.getpid()}"
        with probe.open("w", encoding="ascii") as handle:
            handle.write("probe\n")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"requested /mnt index directory is not writable: {exc}") from exc
    write_permission_probe = {
        "status": "PASS",
        "path_pattern": str(PROBE_ROOT / ".probe_t2_<pid>"),
        "verified_write": True,
        "verified_deleted": True,
    }
    index_write_path = INDEX_PATH
    index_sha_path = INDEX_SHA_PATH
    index_status = "available"
    temp_index = INDEX_PATH.with_name(INDEX_PATH.name + ".tmp")
    with temp_index.open("w", encoding="utf-8", newline="\n") as destination:
        for record_number, record in enumerate(iter_annotation_records(ANN_PATH), start=1):
            video, step, instruction, action, frames = validate_record(
                record, record_number, episode_to_scene, episode_to_instruction
            )
            multiplicity = video_id_counts[record["video_id"]]
            side = "holdout" if video in holdout_videos else "train"
            side_records[side] += 1
            side_actions[side][action] += 1
            side_scene_ids[side].add(video_scene[video])
            row = {
                "video_id": record["video_id"],
                "video": video,
                "step": step,
                "scene_id": video_scene[video],
                "instruction_raw": instruction,
                "instruction_normalized": normalized_instruction(instruction),
                "action_text": action,
                "action_id": ACTION_ID[action],
                "n_frames": len(frames),
                "frames_first": frames[0],
                "frames_last": frames[-1],
                "multiplicity": multiplicity,
                "is_oversampled_copy": multiplicity > 1,
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp_index, index_write_path)

    index_sha, index_bytes = sha256_and_size(index_write_path)
    index_sha_path.write_text(f"{index_sha}  {INDEX_PATH.name}\n", encoding="ascii")
    index_artifact = {
        "path": str(INDEX_PATH),
        "sha256": index_sha,
        "bytes": index_bytes,
        "sha256_sidecar": str(INDEX_SHA_PATH),
        "line_count": full_record_count,
        "status": index_status,
    }
    missing_items: list[dict[str, Any]] = []
    missing_items.append(
        {
            "item": "/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg existence",
            "status": "MISSING",
            "reason": "not checked by P4-T2 because the dispatch assigns frame existence to P4-T1 and forbids reading extracted frames",
        }
    )

    sparse_disposition = []
    for scene in sorted(scene_to_videos, key=lambda value: (len(scene_to_videos[value]), value)):
        count = len(scene_to_videos[scene])
        if count < SPARSE_THRESHOLD:
            sparse_disposition.append(
                {
                    "scene_id": scene,
                    "scene_stem": scene_stem(scene),
                    "video_count": count,
                    "assigned_side": "train",
                    "reason": "fewer than 25 videos; keep the entire tiny scene in training so a single sparse scene does not dominate the holdout baseline",
                }
            )

    def side_payload(name: str, videos: set[str]) -> dict[str, Any]:
        records = side_records[name]
        return {
            "scene_ids": sorted(side_scene_ids[name]),
            "scene_stems": sorted(scene_stem(scene) for scene in side_scene_ids[name]),
            "videos": sorted(videos),
            "video_count": len(videos),
            "record_count": records,
            "action_distribution": action_distribution(side_actions[name], records),
            "majority_class_share_percent": action_distribution(side_actions[name], records)["majority_class"]["share_percent"],
        }

    split_payload = {
        "task_id": "P4-T2",
        "generated_at": deterministic_generated_at(),
        "generator_command": "python3 /home/wxh/go2_short_vln/scripts/p4_build_navila_index.py",
        "write_permission_probe": write_permission_probe,
        "inputs": {
            str(ANN_PATH): {"sha256": EXPECTED_ANN_SHA, "bytes": EXPECTED_ANN_BYTES},
            str(TRAIN_PATH): {"sha256": EXPECTED_TRAIN_SHA, "bytes": TRAIN_PATH.stat().st_size},
        },
        "seed": SEED,
        "target_holdout_fraction": 0.10,
        "selection": selection,
        "train": side_payload("train", train_videos),
        "holdout": side_payload("holdout", holdout_videos),
        "scene_intersection": [],
        "video_intersection": [],
        "sparse_scene_disposition": sparse_disposition,
        "sparse_scene_policy": "All scenes with fewer than 25 videos are explicitly assigned to train.",
        "missing": missing_items,
    }

    full_distribution = action_distribution(full_action_counts, full_record_count)
    multiplicity_video_counts = Counter(video_id_counts.values())
    multiplicity_record_counts = Counter()
    for multiplicity, count in multiplicity_video_counts.items():
        multiplicity_record_counts[multiplicity] = multiplicity * count
    summary_payload = {
        "task_id": "P4-T2",
        "generated_at": deterministic_generated_at(),
        "generator_command": "python3 /home/wxh/go2_short_vln/scripts/p4_build_navila_index.py",
        "write_permission_probe": write_permission_probe,
        "inputs": {
            str(ANN_PATH): {"sha256": EXPECTED_ANN_SHA, "bytes": EXPECTED_ANN_BYTES},
            str(TRAIN_PATH): {"sha256": EXPECTED_TRAIN_SHA, "bytes": TRAIN_PATH.stat().st_size},
        },
        "record_count": full_record_count,
        "distinct_video_count": len(video_counts),
        "distinct_video_id_count": len(video_id_counts),
        "scene_count": len(scene_to_videos),
        "action_id_mapping": {str(identifier): label for label, identifier in ACTION_ID.items()},
        "action_distribution": full_distribution,
        "majority_class_share_percent": full_distribution["majority_class"]["share_percent"],
        "scene_distribution": {
            scene: {
                "scene_stem": scene_stem(scene),
                "video_count": len(scene_to_videos[scene]),
                "record_count": sum(video_counts[video] for video in scene_to_videos[scene]),
            }
            for scene in sorted(scene_to_videos)
        },
        "multiplicity": {
            "videos_by_multiplicity": {str(key): multiplicity_video_counts[key] for key in sorted(multiplicity_video_counts)},
            "records_by_multiplicity": {str(key): multiplicity_record_counts[key] for key in sorted(multiplicity_record_counts)},
            "is_oversampled_copy_definition": "true exactly when multiplicity > 1; all records of an overrepresented video retain that video's multiplicity",
        },
        "frame_path_reconstruction": {
            "source_annotation_rule": "for each row, source frame paths are exactly `<video>/frame_<i>.jpg` for integer i in [0, n_frames-1]",
            "unpacked_absolute_rule": "/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg",
            "index_fields": "frames_first is `<video>/frame_0.jpg`; frames_last is `<video>/frame_<n_frames-1>.jpg`; no full frame list is stored",
            "frame_count_total_expected": 601125,
            "frame_file_paths_stored_in_index": False,
            "frame_existence_verification": {
                "status": "MISSING",
                "reason": "P4-T2 did not read extracted frames; P4-T1 owns frame existence verification per dispatch",
            },
        },
        "index_artifact": index_artifact,
        "scene_split": {
            "seed": SEED,
            "train_scene_count": len(side_scene_ids["train"]),
            "holdout_scene_count": len(side_scene_ids["holdout"]),
            "train_video_count": len(train_videos),
            "holdout_video_count": len(holdout_videos),
            "train_record_count": side_records["train"],
            "holdout_record_count": side_records["holdout"],
            "scene_intersection": [],
            "video_intersection": [],
        },
        "missing": missing_items,
    }
    write_json(SUMMARY_PATH, summary_payload)
    write_json(SPLIT_PATH, split_payload)
    write_markdown(summary_payload, split_payload)
    return summary_payload, split_payload


def deterministic_generated_at() -> str:
    newest_ns = max(ANN_PATH.stat().st_mtime_ns, TRAIN_PATH.stat().st_mtime_ns)
    return datetime.fromtimestamp(newest_ns / 1_000_000_000, timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def write_markdown(summary: dict[str, Any], split: dict[str, Any]) -> None:
    full = summary["action_distribution"]
    train = split["train"]
    holdout = split["holdout"]
    sparse_rows = "\n".join(
        f"| `{item['scene_stem']}` | {item['video_count']} | `{item['assigned_side']}` | {item['reason']} |"
        for item in split["sparse_scene_disposition"]
    )
    action_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in full["action_counts"].items()
    )
    if not summary["missing"]:
        missing_line = "none."
    else:
        missing_line = "; ".join(
            f"{item['item']} — MISSING ({item['reason']})"
            for item in summary["missing"]
        )
    text = f"""# P4-T2 — NaVILA R2R 逐记录索引与 scene 留出集

## 结果摘要

| 项目 | 值 |
|---|---:|
| 索引记录行数 | {summary['record_count']} |
| 不同 video | {summary['distinct_video_count']} |
| 不同 `video_id`（含 step） | {summary['distinct_video_id_count']} |
| scene | {summary['scene_count']} |
| 索引状态 | 已落盘 |
| 索引路径 | `{summary['index_artifact']['path']}` |
| 索引字节数 | {summary['index_artifact']['bytes']} |
| 索引行数 | {summary['index_artifact']['line_count']} |
| 索引 SHA-256 | `{summary['index_artifact']['sha256']}` |
| SHA-256 sidecar | `{summary['index_artifact']['sha256_sidecar']}` |
| `/mnt` 写权限探针 | PASS（写入并删除已核验） |
| 全量 majority action | `{full['majority_class']['action']}` |
| 全量 majority 占比 | {full['majority_class']['share_percent']:.6f}% |
| 全量 stop 占比 | {100.0 * full['action_counts']['I think I should stop because I have finished the instruction.'] / full['total_record_count']:.6f}% |
| 过采样 `video_id` 重数（重复记录组） | `{summary['multiplicity']['videos_by_multiplicity']}` |

输入在解析前已核对 SHA-256；未读取禁止使用的截断 annotations 副本，也未读取 T1 解包帧文件。

## 全量动作分布

| action_text | count |
|---|---:|
{action_rows}

`action_id` 映射在 `t2_index_summary.json` 中显式声明（0–9，固定词典顺序）。

## Scene 切分

固定 seed 为 `{split['seed']}`。按整 scene 切分，目标留出 video 数为
`{split['selection']['target_holdout_video_count']}`（全量 10% 的最近整数）；使用 seeded whole-scene subset-sum，
实际留出 `{holdout['video_count']}` 个 video（{100.0 * holdout['video_count'] / summary['distinct_video_count']:.6f}%）。
训练/留出 scene 与 video 交集均为空。

| 侧 | records | videos | scenes | majority 占比 |
|---|---:|---:|---:|---:|
| train | {train['record_count']} | {train['video_count']} | {len(train['scene_ids'])} | {train['majority_class_share_percent']:.6f}% |
| holdout | {holdout['record_count']} | {holdout['video_count']} | {len(holdout['scene_ids'])} | {holdout['majority_class_share_percent']:.6f}% |

### 稀疏 scene（<25 videos）逐项处置

| scene stem | videos | 归属 | 理由 |
|---|---:|---|---|
{sparse_rows}

六个按字面 `<25 videos` 判定的稀疏 scene 全部固定在训练侧（其中派发单重点列出的五个最稀疏 scene 均包含在内），避免留出侧被单一小 scene 主导；机器可读的完整 scene ID、两侧 scene/video 清单和动作分布见 `t2_scene_split.json`。

## 帧路径重建

索引只保存 `n_frames`、`frames_first`、`frames_last`，不保存 601,125 条完整帧路径。对任意记录，原始标注路径为
`<video>/frame_<i>.jpg`，其中 `i = 0, ..., n_frames-1`；解包后的确定性绝对路径为
`/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg`。

## 复算与缺口

无参数重跑命令：`python3 /home/wxh/go2_short_vln/scripts/p4_build_navila_index.py`。

MISSING: {missing_line}
"""
    MARKDOWN_PATH.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    inputs = verify_inputs()
    # Keep the measured input hashes in the process-visible validation path;
    # output payloads use the same values after successful verification.
    if inputs[str(ANN_PATH)]["sha256"] != EXPECTED_ANN_SHA or inputs[str(TRAIN_PATH)]["sha256"] != EXPECTED_TRAIN_SHA:
        raise AssertionError("verified input hash unexpectedly changed")
    episode_to_scene, episode_to_instruction = load_train(TRAIN_PATH)
    if len(episode_to_scene) != 10819:
        raise ValueError(f"unexpected train episode count {len(episode_to_scene)}; expected 10819")
    full_count, full_actions, video_counts, video_id_counts, video_scene = first_pass(
        episode_to_scene, episode_to_instruction
    )
    build_index_and_split(
        episode_to_scene,
        episode_to_instruction,
        video_counts,
        video_id_counts,
        video_scene,
        full_count,
        full_actions,
    )


if __name__ == "__main__":
    main()
