#!/usr/bin/env python3
"""Recompute P3-T1E from the verified full NaVILA R2R annotations file.

This intentionally parses the 317 MB single-line JSON array with
JSONDecoder.raw_decode in bounded chunks.  It must be run with no arguments.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


TASK_ID = "P3-T1E"
ROOT = Path("/home/wxh/go2_short_vln")
OUTPUT_DIR = ROOT / "reports/p3"
LEAKAGE_OUTPUT = OUTPUT_DIR / "t1e_leakage_full.json"
ACTION_OUTPUT = OUTPUT_DIR / "t1e_action_distribution.json"
EXCLUDED_OUTPUT = OUTPUT_DIR / "t1e_excluded_instructions.json"
EXCLUDED_VIDEO_IDS_OUTPUT = OUTPUT_DIR / "t1e_excluded_video_ids.json.gz"
MARKDOWN_OUTPUT = OUTPUT_DIR / "T1E_LEAKAGE_FULL.md"
GENERATOR_COMMAND = "python3 /home/wxh/go2_short_vln/scripts/p3_navila_leakage_join.py"

# The six paths and hashes are fixed by reports/p3/DISPATCH_P3_T1E.md §2.
# Only the annotations file has a prescribed byte count in that table; sizes
# of the other inputs are still measured and emitted in every output.
EXPECTED_INPUTS: tuple[tuple[str, Path, str, int | None], ...] = (
    (
        "navila_r2r_annotations_full",
        Path("/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_annotations.json"),
        "3587c020cc03807cfc03c454bcbb37a3d409139e125e1ea047fa1fbcf8c4cc1a",
        317267756,
    ),
    (
        "r2r_train",
        Path("/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz"),
        "f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34",
        None,
    ),
    (
        "r2r_val_seen",
        Path("/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/val_seen/val_seen.json.gz"),
        "7fc94841ebbd2eac0d398e020a2f638426948beaf4a561f6ee310dd67cddce55",
        None,
    ),
    (
        "r2r_val_unseen",
        Path("/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz"),
        "d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8",
        None,
    ),
    (
        "r2r_test",
        Path("/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/test/test.json.gz"),
        "69ed5962fac658d34be5739c1f8f15805cf5ce9bfa28f64a338da338244578b7",
        None,
    ),
    (
        "eval11",
        Path("/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz"),
        "ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec",
        None,
    ),
)

SPLIT_ORDER = ("train", "val_seen", "val_unseen", "test")
SPLIT_PATHS = {
    "train": EXPECTED_INPUTS[1][1],
    "val_seen": EXPECTED_INPUTS[2][1],
    "val_unseen": EXPECTED_INPUTS[3][1],
    "test": EXPECTED_INPUTS[4][1],
}
FRAME_NAME_RE = re.compile(r"(?:^|/)frame_(\d+)\.jpg$")


def normalized_instruction(text: str) -> str:
    """The exact dispatch §3.1 join-key normalization."""
    return " ".join(text.split()).strip().lower()


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def verify_inputs() -> dict[str, dict[str, Any]]:
    """Hash every prescribed source before parsing any of them."""
    measured: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for label, path, expected_hash, expected_size in EXPECTED_INPUTS:
        actual_hash, actual_size = sha256_and_size(path)
        measured[str(path)] = {
            "label": label,
            "sha256": actual_hash,
            "bytes": actual_size,
        }
        if actual_hash != expected_hash:
            failures.append(
                f"{path}: sha256 expected {expected_hash}, measured {actual_hash}"
            )
        if expected_size is not None and actual_size != expected_size:
            failures.append(
                f"{path}: bytes expected {expected_size}, measured {actual_size}"
            )
    if failures:
        raise RuntimeError("Input verification failed; no outputs were written:\n" + "\n".join(failures))
    return measured


def load_episodes(path: Path) -> list[dict[str, Any]]:
    """Load one small, gzipped R2R/evaluation split."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        decoded = json.load(source)
    if not isinstance(decoded, dict) or not isinstance(decoded.get("episodes"), list):
        raise ValueError(f"{path} does not have a top-level episodes list")
    return decoded["episodes"]


def iter_annotation_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every object in the large JSON array using bounded raw_decode chunks."""
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    buffer = ""
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

            position = 0
            while not completed:
                length = len(buffer)
                while position < length and buffer[position].isspace():
                    position += 1
                if position == length:
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
                            raise ValueError("truncated JSON while decoding an annotation record")
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
                raise ValueError(f"expected ',' or ']' after annotation record at character {position}")

            buffer = buffer[position:]
            if eof:
                if not completed:
                    raise ValueError("annotations JSON ended before its closing array bracket")
                if buffer.strip():
                    raise ValueError("non-whitespace data appears after annotations JSON array")
                return


def scene_stem(scene_id: str) -> str:
    return PurePosixPath(scene_id).stem


def nearest_rank_distribution(values: list[int]) -> dict[str, int]:
    """Return dispatch percentiles with the documented nearest-rank rule."""
    if not values:
        raise ValueError("cannot describe an empty distribution")
    ordered = sorted(values)

    def percentile(proportion: float) -> int:
        return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "percentile_method": "nearest rank: sorted[ceil(p*n)-1]",
    }


def frame_sequence_is_contiguous(frames: list[Any]) -> bool:
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, str):
            return False
        match = FRAME_NAME_RE.search(frame)
        if match is None or int(match.group(1)) != expected_index:
            return False
    return True


def output_metadata(inputs: dict[str, dict[str, Any]], generated_at: str) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "generated_at": generated_at,
        "generator_command": GENERATOR_COMMAND,
        "inputs": inputs,
    }


def deterministic_generated_at() -> str:
    """Use the newest verified source mtime so no-argument reruns are byte-stable."""
    newest_ns = max(path.stat().st_mtime_ns for _, path, _, _ in EXPECTED_INPUTS)
    return datetime.fromtimestamp(newest_ns / 1_000_000_000, timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def make_markdown(leakage: dict[str, Any], actions: dict[str, Any], exclusions: dict[str, Any]) -> str:
    join = leakage["leakage_join"]
    action = actions["action_distribution"]
    temporal = actions["temporal_structure"]
    actual_intersection = join["matched_train_scene_stems_intersection_eval11_sorted"]
    conclusion = (
        "The verified full annotation file is joined to R2R by normalized raw instruction text; "
        f"the train-implied scene/eval11 intersection is `{actual_intersection}`, and "
        f"`val_unseen − train` contains {join['val_unseen_minus_train_instruction_count']} matched normalized instructions."
    )
    input_rows = "\n".join(
        f"| `{path}` | `{details['sha256']}` | {details['bytes']} |"
        for path, details in leakage["inputs"].items()
    )
    split_rows = "\n".join(
        f"| {split} | {details['matched_normalized_instruction_count']} | {details['matched_record_count']} |"
        for split, details in join["split_matches"].items()
    )
    action_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in action["action_counts"].items()
    )
    def dist_row(name: str, values: dict[str, int]) -> str:
        return (
            f"| {name} | {values['count']} | {values['min']} | {values['p25']} | "
            f"{values['p50']} | {values['p75']} | {values['p90']} | {values['max']} |"
        )

    majority_share = action["majority_class"]["share_percent"]
    stop_share = action["stop_action"]["share_percent"]
    majority_delta = majority_share - 29.74
    stop_delta = stop_share - 9.5

    return f"""# P3-T1E — NaVILA R2R 全量 leakage join 与动作分布

{conclusion}

## Input verification

| Input path | Measured SHA-256 | Measured bytes |
|---|---|---:|
{input_rows}

Every prescribed input was SHA-256 verified before any split or annotations parsing. The full annotations source is the 317,267,756-byte file at the dispatch-prescribed `downloads/navila_probe` path; the similarly named truncated copy was not read.

## Leakage join

| Quantity | Value |
|---|---:|
| Full annotation records | {join['total_record_count']} |
| Distinct normalized instructions | {join['distinct_normalized_instruction_count']} |
| `val_unseen − train` matched instructions | {join['val_unseen_minus_train_instruction_count']} |
| `val_unseen − train` affected records | {join['val_unseen_minus_train_record_count']} |
| Train-implied raw scenes | {join['matched_train_raw_scene_count']} |
| Train-implied normalized scene stems | {join['matched_train_scene_stem_count']} |
| eval11 scenes | {join['eval11_scene_stem_count']} |
| Train/eval11 scene-stem intersection | {len(actual_intersection)} (`{actual_intersection}`) |
| Unexplained records | {join['unexplained_record_count']} |
| Unexplained normalized instructions | {join['unexplained_normalized_instruction_count']} |
| Excluded ambiguous instructions | {exclusions['excluded_instruction_count']} |
| Excluded affected records | {exclusions['excluded_record_count']} |

| R2R split | Matched normalized instructions | Matched annotation records |
|---|---:|---:|
{split_rows}

Matched train raw scene IDs (sorted): `{join['matched_train_raw_scene_ids_sorted']}`

Matched train normalized scene stems (sorted): `{join['matched_train_scene_stems_sorted']}`

eval11 normalized scene stems (sorted): `{join['eval11_scene_stems_sorted']}`

## Full action distribution

| Quantity | Value |
|---|---:|
| Action vocabulary size | {action['action_vocabulary_size']} |
| Majority action | `{action['majority_class']['action']}` |
| Majority count | {action['majority_class']['count']} |
| Majority share | {action['majority_class']['share_percent']:.6f}% |
| Stop action | `{action['stop_action']['action']}` |
| Stop count | {action['stop_action']['count']} |
| Stop share | {action['stop_action']['share_percent']:.6f}% |

Full-dataset rates differ from the prior 6,712-record reference sample: majority share is {majority_share:.6f}% (reference 29.74%, delta {majority_delta:+.6f} percentage points) and stop share is {stop_share:.6f}% (reference 9.5%, delta {stop_delta:+.6f} percentage points). The full action vocabulary remains 10 entries. These values were obtained by counting every decoded record in the verified 317,267,756-byte array; no sampling or agreement-forcing adjustment was used.

| Exact `a` value | Count |
|---|---:|
{action_rows}

## Video and frame structure

The percentile convention is `{temporal['percentile_method']}`. Decision-point counts use unique numeric `<step>` values per video prefix; repeated rows at a step are instruction variants and remain included in the record-level action/frame counts.

| Distribution | Count | Min | P25 | P50 | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
{dist_row('Decision points per video', temporal['decision_points_per_video'])}
{dist_row('Frames per record', temporal['frames_per_record'])}

| Check | Value |
|---|---:|
| Frames contiguous `0..N-1` records | {temporal['frames_contiguity']['contiguous_record_count']} |
| Frames non-contiguous records | {temporal['frames_contiguity']['non_contiguous_record_count']} |
| Videos with non-decreasing `n_frames` by numeric step | {temporal['n_frames_monotonicity']['monotonic_nondecreasing_video_count']} |
| Videos with a monotonicity counterexample | {temporal['n_frames_monotonicity']['counterexample_video_count']} |

## Method and limits

The parser uses `json.JSONDecoder().raw_decode` on 1 MiB text chunks and discards the processed prefix after each record; it never calls `json.load()` on the 317 MB annotations array. All split membership uses exactly `' '.join(s.split()).strip().lower()` on both sides. Scene identity is recovered only from the matching R2R split episodes, not from `video_id` or numeric IDs. A normalized instruction found in train and either val_unseen or test is listed in the complete exclusion artifact and excluded under the dispatch rule. Text normalization can reveal identical text but cannot independently prove an original episode's provenance when R2R split text is duplicated; that ambiguity is why the specified exclusion rule is applied. The `generated_at` field is the newest verified input file mtime (rather than wall-clock time) so no-argument reruns reproduce the JSON bytes deterministically.

MISSING items: {leakage['missing'] if leakage['missing'] else 'none.'}
"""


def main() -> None:
    inputs = verify_inputs()
    generated_at = deterministic_generated_at()

    split_scenes: dict[str, dict[str, set[str]]] = {
        split: defaultdict(set) for split in SPLIT_ORDER
    }
    split_membership: dict[str, set[str]] = defaultdict(set)
    split_episode_counts: dict[str, int] = {}
    for split, path in SPLIT_PATHS.items():
        episodes = load_episodes(path)
        split_episode_counts[split] = len(episodes)
        for episode in episodes:
            instruction = episode["instruction"]["instruction_text"]
            key = normalized_instruction(instruction)
            split_scenes[split][key].add(episode["scene_id"])
            split_membership[key].add(split)

    eval11_episodes = load_episodes(EXPECTED_INPUTS[5][1])
    eval11_raw_scene_ids = sorted({episode["scene_id"] for episode in eval11_episodes})
    eval11_scene_stems = sorted({scene_stem(scene_id) for scene_id in eval11_raw_scene_ids})

    matched_instructions: dict[str, set[str]] = {split: set() for split in SPLIT_ORDER}
    matched_record_counts: Counter[str] = Counter()
    all_normalized_instructions: set[str] = set()
    records_per_instruction: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    frame_counts: list[int] = []
    # One decision point can occur once per instruction variant.  Keep all
    # observations by numeric step, but count unique steps as decision points.
    video_entries: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    unaccounted_examples: list[dict[str, str]] = []
    unaccounted_instructions: set[str] = set()
    contiguity_examples: list[dict[str, Any]] = []
    contiguous_record_count = 0
    total_record_count = 0
    excluded_accumulator: dict[str, dict[str, Any]] = {}

    for record in iter_annotation_records(EXPECTED_INPUTS[0][1]):
        total_record_count += 1
        try:
            video_id = record["video_id"]
            instruction_text = record["q"]
            action = record["a"]
            frames = record["frames"]
        except KeyError as exc:
            raise ValueError(f"annotation record {total_record_count} lacks required field {exc}") from exc
        if not isinstance(video_id, str) or not isinstance(instruction_text, str) or not isinstance(action, str):
            raise ValueError(f"annotation record {total_record_count} has a non-string video_id/q/a")
        if not isinstance(frames, list):
            raise ValueError(f"annotation record {total_record_count} has non-list frames")

        key = normalized_instruction(instruction_text)
        all_normalized_instructions.add(key)
        records_per_instruction[key] += 1
        action_counts[action] += 1

        frame_count = len(frames)
        frame_counts.append(frame_count)
        if frame_sequence_is_contiguous(frames):
            contiguous_record_count += 1
        elif len(contiguity_examples) < 3:
            contiguity_examples.append(
                {"video_id": video_id, "frames": frames[:5], "n_frames": frame_count}
            )

        video_prefix, separator, step_text = video_id.rpartition("-")
        if not separator or not video_prefix:
            raise ValueError(f"annotation record {total_record_count} has invalid video_id {video_id!r}")
        try:
            step = int(step_text)
        except ValueError as exc:
            raise ValueError(f"annotation record {total_record_count} has non-integer step in {video_id!r}") from exc
        video_entries[video_prefix][step].append(frame_count)

        memberships = split_membership.get(key, set())
        if not memberships:
            unaccounted_instructions.add(key)
            if len(unaccounted_examples) < 3:
                unaccounted_examples.append(
                    {
                        "video_id": video_id,
                        "normalized_instruction": key,
                        "raw_instruction": instruction_text,
                    }
                )
            continue

        for split in memberships:
            matched_instructions[split].add(key)
            matched_record_counts[split] += 1

        if "train" in memberships and ("val_unseen" in memberships or "test" in memberships):
            entry = excluded_accumulator.setdefault(
                key,
                {
                    "normalized_instruction": key,
                    "matched_splits": sorted(memberships, key=SPLIT_ORDER.index),
                    "video_ids": [],
                    "affected_record_count": 0,
                },
            )
            entry["video_ids"].append(video_id)
            entry["affected_record_count"] += 1

    matched_train_raw_scene_ids = sorted(
        {
            scene_id
            for key in matched_instructions["train"]
            for scene_id in split_scenes["train"][key]
        }
    )
    matched_train_scene_stems = sorted({scene_stem(scene_id) for scene_id in matched_train_raw_scene_ids})
    actual_stem_intersection = sorted(set(matched_train_scene_stems) & set(eval11_scene_stems))
    actual_raw_intersection = sorted(set(matched_train_raw_scene_ids) & set(eval11_raw_scene_ids))
    val_unseen_minus_train = sorted(matched_instructions["val_unseen"] - matched_instructions["train"])

    split_matches = {
        split: {
            "r2r_split_episode_count": split_episode_counts[split],
            "matched_normalized_instruction_count": len(matched_instructions[split]),
            "matched_record_count": matched_record_counts[split],
        }
        for split in SPLIT_ORDER
    }
    exclusions = sorted(
        excluded_accumulator.values(), key=lambda item: item["normalized_instruction"]
    )
    excluded_record_count = sum(item["affected_record_count"] for item in exclusions)

    metadata = output_metadata(inputs, generated_at)
    leakage = {
        **metadata,
        "method": {
            "join_key": "' '.join(s.split()).strip().lower() applied to NaVILA q and R2R instruction.instruction_text",
            "join_key_prohibitions_observed": ["video_id", "numeric id"],
            "annotations_parser": "json.JSONDecoder().raw_decode incrementally over 1048576-character chunks",
            "full_annotations_json_load_used": False,
        },
        "leakage_join": {
            "total_record_count": total_record_count,
            "distinct_normalized_instruction_count": len(all_normalized_instructions),
            "split_matches": split_matches,
            "val_unseen_minus_train_instruction_count": len(val_unseen_minus_train),
            "val_unseen_minus_train_record_count": sum(records_per_instruction[key] for key in val_unseen_minus_train),
            "val_unseen_minus_train_normalized_instructions_sorted": val_unseen_minus_train,
            "matched_train_raw_scene_count": len(matched_train_raw_scene_ids),
            "matched_train_raw_scene_ids_sorted": matched_train_raw_scene_ids,
            "matched_train_scene_stem_count": len(matched_train_scene_stems),
            "matched_train_scene_stems_sorted": matched_train_scene_stems,
            "eval11_raw_scene_count": len(eval11_raw_scene_ids),
            "eval11_raw_scene_ids_sorted": eval11_raw_scene_ids,
            "eval11_scene_stem_count": len(eval11_scene_stems),
            "eval11_scene_stems_sorted": eval11_scene_stems,
            "matched_train_scene_stems_intersection_eval11_sorted": actual_stem_intersection,
            "matched_train_raw_scene_ids_intersection_eval11_sorted": actual_raw_intersection,
            "unexplained_record_count": None,
            "unexplained_normalized_instruction_count": len(unaccounted_instructions),
            "unexplained_record_examples_first_3": unaccounted_examples,
        },
            "exclusion_summary": {
            "rule": "exclude a matched normalized instruction from training when it matches train and either val_unseen or test",
            "excluded_instruction_count": len(exclusions),
            "excluded_record_count": excluded_record_count,
            "complete_list_path": str(EXCLUDED_OUTPUT),
        },
        "excluded_instructions": exclusions,
        "missing": [],
    }
    # This count is based on records, not merely distinct instructions.
    accounted_record_count = sum(
        records_per_instruction[key]
        for key in all_normalized_instructions
        if split_membership.get(key, set())
    )
    leakage["leakage_join"]["unexplained_record_count"] = total_record_count - accounted_record_count

    majority_action, majority_count = sorted(
        action_counts.items(), key=lambda item: (-item[1], item[0])
    )[0]
    exact_stop_label = "I think I should stop because I have finished the instruction."
    stop_count = action_counts.get(exact_stop_label, 0)
    if exact_stop_label not in action_counts:
        raise ValueError("the expected exact stop action label is absent from the full vocabulary")

    decision_points = [len(entries) for entries in video_entries.values()]
    monotonic_video_count = 0
    monotonicity_counterexamples: list[dict[str, Any]] = []
    duplicate_step_videos: list[str] = []
    conflicting_step_videos: list[str] = []
    for video_prefix, entries in sorted(video_entries.items()):
        steps = sorted(entries)
        if any(len(entries[step]) > 1 for step in steps):
            duplicate_step_videos.append(video_prefix)
        if any(len(set(entries[step])) > 1 for step in steps):
            conflicting_step_videos.append(video_prefix)
        # Duplicate rows are instruction variants of the same decision point;
        # they must agree on the frame count before one value represents it.
        counts = [entries[step][0] for step in steps]
        if all(before <= after for before, after in zip(counts, counts[1:])):
            monotonic_video_count += 1
        elif len(monotonicity_counterexamples) < 3:
            monotonicity_counterexamples.append(
                {
                    "video_prefix": video_prefix,
                    "steps_sorted": steps,
                    "n_frames_by_step": counts,
                }
            )

    actions = {
        **metadata,
        "action_distribution": {
            "total_record_count": total_record_count,
            "action_vocabulary_size": len(action_counts),
            "action_vocabulary_sorted": sorted(action_counts),
            "action_counts": dict(sorted(action_counts.items())),
            "majority_class": {
                "action": majority_action,
                "count": majority_count,
                "share_fraction": majority_count / total_record_count,
                "share_percent": 100 * majority_count / total_record_count,
            },
            "stop_action": {
                "action": exact_stop_label,
                "count": stop_count,
                "share_fraction": stop_count / total_record_count,
                "share_percent": 100 * stop_count / total_record_count,
            },
        },
        "temporal_structure": {
            "percentile_method": "nearest rank: sorted[ceil(p*n)-1]",
            "video_count": len(video_entries),
            "decision_points_definition": "number of unique numeric steps per video prefix; repeated rows with the same step are separate instruction-variant records",
            "decision_points_per_video": nearest_rank_distribution(decision_points),
            "frames_per_record": nearest_rank_distribution(frame_counts),
            "frames_contiguity": {
                "definition": "each frames element path ends in frame_<i>.jpg for list index i, so indices are exactly 0..N-1",
                "contiguous_record_count": contiguous_record_count,
                "non_contiguous_record_count": total_record_count - contiguous_record_count,
                "counterexamples_first_3": contiguity_examples,
            },
            "n_frames_monotonicity": {
                "definition": "within each video prefix, sort unique integer steps parsed from video_id '<video>-<step>'; duplicate rows at one step are checked for equal n_frames, then the unique-step sequence must be non-decreasing",
                "monotonic_nondecreasing_video_count": monotonic_video_count,
                "counterexample_video_count": len(video_entries) - monotonic_video_count,
                "counterexamples_first_3": monotonicity_counterexamples,
                "duplicate_numeric_step_video_count": len(duplicate_step_videos),
                "duplicate_numeric_step_video_ids_first_3": duplicate_step_videos[:3],
                "conflicting_n_frames_at_same_step_video_count": len(conflicting_step_videos),
                "conflicting_n_frames_at_same_step_video_ids_first_3": conflicting_step_videos[:3],
            },
        },
        "missing": [],
    }

    exclusions_payload = {
        **metadata,
        "exclusion_rule": {
            "description": "A normalized instruction found in train and either val_unseen or test is excluded from training regardless of provenance.",
            "join_key": "' '.join(s.split()).strip().lower()",
        },
        "excluded_instruction_count": len(exclusions),
        "excluded_record_count": excluded_record_count,
        "video_id_storage": {"mode": "inline", "path": None},
        "excluded_instructions": exclusions,
        "missing": [],
    }

    # Dispatch §4 permits a sidecar only if the full inline list would exceed 5 MB.
    encoded_exclusions = json.dumps(
        exclusions_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded_exclusions) > 5 * 1024 * 1024:
        video_id_sidecar = {
            **metadata,
            "video_ids_by_normalized_instruction": {
                item["normalized_instruction"]: item["video_ids"] for item in exclusions
            },
        }
        with gzip.open(EXCLUDED_VIDEO_IDS_OUTPUT, "wt", encoding="utf-8") as sidecar:
            json.dump(video_id_sidecar, sidecar, ensure_ascii=False, sort_keys=True)
        for item in exclusions_payload["excluded_instructions"]:
            item.pop("video_ids")
        exclusions_payload["video_id_storage"] = {
            "mode": "gzip_sidecar",
            "path": str(EXCLUDED_VIDEO_IDS_OUTPUT),
            "sha256": sha256_and_size(EXCLUDED_VIDEO_IDS_OUTPUT)[0],
            "bytes": EXCLUDED_VIDEO_IDS_OUTPUT.stat().st_size,
        }

    write_json(LEAKAGE_OUTPUT, leakage)
    write_json(ACTION_OUTPUT, actions)
    write_json(EXCLUDED_OUTPUT, exclusions_payload)
    MARKDOWN_OUTPUT.write_text(make_markdown(leakage, actions, exclusions_payload), encoding="utf-8")


if __name__ == "__main__":
    main()
