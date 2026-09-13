"""Offline-only semantic landmark validation and deterministic instructions."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LANDMARK_WHITELIST = ("door", "doorframe", "chair", "table", "sofa", "bed", "picture", "window")
# The inspected NaVILA mpcat40 table has no separate `doorframe` class: it is a
# synset of the `door` row.  N1 therefore emits `door` unless a future approved
# instance-level source provides a distinct, auditable doorframe label.
NAVILA_MPCAT40_LANDMARKS = ("door", "chair", "table", "sofa", "bed", "picture", "window")
MIN_PIXEL_RATIO = 0.02
MIN_CONSECUTIVE_SAMPLES = 3
ANCHOR_WINDOW_M = 0.5


@dataclass(frozen=True)
class SemanticFrame:
    route_sample_index: int
    route_arc_length_m: float
    class_pixel_ratio: Mapping[str, float]
    segmentation_sha256: str


def load_navila_mpcat40_colors(path: Path) -> dict[tuple[int, int, int], str]:
    """Load only the approved mpcat40 RGB colors from NaVILA's TSV asset."""
    result: dict[tuple[int, int, int], str] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            label = str(row.get("mpcat40", "")).strip().lower()
            color = str(row.get("hex", "")).strip()
            if label not in NAVILA_MPCAT40_LANDMARKS or not color.startswith("#") or len(color) != 7:
                continue
            result[tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))] = label
    if not result:
        raise ValueError("NaVILA mpcat40 file did not contain any approved landmark colors")
    return result


def class_pixel_ratios_from_rgb(
    semantic_rgb: Sequence[Sequence[Sequence[int]]], color_to_label: Mapping[tuple[int, int, int], str]
) -> dict[str, float]:
    """Convert a semantic preview RGB image to full-frame whitelist ratios."""
    pixels = [pixel for row in semantic_rgb for pixel in row]
    if not pixels:
        raise ValueError("semantic preview image is empty")
    counts: Counter[str] = Counter()
    for pixel in pixels:
        if len(pixel) != 3:
            raise ValueError("semantic preview pixel is not RGB")
        label = color_to_label.get(tuple(int(channel) for channel in pixel))
        if label is not None:
            counts[label] += 1
    return {label: count / len(pixels) for label, count in sorted(counts.items())}


def _qualifying_runs(frames: Sequence[SemanticFrame], label: str) -> list[list[SemanticFrame]]:
    runs: list[list[SemanticFrame]] = []
    current: list[SemanticFrame] = []
    for frame in sorted(frames, key=lambda item: item.route_sample_index):
        visible = float(frame.class_pixel_ratio.get(label, 0.0)) >= MIN_PIXEL_RATIO
        contiguous = not current or frame.route_sample_index == current[-1].route_sample_index + 1
        if visible and contiguous:
            current.append(frame)
        else:
            if len(current) >= MIN_CONSECUTIVE_SAMPLES:
                runs.append(current)
            current = [frame] if visible else []
    if len(current) >= MIN_CONSECUTIVE_SAMPLES:
        runs.append(current)
    return runs


def select_landmark(
    candidate: Mapping[str, Any], frames: Sequence[SemanticFrame], *, anchor_window_m: float = ANCHOR_WINDOW_M
) -> dict[str, Any]:
    """Return a unique, anchor-aligned whitelist landmark or a rejection reason."""
    if anchor_window_m <= 0:
        raise ValueError("anchor_window_m must be positive")
    anchor = float(candidate["instruction_anchor_arc_length_m"])
    eligible: list[dict[str, Any]] = []
    for label in LANDMARK_WHITELIST:
        for run in _qualifying_runs(frames, label):
            nearest = min(run, key=lambda frame: abs(frame.route_arc_length_m - anchor))
            distance = abs(nearest.route_arc_length_m - anchor)
            if distance <= anchor_window_m + 1e-9:
                eligible.append(
                    {
                        "label": label,
                        "route_sample_indices": [frame.route_sample_index for frame in run],
                        "pixel_ratios": [float(frame.class_pixel_ratio[label]) for frame in run],
                        "anchor_distance_m": distance,
                        "segmentation_sha256": [frame.segmentation_sha256 for frame in run],
                    }
                )
    labels = {item["label"] for item in eligible}
    if not eligible:
        return {"accepted": False, "reason": "no_whitelist_landmark_visible_for_3_consecutive_samples"}
    if len(labels) != 1:
        return {"accepted": False, "reason": "ambiguous_landmark_categories", "candidates": eligible}
    selected = min(eligible, key=lambda item: (item["anchor_distance_m"], item["label"]))
    return {"accepted": True, "landmark": selected}


def generate_instruction(candidate: Mapping[str, Any], landmark: Mapping[str, Any]) -> dict[str, Any]:
    """Generate the sole permitted short instruction language deterministically."""
    label = str(landmark["label"])
    if label not in LANDMARK_WHITELIST:
        raise ValueError("landmark is outside the controlled whitelist")
    category = str(candidate["category"])
    if category == "straight":
        text, template_id = f"Go to the {label}.", "semantic_end_v1"
    elif category in {"left_turn", "right_turn"}:
        direction = category.removesuffix("_turn")
        text, template_id = f"Turn {direction} at the {label}.", "semantic_turn_v1"
    else:
        raise ValueError(f"unsupported route category: {category}")
    return {
        "text": text,
        "template_id": template_id,
        "generation": "deterministic_semantic_template",
        "uses_external_llm": False,
        "manual_instruction": False,
        "landmark": dict(landmark),
    }


def validate_instruction(candidate: Mapping[str, Any], instruction: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if instruction.get("generation") != "deterministic_semantic_template" or instruction.get("manual_instruction"):
        errors.append("instruction is not generated solely by a deterministic semantic template")
    landmark = instruction.get("landmark")
    if not isinstance(landmark, Mapping) or landmark.get("label") not in LANDMARK_WHITELIST:
        errors.append("instruction landmark is absent or not whitelisted")
        return errors
    if len(landmark.get("route_sample_indices", ())) < MIN_CONSECUTIVE_SAMPLES:
        errors.append("landmark lacks three consecutive visible samples")
    if any(float(value) < MIN_PIXEL_RATIO for value in landmark.get("pixel_ratios", ())):
        errors.append("landmark pixel ratio is below 2%")
    if float(landmark.get("anchor_distance_m", float("inf"))) > ANCHOR_WINDOW_M:
        errors.append("landmark is not spatially aligned with the turn/end anchor")
    expected = generate_instruction(candidate, landmark)
    if instruction.get("text") != expected["text"] or instruction.get("template_id") != expected["template_id"]:
        errors.append("instruction text does not match the declared deterministic template")
    return errors


def build_instruction_swap_set(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Define N2 offline swaps: left/right and a different whitelist category."""
    swaps: list[dict[str, Any]] = []
    for record in records:
        instruction = record["instruction"]
        label = str(instruction["landmark"]["label"])
        category = str(record["category"])
        if category in {"left_turn", "right_turn"}:
            opposite = "right_turn" if category == "left_turn" else "left_turn"
            opposite_candidate = {**record, "category": opposite}
            swaps.append({"kind": "left_right", "source_id": record["candidate_id"], "instruction": generate_instruction(opposite_candidate, instruction["landmark"])})
        replacement = next(name for name in LANDMARK_WHITELIST if name != label)
        swapped_landmark = {**instruction["landmark"], "label": replacement}
        swaps.append({"kind": "landmark_category", "source_id": record["candidate_id"], "instruction": generate_instruction(record, swapped_landmark)})
    return swaps
