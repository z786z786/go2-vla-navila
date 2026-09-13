"""Dense-path resampling and flat short-route candidate generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


Point = tuple[float, float, float]


@dataclass(frozen=True)
class RouteRules:
    resample_spacing_m: float = 0.1
    min_length_m: float = 1.5
    max_length_m: float = 4.0
    max_route_height_change_m: float = 0.05
    max_dense_step_height_change_m: float = 0.03
    max_floor_slope_degrees: float = 2.0
    principal_turn_degrees: float = 45.0
    turn_component_degrees: float = 3.0
    turn_merge_distance_m: float = 0.4


@dataclass(frozen=True)
class ResampledPoint:
    xyz: Point
    arc_length_m: float
    dense_left_index: int
    dense_right_index: int


@dataclass(frozen=True)
class SurfaceProbe:
    """Offline downward surface probe aligned to a 0.1-m route sample."""

    surface_kind: str
    normal_z: float


def _distance(a: Point, b: Point) -> float:
    return math.dist(a, b)


def _xy_heading(a: Point, b: Point) -> float | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return None if math.hypot(dx, dy) <= 1e-9 else math.atan2(dy, dx)


def _wrapped_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def _as_points(path: Sequence[Sequence[float]]) -> list[Point]:
    points = [tuple(float(axis) for axis in point) for point in path]
    if len(points) < 2 or any(len(point) != 3 for point in points):
        raise ValueError("dense path must contain at least two [x, y, z] points")
    if not all(math.isfinite(axis) for point in points for axis in point):
        raise ValueError("dense path contains a non-finite coordinate")
    return points  # type: ignore[return-value]


def resample_dense_path(
    dense_path: Sequence[Sequence[float]], spacing_m: float = 0.1
) -> list[ResampledPoint]:
    """Resample a dense 3-D polyline while retaining exact source provenance."""
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    points = _as_points(dense_path)
    lengths = [_distance(a, b) for a, b in zip(points, points[1:])]
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    if cumulative[-1] <= 1e-9:
        raise ValueError("dense path has zero length")

    targets = [index * spacing_m for index in range(int(cumulative[-1] / spacing_m) + 1)]
    if not math.isclose(targets[-1], cumulative[-1], abs_tol=1e-9):
        targets.append(cumulative[-1])
    result: list[ResampledPoint] = []
    segment = 0
    for target in targets:
        while segment < len(lengths) - 1 and target > cumulative[segment + 1] + 1e-9:
            segment += 1
        length = lengths[segment]
        ratio = 0.0 if length <= 1e-9 else (target - cumulative[segment]) / length
        ratio = min(1.0, max(0.0, ratio))
        a, b = points[segment], points[segment + 1]
        xyz = tuple(a[axis] + (b[axis] - a[axis]) * ratio for axis in range(3))
        result.append(ResampledPoint(xyz, target, segment, segment + 1))  # type: ignore[arg-type]
    return result


def _principal_turns(points: Sequence[ResampledPoint], rules: RouteRules) -> list[dict[str, Any]]:
    headings: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        heading = _xy_heading(start.xyz, end.xyz)
        if heading is not None:
            headings.append((heading, end.arc_length_m))
    changes: list[dict[str, float]] = []
    for (previous, _), (current, location) in zip(headings, headings[1:]):
        delta = math.degrees(_wrapped_delta(current, previous))
        if abs(delta) >= rules.turn_component_degrees:
            changes.append({"angle_degrees": delta, "arc_length_m": location})
    groups: list[list[dict[str, float]]] = []
    for change in changes:
        if groups:
            previous = groups[-1][-1]
            same_direction = change["angle_degrees"] * previous["angle_degrees"] > 0
            close = change["arc_length_m"] - previous["arc_length_m"] <= rules.turn_merge_distance_m
            if same_direction and close:
                groups[-1].append(change)
                continue
        groups.append([change])
    events = []
    for group in groups:
        angle = sum(change["angle_degrees"] for change in group)
        if abs(angle) < rules.principal_turn_degrees:
            continue
        weights = [abs(change["angle_degrees"]) for change in group]
        location = sum(change["arc_length_m"] * weight for change, weight in zip(group, weights)) / sum(weights)
        events.append(
            {
                "direction": "left" if angle > 0 else "right",
                "signed_angle_degrees": angle,
                "arc_length_m": location,
                "component_count": len(group),
            }
        )
    return events


def _flatness_audit(
    dense: Sequence[Point], samples: Sequence[ResampledPoint], probes: Sequence[SurfaceProbe], rules: RouteRules
) -> dict[str, Any]:
    if len(samples) != len(probes):
        return {"passed": False, "reason": "surface_probe_count_mismatch"}
    dense_step = max(abs(b[2] - a[2]) for a, b in zip(dense, dense[1:]))
    height_span = max(point[2] for point in dense) - min(point[2] for point in dense)
    allowed_surface = all(probe.surface_kind.strip().lower() == "floor" for probe in probes)
    max_slope = max(math.degrees(math.acos(min(1.0, max(-1.0, probe.normal_z)))) for probe in probes)
    passed = (
        height_span <= rules.max_route_height_change_m + 1e-9
        and dense_step <= rules.max_dense_step_height_change_m + 1e-9
        and allowed_surface
        and max_slope <= rules.max_floor_slope_degrees + 1e-9
    )
    return {
        "passed": passed,
        "height_span_m": height_span,
        "max_dense_step_height_change_m": dense_step,
        "max_floor_slope_degrees": max_slope,
        "all_surface_probes_are_floor": allowed_surface,
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def enumerate_route_candidates(
    parent_episode_id: str,
    dense_path: Sequence[Sequence[float]],
    surface_probes: Sequence[SurfaceProbe],
    rules: RouteRules = RouteRules(),
) -> list[dict[str, Any]]:
    """Return only continuous, flat, 1.5--4 m geometry candidates.

    Each result retains the parent hash, inclusive dense-index range, dense slice,
    and resampled 0.1-m path.  Semantic landmark checks happen separately after
    the preview renderer produces labels for a candidate.
    """
    full_samples = resample_dense_path(dense_path, rules.resample_spacing_m)
    if len(surface_probes) != len(full_samples):
        raise ValueError("surface probes must align one-for-one with resampled dense path")
    dense = _as_points(dense_path)
    parent_hash = _sha256_json(dense)
    candidates: list[dict[str, Any]] = []
    for start in range(len(full_samples) - 1):
        for end in range(start + 1, len(full_samples)):
            length = full_samples[end].arc_length_m - full_samples[start].arc_length_m
            if length > rules.max_length_m + 1e-9:
                break
            if length < rules.min_length_m - 1e-9:
                continue
            samples = full_samples[start : end + 1]
            dense_start, dense_end = samples[0].dense_left_index, samples[-1].dense_right_index
            dense_slice = dense[dense_start : dense_end + 1]
            flatness = _flatness_audit(dense_slice, samples, surface_probes[start : end + 1], rules)
            if not flatness["passed"]:
                continue
            turns = _principal_turns(samples, rules)
            if len(turns) > 1:
                continue
            category = "straight" if not turns else f"{turns[0]['direction']}_turn"
            relative_turns = [
                {**turn, "arc_length_m": float(turn["arc_length_m"]) - samples[0].arc_length_m}
                for turn in turns
            ]
            anchor = length if not relative_turns else float(relative_turns[0]["arc_length_m"])
            record = {
                "parent_episode_id": str(parent_episode_id),
                "dense_index_range": [dense_start, dense_end],
                "dense_arclength_range_m": [samples[0].arc_length_m, samples[-1].arc_length_m],
                "parent_dense_path_sha256": parent_hash,
                "parent_dense_path": [list(point) for point in dense],
                "subroute_dense_path": [list(point) for point in dense_slice],
                "resampled_path": [
                    {
                        "xyz": list(sample.xyz),
                        "arc_length_m": sample.arc_length_m - samples[0].arc_length_m,
                        "dense_left_index": sample.dense_left_index,
                        "dense_right_index": sample.dense_right_index,
                    }
                    for sample in samples
                ],
                "path_length_m": length,
                "category": category,
                "instruction_anchor": "end" if category == "straight" else "turn",
                "instruction_anchor_arc_length_m": anchor,
                "flatness_audit": flatness,
                "turn_audit": {"principal_turns": relative_turns, "rules": asdict(rules)},
            }
            record["candidate_id"] = _sha256_json(
                [parent_episode_id, dense_start, dense_end, parent_hash, rules.resample_spacing_m]
            )[:20]
            candidates.append(record)
    return candidates


def enumerate_geometry_precandidates(
    parent_episode_id: str,
    dense_path: Sequence[Sequence[float]],
    rules: RouteRules = RouteRules(),
) -> list[dict[str, Any]]:
    """Return height/turn-qualified routes that still require a surface preview.

    This deliberately cannot produce an accepted route: it has no floor-normal
    or semantic evidence.  N1 uses it to construct a bounded preview bank
    without pretending that an all-zero dense Z coordinate proves flatness.
    """
    full_samples = resample_dense_path(dense_path, rules.resample_spacing_m)
    dense = _as_points(dense_path)
    parent_hash = _sha256_json(dense)
    candidates: list[dict[str, Any]] = []
    for start in range(len(full_samples) - 1):
        for end in range(start + 1, len(full_samples)):
            length = full_samples[end].arc_length_m - full_samples[start].arc_length_m
            if length > rules.max_length_m + 1e-9:
                break
            if length < rules.min_length_m - 1e-9:
                continue
            samples = full_samples[start : end + 1]
            dense_start, dense_end = samples[0].dense_left_index, samples[-1].dense_right_index
            dense_slice = dense[dense_start : dense_end + 1]
            dense_step = max(abs(second[2] - first[2]) for first, second in zip(dense_slice, dense_slice[1:]))
            height_span = max(point[2] for point in dense_slice) - min(point[2] for point in dense_slice)
            if (
                height_span > rules.max_route_height_change_m + 1e-9
                or dense_step > rules.max_dense_step_height_change_m + 1e-9
            ):
                continue
            turns = _principal_turns(samples, rules)
            if len(turns) > 1:
                continue
            relative_turns = [
                {**turn, "arc_length_m": float(turn["arc_length_m"]) - samples[0].arc_length_m}
                for turn in turns
            ]
            category = "straight" if not relative_turns else f"{relative_turns[0]['direction']}_turn"
            record = {
                "parent_episode_id": str(parent_episode_id),
                "dense_index_range": [dense_start, dense_end],
                "dense_arclength_range_m": [samples[0].arc_length_m, samples[-1].arc_length_m],
                "parent_dense_path_sha256": parent_hash,
                "parent_dense_path": [list(point) for point in dense],
                "subroute_dense_path": [list(point) for point in dense_slice],
                "resampled_path": [
                    {
                        "xyz": list(sample.xyz),
                        "arc_length_m": sample.arc_length_m - samples[0].arc_length_m,
                        "dense_left_index": sample.dense_left_index,
                        "dense_right_index": sample.dense_right_index,
                    }
                    for sample in samples
                ],
                "path_length_m": length,
                "category": category,
                "instruction_anchor": "end" if category == "straight" else "turn",
                "instruction_anchor_arc_length_m": length
                if not relative_turns
                else float(relative_turns[0]["arc_length_m"]),
                "flatness_audit": {
                    "passed": False,
                    "status": "pending_preview_surface_probe",
                    "height_span_m": height_span,
                    "max_dense_step_height_change_m": dense_step,
                    "required_surface_kind": "floor",
                    "required_max_floor_slope_degrees": rules.max_floor_slope_degrees,
                },
                "turn_audit": {"principal_turns": relative_turns, "rules": asdict(rules)},
                "acceptance_status": "pending_surface_and_semantic_preview",
            }
            record["candidate_id"] = _sha256_json(
                [parent_episode_id, dense_start, dense_end, parent_hash, rules.resample_spacing_m]
            )[:20]
            candidates.append(record)
    return candidates


def select_geometry_precandidates(
    parent_episode_id: str,
    dense_path: Sequence[Sequence[float]],
    *,
    maximum_candidates: int,
    selection_seed: int,
    rules: RouteRules = RouteRules(),
) -> list[dict[str, Any]]:
    """Stream a bounded, category-aware preview bank for one dense parent path.

    Unlike :func:`enumerate_geometry_precandidates`, this never materializes the
    full O(n²) candidate set.  It retains the best straight/left/right rows and
    a tiny global buffer, then expands only the final selected records.
    """
    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    full_samples = resample_dense_path(dense_path, rules.resample_spacing_m)
    dense = _as_points(dense_path)
    parent_hash = _sha256_json(dense)
    selected_by_category: dict[str, tuple[tuple[float, int], tuple[Any, ...]]] = {}
    selected_global: list[tuple[tuple[float, int], tuple[Any, ...]]] = []

    def score(length: float, start: int, end: int) -> tuple[float, int]:
        rank = int.from_bytes(
            hashlib.sha256(f"{selection_seed}|{parent_episode_id}|{start}|{end}".encode("utf-8")).digest()[:8],
            "big",
        )
        return (abs(length - 2.75), rank)

    def consider(item_score: tuple[float, int], metadata: tuple[Any, ...], category: str) -> None:
        current = selected_by_category.get(category)
        if current is None or item_score < current[0]:
            selected_by_category[category] = (item_score, metadata)
        selected_global.append((item_score, metadata))
        selected_global.sort(key=lambda item: item[0])
        del selected_global[maximum_candidates:]

    for start in range(len(full_samples) - 1):
        for end in range(start + 1, len(full_samples)):
            length = full_samples[end].arc_length_m - full_samples[start].arc_length_m
            if length > rules.max_length_m + 1e-9:
                break
            if length < rules.min_length_m - 1e-9:
                continue
            samples = full_samples[start : end + 1]
            dense_start, dense_end = samples[0].dense_left_index, samples[-1].dense_right_index
            dense_slice = dense[dense_start : dense_end + 1]
            dense_step = max(abs(second[2] - first[2]) for first, second in zip(dense_slice, dense_slice[1:]))
            height_span = max(point[2] for point in dense_slice) - min(point[2] for point in dense_slice)
            if (
                height_span > rules.max_route_height_change_m + 1e-9
                or dense_step > rules.max_dense_step_height_change_m + 1e-9
            ):
                continue
            turns = _principal_turns(samples, rules)
            if len(turns) > 1:
                continue
            relative_turns = tuple(
                {**turn, "arc_length_m": float(turn["arc_length_m"]) - samples[0].arc_length_m}
                for turn in turns
            )
            category = "straight" if not relative_turns else f"{relative_turns[0]['direction']}_turn"
            consider(
                score(length, start, end),
                (start, end, length, relative_turns, height_span, dense_step, category),
                category,
            )

    chosen: list[tuple[Any, ...]] = []
    chosen_ranges: set[tuple[int, int]] = set()

    def dense_range_key(metadata: tuple[Any, ...]) -> tuple[int, int]:
        return (
            full_samples[int(metadata[0])].dense_left_index,
            full_samples[int(metadata[1])].dense_right_index,
        )

    for category in ("straight", "left_turn", "right_turn"):
        item = selected_by_category.get(category)
        if item is not None and len(chosen) < maximum_candidates:
            metadata = item[1]
            range_key = dense_range_key(metadata)
            if range_key not in chosen_ranges:
                chosen.append(metadata)
                chosen_ranges.add(range_key)
    for _, metadata in selected_global:
        range_key = dense_range_key(metadata)
        if range_key not in chosen_ranges and len(chosen) < maximum_candidates:
            chosen.append(metadata)
            chosen_ranges.add(range_key)

    records: list[dict[str, Any]] = []
    for start, end, length, relative_turns, height_span, dense_step, category in chosen:
        samples = full_samples[int(start) : int(end) + 1]
        dense_start, dense_end = samples[0].dense_left_index, samples[-1].dense_right_index
        dense_slice = dense[dense_start : dense_end + 1]
        record = {
            "parent_episode_id": str(parent_episode_id),
            "dense_index_range": [dense_start, dense_end],
            "dense_arclength_range_m": [samples[0].arc_length_m, samples[-1].arc_length_m],
            "parent_dense_path_sha256": parent_hash,
            "parent_dense_path": [list(point) for point in dense],
            "subroute_dense_path": [list(point) for point in dense_slice],
            "resampled_path": [
                {
                    "xyz": list(sample.xyz),
                    "arc_length_m": sample.arc_length_m - samples[0].arc_length_m,
                    "dense_left_index": sample.dense_left_index,
                    "dense_right_index": sample.dense_right_index,
                }
                for sample in samples
            ],
            "path_length_m": length,
            "category": category,
            "instruction_anchor": "end" if category == "straight" else "turn",
            "instruction_anchor_arc_length_m": length
            if not relative_turns
            else float(relative_turns[0]["arc_length_m"]),
            "flatness_audit": {
                "passed": False,
                "status": "pending_preview_surface_probe",
                "height_span_m": height_span,
                "max_dense_step_height_change_m": dense_step,
                "required_surface_kind": "floor",
                "required_max_floor_slope_degrees": rules.max_floor_slope_degrees,
            },
            "turn_audit": {"principal_turns": list(relative_turns), "rules": asdict(rules)},
            "acceptance_status": "pending_surface_and_semantic_preview",
        }
        record["candidate_id"] = _sha256_json(
            [parent_episode_id, dense_start, dense_end, parent_hash, rules.resample_spacing_m]
        )[:20]
        records.append(record)
    return records
