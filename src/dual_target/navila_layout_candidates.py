"""Generate simple NaVILA dual-box layout proposals from official route geometry.

The output is intentionally pending rendering.  It contains scene-local poses
and requested relative headings, but no candidate is accepted until the live
renderer supplies the two-box visibility evidence required by
``navila_ood_split``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FORMAT = "navila-dual-target-layout-proposals-v1"
OFFSETS_DEGREES = (-20.0, -16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
TARGET_DISTANCES_M = (4.5, 5.5, 6.5)
BOX_HALF_SEPARATION_M = 0.32
MAX_HEIGHT_SPAN_M = 0.10
MAX_PATH_TO_CHORD_RATIO = 1.05
MAX_CHORD_DEVIATION_M = 0.25


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: Any, length: int = 20) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()[:length]


def _points(value: Any) -> list[tuple[float, float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError("gt_locations must contain at least two xyz points")
    result = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 3:
            raise ValueError("gt_locations contains a non-xyz point")
        xyz = tuple(float(axis) for axis in point)
        if not all(math.isfinite(axis) for axis in xyz):
            raise ValueError("gt_locations contains non-finite coordinates")
        result.append(xyz)
    return result


def _distance_xy(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _path_length_xy(points: Sequence[Sequence[float]]) -> float:
    return sum(_distance_xy(first, second) for first, second in zip(points, points[1:]))


def _chord_deviation(points: Sequence[Sequence[float]]) -> float:
    first, last = points[0], points[-1]
    dx, dy = last[0] - first[0], last[1] - first[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return math.inf
    return max(abs(dx * (first[1] - point[1]) - (first[0] - point[0]) * dy) / norm for point in points)


def _segment_for_distance(points: Sequence[tuple[float, float, float]], start: int, target_m: float) -> tuple[int, list[tuple[float, float, float]]] | None:
    for end in range(start + 1, len(points)):
        segment = list(points[start : end + 1])
        path_length = _path_length_xy(segment)
        if path_length < target_m:
            continue
        chord = _distance_xy(segment[0], segment[-1])
        if not target_m - 0.35 <= chord <= target_m + 0.35:
            return None
        if path_length / chord > MAX_PATH_TO_CHORD_RATIO or _chord_deviation(segment) > MAX_CHORD_DEVIATION_M:
            return None
        if max(point[2] for point in segment) - min(point[2] for point in segment) > MAX_HEIGHT_SPAN_M:
            return None
        return end, segment
    return None


def propose_from_routes(routes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layouts: list[dict[str, Any]] = []
    seen_layout_ids: set[str] = set()
    for route in sorted(routes, key=lambda item: (str(item.get("scene_name", "")), str(item.get("route_id", "")))):
        scene = str(route.get("scene_name") or str(route.get("scene_id", "")).replace("\\", "/").split("/")[1])
        route_id = str(route.get("route_id", ""))
        points = _points(route.get("gt_locations"))
        # Sparse starts keep proposals auditable and prevent a dense sliding
        # window from manufacturing many near-duplicates.
        for start_index in range(0, max(1, len(points) - 1), 4):
            for target_distance in TARGET_DISTANCES_M:
                selected = _segment_for_distance(points, start_index, target_distance)
                if selected is None:
                    continue
                end_index, segment = selected
                start, midpoint = segment[0], segment[-1]
                dx, dy = midpoint[0] - start[0], midpoint[1] - start[1]
                distance = math.hypot(dx, dy)
                unit_x, unit_y = dx / distance, dy / distance
                lateral_x, lateral_y = -unit_y, unit_x
                box_a = [midpoint[0] + lateral_x * BOX_HALF_SEPARATION_M,
                         midpoint[1] + lateral_y * BOX_HALF_SEPARATION_M, midpoint[2] + 0.25]
                box_b = [midpoint[0] - lateral_x * BOX_HALF_SEPARATION_M,
                         midpoint[1] - lateral_y * BOX_HALF_SEPARATION_M, midpoint[2] + 0.25]
                midline_yaw = math.degrees(math.atan2(dy, dx))
                layout_key = {
                    "scene": scene, "start_position": list(start),
                    "box_a_position": box_a, "box_b_position": box_b,
                }
                layout_id = "proposal_layout_" + _hash(layout_key)
                if layout_id in seen_layout_ids:
                    continue
                seen_layout_ids.add(layout_id)
                for offset in OFFSETS_DEGREES:
                    layouts.append({
                        "candidate_id": "proposal_" + _hash([layout_id, offset]),
                        "layout_proposal_id": layout_id,
                        "scene": scene,
                        "scene_id": str(route.get("scene_id", "")),
                        "source_route_id": route_id,
                        "source_episode_id": str(route.get("source_episode_id", "")),
                        "source_gt_index_range": [start_index, end_index],
                        "source_gt_segment": [list(point) for point in segment],
                        "start_position": list(start),
                        "box_a_position": box_a,
                        "box_b_position": box_b,
                        "box_center_separation_m": 2.0 * BOX_HALF_SEPARATION_M,
                        "midline_yaw_degrees": midline_yaw,
                        "camera_yaw_degrees": midline_yaw + offset,
                        "relative_heading_offset_degrees": offset,
                        "nominal_chord_distance_m": distance,
                        "status": "pending_live_two_box_visibility_gate",
                    })
    return {
        "format": FORMAT,
        "generation": {
            "relative_heading_offsets_degrees": list(OFFSETS_DEGREES),
            "target_distances_m": list(TARGET_DISTANCES_M),
            "box_center_separation_m": 2.0 * BOX_HALF_SEPARATION_M,
            "maximum_path_to_chord_ratio": MAX_PATH_TO_CHORD_RATIO,
            "maximum_chord_deviation_m": MAX_CHORD_DEVIATION_M,
            "accepted_as_training_data": False,
        },
        "layout_proposals": len(seen_layout_ids),
        "candidates": layouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.routes.read_text(encoding="utf-8"))
    routes = payload["routes"] if isinstance(payload, Mapping) else payload
    result = propose_from_routes(routes)
    if not result["candidates"]:
        raise SystemExit("no straight, flat 4.5--6.5 m layout proposals")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "layout_proposals": result["layout_proposals"], "pose_candidates": len(result["candidates"])}))


if __name__ == "__main__":
    main()
