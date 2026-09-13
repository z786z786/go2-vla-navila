"""Auditable NaVILA dual-target geometry generation and orthogonal OOD splits.

This module is deliberately simulator-free.  A renderer supplies initial-frame
visibility measurements for each proposed pose; this module applies the fixed
gate, clusters nearby layouts, derives the four color/instruction tasks, assigns
orthogonal geometry/angle partitions, and emits balance statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


FORMAT = "navila-dual-target-ood-split-v1"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
MIN_PIXEL_FRACTION = 0.005
MIN_VISIBILITY_RATIO = 0.50
HORIZONTAL_EDGE_MARGIN_FRACTION = 0.05
SEEN_ANGLE_MIN_DEG = -8.0
SEEN_ANGLE_MAX_DEG = 8.0
OOD_NEGATIVE_DEG = (-20.0, -12.0)
OOD_POSITIVE_DEG = (12.0, 20.0)
DEFAULT_CLUSTER_RADIUS_M = 0.50
DEFAULT_OOD_SCENE_FRACTION = 0.25
RED_INSTRUCTION = "Go to the red box and stop in front of it."
BLUE_INSTRUCTION = "Go to the blue box and stop in front of it."


class DatasetDesignError(ValueError):
    """Raised when a manifest cannot satisfy the declared data contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: Any, length: int = 20) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()[:length]


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise DatasetDesignError(f"{label} must be a {length}-D sequence")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise DatasetDesignError(f"{label} must be finite")
    return result


def _distance_xy(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _bearing_degrees(start: Sequence[float], target: Sequence[float]) -> float:
    return math.degrees(math.atan2(float(target[1]) - float(start[1]), float(target[0]) - float(start[0])))


def bearing_midline_degrees(start: Sequence[float], box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Circular midpoint of the two box bearings in the scene-local frame."""
    first, second = map(math.radians, (_bearing_degrees(start, box_a), _bearing_degrees(start, box_b)))
    x, y = math.cos(first) + math.cos(second), math.sin(first) + math.sin(second)
    if math.hypot(x, y) <= 1e-9:
        raise DatasetDesignError("A/B bearings are antipodal; their midline is undefined")
    return math.degrees(math.atan2(y, x))


def relative_heading_offset_degrees(
    start: Sequence[float], box_a: Sequence[float], box_b: Sequence[float], camera_yaw_degrees: float
) -> float:
    """Camera heading minus the A/B bearing midline; never world yaw itself."""
    return _wrap_degrees(float(camera_yaw_degrees) - bearing_midline_degrees(start, box_a, box_b))


def angle_partition(offset_degrees: float) -> str:
    value = float(offset_degrees)
    if not math.isfinite(value):
        raise DatasetDesignError("relative heading offset must be finite")
    if SEEN_ANGLE_MIN_DEG <= value <= SEEN_ANGLE_MAX_DEG:
        return "seen"
    if (-12.0 < value < -8.0) or (8.0 < value < 12.0):
        return "guard"
    if OOD_NEGATIVE_DEG[0] <= value <= OOD_NEGATIVE_DEG[1] or OOD_POSITIVE_DEG[0] <= value <= OOD_POSITIVE_DEG[1]:
        return "ood"
    return "outside"


def visibility_gate(visibility: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the exact initial-frame two-box visibility gate."""
    width = int(visibility.get("image_width", 0))
    height = int(visibility.get("image_height", 0))
    if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
        return {"passed": False, "errors": ["initial RGB must be exactly 512x512"]}
    errors: list[str] = []
    slots: dict[str, Any] = {}
    for slot in ("A", "B"):
        value = visibility.get(slot)
        if not isinstance(value, Mapping):
            errors.append(f"slot {slot} visibility is missing")
            continue
        try:
            visible = int(value["visible_pixels"])
            projected = int(value["projected_pixels"])
            center_x = float(value["projection_center_x_px"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"slot {slot} visibility fields are invalid")
            continue
        pixel_fraction = visible / (width * height)
        ratio = visible / projected if projected > 0 else 0.0
        edge_fraction = min(center_x / width, (width - center_x) / width)
        slot_errors = []
        if visible < 0 or projected <= 0 or visible > projected:
            slot_errors.append("pixel counts are inconsistent")
        if pixel_fraction + 1e-12 < MIN_PIXEL_FRACTION:
            slot_errors.append("visible pixel fraction is below 0.5%")
        if ratio + 1e-12 < MIN_VISIBILITY_RATIO:
            slot_errors.append("visibility ratio is below 50%")
        if edge_fraction + 1e-12 < HORIZONTAL_EDGE_MARGIN_FRACTION:
            slot_errors.append("projection center is within 5% of a horizontal edge")
        if slot_errors:
            errors.extend(f"slot {slot}: {message}" for message in slot_errors)
        slots[slot] = {
            "visible_pixels": visible,
            "projected_pixels": projected,
            "pixel_fraction": pixel_fraction,
            "visibility_ratio": ratio,
            "projection_center_x_px": center_x,
            "horizontal_edge_margin_fraction": edge_fraction,
            "passed": not slot_errors,
        }
    return {"passed": not errors and set(slots) == {"A", "B"}, "errors": errors, "slots": slots}


def _layout_id(record: Mapping[str, Any]) -> str:
    return "layout_" + _hash({
        "scene": record["scene"],
        "start_position": record["start_position"],
        "box_a_position": record["box_a_position"],
        "box_b_position": record["box_b_position"],
    })


def _nearby(first: Mapping[str, Any], second: Mapping[str, Any], radius_m: float) -> bool:
    return (
        first["scene"] == second["scene"]
        and _distance_xy(first["start_position"], second["start_position"]) <= radius_m
        and _distance_xy(first["box_a_position"], second["box_a_position"]) <= radius_m
        and _distance_xy(first["box_b_position"], second["box_b_position"]) <= radius_m
    )


def geometry_clusters(records: Sequence[Mapping[str, Any]], radius_m: float = DEFAULT_CLUSTER_RADIUS_M) -> dict[str, str]:
    """Conservative connected components over scene/start/A/B proximity."""
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise DatasetDesignError("geometry cluster radius must be positive and finite")
    layout_records: dict[str, Mapping[str, Any]] = {}
    for record in records:
        layout_records.setdefault(_layout_id(record), record)
    identifiers = sorted(layout_records)
    parent = list(range(len(identifiers)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, left_id in enumerate(identifiers):
        for right in range(left + 1, len(identifiers)):
            if _nearby(layout_records[left_id], layout_records[identifiers[right]], radius_m):
                union(left, right)
    components: dict[int, list[str]] = defaultdict(list)
    for index, identifier in enumerate(identifiers):
        components[find(index)].append(identifier)
    result: dict[str, str] = {}
    for members in components.values():
        scene = str(layout_records[members[0]]["scene"])
        cluster_id = "geometry_" + _hash({"scene": scene, "layouts": sorted(members)})
        for identifier in members:
            result[identifier] = cluster_id
    return result


def _slot_sides(record: Mapping[str, Any]) -> dict[str, str]:
    yaw = float(record["camera_yaw_degrees"])
    result = {}
    for slot, key in (("A", "box_a_position"), ("B", "box_b_position")):
        relative = _wrap_degrees(_bearing_degrees(record["start_position"], record[key]) - yaw)
        result[slot] = "left" if relative > 1e-8 else "right" if relative < -1e-8 else "center"
    return result


def _task_rows(base: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    sides = _slot_sides(base)
    rows = []
    for configuration, color_a, color_b in (
        ("A_red_B_blue", "red", "blue"),
        ("A_blue_B_red", "blue", "red"),
    ):
        for target_color, instruction in (("red", RED_INSTRUCTION), ("blue", BLUE_INSTRUCTION)):
            target_slot = "A" if color_a == target_color else "B"
            rows.append({
                "task_id": "task_" + _hash([base["base_geometry_id"], configuration, target_color]),
                "base_geometry_id": base["base_geometry_id"],
                "layout_id": base["layout_id"],
                "geometry_cluster_id": base["geometry_cluster_id"],
                "split": split,
                "color_configuration": configuration,
                "slot_a_color": color_a,
                "slot_b_color": color_b,
                "target_color": target_color,
                "target_slot": target_slot,
                "target_side": sides[target_slot],
                "red_side": sides["A" if color_a == "red" else "B"],
                "blue_side": sides["A" if color_a == "blue" else "B"],
                "instruction": instruction,
            })
    return rows


def _select_ood_scenes(scenes: Sequence[str], fraction: float) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise DatasetDesignError("OOD scene fraction must be in (0,1)")
    unique = sorted(set(scenes))
    if len(unique) < 2:
        raise DatasetDesignError("at least two scenes are required for scene-held-out geometry OOD")
    count = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    ranked = sorted(unique, key=lambda value: (_hash(["ood_scene", value], 64), value))
    return set(ranked[:count])


def _balance(counter: Counter[str], left: str, right: str, expected_total: int) -> dict[str, Any]:
    observed = counter[left] + counter[right]
    return {
        left: counter[left], right: counter[right], "difference": abs(counter[left] - counter[right]),
        "covered": observed, "expected": expected_total,
        "balanced": counter[left] == counter[right] and observed == expected_total,
    }


def summarize(tasks: Sequence[Mapping[str, Any]], bases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for base in bases:
        base_by_split[str(base["split"])].append(base)
    for task in tasks:
        task_by_split[str(task["split"])].append(task)
    result = {}
    for split in ("train", "seen", "orientation_ood", "geometry_ood", "combined_ood"):
        split_tasks, split_bases = task_by_split[split], base_by_split[split]
        counts: Counter[str] = Counter()
        for task in split_tasks:
            counts[f"red-{task['red_side']}"] += 1
            counts[f"blue-{task['blue_side']}"] += 1
            counts[f"target-{task['target_color']}"] += 1
            counts[f"target-{task['target_side']}"] += 1
        result[split] = {
            "base_geometries": len(split_bases),
            "tasks": len(split_tasks),
            "scenes": len({str(item["scene"]) for item in split_bases}),
            "scene_values": sorted({str(item["scene"]) for item in split_bases}),
            "geometry_clusters": len({str(item["geometry_cluster_id"]) for item in split_bases}),
            "relative_heading_degrees": sorted(float(item["relative_heading_offset_degrees"]) for item in split_bases),
            "balance": {
                "red_side": _balance(counts, "red-left", "red-right", len(split_tasks)),
                "blue_side": _balance(counts, "blue-left", "blue-right", len(split_tasks)),
                "target_color": _balance(counts, "target-red", "target-blue", len(split_tasks)),
                "target_side": _balance(counts, "target-left", "target-right", len(split_tasks)),
            },
        }
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("format") != FORMAT:
        errors.append("wrong manifest format")
    bases = manifest.get("base_geometries")
    tasks = manifest.get("tasks")
    if not isinstance(bases, list) or not isinstance(tasks, list):
        return [*errors, "base_geometries and tasks must be arrays"]
    by_base: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cluster_partition: dict[str, set[str]] = defaultdict(set)
    layout_angle_cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    for base in bases:
        base_id = str(base.get("base_geometry_id", ""))
        if not base_id:
            errors.append("base geometry lacks an ID")
        if visibility_gate(base.get("visibility", {})).get("passed") is not True:
            errors.append(f"{base_id}: failed visibility entered final manifest")
        partition = angle_partition(float(base.get("relative_heading_offset_degrees", math.nan)))
        expected = "seen" if base.get("split") in {"train", "seen", "geometry_ood"} else "ood"
        if partition != expected:
            errors.append(f"{base_id}: split mixes angle partitions")
        geometry = str(base.get("geometry_partition", ""))
        split = str(base.get("split", ""))
        if split in {"train", "seen", "orientation_ood"} and geometry != "seen":
            errors.append(f"{base_id}: geometry OOD mixed into {split}")
        if split in {"geometry_ood", "combined_ood"} and geometry != "ood":
            errors.append(f"{base_id}: seen geometry mixed into {split}")
        cluster_partition[str(base.get("geometry_cluster_id", ""))].add(geometry)
        layout_angle_cells[(str(base.get("layout_id", "")), expected)].add(split)
    for task in tasks:
        by_base[str(task.get("base_geometry_id", ""))].append(task)
    expected_combinations = {
        ("A_red_B_blue", "red"), ("A_red_B_blue", "blue"),
        ("A_blue_B_red", "red"), ("A_blue_B_red", "blue"),
    }
    for base in bases:
        base_id = str(base["base_geometry_id"])
        rows = by_base.get(base_id, [])
        observed = {(str(row.get("color_configuration")), str(row.get("target_color"))) for row in rows}
        if len(rows) != 4 or observed != expected_combinations:
            errors.append(f"{base_id}: four derived tasks are incomplete")
        if {row.get("split") for row in rows} != {base.get("split")}:
            errors.append(f"{base_id}: four derived tasks cross splits")
    for cluster, partitions in cluster_partition.items():
        if partitions != ({"seen"} if "seen" in partitions else {"ood"}):
            errors.append(f"{cluster}: nearby geometry crosses geometry partitions")
    for (layout, angle_kind), cells in layout_angle_cells.items():
        if angle_kind == "seen" and "train" in cells and "seen" in cells:
            # Intentional: exact layout is shared, but exact base geometry IDs/angles are not.
            continue
    for split, row in manifest.get("statistics", {}).items():
        for name, balance in row.get("balance", {}).items():
            if row.get("tasks", 0) and balance.get("balanced") is not True:
                errors.append(f"{split}: {name} is imbalanced")
    return errors


def build_manifest(
    candidates: Iterable[Mapping[str, Any]], *, cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    ood_scene_fraction: float = DEFAULT_OOD_SCENE_FRACTION,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in candidates:
        record = dict(raw)
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id or candidate_id in identifiers:
            raise DatasetDesignError("candidate IDs must be non-empty and unique")
        identifiers.add(candidate_id)
        scene = str(record.get("scene", ""))
        if not scene:
            raise DatasetDesignError(f"{candidate_id}: scene is required")
        start = _finite_vector(record.get("start_position"), 3, f"{candidate_id}.start_position")
        box_a = _finite_vector(record.get("box_a_position"), 3, f"{candidate_id}.box_a_position")
        box_b = _finite_vector(record.get("box_b_position"), 3, f"{candidate_id}.box_b_position")
        camera_yaw = float(record.get("camera_yaw_degrees", math.nan))
        declared_offset = float(record.get("relative_heading_offset_degrees", math.nan))
        computed_offset = relative_heading_offset_degrees(start, box_a, box_b, camera_yaw)
        if not math.isfinite(declared_offset) or abs(_wrap_degrees(declared_offset - computed_offset)) > 1e-6:
            raise DatasetDesignError(f"{candidate_id}: relative heading is not camera yaw minus A/B midline")
        angle = angle_partition(declared_offset)
        gate = visibility_gate(record.get("visibility", {}))
        normalized = {
            **record,
            "scene": scene,
            "start_position": list(start),
            "box_a_position": list(box_a),
            "box_b_position": list(box_b),
            "camera_yaw_degrees": camera_yaw,
            "relative_heading_offset_degrees": declared_offset,
            "angle_partition": angle,
            "visibility_gate": gate,
        }
        if angle in {"guard", "outside"} or not gate["passed"]:
            rejected.append({
                "candidate_id": candidate_id,
                "angle_partition": angle,
                "visibility_gate": gate,
                "reasons": ([f"angle_partition:{angle}"] if angle in {"guard", "outside"} else []) + list(gate["errors"]),
            })
        else:
            accepted.append(normalized)
    if not accepted:
        raise DatasetDesignError("no candidate passed angle and visibility gates")
    by_exact_layout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in accepted:
        by_exact_layout[_layout_id(item)].append(item)
    eligible: list[dict[str, Any]] = []
    for layout_id, values in by_exact_layout.items():
        maps = {
            name: {round(float(item["relative_heading_offset_degrees"]), 6): item
                   for item in values if item["angle_partition"] == name}
            for name in ("seen", "ood")
        }
        paired = {
            name: {offset for offset in values_by_offset if offset > 0.0 and -offset in values_by_offset}
            for name, values_by_offset in maps.items()
        }
        keep_seen = ({0.0} if 0.0 in maps["seen"] else set()) | paired["seen"] | {-value for value in paired["seen"]}
        keep_ood = paired["ood"] | {-value for value in paired["ood"]}
        # One symmetric seen pair is reserved for Seen evaluation.  At least
        # one additional seen pose remains for training, and a symmetric OOD
        # pair supplies Orientation/Combined-OOD without directional bias.
        complete = bool(paired["seen"] and keep_seen - {max(paired["seen"]), -max(paired["seen"])} and paired["ood"])
        if not complete:
            for item in values:
                rejected.append({
                    "candidate_id": item["candidate_id"], "angle_partition": item["angle_partition"],
                    "visibility_gate": item["visibility_gate"],
                    "reasons": ["layout_lacks_symmetric_seen_train_eval_or_ood_angle_coverage"],
                })
            continue
        for item in values:
            offset = round(float(item["relative_heading_offset_degrees"]), 6)
            if offset in (keep_seen if item["angle_partition"] == "seen" else keep_ood):
                eligible.append(item)
            else:
                rejected.append({
                    "candidate_id": item["candidate_id"], "angle_partition": item["angle_partition"],
                    "visibility_gate": item["visibility_gate"], "reasons": ["unpaired_relative_heading_offset"],
                })
    if not eligible:
        raise DatasetDesignError("no layout retained symmetric seen/train and OOD angle coverage")
    clusters = geometry_clusters(eligible, cluster_radius_m)
    ood_scenes = _select_ood_scenes([str(item["scene"]) for item in eligible], ood_scene_fraction)
    by_layout_seen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        item["layout_id"] = _layout_id(item)
        item["geometry_cluster_id"] = clusters[item["layout_id"]]
        item["geometry_partition"] = "ood" if item["scene"] in ood_scenes else "seen"
        item["base_geometry_id"] = "base_" + _hash([
            item["layout_id"], round(float(item["relative_heading_offset_degrees"]), 8)
        ])
        if item["geometry_partition"] == "seen" and item["angle_partition"] == "seen":
            by_layout_seen[item["layout_id"]].append(item)
    for layout_id, values in by_layout_seen.items():
        if len(values) < 3:
            raise DatasetDesignError(f"{layout_id}: seen geometry needs at least three passing seen-angle poses for train/Seen isolation")
        by_offset = {round(float(item["relative_heading_offset_degrees"]), 6): item for item in values}
        symmetric = sorted(
            (abs(offset), by_offset[offset], by_offset[-offset])
            for offset in by_offset if offset > 0.0 and -offset in by_offset
        )
        if not symmetric:
            raise DatasetDesignError(f"{layout_id}: seen-angle poses need a passing +/-theta pair")
        _, negative_or_positive, opposite = symmetric[-1]
        reserved_ids = {negative_or_positive["base_geometry_id"], opposite["base_geometry_id"]}
        for item in values:
            item["split"] = "seen" if item["base_geometry_id"] in reserved_ids else "train"
    final_bases: list[dict[str, Any]] = []
    for item in eligible:
        if "split" not in item:
            if item["geometry_partition"] == "seen":
                item["split"] = "orientation_ood"
            elif item["angle_partition"] == "seen":
                item["split"] = "geometry_ood"
            else:
                item["split"] = "combined_ood"
        final_bases.append(item)
    tasks = [task for base in final_bases for task in _task_rows(base, str(base["split"]))]
    statistics = summarize(tasks, final_bases)
    manifest = {
        "format": FORMAT,
        "contract": {
            "base_geometry_fields": ["scene", "start_position", "box_a_position", "box_b_position", "relative_heading_offset_degrees"],
            "relative_heading_definition": "camera_yaw_minus_circular_midline_of_A_and_B_bearings",
            "visibility": {
                "minimum_visible_pixel_fraction": MIN_PIXEL_FRACTION,
                "minimum_visibility_ratio": MIN_VISIBILITY_RATIO,
                "minimum_horizontal_projection_center_margin_fraction": HORIZONTAL_EDGE_MARGIN_FRACTION,
                "both_slots_required": True,
            },
            "angles_degrees": {
                "train_and_seen": [SEEN_ANGLE_MIN_DEG, SEEN_ANGLE_MAX_DEG],
                "guard_excluded": [[-12.0, -8.0], [8.0, 12.0]],
                "orientation_ood": [list(OOD_NEGATIVE_DEG), list(OOD_POSITIVE_DEG)],
            },
            "geometry_cluster_radius_m": cluster_radius_m,
            "geometry_cluster_fields": ["scene", "start_position", "box_a_position", "box_b_position"],
            "four_tasks_per_base_geometry": True,
        },
        "geometry_partition": {
            "policy": "hold_out_complete_scenes; nearby clusters inherit scene ownership",
            "ood_scene_fraction": ood_scene_fraction,
            "seen_scenes": sorted({str(item["scene"]) for item in final_bases} - ood_scenes),
            "ood_scenes": sorted(ood_scenes),
        },
        "base_geometries": sorted(final_bases, key=lambda item: item["base_geometry_id"]),
        "tasks": sorted(tasks, key=lambda item: item["task_id"]),
        "rejected_candidates": rejected,
        "statistics": statistics,
    }
    errors = validate_manifest(manifest)
    if errors:
        raise DatasetDesignError("manifest validation failed: " + "; ".join(errors[:20]))
    return manifest


def write_report(manifest: Mapping[str, Any], path: Path) -> None:
    stats = manifest["statistics"]
    lines = [
        "# NaVILA dual-target OOD split audit", "", "Status: **PASS**.", "",
        f"Accepted base geometries: **{len(manifest['base_geometries'])}**; derived tasks: **{len(manifest['tasks'])}**.",
        f"Rejected candidates: **{len(manifest['rejected_candidates'])}**.", "",
        "| Split | Base | Tasks | Scenes | Geometry clusters | Red L/R | Blue L/R | Target red/blue | Target L/R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "seen", "orientation_ood", "geometry_ood", "combined_ood"):
        row, balance = stats[split], stats[split]["balance"]
        keys = {
            "red_side": ("red-left", "red-right"),
            "blue_side": ("blue-left", "blue-right"),
            "target_color": ("target-red", "target-blue"),
            "target_side": ("target-left", "target-right"),
        }
        pair = lambda name: f"{balance[name][keys[name][0]]}/{balance[name][keys[name][1]]}"  # noqa: E731
        lines.append(
            f"| {split} | {row['base_geometries']} | {row['tasks']} | {row['scenes']} | {row['geometry_clusters']} | "
            f"{pair('red_side')} | {pair('blue_side')} | {pair('target_color')} | {pair('target_side')} |"
        )
    lines.extend(["", "## Relative heading distributions", ""])
    for split in ("train", "seen", "orientation_ood", "geometry_ood", "combined_ood"):
        lines.append(f"- `{split}`: `{stats[split]['relative_heading_degrees']}`")
    lines.extend(["", "## Scene ownership", "",
        f"- Seen scenes: `{manifest['geometry_partition']['seen_scenes']}`",
        f"- Geometry-OOD scenes: `{manifest['geometry_partition']['ood_scenes']}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cluster-radius-m", type=float, default=DEFAULT_CLUSTER_RADIUS_M)
    parser.add_argument("--ood-scene-fraction", type=float, default=DEFAULT_OOD_SCENE_FRACTION)
    args = parser.parse_args()
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    records = payload["candidates"] if isinstance(payload, Mapping) else payload
    manifest = build_manifest(records, cluster_radius_m=args.cluster_radius_m, ood_scene_fraction=args.ood_scene_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(manifest, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "base_geometries": len(manifest["base_geometries"]), "tasks": len(manifest["tasks"])}))


if __name__ == "__main__":
    main()
