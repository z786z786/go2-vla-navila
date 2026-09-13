"""Geometry-group and four-combination bookkeeping without simulator imports."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .contracts import BLUE_TASK, RED_TASK

RED_INSTRUCTION = RED_TASK
BLUE_INSTRUCTION = BLUE_TASK
VALID_SPLITS = frozenset({"train", "validation", "test"})
_GEOMETRY_SIGNATURE_BIN_M = 0.05


class GeometryValidationError(ValueError):
    """Raised for a layout that cannot enter the DT1 simulation precheck."""


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def distance_to(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def translated(self, unit_x: float, unit_y: float, amount: float) -> "Vec2":
        return Vec2(self.x + unit_x * amount, self.y + unit_y * amount)

    def finite(self, label: str) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise GeometryValidationError(f"{label} must contain finite coordinates")


@dataclass(frozen=True)
class TargetSlot:
    slot_id: str
    center: Vec2
    front_unit: Vec2

    def validate(self) -> None:
        if self.slot_id not in {"A", "B"}:
            raise GeometryValidationError("target slots must be named A and B")
        self.center.finite(f"slot {self.slot_id} center")
        self.front_unit.finite(f"slot {self.slot_id} front direction")
        norm = math.hypot(self.front_unit.x, self.front_unit.y)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise GeometryValidationError(f"slot {self.slot_id} front direction must be unit length")


@dataclass(frozen=True)
class ParkingRegion:
    target_slot: str
    center: Vec2
    radius_m: float

    def contains(self, point: Vec2) -> bool:
        return point.distance_to(self.center) <= self.radius_m


@dataclass(frozen=True)
class GeometryGroup:
    """One base geometry; all color/instruction variants inherit its lineage."""

    geometry_group_id: str
    lineage_root_id: str
    slot_a: TargetSlot
    slot_b: TargetSlot
    start: Vec2
    box_half_depth_m: float = 0.25
    # A 0.45 m stand-off with a 0.30 m parking disc left only 0.15 m from
    # the box at the disc edge.  DT1's real Go2 contact audit measured about
    # 0.34 m of forward body/head extent, so the development geometry keeps a
    # 0.45 m edge clearance instead of treating the old geometric minimum as
    # a physical safety claim.
    stand_off_m: float = 0.75
    parking_radius_m: float = 0.30
    min_effective_travel_m: float = 0.80
    # Keep the development layouts fixed to their 0.45 m root-disc-to-box
    # face clearance.  The controlled contact diagnostic supplies its own
    # explicit lower value because it intentionally reaches a box.
    min_box_edge_clearance_m: float = 0.45
    sim_validation_status: str = "pending_sim_validation"

    def slot(self, slot_id: str) -> TargetSlot:
        if slot_id == "A":
            return self.slot_a
        if slot_id == "B":
            return self.slot_b
        raise GeometryValidationError(f"unknown target slot: {slot_id!r}")

    def parking_region(self, slot_id: str) -> ParkingRegion:
        slot = self.slot(slot_id)
        # The parking point follows the fixed box-facing direction, never the
        # start pose or an expert route.
        distance = self.box_half_depth_m + self.stand_off_m
        return ParkingRegion(
            target_slot=slot_id,
            center=slot.center.translated(slot.front_unit.x, slot.front_unit.y, distance),
            radius_m=self.parking_radius_m,
        )

    def validate_geometry(self) -> None:
        if not self.geometry_group_id or not self.lineage_root_id:
            raise GeometryValidationError("geometry_group_id and lineage_root_id are required")
        self.slot_a.validate()
        self.slot_b.validate()
        self.start.finite("start")
        if self.slot_a.slot_id == self.slot_b.slot_id:
            raise GeometryValidationError("two distinct target slots are required")
        for label, value in (
            ("box_half_depth_m", self.box_half_depth_m),
            ("stand_off_m", self.stand_off_m),
            ("parking_radius_m", self.parking_radius_m),
            ("min_effective_travel_m", self.min_effective_travel_m),
            ("min_box_edge_clearance_m", self.min_box_edge_clearance_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise GeometryValidationError(f"{label} must be positive and finite")
        # The nearest edge of the parking disc must still be in front of the
        # target-box face.  Robot-body collision clearance remains a DT1
        # physics check, but this rejects obviously intersecting candidates.
        if self.stand_off_m - self.parking_radius_m < self.min_box_edge_clearance_m:
            raise GeometryValidationError("parking-region edge lacks box clearance")
        regions = (self.parking_region("A"), self.parking_region("B"))
        if regions[0].center.distance_to(regions[1].center) <= regions[0].radius_m + regions[1].radius_m:
            raise GeometryValidationError("parking regions must not overlap")
        for region in regions:
            if region.contains(self.start):
                raise GeometryValidationError("start must be outside every parking region")
            if self.start.distance_to(region.center) < self.min_effective_travel_m:
                raise GeometryValidationError("start lacks the minimum effective travel distance")
        if self.sim_validation_status != "pending_sim_validation":
            raise GeometryValidationError("DT0 candidates must remain pending_sim_validation")

    @property
    def geometry_signature(self) -> tuple[int, ...]:
        """Conservative, translation/mirror-invariant split-leakage key.

        It deliberately identifies renamed identical layouts and left/right
        mirrors as one base geometry.  A 5 cm bin also catches the small
        perturbations that must inherit a lineage instead of becoming a new
        independent group.  This is an audit guard, not a scene generator.
        """
        regions = (self.parking_region("A"), self.parking_region("B"))
        values = (
            *sorted(self.start.distance_to(region.center) for region in regions),
            regions[0].center.distance_to(regions[1].center),
            self.parking_radius_m,
            self.box_half_depth_m,
            self.stand_off_m,
        )
        return tuple(int(round(value / _GEOMETRY_SIGNATURE_BIN_M)) for value in values)


@dataclass(frozen=True)
class TaskSpec:
    geometry_group_id: str
    lineage_root_id: str
    color_configuration: str
    slot_a_color: str
    slot_b_color: str
    target_color: str
    target_slot: str
    instruction: str
    start: Vec2
    parking_region: ParkingRegion
    sim_validation_status: str
    geometry_signature: tuple[int, ...]

    @property
    def combination_key(self) -> tuple[str, str]:
        return (self.color_configuration, self.target_color)


def build_four_tasks(group: GeometryGroup) -> tuple[TaskSpec, ...]:
    """Generate exactly the two color assignments × two fixed instructions."""
    group.validate_geometry()
    result: list[TaskSpec] = []
    configurations = (
        ("A_red_B_blue", "red", "blue"),
        ("A_blue_B_red", "blue", "red"),
    )
    for configuration, color_a, color_b in configurations:
        color_to_slot = {color_a: "A", color_b: "B"}
        for color, instruction in (("red", RED_INSTRUCTION), ("blue", BLUE_INSTRUCTION)):
            slot = color_to_slot[color]
            result.append(
                TaskSpec(
                    geometry_group_id=group.geometry_group_id,
                    lineage_root_id=group.lineage_root_id,
                    color_configuration=configuration,
                    slot_a_color=color_a,
                    slot_b_color=color_b,
                    target_color=color,
                    target_slot=slot,
                    instruction=instruction,
                    start=group.start,
                    parking_region=group.parking_region(slot),
                    sim_validation_status=group.sim_validation_status,
                    geometry_signature=group.geometry_signature,
                )
            )
    return tuple(result)


@dataclass(frozen=True)
class SplitTask:
    task: TaskSpec
    split: str


def validate_grouped_splits(assignments: Sequence[SplitTask]) -> list[str]:
    """Return all split leakage and four-combination errors without mutating data."""
    errors: list[str] = []
    if not assignments:
        return ["split manifest must not be empty"]
    by_group: dict[str, list[SplitTask]] = defaultdict(list)
    by_lineage: dict[str, set[str]] = defaultdict(set)
    by_signature: dict[tuple[int, ...], set[str]] = defaultdict(set)
    signature_groups: dict[tuple[int, ...], set[str]] = defaultdict(set)
    expected_configuration = {
        "A_red_B_blue": ("red", "blue"),
        "A_blue_B_red": ("blue", "red"),
    }
    expected_instruction = {"red": RED_INSTRUCTION, "blue": BLUE_INSTRUCTION}
    for assignment in assignments:
        task = assignment.task
        if assignment.split not in VALID_SPLITS:
            errors.append(f"invalid split {assignment.split!r} for {task.geometry_group_id}")
        if task.sim_validation_status != "pending_sim_validation":
            errors.append(f"{task.geometry_group_id} is not marked pending_sim_validation")
        colors = expected_configuration.get(task.color_configuration)
        if colors is None or (task.slot_a_color, task.slot_b_color) != colors:
            errors.append(f"{task.geometry_group_id} has an invalid color configuration payload")
        expected_slot = "A" if task.target_color == task.slot_a_color else "B" if task.target_color == task.slot_b_color else None
        if expected_slot is None or task.target_slot != expected_slot:
            errors.append(f"{task.geometry_group_id} target slot does not match color configuration")
        if task.instruction != expected_instruction.get(task.target_color):
            errors.append(f"{task.geometry_group_id} instruction is not the frozen text for its target color")
        if task.parking_region.target_slot != task.target_slot:
            errors.append(f"{task.geometry_group_id} parking region does not match target slot")
        if not task.geometry_signature:
            errors.append(f"{task.geometry_group_id} lacks a geometry signature")
        by_group[task.geometry_group_id].append(assignment)
        by_lineage[task.lineage_root_id].add(assignment.split)
        by_signature[task.geometry_signature].add(assignment.split)
        signature_groups[task.geometry_signature].add(task.geometry_group_id)
    for lineage, splits in sorted(by_lineage.items()):
        if len(splits) != 1:
            errors.append(f"derived lineage {lineage!r} leaks across splits: {sorted(splits)!r}")
    for signature, splits in sorted(by_signature.items()):
        if len(splits) != 1:
            errors.append(f"same or mirrored geometry signature {signature!r} leaks across splits: {sorted(splits)!r}")
    for signature, group_ids in sorted(signature_groups.items()):
        if len(group_ids) != 1:
            errors.append(f"same or mirrored geometry was assigned multiple group IDs: {sorted(group_ids)!r}")
    expected = {("A_red_B_blue", "red"), ("A_red_B_blue", "blue"), ("A_blue_B_red", "red"), ("A_blue_B_red", "blue")}
    for group_id, grouped in sorted(by_group.items()):
        splits = {item.split for item in grouped}
        if len(splits) != 1:
            errors.append(f"geometry group {group_id!r} appears in multiple splits: {sorted(splits)!r}")
        observed = [item.task.combination_key for item in grouped]
        if set(observed) != expected or len(observed) != len(expected):
            errors.append(f"geometry group {group_id!r} must contain each four-combination task exactly once")
        starts = {item.task.start for item in grouped}
        if len(starts) != 1:
            errors.append(f"geometry group {group_id!r} changes its reset start across combinations")
        roots = {item.task.lineage_root_id for item in grouped}
        if len(roots) != 1:
            errors.append(f"geometry group {group_id!r} changes its lineage root across combinations")
        signatures = {item.task.geometry_signature for item in grouped}
        if len(signatures) != 1:
            errors.append(f"geometry group {group_id!r} changes its geometry signature across combinations")
        parking_by_slot: dict[str, set[ParkingRegion]] = defaultdict(set)
        for item in grouped:
            parking_by_slot[item.task.target_slot].add(item.task.parking_region)
        if set(parking_by_slot) != {"A", "B"} or any(len(regions) != 1 for regions in parking_by_slot.values()):
            errors.append(f"geometry group {group_id!r} changes parking-region mapping across combinations")
    return errors


def example_group() -> GeometryGroup:
    """A CPU-only candidate used by DT0 checks; it is not a simulator asset."""
    return GeometryGroup(
        geometry_group_id="dt0_example_000",
        lineage_root_id="dt0_example_000",
        slot_a=TargetSlot("A", Vec2(2.0, 1.2), Vec2(-1.0, 0.0)),
        slot_b=TargetSlot("B", Vec2(2.0, -1.2), Vec2(-1.0, 0.0)),
        start=Vec2(0.0, 0.0),
    )
