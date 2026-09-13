from __future__ import annotations

import unittest
from dataclasses import replace

from src.dual_target.contracts import (
    ContractValidationError,
    build_policy_envelope,
    validate_action_chunk,
    validate_policy_input,
)
from src.dual_target.layouts import (
    GeometryGroup,
    GeometryValidationError,
    SplitTask,
    TargetSlot,
    Vec2,
    build_four_tasks,
    example_group,
    validate_grouped_splits,
)


class ContractTests(unittest.TestCase):
    def test_policy_builder_keeps_audit_out_of_policy_input(self) -> None:
        envelope = build_policy_envelope(
            image=b"rgb", body_velocity=[0.1, 0.0, -0.1], task="Go to the red box and stop in front of it.",
            audit={"geometry_group_id": "audit-only"},
        )
        self.assertEqual(set(envelope["policy_input"]), {"observation.images.front", "observation.state", "task"})
        self.assertEqual(envelope["audit"]["geometry_group_id"], "audit-only")

    def test_privileged_or_extra_policy_fields_are_rejected(self) -> None:
        payload = {
            "observation.images.front": b"rgb", "observation.state": [0.0, 0.0, 0.0],
            "task": "Go to the blue box and stop in front of it.", "goal_distance": 0.1,
        }
        with self.assertRaisesRegex(ContractValidationError, "prohibited"):
            validate_policy_input(payload)
        payload.pop("goal_distance")
        payload["audit_id"] = "not-a-policy-input"
        with self.assertRaisesRegex(ContractValidationError, "exactly"):
            validate_policy_input(payload)

    def test_only_the_two_frozen_task_texts_are_allowed(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "frozen"):
            validate_policy_input({
                "observation.images.front": b"rgb", "observation.state": [0.0, 0.0, 0.0],
                "task": "Goal coordinates are (1.0, 2.0); turn left.",
            })

    def test_action_chunk_requires_shape_and_finite_values(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "50"):
            validate_action_chunk([[0.0, 0.0, 0.0]] * 49)
        invalid = [[0.0, 0.0, 0.0] for _ in range(50)]
        invalid[17][0] = float("nan")
        with self.assertRaisesRegex(ContractValidationError, "finite"):
            validate_action_chunk(invalid)


class LayoutTests(unittest.TestCase):
    def test_four_combinations_preserve_start_and_swap_target_slots(self) -> None:
        group = example_group()
        tasks = build_four_tasks(group)
        self.assertEqual(len(tasks), 4)
        self.assertEqual({task.combination_key for task in tasks}, {
            ("A_red_B_blue", "red"), ("A_red_B_blue", "blue"),
            ("A_blue_B_red", "red"), ("A_blue_B_red", "blue"),
        })
        self.assertEqual({task.start for task in tasks}, {group.start})
        by_key = {task.combination_key: task.target_slot for task in tasks}
        self.assertEqual(by_key[("A_red_B_blue", "red")], "A")
        self.assertEqual(by_key[("A_blue_B_red", "red")], "B")

    def test_layout_rejects_start_in_parking_region_and_overlap(self) -> None:
        group = example_group()
        invalid_start = GeometryGroup(
            **{**group.__dict__, "start": group.parking_region("A").center}
        )
        with self.assertRaisesRegex(GeometryValidationError, "outside"):
            invalid_start.validate_geometry()
        overlapping = GeometryGroup(
            **{**group.__dict__, "slot_b": TargetSlot("B", group.slot_a.center, Vec2(-1.0, 0.0))}
        )
        with self.assertRaisesRegex(GeometryValidationError, "overlap"):
            overlapping.validate_geometry()

    def test_split_validator_rejects_derived_lineage_leak_and_missing_variant(self) -> None:
        tasks = build_four_tasks(example_group())
        assignments = [SplitTask(task, "train") for task in tasks]
        self.assertEqual(validate_grouped_splits(assignments), [])
        leaked = [*assignments]
        leaked[-1] = SplitTask(tasks[-1], "test")
        errors = validate_grouped_splits(leaked)
        self.assertTrue(any("leaks" in error for error in errors))
        self.assertTrue(any("multiple splits" in error for error in errors))
        self.assertTrue(any("four-combination" in error for error in validate_grouped_splits(assignments[:-1])))

    def test_split_validator_rejects_empty_semantic_tampering_and_mirrored_renaming(self) -> None:
        self.assertTrue(any("empty" in error for error in validate_grouped_splits([])))
        group = example_group()
        tasks = build_four_tasks(group)
        tampered = [replace(task, target_slot="A", instruction="Ignore language and always go A.") for task in tasks]
        errors = validate_grouped_splits([SplitTask(task, "train") for task in tampered])
        self.assertTrue(any("target slot" in error for error in errors))
        self.assertTrue(any("instruction" in error for error in errors))
        mirror = replace(
            group,
            geometry_group_id="renamed_mirror",
            lineage_root_id="renamed_mirror",
            slot_a=replace(group.slot_a, center=Vec2(2.0, -1.2)),
            slot_b=replace(group.slot_b, center=Vec2(2.0, 1.2)),
        )
        mirrored_assignments = [
            *[SplitTask(task, "train") for task in tasks],
            *[SplitTask(task, "test") for task in build_four_tasks(mirror)],
        ]
        errors = validate_grouped_splits(mirrored_assignments)
        self.assertTrue(any("same or mirrored" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
