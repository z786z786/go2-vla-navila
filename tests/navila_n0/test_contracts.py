from __future__ import annotations

import unittest

from src.navila_n0.contracts import POLICY_CONTRACT, required_training_steps, validate_training_plan


def valid_plan() -> dict[str, object]:
    return {
        **POLICY_CONTRACT,
        "dataset_schema_version": "navila_semantic_short_vln_v1",
        "dataset_root": "/new/navila_semantic/train",
        "source_checkpoint": None,
        "selection_manifest": "/new/navila_semantic/selection.json",
        "acceptance_reference": "/new/navila_semantic/acceptance.json",
        "policy_signals": ["body_vx", "body_vy", "body_yaw_rate"],
        "n_train_samples": 1234,
        "effective_batch_size": 8,
        "training_steps": required_training_steps(1234, 8),
    }


class TestN0PolicyContract(unittest.TestCase):
    def test_fixed_step_formula(self) -> None:
        self.assertEqual(required_training_steps(100, 8), 250)
        self.assertEqual(required_training_steps(100000, 1), 30000)
        with self.assertRaises(ValueError):
            required_training_steps(0, 1)

    def test_valid_new_plan_is_accepted(self) -> None:
        self.assertEqual(validate_training_plan(valid_plan()), [])

    def test_legacy_state_and_artifacts_are_rejected(self) -> None:
        plan = valid_plan()
        plan["state"] = {"shape": [30], "names": ["rpy"]}
        plan["dataset_root"] = "/mnt/wxh/go2_short_vln/data/lerobot/short_vln_v1"
        plan["source_checkpoint"] = "/mnt/wxh/go2_short_vln/outputs/m6/checkpoint"
        plan["policy_signals"] = ["body_vx", "rpy", "goal_distance"]
        errors = validate_training_plan(plan)
        self.assertTrue(any("state must" in error for error in errors))
        self.assertTrue(any("audit-only" in error for error in errors))
        self.assertTrue(any("prohibited" in error for error in errors))
        self.assertTrue(any("legacy checkpoints" in error for error in errors))

    def test_wrong_step_budget_is_rejected(self) -> None:
        plan = valid_plan()
        plan["training_steps"] = 2000
        self.assertTrue(any("training_steps" in error for error in validate_training_plan(plan)))
