import unittest

from src.collector.d5_collection import classify_collection_attempt
from src.smolvla.d5_data_expansion import (
    action_regime,
    classify_normalizer_shift,
    diagnostic_labels,
    exposure_steps,
    select_stratified_episodes,
    summarize_regimes,
)


def _episode(identifier, category, *, scene="scene", path_length=2.5):
    return {
        "short_episode_id": identifier,
        "split": "train",
        "scene_id": scene,
        "path_length": path_length,
        "start_pose": {"rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "instruction_metadata": {"category": category},
    }


class D5DataExpansionTests(unittest.TestCase):
    def test_action_regime_uses_declared_open_boundaries(self):
        self.assertEqual(action_regime(0.049), "STRAIGHT")
        self.assertEqual(action_regime(0.05), "TRANSITION")
        self.assertEqual(action_regime(0.10), "TRANSITION")
        self.assertEqual(action_regime(0.101), "LEFT")
        self.assertEqual(action_regime(-0.101), "RIGHT")

    def test_regime_summary_keeps_near_goal_as_overlapping_label(self):
        records = [
            {"expert_wz": 0.0, "distance_to_goal_xy_m": 0.4, "scene_id": "s1", "episode_id": "e1"},
            {"expert_wz": 0.2, "distance_to_goal_xy_m": 0.6, "scene_id": "s1", "episode_id": "e1"},
            {"expert_wz": -0.2, "distance_to_goal_xy_m": 0.3, "scene_id": "s2", "episode_id": "e2"},
            {"expert_wz": 0.08, "distance_to_goal_xy_m": 0.8, "scene_id": "s2", "episode_id": "e2"},
        ]
        summary = summarize_regimes(records)
        self.assertEqual(summary["wz_regimes"]["STRAIGHT"]["frame_count"], 1)
        self.assertEqual(summary["wz_regimes"]["LEFT"]["frame_count"], 1)
        self.assertEqual(summary["wz_regimes"]["RIGHT"]["frame_count"], 1)
        self.assertEqual(summary["wz_regimes"]["TRANSITION"]["frame_count"], 1)
        self.assertEqual(summary["near_goal"]["frame_count"], 2)
        self.assertEqual(summary["near_goal_by_wz_regime"]["RIGHT"]["frame_count"], 1)

    def test_normalizer_shift_flags_scale_and_location_changes(self):
        old = {"vx": {"mean": 0.2, "std": 0.1, "p5": 0.05, "p50": 0.2, "p95": 0.35}, "vy": {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}, "wz": {"mean": 0.0, "std": 0.2, "p5": -0.3, "p50": 0.0, "p95": 0.3}}
        new = {"vx": {"mean": 0.26, "std": 0.14, "p5": 0.1, "p50": 0.26, "p95": 0.42}, "vy": {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}, "wz": {"mean": 0.0, "std": 0.2, "p5": -0.3, "p50": 0.0, "p95": 0.3}}
        result = classify_normalizer_shift(old, new)
        self.assertTrue(result["substantial"])
        self.assertTrue(result["per_dimension"]["vx"]["mean_shift_substantial"])
        self.assertEqual(result["per_dimension"]["vy"]["status"], "degenerate")

    def test_exposure_steps_uses_trainable_anchor_count_and_rounds_up(self):
        self.assertEqual(exposure_steps(2000, 847, 10_001, save_multiple=500), 24000)

    def test_diagnostic_labels_never_merge_train_and_seen_val_routes(self):
        labels = diagnostic_labels()
        self.assertEqual(labels["short_vln_v1_0000"], "TRAIN-ROUTE DIAGNOSTIC")
        self.assertEqual(labels["short_vln_v1_0004"], "TRAIN-ROUTE DIAGNOSTIC")
        self.assertEqual(labels["short_vln_v1_0001"], "SEEN-VAL GENERALIZATION")
        self.assertEqual(labels["short_vln_v1_0003"], "SEEN-VAL GENERALIZATION")
        self.assertEqual(labels["short_vln_v1_0006"], "UNSEEN-SCENE DIAGNOSTIC")

    def test_reselection_keeps_accepted_and_excludes_planner_reject(self):
        episodes = [
            _episode(f"straight_{index}", "straight", path_length=1.8 + index)
            for index in range(3)
        ] + [
            _episode(f"left_{index}", "left_turn", path_length=1.8 + index)
            for index in range(3)
        ] + [
            _episode(f"right_{index}", "right_turn", path_length=1.8 + index)
            for index in range(3)
        ]
        selected = select_stratified_episodes(
            episodes,
            split="train",
            per_category=2,
            required_ids=("straight_0",),
            fixed_ids=("left_0",),
            excluded_ids=("right_0",),
            min_scenes=1,
            min_heading_bins=1,
            min_distance_per_bin=0,
            seed=7,
        )
        selected_ids = {item["short_episode_id"] for item in selected}
        self.assertIn("straight_0", selected_ids)
        self.assertIn("left_0", selected_ids)
        self.assertNotIn("right_0", selected_ids)
        self.assertEqual(len(selected), 6)

    def test_valid_unsuccessful_episode_is_a_planner_reject(self):
        summary = {
            "status": "terminated_without_success",
            "termination_reason": "environment_done",
            "success": False,
            "final_distance_to_goal_xy_m": 0.6757484,
        }
        checks = {
            "record_count_positive": True,
            "record_summary_count_match": True,
            "frame_indices_contiguous": True,
            "no_missing_frames": True,
            "rgb_sequence_normal": True,
            "timestamps_strictly_monotonic": True,
            "timestamp_period_matches_metadata": True,
            "actions_finite": True,
            "actions_in_expected_range": True,
            "planner_equals_locomotion_command": True,
            "robot_state_finite": True,
            "robot_state_dimension_constant": True,
            "normal_termination": True,
            "success": False,
            "final_distance_within_success_radius": False,
        }
        result = classify_collection_attempt(summary, {"passed": False, "checks": checks})
        self.assertEqual(result["kind"], "planner_reject")
        self.assertEqual(result["reason"], "goal_radius_not_reached")

    def test_verified_large_orientation_is_an_expert_rollout_reject(self):
        summary = {
            "status": "terminated_without_success",
            "termination_reason": "official_large_orientation",
            "success": False,
        }
        checks = {
            "record_count_positive": True,
            "record_summary_count_match": True,
            "frame_indices_contiguous": True,
            "no_missing_frames": True,
            "rgb_sequence_normal": True,
            "timestamps_strictly_monotonic": True,
            "timestamp_period_matches_metadata": True,
            "actions_finite": True,
            "actions_in_expected_range": True,
            "planner_equals_locomotion_command": True,
            "robot_state_finite": True,
            "robot_state_dimension_constant": True,
            "normal_termination": False,
        }
        result = classify_collection_attempt(summary, {"passed": False, "checks": checks})
        self.assertEqual(result, {"kind": "expert_rollout_reject", "reason": "official_large_orientation"})

    def test_verified_zero_frame_orientation_is_an_expert_reset_reject(self):
        summary = {
            "status": "terminated_without_success",
            "termination_reason": "official_large_orientation_before_first_action",
            "success": False,
            "record_count": 0,
        }
        result = classify_collection_attempt(
            summary,
            {
                "passed": False,
                "checks": {
                    "zero_frame_official_orientation_evidence": True,
                    "zero_frame_steps_empty": True,
                    "zero_frame_no_rgb": True,
                },
            },
        )
        self.assertEqual(
            result,
            {
                "kind": "expert_reset_reject",
                "reason": "official_large_orientation_before_first_action",
            },
        )

    def test_zero_frame_orientation_without_terminal_evidence_remains_infrastructure_failure(self):
        result = classify_collection_attempt(
            {
                "status": "terminated_without_success",
                "termination_reason": "official_large_orientation_before_first_action",
                "success": False,
                "record_count": 0,
            },
            {"passed": False, "checks": {"zero_frame_official_orientation_evidence": False}},
        )
        self.assertEqual(result["kind"], "infrastructure_failure")

    def test_protocol_error_is_not_replaceable(self):
        summary = {
            "status": "terminated_without_success",
            "termination_reason": "environment_done",
            "success": False,
        }
        result = classify_collection_attempt(
            summary,
            {"passed": False, "checks": {"actions_finite": False}},
        )
        self.assertEqual(result["kind"], "infrastructure_failure")


if __name__ == "__main__":
    unittest.main()
