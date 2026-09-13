import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.inference.isaac_client import (
    advance_success_streak,
    add_official_runner_compat_args,
    build_policy_request,
    evaluator_distance_from_measurements,
    finalize_summary,
    finalize_startup_failure,
    make_official_episode,
    select_executed_actions,
)
from src.inference.protocol import REQUEST_FIELDS


class IsaacClientCoreTests(unittest.TestCase):
    def test_official_runner_compatibility_defaults_match_demo_planner(self):
        parser = argparse.ArgumentParser()
        add_official_runner_compat_args(parser)

        defaults = parser.parse_args([])
        enabled = parser.parse_args(["--use_cnn", "--use_rnn"])

        self.assertIsNone(defaults.use_cnn)
        self.assertFalse(defaults.use_rnn)
        self.assertTrue(enabled.use_cnn)
        self.assertTrue(enabled.use_rnn)

    def test_success_distance_is_read_only_from_official_evaluator_measurements(self):
        self.assertEqual(
            evaluator_distance_from_measurements({"distance_to_goal": 0.49}),
            0.49,
        )
        with self.assertRaisesRegex(ValueError, "distance_to_goal"):
            evaluator_distance_from_measurements({"distance_to_goal": float("nan")})
        with self.assertRaisesRegex(ValueError, "distance_to_goal"):
            evaluator_distance_from_measurements({})

    def test_policy_request_has_exact_allowlist(self):
        header = build_policy_request(
            state=[0.0] * 30,
            jpeg=b"jpeg",
            episode_id="short_vln_v1_0004",
            instruction="Move forward and stop.",
            replan_index=2,
        )

        self.assertEqual(set(header), set(REQUEST_FIELDS))
        self.assertEqual(header["request_id"], "short_vln_v1_0004:2")
        self.assertNotIn("goal_pose", header)
        self.assertNotIn("reference_path", header)

    def test_only_first_ten_actions_are_selected_and_safed(self):
        chunk = [[0.1, 0.0, 0.2] for _ in range(50)]
        chunk[9] = [0.7, 0.1, -0.8]
        chunk[10] = [0.9, 0.2, 0.9]

        selected = select_executed_actions(chunk, execute_steps=10)

        self.assertEqual(len(selected), 10)
        self.assertEqual(selected[9].applied, (0.5, 0.0, -0.5))
        self.assertFalse(selected[9].in_range)

    def test_success_streak_requires_consecutive_in_radius_frames(self):
        streak = 0
        for distance in [0.49] * 9:
            streak = advance_success_streak(streak, distance, radius=0.5)
        self.assertEqual(streak, 9)
        self.assertEqual(advance_success_streak(streak, 0.51, radius=0.5), 0)
        self.assertEqual(advance_success_streak(9, 0.49, radius=0.5), 10)

    def test_official_episode_uses_short_route_for_environment_only(self):
        short = {
            "source_episode_id": 8,
            "source_trajectory_id": 18,
            "source_episode_new_id": "source-new",
            "scene_id": "mp3d/house/house.glb",
            "start_pose": {"position": [1, 2, 3], "rotation_wxyz": [1, 0, 0, 0]},
            "goal_pose": {"position": [4, 5, 6], "success_radius_m": 0.5},
            "instruction": "Move forward and stop.",
            "reference_path": [[1, 2, 3], [4, 5, 6]],
            "path_length": 3.0,
        }

        official = make_official_episode(short)

        self.assertEqual(official["episode_id"], 8)
        self.assertEqual(official["instruction"]["instruction_text"], short["instruction"])
        self.assertEqual(official["gt_locations"], short["reference_path"])
        self.assertEqual(official["goals"][0]["radius"], 0.5)

    def test_startup_failure_is_persisted_with_empty_audit_streams(self):
        short = {
            "short_episode_id": "short_vln_v1_0004",
            "source_episode_id": 9,
            "split": "train",
            "scene_id": "mp3d/house/house.glb",
            "instruction": "Move forward and stop.",
            "goal_pose": {"position": [4, 5, 6], "success_radius_m": 0.5},
        }
        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory)
            summary = finalize_startup_failure(
                episode_dir=episode_dir,
                short_episode=short,
                started_wall=1.0,
                error=RuntimeError("Kit stopped before wrapper reset"),
            )

            self.assertEqual(summary["status"], "initialization_error")
            self.assertEqual(summary["termination_reason"], "startup_error")
            self.assertFalse(summary["passed"])
            self.assertIn("RuntimeError", summary["error"])
            self.assertEqual((episode_dir / "steps.jsonl").read_text(), "")
            self.assertEqual((episode_dir / "requests.jsonl").read_text(), "")
            self.assertEqual(json.loads((episode_dir / "summary.json").read_text()), summary)

    def test_summary_uses_full_response_chunks_for_unambiguous_range_counts(self):
        first_chunk = [[0.1, 0.0, 0.0] for _ in range(50)]
        first_chunk[0] = [-0.01, 0.0, -0.6]
        first_chunk[10] = [0.1, 0.0, -0.51]
        second_chunk = [[0.1, 0.0, 0.0] for _ in range(50)]
        second_chunk[11] = [0.6, 0.0, 0.0]
        short = {
            "short_episode_id": "short_vln_v1_0004",
            "source_episode_id": 9,
            "split": "train",
            "scene_id": "mp3d/house/house.glb",
            "instruction": "Move forward and stop.",
            "goal_pose": {"position": [4, 5, 6], "success_radius_m": 0.5},
        }
        requests = [
            {
                "status": "ok",
                "roundtrip_wall_ms": 10.0,
                "raw_range_violation_count": 999,
                "response": {"actions": first_chunk},
            },
            {
                "status": "ok",
                "roundtrip_wall_ms": 20.0,
                "raw_range_violation_count": 999,
                "response": {"actions": second_chunk},
            },
        ]
        records = [
            {
                "action_in_range": False,
                "termination_terms": {"base_contact": False, "bad_orientation": False},
                "next_evaluator_distance_to_goal_m": 0.2,
                "measurements": {"success": 1.0},
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            summary = finalize_summary(
                episode_dir=Path(directory),
                short_episode=short,
                status="complete",
                termination_reason="evaluator_goal_stop",
                records=records,
                requests=requests,
                started_wall=0.0,
            )

        self.assertEqual(summary["raw_action_vector_count"], 100)
        self.assertEqual(summary["raw_action_vector_violation_count"], 3)
        self.assertEqual(summary["raw_chunk_with_violation_count"], 2)
        self.assertEqual(summary["executed_policy_action_violation_count"], 1)
        self.assertEqual(summary["discarded_action_violation_count"], 2)
        self.assertEqual(summary["raw_chunk_range_violation_count"], 3)
        self.assertFalse(summary["raw_output_range_passed"])
        self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
