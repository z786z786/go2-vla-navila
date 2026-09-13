import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.inference.preflight import build_preflight_request, evaluate_preflight_response, main
from src.inference.protocol import PROTOCOL_VERSION, REQUEST_FIELDS


class PreflightTests(unittest.TestCase):
    @staticmethod
    def valid_response() -> dict:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": "short_vln_v1_0004:0",
            "status": "ok",
            "actions": [[0.1, 0.0, 0.0] for _ in range(50)],
            "checkpoint_sha256": "a" * 64,
            "timings_ms": {
                "preprocess": 1.0,
                "inference": 2.0,
                "postprocess": 1.0,
                "total": 4.0,
            },
            "seed": 1,
        }

    def test_preflight_request_contains_only_allowlisted_policy_inputs_and_audit_ids(self):
        raw_state = [0.0] * 31
        record = {
            "robot_state": raw_state,
            "current_velocity": {
                "linear_body": [0.1, 0.2, 0.3],
                "angular_body": [0.4, 0.5, 0.6],
            },
            "robot_pose": {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "joint_velocity": [0.0] * 12,
        }

        header = build_preflight_request(
            record=record,
            jpeg=b"four",
            episode_id="short_vln_v1_0004",
            instruction="Move forward and stop.",
            replan_index=0,
        )

        self.assertEqual(set(header), set(REQUEST_FIELDS))
        self.assertEqual(header["version"], PROTOCOL_VERSION)
        self.assertEqual(header["request_id"], "short_vln_v1_0004:0")
        self.assertEqual(len(header["state"]), 30)
        self.assertEqual(header["jpeg_size"], 4)
        for forbidden in ("goal_pose", "goal_direction", "reference_path", "next_waypoint"):
            self.assertNotIn(forbidden, header)

    def test_preflight_marks_one_out_of_range_raw_action_as_failed(self):
        response = self.valid_response()
        response["actions"][17] = [-0.01, 0.0, 0.0]

        result = evaluate_preflight_response(response, roundtrip_ms=100.0)

        self.assertFalse(result["passed"])
        self.assertFalse(result["raw_output_range_passed"])
        self.assertEqual(result["raw_action_vector_violation_count"], 1)
        self.assertEqual(result["raw_chunk_with_violation_count"], 1)
        self.assertTrue(result["checks"]["response_valid"])
        self.assertTrue(result["checks"]["latency_within_timeout"])

    def test_preflight_cli_exits_nonzero_when_raw_action_is_out_of_range(self):
        record = {
            "frame_index": 0,
            "front_rgb": "front_rgb/000000.jpg",
            "robot_state": [0.0] * 31,
            "current_velocity": {
                "linear_body": [0.1, 0.2, 0.3],
                "angular_body": [0.4, 0.5, 0.6],
            },
            "robot_pose": {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "joint_velocity": [0.0] * 12,
        }
        response = self.valid_response()
        response["actions"][0] = [-0.01, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory) / "episode"
            (episode_dir / "front_rgb").mkdir(parents=True)
            (episode_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "short_episode_id": "short_vln_v1_0004",
                        "instruction": "Move forward and stop.",
                    }
                )
            )
            (episode_dir / "steps.jsonl").write_text(json.dumps(record) + "\n")
            (episode_dir / "front_rgb" / "000000.jpg").write_bytes(b"jpeg")
            output = Path(directory) / "preflight.json"
            argv = [
                "preflight.py",
                "--episode-dir",
                str(episode_dir),
                "--output",
                str(output),
            ]
            with patch("src.inference.preflight.unix_request", return_value=response), patch.object(
                sys, "argv", argv
            ), self.assertRaises(SystemExit) as exit_code:
                main()

            self.assertEqual(exit_code.exception.code, 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
