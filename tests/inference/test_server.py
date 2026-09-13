import unittest
from pathlib import Path

from src.inference.server import (
    build_error_response,
    build_success_response,
    derive_request_seed,
    is_m61_checkpoint,
)


class ServerHelperTests(unittest.TestCase):
    def test_request_seed_is_stable_and_changes_per_replan(self):
        self.assertEqual(derive_request_seed(20260831, "short_vln_v1_0004", 0), 2012673106)
        self.assertEqual(derive_request_seed(20260831, "short_vln_v1_0004", 1), 50078916)

    def test_success_response_contains_auditable_model_result(self):
        response = build_success_response(
            request_id="short_vln_v1_0004:0",
            actions=[[0.1, 0.0, 0.2] for _ in range(50)],
            checkpoint_sha256="b" * 64,
            timings_ms={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0, "total": 6.0},
            seed=42,
        )

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["request_id"], "short_vln_v1_0004:0")
        self.assertEqual(len(response["actions"]), 50)

    def test_error_response_never_adds_policy_or_oracle_fields(self):
        response = build_error_response("request-7", ValueError("bad request"))

        self.assertEqual(
            set(response),
            {"version", "request_id", "status", "error_type", "error"},
        )
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["error_type"], "ValueError")

    def test_m61_checkpoint_path_requires_the_bounded_action_codec(self):
        self.assertTrue(is_m61_checkpoint(Path("/mnt/wxh/go2_short_vln/outputs/m6_1/run/checkpoints/step_002000")))
        self.assertFalse(is_m61_checkpoint(Path("/mnt/wxh/go2_short_vln/outputs/m6/run/checkpoints/step_002000")))


if __name__ == "__main__":
    unittest.main()
