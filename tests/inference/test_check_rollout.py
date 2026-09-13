import unittest

from src.inference.action_audit import summarize_action_range
from src.inference.check_rollout import (
    audit_policy_requests,
    audit_provenance,
    evaluate_rollout_records,
    render_markdown,
    validate_gate_stage,
)
from src.inference.protocol import PROTOCOL_VERSION
from src.inference.state import apply_action_safety


def valid_request_row() -> dict:
    return {
        "status": "ok",
        "roundtrip_wall_ms": 900.0,
        "raw_range_violation_count": 0,
        "request": {
            "version": PROTOCOL_VERSION,
            "request_id": "short_vln_v1_0004:0",
            "episode_id": "short_vln_v1_0004",
            "replan_index": 0,
            "instruction": "Move forward and stop.",
            "state": [0.0] * 30,
            "image_encoding": "jpeg",
            "image_height": 512,
            "image_width": 512,
            "image_channels": 3,
            "jpeg_size": 1234,
        },
        "response": {
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
        },
    }


def valid_summary(requests: list[dict], *, frame_count: int = 1) -> dict:
    range_summary = summarize_action_range(
        [request["response"]["actions"] for request in requests], execute_steps=10
    )
    return {
        "frame_count": frame_count,
        "request_count": len(requests),
        "execute_steps_per_chunk": 10,
        "success": True,
        "status": "complete",
        **range_summary.as_dict(),
        "raw_chunk_range_violation_count": range_summary.violating_vectors,
        "executed_action_range_violation_count": range_summary.executed_violating_vectors,
    }


def valid_policy_record(request: dict, *, action_offset: int = 0) -> dict:
    decision = apply_action_safety(request["response"]["actions"][action_offset])
    return {
        "frame_index": 0,
        "timestamp": 0.0,
        "action_source": "smolvla",
        "request_id": request["request"]["request_id"],
        "action_offset": action_offset,
        "policy_state": [0.0] * 30,
        "raw_action": list(decision.raw),
        "applied_action": list(decision.applied),
        "action_in_range": decision.in_range,
        "clipped_dimensions": list(decision.clipped_dimensions),
        "front_rgb": "front_rgb/000000.jpg",
        "termination": True,
    }


class CheckRolloutTests(unittest.TestCase):
    def test_request_audit_accepts_only_exact_policy_allowlist(self):
        checks = audit_policy_requests([valid_request_row()])

        self.assertTrue(checks["request_schema_valid"])
        self.assertTrue(checks["no_oracle_input_leakage"])
        self.assertTrue(checks["responses_valid"])

    def test_request_audit_detects_oracle_input_leakage(self):
        row = valid_request_row()
        row["request"]["next_waypoint"] = [1.0, 2.0, 3.0]

        checks = audit_policy_requests([row])

        self.assertFalse(checks["request_schema_valid"])
        self.assertFalse(checks["no_oracle_input_leakage"])

    def test_rollout_record_audit_separates_infrastructure_from_policy_result(self):
        request = valid_request_row()
        request["response"]["actions"][0] = [-0.01, 0.0, 0.0]
        summary = valid_summary([request])
        records = [valid_policy_record(request)]

        result = evaluate_rollout_records(summary, records, [request])

        self.assertTrue(result["infrastructure_passed"])
        self.assertFalse(result["valid_autonomous_success"])
        self.assertEqual(result["raw_range_violation_count"], 1)

    def test_rollout_audit_reports_vectors_chunks_and_execute_window_separately(self):
        request = valid_request_row()
        request["response"]["actions"][0] = [-0.01, 0.0, -0.6]
        request["response"]["actions"][10] = [0.1, 0.0, -0.6]
        request["raw_range_violation_count"] = 999  # Historical client field is not a count source.
        summary = valid_summary([request])
        records = [valid_policy_record(request)]

        result = evaluate_rollout_records(summary, records, [request])

        self.assertEqual(result["raw_action_vector_violation_count"], 2)
        self.assertEqual(result["raw_chunk_with_violation_count"], 1)
        self.assertEqual(result["executed_policy_action_violation_count"], 1)
        self.assertEqual(result["discarded_action_violation_count"], 1)
        self.assertTrue(result["official_navigation_success"])
        self.assertFalse(result["range_compliant_autonomous_success"])

    def test_rollout_audit_rejects_corrupted_summary_range_flag(self):
        request = valid_request_row()
        request["response"]["actions"][0] = [-0.01, 0.0, 0.0]
        summary = valid_summary([request])
        summary["raw_output_range_passed"] = True

        result = evaluate_rollout_records(summary, [valid_policy_record(request)], [request])

        self.assertFalse(result["checks"]["summary_range_counts_match"])
        self.assertFalse(result["infrastructure_passed"])

    def test_rollout_audit_rejects_step_that_does_not_match_response_offset(self):
        request = valid_request_row()
        record = valid_policy_record(request)
        record["raw_action"] = [0.2, 0.0, 0.0]

        result = evaluate_rollout_records(valid_summary([request]), [record], [request])

        self.assertFalse(result["checks"]["executed_actions_match_response"])
        self.assertFalse(result["infrastructure_passed"])

    def test_full_gate_requires_all_five_specified_episode_ids(self):
        checks = validate_gate_stage(["short_vln_v1_0004"], gate_stage="full")

        self.assertFalse(checks["episode_ids_exact"])
        self.assertEqual(checks["expected_episode_ids"], [
            "short_vln_v1_0000",
            "short_vln_v1_0001",
            "short_vln_v1_0003",
            "short_vln_v1_0004",
            "short_vln_v1_0006",
        ])

    def test_provenance_audit_rejects_altered_checkpoint_or_oracle_hash(self):
        request = valid_request_row()
        oracle = {
            "available": True,
            "planner_sha256": "planner-good",
            "short_dataset_sha256": "dataset-good",
        }

        good = audit_provenance(
            [request],
            oracle,
            expected_checkpoint_sha256="a" * 64,
            expected_planner_sha256="planner-good",
            expected_short_dataset_sha256="dataset-good",
        )
        request["response"]["checkpoint_sha256"] = "b" * 64
        oracle["planner_sha256"] = "planner-bad"
        altered = audit_provenance(
            [request],
            oracle,
            expected_checkpoint_sha256="a" * 64,
            expected_planner_sha256="planner-good",
            expected_short_dataset_sha256="dataset-good",
        )

        self.assertTrue(all(good.values()))
        self.assertFalse(altered["checkpoint_hash_matches"])
        self.assertFalse(altered["planner_hash_matches"])

    def test_markdown_separates_official_and_range_compliant_success(self):
        report = {
            "passed": False,
            "episodes": [
                {
                    "episode_id": "short_vln_v1_0004",
                    "infrastructure_passed": True,
                    "official_navigation_success": True,
                    "range_compliant_autonomous_success": False,
                    "raw_output_range_passed": False,
                    "raw_action_vector_violation_count": 2,
                    "raw_chunk_with_violation_count": 1,
                    "artifacts_complete": True,
                    "summary": {
                        "raw_output_range_passed": False,
                        "final_navigation_error_m": 0.2,
                        "mean_inference_roundtrip_ms": 100.0,
                    },
                }
            ],
        }

        markdown = render_markdown(report)

        self.assertIn("Official success", markdown)
        self.assertIn("Range-compliant success", markdown)
        self.assertIn("2 / 1", markdown)

    def test_markdown_uses_checker_range_result_not_historical_summary_flag(self):
        report = {
            "passed": False,
            "episodes": [
                {
                    "episode_id": "short_vln_v1_0004",
                    "infrastructure_passed": True,
                    "official_navigation_success": True,
                    "range_compliant_autonomous_success": False,
                    "raw_output_range_passed": False,
                    "raw_action_vector_violation_count": 1,
                    "raw_chunk_with_violation_count": 1,
                    "artifacts_complete": True,
                    "summary": {
                        "raw_output_range_passed": True,
                        "final_navigation_error_m": 0.2,
                        "mean_inference_roundtrip_ms": 100.0,
                    },
                }
            ],
        }

        markdown = render_markdown(report)

        self.assertIn("| short_vln_v1_0004 | PASS | PASS | FAIL | FAIL | 1 / 1 |", markdown)


if __name__ == "__main__":
    unittest.main()
