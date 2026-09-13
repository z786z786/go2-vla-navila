import unittest

from src.inference.d3_closed_loop import (
    audit_request_seeds,
    classify_d3,
    oracle_navigation_metrics,
    policy_navigation_metrics,
)
from src.inference.server import derive_request_seed


def policy_record(index, *, distance, next_distance, vx=0.1, wz=0.2):
    return {
        "frame_index": index,
        "timestamp": index * 0.02,
        "robot_pose": {"position_w": [float(index), 0.0, 0.0]},
        "next_robot_pose": {"position_w": [float(index + 1), 0.0, 0.0]},
        "evaluator_distance_to_goal_m": distance,
        "next_evaluator_distance_to_goal_m": next_distance,
        "applied_action": [vx, 0.0, wz],
    }


def oracle_record(index, *, distance, next_distance, vx=0.1, wz=0.2):
    record = policy_record(index, distance=distance, next_distance=next_distance, vx=vx, wz=wz)
    record.pop("applied_action")
    record.update({"expert_vx": vx, "expert_vy": 0.0, "expert_wz": wz})
    record["distance_to_goal_xy_m"] = record.pop("evaluator_distance_to_goal_m")
    record["next_distance_to_goal_xy_m"] = record.pop("next_evaluator_distance_to_goal_m")
    return record


class D3ClosedLoopTests(unittest.TestCase):
    def test_policy_metrics_use_xy_path_minimum_distance_and_success_time(self):
        records = [
            policy_record(0, distance=2.0, next_distance=1.0, vx=0.2, wz=-0.1),
            policy_record(1, distance=1.0, next_distance=0.4, vx=0.4, wz=0.3),
        ]
        summary = {
            "success": True,
            "termination_reason": "evaluator_goal_stop",
            "collision": False,
            "final_navigation_error_m": 0.4,
            "mean_inference_roundtrip_ms": 9.0,
            "p95_inference_roundtrip_ms": 10.0,
        }

        metrics = policy_navigation_metrics(summary, records)

        self.assertEqual(metrics["minimum_distance_m"], 0.4)
        self.assertEqual(metrics["time_to_goal_s"], 0.04)
        self.assertEqual(metrics["path_length_m"], 2.0)
        self.assertAlmostEqual(metrics["mean_vx"], 0.3)
        self.assertAlmostEqual(metrics["mean_abs_wz"], 0.2)
        self.assertAlmostEqual(metrics["cumulative_vx_m"], 0.012)

    def test_failed_policy_has_null_time_to_goal(self):
        summary = {
            "success": False,
            "termination_reason": "client_max_steps",
            "collision": False,
            "final_navigation_error_m": 1.0,
            "mean_inference_roundtrip_ms": 9.0,
            "p95_inference_roundtrip_ms": 10.0,
        }
        metrics = policy_navigation_metrics(summary, [policy_record(0, distance=1.1, next_distance=1.0)])
        self.assertIsNone(metrics["time_to_goal_s"])

    def test_oracle_metrics_share_control_definition(self):
        summary = {"success": True, "termination_reason": "goal_reached", "final_distance_to_goal_xy_m": 0.4}
        metrics = oracle_navigation_metrics(summary, [oracle_record(0, distance=1.0, next_distance=0.4, vx=0.2, wz=-0.3)])
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["final_distance_m"], 0.4)
        self.assertEqual(metrics["mean_abs_wz"], 0.3)
        self.assertIsNone(metrics["mean_inference_latency_ms"])

    def test_seed_audit_rejects_one_mismatched_response(self):
        base_seed = 20260831
        request = {"episode_id": "short_vln_v1_0000", "replan_index": 2, "request_id": "short_vln_v1_0000:2"}
        good = {"request": request, "response": {"seed": derive_request_seed(base_seed, "short_vln_v1_0000", 2)}}
        self.assertTrue(audit_request_seeds([good], base_seed=base_seed)["passed"])
        bad = {"request": request, "response": {"seed": 0}}
        self.assertFalse(audit_request_seeds([bad], base_seed=base_seed)["passed"])

    def test_classification_is_a_when_one_train_route_is_not_three_of_three(self):
        runs = [
            {"episode_id": episode, "official_navigation_success": episode == "short_vln_v1_0000"}
            for episode in ("short_vln_v1_0000", "short_vln_v1_0004")
            for _ in range(3)
        ]
        self.assertEqual(classify_d3(runs, evidence_passed=True)["classification"], "A")

    def test_classification_is_b_only_when_both_routes_are_three_of_three(self):
        runs = [
            {"episode_id": episode, "official_navigation_success": True}
            for episode in ("short_vln_v1_0000", "short_vln_v1_0004")
            for _ in range(3)
        ]
        self.assertEqual(classify_d3(runs, evidence_passed=True)["classification"], "B")

    def test_infrastructure_failure_keeps_classification_unresolved(self):
        self.assertEqual(classify_d3([], evidence_passed=False)["classification"], "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
