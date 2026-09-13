import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.inference.d5_resume import audit_completed_episode


class D5ResumeTests(unittest.TestCase):
    def make_episode(self, *, success=False, frames=1500, reason="client_max_steps"):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "front_rgb").mkdir()
        records = []
        for index in range(frames):
            rel = f"front_rgb/{index:06d}.jpg"
            (root / rel).touch()
            records.append({"front_rgb": rel})
        summary = {
            "status": "complete" if success else "terminated",
            "error": None,
            "short_episode_id": "short_vln_v1_0000",
            "frame_count": frames,
            "success": success,
            "termination_reason": reason,
        }
        (root / "summary.json").write_text(json.dumps(summary))
        (root / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
        (root / "requests.jsonl").write_text("{}\n")
        return temp, root

    @patch("src.inference.d5_resume.evaluate_rollout_records", return_value={"infrastructure_passed": True, "raw_output_range_passed": True})
    def test_accepts_complete_max_step_failure(self, _audit):
        temp, root = self.make_episode()
        self.addCleanup(temp.cleanup)
        result = audit_completed_episode(root, expected_episode_id="short_vln_v1_0000")
        self.assertTrue(result["passed"])

    @patch("src.inference.d5_resume.evaluate_rollout_records", return_value={"infrastructure_passed": True, "raw_output_range_passed": True})
    def test_accepts_normal_success(self, _audit):
        temp, root = self.make_episode(success=True, frames=10, reason="evaluator_goal_stop")
        self.addCleanup(temp.cleanup)
        result = audit_completed_episode(root, expected_episode_id="short_vln_v1_0000")
        self.assertTrue(result["passed"])

    @patch("src.inference.d5_resume.evaluate_rollout_records", return_value={"infrastructure_passed": True, "raw_output_range_passed": True})
    def test_rejects_partial_failed_episode(self, _audit):
        temp, root = self.make_episode(frames=12)
        self.addCleanup(temp.cleanup)
        result = audit_completed_episode(root, expected_episode_id="short_vln_v1_0000")
        self.assertFalse(result["passed"])
        self.assertIn("failed episode did not complete the max-step horizon", result["errors"])

    @patch("src.inference.d5_resume.evaluate_rollout_records", return_value={"infrastructure_passed": True, "raw_output_range_passed": False})
    def test_accepts_complete_diagnostic_with_raw_range_failure(self, _audit):
        temp, root = self.make_episode(success=True, frames=10, reason="evaluator_goal_stop")
        self.addCleanup(temp.cleanup)
        result = audit_completed_episode(root, expected_episode_id="short_vln_v1_0000")
        self.assertTrue(result["passed"])
        self.assertFalse(result["raw_output_range_passed"])

    @patch("src.inference.d5_resume.evaluate_rollout_records", return_value={"infrastructure_passed": True, "raw_output_range_passed": True})
    def test_rejects_success_with_terminated_status(self, _audit):
        temp, root = self.make_episode(success=True, frames=10, reason="evaluator_goal_stop")
        self.addCleanup(temp.cleanup)
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["status"] = "terminated"
        summary_path.write_text(json.dumps(summary))
        result = audit_completed_episode(root, expected_episode_id="short_vln_v1_0000")
        self.assertFalse(result["passed"])
        self.assertIn("successful episode status is not complete", result["errors"])

    @patch("src.inference.d5_resume.evaluate_rollout_records", return_value={"infrastructure_passed": True, "raw_output_range_passed": True})
    def test_rejects_missing_referenced_frame(self, _audit):
        temp, root = self.make_episode()
        self.addCleanup(temp.cleanup)
        (root / "front_rgb/000010.jpg").unlink()
        result = audit_completed_episode(root, expected_episode_id="short_vln_v1_0000")
        self.assertFalse(result["passed"])
        self.assertIn("missing referenced RGB frames: 1", result["errors"])


if __name__ == "__main__":
    unittest.main()
