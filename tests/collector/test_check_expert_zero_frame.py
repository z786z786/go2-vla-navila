import json
from pathlib import Path
import tempfile
import unittest

try:
    from src.collector.check_expert import check_episode
except ModuleNotFoundError as error:
    if error.name != "cv2":
        raise
    check_episode = None


@unittest.skipIf(check_episode is None, "OpenCV is available in the Isaac test environment only")
class ZeroFrameExpertSanityTests(unittest.TestCase):
    def test_writes_a_sanity_record_without_media_for_verified_pre_action_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = Path(temporary) / "short_vln_v1_1050"
            (episode_dir / "front_rgb").mkdir(parents=True)
            planner_hash = "20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432"
            event = {
                "event": "official_large_orientation",
                "stage": "before_first_action",
                "record_count_at_event": 0,
                "roll_rad": -0.3257,
                "pitch_rad": -0.6511,
                "official_threshold_rad": 0.6,
                "strict_threshold_exceeded": True,
            }
            (episode_dir / "terminal_event.json").write_text(json.dumps(event), encoding="utf-8")
            (episode_dir / "collector_config.json").write_text(
                json.dumps({"official_planner_sha256": planner_hash}), encoding="utf-8"
            )
            (episode_dir / "steps.jsonl").write_text("", encoding="utf-8")
            (episode_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "short_episode_id": "short_vln_v1_1050",
                        "status": "terminated_without_success",
                        "termination_reason": "official_large_orientation_before_first_action",
                        "success": False,
                        "record_count": 0,
                        "frame_count": 0,
                        "official_planner_sha256": planner_hash,
                    }
                ),
                encoding="utf-8",
            )

            result = check_episode(episode_dir)

            self.assertFalse(result["passed"])
            self.assertTrue(result["checks"]["zero_frame_official_orientation_evidence"])
            self.assertEqual(result["video"], None)
            self.assertTrue((episode_dir / "sanity.json").is_file())

    def test_rejects_malformed_zero_frame_evidence_without_crashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = Path(temporary) / "short_vln_v1_1050"
            (episode_dir / "front_rgb").mkdir(parents=True)
            (episode_dir / "steps.jsonl").write_text("", encoding="utf-8")
            (episode_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "short_episode_id": "short_vln_v1_1050",
                        "status": "terminated_without_success",
                        "termination_reason": "official_large_orientation_before_first_action",
                        "success": False,
                        "record_count": 0,
                        "frame_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = check_episode(episode_dir)

            self.assertFalse(result["checks"]["zero_frame_official_orientation_evidence"])


if __name__ == "__main__":
    unittest.main()
