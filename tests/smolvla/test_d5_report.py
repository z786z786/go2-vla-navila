import unittest

from src.smolvla.d5_report import collection_summary


class D5ReportCollectionTests(unittest.TestCase):
    def test_reports_accepted_and_rejected_collection_evidence(self):
        manifest = {
            "status": "complete",
            "accepted_ids": {"train": [f"train_{i}" for i in range(30)], "seen-val": [f"val_{i}" for i in range(12)]},
            "rejected_ids": {"train": ["short_vln_v1_0072"], "seen-val": []},
            "attempts": [
                {"decision": {"kind": "planner_reject"}},
                {"decision": {"kind": "expert_rollout_reject"}},
                {"decision": {"kind": "expert_reset_reject"}},
            ],
            "accepted_summary": {
                "splits": {
                    split: {
                        "category_minimum_met": True,
                        "coverage_met": True,
                        "category_counts": {"straight": 10 if split == "train" else 4, "left_turn": 10 if split == "train" else 4, "right_turn": 10 if split == "train" else 4},
                        "scene_count": 9 if split == "train" else 7,
                        "heading_bin_count": 8 if split == "train" else 6,
                        "distance_bin_counts": {},
                        "supplemental_count": 0,
                    }
                    for split in ("train", "seen-val")
                }
            },
        }
        summary = collection_summary(manifest)
        self.assertEqual(summary["accepted_episode_count"], 42)
        self.assertEqual(summary["accepted_by_split"], {"train": 30, "seen-val": 12})
        self.assertEqual(summary["planner_reject_count"], 1)
        self.assertEqual(summary["expert_rollout_reject_count"], 1)
        self.assertEqual(summary["expert_reset_reject_count"], 1)

    def test_allows_bounded_coverage_supplements(self):
        manifest = {
            "status": "complete",
            "accepted_ids": {"train": [f"train_{i}" for i in range(31)], "seen-val": [f"val_{i}" for i in range(13)]},
            "rejected_ids": {"train": [], "seen-val": []}, "attempts": [],
            "accepted_summary": {"splits": {
                "train": {"category_minimum_met": True, "coverage_met": True, "category_counts": {}, "scene_count": 9, "heading_bin_count": 8, "distance_bin_counts": {}, "supplemental_count": 1},
                "seen-val": {"category_minimum_met": True, "coverage_met": True, "category_counts": {}, "scene_count": 7, "heading_bin_count": 6, "distance_bin_counts": {}, "supplemental_count": 1},
            }},
        }
        self.assertEqual(collection_summary(manifest)["accepted_episode_count"], 44)

    def test_rejects_incomplete_collection_manifest(self):
        with self.assertRaises(ValueError):
            collection_summary({"status": "collecting", "accepted_ids": {}, "rejected_ids": {}, "attempts": []})


if __name__ == "__main__":
    unittest.main()
