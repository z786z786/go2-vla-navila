import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


# The local lightweight test environment intentionally lacks the Isaac/LeRobot
# runtime.  Discovery is pure file validation, so provide import-only stubs.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
lerobot = sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
datasets = sys.modules.setdefault("lerobot.datasets", types.ModuleType("lerobot.datasets"))
dataset_module = sys.modules.setdefault(
    "lerobot.datasets.lerobot_dataset", types.ModuleType("lerobot.datasets.lerobot_dataset")
)
setattr(dataset_module, "LeRobotDataset", object)
setattr(lerobot, "datasets", datasets)

from src.dataset.convert_to_lerobot import discover_episodes


class D5AssignmentConverterTests(unittest.TestCase):
    def test_discovery_routes_audited_seen_val_supplement_to_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "short_vln_v1_1051"
            episode.mkdir()
            (episode / "summary.json").write_text(json.dumps({
                "status": "complete", "success": True, "split": "seen-val",
                "short_episode_id": "short_vln_v1_1051",
            }), encoding="utf-8")
            (episode / "sanity.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            (episode / "steps.jsonl").write_text("{}\n", encoding="utf-8")
            (episode / "d5_assignment.json").write_text(json.dumps({
                "format": "go2-short-vln-m6_2-d5-assignment-sidecar-v1",
                "short_episode_id": "short_vln_v1_1051", "source_split": "seen-val",
                "collection_split": "train", "reason": "recover_required_train_scene_coverage",
                "exclude_from_seen_val_evaluation": True, "assignment_policy_sha256": "policy",
                "source_dataset_sha256": "dataset",
            }), encoding="utf-8")

            found = discover_episodes(Path(temporary))

            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["source_split"], "seen-val")
            self.assertEqual(found[0]["collection_split"], "train")
            self.assertTrue(found[0]["assignment"]["exclude_from_seen_val_evaluation"])


if __name__ == "__main__":
    unittest.main()
