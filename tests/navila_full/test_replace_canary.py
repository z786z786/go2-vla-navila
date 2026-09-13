from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.navila_full.finalize_selection import canary_manifest
from src.navila_full.replace_canary import replace_failed_canary
from src.navila_full.selection import DEFAULT_UNSEEN_SCENE, attach_spatial_clusters, candidate_routes
from tests.navila_full.test_contracts_and_selection import _episode


class ReplaceCanaryTests(unittest.TestCase):
    def test_replaces_failed_seen_route_and_preserves_final_capacity(self) -> None:
        episodes = []
        index = 0
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(17):
                episodes.append(_episode(index, category, "seen_house"))
                index += 1
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(4):
                episodes.append(_episode(index, category, DEFAULT_UNSEEN_SCENE))
                index += 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidate.json"
            candidate = {
                "format": "navila-full-episode-selection-v1", "dataset_schema_version": "navila_full_episode_go2_v1",
                "source_provenance": {"official_dataset_sha256": "a" * 64}, "routes": attach_spatial_clusters(candidate_routes(episodes)),
            }
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            previous_path = root / "canary.json"
            previous_path.write_text(json.dumps(canary_manifest(candidate_path)), encoding="utf-8")
            previous = json.loads(previous_path.read_text())
            failed = previous["routes"][2]["route_id"]
            replacement = replace_failed_canary(candidate_path, previous_path, failed)
        self.assertEqual(len(replacement["routes"]), 6)
        self.assertNotIn(failed, [row["route_id"] for row in replacement["routes"]])
        self.assertEqual(replacement["rejected_pd_route_ids"], [failed])
