from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.navila_full.replan_canary import plan_evidence_aware_canary
from src.navila_full.selection import DEFAULT_UNSEEN_SCENE, attach_spatial_clusters, candidate_routes
from tests.navila_full.test_contracts_and_selection import _episode


class ReplanCanaryTests(unittest.TestCase):
    def test_pins_successes_and_prefers_shortest_remaining_routes(self) -> None:
        episodes = []
        index = 0
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(18):
                episodes.append(_episode(index, category, "seen_house"))
                index += 1
        for category in ("straight", "left_turn", "right_turn"):
            for _ in range(4):
                episodes.append(_episode(index, category, DEFAULT_UNSEEN_SCENE))
                index += 1
        routes = attach_spatial_clusters(candidate_routes(episodes))
        with tempfile.TemporaryDirectory() as temporary:
            candidate_path = Path(temporary) / "candidate.json"
            candidate_path.write_text(json.dumps({"format": "navila-full-episode-selection-v1", "dataset_schema_version": "navila_full_episode_go2_v1", "source_provenance": {"official_dataset_sha256": "a" * 64}, "routes": routes}), encoding="utf-8")
            proven = {
                next(route["route_id"] for route in routes if route["category"] == category and route["scene_name"] != DEFAULT_UNSEEN_SCENE)
                for category in ("straight", "left_turn", "right_turn")
            }
            manifest = plan_evidence_aware_canary(candidate_path, proven_success_route_ids=proven, rejected_pd_route_ids=set())
        self.assertEqual({route["route_id"] for route in manifest["routes"]} & proven, proven)
        self.assertEqual(len(manifest["routes"]), 6)
