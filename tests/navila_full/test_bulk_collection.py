from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.navila_full.bulk_collection import plan_bulk_queue, validate_bulk_queue
from src.navila_full.finalize_selection import canary_manifest
from src.navila_full.selection import DEFAULT_UNSEEN_SCENE, attach_spatial_clusters, candidate_routes
from tests.navila_full.test_contracts_and_selection import _episode


class BulkCollectionTests(unittest.TestCase):
    def test_balanced_queue_excludes_canary_and_all_declared_pd_rejects(self) -> None:
        episodes = []
        index = 0
        for scene, count in (("seen_house", 18), (DEFAULT_UNSEEN_SCENE, 6)):
            for category in ("straight", "left_turn", "right_turn"):
                for _ in range(count):
                    episodes.append(_episode(index, category, scene))
                    index += 1
        routes = attach_spatial_clusters(candidate_routes(episodes))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidate.json"
            candidate_path.write_text(json.dumps({
                "format": "navila-full-episode-selection-v1",
                "dataset_schema_version": "navila_full_episode_go2_v1",
                "source_provenance": {"official_dataset_sha256": "a" * 64},
                "routes": routes,
            }), encoding="utf-8")
            canary = canary_manifest(candidate_path, seed=17)
            rejected = next(route["route_id"] for route in routes if route["route_id"] not in {row["route_id"] for row in canary["routes"]})
            canary["rejected_pd_route_ids"] = [rejected]
            canary_path = root / "canary.json"
            canary_path.write_text(json.dumps(canary), encoding="utf-8")
            queue = plan_bulk_queue(candidate_path, canary_path)
        self.assertEqual(validate_bulk_queue(queue), [])
        queued = {row["route_id"] for row in queue["routes"]}
        excluded = {row["route_id"] for row in canary["routes"]} | {rejected}
        self.assertFalse(queued & excluded)
        first_six = {(row["collection_scope"], row["category"]) for row in queue["routes"][:6]}
        self.assertEqual(len(first_six), 6)

