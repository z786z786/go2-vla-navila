from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.collector.full_episode_canary import evaluate


class FullEpisodeCanaryTests(unittest.TestCase):
    def test_missing_collections_fail_closed(self) -> None:
        manifest = {"format": "navila-full-episode-pd-canary-v1", "dataset_schema_version": "navila_full_episode_go2_v1", "routes": [{"route_id": str(index), "category": category, "scene_name": "seen"} for index, category in enumerate(["straight", "straight", "left_turn", "left_turn", "right_turn", "right_turn"])]}
        with tempfile.TemporaryDirectory() as temporary:
            report = evaluate(manifest, Path(temporary))
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["rows"]), 6)
