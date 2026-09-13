from __future__ import annotations

import gzip
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.navila_n1.candidate_manifest import build_geometry_manifest, validate_geometry_manifest
from src.navila_n0.geometry import select_geometry_precandidates


class TestCandidateManifest(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        dataset = root / "official.json.gz"
        payload = {
            "episodes": [
                {"episode_id": 1, "scene_id": "mp3d/scene-a/scene-a.glb", "gt_locations": [[0, 0, 0], [3, 0, 0], [3, 3, 0]]},
                {"episode_id": 2, "scene_id": "mp3d/scene-b/scene-b.glb", "gt_locations": [[0, 0, 0], [0, 1, 0]]},
            ]
        }
        with gzip.open(dataset, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream)
        return dataset

    @patch("src.navila_n1.candidate_manifest._navila_commit", return_value="navila-test-commit")
    def test_builds_fail_closed_geometry_manifest(self, _commit: object) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_geometry_manifest(self._write_dataset(root), root, max_candidates_per_parent=3)
        self.assertEqual(validate_geometry_manifest(manifest), [])
        self.assertTrue(manifest["candidates"])
        self.assertEqual(manifest["parents_without_geometry_precandidate"], ["2"])
        self.assertTrue(all(item["acceptance_status"] == "pending_surface_and_semantic_preview" for item in manifest["candidates"]))
        self.assertTrue(all("instruction" not in item for item in manifest["candidates"]))

    @patch("src.navila_n1.candidate_manifest._navila_commit", return_value="navila-test-commit")
    def test_rejects_prematurely_accepted_or_instructed_record(self, _commit: object) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_geometry_manifest(self._write_dataset(root), root, max_candidates_per_parent=1)
        manifest["candidates"][0]["acceptance_status"] = "accepted"
        manifest["candidates"][0]["instruction"] = "manual"
        errors = validate_geometry_manifest(manifest)
        self.assertTrue(any("fail-closed" in error for error in errors))
        self.assertTrue(any("instruction" in error for error in errors))

    def test_streaming_selector_is_bounded_and_keeps_auditable_paths(self) -> None:
        dense_path = [[float(index) / 10.0, 0.0, 0.0] for index in range(81)]
        records = select_geometry_precandidates(
            "parent-1", dense_path, maximum_candidates=2, selection_seed=17
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(all(1.5 <= record["path_length_m"] <= 4.0 for record in records))
        self.assertTrue(all(record["parent_dense_path"] == dense_path for record in records))
