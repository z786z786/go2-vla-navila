from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.navila_n1.preview_adapter import (
    POLICY_RGB_PRIM,
    build_preview_camera_spec,
    isaac_camera_construction_contract,
    validate_preview_camera_spec,
)


class TestPreviewAdapter(unittest.TestCase):
    def test_missing_mesh_fails_closed_without_policy_leakage(self) -> None:
        spec = build_preview_camera_spec("scene-a", Path("/missing/scene.ply"))
        errors = validate_preview_camera_spec(spec)
        self.assertTrue(any("unavailable" in error for error in errors))
        contract = isaac_camera_construction_contract(spec)
        self.assertEqual(contract["prim_path"], POLICY_RGB_PRIM)
        self.assertEqual(contract["data_types"], ["semantic_segmentation"])
        self.assertEqual(contract["policy_feature_names"], [])

    def test_existing_mesh_produces_isolated_512_semantic_spec(self) -> None:
        with TemporaryDirectory() as temporary:
            mesh = Path(temporary) / "scene.ply"
            mesh.touch()
            spec = build_preview_camera_spec("scene-a", mesh)
            self.assertEqual(validate_preview_camera_spec(spec), [])
