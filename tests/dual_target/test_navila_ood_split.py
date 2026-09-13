import unittest

from src.dual_target.navila_ood_split import (
    DatasetDesignError,
    angle_partition,
    build_manifest,
    relative_heading_offset_degrees,
    validate_manifest,
    visibility_gate,
)


def candidate(scene, layout, offset, *, visible=1800, projected=2600):
    start = [layout * 2.0, 0.0, 0.0]
    box_a = [layout * 2.0 + 5.0, 0.50, 0.25]
    box_b = [layout * 2.0 + 5.0, -0.50, 0.25]
    midline = 0.0
    return {
        "candidate_id": f"{scene}_{layout}_{offset:+g}",
        "scene": scene,
        "start_position": start,
        "box_a_position": box_a,
        "box_b_position": box_b,
        "camera_yaw_degrees": midline + offset,
        "relative_heading_offset_degrees": offset,
        "visibility": {
            "image_width": 512,
            "image_height": 512,
            "A": {"visible_pixels": visible, "projected_pixels": projected, "projection_center_x_px": 210.0},
            "B": {"visible_pixels": visible, "projected_pixels": projected, "projection_center_x_px": 302.0},
        },
    }


class NavilaOodSplitTests(unittest.TestCase):
    def test_relative_heading_uses_box_midline(self):
        self.assertAlmostEqual(relative_heading_offset_degrees([0, 0], [5, 1], [5, -1], 7.0), 7.0)

    def test_angle_boundaries_and_guard(self):
        self.assertEqual([angle_partition(v) for v in (-20, -12, -11.99, -8.01, -8, 0, 8, 8.01, 11.99, 12, 20)],
                         ["ood", "ood", "guard", "guard", "seen", "seen", "seen", "guard", "guard", "ood", "ood"])

    def test_visibility_gate_requires_both_boxes(self):
        good = candidate("s", 0, 0)["visibility"]
        self.assertTrue(visibility_gate(good)["passed"])
        bad = {**good, "B": {**good["B"], "visible_pixels": 100}}
        result = visibility_gate(bad)
        self.assertFalse(result["passed"])
        self.assertTrue(any("slot B" in error for error in result["errors"]))

    def test_manifest_builds_orthogonal_cells_and_balanced_four_tasks(self):
        rows = []
        for scene_index, scene in enumerate(("scene_a", "scene_b", "scene_c", "scene_d")):
            for layout in (scene_index * 3, scene_index * 3 + 1):
                for offset in (-16, -8, -4, 0, 4, 8, 16):
                    rows.append(candidate(scene, layout, offset))
        manifest = build_manifest(rows, cluster_radius_m=0.5, ood_scene_fraction=0.25)
        self.assertEqual(validate_manifest(manifest), [])
        self.assertEqual(len(manifest["tasks"]), 4 * len(manifest["base_geometries"]))
        for split, stats in manifest["statistics"].items():
            if stats["tasks"]:
                self.assertTrue(all(value["balanced"] for value in stats["balance"].values()), split)
        bases = manifest["base_geometries"]
        self.assertTrue(all(row["geometry_partition"] == "seen" for row in bases if row["split"] == "orientation_ood"))
        self.assertTrue(all(row["angle_partition"] == "ood" for row in bases if row["split"] == "orientation_ood"))
        self.assertTrue(all(row["geometry_partition"] == "ood" for row in bases if row["split"] == "geometry_ood"))
        self.assertTrue(all(row["angle_partition"] == "seen" for row in bases if row["split"] == "geometry_ood"))

    def test_manifest_rejects_missing_second_seen_angle(self):
        rows = [candidate("scene_a", 0, 0), candidate("scene_a", 0, 16),
                candidate("scene_b", 2, 0), candidate("scene_b", 2, 16)]
        with self.assertRaisesRegex(DatasetDesignError, "symmetric seen/train and OOD"):
            build_manifest(rows, ood_scene_fraction=0.5)


if __name__ == "__main__":
    unittest.main()
