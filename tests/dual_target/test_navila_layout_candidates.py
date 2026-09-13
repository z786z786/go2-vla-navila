import unittest

from src.dual_target.navila_layout_candidates import OFFSETS_DEGREES, propose_from_routes


class NavilaLayoutCandidateTests(unittest.TestCase):
    def test_straight_route_produces_one_pose_per_offset(self):
        route = {
            "scene_name": "scene_a", "scene_id": "mp3d/scene_a/scene_a.glb",
            "route_id": "route_a", "source_episode_id": "1",
            "gt_locations": [[index * 0.25, 0.0, 1.0] for index in range(27)],
        }
        result = propose_from_routes([route])
        self.assertGreater(result["layout_proposals"], 0)
        self.assertEqual(len(result["candidates"]), result["layout_proposals"] * len(OFFSETS_DEGREES))
        first = result["candidates"][0]
        self.assertEqual(first["relative_heading_offset_degrees"], -20.0)
        self.assertAlmostEqual(first["box_center_separation_m"], 0.64)
        self.assertEqual(first["status"], "pending_live_two_box_visibility_gate")

    def test_turning_route_is_not_claimed_visible(self):
        route = {
            "scene_name": "scene_a", "scene_id": "mp3d/scene_a/scene_a.glb",
            "route_id": "route_turn", "source_episode_id": "2",
            "gt_locations": [[0, 0, 1], [1, 0, 1], [2, 0, 1], [2, 1, 1], [2, 2, 1], [2, 3, 1], [2, 4, 1]],
        }
        self.assertEqual(propose_from_routes([route])["candidates"], [])


if __name__ == "__main__":
    unittest.main()
