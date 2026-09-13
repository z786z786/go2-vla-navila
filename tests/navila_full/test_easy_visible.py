from __future__ import annotations

import unittest

from src.navila_full.easy_visible import (
    EASY_VISIBLE_PREFLIGHT_FORMAT,
    candidate_routes,
    is_single_stage_instruction,
    validate_preflight_manifest,
)


def episode(index: int, instruction: str, *, category: str = "straight") -> dict[str, object]:
    offset = index * 10.0
    if category == "straight":
        path = [[offset, 0, 0], [offset + 5.4, 0, 0]]
    elif category == "left_turn":
        path = [[offset, 0, 0], [offset + 2.7, 0, 0], [offset + 2.7, 2.7, 0]]
    else:
        path = [[offset, 0, 0], [offset + 2.7, 0, 0], [offset + 2.7, -2.7, 0]]
    return {
        "episode_id": f"episode-{index}", "trajectory_id": f"trajectory-{index}",
        "scene_id": "mp3d/house/house.glb", "instruction": {"instruction_text": instruction},
        "reference_path": path, "gt_locations": path, "start_position": path[0],
        "start_rotation": [1, 0, 0, 0], "goals": [{"position": path[-1], "radius": 0.5}],
    }


class EasyVisibleTests(unittest.TestCase):
    def test_language_accepts_direct_command_but_rejects_sequences(self) -> None:
        self.assertTrue(is_single_stage_instruction("Walk to the chair and stop."))
        self.assertFalse(is_single_stage_instruction("Walk to the chair then turn left."))
        self.assertFalse(is_single_stage_instruction("Walk to the chair; stop by the table."))

    def test_candidates_keep_exact_original_instruction_and_require_rgb_review(self) -> None:
        original = "  Go to the chair.  "
        routes = candidate_routes([episode(1, original), episode(2, "Walk to the chair then turn left.")])
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["instruction"], original)
        self.assertEqual(routes[0]["visibility_status"], "pending_marker_free_initial_rgb_review")

    def test_manifest_validation_rejects_unreviewed_state_rewrite(self) -> None:
        route = candidate_routes([episode(1, "Walk to the chair.")])[0]
        manifest = {"format": EASY_VISIBLE_PREFLIGHT_FORMAT, "dataset_schema_version": "navila_full_episode_go2_v1", "routes": [route]}
        self.assertEqual(validate_preflight_manifest(manifest), [])
        route["visibility_status"] = "approved"
        self.assertTrue(validate_preflight_manifest(manifest))
