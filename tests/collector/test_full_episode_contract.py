from __future__ import annotations

import unittest
from unittest import mock

from src.collector.full_episode_contract import (
    TERMINAL_HOLD_FRAMES,
    action3_from_record,
    build_collection_provenance,
    policy_state3_from_velocity,
    validate_collection_provenance,
    validate_record,
)
from src.collector.collect_full_episode import adapter_episode, parse_args
from src.collector.collection_limits import EXPERT_PATH_MARKER_PRIM_PATH, is_expert_path_marker, pd_frame_limit_reached, result_with_forced_done
from src.collector.r7_supervision import PD_ACTION_SOURCE, TERMINAL_HOLD_ACTION_SOURCE


def record(*, action_source=PD_ACTION_SOURCE, action=(0.2, 0.0, -0.1), index=0):
    if action_source == TERMINAL_HOLD_ACTION_SOURCE:
        action = (0.0, 0.0, 0.0)
    return {
        "frame_index": index,
        "timestamp": index / 50.0,
        "control_dt_s": 1.0 / 50.0,
        "front_rgb": f"front_rgb/{index:06d}.jpg",
        "current_velocity": {"linear_body": [0.1, -0.2, 0.3], "angular_body": [0.4, 0.5, -0.6]},
        "expert_vx": action[0], "expert_vy": action[1], "expert_wz": action[2],
        "planner_command": list(action), "locomotion_command": list(action),
        "action_source": action_source,
    }


class FullEpisodeCollectionContractTests(unittest.TestCase):
    def test_pd_frame_watchdog_is_bounded_and_forces_only_the_outer_loop_done(self) -> None:
        self.assertTrue(is_expert_path_marker(EXPERT_PATH_MARKER_PRIM_PATH))
        self.assertTrue(is_expert_path_marker(EXPERT_PATH_MARKER_PRIM_PATH + "_1"))
        self.assertFalse(is_expert_path_marker("/Visuals/Other"))
        self.assertFalse(pd_frame_limit_reached(4_000, 3_999))
        self.assertTrue(pd_frame_limit_reached(4_000, 4_000))
        self.assertFalse(pd_frame_limit_reached(None, 1_000_000))
        observation, reward, done, info = result_with_forced_done(("obs", 1.5, False, {"key": "value"}))
        self.assertEqual((observation, reward, done, info), ("obs", 1.5, True, {"key": "value"}))

    def test_full_collector_forwards_explicit_pd_frame_cap(self) -> None:
        with mock.patch("sys.argv", ["collect_full_episode", "--dataset", "episodes.json.gz", "--source-episode-id", "7", "--episode-dir", "out", "--max-pd-frames", "4000"]):
            args, passthrough = parse_args()
        self.assertEqual(args.max_pd_frames, 4_000)
        self.assertEqual(passthrough, [])

    def test_only_current_body_velocity_is_projected(self) -> None:
        self.assertEqual(policy_state3_from_velocity(record()["current_velocity"]), [0.1, -0.2, -0.6])
        validate_record(record(), expected_index=0)
        self.assertEqual(action3_from_record(record()), [0.2, 0.0, -0.1])

    def test_terminal_label_is_zero_and_pd_labels_stay_bound_to_locomotion(self) -> None:
        terminal = record(action_source=TERMINAL_HOLD_ACTION_SOURCE, index=1)
        validate_record(terminal, expected_index=1)
        terminal["expert_vx"] = 0.1
        with self.assertRaisesRegex(ValueError, "does not equal"):
            action3_from_record(terminal)

    def test_provenance_keeps_original_paths_and_instruction(self) -> None:
        episode = {
            "episode_id": "e", "trajectory_id": "t", "scene_id": "mp3d/s/s.glb",
            "instruction": {"instruction_text": "Go to the red chair and stop."},
            "reference_path": [[0, 0, 0], [1, 0, 0]],
            "gt_locations": [[0, 0, 0], [1, 0, 0]],
        }
        provenance = build_collection_provenance(source_dataset_sha256="a" * 64, source_episode=episode)
        self.assertEqual(validate_collection_provenance(provenance), [])
        self.assertEqual(provenance["terminal_hold_frames"], TERMINAL_HOLD_FRAMES)
        self.assertEqual(provenance["original_reference_path"], episode["reference_path"])
        self.assertTrue(provenance["expert_path_markers_hidden"])

    def test_legacy_instrumentation_adapter_keeps_its_required_metadata(self) -> None:
        episode = {
            "episode_id": "e", "trajectory_id": "t", "scene_id": "mp3d/s/s.glb",
            "instruction": {"instruction_text": "Go to the red chair and stop."},
            "reference_path": [[0, 0, 0], [1, 0, 0]],
            "gt_locations": [[0, 0, 0], [2, 0, 0]],
            "start_position": [0, 0, 0], "start_rotation": [1, 0, 0, 0],
            "goals": [{"position": [2, 0, 0], "radius": 0.5}],
        }
        adapted = adapter_episode(episode)
        self.assertEqual(adapted["split"], "full-episode-pending-final-assignment")
        self.assertEqual(adapted["reference_path"], episode["gt_locations"])
