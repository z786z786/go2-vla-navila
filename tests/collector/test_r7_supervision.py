import json
import tempfile
import unittest
from pathlib import Path

from src.collector.r7_canary import evaluate_canary
from src.collector.r7_supervision import (
    PD_ACTION_SOURCE,
    TERMINAL_HOLD_ACTION_SOURCE,
    command_semantics,
    supervised_action,
    terminal_hold_audit,
)


def _record(*, source, action, pd_action, terminal=False, success=False, state_dim=31):
    return {
        "action_source": source,
        "planner_command": list(action),
        "locomotion_command": list(action),
        "expert_vx": action[0],
        "expert_vy": action[1],
        "expert_wz": action[2],
        "pd_planner_command": list(pd_action),
        "raw_environment_done": source == TERMINAL_HOLD_ACTION_SOURCE,
        "next_current_velocity": {
            "linear_body": [0.5 * pd_action[0], 0.0, 0.0],
            "angular_body": [0.0, 0.0, 0.5 * pd_action[2]],
        },
        "next_distance_to_goal_xy_m": 0.4,
        "goal_pose": {"success_radius_m": 0.5},
        "termination": terminal,
        "success": success,
        "robot_state": [0.0] * state_dim,
    }


def _successful_records(state_dim=31):
    rows = []
    for index in range(10):
        vx = 0.15 + index * 0.01
        wz = 0.2 if index % 2 else -0.2
        rows.append(_record(source=PD_ACTION_SOURCE, action=(vx, 0.0, wz), pd_action=(vx, 0.0, wz), state_dim=state_dim))
    for index in range(50):
        rows.append(
            _record(
                source=TERMINAL_HOLD_ACTION_SOURCE,
                action=(0.0, 0.0, 0.0),
                pd_action=(0.1, 0.0, -0.1),
                terminal=index == 49,
                success=index == 49,
                state_dim=state_dim,
            )
        )
    return rows


class R7SupervisionTests(unittest.TestCase):
    def test_terminal_hold_is_zero_labelled_tail_with_unchanged_pd_evidence(self):
        self.assertEqual(supervised_action((0.2, 0.0, -0.3), terminal_hold=False), ((0.2, 0.0, -0.3), PD_ACTION_SOURCE))
        self.assertEqual(supervised_action((0.2, 0.0, -0.3), terminal_hold=True), ((0.0, 0.0, 0.0), TERMINAL_HOLD_ACTION_SOURCE))
        rows = _successful_records()
        self.assertTrue(command_semantics(rows, allow_legacy=False)["passed"])
        audit = terminal_hold_audit(rows, requested_frames=50)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["frame_indices"], list(range(10, 60)))

    def test_terminal_hold_rejects_nonzero_or_non_tail_label(self):
        rows = _successful_records()
        rows[-2]["planner_command"] = [0.1, 0.0, 0.0]
        rows[-2]["locomotion_command"] = [0.1, 0.0, 0.0]
        rows[-2]["expert_vx"] = 0.1
        self.assertFalse(command_semantics(rows, allow_legacy=False)["passed"])
        self.assertFalse(terminal_hold_audit(rows, requested_frames=50)["passed"])

    def test_canary_requires_six_balanced_successful_r7_episodes(self):
        specs = [
            ("straight", "train", "scene_0"),
            ("straight", "train", "scene_1"),
            ("left_turn", "train", "scene_2"),
            ("left_turn", "seen-val", "scene_3"),
            ("right_turn", "train", "scene_0"),
            ("right_turn", "seen-val", "scene_1"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = []
            for index, (category, split, scene) in enumerate(specs):
                directory = root / f"episode_{index}"
                directory.mkdir()
                instruction = {
                    "straight": "Move forward and stop.",
                    "left_turn": "Turn left and stop.",
                    "right_turn": "Turn right and stop.",
                }[category]
                (directory / "summary.json").write_text(json.dumps({"short_episode_id": directory.name, "split": split, "scene_id": scene, "instruction": instruction, "status": "complete", "success": True}), encoding="utf-8")
                (directory / "sanity.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
                (directory / "collector_config.json").write_text(json.dumps({"terminal_hold_frames": 50}), encoding="utf-8")
                (directory / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in _successful_records()) + "\n", encoding="utf-8")
                directories.append(directory)
            report = evaluate_canary(directories, expected_state_dim=31, expected_hold_frames=50)
        self.assertTrue(report["passed"])
        self.assertEqual(report["coverage"]["category_counts"], {"left_turn": 2, "right_turn": 2, "straight": 2})
        self.assertEqual(report["coverage"]["split_counts"], {"seen-val": 2, "train": 4})
