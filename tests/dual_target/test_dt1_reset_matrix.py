"""CPU contracts for DT1 paired-reset evidence and exact expert matrix."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from src.dual_target.expert import ParkingExpert
from src.dual_target.expert_matrix import (
    DT1_EXPERT_REPEATS,
    ExpertMatrixError,
    build_dt1_expert_pairs,
    expert_matrix_manifest,
    flatten_dt1_expert_pairs,
)
from src.dual_target.low_level import HISTORY_LENGTH, PROPRIO_DIM
from src.dual_target.layouts import Vec2
from src.dual_target.reset_audit import (
    PAIR_SEED_ALGORITHM,
    RESET_AUDIT_SCHEMA,
    ResetAuditError,
    derive_pair_seed,
    reset_audit_from_dict,
    validate_paired_resets,
)
from src.dual_target.runner import (
    Dt1RunnerError,
    _configure_paired_expert_seed,
    _write_paired_reset_audit,
)
from src.dual_target.scene import dt1_development_groups


def _audit(*, target_color: str, rgb_sha: str = "a" * 64) -> dict[str, object]:
    group = "dt1_dev_000"
    configuration = "A_red_B_blue"
    repeat = 1
    return {
        "schema_version": RESET_AUDIT_SCHEMA,
        "geometry_group_id": group,
        "color_configuration": configuration,
        "repeat": repeat,
        "target_color": target_color,
        "scene_metadata_sha256": "c" * 64,
        "pair_seed_algorithm": PAIR_SEED_ALGORITHM,
        "pair_seed": derive_pair_seed(
            geometry_group_id=group, color_configuration=configuration, repeat=repeat
        ),
        "physics_step_after_reset": 0,
        "sim_time_after_reset_s": 0.0,
        "robot_root_pose_w": [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        "robot_root_lin_vel_b": [0.0, 0.0, 0.0],
        "robot_root_ang_vel_b": [0.0, 0.0, 0.0],
        "joint_pos": [0.0] * 12,
        "joint_vel": [0.0] * 12,
        "proprio_history": [[0.0] * PROPRIO_DIM for _ in range(HISTORY_LENGTH)],
        "target_poses_w": {
            "red": [2.0, 1.2, 0.25, 1.0, 0.0, 0.0, 0.0],
            "blue": [2.0, -1.2, 0.25, 1.0, 0.0, 0.0, 0.0],
        },
        "first_observation_seq": 0,
        "first_render_request_seq": 1,
        "first_camera_sensor_frame": 1,
        "first_rgb_path": f"rgb_before_warmup_{target_color}.png",
        "first_rgb_sha256": rgb_sha,
        "render_physics_step_before": 0,
        "render_physics_step_after": 0,
        "render_sim_time_before_s": 0.0,
        "render_sim_time_after_s": 0.0,
        "render_did_not_advance_physics": True,
        "learner_start_physics_step": 400,
        "learner_start_sim_time_s": 2.0,
        "learner_start_robot_root_pose_w": [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        "learner_start_body_velocity_body": [0.0, 0.0, 0.0],
        "learner_first_observation_seq": 2,
        "learner_first_render_request_seq": 2,
        "learner_first_camera_sensor_frame": 2,
        "learner_first_rgb_path": f"rgb_pre_learner_{target_color}.png",
        "learner_first_rgb_sha256": rgb_sha,
        "learner_render_physics_step_before": 400,
        "learner_render_physics_step_after": 400,
        "learner_render_sim_time_before_s": 2.0,
        "learner_render_sim_time_after_s": 2.0,
        "learner_render_did_not_advance_physics": True,
    }


class ResetAuditTests(unittest.TestCase):
    def test_same_seed_comes_from_pair_identity_not_target_or_instruction(self) -> None:
        seed = derive_pair_seed(
            geometry_group_id="dt1_dev_000", color_configuration="A_red_B_blue", repeat=1
        )
        self.assertEqual(seed, _audit(target_color="red")["pair_seed"])
        self.assertGreaterEqual(seed, 0)
        self.assertLessEqual(seed, 0x7FFF_FFFF)
        self.assertNotEqual(seed, derive_pair_seed(
            geometry_group_id="dt1_dev_000", color_configuration="A_red_B_blue", repeat=0
        ))

    def test_exact_paired_reset_validates_state_history_targets_and_first_rgb(self) -> None:
        red = _audit(target_color="red")
        blue = _audit(target_color="blue")
        outcome = validate_paired_resets(red, blue)
        self.assertEqual(outcome["status"], "PAIRED_RESET_MATCHED_NOT_TASK_APPROVED")
        self.assertEqual(outcome["task_colors"], ["red", "blue"])
        self.assertTrue(outcome["physics_and_reset_state_matched"])
        parsed = reset_audit_from_dict(red)
        self.assertEqual(parsed.proprio_history[0], tuple([0.0] * PROPRIO_DIM))

    def test_paired_reset_rejects_color_seed_history_or_rgb_mismatch(self) -> None:
        red = _audit(target_color="red")
        blue = _audit(target_color="blue")
        altered = copy.deepcopy(blue)
        altered["pair_seed"] = int(altered["pair_seed"]) + 1
        with self.assertRaises(ResetAuditError):
            validate_paired_resets(red, altered)
        altered = copy.deepcopy(blue)
        altered["proprio_history"][8][6] = 0.1  # latest command slot is still reset evidence.
        with self.assertRaises(ResetAuditError):
            validate_paired_resets(red, altered)
        validate_paired_resets(red, _audit(target_color="blue", rgb_sha="b" * 64))
        altered = copy.deepcopy(blue)
        altered["learner_first_rgb_sha256"] = "d" * 64
        validate_paired_resets(red, altered)
        altered = copy.deepcopy(blue)
        altered["render_physics_step_after"] = 1
        with self.assertRaises(ResetAuditError):
            reset_audit_from_dict(altered)

    def test_task_equivalence_accepts_small_reset_noise_but_rejects_material_changes(self):
        import math
        red, blue = _audit(target_color="red"), _audit(target_color="blue", rgb_sha="b" * 64)
        blue['robot_root_pose_w'][0] = .01
        blue['learner_start_robot_root_pose_w'][3:] = [math.cos(.01), 0., 0., math.sin(.01)]
        blue['joint_pos'][0] = .001
        blue['joint_vel'][0] = .02
        blue['learner_start_body_velocity_body'] = [.005, -.003, .01]
        result = validate_paired_resets(red, blue)
        self.assertEqual(result['comparison_policy'], 'task_equivalence_v2')
        self.assertFalse(result['cross_run_rgb_equality_required'])
        for field, index, value in [('robot_root_pose_w', 0, .1),
                                     ('learner_start_body_velocity_body', 0, .2)]:
            bad = copy.deepcopy(blue)
            bad[field][index] = value
            with self.assertRaises(ResetAuditError):
                validate_paired_resets(red, bad)
        bad = copy.deepcopy(blue)
        bad['target_poses_w']['red'][0] += .1
        with self.assertRaises(ResetAuditError):
            validate_paired_resets(red, bad)
        bad = copy.deepcopy(blue)
        bad['learner_start_robot_root_pose_w'][3:] = [math.cos(.1), 0., 0., math.sin(.1)]
        with self.assertRaises(ResetAuditError):
            validate_paired_resets(red, bad)

    def test_identical_stale_history_or_moving_starts_are_not_valid_resets(self):
        for mutate in (lambda x: x['proprio_history'][0].__setitem__(0, .1),
                       lambda x: x['learner_start_body_velocity_body'].__setitem__(0, .2)):
            red, blue = _audit(target_color='red'), _audit(target_color='blue')
            mutate(red)
            mutate(blue)
            with self.assertRaises(ResetAuditError):
                validate_paired_resets(red, blue)

    def test_runner_sidecar_writer_accepts_actual_reset_and_runtime_snapshot_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rgb_reset").mkdir()
            (root / "rgb_pre").mkdir()
            (root / "rgb_reset/000001.png").write_bytes(b"reset rgb")
            (root / "rgb_pre/000002.png").write_bytes(b"learner rgb")
            physical = {
                "robot_root_pose_w": [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                "robot_root_lin_vel_b": [0.0, 0.0, 0.0],
                "robot_root_ang_vel_b": [0.0, 0.0, 0.0],
                "joint_pos": [0.0] * 12,
                "joint_vel": [0.0] * 12,
                "proprio_history": [[0.0] * PROPRIO_DIM for _ in range(HISTORY_LENGTH)],
                "target_poses_w": {
                    "red": [2.0, 1.2, 0.25, 1.0, 0.0, 0.0, 0.0],
                    "blue": [2.0, -1.2, 0.25, 1.0, 0.0, 0.0, 0.0],
                },
            }
            reset_observation = {
                "reset_physics_step": 0, "reset_sim_time_s": 0.0,
                "observation_seq": 1, "render_request_seq": 1, "camera_sensor_frame": 1,
                "rgb_path": "rgb_reset/000001.png", "render_physics_step_before": 0,
                "render_physics_step_after": 0, "render_sim_time_before_s": 0.0,
                "render_sim_time_after_s": 0.0, "render_did_not_advance_physics": True,
            }
            learner = {
                # These are exactly _default_runtime_snapshot's public keys,
                # not a second, hand-invented reset schema.
                "robot_pose_w": [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                "body_velocity_body": [0.0, 0.0, 0.0],
            }
            learner_observation = {
                "learner_start_physics_step": 400, "learner_start_sim_time_s": 2.0,
                "observation_seq": 2, "render_request_seq": 2, "camera_sensor_frame": 2,
                "rgb_path": "rgb_pre/000002.png", "render_physics_step_before": 400,
                "render_physics_step_after": 400, "render_sim_time_before_s": 2.0,
                "render_sim_time_after_s": 2.0, "render_did_not_advance_physics": True,
            }
            args = SimpleNamespace(
                group_id="dt1_dev_000", color_configuration="A_red_B_blue", repeat=1,
                target_color="red", seed=derive_pair_seed(
                    geometry_group_id="dt1_dev_000", color_configuration="A_red_B_blue", repeat=1
                ),
            )
            path, digest = _write_paired_reset_audit(
                root, args=args, reset_physical_state=physical, reset_observation=reset_observation,
                learner_physical_state=learner, learner_observation=learner_observation,
                scene_metadata={"ground": "self-contained", "targets": ["red", "blue"]},
            )
            self.assertTrue(path.is_file())
            self.assertEqual(len(digest), 64)
            on_disk = reset_audit_from_dict(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(on_disk.learner_start_robot_root_pose_w, tuple(learner["robot_pose_w"]))
            self.assertEqual(on_disk.learner_start_body_velocity_body, tuple(learner["body_velocity_body"]))


class ExpertMatrixTests(unittest.TestCase):
    def test_fixed_matrix_has_eight_color_independent_reset_pairs_and_sixteen_tasks(self) -> None:
        pairs = build_dt1_expert_pairs()
        self.assertEqual(len(pairs), 8)
        flattened = flatten_dt1_expert_pairs(pairs)
        self.assertEqual(len(flattened), 16)
        for pair in pairs:
            self.assertEqual(pair.task_slots()[0][0], "red")
            self.assertEqual(pair.task_slots()[1][0], "blue")
            self.assertEqual(pair.pair_seed, derive_pair_seed(
                geometry_group_id=pair.scene.group.geometry_group_id,
                color_configuration=pair.scene.color_configuration,
                repeat=pair.repeat,
            ))
        manifest = expert_matrix_manifest()
        self.assertEqual(manifest["paired_repeats"], DT1_EXPERT_REPEATS)
        self.assertEqual(manifest["pair_count"], 8)
        self.assertEqual(manifest["task_count"], 16)
        self.assertFalse(manifest["dt1_approved"])

    def test_matrix_refuses_non_frozen_repeat_count(self) -> None:
        with self.assertRaises(ExpertMatrixError):
            build_dt1_expert_pairs(repeats=1)

    def test_runner_derives_identical_paired_seed_for_red_and_blue_without_user_seed(self) -> None:
        common = {"paired_reset": True, "seed": None, "group_id": "dt1_dev_001", "color_configuration": "A_blue_B_red", "repeat": 1}
        red = SimpleNamespace(**common, target_color="red")
        blue = SimpleNamespace(**common, target_color="blue")
        _configure_paired_expert_seed(red)
        _configure_paired_expert_seed(blue)
        self.assertEqual(red.seed, blue.seed)
        self.assertEqual(red.seed, derive_pair_seed(
            geometry_group_id="dt1_dev_001", color_configuration="A_blue_B_red", repeat=1
        ))
        explicitly_seeded = dict(common)
        explicitly_seeded.update(seed=99, target_color="red")
        with self.assertRaises(Dt1RunnerError):
            _configure_paired_expert_seed(SimpleNamespace(**explicitly_seeded))

    def test_development_parking_disc_has_the_revised_static_head_clearance(self) -> None:
        for group in dt1_development_groups():
            self.assertEqual(group.stand_off_m, 0.75)
            self.assertEqual(group.parking_radius_m, 0.30)
            self.assertEqual(group.min_box_edge_clearance_m, 0.45)
            self.assertGreaterEqual(group.stand_off_m - group.parking_radius_m, group.min_box_edge_clearance_m)


class ExpertTerminalHeadingTests(unittest.TestCase):
    def test_expert_aligns_fixed_box_facing_heading_before_hold_stop(self) -> None:
        expert = ParkingExpert()
        aligning = expert.command(
            robot_xy=Vec2(1.0, 0.0), robot_yaw_rad=0.8, parking_center=Vec2(1.0, 0.0), terminal_heading_rad=0.0,
        )
        self.assertEqual(aligning.phase, "align_terminal_heading")
        self.assertEqual(aligning.vx, 0.0)
        self.assertLess(aligning.wz, 0.0)
        holding = expert.command(
            robot_xy=Vec2(1.0, 0.0), robot_yaw_rad=0.05, parking_center=Vec2(1.0, 0.0), terminal_heading_rad=0.0,
        )
        self.assertEqual(holding.phase, "hold_stop")
        self.assertEqual(holding.as_list(), [0.0, 0.0, 0.0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
