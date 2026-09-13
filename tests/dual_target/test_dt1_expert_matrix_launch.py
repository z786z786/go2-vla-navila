"""CPU-only fixed-plan/collection checks for the DT1 expert matrix."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.dual_target.expert_matrix_launch import (
    PLAN_FORMAT,
    ExpertMatrixLaunchError,
    build_matrix_plan,
    collect_matrix_results,
    create_matrix_plan,
)
from src.dual_target.reset_audit import (
    PAIR_SEED_ALGORITHM,
    RESET_AUDIT_SCHEMA,
    derive_pair_seed,
)
from src.dual_target.low_level import HISTORY_LENGTH, PROPRIO_DIM


def _audit(slot: dict[str, object]) -> dict[str, object]:
    color = str(slot["target_color"])
    return {
        "schema_version": RESET_AUDIT_SCHEMA,
        "geometry_group_id": slot["geometry_group_id"], "color_configuration": slot["color_configuration"],
        "repeat": slot["repeat"], "target_color": color, "scene_metadata_sha256": "f" * 64,
        "pair_seed_algorithm": PAIR_SEED_ALGORITHM, "pair_seed": slot["pair_seed"],
        "physics_step_after_reset": 0, "sim_time_after_reset_s": 0.0,
        "robot_root_pose_w": [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        "robot_root_lin_vel_b": [0.0, 0.0, 0.0], "robot_root_ang_vel_b": [0.0, 0.0, 0.0],
        "joint_pos": [0.0] * 12, "joint_vel": [0.0] * 12,
        "proprio_history": [[0.0] * PROPRIO_DIM for _ in range(HISTORY_LENGTH)],
        "target_poses_w": {"red": [2.0, 1.2, 0.25, 1.0, 0.0, 0.0, 0.0], "blue": [2.0, -1.2, 0.25, 1.0, 0.0, 0.0, 0.0]},
        "first_observation_seq": 1, "first_render_request_seq": 1, "first_camera_sensor_frame": 1,
        "first_rgb_path": f"reset_{color}.png", "first_rgb_sha256": "a" * 64,
        "render_physics_step_before": 0, "render_physics_step_after": 0,
        "render_sim_time_before_s": 0.0, "render_sim_time_after_s": 0.0, "render_did_not_advance_physics": True,
        "learner_start_physics_step": 400, "learner_start_sim_time_s": 2.0,
        "learner_start_robot_root_pose_w": [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        "learner_start_body_velocity_body": [0.0, 0.0, 0.0],
        "learner_first_observation_seq": 2, "learner_first_render_request_seq": 2, "learner_first_camera_sensor_frame": 2,
        "learner_first_rgb_path": f"learner_{color}.png", "learner_first_rgb_sha256": "b" * 64,
        "learner_render_physics_step_before": 400, "learner_render_physics_step_after": 400,
        "learner_render_sim_time_before_s": 2.0, "learner_render_sim_time_after_s": 2.0,
        "learner_render_did_not_advance_physics": True,
    }


class ExpertMatrixLaunchTests(unittest.TestCase):
    def test_plan_predeclares_exact_sixteen_slots_and_never_overwrites(self) -> None:
        plan = build_matrix_plan("dt1_matrix_r1")
        self.assertEqual(plan["format"], PLAN_FORMAT)
        self.assertEqual(plan["slot_count"], 16)
        slots = plan["slots"]
        self.assertEqual(len({slot["run_id"] for slot in slots}), 16)
        for pair_index in range(8):
            pair_slots = [slot for slot in slots if slot["pair_index"] == pair_index]
            self.assertEqual([slot["target_color"] for slot in pair_slots], ["red", "blue"])
            self.assertEqual(pair_slots[0]["pair_seed"], pair_slots[1]["pair_seed"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_matrix_plan(root, "dt1_matrix_r1")
            with self.assertRaises(FileExistsError):
                create_matrix_plan(root, "dt1_matrix_r1")

    def test_collector_requires_all_fixed_slots_and_matching_pair_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = create_matrix_plan(root, "dt1_matrix_r1")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            for slot in plan["slots"]:
                run_root = root / slot["run_id"]
                episode_root = run_root / "episodes" / "episode"
                episode_root.mkdir(parents=True)
                audit_path = episode_root / "paired_reset_audit.json"
                audit = _audit(slot)
                for relative, contents, hash_key in (
                    (audit["first_rgb_path"], b"same reset", "first_rgb_sha256"),
                    (audit["learner_first_rgb_path"], b"same learner", "learner_first_rgb_sha256"),
                ):
                    rgb = episode_root / str(relative)
                    rgb.write_bytes(contents)
                    audit[hash_key] = hashlib.sha256(contents).hexdigest()
                audit_path.write_text(json.dumps(audit), encoding="utf-8")
                audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
                status = {"status": "RUNNING", "episode_result": {
                    "status": "success", "episode_id": slot["run_id"], "paired_reset_audit_path": str(audit_path),
                    "paired_reset_audit_sha256": audit_hash,
                }}
                (run_root / "stage_status.json").write_text(json.dumps(status), encoding="utf-8")
                summary = {
                    "run_id": slot["run_id"], "exit_code": 0, "stage_status": "RUNNING", "result_status": "success",
                }
                (root / f"dt1_expert_{slot['run_id']}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            result = collect_matrix_results(root, plan_path)
            self.assertEqual(result["status"], "EXPERT_MATRIX_COMPLETE_NOT_DT1_APPROVED")
            self.assertEqual(result["successful_slots"], 16)
            self.assertEqual(len(result["pair_reviews"]), 8)
            first_slot = plan["slots"][0]
            first_episode = root / first_slot["run_id"] / "episodes" / "episode"
            # Sidecar hash alone is insufficient: collector must hash the
            # actual RGB bytes referenced by both reset and learner-start
            # observations before accepting a slot.
            (first_episode / "reset_red.png").write_bytes(b"tampered")
            rgb_tampered = collect_matrix_results(root, plan_path)
            self.assertEqual(rgb_tampered["status"], "EXPERT_MATRIX_INCOMPLETE_OR_FAILED")
            (first_episode / "reset_red.png").write_bytes(b"same reset")
            summary_path = root / f"dt1_expert_{first_slot['run_id']}_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["exit_code"] = 1
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            launcher_failed = collect_matrix_results(root, plan_path)
            self.assertEqual(launcher_failed["status"], "EXPERT_MATRIX_INCOMPLETE_OR_FAILED")
            summary["exit_code"] = 0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            # A syntactically valid scalar is still not a stage-status object;
            # collector must report it as a slot failure, never raise .get.
            bad_status_path = root / plan["slots"][1]["run_id"] / "stage_status.json"
            bad_status_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            malformed = collect_matrix_results(root, plan_path)
            self.assertEqual(malformed["status"], "EXPERT_MATRIX_INCOMPLETE_OR_FAILED")
            (root / plan["slots"][0]["run_id"] / "stage_status.json").unlink()
            failed = collect_matrix_results(root, plan_path)
            self.assertEqual(failed["status"], "EXPERT_MATRIX_INCOMPLETE_OR_FAILED")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
