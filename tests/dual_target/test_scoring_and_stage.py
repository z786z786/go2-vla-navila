from __future__ import annotations

import unittest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import patch

from src.dual_target.scoring import AutonomousParkingScorer, ScoreFrame, ScoreStatus
from src.dual_target.stage import (
    GpuSnapshot,
    StageStatus,
    acquire_project_lock,
    create_run_directory,
    evaluate_gpu_wait_dry_run,
    main as stage_main,
    validate_transition,
)


def frame(time_s: float, step: int, seq: int, **overrides: object) -> ScoreFrame:
    values: dict[str, object] = {
        "sim_time_s": time_s, "physics_step": step, "observation_seq": seq,
        "raw_action": (0.0, 0.0, 0.0), "applied_action": (0.0, 0.0, 0.0),
        "body_vx_mps": 0.0, "body_vy_mps": 0.0, "body_yaw_rate_radps": 0.0,
        "in_correct_parking_region": True,
    }
    values.update(overrides)
    return ScoreFrame(**values)  # type: ignore[arg-type]


class ParkingScoringTests(unittest.TestCase):
    def test_requires_full_one_second_not_only_49_intervals(self) -> None:
        scorer = AutonomousParkingScorer()
        decision = None
        for index in range(50):
            decision = scorer.observe(frame(index * 0.02, index * 4, index))
        self.assertIsNotNone(decision)
        self.assertEqual(decision.status, ScoreStatus.IN_PROGRESS)
        self.assertAlmostEqual(decision.window_duration_s, 0.98)
        decision = scorer.observe(frame(1.0, 200, 50))
        self.assertEqual(decision.status, ScoreStatus.SUCCESS)
        self.assertAlmostEqual(decision.window_duration_s, 1.0)

    def test_duplicate_time_or_physics_or_observation_never_accumulates_stop(self) -> None:
        scorer = AutonomousParkingScorer()
        self.assertEqual(scorer.observe(frame(0.0, 0, 0)).status, ScoreStatus.IN_PROGRESS)
        duplicate = scorer.observe(frame(0.02, 0, 1))
        self.assertEqual(duplicate.status, ScoreStatus.INVALID_SAMPLE)
        self.assertIn("physics_step", duplicate.detail)
        self.assertEqual(scorer.observe(frame(1.02, 204, 51)).status, ScoreStatus.INVALID_SAMPLE)

    def test_raw_negative_vx_clamped_to_zero_cannot_fake_stop(self) -> None:
        scorer = AutonomousParkingScorer()
        decision = scorer.observe(frame(0.0, 0, 0, raw_action=(-0.2, 0.0, 0.0), applied_action=(0.0, 0.0, 0.0)))
        self.assertEqual(decision.status, ScoreStatus.IN_PROGRESS)
        self.assertIn("raw", decision.detail)

    def test_terminal_events_and_warmup_never_become_success(self) -> None:
        scorer = AutonomousParkingScorer()
        for index in range(80):
            self.assertNotEqual(scorer.observe(frame(index * 0.02, index * 4, index, warmup=True)).status, ScoreStatus.SUCCESS)
        decision = scorer.observe(frame(1.62, 324, 81, evaluator_stop=True))
        self.assertEqual(decision.status, ScoreStatus.FAILED_EXTERNAL_STOP)
        self.assertEqual(scorer.observe(frame(3.0, 600, 150)).status, ScoreStatus.FAILED_EXTERNAL_STOP)
        for field, expected in (("collision", ScoreStatus.FAILED_COLLISION), ("fallen", ScoreStatus.FAILED_FALLEN), ("wrong_target_stop", ScoreStatus.FAILED_WRONG_TARGET_STOP)):
            with self.subTest(field=field):
                self.assertEqual(AutonomousParkingScorer().observe(frame(0.0, 0, 0, **{field: True})).status, expected)

    def test_valid_collision_event_locks_failure_even_when_other_telemetry_is_nan(self) -> None:
        scorer = AutonomousParkingScorer()
        decision = scorer.observe(frame(0.0, 0, 0, collision=True, body_vx_mps=float("nan")))
        self.assertEqual(decision.status, ScoreStatus.FAILED_COLLISION)
        for index in range(1, 54):
            self.assertEqual(
                scorer.observe(frame(index * 0.02, index * 4, index)).status,
                ScoreStatus.FAILED_COLLISION,
            )

    def test_nonfinite_sample_cannot_contribute(self) -> None:
        scorer = AutonomousParkingScorer()
        self.assertEqual(scorer.observe(frame(0.0, 0, 0)).status, ScoreStatus.IN_PROGRESS)
        self.assertEqual(scorer.observe(frame(0.02, 4, 1, body_vx_mps=float("nan"))).status, ScoreStatus.INVALID_SAMPLE)
        # The first clean sample starts a new window, rather than inheriting
        # the 20 ms of stop time before the invalid telemetry.
        restarted = scorer.observe(frame(1.02, 204, 51))
        self.assertEqual(restarted.status, ScoreStatus.IN_PROGRESS)
        self.assertEqual(restarted.window_duration_s, 0.0)

    def test_bool_and_deployment_action_mapping_are_strict(self) -> None:
        scorer = AutonomousParkingScorer()
        self.assertEqual(scorer.observe(frame(0.0, 0, 0, in_correct_parking_region="false")).status, ScoreStatus.INVALID_SAMPLE)
        self.assertEqual(scorer.observe(frame(0.02, 4, 1, applied_action=(0.0, 1.0, 0.0))).status, ScoreStatus.INVALID_SAMPLE)
        self.assertEqual(scorer.observe(frame(0.04, 8, 2, raw_action=(0.2, 0.0, 0.0), applied_action=(0.0, 0.0, 0.0))).status, ScoreStatus.INVALID_SAMPLE)


class StageTests(unittest.TestCase):
    def test_terra_cannot_approve_and_gpu_wait_dry_run_tracks_three_samples_without_probe(self) -> None:
        with self.assertRaises(PermissionError):
            validate_transition(StageStatus.READY_FOR_REVIEW, StageStatus.APPROVED, actor="terra")
        validate_transition(StageStatus.READY_FOR_REVIEW, StageStatus.APPROVED, actor="parent")
        dry_run = evaluate_gpu_wait_dry_run([
            GpuSnapshot(24 * 1024, 2 * 1024, 0),
            GpuSnapshot(24 * 1024, 3 * 1024, 0),
            GpuSnapshot(24 * 1024, 4 * 1024, 0),
        ])
        self.assertFalse(dry_run.allocation_attempted)
        self.assertFalse(dry_run.resource_probe_executed)
        self.assertEqual(dry_run.result, "ready_for_gpu_acquisition")
        reset = evaluate_gpu_wait_dry_run([
            GpuSnapshot(24 * 1024, 2 * 1024, 0),
            GpuSnapshot(24 * 1024, 2 * 1024, 1),
            GpuSnapshot(24 * 1024, 2 * 1024, 0),
            GpuSnapshot(24 * 1024, 2 * 1024, 0),
        ])
        self.assertEqual(reset.result, "waiting_for_eligible_samples")
        self.assertEqual(reset.sample_decisions[-1]["consecutive_eligible_samples"], 2)

    def test_stage_lock_is_cross_process_safe_and_run_id_cannot_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = acquire_project_lock(root / ".dt0_stage.lock")
            try:
                with self.assertRaisesRegex(RuntimeError, "holds"):
                    acquire_project_lock(root / ".dt0_stage.lock")
            finally:
                import fcntl
                fcntl.flock(first.fileno(), fcntl.LOCK_UN)
                first.close()
            create_run_directory(root, "dt0_once")
            with self.assertRaises(FileExistsError):
                create_run_directory(root, "dt0_once")

    def test_bad_supplied_snapshot_persists_failed_status_with_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "bad_snapshots.json"
            snapshots.write_text(json.dumps([{"total_mib": "not-an-int", "used_mib": 0, "compute_process_count": 0}]))
            output_dir = root / "runs"
            argv = [
                "stage.py", "DT0", "--output-dir", str(output_dir), "--run-id", "dt0_bad_snapshot",
                "--gpu-wait-dry-run", "--gpu-wait-snapshots", str(snapshots),
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                stage_main()
            status = json.loads((output_dir / "dt0_bad_snapshot" / "stage_status.json").read_text())
            self.assertEqual(status["status"], "FAILED")
            self.assertIn("ValueError", status["failure_reason"])
            self.assertIsInstance(status["pid"], int)


if __name__ == "__main__":
    unittest.main()
