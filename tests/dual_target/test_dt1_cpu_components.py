"""CPU-only contract checks for DT1 components; no Isaac/CUDA imports."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.dual_target.expert import ParkingExpert
from src.dual_target.contact_precheck import ContactPrecheckError, _done_at_zero
from src.dual_target.calibration import (
    MotionCalibrationError,
    REQUIRED_ZERO_SAMPLES,
    ZERO_SETTLE_CASES,
    derive_motion_calibration,
)
from src.dual_target.gpu_wait import (
    DEFAULT_REQUIRED_FREE_MIB,
    DEFAULT_REQUIRED_SAMPLES,
    EXCLUSIVE_DT1_POLICY,
    SHARED_SMOKE_DT1_POLICY,
    Dt1GpuWaiter,
    GpuWaitError,
    LiveGpuSnapshot,
    acquire_live_admission,
    default_state_path,
    eligibility,
    parse_nvidia_smi_snapshot,
)
from src.dual_target.layouts import Vec2
from src.dual_target.low_level import HISTORY_LENGTH, PROPRIO_DIM, LowLevelContractError, LowLevelLayout
from src.dual_target.records import EpisodeEvidenceWriter, RecordValidationError, validate_post_step
from src.dual_target.runtime import PreResetEvidenceCapture, SubstepContactLatch
from src.dual_target.scene import (
    GROUND_ONLY_MESH_PATHS,
    GO2_CONTACT_BODY_NAMES,
    ROBOT_CONTACT_FILTER_PATHS,
    SELF_CONTAINED_GROUND_USD,
    DualTargetSceneSpec,
    contact_precheck_scene,
    dt1_development_groups,
    validate_robot_contact_filter_contract,
    validate_scene_metadata,
)
from src.dual_target.runner import (
    Dt1RunnerError,
    LowLevelVelocityAdapter,
    SharedSmokeResourceError,
    SharedSmokeResourceMonitor,
    _camera_rgb,
    _fallen_from_pose,
    _render_without_physics,
    _shared_smoke_usage_from_csv,
    _write_rgb,
    initialize_dt1_run,
    load_motion_calibration,
    validate_live_target_contact_sensor,
    write_dt1_failure_traceback,
    write_dt1_run_status,
)


def _post_record() -> dict[str, object]:
    return {
        "event": "post_step", "phase": "expert", "sim_time_before_s": 0.0, "sim_time_after_s": 0.02,
        "physics_step_before": 0, "physics_step_after": 4, "observation_seq_after": 1,
        "camera_sensor_frame": 1, "camera_timestamp_s": 0.02, "render_request_seq": 1,
        "robot_pose_w": [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], "body_velocity_body": [0.0, 0.0, 0.0],
        "collision_latched_substeps": [False, False, False, False], "target_contact_force_max": {"red": 0.0, "blue": 0.0},
        "in_correct_parking_region": False, "in_other_parking_region": False, "fallen": False,
        "terminated": False, "truncated": False, "evaluator_stop": False, "wrong_target_stop": False,
        "warmup": False, "pre_reset_snapshot_ref": "", "scorer_status": "in_progress", "scorer_detail": "outside",
    }


class SceneSpecTests(unittest.TestCase):
    def test_two_groups_four_color_assignments_have_fixed_geometry(self) -> None:
        groups = dt1_development_groups()
        self.assertEqual(len(groups), 2)
        for group in groups:
            forward = DualTargetSceneSpec(group, "A_red_B_blue")
            swapped = DualTargetSceneSpec(group, "A_blue_B_red")
            self.assertEqual(forward.slot_for_color("red"), group.slot_a)
            self.assertEqual(swapped.slot_for_color("red"), group.slot_b)
            self.assertEqual(forward.slot_for_color("blue"), swapped.slot_for_color("red"))
            metadata = forward.audit_metadata()
            self.assertEqual(metadata["terrain_mesh_paths"], list(GROUND_ONLY_MESH_PATHS))
            self.assertTrue(metadata["terrain_excludes_targets"])
            validate_scene_metadata(metadata)

    def test_scene_metadata_rejects_ambiguous_terrain_or_color(self) -> None:
        metadata = DualTargetSceneSpec(dt1_development_groups()[0], "A_red_B_blue").audit_metadata()
        metadata["terrain_mesh_paths"] = ["/World/ground", "/World/red_target"]
        with self.assertRaises(ValueError):
            validate_scene_metadata(metadata)

    def test_checked_in_ground_is_a_collision_mesh_without_grid_reference(self) -> None:
        contents = SELF_CONTAINED_GROUND_USD.read_text(encoding="utf-8")
        self.assertIn('def Mesh "mesh"', contents)
        self.assertIn("int[] faceVertexCounts = [3, 3]", contents)
        self.assertIn("int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]", contents)
        self.assertIn("PhysicsCollisionAPI", contents)
        self.assertIn("physics:collisionEnabled", contents)
        self.assertIn("PhysicsMaterialAPI", contents)
        self.assertIn("physics:staticFriction = 1.0", contents)
        self.assertIn("physics:dynamicFriction = 1.0", contents)
        self.assertIn("material:binding:physics", contents)
        self.assertNotIn("default_environment.usd", contents)
        self.assertNotIn("@", contents)

    def test_target_contact_filters_are_exact_single_rigid_body_paths(self) -> None:
        paths = validate_robot_contact_filter_contract()
        self.assertEqual(len(GO2_CONTACT_BODY_NAMES), 19)
        self.assertEqual(len(paths), 19)
        self.assertEqual(tuple(paths), ROBOT_CONTACT_FILTER_PATHS)
        self.assertEqual(GO2_CONTACT_BODY_NAMES[0], "base")
        self.assertEqual(GO2_CONTACT_BODY_NAMES[-1], "RR_foot")
        self.assertIn("Head_upper", GO2_CONTACT_BODY_NAMES)
        self.assertIn("Head_lower", GO2_CONTACT_BODY_NAMES)
        self.assertTrue(all(path.startswith("{ENV_REGEX_NS}/Robot/") for path in paths))
        self.assertTrue(all("*" not in path for path in paths))

    def test_contact_precheck_keeps_selected_color_on_physical_forward_slot(self) -> None:
        for color in ("red", "blue"):
            scene = contact_precheck_scene(color)
            self.assertEqual(scene.slot_for_color(color).slot_id, "A")
            self.assertEqual(scene.slot_for_color(color).center, scene.group.slot_a.center)
            self.assertEqual(scene.group.start, Vec2(0.0, 0.0))
            self.assertGreater(scene.group.slot_a.center.x, 1.0)


class LowLevelAndExpertTests(unittest.TestCase):
    def test_write_plan_hits_base_and_latest_flattened_history(self) -> None:
        layout = LowLevelLayout()
        width = 128 + HISTORY_LENGTH * PROPRIO_DIM
        plan = layout.command_write_plan(width)
        self.assertEqual(plan["base_policy"], (6, 9))
        self.assertEqual(plan["latest_history_buffer"], (6, 9))
        self.assertEqual(plan["latest_flattened_history"], (128 + 8 * 45 + 6, 128 + 8 * 45 + 9))
        with self.assertRaises(LowLevelContractError):
            LowLevelLayout(history_length=8).validate()

    def test_expert_does_not_drive_forward_when_target_is_behind(self) -> None:
        expert = ParkingExpert()
        command = expert.command(robot_xy=Vec2(0.0, 0.0), robot_yaw_rad=0.0, parking_center=Vec2(-1.0, 0.0))
        self.assertEqual(command.vy, 0.0)
        self.assertEqual(command.vx, 0.0)
        self.assertNotEqual(command.wz, 0.0)
        self.assertEqual(expert.command(robot_xy=Vec2(0.0, 0.0), robot_yaw_rad=0.0, parking_center=Vec2(0.05, 0.0)).phase, "hold_stop")

    def test_real_zero_windows_calibrate_bounded_stop_thresholds(self) -> None:
        def zero_events(planar: float, yaw: float) -> list[dict[str, object]]:
            return [
                {
                    "raw_action": [0.0, 0.0, 0.0], "applied_action": [0.0, 0.0, 0.0],
                    "warmup": False, "environment_done": False,
                    "collision_latched_substeps": [False, False, False, False],
                    "pre_reset_snapshot_refs": [], "physics_step_after": 4 * (index + 1),
                    "sim_time_after_s": 0.02 * (index + 1),
                    "body_velocity_body": [planar, 0.0, yaw],
                }
                for index in range(REQUIRED_ZERO_SAMPLES)
            ]

        calibration = derive_motion_calibration({
            case: zero_events(0.010, 0.020) for case in ZERO_SETTLE_CASES
        })
        self.assertEqual(calibration.status, "MOTION_CALIBRATED_NOT_TASK_CONTRACT_LOCKED")
        self.assertEqual(calibration.required_zero_samples, REQUIRED_ZERO_SAMPLES)
        self.assertEqual(calibration.derived_body_planar_speed_mps, 0.03)
        self.assertEqual(calibration.derived_body_yaw_rate_radps, 0.05)

    def test_zero_calibration_rejects_nonzero_command_or_unsafe_telemetry(self) -> None:
        def zero_events() -> list[dict[str, object]]:
            return [
                {
                    "raw_action": [0.0, 0.0, 0.0], "applied_action": [0.0, 0.0, 0.0],
                    "warmup": False, "environment_done": False,
                    "collision_latched_substeps": [False, False, False, False],
                    "pre_reset_snapshot_refs": [], "physics_step_after": 4 * (index + 1),
                    "sim_time_after_s": 0.02 * (index + 1),
                    "body_velocity_body": [0.01, 0.0, 0.02],
                }
                for index in range(REQUIRED_ZERO_SAMPLES)
            ]

        events = {case: zero_events() for case in ZERO_SETTLE_CASES}
        events["left_return_zero"][-1]["raw_action"] = [0.1, 0.0, 0.0]
        with self.assertRaises(MotionCalibrationError):
            derive_motion_calibration(events)
        unsafe = {case: zero_events() for case in ZERO_SETTLE_CASES}
        for event in unsafe["right_return_zero"]:
            event["body_velocity_body"] = [0.08, 0.0, 0.0]
        with self.assertRaises(MotionCalibrationError):
            derive_motion_calibration(unsafe)

    def test_runner_loads_only_exact_validated_motion_calibration(self) -> None:
        def events() -> list[dict[str, object]]:
            return [
                {
                    "raw_action": [0.0, 0.0, 0.0], "applied_action": [0.0, 0.0, 0.0],
                    "warmup": False, "environment_done": False,
                    "collision_latched_substeps": [False, False, False, False],
                    "pre_reset_snapshot_refs": [], "physics_step_after": 4 * (index + 1),
                    "sim_time_after_s": 0.02 * (index + 1), "body_velocity_body": [0.0, 0.0, 0.0],
                }
                for index in range(REQUIRED_ZERO_SAMPLES)
            ]

        calibration = derive_motion_calibration({case: events() for case in ZERO_SETTLE_CASES})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion_calibration.json"
            path.write_text(json.dumps(calibration.as_dict()), encoding="utf-8")
            loaded, digest = load_motion_calibration(path)
            self.assertEqual(loaded, calibration)
            self.assertEqual(len(digest), 64)
            bad = calibration.as_dict()
            bad["unexpected"] = True
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(Dt1RunnerError):
                load_motion_calibration(path)


class EvidenceAndRuntimeTests(unittest.TestCase):
    def test_contact_latch_rejects_invalid_force_and_preserves_instant_contact(self) -> None:
        latch = SubstepContactLatch(threshold_n=1.0)
        for bad in (math.nan, math.inf, -0.01, True):
            with self.assertRaises(ValueError):
                latch.capture({"red": bad, "blue": 0.0})
        latch.capture({"red": 0.0, "blue": 2.0})
        latch.capture({"red": 0.0, "blue": 0.0})
        latch.capture({"red": 0.0, "blue": 0.0})
        latch.capture({"red": 0.0, "blue": 0.0})
        hits, maxima = latch.consume_env_step()
        self.assertEqual(hits, [True, False, False, False])
        self.assertEqual(maxima, {"red": 0.0, "blue": 2.0})

    def test_post_record_rejects_force_bypasses_and_non_json(self) -> None:
        for bad in (math.nan, math.inf, -0.1, True):
            record = _post_record()
            record["target_contact_force_max"] = {"red": bad, "blue": 0.0}
            with self.assertRaises(RecordValidationError):
                validate_post_step(record)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "episode"
            with self.assertRaises(ValueError):
                EpisodeEvidenceWriter(root, {"bad": math.nan})
            # Constructor must fail rather than emit a permissive NaN JSON file.
            self.assertFalse(root.exists())

    def test_pre_reset_hook_captures_before_reset_and_four_scene_updates(self) -> None:
        class Scene:
            def __init__(self) -> None:
                self.calls = 0

            def update(self, dt: float) -> None:
                self.calls += 1

        class Env:
            def __init__(self) -> None:
                self.scene = Scene()
                self.resets: list[list[int]] = []

            def _reset_idx(self, env_ids):
                self.resets.append(list(env_ids))

        env = Env()
        substeps: list[int] = []
        capture = PreResetEvidenceCapture(env, lambda _: {"pre": "state"}, lambda: substeps.append(1))
        capture.install()
        for _ in range(4):
            env.scene.update(0.005)
        env._reset_idx([0])
        capture.restore()
        self.assertEqual(substeps, [1, 1, 1, 1])
        self.assertEqual(capture.records, [{"env_ids": [0], "pre": "state"}])
        self.assertEqual(env.resets, [[0]])

    def test_latch_begin_requires_previous_four_substeps_be_consumed(self) -> None:
        latch = SubstepContactLatch()
        latch.begin_env_step()
        latch.capture({"red": 0.0, "blue": 0.0})
        with self.assertRaises(RuntimeError):
            latch.begin_env_step()
        for _ in range(3):
            latch.capture({"red": 0.0, "blue": 0.0})
        latch.consume_env_step()
        latch.begin_env_step()


class RunnerPrimitiveTests(unittest.TestCase):
    def test_episode_writer_creates_reset_rgb_path_before_first_capture(self) -> None:
        class FakeCv2:
            COLOR_RGB2BGR = 1

            @staticmethod
            def cvtColor(image, code):
                self = FakeCv2
                self.converted = (image, code)
                return image

            @staticmethod
            def imwrite(path, image):
                destination = Path(path)
                if not destination.parent.is_dir():
                    return False
                destination.write_bytes(b"fake-rgb")
                return True

        class Rgb:
            ndim = 3
            shape = (2, 2, 3)

        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"cv2": FakeCv2}):
            root = Path(temporary) / "episode"
            EpisodeEvidenceWriter(root, {"paired_reset_required": True})
            destination = root / "rgb_reset" / "000001.png"
            # The test does not pre-create rgb_reset; writer lifecycle must.
            _write_rgb(destination, Rgb())
            self.assertTrue(destination.is_file())
            self.assertEqual(FakeCv2.converted[1], FakeCv2.COLOR_RGB2BGR)

    def test_reset_first_render_needs_only_root_pose_and_clock_not_step_terminal_flags(self) -> None:
        class RootState:
            def __getitem__(self, key):
                self.assert_key = key
                return [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

        class Sim:
            current_time_step_index = 0
            current_time = 0.0

            def __init__(self) -> None:
                self.renders = 0

            def render(self) -> None:
                self.renders += 1

        sim = Sim()
        env = type("ResetOnlyEnv", (), {
            "_sim_step_counter": 0,
            "sim": sim,
            # Deliberately omit reset_terminated/reset_time_outs: these are
            # created by the real manager only after its first step.
            "scene": {"robot": type("Robot", (), {
                "data": type("Data", (), {"root_state_w": RootState()})(),
            })()},
        })()
        before, after, pose_before, pose_after = _render_without_physics(env)
        self.assertEqual(before, after)
        self.assertEqual(pose_before, pose_after)
        self.assertEqual(sim.renders, 1)

    def test_target_contact_view_requires_exact_1_by_19_backend_and_data_shapes(self) -> None:
        class TensorShape:
            def __init__(self, shape) -> None:
                self.shape = shape
                self.ndim = len(shape)

        class ContactView:
            sensor_count = 1
            filter_count = 19
            sensor_paths = ["/World/envs/env_0/red_target"]

            def __init__(self) -> None:
                self.queries: list[float] = []

            def get_contact_force_matrix(self, dt: float):
                self.queries.append(dt)
                return TensorShape((1, 19, 3))

        contact_view = ContactView()
        sensor = type("Sensor", (), {
            "cfg": type("Cfg", (), {
                "filter_prim_paths_expr": [
                    path.format(ENV_REGEX_NS="/World/envs/env_.*") for path in ROBOT_CONTACT_FILTER_PATHS
                ],
            })(),
            "contact_physx_view": contact_view,
            "body_physx_view": type("BodyView", (), {"count": 1})(),
            "data": type("Data", (), {"force_matrix_w": TensorShape((1, 1, 19, 3))})(),
        })()
        with patch("src.dual_target.runner._require_finite_contact_tensor"):
            record = validate_live_target_contact_sensor(sensor, color="red", env_regex_ns="/World/envs/env_.*")
        self.assertEqual(record["sensor_count"], 1)
        self.assertEqual(record["body_count"], 1)
        self.assertEqual(record["filter_count"], 19)
        self.assertEqual(record["force_matrix_shape"], [1, 1, 19, 3])
        self.assertEqual(record["configured_filter_paths"], sensor.cfg.filter_prim_paths_expr)
        self.assertEqual(record["expanded_filter_paths"][0], "/World/envs/env_0/Robot/base")
        self.assertEqual(contact_view.queries, [0.005])

        contact_view.filter_count = 18
        with patch("src.dual_target.runner._require_finite_contact_tensor"), self.assertRaises(Dt1RunnerError):
            validate_live_target_contact_sensor(sensor, color="red", env_regex_ns="/World/envs/env_.*")

    def test_target_contact_view_rejects_reordered_live_filter_config(self) -> None:
        class TensorShape:
            def __init__(self, shape) -> None:
                self.shape = shape
                self.ndim = len(shape)

        configured = [path.format(ENV_REGEX_NS="/World/envs/env_.*") for path in ROBOT_CONTACT_FILTER_PATHS]
        sensor = type("Sensor", (), {
            "cfg": type("Cfg", (), {"filter_prim_paths_expr": list(reversed(configured))})(),
            "contact_physx_view": type("View", (), {
                "sensor_count": 1, "filter_count": 19,
                "sensor_paths": ["/World/envs/env_0/red_target"],
                "get_contact_force_matrix": lambda self, _: TensorShape((1, 19, 3)),
            })(),
            "body_physx_view": type("BodyView", (), {"count": 1})(),
            "data": type("Data", (), {"force_matrix_w": TensorShape((1, 1, 19, 3))})(),
        })()
        with self.assertRaises(Dt1RunnerError):
            validate_live_target_contact_sensor(sensor, color="red", env_regex_ns="/World/envs/env_.*")

    def test_contact_precheck_done_signal_is_single_and_strict(self) -> None:
        self.assertFalse(_done_at_zero(0))
        self.assertTrue(_done_at_zero([1]))
        self.assertTrue(_done_at_zero(True))
        for bad in ([0, 0], 2, "false", 1.0):
            with self.assertRaises(ContactPrecheckError):
                _done_at_zero(bad)

    def test_shared_smoke_monitor_parses_owned_and_foreign_memory_and_fails_closed(self) -> None:
        sample = _shared_smoke_usage_from_csv(
            "24576, 14000\n", "77, 9000\n123, 2500\n", owned_root_pid=123,
        )
        self.assertEqual(sample["free_mib"], 10576)
        self.assertEqual(sample["owned_root_used_mib"], 2500)
        self.assertFalse(sample["compute_processes"][0]["owned_root_pid"])
        self.assertTrue(sample["compute_processes"][1]["owned_root_pid"])
        with self.assertRaises(SharedSmokeResourceError):
            _shared_smoke_usage_from_csv("24576, N/A\n", "", owned_root_pid=123)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "gpu.jsonl"
            monitor = SharedSmokeResourceMonitor(
                output, owned_root_pid=123,
                probe=lambda **_: _shared_smoke_usage_from_csv("24576, 23000\n", "123, 3000\n", owned_root_pid=123),
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            monitor.sample_once()
            with self.assertRaises(SharedSmokeResourceError):
                monitor.check()
            monitor.close()
            self.assertTrue(monitor.summary_path.exists())

    def test_run_root_binds_current_source_without_rewriting_dt0_approval(self) -> None:
        """A DT1 invocation may bind itself, but never grant its own approval."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            root = initialize_dt1_run(output, run_id="dt1_manifest_cpu", actual_argv=["runner", "--live"])
            manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
            status = json.loads((root / "stage_status.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "DT1")
            self.assertTrue(manifest["dt0_parent_binding"]["historical_hashes_match"])
            self.assertIn("src/dual_target/runner.py", manifest["source_sha256"])
            self.assertEqual(status["status"], "RUNNING")
            self.assertEqual(status["approval_authority"], "parent_agent_only")
            with self.assertRaises(FileExistsError):
                initialize_dt1_run(output, run_id="dt1_manifest_cpu", actual_argv=["runner", "--live"])
            with self.assertRaises(Dt1RunnerError):
                write_dt1_run_status(root, "READY_FOR_REVIEW", run_id="dt1_manifest_cpu", detail="forbidden")

    def test_failure_traceback_is_persisted_before_teardown_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                raise AttributeError("NoneType.GetPath")
            except AttributeError as exc:
                evidence = write_dt1_failure_traceback(root, exc=exc)
            traceback_path = Path(evidence["failure_traceback_path"])
            self.assertTrue(traceback_path.is_file())
            self.assertIn("AttributeError: NoneType.GetPath", traceback_path.read_text(encoding="utf-8"))
            self.assertEqual(len(evidence["failure_traceback_sha256"]), 64)

    def test_tensor_dict_like_camera_output_and_fall_signal(self) -> None:
        class TensorDictLike:
            def __init__(self) -> None:
                self.values = {"rgb": object()}

            def __contains__(self, key):
                return key in self.values

            def __getitem__(self, key):
                return self.values[key]

        camera = type("Camera", (), {"data": type("Data", (), {"output": TensorDictLike()})()})()
        self.assertIs(_camera_rgb(camera), camera.data.output.values["rgb"])
        self.assertTrue(_fallen_from_pose([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(_fallen_from_pose([0.0, 0.0, 0.4, math.cos(0.6), math.sin(0.6), 0.0, 0.0]))
        self.assertFalse(_fallen_from_pose([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]))

    def test_adapter_uses_real_counter_and_exact_four_latched_substeps(self) -> None:
        class ShapeOnly:
            def __init__(self, shape) -> None:
                self.shape = shape

        class Base:
            def __init__(self) -> None:
                self._sim_step_counter = 19
                self.sim = type("Sim", (), {"current_time_step_index": 19, "current_time": 19 * .005})()

        latch = SubstepContactLatch()
        base = Base()

        class History:
            def __init__(self) -> None:
                self.unwrapped = base
                self.proprio_obs_buf = ShapeOnly((1, 9, 45))
                self.low = ShapeOnly((1, 128 + 9 * 45))

            def reset(self):
                return self.low, {}

            def step(self, action):
                base._sim_step_counter += 4
                base.sim.current_time_step_index += 4
                base.sim.current_time += 4 * .005
                for substep in range(4):
                    latch.capture({"red": 2.0 if substep == 1 else 0.0, "blue": 0.0})
                return self.low, 0.0, 0, {"observations": {}}

        class InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        fake_torch = type("Torch", (), {"inference_mode": staticmethod(lambda: InferenceMode())})()
        history = History()
        adapter = LowLevelVelocityAdapter(history, lambda _: "low_level_action", physics_dt_s=.005, decimation=4)
        adapter.reset()
        with patch("src.dual_target.runner.synchronized_command_observation", side_effect=lambda obs, *_: obs), patch.dict(sys.modules, {"torch": fake_torch}):
            applied, _, done, action, _, hits, forces = adapter.step((0.2, 0.0, 0.1), latch)
        self.assertEqual(adapter.physics_step, 23)
        self.assertEqual(applied.applied, (0.2, 0.0, 0.1))
        self.assertEqual(done, 0)
        self.assertEqual(action, "low_level_action")
        self.assertEqual(hits, [False, True, False, False])
        self.assertEqual(forces["red"], 2.0)

    def test_adapter_rejects_extra_simulation_step_hidden_from_manager(self) -> None:
        """The reset offset may be nonzero, but it must never drift later."""
        class ShapeOnly:
            def __init__(self, shape) -> None:
                self.shape = shape

        base = type("Base", (), {
            "_sim_step_counter": 7,
            "sim": type("Sim", (), {"current_time_step_index": 12, "current_time": .060})(),
        })()
        latch = SubstepContactLatch()

        class History:
            unwrapped = base
            proprio_obs_buf = ShapeOnly((1, 9, 45))
            low = ShapeOnly((1, 128 + 9 * 45))

            def reset(self):
                return self.low, {}

            def step(self, action):
                base._sim_step_counter += 4
                # SimContext observed an extra step: this is a true timing
                # error even though the initial offset was allowed.
                base.sim.current_time_step_index += 5
                base.sim.current_time += 5 * .005
                for _ in range(4):
                    latch.capture({"red": 0.0, "blue": 0.0})
                return self.low, 0.0, 0, {}

        class InferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        fake_torch = type("Torch", (), {"inference_mode": staticmethod(lambda: InferenceMode())})()
        adapter = LowLevelVelocityAdapter(History(), lambda _: "action", physics_dt_s=.005, decimation=4)
        adapter.reset()
        with patch("src.dual_target.runner.synchronized_command_observation", side_effect=lambda obs, *_: obs), patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(Dt1RunnerError):
                adapter.step((0.1, 0.0, 0.0), latch)


class GpuWaitTests(unittest.TestCase):
    def test_parser_and_three_real_decisions_are_fail_closed(self) -> None:
        parsed = parse_nvidia_smi_snapshot("24576, 1024, 3\n", "")
        self.assertEqual(parsed.free_mib, 23552)
        with self.assertRaises(GpuWaitError):
            parse_nvidia_smi_snapshot("24576, 1024\n", "1805, 100\nnot-a-pid, 20\n")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = iter([
                LiveGpuSnapshot(24576, 4000, ()),
                LiveGpuSnapshot(24576, 4000, (91,)),
                LiveGpuSnapshot(24576, 4000, ()),
                LiveGpuSnapshot(24576, 4000, ()),
                LiveGpuSnapshot(24576, 4000, ()),
                LiveGpuSnapshot(24576, 4000, ()),
            ])
            waiter = Dt1GpuWaiter(
                project_root=root, run_id="dt1_cpu_wait", argv=["test"], interval_s=0.001,
                probe=lambda: next(snapshots),
            )
            state = waiter.wait()
            self.assertEqual(state["status"], "GPU_READY_LOCKED_RECHECKED")
            self.assertEqual(state["consecutive_eligible_samples"], 3)
            self.assertEqual(state["snapshot"]["compute_process_count"], 0)

    def test_project_lock_refuses_duplicate_waiter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = Dt1GpuWaiter(project_root=root, run_id="first", argv=["test"], interval_s=1.0,
                                  probe=lambda: LiveGpuSnapshot(24576, 4000, ()))
            second = Dt1GpuWaiter(project_root=root, run_id="second", argv=["test"], interval_s=1.0,
                                   probe=lambda: LiveGpuSnapshot(24576, 4000, ()))
            first.acquire()
            try:
                with self.assertRaises(GpuWaitError):
                    second.acquire()
            finally:
                first.close()

    def test_live_admission_requires_live_waiter_and_rechecks_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = default_state_path(root)
            path.parent.mkdir(parents=True)
            state = {
                "project_root": str(root.resolve()), "status": "GPU_READY_LOCKED_RECHECKED",
                "admission_source": "live_nvidia_smi", "required_free_mib": DEFAULT_REQUIRED_FREE_MIB,
                "required_consecutive_samples": DEFAULT_REQUIRED_SAMPLES, "sample_interval_s": 30.0,
                "last_checked_utc": datetime.now(timezone.utc).isoformat(), "run_id": "live_one",
                "admission_policy": EXCLUSIVE_DT1_POLICY.policy_id, "allow_existing_compute": False,
            }
            path.write_text(json.dumps(state), encoding="utf-8")
            lease = acquire_live_admission(
                root, run_id="live_one", actual_argv=["runner"],
                probe=lambda: LiveGpuSnapshot(24576, 4000, ()),
            )
            try:
                self.assertEqual(lease.state["run_id"], "live_one")
                self.assertTrue((path.parent / "dt1_live_gpu_admission_live_one.json").exists())
            finally:
                lease.close()
            state["admission_source"] = "injected_test_only"
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaises(GpuWaitError):
                acquire_live_admission(root, run_id="forbidden", actual_argv=["runner"],
                                      probe=lambda: LiveGpuSnapshot(24576, 4000, ()))

    def test_shared_smoke_is_a_separate_fixed_policy_that_records_foreign_compute(self) -> None:
        shared_snapshot = LiveGpuSnapshot(24576, 13000, (777,))
        self.assertEqual(
            eligibility(shared_snapshot, required_free_mib=SHARED_SMOKE_DT1_POLICY.required_free_mib,
                        allow_existing_compute=True),
            (True, "eligible_shared_compute"),
        )
        self.assertFalse(eligibility(shared_snapshot)[0])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = iter([shared_snapshot, shared_snapshot, shared_snapshot, shared_snapshot])
            waiter = Dt1GpuWaiter(
                project_root=root, run_id="shared_one", argv=["absolute-python"],
                required_free_mib=SHARED_SMOKE_DT1_POLICY.required_free_mib,
                required_samples=SHARED_SMOKE_DT1_POLICY.required_samples,
                interval_s=SHARED_SMOKE_DT1_POLICY.sample_interval_s,
                probe=lambda: next(snapshots), policy=SHARED_SMOKE_DT1_POLICY,
            )
            waiter.acquire()
            try:
                waiter._started_at = datetime.now(timezone.utc).isoformat()
                waiter._write_state("WAITING_GPU", snapshot=None, reason="test_started")
                for _ in range(SHARED_SMOKE_DT1_POLICY.required_samples):
                    state = waiter.sample_once()
                state = waiter._locked_recheck()
            finally:
                waiter.close()
            self.assertEqual(state["admission_policy"], "dt1_shared_smoke_v1")
            self.assertTrue(state["allow_existing_compute"])
            self.assertEqual(state["snapshot"]["compute_pids"], [777])
            lease = acquire_live_admission(
                root, run_id="shared_one", actual_argv=["absolute-runner"], policy=SHARED_SMOKE_DT1_POLICY,
                probe=lambda: shared_snapshot,
            )
            try:
                self.assertEqual(lease.state["admission_policy"], "dt1_shared_smoke_v1")
            finally:
                lease.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
