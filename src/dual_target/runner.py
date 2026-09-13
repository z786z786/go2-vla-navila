"""DT1 entry point: CPU audit now, live Isaac Go2 runner only on explicit use.

The CPU path imports neither Isaac nor torch.  The live path is intentionally
separate from NaVILA's ``Planner``/``VLNEnvWrapper``: it has no reference path,
goal command, goal distance stop, marker, or task-derived control input.  It
only loads the audited low-level locomotion checkpoint and replaces its velocity
command in both the base observation and the flattened 9-frame history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import ACTION_BOUNDS, AppliedAction, TaskContract, apply_deployment_bounds
from .calibration import MotionCalibration, MotionCalibrationError
from .expert import ParkingExpert
from .expert_matrix import ExpertMatrixError, build_dt1_expert_pairs
from .gpu_wait import (
    EXCLUSIVE_DT1_POLICY,
    SHARED_SMOKE_DT1_POLICY,
    SHARED_SMOKE_MIN_FREE_MIB_DURING_RUN,
    acquire_live_admission,
)
from .layouts import TaskSpec, Vec2, build_four_tasks
from .low_level import HISTORY_LENGTH, PROPRIO_DIM, LowLevelContractError, command_consistency_report, synchronized_command_observation
from .records import EpisodeEvidenceWriter
from .reset_audit import (
    PAIR_SEED_ALGORITHM,
    ResetAudit,
    ResetAuditError,
    derive_pair_seed,
)
from .runtime import PreResetEvidenceCapture, SubstepContactLatch, camera_audit_fields
from .scene import (
    GO2_CONTACT_BODY_NAMES,
    ROBOT_CONTACT_FILTER_PATHS,
    DualTargetSceneSpec,
    SceneAdapterError,
    apply_flat_dual_target_scene,
    dt1_development_groups,
    validate_robot_contact_filter_contract,
)
from .scoring import AutonomousParkingScorer, ParkingThresholds, ScoreDecision, ScoreFrame, ScoreStatus


DT1_FORMAT = "go2-dual-target-dt1-runner-v1"
PHYSICS_DT_S = 0.005
DECIMATION = 4
ENV_STEP_S = PHYSICS_DT_S * DECIMATION
GO2_WARMUP_STEPS = 100
DEFAULT_CHECKPOINT = Path(
    "<external-data-root>"
    "2024-09-25_23-22-02/model_26499.pt"
)
DEFAULT_CHECKPOINT_SHA256 = "1e21097122ab0bfccaf9d4df2df794d8c1c918a1ddca72c07e38b36768f2e76c"
GO2_USD_RELATIVE_TO_ASSET_ROOT = Path("Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd")
DT1_TIMELINE_END_S = 10_000.0


class Dt1RunnerError(RuntimeError):
    """The runner cannot produce trustworthy DT1 evidence."""


class SharedSmokeResourceError(Dt1RunnerError):
    """The narrowly authorized shared-GPU smoke lost its safety margin."""


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,80}", value):
        raise Dt1RunnerError("DT1 run_id must be 3-81 safe filename characters")
    return value


def _asset_root_from_go2_usd(go2_usd: str) -> Path:
    """Derive and verify the local Isaac asset root from the explicit Go2 USD."""
    path = Path(go2_usd).resolve()
    if not path.is_file():
        raise Dt1RunnerError(f"verified local Go2 USD is absent: {path}")
    try:
        asset_root = path.parents[5]
    except IndexError as exc:
        raise Dt1RunnerError("Go2 USD path is too shallow to identify an Isaac asset root") from exc
    if asset_root / GO2_USD_RELATIVE_TO_ASSET_ROOT != path:
        raise Dt1RunnerError("Go2 USD must have the audited Isaac asset-root layout")
    return asset_root


def _configure_dt1_kit_args_before_app(asset_root: Path) -> list[str]:
    """Pin local assets before AppLauncher without importing the legacy client."""
    kit_args = [
        f"--/persistent/isaac/asset_root/default={asset_root}",
        f"--/persistent/isaac/asset_root/cloud={asset_root}",
        f"--/persistent/isaac/asset_root/nvidia={asset_root}",
        "--/persistent/isaac/asset_root/timeout=30.0",
    ]
    appended: list[str] = []
    for argument in kit_args:
        prefix = argument.split("=", 1)[0] + "="
        if not any(item.startswith(prefix) for item in sys.argv):
            sys.argv.append(argument)
            appended.append(argument)
    return appended


def _configure_dt1_runtime_after_app(asset_root: Path) -> dict[str, object]:
    """Set process-local assets and extend Kit's finite camera timeline.

    This runs after AppLauncher and before ``parse_env_cfg``/``gym.make``.  It
    avoids importing or monkey-patching the legacy VLN client, planner, gym
    constructor, or SimulationContext reset path.
    """
    try:
        import carb
        import omni.timeline
        import omni.usd
    except ImportError as exc:  # pragma: no cover - Isaac only
        raise Dt1RunnerError("Kit runtime services are unavailable after AppLauncher") from exc
    settings = carb.settings.get_settings()
    for key in ("default", "cloud", "nvidia"):
        settings.set_string(f"/persistent/isaac/asset_root/{key}", str(asset_root))
    settings.set_float("/persistent/isaac/asset_root/timeout", 30.0)
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise Dt1RunnerError("Kit stage is unavailable before DT1 SimulationContext construction")
    previous_end_s = float(timeline.get_end_time())
    previous_stage_end = float(stage.GetEndTimeCode())
    stage.SetEndTimeCode(DT1_TIMELINE_END_S * stage.GetTimeCodesPerSecond())
    timeline.set_end_time(DT1_TIMELINE_END_S)
    timeline.commit()
    return {
        "asset_root": str(asset_root), "asset_root_settings": ["default", "cloud", "nvidia"],
        "asset_timeout_s": 30.0, "timeline_end_before_s": previous_end_s,
        "timeline_end_after_s": float(timeline.get_end_time()),
        "stage_end_before_timecodes": previous_stage_end,
        "stage_end_after_timecodes": float(stage.GetEndTimeCode()),
        "timeline_extension_s": DT1_TIMELINE_END_S,
        "legacy_client_imported": False,
    }


class LiveClock(tuple):
    """Manager and simulator time sources that must agree on every action."""

    __slots__ = ()

    def __new__(cls, manager_step: int, sim_step: int, sim_time_s: float):
        return tuple.__new__(cls, (manager_step, sim_step, sim_time_s))

    @property
    def manager_step(self) -> int:
        return self[0]

    @property
    def sim_step(self) -> int:
        return self[1]

    @property
    def sim_time_s(self) -> float:
        return self[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_motion_calibration(path: Path) -> tuple[MotionCalibration, str]:
    """Load only an exact, bounded zero-command calibration artifact."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise Dt1RunnerError(f"motion calibration is absent: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Dt1RunnerError(f"motion calibration cannot be parsed: {resolved}") from exc
    expected = {field.name for field in fields(MotionCalibration)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise Dt1RunnerError("motion calibration keys do not match the frozen schema")
    payload = dict(payload)
    payload["zero_cases"] = tuple(payload["zero_cases"]) if isinstance(payload["zero_cases"], list) else payload["zero_cases"]
    try:
        calibration = MotionCalibration(**payload)
        calibration.validate()
    except (TypeError, MotionCalibrationError) as exc:
        raise Dt1RunnerError(f"motion calibration is invalid: {exc}") from exc
    return calibration, sha256(resolved)


def _dt0_approval_binding(project_root: Path) -> dict[str, object]:
    """Verify historical DT0 approval without mistaking current DT1 edits for it."""
    approval_path = project_root / "reports/dual_target_v1/dt0_approval.json"
    if not approval_path.is_file():
        raise Dt1RunnerError("DT1 requires the parent DT0 approval record")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("stage") != "DT0" or approval.get("status") != "APPROVED":
        raise Dt1RunnerError("DT0 parent approval record is not APPROVED")
    gate_path = project_root / str(approval.get("local_gate", ""))
    if not gate_path.is_file():
        raise Dt1RunnerError("DT0 parent approval local gate is absent")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("stage") != "DT0" or gate.get("status") != "READY_FOR_REVIEW":
        raise Dt1RunnerError("DT0 local gate no longer has its reviewed state")
    approved_hashes = approval.get("source_sha256")
    if not isinstance(approved_hashes, dict) or gate.get("source_sha256") != approved_hashes:
        raise Dt1RunnerError("DT0 approval no longer binds the reviewed historical gate source hashes")
    return {"approval_path": str(approval_path), "approval_sha256": sha256(approval_path),
            "local_gate": str(gate_path), "local_gate_sha256": sha256(gate_path),
            "historical_hashes_match": True,
            "note": "DT1 source manifest is separate; current stage/script hashes are intentionally not compared to DT0."}


def _current_source_manifest(project_root: Path, *, run_id: str, actual_argv: Sequence[str]) -> dict[str, object]:
    roots = [
        project_root / "src/dual_target", project_root / "config/dual_target_v1",
        project_root / "scripts/dual_target_run_stage.sh", project_root / "scripts/dual_target_shared_smoke.sh",
        project_root / "scripts/dual_target_contact_precheck.sh",
        project_root / "scripts/dual_target_dt1_expert.sh",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts))
    if not paths:
        raise Dt1RunnerError("DT1 source manifest has no owned source files")
    return {
        "format": "go2-dual-target-dt1-source-manifest-v1", "stage": "DT1", "run_id": run_id,
        "created_at_utc": _utc_now(), "actual_argv": list(actual_argv),
        "source_sha256": {str(path.relative_to(project_root)): sha256(path) for path in paths},
        "dt0_parent_binding": _dt0_approval_binding(project_root),
    }


def initialize_dt1_run(output_dir: Path, *, run_id: str, actual_argv: Sequence[str]) -> Path:
    """Create one non-overwriting DT1 run root and bind its current source."""
    safe_id = _safe_run_id(run_id)
    run_root = output_dir.resolve() / safe_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing DT1 run: {run_root}")
    project_root = Path(__file__).resolve().parents[2]
    manifest = _current_source_manifest(project_root, run_id=safe_id, actual_argv=actual_argv)
    run_root.mkdir(parents=True)
    (run_root / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_dt1_run_status(run_root, "RUNNING", run_id=safe_id, detail="live runner initialized; no self-approval")
    return run_root


def write_dt1_run_status(run_root: Path, status: str, *, run_id: str, detail: str, extra: dict[str, object] | None = None) -> None:
    if status not in {"RUNNING", "FAILED", "INTERRUPTED_RESUMABLE"}:
        raise Dt1RunnerError("Terra DT1 runner cannot write an approval or review-ready status")
    payload: dict[str, object] = {
        "format": "go2-dual-target-dt1-stage-status-v1", "stage": "DT1", "run_id": run_id,
        "status": status, "actor": "terra", "pid": os.getpid(), "updated_at_utc": _utc_now(),
        "approval_authority": "parent_agent_only", "detail": detail,
    }
    if extra:
        payload.update(extra)
    temporary = run_root / f".stage_status.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(run_root / "stage_status.json")


def write_dt1_failure_traceback(run_root: Path, *, exc: BaseException) -> dict[str, str]:
    """Persist the launch/build exception before Kit close can alter exit state."""
    destination = run_root / "failure_traceback.txt"
    if destination.exists():
        raise Dt1RunnerError(f"refusing to overwrite DT1 failure traceback: {destination}")
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if not rendered:
        rendered = f"{type(exc).__name__}: {exc}\n"
    temporary = run_root / f".failure_traceback.{os.getpid()}.tmp"
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(destination)
    return {
        "failure_traceback_path": str(destination),
        "failure_traceback_sha256": sha256(destination),
    }


def write_dt1_partial_gate(
    run_root: Path,
    *,
    run_id: str,
    result: dict[str, object],
    admission: dict[str, object],
    evidence_kind: str = "expert_episode",
) -> None:
    """Record one live sub-run without claiming the full DT1 stage passed."""
    manifest = run_root / "source_manifest.json"
    payload = {
        "format": "go2-dual-target-dt1-partial-gate-v1", "stage": "DT1", "run_id": run_id,
        "status": "RUNNING", "approval_authority": "parent_agent_only", "created_at_utc": _utc_now(),
        "source_manifest": str(manifest), "source_manifest_sha256": sha256(manifest),
        "gpu_admission": admission, "evidence_kind": evidence_kind, "episode_result": result,
        "next_step": "retain this partial evidence; complete all DT1 gates and await parent review",
    }
    temporary = run_root / f".gate.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(run_root / "gate.json")


def dt1_task_matrix(*, repeats: int = 2) -> tuple[tuple[DualTargetSceneSpec, TaskSpec, int], ...]:
    """Return the fixed 2 geometry × 4 combination × repeat schedule."""
    if repeats <= 0:
        raise ValueError("DT1 repeats must be positive")
    matrix: list[tuple[DualTargetSceneSpec, TaskSpec, int]] = []
    for group in dt1_development_groups():
        for configuration in ("A_red_B_blue", "A_blue_B_red"):
            scene = DualTargetSceneSpec(group, configuration)
            tasks = {task.target_color: task for task in build_four_tasks(group) if task.color_configuration == configuration}
            if set(tasks) != {"red", "blue"}:
                raise Dt1RunnerError("scene configuration does not produce exactly red/blue instruction tasks")
            for color in ("red", "blue"):
                for repeat in range(repeats):
                    matrix.append((scene, tasks[color], repeat))
    return tuple(matrix)


def cpu_self_check() -> dict[str, object]:
    """Audit the live-run plan without importing a simulator or allocator."""
    matrix = dt1_task_matrix(repeats=2)
    expected = 2 * 2 * 2 * 2
    scenes = {(scene.group.geometry_group_id, scene.color_configuration) for scene, _, _ in matrix}
    return {
        "format": DT1_FORMAT,
        "stage": "DT1",
        "mode": "cpu_self_check",
        "real_simulation_evidence": False,
        "status": "RUNNING_CPU_IMPLEMENTATION",
        "checks": {
            "development_expert_schedule_has_16_slots": len(matrix) == expected,
            "two_geometry_groups": len({scene.group.geometry_group_id for scene, _, _ in matrix}) == 2,
            "four_color_instruction_combinations": len(scenes) == 4,
            "old_low_level_proprio_45_and_history_9_frozen": PROPRIO_DIM == 45 and HISTORY_LENGTH == 9,
            "history_rebuild_plan_present": command_consistency_report(128 + 45 * 9)["rebuild_required"] is True,
            "task_contract_still_pending_sim_validation": TaskContract().status == "pending_sim_validation",
        },
        "checkpoint": {"path": str(DEFAULT_CHECKPOINT), "sha256": DEFAULT_CHECKPOINT_SHA256, "loaded": False},
        "live_entry_requires": ["--live", "--go2-usd", "GPU_READY_LOCKED_RECHECKED live admission"],
        "shared_smoke_exception": {
            "available_only_with": ["--live", "--shared-smoke", "--max-env-steps=50"],
            "admission_policy": SHARED_SMOKE_DT1_POLICY.policy_id,
            "required_free_mib": SHARED_SMOKE_DT1_POLICY.required_free_mib,
            "foreign_compute_recorded_not_owned": True,
            "not_general_training_or_DT2_authorization": True,
        },
    }


def _as_scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise Dt1RunnerError("simulator supplied a non-finite scalar")
    return result


def _as_nonnegative_int(value: Any, label: str) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Dt1RunnerError(f"{label} must be a non-negative integer")
    return value


def _live_clock(base_env: Any) -> LiveClock:
    """Read independent manager/simulator physical clock fields."""
    manager_step = _as_nonnegative_int(getattr(base_env, "_sim_step_counter", None), "manager sim step counter")
    sim = getattr(base_env, "sim", None)
    sim_step = _as_nonnegative_int(getattr(sim, "current_time_step_index", None), "SimulationContext current_time_step_index")
    sim_time = _as_scalar(getattr(sim, "current_time", None))
    return LiveClock(manager_step, sim_step, sim_time)


def _as_vector(value: Any, length: int, label: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise Dt1RunnerError(f"{label} has unexpected shape")
    result = [_as_scalar(item) for item in value]
    return result


def _bool_at_zero(value: Any, label: str) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if not value:
            raise Dt1RunnerError(f"{label} is empty")
        value = value[0]
    if isinstance(value, bool):
        return value
    # RslRlVecEnvHistoryWrapper turns (terminated | truncated) into a long
    # done tensor.  Accept exactly 0/1, never generic integer truthiness.
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    raise Dt1RunnerError(f"{label} must be a bool or 0/1 done flag")


def _camera_rgb(camera: Any) -> Any:
    output = getattr(getattr(camera, "data", None), "output", None)
    # Isaac Camera uses TensorDict, not a builtin dict.  Require mapping-like
    # containment/index access so a future output container cannot silently
    # become an unrelated object.
    if output is None or not hasattr(output, "__contains__") or not hasattr(output, "__getitem__") or "rgb" not in output:
        raise Dt1RunnerError("standard rgbd_camera did not provide rgb")
    return output["rgb"]


def _write_rgb(path: Path, rgb: Any) -> None:
    """Persist a pre-action RGB audit frame without adding it to learner state."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - Isaac host dependency
        raise Dt1RunnerError("OpenCV is required to save DT1 RGB evidence") from exc
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu()
    if hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    if getattr(rgb, "ndim", None) == 4:
        rgb = rgb[0]
    if getattr(rgb, "ndim", None) != 3 or rgb.shape[-1] not in {3, 4}:
        raise Dt1RunnerError("rgbd_camera image must be H×W×3 or H×W×4")
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise Dt1RunnerError(f"failed to write RGB evidence {path}")


def _shared_smoke_usage_from_csv(gpu_csv: str, process_csv: str, *, owned_root_pid: int) -> dict[str, object]:
    """Parse one strict shared-smoke sample, retaining foreign PID memory.

    This parser is intentionally separate from admission: shared smoke is
    allowed to *observe* existing compute, but a malformed query never means
    the resource is safe.  The runner owns exactly its root Python PID; any
    other PID remains an audit record rather than an inferred owned process.
    """
    if isinstance(owned_root_pid, bool) or not isinstance(owned_root_pid, int) or owned_root_pid <= 0:
        raise SharedSmokeResourceError("shared-smoke owned root PID must be positive")
    rows = [line.strip() for line in gpu_csv.splitlines() if line.strip()]
    if len(rows) != 1:
        raise SharedSmokeResourceError("shared-smoke monitor needs exactly one GPU memory row")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2:
        raise SharedSmokeResourceError("shared-smoke monitor GPU row must contain total and used MiB")
    try:
        total_mib, used_mib = (int(field) for field in fields)
    except ValueError as exc:
        raise SharedSmokeResourceError("shared-smoke monitor received non-integer GPU memory") from exc
    if total_mib <= 0 or used_mib < 0 or used_mib > total_mib:
        raise SharedSmokeResourceError("shared-smoke monitor GPU memory is out of range")
    processes: list[dict[str, int | bool]] = []
    for line in process_csv.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise SharedSmokeResourceError("shared-smoke monitor process row is malformed")
        try:
            pid, memory_mib = (int(field) for field in fields)
        except ValueError as exc:
            raise SharedSmokeResourceError("shared-smoke monitor process row is non-numeric") from exc
        if pid <= 0 or memory_mib < 0:
            raise SharedSmokeResourceError("shared-smoke monitor process values are out of range")
        processes.append({"pid": pid, "used_mib": memory_mib, "owned_root_pid": pid == owned_root_pid})
    return {
        "total_mib": total_mib, "used_mib": used_mib, "free_mib": total_mib - used_mib,
        "compute_processes": processes,
        "owned_root_pid": owned_root_pid,
        "owned_root_used_mib": sum(item["used_mib"] for item in processes if item["owned_root_pid"]),
    }


def _probe_shared_smoke_usage(*, owned_root_pid: int) -> dict[str, object]:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10.0,
        )
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SharedSmokeResourceError(f"shared-smoke GPU monitor query failed: {exc}") from exc
    return _shared_smoke_usage_from_csv(gpu.stdout, processes.stdout, owned_root_pid=owned_root_pid)


class SharedSmokeResourceMonitor:
    """Per-second GPU accounting for the sole user-authorized shared smoke.

    It starts before AppLauncher, records all compute PIDs plus the runner PID,
    and fail-closes the runner on an unavailable query or <2 GiB free margin.
    The outer dedicated launcher also owns a process-group timeout for a Kit
    startup hang before this Python loop can check the event.
    """

    def __init__(self, output_path: Path, *, owned_root_pid: int, interval_s: float = 1.0,
                 probe: Callable[..., dict[str, object]] = _probe_shared_smoke_usage) -> None:
        if interval_s <= 0 or not math.isfinite(interval_s):
            raise ValueError("shared-smoke monitor interval must be positive and finite")
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite shared-smoke GPU monitor {output_path}")
        self.output_path = output_path
        self.summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        self.owned_root_pid = owned_root_pid
        self.interval_s = interval_s
        self.probe = probe
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: str | None = None
        self._sample_count = 0
        self._min_free_mib: int | None = None
        self._owned_peak_mib = 0

    @property
    def failure(self) -> str | None:
        return self._failure

    def _append(self, payload: dict[str, object]) -> None:
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def sample_once(self) -> dict[str, object]:
        if self._failure is not None:
            return {"failure": self._failure}
        try:
            sample = self.probe(owned_root_pid=self.owned_root_pid)
            free_mib = sample.get("free_mib")
            owned_mib = sample.get("owned_root_used_mib")
            if (isinstance(free_mib, bool) or not isinstance(free_mib, int)
                    or isinstance(owned_mib, bool) or not isinstance(owned_mib, int)):
                raise SharedSmokeResourceError("shared-smoke probe returned malformed memory values")
            self._sample_count += 1
            self._min_free_mib = free_mib if self._min_free_mib is None else min(self._min_free_mib, free_mib)
            self._owned_peak_mib = max(self._owned_peak_mib, owned_mib)
            payload = {"timestamp_utc": _utc_now(), "sample_index": self._sample_count, **sample}
            if free_mib < SHARED_SMOKE_MIN_FREE_MIB_DURING_RUN:
                self._failure = f"shared-smoke free GPU memory fell below {SHARED_SMOKE_MIN_FREE_MIB_DURING_RUN} MiB"
                payload["failure"] = self._failure
            self._append(payload)
            return payload
        except BaseException as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            payload = {"timestamp_utc": _utc_now(), "sample_index": self._sample_count + 1,
                       "monitor_error": self._failure, "owned_root_pid": self.owned_root_pid}
            self._append(payload)
            return payload

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_once()
        self._thread = threading.Thread(target=self._run, name="dt1-shared-smoke-gpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample_once()
            if self._failure is not None:
                return

    def check(self) -> None:
        if self._failure is not None:
            raise SharedSmokeResourceError(self._failure)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
        summary = {
            "format": "go2-dual-target-dt1-shared-smoke-gpu-monitor-v1",
            "owned_root_pid": self.owned_root_pid, "sample_interval_s": self.interval_s,
            "sample_count": self._sample_count, "min_free_mib": self._min_free_mib,
            "owned_root_peak_mib": self._owned_peak_mib, "failure": self._failure,
            "closed_at_utc": _utc_now(), "samples_jsonl": str(self.output_path),
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _contact_tensor_shape(value: Any, expected: tuple[int, ...], label: str) -> None:
    if getattr(value, "ndim", None) != len(expected) or tuple(getattr(value, "shape", ())) != expected:
        raise Dt1RunnerError(f"{label} has unexpected shape {tuple(getattr(value, 'shape', ()))}, expected {expected}")


def _require_finite_contact_tensor(value: Any, label: str) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Isaac host dependency
        raise Dt1RunnerError("torch is required for live target-contact inspection") from exc
    if not bool(torch.all(torch.isfinite(value))):
        raise Dt1RunnerError(f"{label} contains non-finite force telemetry")


def _target_contact_matrix(sensor: Any, *, color: str) -> Any:
    """Read the current [env,target-body,19 robot bodies,xyz] force matrix."""
    matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
    if matrix is None:
        raise Dt1RunnerError(f"{color} target contact sensor lacks filtered force_matrix_w")
    _contact_tensor_shape(matrix, (1, 1, len(GO2_CONTACT_BODY_NAMES), 3), f"{color} target contact force matrix")
    _require_finite_contact_tensor(matrix, f"{color} target contact force matrix")
    return matrix


def validate_live_target_contact_sensor(
    sensor: Any,
    *,
    color: str,
    physics_dt_s: float = PHYSICS_DT_S,
    env_regex_ns: str = "{ENV_REGEX_NS}",
) -> dict[str, object]:
    """Fail closed unless the live backend has the audited 1×19 contact view.

    Isaac's ContactSensor initializer checks that its target side has one
    body, but it does not reject a wildcard that expands the filter side into
    the wrong rigid-body view.  Inspect both public backend views, then force
    a real backend matrix query before any DT1 evidence is collected.
    """
    if color not in {"red", "blue"}:
        raise Dt1RunnerError("target-contact validation needs red or blue")
    filters = validate_robot_contact_filter_contract()
    if tuple(filters) != ROBOT_CONTACT_FILTER_PATHS:
        raise Dt1RunnerError("scene contact filter contract changed after configuration")
    if not isinstance(env_regex_ns, str) or not env_regex_ns:
        raise Dt1RunnerError("live scene has no usable environment regex namespace")
    # InteractiveScene replaces ``{ENV_REGEX_NS}`` in asset configs before it
    # builds sensor views.  Compare the fully expanded config value, rather
    # than incorrectly requiring the pre-scene template to survive live.
    expected_configured_filters = tuple(path.format(ENV_REGEX_NS=env_regex_ns) for path in filters)
    configured_filters = getattr(getattr(sensor, "cfg", None), "filter_prim_paths_expr", None)
    if (not isinstance(configured_filters, (list, tuple))
            or tuple(configured_filters) != expected_configured_filters):
        raise Dt1RunnerError(
            f"{color} target ContactSensor config no longer has the ordered exact 19-body filter list "
            "after InteractiveScene namespace expansion"
        )
    contact_view = getattr(sensor, "contact_physx_view", None)
    body_view = getattr(sensor, "body_physx_view", None)
    if contact_view is None or body_view is None:
        raise Dt1RunnerError(f"{color} target contact sensor has no live PhysX views")
    sensor_count = getattr(contact_view, "sensor_count", None)
    filter_count = getattr(contact_view, "filter_count", None)
    body_count = getattr(body_view, "count", None)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (sensor_count, filter_count, body_count)):
        raise Dt1RunnerError(f"{color} target contact view reports non-integer body/filter counts")
    if sensor_count != 1 or body_count != 1 or filter_count != len(filters):
        raise Dt1RunnerError(
            f"{color} target contact view must be sensor_count=1, body_count=1, "
            f"filter_count={len(filters)}; got {sensor_count},{body_count},{filter_count}"
        )
    paths = getattr(contact_view, "sensor_paths", None)
    if not isinstance(paths, (list, tuple)) or len(paths) != 1 or not isinstance(paths[0], str):
        raise Dt1RunnerError(f"{color} target contact view has no unique live sensor path")
    if not paths[0].rstrip("/").endswith(f"/{color}_target"):
        raise Dt1RunnerError(f"{color} target contact view is bound to unexpected prim {paths[0]!r}")
    environment_prefix = paths[0].rsplit("/", 1)[0]
    if "/env_" not in environment_prefix:
        raise Dt1RunnerError(f"{color} target contact path does not identify one concrete environment")
    expanded_filters = [f"{environment_prefix}/Robot/{body}" for body in GO2_CONTACT_BODY_NAMES]
    if len(set(expanded_filters)) != len(expanded_filters):
        raise Dt1RunnerError(f"{color} target contact filters do not expand to unique same-environment prim paths")
    try:
        backend_matrix = contact_view.get_contact_force_matrix(physics_dt_s)
    except BaseException as exc:
        raise Dt1RunnerError(f"{color} target backend contact-matrix query failed: {type(exc).__name__}: {exc}") from exc
    _contact_tensor_shape(backend_matrix, (1, len(filters), 3), f"{color} target backend contact matrix")
    _require_finite_contact_tensor(backend_matrix, f"{color} target backend contact matrix")
    matrix = _target_contact_matrix(sensor, color=color)
    return {
        "color": color,
        "sensor_paths": list(paths),
        "sensor_count": sensor_count,
        "body_count": body_count,
        "filter_count": filter_count,
        "filter_path_templates": list(filters),
        "configured_filter_paths": list(configured_filters),
        "expanded_filter_paths": expanded_filters,
        "backend_matrix_shape": list(backend_matrix.shape),
        "force_matrix_shape": list(matrix.shape),
        "backend_query_succeeded": True,
    }


def validate_live_target_contact_sensors(base_env: Any) -> dict[str, object]:
    """Validate both physical box contact views before a DT1 action can run."""
    try:
        red_sensor = base_env.scene["red_target_contacts"]
        blue_sensor = base_env.scene["blue_target_contacts"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise Dt1RunnerError("DT1 physical target contact sensors are unavailable") from exc
    try:
        live_body_names = list(base_env.scene["robot"].body_names)
    except (KeyError, TypeError, AttributeError) as exc:
        raise Dt1RunnerError("live Go2 robot body names are unavailable for contact validation") from exc
    if (len(live_body_names) != len(GO2_CONTACT_BODY_NAMES)
            or any(not isinstance(name, str) for name in live_body_names)
            or set(live_body_names) != set(GO2_CONTACT_BODY_NAMES)):
        raise Dt1RunnerError("live Go2 body-name set does not equal the audited 19 rigid bodies")
    env_regex_ns = getattr(base_env.scene, "env_regex_ns", None)
    if not isinstance(env_regex_ns, str) or not env_regex_ns:
        raise Dt1RunnerError("live scene does not expose its environment regex namespace")
    return {
        "format": "go2-dual-target-dt1-contact-view-contract-v1",
        "expected_robot_body_count": len(GO2_CONTACT_BODY_NAMES),
        "expected_robot_body_names": list(GO2_CONTACT_BODY_NAMES),
        "live_robot_body_names": live_body_names,
        "live_environment_regex_ns": env_regex_ns,
        "sensors": {
            "red": validate_live_target_contact_sensor(red_sensor, color="red", env_regex_ns=env_regex_ns),
            "blue": validate_live_target_contact_sensor(blue_sensor, color="blue", env_regex_ns=env_regex_ns),
        },
    }


def _target_contact_force(sensor: Any, *, color: str) -> float:
    """Read one target's strict filtered force matrix at its physical substep."""
    matrix = _target_contact_matrix(sensor, color=color)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Isaac host dependency
        raise Dt1RunnerError("torch is required to inspect target-contact force") from exc
    value = float(torch.linalg.vector_norm(matrix[0], dim=-1).max().detach().cpu().item())
    if not math.isfinite(value) or value < 0.0:
        raise Dt1RunnerError(f"{color} target contact sensor supplied invalid force telemetry")
    return value


def target_contact_body_norms(sensor: Any, *, color: str) -> dict[str, float]:
    """Return the current physical force norm for every audited robot body."""
    matrix = _target_contact_matrix(sensor, color=color)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Isaac host dependency
        raise Dt1RunnerError("torch is required to inspect per-body target-contact force") from exc
    norms = torch.linalg.vector_norm(matrix[0, 0], dim=-1).detach().cpu().tolist()
    if not isinstance(norms, list) or len(norms) != len(GO2_CONTACT_BODY_NAMES):
        raise Dt1RunnerError(f"{color} target contact body-norm vector has wrong length")
    result: dict[str, float] = {}
    for body, raw in zip(GO2_CONTACT_BODY_NAMES, norms):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) < 0.0:
            raise Dt1RunnerError(f"{color} target contact body-norm telemetry is invalid")
        result[body] = float(raw)
    return result


class LowLevelVelocityAdapter:
    """Minimal replacement for legacy VLNEnvWrapper velocity injection.

    The only control path is the provided `[vx, 0, wz]` command.  In
    particular, this class has no planner/reference path/goal state and does
    not request any pose-derived command from the base environment.
    """

    def __init__(self, history_env: Any, low_level_policy: Callable[[Any], Any], *, physics_dt_s: float, decimation: int) -> None:
        self.history_env = history_env
        self.low_level_policy = low_level_policy
        self.physics_dt_s = physics_dt_s
        self.decimation = decimation
        self.low_level_obs: Any | None = None
        self._initial_manager_to_sim_offset: int | None = None

    @property
    def base_env(self) -> Any:
        return self.history_env.unwrapped

    def reset(self) -> tuple[Any, dict[str, Any]]:
        low_level_obs, info = self.history_env.reset()
        self.low_level_obs = low_level_obs
        self._assert_live_shape()
        clock = self.live_clock()
        self._initial_manager_to_sim_offset = clock.sim_step - clock.manager_step
        return low_level_obs, info

    @property
    def physics_step(self) -> int:
        """Read the verified physical substep index, never a local estimate."""
        return self.live_clock().sim_step

    def live_clock(self) -> LiveClock:
        return _live_clock(self.base_env)

    @property
    def sim_time_s(self) -> float:
        return self.live_clock().sim_time_s

    def _assert_clock_offset(self, clock: LiveClock) -> None:
        if self._initial_manager_to_sim_offset is None:
            raise Dt1RunnerError("adapter must freeze manager/simulator clock offset at reset")
        if clock.sim_step - clock.manager_step != self._initial_manager_to_sim_offset:
            raise Dt1RunnerError("manager/simulator step offset changed during a live DT1 episode")

    def _assert_live_shape(self) -> None:
        if self.low_level_obs is None:
            raise Dt1RunnerError("low-level observation is absent")
        buf = self.history_env.proprio_obs_buf
        if tuple(buf.shape[1:]) != (HISTORY_LENGTH, PROPRIO_DIM):
            raise LowLevelContractError(f"expected live proprio history [1,9,45], got {tuple(buf.shape)}")
        if self.low_level_obs.shape[0] != 1:
            raise LowLevelContractError("DT1 only permits one audited environment at a time")
        expected_base = int(self.low_level_obs.shape[1]) - HISTORY_LENGTH * PROPRIO_DIM
        if expected_base < 9:
            raise LowLevelContractError("live old low-level policy observation is smaller than expected")

    def step(self, command: Sequence[object], latch: SubstepContactLatch) -> tuple[AppliedAction, Any, Any, Any, dict[str, Any], list[bool], dict[str, float]]:
        if self.low_level_obs is None:
            raise Dt1RunnerError("adapter must be reset before a command is stepped")
        applied = apply_deployment_bounds(command)
        clock_before = self.live_clock()
        self._assert_clock_offset(clock_before)
        latch.begin_env_step()
        self.low_level_obs = synchronized_command_observation(
            self.low_level_obs, self.history_env.proprio_obs_buf, applied.applied
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - Isaac host dependency
            raise Dt1RunnerError("torch is required to execute old low-level policy") from exc
        with torch.inference_mode():
            low_level_action = self.low_level_policy(self.low_level_obs)
        low_level_obs, reward, done, info = self.history_env.step(low_level_action)
        self.low_level_obs = low_level_obs
        clock_after = self.live_clock()
        self._assert_clock_offset(clock_after)
        if (clock_after.sim_step - clock_before.sim_step != self.decimation
                or clock_after.manager_step - clock_before.manager_step != self.decimation):
            raise Dt1RunnerError(
                "manager and simulator counters must each advance by exactly the configured decimation"
            )
        if not math.isclose(clock_after.sim_time_s - clock_before.sim_time_s, self.decimation * self.physics_dt_s,
                              rel_tol=0.0, abs_tol=1e-6):
            raise Dt1RunnerError("SimulationContext physical time did not advance by exactly four substeps")
        hits, forces = latch.consume_env_step(expected_substeps=self.decimation)
        return applied, reward, done, low_level_action, info, hits, forces


def _default_runtime_snapshot(base_env: Any) -> dict[str, Any]:
    robot = base_env.scene["robot"]
    pose = _as_vector(robot.data.root_state_w[0, :7], 7, "robot root pose")
    linear = _as_vector(robot.data.root_lin_vel_b[0], 3, "robot body linear velocity")
    angular = _as_vector(robot.data.root_ang_vel_b[0], 3, "robot body angular velocity")
    return {
        "robot_pose_w": pose,
        "body_velocity_body": [linear[0], linear[1], angular[2]],
        "terminated": _bool_at_zero(base_env.reset_terminated, "reset_terminated"),
        "truncated": _bool_at_zero(base_env.reset_time_outs, "reset_time_outs"),
    }


def _robot_pose_w(base_env: Any) -> list[float]:
    """Read root pose without touching step-created termination buffers.

    Camera capture immediately after ``reset`` is valid DT1 evidence, while
    ``ManagerBasedRLEnv.reset_terminated`` is created only after its first
    environment step on the audited runtime.  Scored post-step snapshots keep
    using :func:`_default_runtime_snapshot`; this narrow helper is solely for
    render/pause invariance checks where terminal state is neither available
    nor relevant.
    """
    try:
        return _as_vector(base_env.scene["robot"].data.root_state_w[0, :7], 7, "robot root pose")
    except (KeyError, TypeError, AttributeError) as exc:
        raise Dt1RunnerError("robot root pose is unavailable for render audit") from exc


def _target_poses_w(base_env: Any) -> dict[str, list[float]]:
    """Read both physical box poses for reset audit only."""
    result: dict[str, list[float]] = {}
    for color in ("red", "blue"):
        try:
            target = base_env.scene[f"{color}_target"]
            result[color] = _as_vector(target.data.root_state_w[0, :7], 7, f"{color} target pose")
        except (KeyError, TypeError, AttributeError) as exc:
            raise Dt1RunnerError(f"{color} target pose is unavailable for paired-reset audit") from exc
    return result


def _paired_reset_physical_state(base_env: Any, history_env: Any) -> dict[str, object]:
    """Snapshot all post-reset physical and low-level history state once.

    This stays on the evaluator/audit path.  It is intentionally taken before
    warmup and never included in a learner request or the old low-level
    observation write path.
    """
    try:
        robot = base_env.scene["robot"]
        history = history_env.proprio_obs_buf
        history_shape = tuple(history.shape)
        if history_shape != (1, HISTORY_LENGTH, PROPRIO_DIM):
            raise Dt1RunnerError(
                f"paired-reset history must be [1,{HISTORY_LENGTH},{PROPRIO_DIM}], got {history_shape}"
            )
        history_rows = [
            _as_vector(history[0, index], PROPRIO_DIM, f"paired reset proprio history[{index}]")
            for index in range(HISTORY_LENGTH)
        ]
        return {
            "robot_root_pose_w": _as_vector(robot.data.root_state_w[0, :7], 7, "paired reset robot root pose"),
            "robot_root_lin_vel_b": _as_vector(robot.data.root_lin_vel_b[0], 3, "paired reset robot linear velocity"),
            "robot_root_ang_vel_b": _as_vector(robot.data.root_ang_vel_b[0], 3, "paired reset robot angular velocity"),
            "joint_pos": _as_vector(robot.data.joint_pos[0], 12, "paired reset joint positions"),
            "joint_vel": _as_vector(robot.data.joint_vel[0], 12, "paired reset joint velocities"),
            "proprio_history": history_rows,
            "target_poses_w": _target_poses_w(base_env),
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise Dt1RunnerError("live robot reset state is unavailable for paired-reset audit") from exc


def _write_paired_reset_audit(
    episode_root: Path,
    *,
    args: argparse.Namespace,
    reset_physical_state: dict[str, object],
    reset_observation: Mapping[str, Any],
    learner_physical_state: dict[str, object],
    learner_observation: Mapping[str, Any],
    scene_metadata: Mapping[str, object],
) -> tuple[Path, str]:
    """Write the exact first-frame pair sidecar, refusing a partial record."""
    try:
        audit = ResetAudit(
            schema_version="go2-dual-target-dt1-paired-reset-audit-v1",
            geometry_group_id=args.group_id,
            color_configuration=args.color_configuration,
            repeat=args.repeat,
            target_color=args.target_color,
            scene_metadata_sha256=hashlib.sha256(json.dumps(
                scene_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            pair_seed_algorithm=PAIR_SEED_ALGORITHM,
            pair_seed=args.seed,
            physics_step_after_reset=int(reset_observation["reset_physics_step"]),
            sim_time_after_reset_s=float(reset_observation["reset_sim_time_s"]),
            robot_root_pose_w=tuple(reset_physical_state["robot_root_pose_w"]),  # type: ignore[arg-type]
            robot_root_lin_vel_b=tuple(reset_physical_state["robot_root_lin_vel_b"]),  # type: ignore[arg-type]
            robot_root_ang_vel_b=tuple(reset_physical_state["robot_root_ang_vel_b"]),  # type: ignore[arg-type]
            joint_pos=tuple(reset_physical_state["joint_pos"]),  # type: ignore[arg-type]
            joint_vel=tuple(reset_physical_state["joint_vel"]),  # type: ignore[arg-type]
            proprio_history=tuple(tuple(row) for row in reset_physical_state["proprio_history"]),  # type: ignore[arg-type]
            target_poses_w=tuple((color, tuple(pose)) for color, pose in reset_physical_state["target_poses_w"].items()),  # type: ignore[union-attr]
            first_observation_seq=int(reset_observation["observation_seq"]),
            first_render_request_seq=int(reset_observation["render_request_seq"]),
            first_camera_sensor_frame=int(reset_observation["camera_sensor_frame"]),
            first_rgb_path=str(reset_observation["rgb_path"]),
            first_rgb_sha256=sha256(episode_root / str(reset_observation["rgb_path"])),
            render_physics_step_before=int(reset_observation["render_physics_step_before"]),
            render_physics_step_after=int(reset_observation["render_physics_step_after"]),
            render_sim_time_before_s=float(reset_observation["render_sim_time_before_s"]),
            render_sim_time_after_s=float(reset_observation["render_sim_time_after_s"]),
            render_did_not_advance_physics=reset_observation["render_did_not_advance_physics"],
            learner_start_physics_step=int(learner_observation["learner_start_physics_step"]),
            learner_start_sim_time_s=float(learner_observation["learner_start_sim_time_s"]),
            learner_start_robot_root_pose_w=tuple(learner_physical_state["robot_pose_w"]),  # type: ignore[arg-type]
            learner_start_body_velocity_body=tuple(learner_physical_state["body_velocity_body"]),  # type: ignore[arg-type]
            learner_first_observation_seq=int(learner_observation["observation_seq"]),
            learner_first_render_request_seq=int(learner_observation["render_request_seq"]),
            learner_first_camera_sensor_frame=int(learner_observation["camera_sensor_frame"]),
            learner_first_rgb_path=str(learner_observation["rgb_path"]),
            learner_first_rgb_sha256=sha256(episode_root / str(learner_observation["rgb_path"])),
            learner_render_physics_step_before=int(learner_observation["render_physics_step_before"]),
            learner_render_physics_step_after=int(learner_observation["render_physics_step_after"]),
            learner_render_sim_time_before_s=float(learner_observation["render_sim_time_before_s"]),
            learner_render_sim_time_after_s=float(learner_observation["render_sim_time_after_s"]),
            learner_render_did_not_advance_physics=learner_observation["render_did_not_advance_physics"],
        )
        payload = audit.as_dict()
    except (KeyError, TypeError, ValueError, ResetAuditError) as exc:
        raise Dt1RunnerError(f"paired-reset audit is invalid: {exc}") from exc
    destination = episode_root / "paired_reset_audit.json"
    if destination.exists():
        raise Dt1RunnerError(f"refusing to overwrite paired-reset audit: {destination}")
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination, sha256(destination)


def _inside(region_center: Vec2, radius: float, pose: Sequence[float]) -> bool:
    return math.hypot(float(pose[0]) - region_center.x, float(pose[1]) - region_center.y) <= radius


def _fallen_from_pose(pose: Sequence[float], *, min_root_height_m: float = 0.18, max_abs_tilt_rad: float = 0.85) -> bool:
    """Conservative physical fall signal, independent of score regions."""
    if len(pose) != 7:
        raise Dt1RunnerError("cannot derive fall state from malformed root pose")
    if float(pose[2]) < min_root_height_m:
        return True
    w, x, y, z = (float(value) for value in pose[3:])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_argument)
    return abs(roll) > max_abs_tilt_rad or abs(pitch) > max_abs_tilt_rad


def _same_pose(before: Sequence[float], after: Sequence[float]) -> bool:
    return len(before) == len(after) == 7 and all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(before, after)
    )


def _render_without_physics(base_env: Any) -> tuple[LiveClock, LiveClock, list[float], list[float]]:
    """Render one camera update and prove it did not advance physical state."""
    before_clock = _live_clock(base_env)
    before_pose = _robot_pose_w(base_env)
    base_env.sim.render()
    after_clock = _live_clock(base_env)
    after_pose = _robot_pose_w(base_env)
    if after_clock != before_clock or not _same_pose(before_pose, after_pose):
        raise Dt1RunnerError("explicit render advanced physics clock or robot pose")
    return before_clock, after_clock, before_pose, after_pose


def verify_pause_does_not_advance_physics(base_env: Any, *, duration_s: float = 1.0) -> dict[str, object]:
    """Wall-clock pause probe: neither Kit update nor physics stepping is issued."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("pause duration must be positive and finite")
    before_clock = _live_clock(base_env)
    before_pose = _robot_pose_w(base_env)
    wall_started = time.monotonic()
    time.sleep(duration_s)
    wall_elapsed = time.monotonic() - wall_started
    after_clock = _live_clock(base_env)
    after_pose = _robot_pose_w(base_env)
    if after_clock != before_clock or not _same_pose(before_pose, after_pose):
        raise Dt1RunnerError("wall-clock pause advanced physics clock or robot pose")
    return {
        "requested_wall_pause_s": duration_s, "observed_wall_pause_s": wall_elapsed,
        "physics_step_before": before_clock.sim_step, "physics_step_after": after_clock.sim_step,
        "sim_time_before_s": before_clock.sim_time_s, "sim_time_after_s": after_clock.sim_time_s,
        "pose_before": before_pose, "pose_after": after_pose, "physics_advanced": False,
    }


def _scene_for_task(group_id: str, configuration: str, target_color: str) -> tuple[DualTargetSceneSpec, TaskSpec]:
    for scene, task, _ in dt1_task_matrix(repeats=1):
        if (scene.group.geometry_group_id, scene.color_configuration, task.target_color) == (group_id, configuration, target_color):
            return scene, task
    raise Dt1RunnerError("requested DT1 task is not in the frozen development task matrix")


def _configure_paired_expert_seed(args: argparse.Namespace) -> None:
    """Bind one red/blue task pair to a color-independent reset seed."""
    if not args.paired_reset:
        if args.seed is None:
            args.seed = 3401
        return
    if args.seed is not None:
        raise Dt1RunnerError("--paired-reset derives its own seed; --seed is prohibited")
    try:
        candidates = [
            pair for pair in build_dt1_expert_pairs()
            if (pair.scene.group.geometry_group_id, pair.scene.color_configuration, pair.repeat)
            == (args.group_id, args.color_configuration, args.repeat)
        ]
    except (ExpertMatrixError, ResetAuditError) as exc:
        raise Dt1RunnerError(f"paired expert schedule is invalid: {exc}") from exc
    if len(candidates) != 1:
        raise Dt1RunnerError("paired reset request is outside the frozen 2×4×2 DT1 matrix")
    pair = candidates[0]
    expected_task = pair.red_task if args.target_color == "red" else pair.blue_task
    if expected_task.target_color != args.target_color:
        raise Dt1RunnerError("paired target color is not in the frozen matrix")
    args.seed = pair.pair_seed


def _assert_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise Dt1RunnerError(f"old low-level checkpoint is absent: {path}")
    actual = sha256(path)
    if path == DEFAULT_CHECKPOINT and actual != DEFAULT_CHECKPOINT_SHA256:
        raise Dt1RunnerError(f"default checkpoint hash mismatch: expected {DEFAULT_CHECKPOINT_SHA256}, got {actual}")
    return actual


def _live_imports() -> dict[str, Any]:
    """Import live stack only after AppLauncher started a process."""
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from omni.isaac.lab.envs import ManagerBasedEnv
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        from omni.isaac.vlnce.config.go2.go2_matterport_vision_cfg import Go2VisionRoughPPORunnerCfg
        from omni.isaac.vlnce.utils import RslRlVecEnvHistoryWrapper
    except ImportError as exc:  # pragma: no cover - Isaac host dependency
        raise Dt1RunnerError("NaVILA IsaacLab extensions are not importable in this process") from exc
    return {
        "gym": gym, "torch": torch, "OnPolicyRunner": OnPolicyRunner, "parse_env_cfg": parse_env_cfg,
        "Go2VisionRoughPPORunnerCfg": Go2VisionRoughPPORunnerCfg,
        "RslRlVecEnvHistoryWrapper": RslRlVecEnvHistoryWrapper, "ManagerBasedEnv": ManagerBasedEnv,
    }


@dataclass
class LiveLowLevelRuntime:
    """One constructed Isaac env using the frozen Go2 low-level contract."""

    checkpoint: Path
    checkpoint_hash: str
    scene_metadata: dict[str, object]
    history_env: Any
    adapter: LowLevelVelocityAdapter
    policy_history_length: int


def build_live_low_level_runtime(args: argparse.Namespace, scene: DualTargetSceneSpec) -> LiveLowLevelRuntime:
    """Construct the only permitted old-low-level adapter for DT1 live probes."""
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = _assert_checkpoint(checkpoint)
    modules = _live_imports()
    gym, torch = modules["gym"], modules["torch"]
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    modules["ManagerBasedEnv"].seed(args.seed)
    env_cfg = modules["parse_env_cfg"]("go2_matterport_vision", num_envs=1)
    # Pairing must seed the environment configuration before construction, not
    # merely torch after a world exists.  This is recorded alongside the raw
    # post-reset state and re-applied on the constructed environment below.
    env_cfg.seed = args.seed
    scene_metadata = apply_flat_dual_target_scene(env_cfg, scene, go2_usd_path=args.go2_usd)
    if not math.isclose(float(env_cfg.sim.dt), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1e-12) or env_cfg.decimation != DECIMATION:
        raise Dt1RunnerError("live environment no longer has frozen 0.005s/decimation4 timing")
    raw_env = gym.make("go2_matterport_vision", cfg=env_cfg, render_mode=None)
    try:
        reset_seed = raw_env.unwrapped.seed(args.seed)
    except (AttributeError, TypeError, ValueError) as exc:
        raise Dt1RunnerError("constructed Go2 environment refused the audited reset seed") from exc
    if isinstance(reset_seed, bool) or not isinstance(reset_seed, int) or reset_seed < 0:
        raise Dt1RunnerError("constructed Go2 environment returned an invalid reset seed")
    if reset_seed != args.seed:
        raise Dt1RunnerError("constructed Go2 environment did not retain the paired reset seed")
    history_env = modules["RslRlVecEnvHistoryWrapper"](raw_env, history_length=HISTORY_LENGTH)
    agent_cfg = modules["Go2VisionRoughPPORunnerCfg"]()
    agent_cfg.device = args.low_level_device
    if not hasattr(agent_cfg, "policy"):
        raise Dt1RunnerError("old Go2 RSL-RL policy config is unavailable")
    # The legacy confirmed launch path set this config field in addition to
    # constructing a 9-frame wrapper.  Keep both mechanisms in lockstep.
    agent_cfg.policy.history_length = HISTORY_LENGTH
    if agent_cfg.policy.history_length != HISTORY_LENGTH:
        raise Dt1RunnerError("old Go2 RSL-RL policy config refused history_length=9")
    policy_runner = modules["OnPolicyRunner"](history_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    policy_runner.load(str(checkpoint))
    low_level_policy = policy_runner.get_inference_policy(device=history_env.unwrapped.device)
    adapter = LowLevelVelocityAdapter(history_env, low_level_policy, physics_dt_s=PHYSICS_DT_S, decimation=DECIMATION)
    return LiveLowLevelRuntime(
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        scene_metadata=scene_metadata,
        history_env=history_env,
        adapter=adapter,
        policy_history_length=agent_cfg.policy.history_length,
    )


def run_live_expert(
    args: argparse.Namespace,
    simulation_app: Any,
    *,
    resource_monitor: SharedSmokeResourceMonitor | None = None,
    scene_task: tuple[DualTargetSceneSpec, Any] | None = None,
) -> dict[str, object]:
    """Run one true-physics expert episode.  Called only after GPU admission."""
    scene, task = scene_task if scene_task is not None else _scene_for_task(args.group_id, args.color_configuration, args.target_color)
    live_runtime = build_live_low_level_runtime(args, scene)
    checkpoint = live_runtime.checkpoint
    checkpoint_hash = live_runtime.checkpoint_hash
    scene_metadata = live_runtime.scene_metadata
    history_env = live_runtime.history_env
    adapter = live_runtime.adapter
    episode_id = f"{scene.group.geometry_group_id}_{scene.color_configuration}_{task.target_color}_r{args.repeat:02d}"
    episode_root = Path(args.output_dir).resolve() / args.run_id / "episodes" / episode_id
    calibration: MotionCalibration | None = None
    calibration_hash = ""
    calibration_path = getattr(args, "motion_calibration", None)
    if calibration_path is not None:
        calibration, calibration_hash = load_motion_calibration(calibration_path)
        thresholds = ParkingThresholds(
            body_linear_speed_mps=calibration.derived_body_planar_speed_mps,
            body_yaw_rate_radps=calibration.derived_body_yaw_rate_radps,
        )
    else:
        thresholds = ParkingThresholds()
    expert = ParkingExpert()
    writer = EpisodeEvidenceWriter(episode_root, {
        **scene_metadata,
        "episode_id": episode_id,
        "seed": args.seed,
        "paired_reset_required": bool(args.paired_reset),
        "pair_seed_algorithm": PAIR_SEED_ALGORITHM if args.paired_reset else "",
        "repeat": args.repeat,
        "target_color": task.target_color,
        "target_slot": task.target_slot,
        "instruction": task.instruction,
        "physics_dt_s": PHYSICS_DT_S,
        "decimation": DECIMATION,
        "thresholds": asdict(thresholds),
        "motion_calibration": calibration.as_dict() if calibration is not None else None,
        "motion_calibration_path": str(Path(calibration_path).resolve()) if calibration_path is not None else "",
        "motion_calibration_sha256": calibration_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "agent_policy_history_length": live_runtime.policy_history_length,
        "policy_phase": "expert",
        "learner_inputs": ["RGB", "body_vx", "body_vy", "body_yaw_rate", "task"],
        "truth_is_audit_or_expert_only": True,
        "seed_protocol": "python_random,numpy,torch,ManagerBasedEnv.seed set before gym.make/reset",
        "physics_clock_source": "ManagerBasedRLEnv._sim_step_counter verified to increment inside each physics substep",
        "fall_detection": {"min_root_height_m": 0.18, "max_abs_roll_or_pitch_rad": 0.85, "env_terminated_is_fall": True},
        "expert_terminal_heading": {
            "enabled": True,
            "definition": "face the fixed target-box front from the parking centre before autonomous hold",
            "tolerance_rad": expert.gains.terminal_heading_tolerance_rad,
        },
    })
    base_env = adapter.base_env
    latch = SubstepContactLatch(threshold_n=args.contact_threshold_n)
    snapshot_capture = PreResetEvidenceCapture(
        base_env,
        snapshot=lambda env: _default_runtime_snapshot(env),
        on_substep=lambda: latch.capture({
            "red": _target_contact_force(base_env.scene["red_target_contacts"], color="red"),
            "blue": _target_contact_force(base_env.scene["blue_target_contacts"], color="blue"),
        }),
    )
    scorer = AutonomousParkingScorer(thresholds)
    wrong_target_scorer = AutonomousParkingScorer(thresholds)
    render_request_seq = observation_seq = 0
    pre_reset_index = 0
    paired_reset_audit_path = ""
    paired_reset_audit_hash = ""
    result: dict[str, object] = {"episode_id": episode_id, "status": "FAILED", "reason": "not_started"}
    try:
        if resource_monitor is not None:
            resource_monitor.check()
        adapter.reset()
        contact_view_contract = validate_live_target_contact_sensors(base_env)
        (episode_root / "target_contact_view_contract.json").write_text(
            json.dumps(contact_view_contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        initial_clock = adapter.live_clock()
        (episode_root / "runtime_clock_initial.json").write_text(json.dumps({
            "manager_sim_step": initial_clock.manager_step, "simulation_context_step": initial_clock.sim_step,
            "simulation_context_time_s": initial_clock.sim_time_s,
            "frozen_manager_to_sim_step_offset": initial_clock.sim_step - initial_clock.manager_step,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        camera = base_env.scene["rgbd_camera"]

        def capture_rgb(kind: str) -> dict[str, Any]:
            """Request/render one audit observation without advancing physics."""
            nonlocal render_request_seq, observation_seq
            before_clock, after_clock, _, _ = _render_without_physics(base_env)
            render_request_seq += 1
            rgb = _camera_rgb(camera)
            fields = camera_audit_fields(camera)
            if fields["camera_sensor_frame"] is None or fields["camera_timestamp_s"] is None:
                raise Dt1RunnerError("camera audit bookkeeping is unavailable")
            observation_seq += 1
            relative = f"rgb_{kind}/{observation_seq:06d}.png"
            _write_rgb(episode_root / relative, rgb)
            return {
                "observation_seq": observation_seq, "render_request_seq": render_request_seq,
                "rgb_path": relative, "render_physics_step_before": before_clock.sim_step,
                "render_physics_step_after": after_clock.sim_step,
                "render_sim_time_before_s": before_clock.sim_time_s,
                "render_sim_time_after_s": after_clock.sim_time_s,
                "render_did_not_advance_physics": True,
                **fields,
            }

        reset_state: dict[str, object] | None = None
        reset_observation: dict[str, Any] | None = None
        if args.paired_reset:
            reset_state = _paired_reset_physical_state(base_env, history_env)
            reset_observation = capture_rgb("reset")
            reset_observation["reset_physics_step"] = initial_clock.sim_step
            reset_observation["reset_sim_time_s"] = initial_clock.sim_time_s

        snapshot_capture.install()
        # Legacy Go2 warmup is evidence only.  It never reaches scorer/pre-action logs.
        for warmup_step in range(GO2_WARMUP_STEPS):
            before = adapter.physics_step
            _, _, done, _, _, hits, forces = adapter.step((0.0, 0.0, 0.0), latch)
            if resource_monitor is not None:
                resource_monitor.check()
            if any(hits) or _bool_at_zero(done, "warmup_done") or snapshot_capture.records:
                raise Dt1RunnerError("target contact or auto-reset occurred during warmup")
            writer.warmup({
                "event": "warmup", "warmup_step": warmup_step,
                "sim_time_s": adapter.sim_time_s,
                "physics_step": adapter.physics_step, "command": [0.0, 0.0, 0.0],
            })
        if snapshot_capture.records:
            raise Dt1RunnerError("environment auto-reset during warmup; no scored episode may continue")
        pause_probe = verify_pause_does_not_advance_physics(base_env, duration_s=1.0)
        (episode_root / "pause_no_physics.json").write_text(
            json.dumps(pause_probe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # The initial observation is the RGB/state directly before its action.
        # Each following pre-action record reuses the immediately post-action
        # observation from the preceding action; no fake time or camera change
        # is invented between them.
        observation = capture_rgb("pre")
        if args.paired_reset:
            if reset_state is None or reset_observation is None:
                raise Dt1RunnerError("paired-reset audit lost its reset state before learner start")
            learner_state = _default_runtime_snapshot(base_env)
            observation["learner_start_physics_step"] = adapter.physics_step
            observation["learner_start_sim_time_s"] = adapter.sim_time_s
            audit_path, audit_hash = _write_paired_reset_audit(
                episode_root,
                args=args,
                reset_physical_state=reset_state,
                reset_observation=reset_observation,
                learner_physical_state=learner_state,
                learner_observation=observation,
                scene_metadata=scene_metadata,
            )
            paired_reset_audit_path = str(audit_path)
            paired_reset_audit_hash = audit_hash
        for step_index in range(args.max_env_steps):
            if resource_monitor is not None:
                resource_monitor.check()
            before_snapshot = _default_runtime_snapshot(base_env)
            pose_before = before_snapshot["robot_pose_w"]
            center = task.parking_region.center
            target_slot = scene.slot_for_color(task.target_color)
            terminal_heading = math.atan2(-target_slot.front_unit.y, -target_slot.front_unit.x)
            command = expert.command(
                robot_xy=Vec2(float(pose_before[0]), float(pose_before[1])),
                robot_yaw_rad=_yaw_from_wxyz(pose_before[3:]), parking_center=center,
                terminal_heading_rad=terminal_heading,
            )
            physics_before = adapter.physics_step
            time_before = adapter.sim_time_s
            body_before = before_snapshot["body_velocity_body"]
            writer.pre_action({
                "event": "pre_action", "phase": "expert", "sim_time_s": time_before,
                "physics_step": physics_before, "observation_seq": observation["observation_seq"],
                "camera_sensor_frame": observation["camera_sensor_frame"],
                "camera_timestamp_s": observation["camera_timestamp_s"],
                "render_request_seq": observation["render_request_seq"], "rgb_path": observation["rgb_path"],
                "body_velocity_body": body_before, "raw_action": command.as_list(),
                "applied_action": list(apply_deployment_bounds(command.as_list()).applied), "task": task.instruction,
            })
            applied, _, done, _, _, hits, forces = adapter.step(command.as_list(), latch)
            if resource_monitor is not None:
                resource_monitor.check()
            post_snapshot = _default_runtime_snapshot(base_env)
            pre_reset_ref = ""
            if snapshot_capture.records:
                # ManagerBasedRLEnv has already reset before its public step
                # returns.  Persist the hook's pre-reset values and use them
                # for this terminal post-step instead of reset pose/velocity.
                snapshot = snapshot_capture.records.pop(0)
                pre_reset_ref = f"pre_reset_{pre_reset_index:04d}.json"
                (episode_root / pre_reset_ref).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                pre_reset_index += 1
                post_snapshot = snapshot
            # This is a genuinely post-action renderer request.  Its RGB is
            # retained even for terminal failures, but never used as the
            # current action's learner input.
            post_observation = capture_rgb("post")
            correct = _inside(task.parking_region.center, scene.group.parking_radius_m, post_snapshot["robot_pose_w"])
            other_slot = "B" if task.target_slot == "A" else "A"
            other = _inside(scene.group.parking_region(other_slot).center, scene.group.parking_radius_m, post_snapshot["robot_pose_w"])
            body = post_snapshot["body_velocity_body"]
            terminated = bool(post_snapshot["terminated"])
            truncated = bool(post_snapshot["truncated"])
            environment_done = _bool_at_zero(done, "low_level_done")
            fallen = _fallen_from_pose(post_snapshot["robot_pose_w"]) or terminated
            # An environment reset/timeout is terminal failure evidence before
            # the scorer sees this frame.  It therefore cannot be re-labelled
            # as success merely because a pre-reset pose happened to be parked.
            evaluator_stop = truncated or (environment_done and not fallen)
            # Wrong target is a failure only after the same real, continuous
            # one-second autonomous stop evidence used for the correct target.
            # A transient zero command near the other box cannot end a task.
            wrong_decision = wrong_target_scorer.observe(ScoreFrame(
                sim_time_s=adapter.sim_time_s, physics_step=adapter.physics_step,
                observation_seq=post_observation["observation_seq"], raw_action=applied.raw, applied_action=applied.applied,
                body_vx_mps=float(body[0]), body_vy_mps=float(body[1]), body_yaw_rate_radps=float(body[2]),
                in_correct_parking_region=other, warmup=False, collision=any(hits), fallen=fallen,
                wrong_target_stop=False, evaluator_stop=evaluator_stop,
            ))
            wrong_stop = wrong_decision.status is ScoreStatus.SUCCESS
            frame = ScoreFrame(
                sim_time_s=adapter.sim_time_s, physics_step=adapter.physics_step,
                observation_seq=post_observation["observation_seq"], raw_action=applied.raw, applied_action=applied.applied,
                body_vx_mps=float(body[0]), body_vy_mps=float(body[1]), body_yaw_rate_radps=float(body[2]),
                in_correct_parking_region=correct, warmup=False, collision=any(hits), fallen=fallen,
                wrong_target_stop=wrong_stop, evaluator_stop=evaluator_stop,
            )
            decision = scorer.observe(frame)
            writer.post_step({
                "event": "post_step", "phase": "expert", "sim_time_before_s": time_before,
                "sim_time_after_s": adapter.sim_time_s,
                "physics_step_before": physics_before, "physics_step_after": adapter.physics_step,
                "observation_seq_after": post_observation["observation_seq"],
                "camera_sensor_frame": post_observation["camera_sensor_frame"],
                "camera_timestamp_s": post_observation["camera_timestamp_s"],
                "render_request_seq": post_observation["render_request_seq"],
                "robot_pose_w": post_snapshot["robot_pose_w"], "body_velocity_body": body,
                "collision_latched_substeps": hits, "target_contact_force_max": forces,
                "in_correct_parking_region": correct, "in_other_parking_region": other, "fallen": fallen,
                "terminated": terminated, "truncated": truncated,
                "evaluator_stop": evaluator_stop, "wrong_target_stop": wrong_stop, "warmup": False,
                "pre_reset_snapshot_ref": pre_reset_ref, "scorer_status": decision.status.value, "scorer_detail": decision.detail,
            })
            if decision.terminal:
                result = {"episode_id": episode_id, "status": decision.status.value, "reason": decision.detail,
                          "steps": step_index + 1, "sim_time_s": adapter.sim_time_s}
                break
            if environment_done:
                result = {"episode_id": episode_id, "status": "FAILED_AUTO_RESET", "reason": "environment_done_before_scorer_success"}
                break
            observation = post_observation
        else:
            result = {"episode_id": episode_id, "status": "FAILED_TIMEOUT", "reason": "expert did not finish before max_env_steps"}
        if paired_reset_audit_path:
            result["paired_reset_audit_path"] = paired_reset_audit_path
            result["paired_reset_audit_sha256"] = paired_reset_audit_hash
            result["pair_seed"] = args.seed
            result["pair_seed_algorithm"] = PAIR_SEED_ALGORITHM
        return result
    finally:
        snapshot_capture.restore()
        history_env.close()


def _yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    if len(quaternion) != 4:
        raise Dt1RunnerError("robot quaternion must have wxyz order")
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _live_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="explicitly launch an Isaac DT1 expert episode")
    parser.add_argument("--shared-smoke", action="store_true",
                        help="only user-authorized DT1 50-step shared-GPU Isaac/Go2 smoke mode")
    parser.add_argument("--contact-precheck", action="store_true",
                        help="bounded physical foot/target contact diagnostic; never a navigation episode")
    parser.add_argument("--motion-precheck", action="store_true",
                        help="bounded physical forward/turn/return-zero calibration; never a navigation episode")
    parser.add_argument("--cpu-self-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("<external-data-root>"))
    parser.add_argument("--run-id", default=f"dt1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}")
    parser.add_argument("--group-id", default="dt1_dev_000", choices=[group.geometry_group_id for group in dt1_development_groups()])
    parser.add_argument("--color-configuration", default="A_red_B_blue", choices=["A_red_B_blue", "A_blue_B_red"])
    parser.add_argument("--target-color", default="red", choices=["red", "blue"])
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int,
                        help="non-paired diagnostic seed; paired expert tasks derive a color-independent seed")
    parser.add_argument("--paired-reset", action="store_true",
                        help="require the frozen red/blue paired-reset seed and write both reset/learner-start sidecars")
    parser.add_argument("--go2-usd", help="verified local Go2 USD path, applied to final env_cfg before gym.make")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--motion-calibration", type=Path,
                        help="validated DT1 zero-command calibration; required by the bounded expert launcher")
    parser.add_argument("--low-level-device", default="cuda:0")
    parser.add_argument("--contact-threshold-n", type=float, default=1.0)
    parser.add_argument("--max-env-steps", type=int, default=1500)
    parser.add_argument("--gpu-project-root", type=Path, default=Path("<external-data-root>"),
                        help="fixed project-owned GPU waiter lock/state root")
    parser.add_argument("--gpu-admission-state", type=Path,
                        help="optional explicit state path; must still match --gpu-project-root")
    return parser


def main() -> None:
    # Do not import AppLauncher for CPU self-check.  It would construct an
    # Isaac application and violate the wait-before-GPU contract.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--cpu-self-check", action="store_true")
    bootstrap.add_argument("--live", action="store_true")
    early, _ = bootstrap.parse_known_args()
    if early.cpu_self_check and not early.live:
        args = _live_parser().parse_args()
        print(json.dumps(cpu_self_check(), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not early.live:
        raise SystemExit("DT1 runner requires --cpu-self-check or explicit --live")
    # AppLauncher must be configured before importing any live Omni modules.
    from omni.isaac.lab.app import AppLauncher  # pragma: no cover - live Isaac only
    parser = _live_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.go2_usd:
        raise SystemExit("--go2-usd is required for live DT1; asset aliases are not accepted")
    if args.max_env_steps <= 0 or args.repeat < 0 or args.contact_threshold_n <= 0:
        raise SystemExit("repeat/max-env-steps/contact-threshold-n must be positive where applicable")
    if args.shared_smoke and args.max_env_steps != 50:
        raise SystemExit("--shared-smoke is restricted to exactly --max-env-steps 50")
    if args.motion_precheck and (args.shared_smoke or args.contact_precheck):
        raise SystemExit("--motion-precheck is an exclusive diagnostic and cannot combine with shared/contact modes")
    if args.motion_precheck and args.motion_calibration is not None:
        raise SystemExit("--motion-precheck derives a calibration and cannot consume one")
    if args.paired_reset and (args.motion_precheck or args.contact_precheck or args.shared_smoke):
        raise SystemExit("--paired-reset is only valid for one exclusive calibrated expert task")
    if args.paired_reset and args.motion_calibration is None:
        raise SystemExit("--paired-reset requires the validated zero-command --motion-calibration")
    try:
        _configure_paired_expert_seed(args)
    except Dt1RunnerError as exc:
        raise SystemExit(str(exc)) from exc
    if args.contact_precheck:
        if args.shared_smoke:
            raise SystemExit("--contact-precheck is an exclusive diagnostic and cannot use --shared-smoke")
        if not 1 <= args.max_env_steps <= 300:
            raise SystemExit("--contact-precheck requires --max-env-steps in 1..300")
        if not math.isclose(args.contact_threshold_n, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit("--contact-precheck requires --contact-threshold-n 1.0")
    asset_root = _asset_root_from_go2_usd(args.go2_usd)
    _configure_dt1_kit_args_before_app(asset_root)
    safe_run_id = _safe_run_id(args.run_id)
    actual_argv = [sys.executable, "-m", "src.dual_target.runner", *sys.argv[1:]]
    # This is deliberately before GPU admission and before AppLauncher: the
    # live request is non-overwriting, records its exact source/argv input,
    # and makes no graphics/CUDA allocation.  It verifies the immutable DT0
    # approval binding without trying to compare the intentionally changed
    # DT1 entrypoint against DT0's historical source hashes.
    run_root = initialize_dt1_run(args.output_dir, run_id=safe_run_id, actual_argv=actual_argv)
    admission = None
    terminal_status_recorded = False
    resource_monitor: SharedSmokeResourceMonitor | None = None
    # Do not construct AppLauncher until the waiter state and a fresh
    # nvidia-smi sample have been checked while holding the project lock.
    app = None
    try:
        if args.shared_smoke:
            resource_monitor = SharedSmokeResourceMonitor(
                run_root / "shared_smoke_gpu_samples.jsonl", owned_root_pid=os.getpid(), interval_s=1.0,
            )
            resource_monitor.start()
            resource_monitor.check()
        admission = acquire_live_admission(
            args.gpu_project_root, run_id=safe_run_id, actual_argv=actual_argv,
            state_path=args.gpu_admission_state,
            policy=SHARED_SMOKE_DT1_POLICY if args.shared_smoke else EXCLUSIVE_DT1_POLICY,
        )
        write_dt1_run_status(
            run_root, "RUNNING", run_id=safe_run_id,
            detail="fresh GPU admission consumed; AppLauncher has not yet been constructed",
            extra={"gpu_admission_receipt": admission.state},
        )
        app = AppLauncher(args)
        kit_runtime = _configure_dt1_runtime_after_app(asset_root)
        (run_root / "kit_runtime.json").write_text(
            json.dumps(kit_runtime, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if resource_monitor is not None:
            resource_monitor.check()
        if args.motion_precheck:
            # The control probe has a distinct, bounded command sequence and
            # produces calibration evidence only.  It cannot be confused with
            # a task success or an expert demonstration.
            from .motion_precheck import run_live_motion_precheck
            result = run_live_motion_precheck(args, app.app, resource_monitor=resource_monitor)
            write_dt1_partial_gate(
                run_root, run_id=safe_run_id, result=result, admission=admission.state,
                evidence_kind="physical_motion_calibration_precheck",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            write_dt1_run_status(
                run_root, "RUNNING", run_id=safe_run_id,
                detail="physical motion calibration completed; no task-contract lock, navigation success, or DT1 approval was produced",
                extra={"motion_precheck_result": result},
            )
        elif args.contact_precheck:
            # Import after AppLauncher only: this small diagnostic shares the
            # same audited low-level adapter, but has no planner or task
            # success path and deliberately cannot self-approve DT1.
            from .contact_precheck import run_live_contact_precheck
            result = run_live_contact_precheck(args, app.app, resource_monitor=resource_monitor)
            write_dt1_partial_gate(
                run_root, run_id=safe_run_id, result=result, admission=admission.state,
                evidence_kind="physical_contact_precheck",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            write_dt1_run_status(
                run_root, "RUNNING", run_id=safe_run_id,
                detail="physical contact precheck completed; no navigation success or DT1 approval was produced",
                extra={"contact_precheck_result": result},
            )
        else:
            result = run_live_expert(args, app.app, resource_monitor=resource_monitor)
            write_dt1_partial_gate(run_root, run_id=safe_run_id, result=result, admission=admission.state)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            if result["status"] != ScoreStatus.SUCCESS.value:
                write_dt1_run_status(
                    run_root, "FAILED", run_id=safe_run_id,
                    detail="live expert sub-run ended without a successful terminal score",
                    extra={"episode_result": result},
                )
                terminal_status_recorded = True
                raise SystemExit(1)
            # One expert episode is partial DT1 evidence only.  Never write a
            # review-ready or approved state from this implementation agent.
            write_dt1_run_status(
                run_root, "RUNNING", run_id=safe_run_id,
                detail="one live expert sub-run completed; full DT1 matrix and parent review remain",
                extra={"episode_result": result},
            )
    except BaseException as exc:
        # Preserve a recoverable, explicit failure rather than leaving a
        # RUNNING file after a failed admission, Isaac startup, or telemetry
        # error.  Do this before ``app.close()``: Kit teardown may otherwise
        # make a shell supervisor see exit 0 after a scene-build exception.
        # The completed non-success branch above already wrote the richer
        # result and is safely overwritten with this concise cause.
        if run_root.exists() and not terminal_status_recorded:
            failure_evidence = write_dt1_failure_traceback(run_root, exc=exc)
            write_dt1_run_status(
                run_root, "FAILED", run_id=safe_run_id,
                detail=f"{type(exc).__name__}: {exc}",
                extra=failure_evidence,
            )
        raise
    finally:
        if app is not None:
            app.app.close()
        if admission is not None:
            admission.close()
        if resource_monitor is not None:
            resource_monitor.close()


if __name__ == "__main__":
    main()
