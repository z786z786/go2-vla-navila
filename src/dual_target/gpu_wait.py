"""One real, cancellable DT1 GPU waiter with no CUDA allocation.

The waiter only samples ``nvidia-smi``.  It never starts Isaac, constructs a
CUDA tensor, or kills a process it did not create.  A project-owned POSIX lock
prevents duplicate waiters even when callers select different output folders.
``flock`` is an intra-project coordination mechanism, not a GPU reservation
against other users; a live runner must recheck the resource snapshot after the
waiter declares it eligible.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_REQUIRED_FREE_MIB = 20 * 1024
DEFAULT_REQUIRED_SAMPLES = 3
DEFAULT_INTERVAL_S = 30.0
NVIDIA_SMI_TIMEOUT_S = 10.0
SHARED_SMOKE_REQUIRED_FREE_MIB = 10 * 1024
SHARED_SMOKE_MIN_FREE_MIB_DURING_RUN = 2 * 1024


class GpuWaitError(RuntimeError):
    """Sampling or waiter-state safety failed before a live run could start."""


@dataclass(frozen=True)
class GpuAdmissionPolicy:
    """A frozen resource-admission policy, not a caller-tunable threshold."""

    policy_id: str
    required_free_mib: int
    required_samples: int = DEFAULT_REQUIRED_SAMPLES
    sample_interval_s: float = DEFAULT_INTERVAL_S
    allow_existing_compute: bool = False

    def __post_init__(self) -> None:
        if (not self.policy_id or self.required_free_mib <= 0 or self.required_samples <= 0
                or self.sample_interval_s <= 0):
            raise ValueError("GPU admission policy must have positive fixed thresholds")


EXCLUSIVE_DT1_POLICY = GpuAdmissionPolicy(
    policy_id="dt1_exclusive_v1", required_free_mib=DEFAULT_REQUIRED_FREE_MIB,
)
# This is deliberately not a general low-memory admission.  It exists only
# for the user-authorized first DT1 50-step Isaac/Go2 smoke check.  A foreign
# PID is recorded as shared-resource evidence, never treated as owned.
SHARED_SMOKE_DT1_POLICY = GpuAdmissionPolicy(
    policy_id="dt1_shared_smoke_v1", required_free_mib=SHARED_SMOKE_REQUIRED_FREE_MIB,
    allow_existing_compute=True,
)
SHARED_COEXISTENCE_DT1_POLICY = GpuAdmissionPolicy(
    policy_id="dt1_shared_coexistence_16g_v1", required_free_mib=16 * 1024,
    allow_existing_compute=True,
)


@dataclass(frozen=True)
class LiveGpuSnapshot:
    total_mib: int
    used_mib: int
    compute_pids: tuple[int, ...]
    utilization_pct: int | None = None
    reported_free_mib: int | None = None

    def validate(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.total_mib, self.used_mib)):
            raise GpuWaitError("GPU memory values must be integers")
        if self.total_mib <= 0 or self.used_mib < 0 or self.used_mib > self.total_mib:
            raise GpuWaitError("GPU memory values are out of range")
        if self.reported_free_mib is not None and (type(self.reported_free_mib) is not int
                or not 0 <= self.reported_free_mib <= self.total_mib - self.used_mib):
            raise GpuWaitError("invalid actual free memory counter")
        if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in self.compute_pids):
            raise GpuWaitError("GPU compute PIDs must be positive integers")
        if self.utilization_pct is not None and (not isinstance(self.utilization_pct, int) or not 0 <= self.utilization_pct <= 100):
            raise GpuWaitError("GPU utilization must be an integer percentage when supplied")

    @property
    def free_mib(self) -> int:
        # Drivers can reserve memory outside memory.used. Use the actual free
        # counter for live admission; older injected fixtures remain supported.
        return self.reported_free_mib if self.reported_free_mib is not None else self.total_mib - self.used_mib

    def as_dict(self) -> dict[str, object]:
        return {
            "total_mib": self.total_mib,
            "used_mib": self.used_mib,
            "free_mib": self.free_mib,
            "compute_pids": list(self.compute_pids),
            "compute_process_count": len(self.compute_pids),
            "utilization_pct": self.utilization_pct,
        }


def _int_cell(value: str, label: str) -> int:
    try:
        result = int(value.strip())
    except ValueError as exc:
        raise GpuWaitError(f"nvidia-smi supplied an invalid {label}: {value!r}") from exc
    return result


def parse_nvidia_smi_snapshot(gpu_csv: str, compute_csv: str) -> LiveGpuSnapshot:
    """Parse injected or live `nvidia-smi --format=csv,noheader,nounits` text."""
    rows = [line.strip() for line in gpu_csv.splitlines() if line.strip()]
    if len(rows) != 1:
        raise GpuWaitError("DT1 requires exactly one selected GPU snapshot")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) not in {2, 3, 4}:
        raise GpuWaitError("GPU snapshot must contain total, used, and optional utilization")
    total, used = _int_cell(fields[0], "total memory"), _int_cell(fields[1], "used memory")
    utilization = _int_cell(fields[2], "utilization") if len(fields) >= 3 else None
    free = _int_cell(fields[3], "free memory") if len(fields) == 4 else None
    if free is not None and not 0 <= free <= total - used:
        raise GpuWaitError("invalid actual free memory counter")
    pids: list[int] = []
    normalized = compute_csv.strip()
    # An empty, successful query is the only accepted representation of zero
    # compute jobs.  Text such as N/A or an error banner is unknown state, not
    # idle GPU evidence.
    if normalized:
        for line in normalized.splitlines():
            cell = line.strip().split(",", maxsplit=1)[0].strip()
            if cell:
                pids.append(_int_cell(cell, "compute PID"))
    snapshot = LiveGpuSnapshot(total, used, tuple(sorted(set(pids))), utilization, free)
    snapshot.validate()
    return snapshot


def probe_gpu() -> LiveGpuSnapshot:
    """Read current GPU state without allocating any GPU resource."""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S,
        )
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise GpuWaitError("nvidia-smi is unavailable; cannot make a DT1 GPU admission claim") from exc
    except subprocess.CalledProcessError as exc:
        raise GpuWaitError(f"nvidia-smi query failed: {(exc.stderr or exc.stdout).strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GpuWaitError(f"nvidia-smi query exceeded {NVIDIA_SMI_TIMEOUT_S:g}s") from exc
    return parse_nvidia_smi_snapshot(gpu.stdout, processes.stdout)


def eligibility(
    snapshot: LiveGpuSnapshot,
    *,
    required_free_mib: int = DEFAULT_REQUIRED_FREE_MIB,
    allow_existing_compute: bool = False,
) -> tuple[bool, str]:
    snapshot.validate()
    if snapshot.free_mib < required_free_mib:
        return False, f"free_memory_below_{required_free_mib}_MiB"
    if snapshot.compute_pids and not allow_existing_compute:
        return False, "foreign_or_unknown_compute_process_present"
    return True, "eligible_shared_compute" if snapshot.compute_pids else "eligible"


def project_gpu_lock(project_root: Path) -> Path:
    return project_root / "outputs" / "dual_target_v1" / ".dt1_gpu_wait.lock"


def default_state_path(project_root: Path) -> Path:
    return project_root / "outputs" / "dual_target_v1" / "dt1_gpu_wait_state.json"


def default_cancel_path(project_root: Path) -> Path:
    return project_root / "outputs" / "dual_target_v1" / "dt1_gpu_wait_cancel.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_age_seconds(value: object) -> float:
    if not isinstance(value, str):
        raise GpuWaitError("GPU waiter state lacks an ISO last_checked_utc timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GpuWaitError("GPU waiter last_checked_utc is not ISO-8601") from exc
    if stamp.tzinfo is None:
        raise GpuWaitError("GPU waiter last_checked_utc must include timezone")
    age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
    if age < 0.0 or age > 90.0:
        raise GpuWaitError("GPU waiter eligibility is stale or clock-skewed; run a new live waiter")
    return age


class Dt1GpuWaiter:
    """Exclusive, stateful sampler usable with an injected CPU test probe."""

    def __init__(
        self,
        *,
        project_root: Path,
        run_id: str,
        argv: Sequence[str],
        state_path: Path | None = None,
        required_free_mib: int = DEFAULT_REQUIRED_FREE_MIB,
        required_samples: int = DEFAULT_REQUIRED_SAMPLES,
        interval_s: float = DEFAULT_INTERVAL_S,
        probe: Callable[[], LiveGpuSnapshot] = probe_gpu,
        test_only: bool = False,
        policy: GpuAdmissionPolicy = EXCLUSIVE_DT1_POLICY,
    ) -> None:
        if required_free_mib <= 0 or required_samples <= 0 or interval_s <= 0:
            raise ValueError("GPU waiter thresholds and interval must be positive")
        self.project_root = project_root.resolve()
        self.run_id = run_id
        self.argv = list(argv)
        self.state_path = (state_path or default_state_path(self.project_root)).resolve()
        self.required_free_mib = required_free_mib
        self.required_samples = required_samples
        self.interval_s = interval_s
        self.probe = probe
        self.test_only = test_only
        self.policy = policy
        self._lock: Any | None = None
        self._consecutive = 0

    def acquire(self) -> None:
        lock_path = project_gpu_lock(self.project_root)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise GpuWaitError(f"another DT1 GPU waiter holds {lock_path}") from None
        self._lock = handle

    def close(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def _write_state(self, status: str, *, snapshot: LiveGpuSnapshot | None, reason: str | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": "go2-dual-target-dt1-gpu-wait-v1",
            "stage": "DT1",
            "run_id": self.run_id,
            "status": status,
            "pid": os.getpid(),
            "project_root": str(self.project_root),
            "started_at_utc": getattr(self, "_started_at", _now()),
            "last_checked_utc": _now(),
            "required_free_mib": self.required_free_mib,
            "required_consecutive_samples": self.required_samples,
            "sample_interval_s": self.interval_s,
            "admission_policy": self.policy.policy_id,
            "allow_existing_compute": self.policy.allow_existing_compute,
            "consecutive_eligible_samples": self._consecutive,
            "actual_argv": self.argv,
            "admission_source": "injected_test_only" if self.test_only else "live_nvidia_smi",
            "snapshot": snapshot.as_dict() if snapshot else None,
            "reason": reason,
            "cancel_command": [sys.executable, "-m", "src.dual_target.gpu_wait", "--cancel", "--project-root", str(self.project_root)],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)
        return payload

    def sample_once(self) -> dict[str, object]:
        if self._lock is None:
            raise GpuWaitError("GPU waiter must hold its project lock before sampling")
        snapshot = self.probe()
        ok, reason = eligibility(
            snapshot, required_free_mib=self.required_free_mib,
            allow_existing_compute=self.policy.allow_existing_compute,
        )
        self._consecutive = self._consecutive + 1 if ok else 0
        status = "GPU_READY_RECHECK_REQUIRED" if self._consecutive >= self.required_samples else "WAITING_GPU"
        return self._write_state(status, snapshot=snapshot, reason=reason)

    def _cancel_requested(self) -> bool:
        path = default_cancel_path(self.project_root)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GpuWaitError("DT1 cancellation request is unreadable") from exc
        if payload.get("project_root") != str(self.project_root) or payload.get("run_id") != self.run_id:
            raise GpuWaitError("DT1 cancellation request does not match the held project waiter")
        path.unlink()
        return True

    def _locked_recheck(self) -> dict[str, object]:
        """One new real sample while the exclusive project lock is still held."""
        snapshot = self.probe()
        ok, reason = eligibility(
            snapshot, required_free_mib=self.required_free_mib,
            allow_existing_compute=self.policy.allow_existing_compute,
        )
        if not ok:
            self._consecutive = 0
            return self._write_state("WAITING_GPU", snapshot=snapshot, reason=f"locked_recheck_{reason}")
        return self._write_state("GPU_READY_LOCKED_RECHECKED", snapshot=snapshot, reason="locked_recheck_eligible")

    def wait(self) -> dict[str, object]:
        if self._lock is None:
            self.acquire()
        self._started_at = _now()
        self._write_state("WAITING_GPU", snapshot=None, reason="waiter_started")
        try:
            while True:
                if self._cancel_requested():
                    return self._write_state("INTERRUPTED_RESUMABLE", snapshot=None, reason="cancel_request_consumed")
                state = self.sample_once()
                if state["status"] == "GPU_READY_RECHECK_REQUIRED":
                    state = self._locked_recheck()
                    if state["status"] == "GPU_READY_LOCKED_RECHECKED":
                        return state
                time.sleep(self.interval_s)
        except KeyboardInterrupt:
            self._write_state("INTERRUPTED_RESUMABLE", snapshot=None, reason="waiter_interrupted")
            raise
        except BaseException as exc:
            self._write_state("FAILED", snapshot=None, reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.close()


@dataclass
class LiveGpuAdmissionLease:
    """A project lock retained from final recheck through live process cleanup."""

    handle: Any
    lock_path: Path
    snapshot: LiveGpuSnapshot
    state: dict[str, Any]

    def close(self) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def acquire_live_admission(
    project_root: Path,
    *,
    run_id: str,
    actual_argv: Sequence[str],
    state_path: Path | None = None,
    probe: Callable[[], LiveGpuSnapshot] = probe_gpu,
    policy: GpuAdmissionPolicy = EXCLUSIVE_DT1_POLICY,
) -> LiveGpuAdmissionLease:
    """Hold the project lock and do one fresh real probe before Isaac starts.

    A direct ``runner --live`` call cannot replace the preceding waiter with a
    synthetic snapshot: the state must say it was created by live nvidia-smi,
    and this function always probes again while holding the same lock.
    """
    root = project_root.resolve()
    lock_path = project_gpu_lock(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = (state_path or default_state_path(root)).resolve()
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("project_root") != str(root):
            raise GpuWaitError("GPU waiter state belongs to another project root")
        if state.get("status") != "GPU_READY_LOCKED_RECHECKED":
            raise GpuWaitError("DT1 live launch requires GPU_READY_LOCKED_RECHECKED waiter state")
        if state.get("run_id") != run_id:
            raise GpuWaitError("DT1 live run_id must equal the freshly prepared waiter run_id")
        if state.get("admission_source") != "live_nvidia_smi":
            raise GpuWaitError("injected test snapshots can never admit a live DT1 runner")
        if state.get("admission_policy") != policy.policy_id:
            raise GpuWaitError("GPU waiter policy does not match this DT1 live mode")
        if state.get("allow_existing_compute") is not policy.allow_existing_compute:
            raise GpuWaitError("GPU waiter shared-compute setting does not match this DT1 live mode")
        if (state.get("required_free_mib") != policy.required_free_mib
                or state.get("required_consecutive_samples") != policy.required_samples):
            raise GpuWaitError("live admission does not match the frozen policy memory/sample thresholds")
        if state.get("sample_interval_s") != policy.sample_interval_s:
            raise GpuWaitError("live admission does not match the frozen 30-second sample interval")
        state_age_s = _utc_age_seconds(state.get("last_checked_utc"))
        snapshot = probe()
        ok, reason = eligibility(
            snapshot, required_free_mib=policy.required_free_mib,
            allow_existing_compute=policy.allow_existing_compute,
        )
        if not ok:
            raise GpuWaitError(f"locked live admission recheck failed: {reason}")
        receipt = {
            "format": "go2-dual-target-dt1-live-gpu-admission-v1", "stage": "DT1", "run_id": run_id,
            "project_root": str(root), "acquired_at_utc": _now(), "pid": os.getpid(),
            "waiter_run_id": state.get("run_id"), "waiter_state_path": str(path),
            "waiter_last_checked_age_s": state_age_s,
            "actual_argv": list(actual_argv), "fresh_snapshot": snapshot.as_dict(),
            "lock_path": str(lock_path), "admitted": True,
            "admission_policy": policy.policy_id,
            "allow_existing_compute": policy.allow_existing_compute,
        }
        receipt_path = root / "outputs" / "dual_target_v1" / f"dt1_live_gpu_admission_{run_id}.json"
        temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)
        consumed = dict(state)
        consumed.update({"status": "LIVE_ADMISSION_CONSUMED", "consumer_run_id": run_id,
                         "consumed_at_utc": _now(), "admission_receipt": str(receipt_path)})
        state_temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        state_temporary.write_text(json.dumps(consumed, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        state_temporary.replace(path)
        return LiveGpuAdmissionLease(handle=handle, lock_path=lock_path, snapshot=snapshot, state=receipt)
    except BaseException:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise


def request_cancel(project_root: Path, state_path: Path | None = None) -> dict[str, object]:
    """Request cancellation; only the waiter holding the lock consumes it.

    The caller never signals a PID.  That avoids PID-reuse and same-module
    cross-project hazards, and preserves a deterministic terminal state.
    """
    root = project_root.resolve()
    path = (state_path or default_state_path(root)).resolve()
    if not path.exists():
        raise GpuWaitError(f"no DT1 GPU waiter state exists at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("project_root") != str(root):
        raise GpuWaitError("waiter state belongs to another project root")
    if state.get("status") != "WAITING_GPU":
        raise GpuWaitError("only an active WAITING_GPU waiter may receive cancellation")
    request = {
        "format": "go2-dual-target-dt1-gpu-cancel-v1", "project_root": str(root),
        "run_id": state.get("run_id"), "requested_at_utc": _now(), "requested_by_pid": os.getpid(),
    }
    cancel_path = default_cancel_path(root)
    temporary = cancel_path.with_name(f".{cancel_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(cancel_path)
    return request


def _parse_snapshot_file(path: Path) -> list[LiveGpuSnapshot]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise GpuWaitError("--test-snapshots must be a JSON list")
    snapshots: list[LiveGpuSnapshot] = []
    for item in payload:
        if not isinstance(item, dict):
            raise GpuWaitError("each test snapshot must be an object")
        pids = item.get("compute_pids", [])
        if not isinstance(pids, list):
            raise GpuWaitError("test compute_pids must be a list")
        snapshot = LiveGpuSnapshot(item.get("total_mib"), item.get("used_mib"), tuple(pids), item.get("utilization_pct"))
        snapshot.validate()
        snapshots.append(snapshot)
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-id", default="dt1_gpu_wait")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--shared-smoke", action="store_true",
                        help="only the user-authorized DT1 shared 50-step smoke admission")
    parser.add_argument("--shared-coexistence", action="store_true",
                        help="user-authorized DT1 inference-only shared resource probe")
    parser.add_argument("--required-free-mib", type=int, help="test-snapshots only")
    parser.add_argument("--required-samples", type=int, help="test-snapshots only")
    parser.add_argument("--interval-s", type=float, help="test-snapshots only")
    parser.add_argument("--test-snapshots", type=Path, help="CPU test only: injected snapshots, never probes")
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    if args.cancel:
        if args.shared_smoke or args.shared_coexistence:
            parser.error("--cancel does not select or change a GPU admission policy")
        print(json.dumps(request_cancel(args.project_root, args.state_path), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.shared_smoke and args.shared_coexistence:
        parser.error("shared modes are mutually exclusive")
    policy = (SHARED_COEXISTENCE_DT1_POLICY if args.shared_coexistence else
              SHARED_SMOKE_DT1_POLICY if args.shared_smoke else EXCLUSIVE_DT1_POLICY)
    provided = (args.required_free_mib, args.required_samples, args.interval_s)
    if args.test_snapshots is None and any(value is not None for value in provided):
        parser.error("live DT1 admission thresholds are frozen by its policy; overrides are test-only")
    required_free_mib = args.required_free_mib if args.required_free_mib is not None else policy.required_free_mib
    required_samples = args.required_samples if args.required_samples is not None else policy.required_samples
    interval_s = args.interval_s if args.interval_s is not None else policy.sample_interval_s
    snapshots = _parse_snapshot_file(args.test_snapshots) if args.test_snapshots else None
    probe_index = 0

    def test_probe() -> LiveGpuSnapshot:
        nonlocal probe_index
        if snapshots is None:
            return probe_gpu()
        if probe_index >= len(snapshots):
            raise GpuWaitError("test snapshots exhausted before eligibility")
        result = snapshots[probe_index]
        probe_index += 1
        return result

    waiter = Dt1GpuWaiter(
        project_root=args.project_root,
        run_id=args.run_id,
        argv=[sys.executable, "-m", "src.dual_target.gpu_wait", *sys.argv[1:]],
        state_path=args.state_path,
        required_free_mib=required_free_mib,
        required_samples=required_samples,
        interval_s=interval_s,
        probe=test_probe,
        test_only=snapshots is not None,
        policy=policy,
    )
    result = waiter.wait()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    # A cancelled waiter is not successful admission: shell callers must not
    # proceed into their launch branch just because the JSON was written.
    if result["status"] == "INTERRUPTED_RESUMABLE":
        raise SystemExit(130)
    if result["status"] != "GPU_READY_LOCKED_RECHECKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
