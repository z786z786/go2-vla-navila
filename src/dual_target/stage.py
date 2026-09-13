"""DT0-only, CPU-safe stage self-check and state/gpu-wait dry-run entry."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from .contracts import TaskContract, build_policy_envelope, validate_action_chunk
from .layouts import SplitTask, build_four_tasks, example_group, validate_grouped_splits


class StageStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    WAITING_GPU = "WAITING_GPU"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    APPROVED = "APPROVED"
    FAILED = "FAILED"
    INTERRUPTED_RESUMABLE = "INTERRUPTED_RESUMABLE"


_TRANSITIONS = {
    StageStatus.NOT_STARTED: {StageStatus.RUNNING, StageStatus.WAITING_GPU, StageStatus.FAILED},
    StageStatus.RUNNING: {StageStatus.WAITING_GPU, StageStatus.READY_FOR_REVIEW, StageStatus.FAILED, StageStatus.INTERRUPTED_RESUMABLE},
    StageStatus.WAITING_GPU: {StageStatus.RUNNING, StageStatus.FAILED, StageStatus.INTERRUPTED_RESUMABLE},
    StageStatus.READY_FOR_REVIEW: {StageStatus.APPROVED, StageStatus.FAILED},
    StageStatus.INTERRUPTED_RESUMABLE: {StageStatus.RUNNING, StageStatus.WAITING_GPU, StageStatus.FAILED},
    StageStatus.APPROVED: set(),
    StageStatus.FAILED: set(),
}


def validate_transition(current: StageStatus, target: StageStatus, *, actor: str) -> None:
    if target not in _TRANSITIONS[current]:
        raise ValueError(f"invalid stage transition {current.value} -> {target.value}")
    if target is StageStatus.APPROVED and actor != "parent":
        raise PermissionError("only the parent agent may mark a stage APPROVED")
    if actor not in {"terra", "parent"}:
        raise PermissionError("unknown stage actor")


@dataclass(frozen=True)
class GpuSnapshot:
    """A supplied snapshot, never the result of a probe from DT0."""

    total_mib: int
    used_mib: int
    compute_process_count: int

    def validate(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in vars(self).values()):
            raise ValueError("GPU snapshot values must be integers")
        if self.total_mib <= 0 or self.used_mib < 0 or self.used_mib > self.total_mib or self.compute_process_count < 0:
            raise ValueError("GPU snapshot values are out of range")


@dataclass(frozen=True)
class GpuWaitDryRun:
    format: str = "go2-dual-target-gpu-wait-dry-run-v1"
    allocation_attempted: bool = False
    resource_probe_executed: bool = False
    required_free_mib: int = 20 * 1024
    required_consecutive_samples: int = 3
    sample_interval_s: int = 30
    sample_decisions: tuple[dict[str, Any], ...] = ()
    result: str = "not_ready_no_samples"


def evaluate_gpu_wait_dry_run(samples: Sequence[GpuSnapshot]) -> GpuWaitDryRun:
    """Evaluate supplied samples without calling nvidia-smi or acquiring a GPU."""
    consecutive = 0
    decisions: list[dict[str, Any]] = []
    required_free_mib = 20 * 1024
    required_consecutive_samples = 3
    for index, sample in enumerate(samples):
        sample.validate()
        free_mib = sample.total_mib - sample.used_mib
        eligible = free_mib >= required_free_mib and sample.compute_process_count == 0
        consecutive = consecutive + 1 if eligible else 0
        decisions.append({
            "sample_index": index,
            "free_mib": free_mib,
            "compute_process_count": sample.compute_process_count,
            "eligible": eligible,
            "consecutive_eligible_samples": consecutive,
        })
    result = "ready_for_gpu_acquisition" if consecutive >= required_consecutive_samples else "waiting_for_eligible_samples"
    if not samples:
        result = "not_ready_no_samples"
    return GpuWaitDryRun(sample_decisions=tuple(decisions), result=result)


def load_gpu_snapshots(path: Path | None) -> tuple[GpuSnapshot, ...]:
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--gpu-wait-snapshots must contain a JSON list")
    snapshots: list[GpuSnapshot] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each supplied GPU snapshot must be an object")
        snapshots.append(GpuSnapshot(
            total_mib=item.get("total_mib"),
            used_mib=item.get("used_mib"),
            compute_process_count=item.get("compute_process_count"),
        ))
    return tuple(snapshots)


def safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,80}", value):
        raise ValueError("run_id must be 3-81 safe filename characters")
    return value


def default_run_id() -> str:
    return f"dt0_cpu_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}"


def acquire_project_lock(lock_path: Path):
    """Return a non-blocking POSIX flock handle usable on macOS and Linux."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another dual-target stage process holds {lock_path}")
    return handle


def create_run_directory(output_dir: Path, run_id: str) -> Path:
    """Create exactly one run directory; a duplicate ID is a hard error."""
    run_dir = output_dir / safe_run_id(run_id)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing DT0 run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_check() -> dict[str, bool]:
    group = example_group()
    tasks = build_four_tasks(group)
    envelope = build_policy_envelope(
        image=object(), body_velocity=(0.0, 0.0, 0.0), task=tasks[0].instruction,
        audit={"geometry_group_id": group.geometry_group_id},
    )
    actions = [[0.0, 0.0, 0.0] for _ in range(50)]
    return {
        "contract_allows_only_current_rgb_state_task": set(envelope["policy_input"]) == {
            "observation.images.front", "observation.state", "task"
        },
        "four_combinations_generated": len(tasks) == 4 and not validate_grouped_splits(
            [SplitTask(task, "train") for task in tasks]
        ),
        "candidate_is_pending_sim_validation": all(task.sim_validation_status == "pending_sim_validation" for task in tasks),
        "action_chunk_50x3_cpu_validated": len(validate_action_chunk(actions)) == 50,
        "gpu_wait_dry_run_has_no_allocation": not evaluate_gpu_wait_dry_run(()).allocation_attempted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", help="only DT0 is implemented in this entry")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/dual_target_v1/dt0_runs"))
    parser.add_argument("--run-id", help="unique DT0 self-check ID; an existing ID is refused")
    parser.add_argument("--gpu-wait-dry-run", action="store_true")
    parser.add_argument("--gpu-wait-snapshots", type=Path, help="JSON snapshots used only with --gpu-wait-dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage != "DT0":
        raise SystemExit(f"refusing {args.stage}: only DT0 is implemented; parent approval is required before DT1+")
    if args.gpu_wait_snapshots is not None and not args.gpu_wait_dry_run:
        raise SystemExit("--gpu-wait-snapshots requires --gpu-wait-dry-run")
    transition = [StageStatus.NOT_STARTED, StageStatus.RUNNING, StageStatus.READY_FOR_REVIEW]
    for current, target in zip(transition, transition[1:]):
        validate_transition(current, target, actor="terra")
    output_dir = args.output_dir.resolve()
    run_id = safe_run_id(args.run_id or default_run_id())
    root = Path(__file__).resolve().parents[2]
    # This lock is project-owned rather than output-dir-owned, so a caller
    # cannot evade stage mutual exclusion by choosing another report folder.
    try:
        lock = acquire_project_lock(root / "reports/dual_target_v1/.dt0_stage.lock")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    run_dir: Path | None = None
    status_path: Path | None = None
    try:
        run_dir = create_run_directory(output_dir, run_id)
        status_path = run_dir / "stage_status.json"
        status_path.write_text(json.dumps({
            "stage": "DT0", "run_id": run_id, "status": StageStatus.RUNNING.value,
            "actor": "terra", "pid": os.getpid(), "approval_authority": "parent_agent_only",
        }, indent=2) + "\n", encoding="utf-8")
        checks = self_check()
        gpu_wait = evaluate_gpu_wait_dry_run(load_gpu_snapshots(args.gpu_wait_snapshots)) if args.gpu_wait_dry_run else None
        source_paths = [
            root / "src/dual_target/contracts.py", root / "src/dual_target/layouts.py",
            root / "src/dual_target/scoring.py", root / "src/dual_target/stage.py",
            root / "scripts/dual_target_run_stage.sh", root / "config/dual_target_v1/dt0_contract.json",
            root / "config/dual_target_v1/layout_policy.json",
        ]
        gate_path = run_dir / "gate.json"
        report: dict[str, Any] = {
            "format": "go2-dual-target-dt0-gate-v1",
            "stage": "DT0",
            "run_id": run_id,
            "status": StageStatus.READY_FOR_REVIEW.value,
            "approval_authority": "parent_agent_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "state_transition": [item.value for item in transition],
            "checks": checks,
            "gpu_wait_dry_run": asdict(gpu_wait) if gpu_wait is not None else None,
            "input_contract": TaskContract().as_dict(),
            "input_hashes": {
                "task_contract_draft_sha256": sha256(root / "config/dual_target_v1/dt0_contract.json"),
                "layout_policy_draft_sha256": sha256(root / "config/dual_target_v1/layout_policy.json"),
                "data_sha256": None,
                "model_sha256": None,
                "not_applicable_reason": "DT0 consumes no dataset or model and never constructs a CUDA policy.",
            },
            "source_sha256": {str(path.relative_to(root)): sha256(path) for path in source_paths},
            "output_paths": {"run_dir": str(run_dir), "gate": str(gate_path), "status": str(status_path)},
            "actual_argv": ["scripts/dual_target_run_stage.sh", *sys.argv[1:]],
            "exit_code": 0 if all(checks.values()) else 1,
            "failure_reason": None,
            "next_step": "await parent-agent review; do not start DT1",
        }
        gate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status_path.write_text(json.dumps({
            "stage": "DT0", "run_id": run_id, "status": StageStatus.READY_FOR_REVIEW.value,
            "actor": "terra", "pid": os.getpid(), "approval_authority": "parent_agent_only",
            "gate": str(gate_path),
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not all(checks.values()):
            raise SystemExit(1)
    except (FileExistsError, ValueError) as exc:
        if status_path is not None and run_dir is not None:
            status_path.write_text(json.dumps({
                "stage": "DT0", "run_id": run_id, "status": StageStatus.FAILED.value,
                "actor": "terra", "pid": os.getpid(), "approval_authority": "parent_agent_only",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(str(exc)) from None
    except BaseException as exc:
        if status_path is not None and run_dir is not None:
            status_path.write_text(json.dumps({
                "stage": "DT0", "run_id": run_id, "status": StageStatus.FAILED.value,
                "actor": "terra", "pid": os.getpid(), "approval_authority": "parent_agent_only",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
