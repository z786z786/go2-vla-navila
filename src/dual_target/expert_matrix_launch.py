"""Non-training DT1 16-slot expert plan and immutable result collector.

This module never starts Isaac or a GPU waiter.  The accompanying shell
launcher invokes one exclusive waiter and one already-bounded expert process
per predeclared slot.  Keeping scheduling and collection CPU-only makes the
fixed task list reviewable before expensive work begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .expert_matrix import ExpertPair, build_dt1_expert_pairs
from .reset_audit import ResetAuditError, reset_audit_from_dict, validate_paired_resets


PLAN_FORMAT = "go2-dual-target-dt1-expert-matrix-plan-v1"
RESULT_FORMAT = "go2-dual-target-dt1-expert-matrix-result-v1"


class ExpertMatrixLaunchError(RuntimeError):
    """The fixed DT1 matrix cannot be created or collected safely."""


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 48:
        raise ExpertMatrixLaunchError(f"{label} must be a non-empty compact identifier")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ExpertMatrixLaunchError(f"{label} contains unsafe filename characters")
    return value


def _slot_run_id(matrix_run_id: str, pair: ExpertPair, color: str) -> str:
    value = f"{matrix_run_id}__{pair.pair_id}__{color}"
    if len(value) > 81:
        raise ExpertMatrixLaunchError("matrix slot run_id exceeds runner's safe length limit")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_launcher_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "dual_target_dt1_matrix.sh"


def build_matrix_plan(matrix_run_id: str, *, launcher_script: Path | None = None) -> dict[str, object]:
    """Create exactly 16 immutable expert slots before any slot executes."""
    identifier = _safe_id(matrix_run_id, "matrix_run_id")
    launcher = (launcher_script or _default_launcher_script()).resolve()
    if not launcher.is_file():
        raise ExpertMatrixLaunchError(f"matrix launcher script is absent: {launcher}")
    pairs = build_dt1_expert_pairs()
    slots: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        for color, task in pair.task_slots():
            slots.append({
                "slot_index": len(slots), "pair_index": pair_index, "pair_id": pair.pair_id,
                "run_id": _slot_run_id(identifier, pair, color),
                "geometry_group_id": pair.scene.group.geometry_group_id,
                "color_configuration": pair.scene.color_configuration,
                "repeat": pair.repeat, "target_color": color, "pair_seed": pair.pair_seed,
                "instruction": task.instruction, "target_slot": task.target_slot,
            })
    if len(slots) != 16 or {slot["target_color"] for slot in slots} != {"red", "blue"}:
        raise ExpertMatrixLaunchError("DT1 matrix must contain the fixed 16 red/blue expert slots")
    return {
        "format": PLAN_FORMAT, "stage": "DT1", "matrix_run_id": identifier,
        "status": "SCHEDULED_NOT_TASK_APPROVED", "pair_count": 8, "slot_count": 16,
        "retry_policy": "no_per_slot_retry_or_best_of_selection; create_a_new_matrix_revision_after_complete_diagnostic_review",
        "per_slot_gpu_admission": "exclusive_3_samples_30s_then_locked_recheck_before_each_slot",
        "matrix_launcher_path": str(launcher), "matrix_launcher_sha256": _sha256(launcher),
        "matrix_collector_sha256": _sha256(Path(__file__).resolve()),
        "slots": slots, "navigation_success_approved": False, "dt1_approved": False,
    }


def create_matrix_plan(output_dir: Path, matrix_run_id: str, *, launcher_script: Path | None = None) -> Path:
    """Persist one non-overwriting plan before its first waiter is started."""
    plan = build_matrix_plan(matrix_run_id, launcher_script=launcher_script)
    root = output_dir.resolve() / matrix_run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite matrix run root: {root}")
    root.mkdir(parents=True)
    path = root / "expert_matrix_plan.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpertMatrixLaunchError(f"matrix plan is unreadable: {path}") from exc
    expected = {
        "format", "stage", "matrix_run_id", "status", "pair_count", "slot_count", "retry_policy",
        "per_slot_gpu_admission", "matrix_launcher_path", "matrix_launcher_sha256", "matrix_collector_sha256",
        "slots", "navigation_success_approved", "dt1_approved",
    }
    if not isinstance(plan, dict) or set(plan) != expected:
        raise ExpertMatrixLaunchError("matrix plan keys do not match the frozen schema")
    canonical = build_matrix_plan(plan.get("matrix_run_id"), launcher_script=Path(plan.get("matrix_launcher_path", "")))
    if plan != canonical:
        raise ExpertMatrixLaunchError("matrix plan differs from the frozen 16-slot schedule")
    return plan


def _load_slot_success(output_dir: Path, slot: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    run_root = output_dir.resolve() / str(slot["run_id"])
    status_path = run_root / "stage_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "stage_status_missing_or_invalid"
    if not isinstance(status, dict):
        return None, "stage_status_must_be_an_object"
    result = status.get("episode_result")
    if status.get("status") != "RUNNING" or not isinstance(result, dict) or result.get("status") != "success":
        return None, "slot_not_successful"
    if result.get("episode_id") is None:
        return None, "episode_result_lacks_episode_id"
    summary_path = output_dir.resolve() / f"dt1_expert_{slot['run_id']}_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "launcher_summary_missing_or_invalid"
    if (not isinstance(summary, dict) or summary.get("run_id") != slot["run_id"]
            or summary.get("exit_code") != 0 or summary.get("stage_status") != "RUNNING"
            or summary.get("result_status") != "success"):
        return None, "launcher_summary_does_not_confirm_success"
    return result, None


def collect_matrix_results(output_dir: Path, plan_path: Path) -> dict[str, object]:
    """Collect every scheduled slot once; never select among repeat attempts."""
    plan = _load_plan(plan_path)
    slots = plan["slots"]
    outcomes: list[dict[str, object]] = []
    audits_by_pair: dict[str, list[Any]] = {}
    all_success = True
    for slot in slots:
        run_root = output_dir.resolve() / str(slot["run_id"])
        result, error = _load_slot_success(output_dir, slot)
        outcome: dict[str, object] = {
            "slot_index": slot["slot_index"], "run_id": slot["run_id"],
            "pair_id": slot["pair_id"], "target_color": slot["target_color"],
            "success": error is None, "error": error or "",
        }
        if result is not None:
            audit_path = result.get("paired_reset_audit_path")
            if not isinstance(audit_path, str) or not audit_path:
                outcome.update(success=False, error="successful_slot_lacks_paired_reset_audit")
                all_success = False
            else:
                try:
                    resolved_audit = Path(audit_path).resolve()
                    if not resolved_audit.is_relative_to(run_root.resolve()):
                        raise ResetAuditError("paired reset audit is outside the slot's own run root")
                    actual_audit_hash = _sha256(resolved_audit)
                    if result.get("paired_reset_audit_sha256") != actual_audit_hash:
                        raise ResetAuditError("episode result does not bind the paired-reset audit hash")
                    audit = reset_audit_from_dict(json.loads(resolved_audit.read_text(encoding="utf-8")))
                    if (audit.geometry_group_id, audit.color_configuration, audit.repeat, audit.target_color, audit.pair_seed) != (
                        slot["geometry_group_id"], slot["color_configuration"], slot["repeat"],
                        slot["target_color"], slot["pair_seed"],
                    ):
                        raise ResetAuditError("slot audit identity differs from predeclared matrix identity")
                    for relative, expected_hash, label in (
                        (audit.first_rgb_path, audit.first_rgb_sha256, "reset RGB"),
                        (audit.learner_first_rgb_path, audit.learner_first_rgb_sha256, "learner-start RGB"),
                    ):
                        rgb_path = (resolved_audit.parent / relative).resolve()
                        if not rgb_path.is_relative_to(resolved_audit.parent.resolve()) or _sha256(rgb_path) != expected_hash:
                            raise ResetAuditError(f"{label} is missing, escapes its episode, or differs from the sidecar hash")
                    audits_by_pair.setdefault(str(slot["pair_id"]), []).append(audit)
                    outcome["paired_reset_audit_path"] = str(resolved_audit)
                    outcome["paired_reset_audit_sha256"] = actual_audit_hash
                except (OSError, json.JSONDecodeError, ResetAuditError) as exc:
                    outcome.update(success=False, error=f"paired_reset_audit_invalid:{type(exc).__name__}")
                    all_success = False
        else:
            all_success = False
        outcomes.append(outcome)
    pair_reviews: list[dict[str, object]] = []
    for pair_id in sorted({str(slot["pair_id"]) for slot in slots}):
        audits = audits_by_pair.get(pair_id, [])
        if len(audits) != 2:
            all_success = False
            pair_reviews.append({"pair_id": pair_id, "matched": False, "error": "pair_audits_missing"})
            continue
        try:
            review = validate_paired_resets(audits[0], audits[1])
            pair_reviews.append({"pair_id": pair_id, "matched": True, "review": review})
        except ResetAuditError as exc:
            all_success = False
            pair_reviews.append({"pair_id": pair_id, "matched": False, "error": str(exc)})
    return {
        "format": RESULT_FORMAT, "stage": "DT1", "matrix_run_id": plan["matrix_run_id"],
        "status": "EXPERT_MATRIX_COMPLETE_NOT_DT1_APPROVED" if all_success else "EXPERT_MATRIX_INCOMPLETE_OR_FAILED",
        "plan_path": str(plan_path.resolve()), "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "slot_count": len(slots), "successful_slots": sum(1 for outcome in outcomes if outcome["success"]),
        "slots": outcomes, "pair_reviews": pair_reviews,
        "retry_policy": plan["retry_policy"], "navigation_success_approved": False, "dt1_approved": False,
    }


def _write_result(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite matrix result: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--launcher-script", type=Path,
                        help="exact matrix shell to bind into the immutable pre-execution plan")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create-plan", action="store_true")
    action.add_argument("--print-slots", action="store_true")
    action.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    plan_path = args.output_dir.resolve() / args.matrix_run_id / "expert_matrix_plan.json"
    if args.create_plan:
        print(create_matrix_plan(args.output_dir, args.matrix_run_id, launcher_script=args.launcher_script))
        return
    plan = _load_plan(plan_path)
    if args.print_slots:
        for slot in plan["slots"]:
            print("\t".join(str(slot[key]) for key in (
                "run_id", "geometry_group_id", "color_configuration", "target_color", "repeat",
            )))
        return
    result = collect_matrix_results(args.output_dir, plan_path)
    result_path = plan_path.parent / "expert_matrix_result.json"
    _write_result(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "EXPERT_MATRIX_COMPLETE_NOT_DT1_APPROVED":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
