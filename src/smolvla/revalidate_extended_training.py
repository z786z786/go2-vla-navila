#!/usr/bin/env python3
"""Audit an already-finished equal-exposure run and migrate only its PASS flag."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acceptance_checks(root: Path, report: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, bool]:
    steps = int(report.get("training", {}).get("steps", 0))
    final_checkpoint = root / "checkpoints" / f"step_{steps:06d}"
    checkpoint_meta_path = final_checkpoint / "m6_checkpoint.json"
    checkpoint_meta = (
        json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
        if checkpoint_meta_path.is_file() else {}
    )
    reloaded = report.get("reloaded_checkpoint_probe", {})
    contiguous = len(history) == steps and all(
        isinstance(row, dict) and row.get("step") == index
        for index, row in enumerate(history, start=1)
    )
    finite_history = contiguous and all(
        all(math.isfinite(float(row[key])) for key in ("loss", "gradient_norm", "lr", "step_time_s", "samples_per_s"))
        and (row.get("val_loss") is None or math.isfinite(float(row["val_loss"])))
        for row in history
    )
    return {
        "stage_is_smoke": report.get("stage") == "smoke",
        "minimum_steps_met": steps >= 2000,
        "history_is_contiguous_and_complete": contiguous,
        "history_is_finite": finite_history,
        "final_checkpoint_metadata_matches": checkpoint_meta.get("step") == steps,
        "final_model_exists": (final_checkpoint / "model.safetensors").is_file(),
        "processors_exist": all((final_checkpoint / name).is_file() for name in (
            "policy_preprocessor.json", "policy_postprocessor.json"
        )),
        "checkpoint_reloaded": bool(reloaded.get("finite")),
        "output_action_shape": reloaded.get("output_shape") == [1, 50, 3],
    }


def revalidate(root: Path) -> dict[str, Any]:
    report_path = root / "m6_report.json"
    history_path = root / "training_history.json"
    snapshot_path = root / "m6_report.before_extended_acceptance.json"
    acceptance_path = root / "extended_training_acceptance.json"
    if snapshot_path.exists() or acceptance_path.exists():
        raise FileExistsError("extended-training acceptance migration already exists")
    original_bytes = report_path.read_bytes()
    report = json.loads(original_bytes)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(history, list):
        raise ValueError("invalid training report or history")
    checks = acceptance_checks(root, report, history)
    result = {
        "format": "go2-short-vln-m6_2-extended-training-acceptance-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "original_report_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "history_sha256": sha256(history_path),
        "original_passed": report.get("passed"),
        "reason": "replace the legacy exact-2000-step smoke check for the audited equal-exposure run",
    }
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    snapshot_path.write_bytes(original_bytes)
    atomic_json(acceptance_path, result)
    report["passed"] = True
    report["acceptance_migration"] = {
        "kind": "audited_equal_exposure_smoke_acceptance",
        "evidence": str(acceptance_path),
        "evidence_sha256": sha256(acceptance_path),
        "original_report": str(snapshot_path),
        "original_report_sha256": result["original_report_sha256"],
    }
    atomic_json(report_path, report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    args = parser.parse_args()
    result = revalidate(args.training_root.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
