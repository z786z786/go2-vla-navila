#!/usr/bin/env python3
"""D3: audit exact-train-episode SmolVLA closed-loop rollouts.

The runner creates three policy-seed groups, each with the two frozen train
episodes.  This module intentionally consumes only persisted evidence: it
reuses the M7 rollout checker, verifies the deterministic policy-noise seed
for every socket response, and derives comparable navigation statistics from
the policy and retained M4 PD-oracle streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.inference.check_rollout import (
    EXPECTED_M4_PLANNER_SHA256,
    EXPECTED_SHORT_DATASET_SHA256,
    check_episode,
    load_jsonl,
)
from src.inference.server import derive_request_seed


EXPECTED_EPISODE_IDS = ("short_vln_v1_0000", "short_vln_v1_0004")
EXPECTED_BASE_SEEDS = (20260831, 20260832, 20260833)
EXPECTED_M61_CHECKPOINT_SHA256 = "facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7"
CONTROL_DT_S = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _position(record: dict[str, Any], key: str) -> tuple[float, float]:
    try:
        values = record[key]["position_w"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing {key}.position_w") from error
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"{key}.position_w must have at least two values")
    return _finite(values[0], f"{key}.position_w[0]"), _finite(values[1], f"{key}.position_w[1]")


def _path_length(records: Sequence[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    positions = [_position(records[0], "robot_pose")]
    positions.extend(_position(record, "next_robot_pose") for record in records)
    return sum(math.dist(before, after) for before, after in zip(positions, positions[1:]))


def _action(record: dict[str, Any], *, oracle: bool) -> tuple[float, float, float]:
    if oracle:
        values = [record.get("expert_vx"), record.get("expert_vy"), record.get("expert_wz")]
        label = "expert action"
    else:
        values = record.get("applied_action")
        label = "applied_action"
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return tuple(_finite(value, f"{label}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


def _policy_distances(records: Sequence[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        for key in ("evaluator_distance_to_goal_m", "next_evaluator_distance_to_goal_m"):
            if key in record:
                values.append(_finite(record[key], key))
    if not values:
        raise ValueError("policy records have no evaluator distance")
    return values


def _oracle_distances(records: Sequence[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        for key in ("distance_to_goal_xy_m", "next_distance_to_goal_xy_m"):
            if key in record:
                values.append(_finite(record[key], key))
    if not values:
        raise ValueError("oracle records have no XY goal distance")
    return values


def _metric_summary(
    records: Sequence[dict[str, Any]],
    *,
    success: bool,
    termination_reason: str,
    collision: bool,
    oracle: bool,
    latency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("rollout contains no step records")
    distances = _oracle_distances(records) if oracle else _policy_distances(records)
    actions = [_action(record, oracle=oracle) for record in records]
    final_distance = distances[-1]
    return {
        "success": bool(success),
        "final_distance_m": final_distance,
        "minimum_distance_m": min(distances),
        "time_to_goal_s": (len(records) * CONTROL_DT_S) if success else None,
        "path_length_m": _path_length(records),
        "mean_vx": statistics.fmean(action[0] for action in actions),
        "mean_abs_wz": statistics.fmean(abs(action[2]) for action in actions),
        "cumulative_vx_m": sum(action[0] for action in actions) * CONTROL_DT_S,
        "control_dt_s": CONTROL_DT_S,
        "frame_count": len(records),
        "termination_reason": termination_reason,
        "collision": bool(collision),
        "mean_inference_latency_ms": None if latency is None else latency.get("mean"),
        "p95_inference_latency_ms": None if latency is None else latency.get("p95"),
    }


def policy_navigation_metrics(summary: dict[str, Any], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = _metric_summary(
        records,
        success=bool(summary.get("success")),
        termination_reason=str(summary.get("termination_reason", "unknown")),
        collision=bool(summary.get("collision")),
        oracle=False,
        latency={
            "mean": _finite(summary["mean_inference_roundtrip_ms"], "mean inference latency"),
            "p95": _finite(summary["p95_inference_roundtrip_ms"], "p95 inference latency"),
        },
    )
    expected_final = summary.get("final_navigation_error_m")
    if expected_final is not None and not math.isclose(
        result["final_distance_m"], _finite(expected_final, "summary final navigation error"), abs_tol=1e-6
    ):
        raise ValueError("policy final distance differs from its final step record")
    return result


def oracle_navigation_metrics(summary: dict[str, Any], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = _metric_summary(
        records,
        success=bool(summary.get("success")),
        termination_reason=str(summary.get("termination_reason", "unknown")),
        collision=False,
        oracle=True,
    )
    expected_final = summary.get("final_distance_to_goal_xy_m")
    if expected_final is not None and not math.isclose(
        result["final_distance_m"], _finite(expected_final, "oracle final distance"), abs_tol=1e-6
    ):
        raise ValueError("oracle final distance differs from its final step record")
    return result


def audit_request_seeds(requests: Sequence[dict[str, Any]], *, base_seed: int) -> dict[str, Any]:
    checks = {
        "requests_present": bool(requests),
        "request_metadata_valid": True,
        "response_seed_matches": True,
    }
    actual: list[dict[str, Any]] = []
    for row in requests:
        request, response = row.get("request"), row.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            checks["request_metadata_valid"] = False
            checks["response_seed_matches"] = False
            continue
        try:
            episode_id = str(request["episode_id"])
            replan_index = int(request["replan_index"])
            expected = derive_request_seed(base_seed, episode_id, replan_index)
            observed = int(response["seed"])
        except (KeyError, TypeError, ValueError):
            checks["request_metadata_valid"] = False
            checks["response_seed_matches"] = False
            continue
        match = observed == expected
        if not match:
            checks["response_seed_matches"] = False
        actual.append({"request_id": request.get("request_id"), "expected": expected, "observed": observed, "match": match})
    return {"base_seed": base_seed, "passed": all(checks.values()), "checks": checks, "requests": actual}


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty series")
    return {
        "mean": statistics.fmean(materialized),
        "std": statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
    }


def classify_d3(results: Sequence[dict[str, Any]], *, evidence_passed: bool) -> dict[str, Any]:
    if not evidence_passed:
        return {"classification": "UNRESOLVED", "reason": "D3 infrastructure or evidence validation failed."}
    per_episode = {
        episode_id: [item for item in results if item["episode_id"] == episode_id]
        for episode_id in EXPECTED_EPISODE_IDS
    }
    if any(len(items) != len(EXPECTED_BASE_SEEDS) for items in per_episode.values()):
        return {"classification": "UNRESOLVED", "reason": "D3 requires exactly three runs per train episode."}
    unstable = [
        episode_id
        for episode_id, items in per_episode.items()
        if sum(bool(item["official_navigation_success"]) for item in items) < len(EXPECTED_BASE_SEEDS)
    ]
    if unstable:
        return {
            "classification": "A",
            "reason": "At least one exact train episode was not 3/3 official-successful; prioritize underfit, action scaling, timing, and chunk execution.",
            "unstable_episodes": unstable,
        }
    return {
        "classification": "B",
        "reason": "Both exact train episodes were 3/3 official-successful while the frozen M7 held-out reference failed; prioritize data coverage, generalization, and closed-loop covariate shift.",
        "unstable_episodes": [],
    }


def _plot_episode(output: Path, entries: Sequence[dict[str, Any]], oracle_records: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    oracle_time = [float(row["timestamp"]) for row in oracle_records]
    oracle_distance = [float(row["distance_to_goal_xy_m"]) for row in oracle_records]
    oracle_actions = [_action(row, oracle=True) for row in oracle_records]
    axes[0, 0].plot(oracle_time, oracle_distance, color="black", linewidth=2, label="PD oracle")
    axes[0, 1].plot(oracle_time, [item[0] for item in oracle_actions], color="black", linewidth=2, label="PD oracle")
    axes[1, 0].plot(oracle_time, [item[2] for item in oracle_actions], color="black", linewidth=2, label="PD oracle")
    oracle_positions = [_position(oracle_records[0], "robot_pose")] + [_position(row, "next_robot_pose") for row in oracle_records]
    axes[1, 1].plot([pos[0] for pos in oracle_positions], [pos[1] for pos in oracle_positions], color="black", linewidth=2, label="PD oracle")
    for entry in entries:
        records = entry["records"]
        label = f"SmolVLA seed {entry['base_seed']}"
        time = [float(row["timestamp"]) for row in records]
        axes[0, 0].plot(time, [float(row["evaluator_distance_to_goal_m"]) for row in records], alpha=0.8, label=label)
        actions = [_action(row, oracle=False) for row in records]
        axes[0, 1].plot(time, [item[0] for item in actions], alpha=0.8, label=label)
        axes[1, 0].plot(time, [item[2] for item in actions], alpha=0.8, label=label)
        positions = [_position(records[0], "robot_pose")] + [_position(row, "next_robot_pose") for row in records]
        axes[1, 1].plot([pos[0] for pos in positions], [pos[1] for pos in positions], alpha=0.8, label=label)
    for axis, title, ylabel in (
        (axes[0, 0], "Goal distance", "m"),
        (axes[0, 1], "Applied vx", "m/s"),
        (axes[1, 0], "Applied wz", "rad/s"),
    ):
        axis.set_title(title)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[1, 1].set_title("XY trajectory")
    axes[1, 1].set_xlabel("world x (m)")
    axes[1, 1].set_ylabel("world y (m)")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for item in report["runs"]:
        metrics = item["metrics"]
        rows.append(
            "| {episode} | {seed} | {infra} | {success} | {final:.3f} | {minimum:.3f} | {time} | {path:.3f} | {vx:.3f} | {wz:.3f} | {latency:.1f} | {term} |".format(
                episode=item["episode_id"], seed=item["base_seed"],
                infra="PASS" if item["infrastructure_passed"] and item["artifacts_complete"] and item["seed_audit"]["passed"] else "FAIL",
                success="PASS" if item["official_navigation_success"] else "FAIL",
                final=metrics["final_distance_m"], minimum=metrics["minimum_distance_m"],
                time="{:.2f}".format(metrics["time_to_goal_s"]) if metrics["time_to_goal_s"] is not None else "N/A",
                path=metrics["path_length_m"], vx=metrics["mean_vx"], wz=metrics["mean_abs_wz"],
                latency=metrics["mean_inference_latency_ms"], term=metrics["termination_reason"],
            )
        )
    classification = report["classification"]
    return (
        "# D3 Exact Train-Episode Closed-Loop Report\n\n"
        "| Episode | Policy seed | Evidence | Official success | Final distance (m) | Minimum distance (m) | Time-to-goal (s) | Path length (m) | Mean vx | Mean |wz| | Mean latency (ms) | Termination |\n"
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n\nD3 evidence gate: **{}**. Classification: **{}** — {}\n".format(
            "PASS" if report["passed"] else "FAIL", classification["classification"], classification["reason"]
        )
    )


def _append_diagnosis(path: Path, report: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    if "## D3 — Exact Train-Episode Closed-Loop Test" in text:
        raise FileExistsError(f"D3 is already recorded in {path}; refusing duplicate diagnosis entry")
    classification = report["classification"]
    per_episode = report["per_episode"]
    lines = [
        "\n## D3 — Exact Train-Episode Closed-Loop Test\n",
        "**Stage:** D3  ",
        f"**Status:** {'PASS' if report['passed'] else 'FAIL'}  ",
        f"**Evidence root:** `{report['run_root']}`\n",
        "Three fixed policy-noise seeds (`20260831/20260832/20260833`) were evaluated on each exact train episode with Isaac seed `20260831`. Inputs remained current RGB, 30-D state, and instruction only; no route, waypoint, goal, or PD state was sent to the policy.\n",
    ]
    for episode_id in EXPECTED_EPISODE_IDS:
        aggregate = per_episode[episode_id]
        oracle = aggregate["pd_oracle_metrics"]
        lines.append(
            f"- `{episode_id}`: SmolVLA official success `{aggregate['official_success_count']}/3`; "
            f"final distance mean±std `{aggregate['final_distance_m']['mean']:.3f}±{aggregate['final_distance_m']['std']:.3f} m`; "
            f"PD oracle success `{oracle['success']}`, final distance `{oracle['final_distance_m']:.3f} m`.\n"
        )
    lines.extend([
        f"\n**Classification:** `{classification['classification']}`. {classification['reason']}\n",
        "Artifacts include all six request/step streams, raw and applied actions, RGB frames, decodable rollout videos, per-run M7 sanity files, seed audits, oracle metrics, and train-route comparison plots.\n",
        "\n**Stop here.** Wait for `CONTINUE D4`.\n",
    ])
    path.write_text(text.rstrip() + "\n" + "".join(lines), encoding="utf-8")


def build_report(
    run_root: Path,
    oracle_root: Path,
    *,
    expected_checkpoint_sha256: str = EXPECTED_M61_CHECKPOINT_SHA256,
    expected_planner_sha256: str = EXPECTED_M4_PLANNER_SHA256,
    expected_short_dataset_sha256: str = EXPECTED_SHORT_DATASET_SHA256,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    per_episode_records: dict[str, list[dict[str, Any]]] = {episode_id: [] for episode_id in EXPECTED_EPISODE_IDS}
    oracle_metrics: dict[str, dict[str, Any]] = {}
    for episode_id in EXPECTED_EPISODE_IDS:
        oracle_dir = oracle_root / episode_id
        oracle_summary = json.loads((oracle_dir / "summary.json").read_text(encoding="utf-8"))
        oracle_records = load_jsonl(oracle_dir / "steps.jsonl")
        oracle_metrics[episode_id] = oracle_navigation_metrics(oracle_summary, oracle_records)
        if oracle_summary.get("official_planner_sha256") != expected_planner_sha256:
            raise ValueError(f"oracle planner hash mismatch for {episode_id}")
        if oracle_summary.get("short_dataset_sha256") != expected_short_dataset_sha256:
            raise ValueError(f"oracle dataset hash mismatch for {episode_id}")
    for base_seed in EXPECTED_BASE_SEEDS:
        seed_root = run_root / f"seed_{base_seed}"
        config = json.loads((seed_root / "seed_config.json").read_text(encoding="utf-8"))
        if config.get("policy_base_seed") != base_seed:
            raise ValueError(f"seed config mismatch in {seed_root}")
        if config.get("isaac_seed") != 20260831:
            raise ValueError(f"Isaac seed mismatch in {seed_root}")
        for episode_id in EXPECTED_EPISODE_IDS:
            episode_dir = seed_root / episode_id
            checked = check_episode(
                episode_dir, oracle_root,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_planner_sha256=expected_planner_sha256,
                expected_short_dataset_sha256=expected_short_dataset_sha256,
            )
            records = load_jsonl(episode_dir / "steps.jsonl")
            requests = load_jsonl(episode_dir / "requests.jsonl")
            seed_audit = audit_request_seeds(requests, base_seed=base_seed)
            item = {
                "episode_id": episode_id,
                "base_seed": base_seed,
                **checked,
                "seed_audit": seed_audit,
                "metrics": policy_navigation_metrics(checked["summary"], records),
            }
            runs.append(item)
            per_episode_records[episode_id].append({"base_seed": base_seed, "records": records})
    expected_pairs = {(episode_id, seed) for episode_id in EXPECTED_EPISODE_IDS for seed in EXPECTED_BASE_SEEDS}
    actual_pairs = {(item["episode_id"], item["base_seed"]) for item in runs}
    structure_passed = actual_pairs == expected_pairs and len(runs) == len(expected_pairs)
    per_episode: dict[str, Any] = {}
    for episode_id, items in per_episode_records.items():
        matching = [run for run in runs if run["episode_id"] == episode_id]
        per_episode[episode_id] = {
            "run_count": len(matching),
            "official_success_count": sum(item["official_navigation_success"] for item in matching),
            "final_distance_m": _mean_std(item["metrics"]["final_distance_m"] for item in matching),
            "minimum_distance_m": _mean_std(item["metrics"]["minimum_distance_m"] for item in matching),
            "path_length_m": _mean_std(item["metrics"]["path_length_m"] for item in matching),
            "mean_vx": _mean_std(item["metrics"]["mean_vx"] for item in matching),
            "mean_abs_wz": _mean_std(item["metrics"]["mean_abs_wz"] for item in matching),
            "cumulative_vx_m": _mean_std(item["metrics"]["cumulative_vx_m"] for item in matching),
            "mean_inference_latency_ms": _mean_std(item["metrics"]["mean_inference_latency_ms"] for item in matching),
            "pd_oracle_metrics": oracle_metrics[episode_id],
        }
        oracle_records = load_jsonl(oracle_root / episode_id / "steps.jsonl")
        _plot_episode(run_root / f"{episode_id}_d3_comparison.png", items, oracle_records)
    evidence_passed = bool(
        structure_passed
        and all(item["infrastructure_passed"] and item["artifacts_complete"] and item["seed_audit"]["passed"] for item in runs)
    )
    classification = classify_d3(runs, evidence_passed=evidence_passed)
    return {
        "format": "go2-short-vln-m6-2-d3-closed-loop-v1",
        "run_root": str(run_root.resolve()),
        "expected_episode_ids": list(EXPECTED_EPISODE_IDS),
        "expected_base_seeds": list(EXPECTED_BASE_SEEDS),
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "expected_planner_sha256": expected_planner_sha256,
        "expected_short_dataset_sha256": expected_short_dataset_sha256,
        "structure_passed": structure_passed,
        "runs": runs,
        "per_episode": per_episode,
        "classification": classification,
        "passed": evidence_passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--diagnosis-report", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_M61_CHECKPOINT_SHA256)
    parser.add_argument("--expected-planner-sha256", default=EXPECTED_M4_PLANNER_SHA256)
    parser.add_argument("--expected-short-dataset-sha256", default=EXPECTED_SHORT_DATASET_SHA256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.run_root.expanduser().resolve(), args.oracle_root.expanduser().resolve(),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_planner_sha256=args.expected_planner_sha256,
        expected_short_dataset_sha256=args.expected_short_dataset_sha256,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(_render_markdown(report), encoding="utf-8")
    if args.diagnosis_report is not None:
        _append_diagnosis(args.diagnosis_report.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
