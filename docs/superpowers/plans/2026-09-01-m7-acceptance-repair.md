# M7 Acceptance Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct M7's acceptance accounting, turn the offline checkpoint check into an independent hard gate, produce a physically bounded SmolVLA checkpoint only with explicit scope approval, and rerun the five-route gate without entering M8.

**Architecture:** Keep the policy server and Isaac client isolated. Centralize range accounting in one pure module used by the client, preflight, and independent checker; never trust client-written summary booleans. If checkpoint remediation is authorized, train in a new M6.1 output tree with an invertible bounded action codec saved beside the processors, then return decoded physical actions before the unchanged safety gate.

**Tech Stack:** Python 3.12, PyTorch, LeRobot 0.6.0, SmolVLA, Unix sockets, Isaac Sim/Isaac Lab, `unittest`, Bash.

**Spec:** `reports/MILESTONES.md` M7 section and the original “M7 — SmolVLA 闭环导航集成” acceptance plan in this task.

## Global Constraints

- Do not overwrite `/mnt/wxh/go2_short_vln/outputs/m7/gate_20260901T084954+0800` or any earlier M7 evidence.
- Keep the policy request allowlist unchanged: rotated 512×512 RGB JPEG, 30-D state, short instruction, and audit-only episode/replan IDs.
- Never send path, waypoint, goal pose/direction/distance, PD state, planner action, or oracle heading to SmolVLA.
- Keep action bounds exactly `vx=[0,0.5]`, `vy=0`, `wz=[-0.5,0.5]`; execute 10 of 50 predicted actions.
- Keep timeout at 10 seconds, episode limit at 1,500 frames, and success at 10 consecutive official-evaluator frames within 0.5 m.
- Do not run the remaining four routes until `short_vln_v1_0004` passes the first gate.
- Reuse M4 oracle artifacts only when planner and dataset hashes match.
- Do not train or replace the M6 checkpoint unless the user explicitly authorizes M6.1.
- Do not enter M8. The workspace has no Git repository, so record SHA-256 hashes instead of commits or tags.

---

## Milestone M7-R1: Correct Range Accounting and Report Semantics

**Outcome:** Existing evidence is described accurately: 37/2,100 action vectors, 10/42 chunks, and 15/420 executed policy actions violated the raw range. M7 remains FAIL.

**Files:**
- Create: `src/inference/action_audit.py`
- Create: `tests/inference/test_action_audit.py`
- Modify: `src/inference/isaac_client.py`
- Modify: `src/inference/check_rollout.py`
- Modify: `reports/MILESTONES.md`
- Modify: `reports/EXPERIMENTS.md`

**Interfaces:**
- Produce `ActionRangeSummary(total_vectors, violating_vectors, chunks_with_violation, executed_vectors, executed_violations, per_dimension)`.
- Produce `summarize_action_range(chunks: Sequence[Sequence[Sequence[float]]], execute_steps: int) -> ActionRangeSummary`.

- [x] Write tests proving that vector violations and chunk violations are counted separately, overlapping dimension violations count as one vector, and executed/tail denominators are correct.
- [x] Run `python -m unittest tests.inference.test_action_audit -v`; it initially failed because the module did not exist.
- [x] Implement the pure range summarizer using `ACTION_BOUNDS` from `src/inference/state.py`.
- [x] Replace `raw_chunk_range_violation_count` generation in `isaac_client.py` with explicit vector/chunk/executed fields while retaining the old field only as a documented compatibility alias.
- [x] Change the Markdown gate table to show `Official success`, `Raw range`, and `Range-compliant success` as separate columns.
- [x] Add a correction note to both reports; do not modify the immutable rollout JSONL or original gate report.
- [x] Run `python -m unittest discover -s tests -q`; all tests pass locally and on the server.
- [x] Record SHA-256 hashes for changed source, tests, reports, and the derived server recomputation.

**Acceptance:** Synthetic count tests pass; recomputation of the retained episode yields exactly 37 violating vectors across 10 chunks, including 15 executed policy actions; the report still says M7 FAIL.

---

## Milestone M7-R2: Make Preflight and the Independent Checker Hard Gates

**Outcome:** The current fixed checkpoint fails before Isaac starts, and no client-written range flag can make the independent checker pass.

**Files:**
- Modify: `src/inference/preflight.py`
- Modify: `src/inference/check_rollout.py`
- Modify: `scripts/m7_run_gate.sh`
- Modify: `tests/inference/test_preflight.py`
- Modify: `tests/inference/test_check_rollout.py`

**Interfaces:**
- Produce `evaluate_preflight_response(response: dict, roundtrip_ms: float) -> dict` with a computed `passed` field.
- Extend the checker CLI with `--gate-stage {first,full}`.
- First stage expects exactly `short_vln_v1_0004`; full stage expects exactly `0000/0001/0003/0004/0006`.

- [x] Write a failing preflight test where one of 50 actions is below `vx=0`; assert `passed=False` and a nonzero CLI exit.
- [x] Write failing checker tests that corrupt the summary range flag, mismatch a step's request offset, omit one full-gate episode, and alter an oracle hash.
- [x] Run the two targeted test modules and confirm the new tests fail for the expected reasons.
- [x] Compute preflight range status from all 50 returned actions; remove the hard-coded `passed=True`.
- [x] Insert the real socket preflight into `m7_run_gate.sh` after server readiness and before `m7_run_episode.sh`.
- [x] Recompute all response-action ranges inside `check_rollout.py`; cross-check every executed step against `(request_id, action_offset)`, recompute applied clipping, and compare all derived counts with the summary.
- [x] Enforce stage-specific episode IDs, artifact completeness, checkpoint hash, M4 planner hash, and short-dataset hash.
- [x] Run all unit tests, Python compilation, and shell syntax checks.
- [x] Run only the real offline/socket preflight with the existing M6 checkpoint; expect a deliberate FAIL and verify Isaac is not launched.

**Acceptance:** The retained M6 checkpoint is rejected before Isaac; tests demonstrate the checker cannot be passed by editing summary booleans or omitting episodes.

### Scope Decision Gate

- If the user does not authorize M6.1, stop here, publish the corrected M7 FAIL report, and do not rerun the closed loop.
- If the user explicitly authorizes M6.1, continue to M7-R3. This is a checkpoint/training change, not an integration-only repair.

---

## Milestone M7-R3: Produce a Bounded M6.1 Checkpoint (Conditional)

**Outcome:** Every finite model latent action decodes to a physical `[vx,0,wz]` inside the immutable bounds before the M7 safety gate.

**Files:**
- Create: `src/smolvla/bounded_actions.py`
- Create: `tests/smolvla/test_bounded_actions.py`
- Modify: `src/smolvla/m6_training.py`
- Modify: `src/inference/server.py`
- Create: `scripts/m61_run_stage.sh`
- Modify: `tests/inference/test_server.py`

**Interfaces:**
- `encode_physical_actions(actions: Tensor, epsilon: float = 1e-4) -> Tensor`
- `decode_latent_actions(latent: Tensor) -> Tensor`
- Decode rules: `vx=0.25*(tanh(zx)+1)`, `vy=0`, `wz=0.5*tanh(zw)`.
- Save `bounded_action_codec.json` in every M6.1 checkpoint with codec version, epsilon, bounds, and source checkpoint hash.

- [x] Write property tests covering exact physical boundaries, round-trip interior values, arbitrary large finite latent values, zero lateral velocity, dtype/device preservation, and NaN/Inf rejection.
- [x] Run `python -m unittest tests.smolvla.test_bounded_actions -v`; expect failures before implementation.
- [x] Implement the codec and make the property tests pass.
- [x] In the training path, encode physical action chunks before normalization and calculate replacement action statistics from all 847 encoded train frames; leave images and 30-D state unchanged.
- [x] In probe metrics, decode postprocessor output before comparing it with physical expert actions.
- [x] Save the codec metadata beside the preprocessor/postprocessor and require it when loading an M6.1 checkpoint.
- [x] In the server, return decoded physical actions as policy output before `apply_action_safety`; store latent output only in the server-side audit log.
- [x] Train or fine-tune into a new `outputs/m6_1/...` directory without altering `step_002000`.
- [x] Reload the new checkpoint in a fresh process and verify deterministic `1×50×3` finite output, zero raw range violations, processor/codec hashes, and socket round trip.
- [x] Run the full server-host unit suite and record loss, physical-action MAE, prediction range, resource peaks, and checkpoint hashes.

**Acceptance:** Codec property tests prove bounds for all finite latent inputs; the saved M6.1 checkpoint reloads offline and its M7 preflight passes with zero raw range violations.

---

## Milestone M7-R4: Rerun the Closed-Loop Gates and Close M7

**Outcome:** A new evidence root contains the first gate and, only after it passes, all five required task artifacts and a corrected final verdict.

**Files:**
- Modify only if needed: `scripts/m7_run_server.sh` to select the approved M6.1 checkpoint.
- Modify: `reports/MILESTONES.md`
- Modify: `reports/EXPERIMENTS.md`
- Generate under a new immutable directory: `/mnt/wxh/go2_short_vln/outputs/m7/<new-gate-tag>`.

- [x] Verify at least 20,480 MiB free VRAM without terminating or pausing other GPU jobs.
- [x] Run all unit, compilation, shell syntax, real checkpoint, and socket preflight gates.
- [x] Verify M4 planner hash `20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432` and dataset hash `88255ffb8e175372048c4948531d7ad059a9d8aa9217b7be95e2f60f520b34b6`; reuse the oracle only on an exact match.
- [x] Run `short_vln_v1_0004` and independently check official success, zero raw violations, no leakage, latency, logs, plots, and decodable video.
- [x] If and only if the first gate passes, run `0000`, `0001`, `0003`, and `0006` (not authorized by the failed first gate).
- [x] Run the full-stage checker with the exact five IDs and generate JSON/Markdown reports (not authorized by the failed first gate).
- [x] Update both project reports with commands, environment, hashes, resource peaks, failures, and the final M7 verdict.
- [x] Synchronize approved source/tests/reports to the server and verify local/remote hashes.
- [x] Stop after M7; do not create or start M8 work.

**Acceptance:** All infrastructure checks pass; all five episodes have complete logs and videos; no forbidden policy input exists; at least one official success is raw-range compliant. Otherwise record M7 FAIL and stop.

## Recommended Execution Order

1. Execute M7-R1 and review the corrected historical accounting.
2. Execute M7-R2 and confirm the existing checkpoint fails before Isaac.
3. Obtain explicit approval for M6.1.
4. Execute M7-R3, review the offline bounded-checkpoint evidence, then execute M7-R4.
