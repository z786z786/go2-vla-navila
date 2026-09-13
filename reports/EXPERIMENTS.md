# Experiments

## EXP-001 — M6 SmolVLA Training Bring-Up

### Hypothesis

The M5 Go2 short-VLN LeRobot dataset can drive the pinned SmolVLA training stack on one RTX 3090, overfit its two training episodes, complete a stable 2,000-step run, and reload a checkpoint for finite `50×3` action inference.

### Baseline

`lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`, with the required SmolVLM2 base cached at snapshot `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

### Change

Replace only the pretrained checkpoint's robot-specific feature declaration with the M5 interface: front RGB, 30-D state, task text, and 3-D `[vx, vy, wz]` action. Retain the pinned architecture, frozen VLM/vision encoder, and trainable action expert/state projection.

### Controlled Variables

Seed `20260831`; batch size 1; chunk size 50; PyAV; no AMP, PEFT, image augmentation, WandB, Hub push, Isaac, or M7 integration. Every successful stage ran with Hugging Face and Transformers offline modes enabled.

### Dataset / Split

M5 `short_vln_v1`: train 2 episodes/847 frames, seen-val 2/1,244, unseen-test 1/355. Training uses train only; the smoke run evaluates fixed windows from seen-val without optimizing on them.

### Command

```text
bash /home/wxh/go2_short_vln/scripts/m6_run_stage.sh single-batch
bash /home/wxh/go2_short_vln/scripts/m6_run_stage.sh tiny-overfit
bash /home/wxh/go2_short_vln/scripts/m6_run_stage.sh smoke
```

### Metrics

- Stage A: finite loss/gradient, optimizer update, tensor shapes, step time, peak VRAM.
- Stage B: first/last 25-step loss ratio and fixed-noise train-probe action MAE ratio.
- Stage C: 2,000-step numerical stability, fixed seen-val loss every 500 steps, throughput, checkpoint integrity/reload, and final action shape.

### Result

PASS. Stage A loss was 0.443519 and the update took 0.840 s. Tiny overfit loss ratio was 0.3462 and probe MAE ratio was 0.3526. The independent smoke run completed 2,000 steps at 0.1496 s/step and 6.71 samples/s; its loss ratio was 0.2369. Final checkpoint reload emitted finite `1×50×3` actions and reduced fixed train-probe MAE from 0.201014 to 0.080213. External peak VRAM was 2,209 MiB.

### Failure Cases

Two retained Stage A preflight failures occurred before any update: direct concrete-config loading failed because typed LeRobot checkpoints require registry dispatch, and the next attempt waited for an uncached SmolVLM2 base. The first SmolVLM2 download then exposed Xet CAS 401 behavior through the mirror; forcing standard HTTP with `HF_HUB_DISABLE_XET=1` resolved it. All successful stages subsequently ran offline.

### Conclusion

The data/model/optimizer/checkpoint pipeline is operational and can memorize the tiny M5 gate dataset. The experiment does not establish seen/unseen navigation performance or lateral-control learning.

### Next Step

Stop at M6. On explicit `CONTINUE M7`, integrate the final smoke checkpoint with an isolated Isaac client and SmolVLA inference server, preserving the M5 input allowlist and expert-information leakage ban.

## EXP-002 — M7 SmolVLA Isolated Closed-Loop First Gate

### Hypothesis

The M6 final smoke checkpoint can autonomously complete the training straight route when it receives only current rotated RGB, M5's 30-D state, and the short instruction through a local Unix socket, while remaining within the required raw velocity-action range.

### Controlled Variables

Checkpoint `step_002000` SHA-256 `8e72db4f1bf46c12dc3eb0f6e58b2cfc18d97e448dea1ecd9116d439a03145dd`; route `short_vln_v1_0004`; request timeout 10 s; JPEG quality 90; chunk `50×3`; execute/replan period 10 frames at 50 Hz; max 1,500 frames; success radius 0.5 m for 10 consecutive official-evaluator frames; action application bounds `vx=[0,0.5]`, `vy=0`, `wz=[-0.5,0.5]`; fixed episode/replan derived inference seeds. The PD planner was not in the policy rollout.

### Command

```text
bash /home/wxh/go2_short_vln/scripts/m7_run_gate.sh \
  /mnt/wxh/go2_short_vln/outputs/m7/gate_20260901T084954+0800
```

### Result

The isolated policy completed the official route: 421 frames, 42 SmolVLA requests, official `success=1.0`, `SPL=1.0`, final `DistanceToGoal=0.279893 m`, no collision, no bad orientation, mean/P95/max socket round-trip latency `371.1/415.8/1183.8 ms`, and a decodable 421-frame MP4. The request/response audit, 30-D state checks, finite-value checks, applied-action bounds, no-leakage checks, trajectory plot, and video checks all passed. The matching M4 oracle also succeeded (final navigation error 0.454139 m) with matching planner/dataset hashes.

### Failure Case

FAIL under the M7 acceptance rule. M7-R1 recomputation corrected the original wording: 37 of 2,100 raw action vectors were out of range, distributed across 10 of 42 chunks—not 37 of 42 chunks. Fifteen of 420 first-10 action positions required clipping; 22 violations were in discarded chunk tails. The breakdown is 3 `vx<0` and 34 `wz<-0.5`, with no `vy`, positive-`vx`, or positive-`wz` violations. The applied safe commands still completed the route, but raw-range compliance is explicitly required, so the range-compliant autonomous-success count is zero while the official navigation-success count is one. No model training, range relaxation, PD action interception, or goal/path input was used to change this outcome.

### Resources and Artifacts

The server launch gate saw 24,071 MiB free VRAM and the sampled peak was 5,637 MiB; Isaac-client peak RSS was 7,579,088 KiB. Evidence is in `/mnt/wxh/go2_short_vln/outputs/m7/gate_20260901T084954+0800`, including `first_gate_report.json` SHA-256 `e7ed7775eefbb5e194cb5ccb80c5c20cff20e6471d17204dd0d766942b01b5d2`, the summary SHA-256 `0971940689567b2931f1fb6fe4e9505bac5557755dbca7189f7f556e334c7747`, action/trajectory plot, raw/applied logs, and video.

### M7-R1 Acceptance-Audit Correction

M7-R1 added a shared action-range counter and made official navigation success distinct from range-compliant autonomous success. The server-host suite passed 36 tests. The retained episode was copied, never overwritten, into `/mnt/wxh/go2_short_vln/outputs/m7/r1_recompute_20260901T102653+0800`; its independent recheck returned the expected failing gate (`official_navigation_success_count=1`, `range_compliant_autonomous_success_count=0`) and produced JSON SHA-256 `6fb655ba05971e151961165b939e7aac90778d0de498b984a12e47e4b9dbdf78`. The original report and summary hashes are unchanged. This is an accounting and audit repair only; it did not run Isaac, retrain the policy, modify the M6 checkpoint, relax bounds, or change policy inputs.

### M7-R2 Preflight and Checker Hard Gate

M7-R2 makes the real socket preflight a mandatory barrier after server readiness and before `m7_run_episode.sh`. It independently checks every raw action in the returned `50×3` block, protocol/schema validity, finite values, and the 10 s round-trip limit; a failing check exits nonzero. The independent rollout checker now recomputes response ranges, validates every policy step against its `(request_id, action_offset)` raw/applied safety transformation, compares the derived range counts with the client summary, requires the fixed M6 checkpoint hash and exact M4 oracle planner/dataset hashes, and requires the exact first/full route lists plus complete artifacts. The server-host suite passed 42 tests, compilation, and shell syntax validation.

The real command `bash -x /home/wxh/go2_short_vln/scripts/m7_run_gate.sh /mnt/wxh/go2_short_vln/outputs/m7/r2_preflight_20260901T111500+0800` deliberately exited `1`: frame 0 of the stored `short_vln_v1_0004` oracle observation produced 21/50 raw violations (all `vx<0`; 7 execute-window and 14 tail). The response was otherwise valid, used checkpoint SHA-256 `8e72db4f1bf46c12dc3eb0f6e58b2cfc18d97e448dea1ecd9116d439a03145dd`, and completed in 1,017.6 ms. The preflight artifact SHA-256 is `60b46096be17ab0496bba30d0bee00afe867a62fb4d168254fc4282bdcbf79e9`; no `short_vln_v1_0004` episode directory was created and the socket cleanup completed. This confirms the fixed checkpoint is now rejected before Isaac, without retraining or changing the policy interface.

### M7-R3 Bounded M6.1 Checkpoint

With explicit authorization, M6.1 fine-tuned the M6 `step_002000` weights into the new immutable root `/mnt/wxh/go2_short_vln/outputs/m6_1/smoke_20260901T104846+0800`; the source checkpoint was never changed. Training encodes every physical action before normalization and replaces only the action statistics, calculated from all 847 train frames. The saved codec uses `vx=0.25(tanh(zx)+1)`, forced `vy=0`, and `wz=0.5*tanh(zw)`, with epsilon `1e-4` at the physical boundaries. Its metadata is saved beside the processors and records the source M6 hash. At inference, the server decodes the postprocessor latent output before the standard action-safety gate; latent actions remain server-audit-only.

The full 2,000-step run passed: first/last-25 loss `0.258185/0.153843` (ratio `0.595864`), decoded physical probe MAE `0.072032`, and fresh reload emitted finite `1×50×3` actions inside `[0,0.5]×{0}×[-0.5,0.5]`. The final model SHA-256 is `facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7`; training report SHA-256 is `9926f6cdd726e691d967cec8bd8692778aaf385ca4e28248b55fcb6dc20e600f`. Two fixed-seed real Unix-socket preflights passed identically with zero raw violations; their wall latencies were 997.0 and 321.4 ms. No Isaac process was started. The initial launcher import failure and an incomplete direct socket test (short readiness timeout and missing offline variables) occurred before any usable M6.1 evidence, were corrected, and are retained as integration notes only.

### M7-R4 Final Closed-Loop Gate

The M6.1 server passed the real preflight and then ran the required first route `short_vln_v1_0004` in a new immutable root. The policy interface, M4 oracle hashes, timeout, replanning period, safety bounds, and official evaluator were unchanged. All integration checks passed: no policy-input leakage, correct checkpoint provenance, valid action-offset mappings, finite states/actions, zero raw-range violations across all 7,500 response actions, safe applications, complete JSONL/RGB/video/plot artifacts, and a decodable 1,500-frame MP4. Mean/P95/max socket latency was `407.2/571.9/1495.3` ms.

The route nevertheless failed its policy-performance gate: it reached the 1,500-frame limit with official `success=0`, `SPL=0`, final navigation error `1.030050 m`, no collision, and no bad orientation. Consequently the first-stage independent checker failed because there was no raw-range-compliant official success; the remaining four routes were not started. This is a policy-navigation failure, not an integration or raw-action-range failure. M7 remains FAIL and no M8 work is started.

### M7 Offline Training-Set Action Replay

To separate policy fitting from closed-loop distribution shift, the bounded M6.1 checkpoint was evaluated without Isaac or the Unix socket on the exact two training episodes (`short_vln_v1_0000`, 427 frames; `short_vln_v1_0004`, 420 frames). Each of the 847 saved GT observations supplied only its LeRobot RGB, M5-compatible 30-D state, and instruction. Three independently derived but repeatable noise seeds (`20260831/20260832/20260833`) each produced a physical decoded `50x3` chunk. Every physical action was finite and in `[0,0.5]x{0}x[-0.5,0.5]`; the archive retains all decoded and latent blocks, GT blocks, and padding masks.

The ensemble first-step comparison (847 pairs) gives `vx` MSE/MAE/Pearson/R2 `0.002506/0.037369/0.861076/0.728022` and `wz` `0.032434/0.118126/0.856329/0.675405`; full valid-chunk comparison (39,900 pairs per seed) gives `vx` `0.002263/0.037015/0.863266/0.695499` and `wz` `0.031245/0.115402/0.867968/0.703378`. `vy` is identically zero in both data and model, so its correlation is undefined rather than interpreted as learned lateral control. First-step `wz` sign accuracy at `|wz_gt|>=0.05` is `0.707143`, above the `0.678571` majority-sign baseline. Model/expert first-step saturation ratios are close (`any=0.432113/0.452184`), and action-L2 correlation is `0.809700`.

This establishes genuine, time-aligned training-set action prediction, not merely matching an action histogram. The per-episode caveat is important: the first-gate route `0004` has high `vx` fit (`r=0.904585`, `R2=0.788216`) but weak `wz` explanatory power (`r=0.644092`, `R2=-0.493863`), while turn-containing `0000` has positive `wz` R2 (`0.663262`). The time-series plots show that the seed ensemble follows the expert's coarse command structure but has substantial per-frame yaw noise. Therefore the R4 closed-loop failure is consistent with error accumulation and sampling sensitivity under state drift, despite meaningful teacher-forced training-set fitting; it is not evidence of a socket, codec, or raw-range regression.

Evidence root: `/mnt/wxh/go2_short_vln/outputs/m7/offline_train_action_eval_20260901T120600+0800`. The real offline command took 683.76 s and saved `predictions.npz` SHA-256 `834cb049f2dda4bd74498ebd68129f3c3c160cbec98b45f091d9ade775d0292e`, `metrics.json` `ccca94d1302ec57cea55a65729c74e4a4bffea5657741b10900b40e2847b153b`, and report `a98245cfd2c2c8e1bebbdf5dc8d8ba4385336528fe566b8a92f639faa026bc73`. The server-host regression suite was 43/43 PASS; the additional offline evaluator source hashes are `evaluate_train_actions.py` `7cda15a12b35dbc2b2bde805b9f3eaf3f48a0c40491d047e68adce99f30affaa`, test `581b6917b0381325e024f8d01c667d5abc2cd1f77d0bcc38fbf0db490637d1c6`, and launcher `6eef8e7a4c6f3d959beb5ddfdb2b33fe1571d3b1306e9b82b117af59628e29d2`.

### Conclusion

The M7 integration is operational, and M6.1 restores raw-range compliance, but M7 is not accepted because the first bounded-checkpoint closed-loop route did not reach the official success radius. The first-gate rule correctly prevented the remaining four rollouts and M8 is not entered.
