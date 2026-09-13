# Project Milestones

## M0 — Environment Audit

Status: PASS  
Commit: N/A — `/home/wxh/go2_short_vln` was not an existing Git repository, and M0 did not initialize one.  
Date: 2026-08-30  
Commands: Complete command log is recorded in [`ENVIRONMENT.md`](ENVIRONMENT.md#commands-executed).  
Artifacts: `/home/wxh/go2_short_vln/reports/ENVIRONMENT.md`, `/home/wxh/go2_short_vln/reports/MILESTONES.md`  
Results: RTX 3090 and `nvidia-smi` verified; driver 580.173.02; system CUDA toolkit 12.2; disk, CPU, RAM, Conda, Python, PyTorch, Isaac Sim, Isaac Lab, NaVILA-Bench, LeRobot, and SmolVLA status documented.  
Peak VRAM: N/A for audit-only M0; observed snapshot was 9623 MiB used and 14493 MiB free of 24576 MiB.  
Known Issues: Root filesystem is 95% used; installed Isaac Sim 5.1 / Isaac Lab 2.3.1 does not match NaVILA-Bench's documented Isaac Sim 4.1 / Isaac Lab 1.1 stack; GPU was occupied by unrelated workloads; installed Isaac Lab lacks Git metadata; runbook SSH alias differs from local configuration.  
Next Risks: Before M1, choose `/mnt` placement for large assets and isolated environments, re-check free VRAM, and preserve current driver, CUDA, and Isaac installations.

### Acceptance

- [x] RTX 3090 detected
- [x] `nvidia-smi` normal
- [x] CUDA state explicit
- [x] Disk state explicit
- [x] Driver unchanged
- [x] System CUDA unchanged
- [x] `ENVIRONMENT.md` created

M0 passed. STOP and wait for `CONTINUE M1`.

## M1 — NaVILA-Bench Go2 Matterport PD Planner

Status: PASS — official NaVILA-Bench Go2 Matterport PD-planner episode 0 completed headlessly with cameras.  
Commit: N/A — workspace remains uninitialized by design.  
Date: 2026-08-31  
Commands: Source/installation/preflight logs remain under `/mnt/wxh/go2_short_vln/outputs/m1/logs/`, including `source-install.log`, `environment-install.log`, `isaacsim-4.1-install.log`, and `preflight-20260830T065222Z/summary.txt`. Final command: `bash /home/wxh/go2_short_vln/scripts/m1_run_pd_planner.sh`, which invoked the byte-identical official `NaVILA-Bench/scripts/demo_planner.py` with `--task=go2_matterport_vision --history_length=9 --load_run=2024-09-25_23-22-02 --num_envs=1 --episode_index=0 --headless --enable_cameras`; final complete-rollout log: `/mnt/wxh/go2_short_vln/outputs/m1/logs/pd-planner-rollout-20260831T023819+0800.log`.  
Artifacts: NaVILA-Bench checkout `/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench` at `e9d2db12ce5788c0f987d734c0094100b6bc0d3a`; modified Isaac Lab checkout `/mnt/wxh/go2_short_vln/third_party/IsaacLab` at `4d558ec83878c4892a46591c85ba91ac9d3c1834` (1.1.0); isolated Python 3.10 / Isaac Sim 4.1 stack `/mnt/wxh/go2_short_vln/envs/conda/navila-isaac`; Matterport asset root `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac`; final screenshots `/mnt/wxh/go2_short_vln/outputs/m1/screenshot/20260831T023819+0800/`; playable evidence video `/mnt/wxh/go2_short_vln/outputs/m1/video/20260831T023819+0800/pd_planner_camera_evidence.mp4`.  
Results: Scene `zsNo4HB9uLZ` loaded; Unitree Go2 spawned; RGB camera emitted 512×512 RGB frames; the official PD planner produced nonzero linear/yaw commands and completed episode 0. Measurements: `path_length=7.473894219117938`, `distance_to_goal=0.5609713776607296`, `success=1.0`, `spl=1.0`, `oracle_navigation_error=0.5609713776607296`, `oracle_success=1.0`. The official planner SHA-256 remained `20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432`. The project-only capture wrapper used only a process-local local-asset-root / timeline compatibility layer and a camera side-channel; it did not modify the official planner source or planner logic. LeRobot and SmolVLA are absent from the isolated environment.  
Peak VRAM: `14727 MiB / 24576 MiB` total GPU memory sampled at 2026-08-31 02:41:09 +08:00; GPU remains shared with unrelated workloads.  
Known Issues: Isaac Kit fast shutdown occurs after the official planner calls `simulation_app.close()`, before OpenCV releases the live writer. The raw `pd_planner_camera.mp4` is retained but lacks an MP4 `moov` atom. A valid 17.0-second H.264 evidence video (512×512, 34 frames, SHA-256 `5e71647e61c6c01a0340347c8146d9a743d86e2885517975dbd143ea3e1af452`) was generated solely from the saved numbered rollout screenshots; no simulator/planner behavior was altered. Camera frames are rotated by the official planner's `cv2.rotate` call.  
Protection check: `/home/wxh/isaacsim` and `/home/wxh/IsaacLab` remain directories resolving to themselves; their protected existing links were not moved, modified, uninstalled, or relinked. All M1 environments, assets, caches, logs, and evidence are under `/mnt/wxh/go2_short_vln`; final free space is 51 GB on `/` and 2.1 TB on `/mnt`. No driver, system CUDA, global Conda/Git, or global proxy configuration was changed.  

### M1 Acceptance

- [x] A. Matterport scene loads
- [x] B. Unitree Go2 spawns
- [x] C. RGB camera valid
- [x] D. Go2 low-level locomotion works
- [x] E. PD planner follows reference path
- [x] F. Complete trajectory runs
- [x] G. Video or equivalent visual evidence saved
- [x] H. Peak VRAM recorded for rollout
- [x] I. Existing Isaac Sim 5.1 / Isaac Lab 2.3.1 unchanged

M1 passed. STOP and wait for explicit M2 direction.

## M2 — Episode & Reference Path Inspector

Status: PASS — the real VLN-CE-Isaac schema is parsed directly, arbitrary episode IDs are inspectable, reference paths are visualized, and a 100-episode scene-stratified statistical sample is reported.  
Commit: N/A — workspace remains uninitialized by design.  
Date: 2026-08-31  
Commands: Main artifact command: `/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python src/episodes/inspect_episode.py --episode-id 1 --sample-size 100 --seed 20260831 --major-turn-degrees 45`. Arbitrary-ID check: the same command with `--episode-id 1833 --no-plot`. Remote syntax and CLI checks used `python -m py_compile src/episodes/inspect_episode.py` and `python src/episodes/inspect_episode.py --help`. Independent assertions loaded all episodes through the module and checked count, ID uniqueness, uniform keys, pose shapes, start/reference-start equality, goal/reference-end equality, episode 1 geometry, episode 1833 lookup, report content, and Pillow PNG verification; result is saved in `outputs/m2/logs/validation.log`. Resource command: `/usr/bin/time -v /mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python src/episodes/inspect_episode.py --episode-id 1 --sample-size 100 --seed 20260831 --major-turn-degrees 45`.  
Artifacts: Inspector `/home/wxh/go2_short_vln/src/episodes/inspect_episode.py`; report `/home/wxh/go2_short_vln/reports/episode_statistics.md`; visualization `/mnt/wxh/go2_short_vln/outputs/m2/episode_1_path.png`; metadata `/mnt/wxh/go2_short_vln/outputs/m2/episode_1_metadata.json` and `episode_1833_metadata.json`; execution, validation, and resource logs under `/mnt/wxh/go2_short_vln/outputs/m2/logs/`.  
Results: The gzip JSON has one top-level field, `episodes`, containing 1077 episodes from 11 scenes. All 1077 IDs are unique; episode, instruction, and goal key sets are uniform; no instruction is empty; no reference/expert path has fewer than two points; every start matches `reference_path[0]` and every configured goal matches `reference_path[-1]` within 1e-6 m. The observed episode fields are `episode_id`, `episode_new_id`, `trajectory_id`, `scene_id`, `start_position`, `start_rotation`, `info`, `goals`, `instruction`, `reference_path`, `gt_locations`, `gt_actions`, and `gt_forward_steps`. The 100-episode deterministic sample covers all 11 scenes. Its reference-path length median/mean is 9.17/9.45 m, waypoint-count median/mean is 6.0/5.8, and major-turn-count median/mean is 1.0/1.2. Episode 1 has a 8.7038 m reference path, 6 sparse waypoints, 33 expert waypoints, 6.6726 m start-goal distance, and one 88.03-degree major turn. Episode ID 1833 at list index 1076 also parsed successfully. `reference_path` is treated as the sparse task route; `gt_locations` is kept separate as the dense expert/metric path used by the official code. A major turn is a wrapped XY heading change of at least 45 degrees.  
CPU/RAM/GPU: The full inspector + report + plot command completed in 0.96 s wall time, used 731% CPU, and peaked at 93,280 KiB RSS. M2 does not import or start Isaac Sim and does not use the GPU. The contemporaneous GPU snapshot was 6749 MiB / 24576 MiB in use by pre-existing shared workloads; system memory snapshot was 9.3 GiB used and 490 GiB available of 503 GiB.  
Known Issues: The 45-degree major-turn threshold is an explicit M2 analysis definition, not a field supplied by the dataset. Original routes are longer than the V1 1–4 m target (the 100-episode sample minimum is 5.19 m), so M3 must construct auditable subsegments. Full-route natural-language instructions cannot be copied unchanged onto arbitrary cropped paths. The inspector does not render a Matterport floorplan; its path figure is a metric XY top-down plot plus height profile, which is sufficient for geometry inspection without launching Isaac.  
Next Risks: M3 must preserve scene-level split isolation, retain source index ranges, avoid fabricated landmark labels, and decide how to generate truthful short instructions when a cropped segment is only supported geometrically.

### M2 Acceptance

- [x] Any episode can be parsed by `episode_id` or list index
- [x] Reference-path and start-goal lengths are computed
- [x] Reference path is visualized
- [x] Sparse reference waypoints and dense expert waypoints are distinguished
- [x] Short-VLN-required fields are explicit
- [x] At least 50 episodes are sampled (100, covering all 11 scenes)
- [x] `reports/episode_statistics.md` created
- [x] Validation and resource logs saved
- [x] No short episode was constructed

M2 passed. STOP and wait for explicit `CONTINUE M3`.

## M3 — Short-Horizon VLN Episode Builder

Status: PASS — 1077 auditable 1–4 m short-VLN episodes were generated from contiguous official expert-path slices, with at most one major turn, geometry-supported instructions, and scene-isolated splits.  
Commit: N/A — workspace remains uninitialized by design.  
Date: 2026-08-31  
Commands: Final generation and resource command: `/usr/bin/time -v /mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python src/episodes/build_short_vln.py --dataset /mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz --output data/short_vln_v1.json --statistics-report reports/short_vln_v1_statistics.md --spot-check-report reports/m3_spot_check.md --seed 20260831 --min-path-length 1 --max-path-length 4 --major-turn-degrees 45 --turn-merge-distance 0.75 --turn-min-delta-degrees 5 --turn-min-leg-distance 0.75 --unseen-scene-fraction 0.20 --seen-val-episode-fraction 0.15 --spot-check-count 20`. Independent validation: `/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python src/episodes/check_short_vln.py --dataset data/short_vln_v1.json --source-dataset /mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz --spot-check-report reports/m3_spot_check.md`. Syntax checks used `python -m py_compile` for both scripts. Complete generation/resource output and independent validation output are saved under `outputs/m3/logs/`.  
Artifacts: Builder `/home/wxh/go2_short_vln/src/episodes/build_short_vln.py`; independent validator `/home/wxh/go2_short_vln/src/episodes/check_short_vln.py`; dataset `/home/wxh/go2_short_vln/data/short_vln_v1.json` (5,018,835 bytes, SHA-256 `88255ffb8e175372048c4948531d7ad059a9d8aa9217b7be95e2f60f520b34b6`); statistics `/home/wxh/go2_short_vln/reports/short_vln_v1_statistics.md`; 20-row audit `/home/wxh/go2_short_vln/reports/m3_spot_check.md`; logs `/mnt/wxh/go2_short_vln/outputs/m3/logs/build-short-vln.log` and `independent-validation.log`.  
Results: One short episode was generated for each of the 1077 source episodes across all 11 source scenes. Every `reference_path` is an exact contiguous slice of the corresponding official `gt_locations`, declared by `source_path_field` and an inclusive `source_path_index_range`; the original sparse path and full-route instruction remain audit-only metadata. Path length min/median/mean/max is 1.500/2.750/2.720/4.000 m. Major-turn count min/median/mean/max is 0/0/0.40/1. Instruction categories are 647 straight, 171 left-turn, and 259 right-turn. No external LLM or landmark label was used. Start WXYZ yaw is synthesized from the outgoing path tangent, and turn instructions contain audited pre/post-turn distances with both legs at least 0.75 m. Splits are 735 train, 129 seen-val, and 213 unseen-test. Train and seen-val share the same 9 houses but disjoint source episodes; unseen-test holds out the complete `2azQ1b91cZZ` and `8194nk5LbLH` houses, with zero train/unseen scene leakage. The independent validator recomputed source slicing, path lengths, tangent-aligned normalized quaternions, start/goal endpoints, instruction text/parameters, unique IDs, split isolation, and 20 spot-check rows; it passed. The 20 checks cover every source scene, all three splits, and all three instruction categories.  
CPU/RAM/GPU: Full generation, JSON write/reload, report generation, and internal validation completed in 9.57 s wall time at 99% CPU and 64,744 KiB peak RSS. M3 does not import Isaac or use the GPU. The final shared-machine snapshot was 6749 MiB / 24576 MiB GPU memory in use by pre-existing workloads and 9.3 GiB system RAM used with 490 GiB available of 503 GiB.  
Known Issues: The generated language is deliberately geometry-only and does not claim an RGB-visible landmark; M3 therefore guarantees instruction/path consistency but not visual landmark grounding. Internal segment start rotations are synthesized and require fresh simulator observations rather than reuse of source-start imagery. A major turn is defined as at least 45 degrees after merging nearby same-direction heading components within 0.75 m; this is an explicit auditable V1 rule, not native dataset metadata. No short episode has yet been rolled out in Isaac.  
Next Risks: M4 must load the synthesized internal start pose, treat the short `reference_path` as the PD expert path, use only the short `instruction` field, verify collision-free spawn/camera state on representative straight/left/right and stair cases, record the actual control frequency, and perform a small rollout smoke test before bulk collection.

### M3 Acceptance

- [x] JSON reload succeeds
- [x] 1077 candidates generated from multiple scenes (minimum was 100)
- [x] Every path length is within 1–4 m
- [x] Every route has at most one major turn
- [x] `train`, `seen-val`, and `unseen-test` are non-empty
- [x] No house appears in both train and unseen-test
- [x] Instructions are geometry-supported and do not fabricate landmarks
- [x] Original long-route instruction is never used as the short training instruction
- [x] 20 deterministic spot checks pass and cover every scene/split/category
- [x] Independent validation passes
- [x] No M4 collector or expert data was created

M3 passed. STOP and wait for explicit `CONTINUE M4`.

## M4 — Go2 PD Expert Collection

Status: PASS — 5/5 representative short-VLN episodes were collected successfully with complete RGB, robot state, expert velocity action, task text, timestamps, plots, and videos.  
Commit: N/A — workspace remains uninitialized by design.  
Date: 2026-08-31  
Artifacts: Collector `/home/wxh/go2_short_vln/src/collector/collect_expert.py`; independent checker `/home/wxh/go2_short_vln/src/collector/check_expert.py`; final gate data `/mnt/wxh/go2_short_vln/data/expert_v1/m4_gate_final_20260831`; per-episode `summary.json`, `steps.jsonl`, `sanity.json`, trajectory plot, and video; collection/check logs under `/mnt/wxh/go2_short_vln/outputs/m4/`.  
Results: The final gate contains 2 train episodes (427 and 420 frames), 2 seen-val episodes (577 and 667 frames), and 1 unseen-test episode (355 frames), for 2,446 frames at 50 Hz. Train and seen-val cover `zsNo4HB9uLZ` and `x8F5xyUWy9e`; unseen-test uses held-out `2azQ1b91cZZ`. Every episode completed successfully, had contiguous timestamps and frames, finite actions and state, command/motion consistency, normal termination, a saved trajectory plot, and a decodable video.  
Known Issue Found During M5: M4 declared `robot_state` as 33-D (`linear velocity[3], angular velocity[3], RPY[3], relative joint position[12], joint velocity[12]`) but all 2,446 serialized rows are 31-D. Isaac Lab returns RPY as a tuple and the collector's `[0]` retained only roll. M5 preserves the source audit trail and reconstructs complete RPY from the same frame's legitimate WXYZ body quaternion; no path, goal, or planner data is introduced.  

### M4 Acceptance

- [x] 5/5 episodes collected successfully
- [x] No missing RGB frames
- [x] Expert actions finite and in range
- [x] Command and motion consistent
- [x] Trajectory plots generated
- [x] Videos saved and decodable

M4 passed. STOP and wait for explicit `CONTINUE M5`.

## M5 — LeRobotDataset Conversion

Status: PASS — all 18 independent dataset and SmolVLA preprocessing checks passed; no training was started.  
Commit: N/A — workspace remains uninitialized by design.  
Date: 2026-08-31  
Commands: Conversion used `/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python src/dataset/convert_to_lerobot.py --expert-root /mnt/wxh/go2_short_vln/data/expert_v1/m4_gate_final_20260831 --output-root /mnt/wxh/go2_short_vln/data/lerobot/short_vln_v1`. Independent gate used the same environment with `src/dataset/check_dataset.py --dataset-root /mnt/wxh/go2_short_vln/data/lerobot/short_vln_v1 --report-dir /mnt/wxh/go2_short_vln/outputs/m5/report --seed 20260831 --batch-size 4 --video-backend pyav`. Full logs are `/mnt/wxh/go2_short_vln/outputs/m5/logs/conversion.log` and `dataset-check.log`.  
Environment: Isolated Python 3.12.14 environment `/mnt/wxh/go2_short_vln/envs/conda/smolvla`; LeRobot 0.6.0; PyTorch 2.11.0+cu130; Transformers 5.5.4; Datasets 4.8.5; OpenCV 4.13.0; Matplotlib 3.11.1. `pip check` reports no broken requirements, CUDA is available, and the RTX 3090 is detected. LeRobot source is the official v0.6.0 archive corresponding to commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`, archive SHA-256 `a4451766b7b450c7067a7e45671117511212a51c693b05aa711df2e101e4f7fd`.  
Artifacts: Converter `/home/wxh/go2_short_vln/src/dataset/convert_to_lerobot.py` (SHA-256 `81fc2374fc274fe8db692ce115a33bbfe6b19812e1699b0bf3e8f2f051d817f8`); checker `/home/wxh/go2_short_vln/src/dataset/check_dataset.py` (SHA-256 `99f60c78f1050ec7829ad984309f89c92b2feb2ee49a66e2b425a5390fcea300`); dataset `/mnt/wxh/go2_short_vln/data/lerobot/short_vln_v1` (23 MiB); conversion manifest SHA-256 `d55f2e388c62350a63c55cc7954208d9d77a16b838725965251a28e218f3d402`; check JSON SHA-256 `645f4f26babd5088d51b2c2b09476b0059fe7b00b52b407c9b7dba61f7f2d649`; reports and figures under `/mnt/wxh/go2_short_vln/outputs/m5/report/`.  
Mapping: Policy-facing fields are exactly `observation.images.front`, `observation.state`, `action`, and `task`. Images are 512×512 RGB video. State is 30-D: planar body linear velocity (2), yaw angular velocity (1), reconstructed base RPY (3), relative joint positions (12), and joint velocities (12). Action is exactly `[vx, vy, wz]`. Reference path, next waypoint, oracle heading, PD action history, goal pose/direction, and distance-to-goal values are not copied.  
Temporal Sampling: The dataset stores one 3-D action per frame. SmolVLA v0.6.0 resolves `action_delta_indices=0..49` at load time, producing DataLoader batches shaped `B×50×3`; observations use index `[0]`. No action chunk was manually duplicated.  
Results: 5 episodes and 2,446 frames converted at 50 Hz: train 2/847, seen-val 2/1,244, unseen-test 1/355. All LeRobotDataset loads, DataLoader batches, image/state/action shape/dtype/range checks, timestamp and episode-boundary checks, NaN/Inf checks, SmolVLA tokenizer/normalizer preprocessing, padding masks, and artifact checks passed (18/18). Action min/max is `[0.0, 0.0, -0.5]` / `[0.388357, 0.0, 0.5]`; mean is `[0.220189, 0.0, -0.044908]`. Train/seen-val share their two seen houses by design, while unseen-test `2azQ1b91cZZ` has zero overlap with both.  
Resource Use: Conversion completed in 50.50 s at 1,285,200 KiB peak RSS. Final independent check completed in 13.77 s at 1,141,652 KiB peak RSS. M5 did not start Isaac Sim or training; final shared GPU snapshot remained 6,749 / 24,576 MiB used.  
Known Issues: Hugging Face tokenizer access required the existing temporary reverse proxy; PyPI installation was faster via direct remote access. Hub access was unauthenticated but completed successfully. The first failed conversion left only a 3 KiB metadata file and was preserved at `/mnt/wxh/go2_short_vln/data/lerobot/short_vln_v1_failed_31d_20260831T1302` for auditability.  

### M5 Acceptance

- [x] LeRobotDataset loads for train, seen-val, and unseen-test
- [x] DataLoader batches successfully
- [x] SmolVLA preprocessing succeeds with language tokens and finite normalized tensors
- [x] Action dimension is 3
- [x] No NaN or Inf
- [x] Unseen-test scene is isolated
- [x] 10 random samples emitted
- [x] Action and episode-length histograms emitted
- [x] Scene split report emitted
- [x] No training started

M5 passed. STOP and wait for explicit `CONTINUE M6`.

## M6 — SmolVLA Training Bring-Up

Status: PASS — pretrained loading, single-batch optimization, 500-step tiny overfit, independent 2,000-step smoke training, checkpoint reload, and finite `50×3` action-chunk inference all passed; M7 was not started.  
Commit: N/A — workspace remains uninitialized by design.  
Date: 2026-08-31  
Commands: All stages used `/home/wxh/go2_short_vln/scripts/m6_run_stage.sh` in strict offline mode after caching the pinned models. Successful runs were `single-batch`, `tiny-overfit`, and `smoke`; their full stdout/resource logs and 1 Hz GPU samples are under `/mnt/wxh/go2_short_vln/outputs/m6/logs/`. The first two single-batch attempts are retained: the first exposed the required `PreTrainedConfig` registry dispatch for a typed checkpoint, and the second established that SmolVLA construction also requires the SmolVLM2 base weights. Neither attempt reached a training update.  
Environment and Models: Isolated `/mnt/wxh/go2_short_vln/envs/conda/smolvla` environment with LeRobot 0.6.0 and RTX 3090. SmolVLA source checkpoint `lerobot/smolvla_base` was pinned at revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`; local `model.safetensors` SHA-256 is `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`. Its required `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` base resolved to snapshot `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; base `model.safetensors` SHA-256 is `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`. Training then ran with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.  
Artifacts: Training/evaluation entry point `/home/wxh/go2_short_vln/src/smolvla/m6_training.py` (SHA-256 `8bc54a08c4f56bf6f058bbe6d67750567c868c76060ebb663780e349ddb84ebc`); stage wrapper `/home/wxh/go2_short_vln/scripts/m6_run_stage.sh` (SHA-256 `8ad1f92b49d22b3b9e73adedc8a71ac0b099fb57e1f1894a1d0c62c7d49ee5ad`). Successful reports are `/mnt/wxh/go2_short_vln/outputs/m6/single-batch_20260831T223753+0800/m6_report.json`, `/mnt/wxh/go2_short_vln/outputs/m6/tiny-overfit_20260831T223911+0800/m6_report.json`, and `/mnt/wxh/go2_short_vln/outputs/m6/smoke_20260831T224205+0800/m6_report.json`. The smoke directory also contains the complete 2,000-row history, training curve, and checkpoints at steps 500/1000/1500/2000; each checkpoint is about 1.319 GB and includes the policy, processors, optimizer, scheduler, and M6 metadata.  
Configuration: The pinned pretrained architecture was adapted from its original three-camera/6-D metadata to the audited M5 interface: one `observation.images.front`, 30-D `observation.state`, task text, and 3-D `[vx, vy, wz]` action. Action chunks are loader-sampled `50×3`; vision and VLM parameters remain frozen (`freeze_vision_encoder=true`, `train_expert_only=true`), while the action expert and state projection are trained. Batch size is 1, FP32 training/official BF16 frozen VLM loading is used, and AMP, PEFT, WandB, Hub push, and Isaac Sim are disabled. Tiny overfit used 50-step warmup and 500-step decay; the independent smoke run used the official 1,000-step warmup and 30,000-step decay schedule.  
Results: Single-batch input shapes were image `1×3×512×512`, state `1×1×30`, and action `1×50×3`; forward loss was `0.443519`, backward/gradient clipping/optimizer step succeeded, and the update took 0.840 s. Tiny overfit completed 500 steps: first/last 25-step mean loss was `0.790343/0.273638` (ratio `0.3462`), while fixed-noise demonstration action MAE improved from `0.201014` to `0.070884` (ratio `0.3526`, versus required maximum `0.8`). The independent smoke run completed all 2,000 steps in 326.44 s of core training and 350.46 s total, averaging 0.1496 s/step and 6.71 samples/s. Its first/last 25-step loss was `0.999238/0.236692` (ratio `0.2369`). Fixed seen-val losses at steps 500/1000/1500/2000 were `0.5197/1.2193/0.6346/0.5655`; all were finite. The final checkpoint reloaded in a fresh model, emitted finite `1×50×3` actions, and improved the fixed train-probe MAE from `0.201014` to `0.080213`.  
Resource Use: External GPU sampling peaked at 2,209 MiB in both training stages; PyTorch peak allocated/reserved memory was 1,713.9/1,830 MiB. Peak host RSS was 3,836,388 KiB for Stage A, 5,595,136 KiB for tiny overfit, and 5,524,252 KiB for smoke training. GPU returned to 45 MiB used after completion.  
Known Issues: Only two train episodes (847 frames), two seen-val episodes, and one unseen-test episode exist. M6 therefore validates the training pipeline and memorization capability, not navigation generalization. Seen-val loss is noisy and worsened near the peak learning rate before recovering; it must not be reported as a benchmark result. The dataset has zero `vy` variation, so the zero lateral-action MAE is not evidence that lateral control was learned. M4's 31-D serialization issue remains deferred until new expert collection because M5's audited quaternion reconstruction already fixes the current data.  
Next Risks: Before M7, use the exact saved pre/postprocessors and 30-D state convention; execute only the first configured portion of each `50×3` chunk under closed-loop replanning; enforce finite/range/latency checks; never expose reference paths, waypoints, goal direction, or PD state to SmolVLA; and keep Isaac and SmolVLA in isolated processes/environments.

### M6 Acceptance

- [x] Pinned pretrained SmolVLA and required SmolVLM2 base load offline
- [x] Single-batch forward, finite loss, backward, clipping, and optimizer step succeed
- [x] Tiny overfit passes both loss-ratio and fixed action-MAE gates
- [x] Independent 2,000-step smoke training completes without NaN, Inf, or OOM
- [x] Seen-val loss is sampled at four fixed checkpoints and remains finite
- [x] Final checkpoint reload succeeds in a fresh model
- [x] Reloaded policy emits finite `1×50×3` action chunks
- [x] Timing, throughput, CPU/RAM, checkpoint sizes, and peak VRAM are recorded
- [x] No Isaac Sim process or M7 closed-loop integration was started

M6 passed. STOP and wait for explicit `CONTINUE M7`.

## M7 — SmolVLA Closed-Loop Navigation Integration

Status: FAIL — the independent closed loop and official evaluator succeeded on the first gate route, but strict raw-action range compliance failed; the remaining four routes were deliberately not started and M8 must not begin.  
Commit: N/A — workspace remains uninitialized by design.  
Date: 2026-09-01  
Commands: Server and client were launched exclusively through `bash /home/wxh/go2_short_vln/scripts/m7_run_gate.sh /mnt/wxh/go2_short_vln/outputs/m7/gate_20260901T084954+0800`; it starts `/home/wxh/go2_short_vln/scripts/m7_run_server.sh`, then `/home/wxh/go2_short_vln/scripts/m7_run_episode.sh short_vln_v1_0004 ...`. The final server-host test gate was `/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python -m unittest discover -s tests -q` (30/30 PASS, including the real Unix-socket round trip), followed by `py_compile` and `bash -n` checks.  
Environment: Isaac client: `/mnt/wxh/go2_short_vln/envs/conda/navila-isaac` (Isaac Sim 4.1/Isaac Lab 1.1/NaVILA-Bench); inference server: `/mnt/wxh/go2_short_vln/envs/conda/smolvla` (LeRobot 0.6.0). The server required 20 GiB free VRAM before launch; the pre-launch free amount was 24,071 MiB. The final checkpoint was `/mnt/wxh/go2_short_vln/outputs/m6/smoke_20260831T224205+0800/checkpoints/step_002000`, SHA-256 `8e72db4f1bf46c12dc3eb0f6e58b2cfc18d97e448dea1ecd9116d439a03145dd`.  
Artifacts: Final gate root `/mnt/wxh/go2_short_vln/outputs/m7/gate_20260901T084954+0800`; first-gate report SHA-256 `e7ed7775eefbb5e194cb5ccb80c5c20cff20e6471d17204dd0d766942b01b5d2`; episode summary SHA-256 `0971940689567b2931f1fb6fe4e9505bac5557755dbca7189f7f556e334c7747`; socket-ready metadata SHA-256 `c9b71a7371dfdc32b482e3151c6449906c33fac59e4e849a1d1df3a0a5fd843e`. The episode has complete `steps.jsonl`, `requests.jsonl`, 421 RGB JPEGs, raw/applied actions, `sanity.json`, trajectory/action plot `m7_rollout_diagnostics.png`, and decodable 421-frame `rollout.mp4`.  
Implementation: A local Unix-socket service at `/tmp/go2_smolvla_m7.sock` accepts only 512×512 rotated RGB JPEG, M5-compatible 30-D state, short instruction, and audit-only episode/replan IDs. It returns the saved-processor checkpoint's finite `50×3` raw action chunk with per-stage latency. The Isaac client executes exactly 10 actions (0.2 s) then re-observes, saves all raw values, clips application to `[vx 0..0.5, vy=0, wz -0.5..0.5]`, and never uses PD navigation commands. Goal distance is read only from the official `DistanceToGoal` evaluator; ten consecutive evaluator frames within 0.5 m trigger a zero command and official stop/success.  
Results: `short_vln_v1_0004` completed autonomously in 421 frames / 42 replans. Official metrics were `success=1.0`, `spl=1.0`, `distance_to_goal=0.279893 m`, `oracle_navigation_error=0.279893 m`; collision and bad orientation were both false. All infrastructure checks passed: exact request schema, no oracle-information leakage, finite action/state values, safe applied actions, contiguous records, valid frames, and decoded MP4. Mean/P95/max inference round-trip latency was 371.1/415.8/1183.8 ms. The matching M4 PD oracle succeeded with final navigation error 0.454139 m; its planner hash and short-dataset hash matched (`20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432` and `88255ffb8e175372048c4948531d7ad059a9d8aa9217b7be95e2f60f520b34b6`). Peak sampled GPU memory was 5,637 MiB; peak Isaac-client RSS was 7,579,088 KiB.  
Failure Case: The retained `50×3` responses contain 37 out-of-range action vectors of 2,100 total; they occur in 10 of 42 chunks. Of the 420 first-10 action positions considered for execution, 15 were out of range and required clipping; the remaining 22 violations are in discarded tail positions. The violations are 3 `vx<0` and 34 `wz<-0.5`; there are no `vy`, positive-`vx`, or positive-`wz` violations. This invalidates the required raw-range condition even though the clipped closed loop reached the official goal. Official navigation is `PASS`; raw range and range-compliant autonomous success are both `FAIL`, so overall M7 remains `FAIL`.  
Compatibility Fixes Retained: M7 installs the long-timeline hook during `AppLauncher` initialization (the verified M4 lifecycle), mirrors the official `seed`, `use_cnn`, and `use_rnn` runtime parser fields for the frozen locomotion runner, persists startup errors as audit summaries, and runs the checker from the project directory. These are integration-only repairs; the model, action bounds, checkpoint, task, and PD-control ban were not changed.  
Original M7 source hashes: `src/inference/isaac_client.py` `295e4eea440eef61a8ed66677785d335d98588a699236907150e7c7d1ead800f`; `src/inference/check_rollout.py` `09bb2de399e2da00a4de385a92c45290443281621d1fb1984605e7b97e838cd8`; `scripts/m7_run_gate.sh` `c568061d4aa59d0047bc7e65924d137dfa8add9a00b3988ed0567606fd49e927`.  
M7-R1 correction: the old `raw_chunk_range_violation_count=37` was a sum of violating action vectors, not a count of violating chunks. M7-R1 introduces explicit vector, chunk, executed-window, tail-window, and per-dimension fields; it does not alter retained rollout JSONL, the checkpoint, control commands, or the original gate report.  
M7-R1 server verification: 36 server-host unit tests passed, followed by Python compilation and shell syntax checks. A copied, derived evidence root `/mnt/wxh/go2_short_vln/outputs/m7/r1_recompute_20260901T102653+0800` reran the checker without modifying the original episode. Its report confirms 42 chunks / 2,100 vectors / 37 violating vectors / 10 violating chunks / 420 execute-window positions / 15 execute-window violations / 1,680 discarded positions / 22 discarded violations. The derived JSON and Markdown report SHA-256 values are `6fb655ba05971e151961165b939e7aac90778d0de498b984a12e47e4b9dbdf78` and `353f77dddde33a5d22bcded4d710cdf0e580e5190fe8132269d71ffa3278e822`. The original first-gate report and episode summary retained SHA-256 values `e7ed7775eefbb5e194cb5ccb80c5c20cff20e6471d17204dd0d766942b01b5d2` and `0971940689567b2931f1fb6fe4e9505bac5557755dbca7189f7f556e334c7747`. Current R1 source hashes are `src/inference/action_audit.py` `e05a5daf1a8f950209d11717f6e00c7abee181c025c93818932a07220f455f31`, `src/inference/isaac_client.py` `438534e41693edc28b17dc9667c0b62f2db3b57482360379570c1aa0ad3e2bb4`, and `src/inference/check_rollout.py` `2b59f915dd1f74d23756346613e81c323123897b6731a8fe1d582618521cbe7d`.  
M7-R2 hard-gate verification: `src/inference/preflight.py` now independently validates the full returned `50×3` raw block and returns nonzero on protocol, timeout, non-finite, shape, or range failure; `scripts/m7_run_gate.sh` invokes it after server readiness and before any episode runner. The independent checker recomputes ranges, action-offset/safety mappings, summary counts, provenance hashes, artifacts, and exact first/full-stage IDs. The server-host suite passed 42 tests, followed by Python compilation and shell syntax checks. The real command `bash -x /home/wxh/go2_short_vln/scripts/m7_run_gate.sh /mnt/wxh/go2_short_vln/outputs/m7/r2_preflight_20260901T111500+0800` intentionally exited `1` before Isaac. Its saved M4 frame-0 request returned a valid block from the fixed checkpoint hash but had 21/50 raw violations (all `vx<0`; 7 in the first 10 and 14 in the discarded tail), so `preflight.json` recorded `passed=false`; wall latency was 1,017.6 ms. No `short_vln_v1_0004` episode directory was created and the Unix socket was removed by cleanup. Evidence hashes: preflight `60b46096be17ab0496bba30d0bee00afe867a62fb4d168254fc4282bdcbf79e9`, server-ready `1ca00591eab74d12ab812892e754e74835989a315460ecf34cb87c27421b214e`, server log `db0f325d3a17b04af752366fefb57d3fb5603e0e5ed0ad75b7d1a34281cb05d8`, console trace `e64271905a1bde9688a2d77c070ba67e636c7ae7b4673996a0bd4b2ff13d5ad4`. R2 source hashes are `preflight.py` `2757d416545125f70383f9a32c68c3eebbf2ce6340886034e478e71311f99b20`, `check_rollout.py` `73f4d4deab4de23dd196f1f48a07fef3f4e2cd96b7947eb824c4e16b03cb92d7`, and `m7_run_gate.sh` `d3aaade4e209b3e3f3b6a4e6e05072a604786878a16fdf2299d76493a21b7dc7`. No checkpoint, bounds, policy inputs, training, or historical evidence was changed.  
M7-R3 bounded M6.1 checkpoint: With explicit authorization, a new smoke fine-tune loaded (but did not modify) the M6 `step_002000` source and wrote only `/mnt/wxh/go2_short_vln/outputs/m6_1/smoke_20260901T104846+0800`. Physical targets are encoded before the action normalizer using `vx=0.25(tanh(zx)+1)`, `vy=0`, `wz=0.5*tanh(zw)` inverses with epsilon `1e-4`; replacement action statistics were calculated once from all 847 train frames. The checkpoint stores `bounded_action_codec.json` beside the processors, hashes the source M6 checkpoint, and the server requires this metadata for `m6_1` paths, decodes before returning policy actions, and stores latent actions only in its server log. The 2,000-step run passed with finite losses/gradients and fresh reload: first/last-25 loss `0.258185/0.153843` (ratio `0.595864`), decoded physical probe MAE `0.072032`, output `1×50×3`, range `[0.0000004,0.407157]×{0}×[-0.499999,0.499966]`, core train wall time 331.63 s (end-to-end 373.30 s), 2,209 MiB sampled GPU peak and 5,541,684 KiB peak RSS. The final model SHA-256 is `facc4a73b500e52468a3c57fa985656ca085ed482211e9b97bb13328b58748e7`; pre/postprocessor and codec hashes are `78423ba2bf38e113f59139327ea7d009272f97c60855a02da2e39f5cdf11a80a`, `47dbdc6d1be74f738990bbf2a3c66d5ddf639481482f3984fc766e3bf144e3aa`, and `143c8fce264be7906bd1aea11d3e502a4d67f40e7d8a691fac678f752657af27`. Two real offline socket preflights in `/mnt/wxh/go2_short_vln/outputs/m6_1/r3_socket_offline_20260901T110900+0800` passed deterministically with zero raw violations, valid `1×50×3` output, 997.0/321.4 ms wall latency, and no remaining socket; evidence hashes are `cedda0ae14f0557ca160e0049570a87f8c5f05cbd5779b54a241bcb9eb39d050`, `c0b40fb2a2c52f24132683ade3004a22a6009204739c023542c65a9a8f6ee515`, and request log `256554f873646ae869b6a8106c0164b9cd2b7c1896276bde67df240e211e801e`. An initial M6.1 launcher attempt failed before training because file-mode execution could not import the package; it was corrected to module-mode execution. A first direct socket test used a 90-second readiness window and omitted offline variables; it performed no request and was replaced by the final 180-second offline validation.  
M7-R4 final bounded-checkpoint gate: After 24,071 MiB free-VRAM, M6.1 checkpoint, M4 planner/dataset hash, 43-test, compile, and shell checks passed, the new root `/mnt/wxh/go2_short_vln/outputs/m7/gate_m61_20260901T112400+0800` ran only `short_vln_v1_0004`. Its real preflight passed (`50/50` finite, raw-range-compliant actions; SHA-256 `ba420cd00fa08f5ae33125f149a5f051db70fd4cfdc3757b4021c215db060011`). The closed loop retained all infrastructure checks: exact request allowlist/no leakage, response-action offset mapping, applied safety, M6.1 hash, M4 oracle hashes, RGB frames, action/trajectory plot, and decodable 1,500-frame MP4. It had 150 chunks/7,500 raw actions and zero violations in every raw/executed/tail dimension, with mean/P95/max round-trip latency 407.2/571.9/1495.3 ms; collision and bad orientation were false. However official navigation did not reach the 0.5 m success radius: it terminated at the immutable 1,500-frame limit with final navigation error 1.030050 m, official success/SPL 0, and range-compliant autonomous success false. The independent first-stage report therefore failed (JSON `69aa5fb94f5a8f882234dd515f7cbdce3abcb1a7556f945583ede510d6f15ae2`; summary `04e2e0bbeedbaf27302d671d6d5b6d014b38bb2cc31370b849edc1b337a934c8`). The first-gate rule correctly prevented `0000/0001/0003/0006`; no full-stage report was produced. An initial R4 output root stopped before service start because uploaded execute bits were absent; it contains no episode and is retained. The retry's source hashes are server `e9b10225a02a5641e09139d80d4d16edc54975eb61db250945c497799feb1643`, server wrapper `34f343621838cd0747231bfbf86918822f69836fc49e5acd2947a98dd5bc7647`, and gate wrapper `7b35ebf4d76b7d5c1f561723956e1f0978c81412bb12505c9bd9d6f1af24d767`. Socket cleanup and GPU release were verified.  
M7 offline training-set action replay: A separate offline evaluator loaded the fixed M6.1 checkpoint and its saved processors/codec, then fed only GT LeRobot RGB, M5 30-D state, and instruction for all 847 training observations (`0000`: 427, `0004`: 420). No Isaac process, socket, planner, goal field, or generated action feedback was used. Three fixed, episode/frame-derived noise seeds produced finite physical `50x3` chunks in bounds for every observation. Both requested alignments were retained: first step (847 pairs) and all valid action offsets (39,900 pairs per seed). Ensemble first-step `vx` MSE/MAE/Pearson/R2 were `0.002506/0.037369/0.861076/0.728022`; `wz` was `0.032434/0.118126/0.856329/0.675405`; full-chunk values were respectively `0.002263/0.037015/0.863266/0.695499` and `0.031245/0.115402/0.867968/0.703378`. The `wz` sign accuracy at `|GT|>=0.05` was `0.707143` versus the majority-sign baseline `0.678571`. On the failed M7-R4 first route specifically, `vx` fit is strong (`r=0.904585`, `R2=0.788216`) but `wz` R2 is negative (`-0.493863`); per-frame yaw sampling variance is visibly high. This establishes training-set time-aligned action prediction but identifies yaw stochasticity/distribution shift and rollout error accumulation—not protocol, action bounds, or codec—as the likely remaining closed-loop mechanism. Evidence is `/mnt/wxh/go2_short_vln/outputs/m7/offline_train_action_eval_20260901T120600+0800`, with metrics SHA-256 `ccca94d1302ec57cea55a65729c74e4a4bffea5657741b10900b40e2847b153b`, predictions SHA-256 `834cb049f2dda4bd74498ebd68129f3c3c160cbec98b45f091d9ade775d0292e`, all requested plots, and 43/43 server-host tests passing.

### M7 Acceptance

- [x] Protocol, state, action-safety, image, preflight, and rollout tests pass on the server
- [x] Exact M5 30-D state and RGB/JPEG protocol are used
- [x] No path, waypoint, goal, distance, PD, or unknown policy field reaches SmolVLA
- [x] At least one official autonomous success is recorded with complete logs and video
- [x] 10-frame replanning, finite-output, timeout, clipping, and zero-stop controls are exercised
- [x] R2 independent preflight rejects the current checkpoint before any Isaac episode starts
- [x] R2 checker enforces response/action mapping, summary counts, provenance hashes, artifacts, and exact gate-stage IDs
- [ ] Raw output is range compliant (37/2,100 action vectors violated across 10/42 chunks)
- [x] M6.1 R4 first-gate raw output is range compliant (0/7,500 vectors violated)
- [ ] At least one raw-range-compliant official navigation success (R4 `0004` timed out at 1.030 m)
- [ ] All five simple-task rollouts are complete (not run because the required first gate failed)
- [ ] M7 PASS

M7 failed the final acceptance gate: the bounded M6.1 checkpoint fixed raw-range compliance but did not achieve an official autonomous success on the required first route. STOP; do not enter M8.
