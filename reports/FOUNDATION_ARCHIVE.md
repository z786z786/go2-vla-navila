# P0 Foundation Archive

Generated 2026-09-12 for the immutable P0 inventory baseline. W1 evidence is preserved; this archive applies the nine mandatory corrections in the dispatch order.

## §2 eight validated capabilities

| Capability | Evidence (home / mnt mapping) | SHA-256 | Producing command | Status |
|---|---|---|---|---|
| Go2 locomotion and velocity tracking | `/mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_ep1_v3` (claimed `/home/.../reports/r2r_ablation/...`) | manifest `7d2d9fc83294590563e10c66853f3979a1fd3a48f7c8220821175cfa6427405e` (file_manifest.json SHA-256); 249 files, 17,996,116 bytes (248 entries totaling 17,961,385 + file_manifest.json 34,731) | MISSING — 日志无 argv；manifest collector_sha256 e734e0b1…f3b82e 与树内任何现存脚本均不匹配（已查 collect_r2r_ablation.py 71eea6d5…、collect_r2r_continuous_v1.py 71eea6d5…（与前者字节相同）、collect_r2r_ablation.py.pre_platform_sync_20260912 147c6868…），产生该证据的 collector 源码已不在仓库内 | VERIFIED |
| R2R-VLNCE full deployment | `/home/wxh/go2_short_vln/reports/r2r_vlnce/audit_report.md` (no `/mnt` counterpart) | `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34` (`train.json.gz`; 10,819 episodes / 61 scenes via `reports/split_disjoint_check.json`) | n/a — 外部数据集 R2R-VLNCE v1-3，非本仓库生成；溯源为 train.json.gz SHA-256 f411066b…56a34 | PARTIAL (capability VERIFIED; audit report MISSING; “6744 file checks” MISSING; “two-version SHA-256 complete” MISSING) |
| Habitat ↔ Isaac coordinate conversion | `/home/wxh/go2_short_vln/scripts/validate_r2r_episode_load.py` | `d0b5c2787cf089113afc85e7726679d09c925a0323ac1f85038ed072e9a1b1ce` | n/a — 源码文件，非生成产物 | VERIFIED |
| Path Expert | `/home/wxh/go2_short_vln/scripts/collect_r2r_ablation.py` | `71eea6d5f24d70ac334192f4a632d8781be1fce255ca4efc0e38b585f377f67e` | n/a — 源码文件，非生成产物 | VERIFIED |
| Collection chain + audit | `/home/wxh/go2_short_vln/scripts/audit_r2r_ablation.py` | `d235a679e97efd767cfc81b2f9508322b06ef469dba869f91be2f764dbdb8f63` | n/a — 源码文件，非生成产物 | VERIFIED |
| Ablation array interface | `/mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_ep1_v3/ablation_arrays.npz` | `ae43c4ba37026fe5854175f85be473a63e01d17cca39c277a2170fdfe5cabd0b` | MISSING — 日志无 argv；manifest collector_sha256 e734e0b1…f3b82e 与树内任何现存脚本均不匹配（已查 collect_r2r_ablation.py 71eea6d5…、collect_r2r_continuous_v1.py 71eea6d5…（与前者字节相同）、collect_r2r_ablation.py.pre_platform_sync_20260912 147c6868…），产生该证据的 collector 源码已不在仓库内 | VERIFIED |
| GPU sharing governance | `/home/wxh/go2_short_vln/src/dual_target/gpu_wait.py` | `cd82d3dceb4ed9d4a9234e10ec7104fce6b9f4a7c671e73237f73258842c2f14` | n/a — 源码文件，非生成产物 | VERIFIED |
| Minimal controllable scene | `/home/wxh/go2_short_vln/config/dual_target_v1/task_contract.lock.json` | `ae040c5b66c7405197874d9d24408f03736606704c5d3cd278ad5827af5e0e14` | n/a — 配置文件，非生成产物 | VERIFIED |

## Producing-command provenance

(a) The all61 batch command is fully recoverable from the `command` field in `20260910_all61_*_guard.json`; for example, ep1023 records `/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python /home/wxh/go2_short_vln/scripts/collect_r2r_continuous_v1.py --expert continuous --output /mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_all61_ep1023 --episode-id 1023 --seed 20260910 --max-seconds 120 --load_run 2024-09-25_23-22-02 --headless --enable_cameras`. This belongs to all61 ep1023 and does not belong to any row in §2.

(b) `collect_r2r_ablation.py` and `collect_r2r_continuous_v1.py` are byte-identical; legacy/continuous are distinguished by the runtime `--expert` flag.

(c) The ep1_v3 collector source cannot be recovered, so P2 cannot rerun ep1 with the original collector.

## Corrected measurements and constraints

1. `ablation_arrays.npz` contains `command_raw` and `command_applied`, shape `(N,3)` float32 `[vx,vy,wz]`. ep1_v3 raw wz `[-2.536474,+1.036563]`, maxabs `2.536474`; applied `[-0.5,+0.5]`. continuous_v2 raw maxabs `2.504359`, applied ±0.5. Thus ±0.5 applies only to applied and clipping occurred.
2. Raw/applied vx maxima are 0.349996 (ep1_v3) and 0.349925 (continuous_v2); contract is [0,0.5], leaving `(0.35,0.5]` unsupervised. P1 decision item.
3. Stop counts are source-dependent: actions.jsonl 13 over 196 intervals; NPZ 12 over 195 complete intervals; continuous_v2 is 12 in both.
4. all61 `training_eligible` is false for all 60 completed samples; 34 successes are not 34 trainable samples.
5. all61 `maximum_contact_force_n`: 54 completed samples are 0.0 and 6 failed samples are null, supporting the X1 contact-signal limitation.
6. all61 `stop_positive_actions` has outlier 113; all others are 0–18. P4 QC must investigate latch/segmentation.
7. Credential scan has zero real leaks. The 19 T5 hits are false positives: 13 `sk-` substrings in `task-`, 6 self-referential scan-pattern hits.
8. `nvidia-smi` works: RTX 3090 38 MiB / 24576 MiB, zero compute processes and no non-project use. The earlier MISSING was a Codex sandbox artifact.
9. The official 1077 queue was not stopped by P0: progress was `completed=4`, `planned=1077`, `FAILED_NEEDS_REVIEW`; it failed itself at 4/1077. Root cause is MISSING because the `/tmp` log no longer exists.

## Known missing and non-recoverable items

The claimed `reports/r2r_ablation/20260910_ep1_v3/` is a path mapping error; evidence exists under `/mnt/wxh/go2_short_vln/outputs/...`. `reports/r2r_vlnce/audit_report.md` and `reports/OFFICIAL_NAVILA_VELOCITY_0911.md` are absent as claimed evidence and remain MISSING. all61 is not independently QC-approved: scene `XcA2TqTSSAj` is unavailable, six scenes have only failed warmups, and all training eligibility flags are false. ep1065 warmup failed; retain the failure and do not infer a repair. Official queue failure root cause is uninvestigated (missing `/tmp` log).

## Appendix A — historical asset disposition

- **A.1 M0–M7 short VLN:** retained at `reports/MILESTONES.md`; not reused because 30-D state, no stop head, and bounded codec conflict with the new contract.
- **A.2 M7-N semantic short-route:** retained in `reports/M7_N_N0_DESIGN.md` and `reports/M7_N_N1_GEOMETRY_PREVIEW.md`; not reused as training data because semantic mesh assets are missing and candidates remain pending.
- **A.3 official NaVILA velocity benchmark:** retained at `/mnt/wxh/go2_short_vln/outputs/official_navila_velocity_0911_v1/` and related reports where present; not reused because its constant-zero state/raw-stop interface is incompatible. It failed autonomously at 4/1077, not P0-stopped.
- **A.4 dual_target:** retained and used for pre-R2R sanity checks (`config/dual_target_v1/task_contract.lock.json`).
- **A.5 diagnostic assets:** retained; methods are reused (temporal alignment, action distribution, codec analysis), while old checkpoints/data are not reused.

## Git inclusion decisions

`data/short_vln_v1.json` is 4.8 MB, below the 10 MB file limit, and is included in the baseline after removing the blanket `data/` ignore. SHA-256: `88255ffb8e175372048c4948531d7ad059a9d8aa9217b7be95e2f60f520b34b6`. `third_party/` contains only symlinks `IsaacLab` and `NaVILA-Bench` targeting `/mnt`; it remains ignored. `.gitignore` uses `*.token`, `*credentials*.json`, and `*.key`; the broad `*token*` trap was removed.
