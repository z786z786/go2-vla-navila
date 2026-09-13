# 实验时间线子笔记

## 用途

这份子笔记专门记录：

- 每轮实验的目标
- 配置 / checkpoint / 数据集
- 预期结果
- 实际结果
- 结论
- 下一步动作

适合沉淀“从开始到现在”的逐轮实验脉络，而不是放架构细节。

## 建议结构

### 真实数据主线

- Stage 0 smoke
- Stage 1 projector-only
- Stage 1.5 validation gate
- Stage 1.6 delta/full-mask
- Stage 1.7A event-heavy real data

### 仿真数据主线

- sim-only baseline
- office-clean / projector-only
- anchor-residual
- turnpilot
- weighted subset
- targeted-v2

### 每轮实验记录模板

#### 实验名

- 日期：
- 目标：
- 配置：
- 数据：
- checkpoint：
- 关键指标：
- 结果判断：
- 下一步：

## 当前状态

- 当前总览仍集中在 `docs/llada_vla_go2_project_notes.md`。
- 如果后续实验日志继续增长，优先把逐轮结果迁到本文件。

## 2026-05-10 Memory-conditioned World Model MVP 准备

- 目标：为 MemoryVLA memory / retrieved memory representation 是否降低 world model latent prediction error 建立 offline benchmark。
- 新项目目录：`/home/wxh/memory_world_model`，代码与脚本均放在该目录。
- 大文件目录：统一放 `/mnt`；任务开始时 `/mnt` 可用约 3.2T。
- 关键路径：
  - feature cache: `/mnt/llada-vla-go2/memory_world_model/cache`
  - outputs/checkpoints/plots: `/mnt/llada-vla-go2/memory_world_model/outputs`
  - uv env: `/mnt/llada-vla-go2/envs/memory_world_model`，项目内 `.venv` 为 symlink
  - HF cache: `/mnt/hf_cache`
  - Torch cache: `/mnt/torch_cache`
  - models: `/mnt/models`
  - LIBERO data: `/mnt/libero`
- 已完成：MemoryVLA official repo clone 到 `/mnt/llada-vla-go2/external/MemoryVLA`，观测 commit `e4d27ec`。
- 未完成/阻塞：LIBERO clone 初次超时，重试命令被权限层拦截，未继续重试；已写入 `/home/wxh/memory_world_model/docs/remote_training_runbook.md` 作为远端/手动步骤。
- 已实现 MVP：
  - PyTorch 模块：WM-only、WM+history、WM+memory、Residual Memory WM。
  - feature cache schema，synthetic/pseudo-memory extractor，training/eval/plot scripts。
  - memory ablation：correct / shuffled / zero memory。
  - 远端迁移 runbook：`/home/wxh/memory_world_model/docs/remote_training_runbook.md`。
- 本地验证：只运行 smoke/dry-run，没有启动正式训练；`pytest` 通过，`scripts/run_all.sh configs/smoke.yaml dry-run` 生成 smoke checkpoints、results 和 plots。
- 重要限制：true MemoryVLA memory token extractor 目前是 adapter stub；第一阶段 MVP 使用 pseudo-memory = past K latent states。正式 LIBERO/MemoryVLA 训练应在远端接好 adapter 后启动。
- 2026-05-10 后续准备：已将 MVP 推进到可迁移远端并直接启动训练的状态。LIBERO hdf5 demos 完整，DINOv2 base 已下载并通过 local-load preflight；新增真实 LIBERO feature extraction 路径（DINOv2 -> z_t/z_future，pseudo-memory -> z_history）、`configs/libero_mvp.yaml`、`configs/dino_preflight.yaml`、`scripts/start_remote_training.sh`、`scripts/rsync_to_remote.sh`、`docs/remote_asset_manifest.json` 和更新版 `docs/remote_training_runbook.md`。本地只跑了 preflight/dry-run，未启动正式训练。
- 2026-05-10 review 结果：关键代码 review 后修复了 README 中错误的 `--synthetic` 正式训练命令、`rsync_to_remote.sh` 参数歧义、formal split 的 episode 泄漏风险、DINO latent dim mismatch 只 warning 的问题，并将 residual model 默认设为加载 WM-only 后冻结 base（`residual_freeze_base: true`）。复验：`pytest` 6 passed、`ruff check .` 通过、DINO preflight 通过；formal `libero_mvp` 预计 64,913 windows，按 episode-level split 为 train 52,243 / val 6,317 / test 6,353，2000 episode groups，泄漏 0。
- 2026-05-10 本地 RTX 3090 MVP 正式训练完成：使用 `configs/libero_mvp.yaml` 在本地 3090 上完成 DINOv2 feature extraction、4 个 world model 训练与评估。feature cache 位于 `/mnt/llada-vla-go2/memory_world_model/cache/libero_mvp`（train 2.7G、val 327M、test 329M）；checkpoints 位于 `/mnt/llada-vla-go2/memory_world_model/outputs/libero_mvp/checkpoints`；plots 位于 `/mnt/llada-vla-go2/memory_world_model/outputs/libero_mvp/plots`。结果：WM-only H-step MSE 0.183464；WM+history 0.181371（+1.14%）；WM+Memory 0.177826（+3.07%）；Residual Memory WM 0.158162（+13.79%）；Residual shuffled M 0.193331（-5.38%）；Residual zero M 0.198405（-8.14%）。初步结论：pseudo-memory 明显降低 latent prediction error，residual correction 有效；shuffled/zero memory 退化，说明 memory 信息不是单纯参数量收益。
- 2026-05-10 多 seed 稳定性验证完成：复用 `libero_mvp` feature cache，新增 `configs/libero_mvp_seed8.yaml` / `configs/libero_mvp_seed9.yaml` 和 `scripts/run_training_from_cache.sh`，完成 seed 8/9 训练与评估。汇总文件：`/mnt/llada-vla-go2/memory_world_model/outputs/multiseed_results.md`。3 seeds H-step MSE 均值±std：WM-only 0.183859±0.000289；WM+history 0.181071±0.000622（+1.52%）；WM+Memory 0.178225±0.000370（+3.06%）；Residual Memory WM 0.158314±0.000148（+13.89%）；Residual shuffled M 0.193050±0.000200（-5.00%）；Residual zero M 0.197002±0.001059（-7.15%）。结论：memory/residual 提升跨 seed 稳定，shuffled/zero ablation 稳定退化。
- 2026-05-10 LIBERO-90 long-horizon follow-up 完成：使用 `configs/libero90_stride10.yaml`，suite=`libero_90`、stride=10、H=5、K=4、DINOv2-base frozen、pseudo-memory=`z_history`。episode-level feature cache 位于 `/mnt/llada-vla-go2/memory_world_model/cache/libero90_stride10`，窗口数 train/val/test=51,779/6,577/6,509，总 64,865。输出位于 `/mnt/llada-vla-go2/memory_world_model/outputs/libero90_stride10`，对比汇总文件为 `libero90_vs_mvp_summary.md`。结果：WM-only H-step MSE 0.187646；WM+history 0.185914（+0.92%）；WM+Memory 0.182005（+3.01%）；Residual Memory WM 0.160035（+14.71%）；Residual shuffled M 0.195689（-4.29%）；Residual zero M 0.198879（-5.99%）。结论：在 LIBERO-90 长任务上，pseudo-memory/residual 的收益仍成立，Residual Memory WM 比 MVP multi-seed 均值略强；shuffled/zero memory 退化，支持继续实现 true MemoryVLA memory token extractor，但当前结论仍限于 pseudo-memory。

## 2026-05-14 Isaac Go2 专家控制视频录制

- 目标：录制两个专家控制 Go2 在 Isaac 仿真中完成目标导航任务的视频。
- 入口：`collectors/sim_go2/scripts/collect_raw_trajectories.py`，配置 `collectors/sim_go2/configs/collection_dual_target_contrast_room_local_smoke.yaml`。
- 运行环境要点：需要用 `CONDA_PREFIX= UNITREE_ROS_DIR=/home/zxq/zxq/unitree_ros /home/zxq/zxq/IsaacLab/isaaclab.sh -p ...`；如果不清空 `CONDA_PREFIX`，launcher 会误用 base conda python 并缺少 `isaacsim`；如果不设 `UNITREE_ROS_DIR`，会找不到 Go2 URDF。
- 输出 raw session：`/home/wxh/llada-vla-go2/outputs/expert_video_raw_20260514`。
- 导出视频：
  - `/home/wxh/llada-vla-go2/outputs/expert_video_raw_20260514/videos/expert_go2_isaac_ep_000001_door.mp4`：`approach the door`，78 帧，20 fps，3.90s，`success/goal_reached`。
  - `/home/wxh/llada-vla-go2/outputs/expert_video_raw_20260514/videos/expert_go2_isaac_ep_000002_suitcase.mp4`：`approach the dark gray suitcase`，81 帧，20 fps，4.05s，`success/goal_reached`。
- 验证：`ffprobe` 确认两个 MP4 均为 1280x960、20 fps、H.264/yuv420p；`index.json` 记录两个 episode 均成功。
- 补充摘要：`/home/wxh/llada-vla-go2/outputs/expert_video_raw_20260514/recording_summary.md`。

## 2026-05-14 第三人称 rollout 对比视频

- 用户目标：需要一个真实闭环推理 rollout 的第三人称视频，以及一个预期成功的第三人称参考视频。
- 直接 Isaac 第三人称 RGB 尝试：尝试添加/使用第三人称相机和 replicator capture，但本机 headless Isaac 渲染在场景/ground-plane/viewport 初始化阶段反复卡住或失败；日志出现大量 Carb `errno=28/No space left on device` watcher 错误，并在一次 smoke 中触发 `ChangePropertyCommand` ground-plane setup 失败。未把这些失败产物当作成功视频交付。
- 交付的可用版本是基于真实 telemetry 的第三人称轨迹可视化，不是原始 Isaac RGB 相机画面：
  - 真实模型闭环推理：`/home/wxh/llada-vla-go2/outputs/third_person_trace_videos_20260514/real_model_rollout_third_person.mp4`，来源 `/home/wxh/llada-vla-go2/outputs/closed_loop_reacquire_plus050_local_turnpilot_trace_20260506_0001/episodes/episode_000.json`；结果为真实模型 timeout，`success=false`，180 command steps，最终 `goal_distance≈2.886`。
  - 预期成功参考：`/home/wxh/llada-vla-go2/outputs/third_person_trace_videos_20260514/expected_success_expert_third_person.mp4`，来源 `/home/wxh/llada-vla-go2/outputs/expert_video_raw_20260514/episodes/ep_000001.json`；expert 成功，`termination_reason=goal_reached`，78 frames。
- 生成脚本：`/home/wxh/llada-vla-go2/outputs/render_trace_third_person_video.py`；输出目录 README：`/home/wxh/llada-vla-go2/outputs/third_person_trace_videos_20260514/README.md`。
