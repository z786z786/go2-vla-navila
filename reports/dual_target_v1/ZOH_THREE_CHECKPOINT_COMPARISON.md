# 三个 5k checkpoint 自动对照

## 2026-09-07 恢复记录

B与C均已完成5000步。原 `evaluation/` 在0/72时暴露fresh-process config注册顺序错误，模型尚未输出动作，故不构成任何checkpoint的任务结果。失败证据保留。修复后A/C真实CPU加载和GPU worker启动通过，恢复到新目录 `evaluation_v2/`：父PID2181558、队列PID2181599；第一条A/train/slot0已进入真实闭环并开始写模型chunk和50Hz轨迹。本地同步改由PID19513继续。最终报告仍须等待72/72，当前不可据训练loss宣布优胜者。

2026-09-07：用户明确授权当前 scheduler 对照结束后自动启动全量微调，并在两组完成后自动比较三个 5000-step checkpoint。该流水线不再逐阶段等待确认；不授权补采数据混入本次对照、使用独立 test 集或继续加训。

## 固定条件

|组|训练根目录尾名|LR|微调范围|
|---|---|---|---|
|A|zoh_train_5000_0906_v2|常数 1e-4|原 expert＋投影层|
|B|zoh_scheduler_5000_0907_v1|warmup＋cosine|同 A|
|C|zoh_full_5000_0907_v1|同 B|全部 450,046,176 参数启用梯度，含视觉/VLM|

共同使用原 base、批准的 16 条 train 轨迹（770 个 5 Hz 帧）、同一 80000 项采样索引、micro4/accum4、有效 batch16、seed20260906、5Hz/chunk5/execute1、同损失/归一化/命令限幅及变化率限制。B/C 都从 base 开始，不由 A/B 继续训练。B/C 实际 warmup166、衰减跨度5000、peak1e-4、末2.5e-6。C 全部参数使用 FP32 master，BF16 autocast；视觉非重入激活重计算仅作显存优化。这一精度实现差异需在解释 C−B 时明确记录，不宣称所有浮点计算严格相同。

架构未被实际使用的终端 head/norm 允许 grad=None，会记录完整名单；必须检测 vision/text/action expert 的非零有限梯度，以及视觉、文本 q_proj、动作输出投影的实际权重更新。首步需真实 checkpoint 重载预测一致，再继续训练。不可因显存不足悄悄减 batch 或冻结模块。

## 自动执行与交付

远端源 `/home/wxh/go2_short_vln`，数据根 `/mnt/wxh/go2_short_vln`。

- 流水线 PID **2121054**，入口 `scripts.zoh_three_pipeline`。
- 输出 `outputs/dual_target_v2/three_ckpt_compare_0907_v1`。
- 日志 `/tmp/go2_three_ckpt_compare_0907_v1_pipeline.log`。
- 已确认 `WAITING_SCHEDULER_TRAINING`，等待现有 scheduler 父 PID2113786 完整退出后启动 C。
- C 独立 supervisor：14GiB 共享准入、2GiB 运行保护、8小时单次训练限制；当前 B 代码及资源策略不改。
- 测试队列统一9GiB共享准入、2GiB运行保护、每条1800秒墙钟限制。项目锁保证串行，不终止其他用户进程。
- 每个 checkpoint 重跑 train slots0–15、validation slots16–23，共72条；按 slot 配对交错 A/B/C。统一 policy seed 和现有成对 reset 规则。不要求物理初态逐比特相同，不使用 test slots24–31。
- 正常模型失败（超时、停错目标等）记为结果继续；基础设施/数据完整性/显存保护失败保留证据并标记 `FAILED_NEEDS_REVIEW`，不冒充模型失败，不无限自动重试。
- 每条审计原始输入、chunk执行、ZOH、物理时序、限幅/变化率；成功结果独立重算停车。自动轨迹复核不等于主代理人工逐条观看视频。
- 导出全部实际50Hz帧，包含起始帧及实际末段；不足0.2秒末段不补满、不截掉。保留目标在画面边缘的实际影像，不裁切。
- 最终 `delivery/README.md`、`summary.csv`、`results.json`、三组训练配置/5k hash/gate证据、72个视频及预览图、全交付 SHA256 清单。
- 比较 train/eval 各自成功、停错目标、超时、碰撞、进入正确/错误停车区、末停车距离，以及 B−A / C−B 的描述性差异。单训练seed/推理seed，不作统计显著或唯一因果结论。

本地同步 PID **69972**，入口 `scripts/sync_three_ckpt_results.py`，每60秒读取远端交付、断网重试、最长48小时；远端计算不依赖本地连接，但本地自动下载要求电脑在线。日志 `reports/dual_target_v1/three_ckpt_sync.log`。下载目录：

`reports/dual_target_v1/remote_runs/three_ckpt_compare_0907_v1/`

完成后逐文件校验哈希。当前只是队列就绪，不是三组实验已经完成。

## 已完成的启动前验证

- 远端19项测试通过：矩阵完整性、待完成不计失败、全量scope/梯度/更新检测、推理loader分派、scheduler、控制和policy契约。
- 实际 base CPU 加载：450,046,176 / 450,046,176 参数可训练，全部 FP32 master，视觉激活重计算开启，CUDA 未初始化。
- A 的完整5k文件hash核验通过：manifest SHA256 `183c3941b0260c0e2079f91793f0639d99351c390e58cc6008c611534bb15e88`。
- 视频实测：原1465低层步、末hold5步，输出1466帧、29.32秒，未丢末0.1秒。远端派生预检视频位于 `/tmp/zoh_three_video_preflight_u67fjpky/preflight_partial.mp4`，原轨迹未修改。
- 远端磁盘约2TB可用。全量训练GPU前向、梯度和首步重载gate尚待 B 完成后执行，不能提前宣称通过。

恢复时先查 `pipeline_status.json` 和已有父/子进程，勿重复启动、覆盖目录、改冻结代码。pipeline_plan/eval_plan 保存源码hash，实际状态以其证据为准。
