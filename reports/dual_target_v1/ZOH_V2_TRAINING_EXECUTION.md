# 新版 5 Hz / chunk 5 训练执行

## TensorBoard

用户已授权并配置旁路可视化。浏览器入口 `http://localhost:16007`，本地 SSH 隧道转发至 neu-3070 `127.0.0.1:6007`；旧6006服务未改。专用脚本 `scripts/zoh_training_to_tensorboard.py` 仅解析本次 `events.jsonl` 的 update 事件，横轴 optimizer_updates，使用原始 wall_time；历史回填+2秒轮询，写入 `tensorboard/zoh_train_1000`，不使用 GPU。桥接 PID2041992、TensorBoard PID2041993；日志分别在训练输出根的 `tensorboard_bridge.log`、`tensorboard_server.log`。已通过本地 HTTP scalars 接口验证，至少回填至660步。

显示 Loss/train、裁剪前梯度范数、恒定学习率、单步时间、16/单步时间的更新段吞吐、PyTorch历史峰值 allocated MiB（不是整卡空闲显存）。训练只每10更新记录一次 loss，首步另记；界面不伪造每步或 validation loss。曲线平滑只影响显示，不代表模型导航已成功。

本地隧道断开可执行：`ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:16007:127.0.0.1:6007 neu-3070`。服务只监听回环，不开放公网。恢复先查已有端口/进程，不重复启动桥接或服务。

用户在数据集主代理批准后要求准备好即直接启动训练。训练优先，采集提速候选未接入、未影响本批数据。

启动验收：真实训练已进行到日志390/1000，micro4×accum4；首步梯度与投影更新有效，保存/重载固定输入噪声预测最大差0，single_update_gate通过。checkpoint1/250已落盘。当前自有显存3182 MiB、actual free4190 MiB，高于2048保护线。该进度为检查时快照，后续查实际事件日志。

## 固定范围

- 独立 `zoh_train.py` / `zoh_train_core.py` / `zoh_train_launch.py`；旧训练源码和 250 步旧模型不变。读取批准的 `zoh_dataset_0906_v2/dataset/train`，16 条/770 帧。验证、测试不参与更新；train-only normalizer，episode 等权抽样。
- 从已核验 `smolvla_base_c83c316` 重新初始化，非旧 250 步续训；RGB 512、实测 3D 速度状态、高层 3D 速度动作。
- 5 Hz、chunk 5、n_action_steps 1；仅真实 episode 内组成 action chunk，尾部 padding mask 与 vy 通道不计损失。
- VLM/vision 冻结，动作专家与 state/action/time 投影可训练；可训练主权重和 Adam 状态 FP32，BF16 autocast。逐参数验证可训练范围。
- 保留既有 LR=1e-4、AdamW betas(.9,.95)、eps1e-8、weight_decay1e-10、clip10、恒定 LR。有效 batch16，micro1/2/4 实测选择，accum=16/micro。没有直接采用未锁定的 LR5e-5/6000 步提案。
- 首阶段 1000 次更新；checkpoint1/250/500/750/1000。仍处于既有累计 5000 上限内（旧250+本次1000，后续剩3750）；不得自动扩展。离线 loss 不作为导航成功或 checkpoint 主排序依据；后续按 validation 闭环比较，test 暂不评估。
- 单次更新后强制真实 checkpoint/processor 重载，固定输入与噪声预测一致；门槛失败则停止。完成1000后状态 `ZOH_1000_COMPLETE_AWAITING_REVIEW`，不自动启动模型导航/下一阶段。

## 资源与记录

- 独立训练准入 `zoh_train_shared_profile_7g_v1`：空闲7168 MiB、连续3次/30秒、持锁复查；运行实际空闲低于2048 MiB停止自有 PGID。allocator 上限准入空闲减2304 MiB；micro profile 预留 Adam moment，失败不计作训练更新。4小时墙钟上限。
- 输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_train_1000_0906_v1`。
- 监督父 PID 2039433；父日志 `/tmp/go2_zoh_train_1000_0906_v1_supervisor.log`。恢复先读 `supervisor_status.json`、`events.jsonl`、`training_config.json`，勿重复启动。
- 远端实际训练环境9项测试全部通过：旧批准拒绝、heldout路径拒绝、chunk5跨episode/padding、vy/padding零梯度、累积与整batch一致。启动前新版批准与全部导出文件哈希核验、770帧/fps5/chunk5元数据检查和训练依赖导入通过。
- 启动监督队列不等于已经发生优化器更新，以 `events.jsonl` 的 `update` 与 checkpoint 为准。
