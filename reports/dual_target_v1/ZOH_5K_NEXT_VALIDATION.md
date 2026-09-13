# 5k 训练验收与四条配对闭环

2026-09-06 用户授权“训练完了，开始下一步”。范围：验收本轮训练，固定最终 checkpoint 5000 做验证集一组布局四条配对任务；完成后停下，不自动进入完整矩阵、测试集或再训练。

## 训练验收

`zoh_train_5000_0906_v2` 正常完成，supervisor 状态 ZOH_5000_COMPLETE_AWAITING_REVIEW，模型/训练进程已退出。共新增 4000 更新至累计 5000。1001/2000/3000/4000/5000 五份 checkpoint 的 manifest 与全部文件哈希通过独立复核；保存重载 gate 已通过。

401 个记录更新的 loss/gradient_norm 均有限，末次 loss 0.04303884，末 10 个记录点均值 0.07374439（非最后 10 个实际更新）。训练损失不构成导航成功验收。

## 固定输入对照

脚本 `scripts/audit_zoh_after_5k.py`，本地证据 `remote_runs/zoh_post5k_audit_0906_v1/report.json`，远端 `outputs/dual_target_v2/zoh_post5k_audit_0906_v1`。

重新验证数据绑定及 checkpoint 预处理与 train normalizer 构造处理完全一致；测试相同训练起点、seed 20260906 的固定噪声，交换 red/blue task tokens。

- 16 起点中预测首动作正 wz 为 10 个，方向与对应专家一致 8/16。
- 更换指令后首动作 wz 平均绝对变化 0.00307835 rad/s，原 checkpoint 1000 为 0.00288131。
- 这不是导航成功率，也不是跨 seed 的统计检验。不能依据微小变化声称语言响应改善；仍有明显语言利用不足的警讯。
- 不据此选择其它 checkpoint；先固定 5000 进行真实配对轨迹检查。

## 闭环执行

新队列模块 `src/dual_target/zoh_eval_5k_pair_queue.py`，沿用已修复的 `zoh_closed_loop_result`、`zoh_result_loop`、`zoh_eval_review_v2`；11 项推理合同/结果/评分回归通过。只新增队列，不改数据、训练或控制评分模块。

运行根 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_val_5k_pair_0906_v1`，父 PID 2089283，日志 `/tmp/go2_zoh_val_5k_pair_0906_v1_queue.log`。

固定 validation slots 16/17/18/19：布局 v2_val_000，A红B蓝去红/去蓝，A蓝B红去红/去蓝；policy seed 20260906，配对重置种子规则不变。每条最长 30 秒仿真时间（1800 秒墙钟保护）。5 Hz 仿真控制、chunk5/execute1、共享限幅/变化率、停车半径 0.3m、连续停稳 1s 均不变。

保留前轮闭环的 7168 MiB 准入和 1024 MiB 运行保护；连续检查、锁复查、只终止自有组。普通超时/碰撞/错目标等结果计入并继续下一配对；基础设施/资源/数据完整性失败停止、不自动重试。首任务失败不伪装为资源失败。

后续验收分别报告接近哪个目标、是否进入正确/另一停车区、自主停稳和完整任务成功，并查看图像与轨迹；本四条完成后停止，等待用户决定是否扩展。
