# ZOH 评分兼容修复与验证重启

## 根因与修复

原 `ScoreFrame.validate` 假设 applied=逐轴clip(raw)，但 ZOH 部署已是 clip→slew→hold。合法变化率限制生效时，旧校验返回 INVALID_SAMPLE 并重置停车计时。这是部署/评分集成遗漏，不能归因于模型未学会。

新 `zoh_scoring.py` 提供独立 `ZohActionAudit`：从零初值按0.2秒更新一次，复算限幅、变化率限制，确认十个低层步内raw/applied不变，拒绝步号丢失、重复、非有限值和错误施加动作。`ZohScoreFrame` 保留原字段/有限值/flags/时钟校验，仅将无状态映射断言替换为独立ZOH期望；真正送入停车判定的原始raw不替换、不归零。停车仍须raw vx/wz、applied vx/wz、实测速度、正确区域连续1秒满足阈值。碰撞等失败优先级不变。

独立 `zoh_scored_loop.py` 复用原循环，只注入独立动作审计和新frame类型；若正确/错误目标评分器再报告INVALID_SAMPLE立即报错停止，不能继续运行到timeout后误标模型失败。旧 `scoring.py`、`zoh_loop.py`、数据集/物理gate源码和证据不修改。

## 测试与真实回放

- 本地与远端8项评分/协议测试全部通过：slew合法、段中更新拒绝、错误slew/重复步拒绝、原始非零不得被clip掩盖、NaN拒绝、真实运动不准停车、碰撞优先级、连续1秒停车门槛、新chunk协议。
- 原受影响闭环已记录593步，旧INVALID_SAMPLE60次，新评分回放0次，593步仍IN_PROGRESS，不伪造任务成功。
- 对32条已批准专家轨迹做固定代码完整性回归，全部仍成功，新INVALID_SAMPLE均0。因此不需要重采或重训。包括test_audit的读取仅核验已有专家数据评分兼容性，不运行模型test任务，不按该数据选择checkpoint或调阈值。
- 证据：`outputs/dual_target_v2/zoh_scoring_fix_replay_0906_v1.json`，本地副本在本报告目录的 `remote_runs/`。

## 新运行

原PID2045780已定向停止，Isaac2045934、模型2046571确认退出；`zoh_val_probe7_floor1_0906_v1`完整保留，受影响结果不算有效模型评估。

新队列 `zoh_eval_scored_queue.py` / `zoh_closed_loop_scored.py`，PID2048028，输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_val_scored_0906_v1`，父日志 `/tmp/go2_zoh_val_scored_0906_v1_queue.log`。保留原32任务、checkpoint顺序、种子、7GiB准入/1GiB保护和1800秒上限；首条成功并独立审计/资源合格才继续矩阵，否则停止。无追加训练，无模型test评估。队列冻结新源码，不能热改。

回放通过不等于新实时闭环已经通过；以该目录queue_status、逐帧评分和实际任务结果为准。
