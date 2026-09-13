# D5-R4 — 事务化编排与第九场景补充

## 原因与边界

D5-R3 的 `1074` 和 `0104` 均由不可变官方 PD 成功采集，使 train 达到 30 条和三类动作各 10 条，但成功场景仍为 8。缺失场景 `pLe4wQe7qrG` 的 train 路线 `1050/1052` 已分别作为初始化姿态拒绝和正常 planner 拒绝保留；没有其他未尝试 train 路线。旧实现随后本应写入 `blocked_selection_infeasible`，却把内部 coverage set 写入 JSON 而退出。

## D5-R4 修复

- manifest 在内存完成确定性 JSON 序列化后才原子替换；集合会稳定地写为列表，未知类型在替换旧文件前失败。
- blocked 决策保存理论最大可达类别、场景、朝向和距离覆盖，不再以异常掩盖真正的可行性结论。
- 原始 `short_vln_v1.json` 和其 SHA-256 保持不变。唯一未被 D5 seen-val 预选的同场景路线 `short_vln_v1_1051` 由独立 assignment policy 作为 D5 train 覆盖补充。
- M4 summary 继续保留 `source_split=seen-val`。成功轨迹旁写入 `d5_assignment.json`，声明 `collection_split=train`、策略哈希、数据集哈希和“禁止进入 seen-val evaluation”；转换清单保留这两种 split。
- 现有 41 次 attempt、accepted/rejected 原始证据不被覆盖或重跑。首次迁移先保存不可变 resume snapshot 和 assignment-policy 迁移事件。

## 恢复 gate

1. 本地与远端单元测试、编译和 shell 检查通过。
2. 只读 dry-run 必须把 `1051` 选作 train coverage supplement，覆盖增益为 1。
3. 空闲显存不少于 12,288 MiB 时恢复 D5；PD runner 与官方 planner 均不修改。
4. `1051` 若成功，train 为 31 条、9 场景；若正常失败，干净地记录 `blocked_selection_infeasible`，不降低阈值或重跑历史路线。
5. D5-R4 后连续三条新路线须无基础设施或 manifest 错误，之后才改为每 30 分钟监控。
