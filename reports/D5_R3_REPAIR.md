# D5-R3 — 约束感知的路线重选与补充

## 触发条件

D5-R2 对 `short_vln_v1_1052` 的分类是正常的 `planner_reject`：PD rollout 完整运行但未进入成功半径，并非基础设施错误。旧的重选器为每个缺口枚举整组组合，达到 50,000 次内部搜索上限后错误地报告“没有满足覆盖约束的替代路线”，并把 manifest 留在 `collecting`。

## 修复范围

- 保留所有 28 条已接纳轨迹、39 次已发生 attempt，以及所有被拒绝路线的原始证据；不重跑 `1052`。
- 删除组合枚举路径，改为每轮只挑选一条候选路线。分类最低配额优先；配额满足后，仅挑选能改进 scene、heading-bin 或 distance-bin 覆盖的补充路线。
- 接纳最小值保持 train `30`、seen-val `12`；为覆盖不足允许补充到 train `34`、seen-val `16`，总计最多 `50`。超过预算、无可行候选和分类 attempt 预算分别形成明确、可恢复的 blocked 状态。
- 每次选择在运行前写入 `selection_events`，包括阶段、候选属性、覆盖增益、稳定随机排序分数和选择前的接纳覆盖。manifest 同时写入只基于 accepted 证据计算的 `accepted_summary`。
- 增加同 manifest 的互斥锁，以及 `--selection-dry-run`；干跑不写入、也不启动 Isaac。
- D5 总报告不再强制恰好 `30/12`，而是验证最小/最大/总数上限及每个 split 的分类和覆盖验收状态。

## 执行顺序

1. 本地单元测试和语法检查。
2. 上传源码、测试和本报告；在远端针对 D5-R3 测试执行。
3. 对现有 D5 root 运行 `--resume --selection-dry-run`，确认候选、覆盖增益和上限；该步骤不得写 manifest 或启动 Isaac。
4. 在空闲显存至少 12,288 MiB 时，以原 run root 执行 `bash scripts/m62_run_d5.sh --resume ...`。不终止或干预其他 GPU 任务。
5. 收集完成后，检查 manifest 的接纳摘要、视频/日志和 D5 后续数据转换、训练、闭环、报告 gate。

## 验收

- 不再存在组合搜索上限；无候选时给出 `blocked_selection_infeasible`，不伪报基础设施故障。
- 仅使用 accepted 路线评估覆盖；保留 planner/recovery reject 的审计链。
- 全部 accepted 数量在 train `30..34`、seen-val `12..16`、总数 `<=50`，并满足各 split 分类和覆盖阈值。
