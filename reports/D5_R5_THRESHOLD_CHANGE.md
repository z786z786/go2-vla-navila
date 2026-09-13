# D5-R5 — Train 场景覆盖阈值调整（9 → 8）

## 授权与原因

依据 2026-09-02 的明确用户指令，D5 train accepted-scene 阈值由 9 调整为 8。D5-R4 已保留可行性证据：在 30 条成功、三个动作类别各 10 条、朝向与距离覆盖均满足的前提下，最大可达成功场景数为 8；唯一用于第九场景补充的 `short_vln_v1_1051` 已由不可变官方 PD rollout 正常拒绝（`official_large_orientation`）。

## 实施边界

- 仅修改 D5 train 的 `min_scenes: 9 → 8`；seen-val 覆盖规则、动作类别配额、上限和 PD 执行器均不变。
- 恢复时先保存新的不可变 manifest snapshot，再写入 `approved_train_scene_threshold_change` 迁移事件、旧值、新值和用户授权标识。
- manifest 持久化完整 ruleset；后续恢复若规则不一致将拒绝运行，不能静默改变 accepted cohort 的含义。
- 既有 attempt、accepted/rejected 证据、原始短数据集、M4 官方 planner 和其哈希均不覆盖、不重跑、不修改。

## 验收含义

这使现有 30 条、8 场景的 train cohort 满足 D5 覆盖约束，采集编排将继续进入独立的 seen-val 队列。报告必须表述为“8 场景覆盖标准”，不得继续声称满足原 9 场景标准；这仍是数据覆盖约束，而不是统计显著性或泛化结论。
