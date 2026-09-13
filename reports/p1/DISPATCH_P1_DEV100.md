# 派发单 P1-DEV100 — 冻结 benchmark dev100 的 episode 清单

授权：用户 2026-09-13。属 **P1′ 契约交付项**（契约 §4 与 P1 任务 5）。
波次：与 P4-T6v2 并发（写入路径不重叠、均不占 GPU、互不依赖）。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P1-DEV100** |
| 资源 | **CPU only。零 GPU，零网络。** |

## 1. 目标

产出**冻结的** dev100 episode 清单。它是日常迭代与全部 ablation 的评测子集，一经冻结长期使用，
改动会使已跑的评测作废。

## 2. 输入

| 路径 | SHA-256 |
|---|---|
| `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz` | `ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec` |

开工前 `sha256sum` 核对，不符即停。

## 3. 已确立的事实（主代理实测，作为实现依据与自检基准）

| 事实 | 值 |
|---|---|
| episode 总数 | **1077** |
| scene 数 | **11**（恰为 R2R `val_unseen` 的 scene 集合） |
| **不同 trajectory 数** | **359**，且**每条 trajectory 恰有 3 条 episode** |
| 同一 trajectory 的 3 条 episode | **起点、目标、`gt_locations` 完全相同，仅 `instruction_text` 不同** |
| 成功半径 | 全部 1077 条 `goals[0].radius == 3.0`（P0 已取证） |

每 scene 的 episode / trajectory 数（主代理实测，须复现）：

| scene | episodes | routes |
|---|---:|---:|
| `zsNo4HB9uLZ` | 270 | 90 |
| `QUCTc6BB5sX` | 240 | 80 |
| `2azQ1b91cZZ` | 204 | 68 |
| `TbHJrupSAjP` | 108 | 36 |
| `X7HyMhZNoso` | 69 | 23 |
| `Z6MFQCViBuw` | 66 | 22 |
| `EU6Fwq7SyZv` | 54 | 18 |
| `x8F5xyUWy9e` | 30 | 10 |
| `oLBMNvg9in8` | 24 | 8 |
| `8194nk5LbLH` | 9 | 3 |
| `pLe4wQe7qrG` | 3 | **1** |

## 4. 抽样规则（用户 2026-09-13 裁定，照此实现，不得自行改动）

**方案 A：每条 trajectory 最多取 1 条 episode，故 dev100 = 100 条互不相同的物理路线。**

裁定理由（记录在案）：dev100 的用途是**筛选信号**（契约 §4，SR 置信区间约 ±10 点，明确不作结论）。
每路线取 1 条使每 GPU 小时买到的路线多样性最大。「同一路线不同说法」的考察由已批准的
**乱指令对照**（v3.1 #14）以更严厉的方式承担（喂错误指令，强于三种正确改写），故不在 dev100 中重复。

实现细则：

1. **固定 seed**，写入输出。抽样必须完全可复现。
2. 按 scene 分层。**每个 scene 至少分配 1 条**（`pLe4wQe7qrG` 仅 1 条路线，只能且必须贡献 1 条）。
3. 剩余名额按各 scene 的**路线数**（非 episode 数）比例分配，用**最大余数法**配平到**恰好 100**。
4. 任一 scene 的分配数不得超过其可用路线数。
5. 选中 trajectory 后，从该 trajectory 的 3 条 episode 中**按固定规则选 1 条**
   （例如 `episode_id` 最小，或 seed 洗牌后取首条——**选定一种并写明**）。
6. 输出按 `(scene, episode_id)` 排序，保证清单稳定。

## 5. 输出

| 路径 | 内容 |
|---|---|
| `/home/wxh/go2_short_vln/configs/benchmark_dev100.json` | **冻结清单**：`frozen_at`、`seed`、`selection_rule`、`episode_ids`(100)、`trajectory_ids`(100)、每 scene 分配表、输入 sha256 |
| `/home/wxh/go2_short_vln/reports/p1/DEV100_SELECTION.md` | 人读摘要：分配表（分配数 / 可用路线数）、规则说明、裁定理由 |
| `/home/wxh/go2_short_vln/scripts/p1_build_dev100.py` | **无参数可重跑且确定性** |

**不要写 `configs/benchmark.yaml`**——那是 P1′ 完整契约冻结时的产物，本任务只产出 dev100 清单供其引用。
`configs/` 为独占写资源，本波次只有本任务写它。

## 6. 验收标准（主代理如何独立复算）

1. 主代理独立重算每 scene 的 episode 数与 trajectory 数，须与 §3 表格逐行一致。
2. 主代理断言 `len(episode_ids) == 100` 且 **`len(set(trajectory_ids)) == 100`**（即 100 条互不相同的路线）。
3. 主代理断言 100 个 `episode_id` **全部存在于 1077 条评测集**中，且每个 episode 的 `trajectory_id` 与清单对应。
4. 主代理独立按 §4 规则重算分配表（含最大余数法配平），须与输出一致；并断言每 scene 分配数 ≤ 可用路线数、每 scene ≥1。
5. 主代理无参数重跑脚本，产物关键字段一致（确定性）。
6. 主代理断言所选 100 条的 scene 集合 ⊆ 那 11 个评测 scene，且**与 61 个训练 scene 交集为空**。
7. 输入 sha256 与 §2 逐字一致。

**子代理的 `passed` 与自我结论不构成 PASS 依据。**

## 7. 禁止事项

1. 不得改动 §5 三个输出以外的任何文件；**特别不得写 `configs/benchmark.yaml`**、不得动
   `reports/p3/`、`reports/p4/`、`actions/`、`datasets/`、`src/`、`.gitignore`、里程碑文档。
2. 不得改动 §4 的抽样规则，包括「每 trajectory 取 1 条」这一条。
3. 不得从训练集（NaVILA / R2R train）取任何数据——dev100 只能来自那 1077 条。
4. 不得 `git add` / `git commit` / `git tag`；不得联网；不得占 GPU。
5. 证据缺失标记 `MISSING`，不得自行追认 PASS。

## 8. 验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| — | — | 待派发 | — |
