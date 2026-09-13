# Go2 + Isaac VLN 实验平台里程碑计划（合并版 v1）

本文档由 `new_milestone.md`（平台化重构设想）与仓库实际已完成工作（`CODEX_GO2_SMOLVLA_SHORT_VLN_RUNBOOK.md`、`CODEX_SMOLVLA_GO2_DUAL_TARGET_MILESTONES.md`、`reports/MILESTONES.md`、`reports/r2r_ablation/`、`reports/OFFICIAL_NAVILA_VELOCITY_0911.md`）合并而成，取代前述三份文档作为**唯一执行入口**。旧文档降级为审计参考。

生成日期：2026-09-12。

---

## 0. 本文档相对 `new_milestone.md` 的九处修订

逐条列出，便于核对为什么和你的草稿不同。

| # | `new_milestone.md` 原案 | 本文档 | 理由 |
|---|---|---|---|
| 1 | `NavCommand = [vx,vy,wz,duration,stop]` | `[vx, vy, wz, stop]`，**无 duration** | 保住全部已采数据与 M6.1 bounded codec；stop 改为显式头，`ablation_arrays.npz` 已有四维 `action_command_stop` 可直接用 |
| 2 | `wz ∈ ±1.0 rad/s` | `wz ∈ ±0.5 rad/s` | 现有 expert 全部在 ±0.5 内采集，扩范围等于让模型在无监督区域自由输出 |
| 3 | 决策频率 M0 冻结 2 Hz，§14 又列 1/2/4 Hz ablation | **固定 5 Hz，不做频率 ablation** | 全部已采数据、history 关键帧规则、M6.1 codec 都绑在 5 Hz；频率 ablation 会稀释主研究问题（action representation）的实验预算 |
| 4 | STOP 与成功判据混写为 "3m" | **两层分离**：expert 停止意图 0.30m（监督），主表 SR 用官方 3m（评分） | 草稿把行为目标和评分规则写在同一节，易实现成 expert 在 3m 外就停 |
| 5 | M1–M4 从零执行（约 7 周） | 折叠为 **P0 存量追认 + P2 重构后回归** | 四者在仓库已有实测证据（见 §2），重做是纯浪费 |
| 6 | 无接触力议题 | 单列 **X1 可选修复里程碑**，主线用代理指标 | 19 个 body 接触力恒零、四种排查全失败，放进主线等于埋雷 |
| 7 | M4 目标 500–1000 条 Go2 rollout | **P3 数据可用性探针先行**，规模在探针后定 | 实测实时率 0.203，1000 条 ≈ 75 小时墙钟 + 22 GB，共享 GPU 下不可行 |
| 8 | LLaDA-V 两个 head 并行两周 | **language head 先行**，discrete head 排到 X2 | 仓库零前期工作；两个 head 工程量完全不同 |
| 9 | 16 周甘特图 | **去掉周数，保留 CONTINUE 门禁** | 共享 GPU + 0.2 实时率下周数只制造赶工压力，不构成调度依据 |

草稿中被完整采纳的三点：冻结统一 Observation / Policy API（当前三条线三套协议，是最真实的技术债）；把 **action representation 对比**立为主研究问题；`§21` 五大失败点作为全程风险清单。

---

## 1. 项目目标

不追求 R2R / VLN-CE 的 SOTA，而是建成一个可复用、可扩展、能**公平比较不同 VLM/VLA 导航策略**的 Isaac-Go2 实验平台。

主研究问题（按优先级）：

1. Continuous action 与 language action，哪种更适合 embodied VLN？
2. History 是否显著改善 navigation？
3. Robot state 是否有帮助？
4. 三种表示在 SR / 路径效率 / collision / latency 上的 trade-off？

工程铁律：**所有模型必须通过同一个 `NavCommand` API 控制 Go2，不得有任一模型获得其他模型没有的信息。**

---

## 2. 已完成的基础能力（validated foundation）

以下能力**已有实测证据，不重做**。P0 负责归档，P2 负责重构后回归。

| 能力 | 证据 | 位置 |
|---|---|---|
| Go2 低层 locomotion + 速度跟踪 | 50 Hz `low_level.jsonl`：动作前/后位姿、机体/世界速度、关节位置/速度/目标、实际力矩、低层策略输出 | `reports/r2r_ablation/20260910_ep1_v3/` |
| R2R-VLNCE 全量部署 | 10819 episodes / 61 scenes；场景文件 61/61、6744 文件校验；两版本 SHA-256 齐全 | `reports/r2r_vlnce/audit_report.md` |
| Habitat ↔ Isaac 坐标转换 | `convert` / `rotate`，实测走完 7.09 m、末端距目标 0.216 m、零意外 reset | `scripts/validate_r2r_episode_load.py` |
| 连续路径 Expert | `ContinuousRouteExpert`：look-ahead waypoint + heading controller（0.25 m 切换半径、1.5 增益、0.75 rad 转向阈值） | `src/dual_target/r2r_path_expert.py` |
| 采集链路 + 独立审计 | 1956 个 50 Hz 转移、196 个高层区间、236 张有效 RGB、13 个停止意图区间；时间连续性/图像 freshness/ZOH 保持/history 无未来泄漏/停止事件全部通过 | `scripts/collect_r2r_ablation.py`、`scripts/audit_r2r_ablation.py` |
| 消融数组接口 | `ablation_arrays.npz`：`command_raw`、`command_applied`、`velocity_actual_post`、`velocity_actual_interval_mean`、`state_velocity_pre`、`stop_intent`、`action_command_stop`、`rgb_frame_id`、`history_frame_ids`、`history_padding_mask` | 同上 |
| GPU 共享治理 | `gpu_wait.py`：共享锁、显存准入门槛、运行余量、只清理自有进程组 | `src/dual_target/gpu_wait.py` |
| 极小可控场景 | dual_target 红蓝箱：冻结任务契约、布局策略、评分 | `config/dual_target_v1/task_contract.lock.json` |

**实测成本基线**（用于一切规模估算）：实时率 0.203；ep1 走 7.09 m 用 39.12 s 仿真 / 192.46 s 墙钟；原始记录 16.15 MB；每仿真分钟约 24.76 MB。

---

## 3. 冻结契约

P1 通过后，未经显式授权不得修改本节任何数值。

### 3.1 Observation API

```python
Observation:
    rgb:          uint8 [512, 512, 3]      # 不旋转
    instruction:  str                       # R2R 原文，逐字，不改写不截断
    history_rgb:  Optional[List[rgb]]       # NaVILA 8 帧 prefix-uniform，0.5 s 关键帧
    state:        Optional[[float, float, float]]   # [body_vx, body_vy, body_yaw_rate]
    previous_action: Optional[NavCommand]
```

`history_rgb` 不足 8 帧时以黑帧 padding，并提供 `padding_mask`；**只取当前及过去帧**。

**模型输入绝对禁止**（仅 evaluator / expert 可见）：

```
GT global position · GT goal direction · GT distance-to-goal
shortest path · reference path · oracle heading · PD planner action
```

每条 rollout 必须留存 request 字段白名单校验记录，能独立复算"无泄漏"。

### 3.2 NavCommand

```python
NavCommand:
    vx:   float   # [0.0, 0.5]  m/s
    vy:   float   # 固定 0.0，本阶段不作为自由维度
    wz:   float   # [-0.5, 0.5] rad/s
    stop: bool    # 显式 stop 头输出
```

- 模型输出 **raw** 值先记录，再做安全 clipping 与变化率限制（每次决策 `Δvx ≤ 0.1 m/s`、`Δwz ≤ 0.2 rad/s`），两者都存。
- `stop` 由**显式分类头**给出，不再从速度阈值反推。旧的 `|vx|<0.03 / |vy|<0.03 / |wz|<0.05` 阈值法作为历史兼容路径保留在归档里，不进新主线。

### 3.3 时序

```
VLA 决策频率:      5 Hz  (0.2 s)     ← 固定，所有模型一致，不作为实验变量
命令保持:          ZOH，10 个低层控制步
低层 locomotion:   50 Hz (0.02 s)
物理步长:          0.005 s
```

**时间语义按仿真时间对齐，允许墙钟慢于实时。** 延迟单独作为指标报告，不作为约束——否则 LLaDA-V 的生成延迟会直接摧毁公平对比。

决策频率是**冻结常量，不是 ablation 轴**。50 Hz 原始记录仍然全量保留，日后若有人要做频率研究可以重采样，但不在本计划范围内。

### 3.4 成功与停止（两层，不得混用）

| 层 | 阈值 | 用途 |
|---|---|---|
| Expert 停止意图 | 距原始目标 **0.30 m** 且连续 2 s 实际静止（平面速度 `<0.05 m/s`、yaw rate `<0.10 rad/s`） | 行为目标 / 训练监督 |
| Benchmark 主表 SR | 官方 **3.0 m**（已核对，见下） | 评分 |

副表并行记录 1 m / 2 m 半径 SR，防止 3 m 判据掩盖真实差异。

**官方半径核对结论（2026-09-12 完成）**：官方 `Success` 与 `OracleSuccess` 均以
`self._success_distance = episode["goals"][0]["radius"]` 逐 episode 从数据读取，不是硬编码常量
（源：`isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/measures.py`）。
本项目已抽取的 12 个 episode goal 中 `success_radius_m` 全部为 `3.0`，无例外
（`artifacts/navila_canary_r5_review/extracted/**/collector_config.json`）。

因此 P1 的核对项不是"去查官方阈值"，而是一行断言：**全部 1077 条 episode 的
`goals[0].radius == 3.0`**。若出现非 3.0 的 episode，必须单独列出并决定是排除还是分组报告——
因为半径是逐条读的，混入不同半径会让 SR 失去统一含义。

### 3.5 配置文件

```
configs/
    observation.yaml    # §3.1
    action.yaml         # §3.2 + §3.3
    robot.yaml          # Go2 相机参数、关节/接触体名称、低层 checkpoint 哈希
    benchmark.yaml      # §3.4 + 评测集 ID 列表 + 指标定义
```

每份配置带 `frozen_at` 与 SHA-256，evaluator 启动时校验；模型代码不得覆盖。

---

## 4. 数据与 split

| 用途 | 来源 | 规模 |
|---|---|---|
| 训练 | R2R-VLNCE TRAIN | 10819 ep / 61 scenes |
| 评测 | VLN-CE-Isaac | 1077 ep / 11 scenes |

两者 scene 集合不相交，评测天然为 unseen。**P0 必须用一条机械核对把这个结论钉成证据**（打印两个 scene 集合的交集，断言为空并存档），而不是停留在口头结论。

评测集切分：

- **dev100**：从 1077 条按 scene 分层抽样 100 条，固定 seed 与 ID 列表，写入 `configs/benchmark.yaml`。日常迭代与全部 ablation 跑 dev100。
- **full1077**：只有进入主对比表的最终模型才跑全量。

注意 dev100 的 SR 置信区间约 ±10 个点，dev100 上的差异只作为"是否值得上全量"的筛选信号，不作为结论。

---

## 5. 里程碑

**执行协议**：不挂日期，只挂 Go/No-Go 条件。每个里程碑完成并自检后**停下**，等待显式 `CONTINUE Pn`，不得自动进入下一阶段。同一失败原因在两次有证据的修复尝试后仍存在，先交诊断再动方案。

### 角色分工

| 角色 | 职责 |
|---|---|
| **主代理** | 只负责**写计划与验收**：拆解里程碑、定义验收标准、审查 codex 子代理交回的证据、判定 PASS/FAIL、维护本文档与 `reports/` |
| **codex 子代理**（可并发多个） | 负责**实现**：写代码、跑实验、采数据、出报告 |

派发规则：

1. 主代理把一个里程碑拆成**互不冲突的子任务**再并发派发。判定"互不冲突"看三件事：是否写同一批文件、是否争抢 GPU、是否依赖彼此的产出。三者任一存在就串行。
2. 每个子任务的派发单必须自带：目标、**输入文件的确切路径与 SHA-256**、输出路径、验收标准、禁止事项。子代理不共享上下文，含糊的派发单必然返工。
3. 涉及 GPU 的子任务**最多一个在跑**（3090 单卡且共享），必须走 `gpu_wait` 的准入与锁。CPU/网络类子任务（如 P3 数据探针、parser 单元测试、配置生成）可自由并发。
4. 主代理**不接受子代理的自我验收**。子代理交回的 `passed` 字段、测试通过数、loss 下降曲线都不构成 PASS 依据；主代理独立复算关键数字、核对哈希、必要时重跑最小验证。
5. 子代理不得跨里程碑推进。做完自己那一格就交回，由主代理决定下一步。

适合并发派发的例子：P1 的四份配置各一个子代理；P2 重构按目标树的 `envs/ robots/ policies/ actions/ experts/ datasets/ evaluation/` 分派（但回归验证由主代理统一跑）；P3 探针的五项调查各一个；P8 的各条 ablation 在 GPU 排队下串行、但数据准备与结果分析可并发。

必须串行的例子：P0 → P1 → P2（证据冻结、契约、重构有严格先后）；P5 的 overfit 阶梯；任何两个都要写 `configs/` 的任务。

```
P0 存量冻结  →  P1 契约冻结  →  P2 重构+回归  →  P3 数据探针
     →  P4 数据生产  →  P5 SmolVLA 基线  →  P6 LLaDA-V language
     →  P7 评测框架  →  P8 主对比+ablation  →  P9 Robustness  →  P10 交付

旁支（不阻塞主线）：X1 接触力修复   X2 LLaDA-V discrete head
```

---

### P0 — 存量冻结与证据归档

**目标**：在任何重构动作之前，把现有实验证据变成不可变基线。

任务：

1. **停止正在运行的官方 1077-episode 队列**（PID 779548，日志 `/tmp/official_navila_velocity_0911_full_v1.log`）。按 `gpu_wait` 约定只停自有进程组，保留 `progress.json` / `plan.json` / `results.json` 与所有已完成 episode 目录。
2. 对现有成果打 `git tag foundation-v0`，并生成 `reports/FOUNDATION_ARCHIVE.md`：逐项记录 §2 表格中每个能力的证据路径、SHA-256、产生它的命令、以及远端 `/home/wxh/go2_short_vln` 与 `/mnt/wxh/go2_short_vln` 下对应产物。
3. **scene 交集核对**：打印 R2R TRAIN 61 scenes 与 VLN-CE-Isaac 11 scenes 的交集，断言为空，存档为 `reports/split_disjoint_check.json`。
4. 归档三块不进新主线的历史资产（详见附录 A）。

验收：

- [ ] 官方队列已停止，GPU 已释放，已完成 episode 与 progress 完整保留
- [ ] `foundation-v0` tag 存在
- [ ] `FOUNDATION_ARCHIVE.md` 覆盖 §2 全部七项，每项都有可复算的哈希
- [ ] scene 交集为空，有存档证据
- [ ] 附录 A 三块历史资产各有明确去向说明

停下，等 `CONTINUE P1`。

---

### P1 — 接口契约冻结

**目标**：把 §3 落成可校验的配置，并消除当前三套并存的协议（M7 的 30-D state、dual_target 的 zero-state、官方评测的 constant-zero state）。

任务：

1. 写出 `configs/` 四份 YAML（§3.5），每份带 `frozen_at` 与哈希。
2. **断言全部 1077 条 episode 的 `goals[0].radius == 3.0`**，把结果写入 `benchmark.yaml` 与 `reports/success_radius_check.json`，并记录官方 commit（`e9d2db12ce5788c0f987d734c0094100b6bc0d3a`）与 `measures.py` 的 SHA-256。半径来源已核对为逐 episode 读取（§3.4），此处只验证数据端一致性。
3. 定义 `BaseVLNPolicy` 抽象：输入 `Observation`，输出 `NavCommand`。evaluator 只通过它与模型交互。
4. 定义 request 字段白名单与无泄漏校验器（复用 `src/inference/preflight.py` 的思路）。
5. 抽样生成 dev100 ID 列表，写入 `benchmark.yaml`。

验收：

- [ ] 四份配置存在且带哈希，evaluator 能在启动时校验并在不匹配时拒绝运行
- [ ] 1077 条 episode 的 `goals[0].radius` 全等于 3.0（或例外已列出并有处置决定）
- [ ] `BaseVLNPolicy` 有至少两个 stub 实现（random、expert）通过同一 evaluator
- [ ] 无泄漏校验器对一个人为注入 `goal_direction` 的请求返回失败
- [ ] dev100 ID 列表冻结，按 scene 分层，可复现

停下，等 `CONTINUE P2`。

---

### P2 — 全量重构 + 回归验证

**目标**：按目标树重构，并用数值一致性证明重构没有改变行为。

目标结构：

```
vln-go2/
├── configs/            robot/ model/ dataset/ benchmark/
├── envs/               isaac_vln_env.py  episode_loader.py  observation.py
├── robots/             go2.py  locomotion_policy.py
├── policies/           base_policy.py  smolvla_policy.py  llada_language_policy.py
├── actions/            nav_command.py  language_parser.py  discrete_adapter.py
├── experts/            path_follower.py  velocity_controller.py
├── datasets/           r2r.py  isaac_go2_dataset.py  history_sampler.py
├── evaluation/         evaluator.py  vln_metrics.py  robot_metrics.py
├── scripts/            generate_data.py  train_smolvla.py  train_llada.py  evaluate.py
└── tests/
```

顺序严格为：**P0 已冻结旧证据 → 本阶段重构 → 重构后跑回归**。

回归验证（这是本阶段唯一有意义的验收）：

用重构后的代码重跑 R2R episode 1（scene `7y3sRwLe3Va`），与归档的 `20260910_ep1_v3` 逐项比对：

- `ablation_arrays.npz` 中 `command_raw` / `command_applied` / `velocity_actual_post` 数值一致（浮点容差需在报告中声明）
- 高层区间数 196、有效 RGB 数 236、停止意图区间数 13 一致
- 末端距目标 0.216 m 一致
- 独立审计全部通过项目与归档一致

不一致即为重构引入的行为改变，必须定位到具体模块后才能继续。

验收：

- [ ] 目标树完成，旧 `src/` 下无孤立活跃代码
- [ ] 远端部署路径已同步更新，`FOUNDATION_ARCHIVE.md` 补记新旧路径映射
- [ ] ep1 回归数值一致，差异项已解释
- [ ] `tests/` 全部通过
- [ ] `BaseVLNPolicy` 的 random / expert stub 在重构后仍能跑通 dev100 中至少 1 条

停下，等 `CONTINUE P3`。

---

### P3 — 数据可用性探针

**目标**：在投入任何大规模采集之前，查清公开 NaVILA R2R converted training data 能否复用，以及复用的真实代价。

这是纯 CPU / 网络工作，不占 GPU，可与 P2 并行。

必须查清并写进报告的事项：

1. **可获取性**：数据是否可下载、许可、体积、格式。
2. **Scene 覆盖**：是否包含评测用的 11 个 VLN-CE-Isaac scene。**若包含即构成 leakage**，必须剔除或改评测集。
3. **动作语义**：离散语言动作的确切集合与数值（forward 多少 cm、turn 多少度、有无 stop），以及它到本项目 `NavCommand` 的映射假设。注意：`wz ≤ 0.5 rad/s` 下 "turn 30°" 需 1.05 s ≈ 5 个 0.2 s 命令，这个映射是**假设而非监督**，必须显式记录。
4. **相机参数**：渲染高度、FOV、分辨率，与 Go2 车载相机的差值。
5. **Embodiment gap 量化**：从已有 Go2 数据里取同场景帧，与 NaVILA 帧做视角/高度对比，给出定量或至少可视化的差距证据。

产出 `reports/DATA_SOURCE_PROBE.md`，并给出三选一建议：

- **A 两阶段**：NaVILA 公开数据预训练 + 少量 Go2 rollout 做 embodiment 适配
- **B 纯 Go2 rollout**：探针发现公开数据不可用或 leakage 不可剔除时
- **C 只给 language 线用**：若 continuous 线的动作映射假设不可接受

提醒：`new_milestone.md §21` 第四大失败点正是"训练数据由 point-agent 产生，测试却用 Go2"。选 A 时必须在报告中说明用什么手段补 gap；选任何方案都不得在论文中把结论归因到未经验证的数据等价性上。

验收：

- [ ] 五项事项全部有答案，无"待定"
- [ ] Scene leakage 结论有机械核对证据
- [ ] 动作映射假设写成可执行的转换函数 + 单元测试
- [ ] 三选一建议有明确推荐及代价说明

停下，等 `CONTINUE P4`。

---

### P4 — 数据生产

**目标**：产出可训练数据集。具体规模与来源**由 P3 结论决定**，本文档不预设。

若涉及 Go2 rollout，采用**分级门禁**，每级采完先训一次、跑一次 dev100，看 SR 曲线是否还在涨：

```
50 条  →  200 条  →  500 条
```

涨幅撑不住采集成本即停。按实测基线，50 条约 4 小时墙钟 / 1.1 GB，200 条约 15 小时 / 4.4 GB。

每条 trajectory 同时保存三种 label（这是 `new_milestone.md` 最有价值的设计，完整保留）：

```json
{
  "step": 42,
  "instruction": "Walk down the hallway and turn left.",
  "rgb": "rgb/000042.jpg",
  "state": {"vx": 0.38, "vy": 0.0, "wz": 0.02},
  "action": {"vx": 0.4, "vy": 0.0, "wz": 0.0, "stop": false},
  "discrete_action": "FORWARD_M",
  "language_action": "move forward 50 centimeters"
}
```

一次采集同时供给 SmolVLA（continuous）、LLaDA-V language head、以及后续 X2 的 discrete head，不重复采集。

History 只存 frame index，训练时由 loader 按 NaVILA prefix-uniform 规则构造，不在数据层重复存图。

数据 QC（自动，沿用并扩展 `audit_r2r_ablation.py`）：轨迹是否成功、是否摔倒、是否长时间卡住、是否 collision loop、action 是否异常、RGB 是否全黑、frame 是否缺失、时间是否单调、history 是否含未来帧。

验收：

- [ ] 当级数量达标且 QC 全通过
- [ ] 三种 label 同时存在且互相一致（离散/语言标签能反算回连续命令，误差在声明容差内）
- [ ] PyTorch Dataset 能正常加载并 batch
- [ ] scene split 无泄漏
- [ ] 该级的 SR 曲线数据已产出，支撑"继续/停止"的决定

停下，等 `CONTINUE P5`。

---

### P5 — SmolVLA 基线（continuous）

**目标**：在冻结契约下重建 SmolVLA 策略，输出 `[vx, wz, stop]`。

关键变更（相对 M6/M7）：加显式 stop 头；state 改为 3 维 `[body_vx, body_vy, body_yaw_rate]`（可选输入）；history 可选。

Loss：

```
L = L_velocity + λ · L_stop
```

`vx / wz` 回归，`stop` 二分类。stop 正样本稀疏，需明确处理（加权或重采样），并在报告中写明。

强制 sanity check 阶梯（`new_milestone.md §21` 第五点，完整保留）：

```
10 episodes overfit  →  P4 当级全量  →  dev100 闭环
```

**若 10 episodes 都无法明显 overfit，禁止进入全量训练。** 先查 normalization / action scale / state encoding / image preprocessing / instruction 文本 / chunk indexing / 时间对齐——这正是 M6.2 诊断阶段付出过代价的清单。

必须复用的历史教训：M7 离线 replay 显示模型在训练集上 `vx` 拟合良好（r=0.86、R²=0.73）但 `wz` 在失败路线上 R² 为负（-0.49），闭环仍失败。**因此训练集拟合指标不构成 P5 通过条件**，必须有闭环证据。

先在 dual_target 红蓝箱上跑通再上 R2R——极小可控场景 debug 比在 R2R 上快一个量级。

验收：

- [ ] 10 episodes overfit 成功（loss 比值 + 动作 MAE 双门槛）
- [ ] 全量训练稳定，无 NaN / Inf / OOM，checkpoint 可 reload
- [ ] 输出 shape 正确，raw 值在契约范围内（零违例）
- [ ] 在 dual_target 红蓝箱上闭环跑通
- [ ] dev100 上完成至少 1 条自主成功 episode，全程无禁止输入
- [ ] 训练集 `wz` 拟合质量单独报告（不作为通过条件，但必须记录）

停下，等 `CONTINUE P6`。

---

### P6 — LLaDA-V language head

**目标**：接入第二个模型，验证同一 `BaseVLNPolicy` 下可插拔。

架构：

```
RGB + history + instruction  →  LLaDA-V  →  "move forward 50 centimeters"
                                              ↓
                                   deterministic parser
                                              ↓
                                         NavCommand
```

Parser 必须是 **regex / grammar / 有限状态机**，**禁止用另一个 LLM 做 parser**。支持 `move forward X cm`、`turn left X degrees`、`turn right X degrees`、`stop`。

无法解析时的处理必须显式定义（NOOP 或 STOP，二选一并固定），并把 `parse_error_rate` 作为模型指标记录——它本身是 language action 表示的固有成本。

延迟：按 §3.3，仿真时间对齐，墙钟可慢于实时。延迟单独报，不作为通过条件。

验收：

- [ ] Parser 单元测试 100% 通过，含畸形输入、单位变体、超范围数值
- [ ] LLaDA-V 在 3090/24GB 上可加载并推理，显存峰值已记录
- [ ] 切换 `policy=llada_language` 后 evaluator **零修改**即可运行
- [ ] `parse_error_rate` 已统计
- [ ] dev100 上完成至少 1 条自主成功 episode

停下，等 `CONTINUE P7`。

---

### P7 — 评测框架

**目标**：模型无关的 evaluator，模型只看 observation，evaluator 可见 GT。

指标三类：

**VLN**：SR（主表 = 官方 3 m；副表 1 m / 2 m）、SPL、Navigation Error、Oracle Success、nDTW、Path Length

**Robotics**：collision_count、collision_time、fall_rate、timeout_rate、episode_time、distance_traveled、average_velocity、stop_accuracy

> collision 相关指标在 X1 完成前**使用代理来源**：机体接触终止事件、跌倒、卡住、轨迹穿墙几何检测。报告中必须显式标注来源与局限，不得声称做过独立碰撞认证。

**Model**：inference_latency（mean / P95）、GPU memory、decision frequency、invalid_action_rate、parse_error_rate

每条 episode 存：

```
logs/<episode_id>/
    metrics.json  commands.jsonl  trajectory.csv  video.mp4
```

视频必存——M7 阶段的失败定位几乎全靠它。

验收：

- [ ] 同一条 recorded trajectory 重复评测得到**完全相同**的指标（确定性）
- [ ] random / expert / SmolVLA / LLaDA-V 四种 policy 走同一 evaluator，零分支代码
- [ ] 无泄漏校验在每条 episode 上自动执行并留证
- [ ] collision 指标来源已标注

停下，等 `CONTINUE P8`。

---

### P8 — 主对比 + Ablation

**Baselines**（必做）：

| Baseline | 作用 |
|---|---|
| Random Policy | 下界 |
| Expert Oracle | 上界；同时验证 episode 本身可达 |
| Image-Free Policy | `instruction embedding + previous action`，**证明模型真的用了视觉** |

Image-Free 是这三个里最重要的——没有它，"模型学会了导航"和"模型学会了数据集的动作先验"无法区分。

**主实验矩阵**：

| ID | Model | History | State | Output |
|---|---|---|---|---|
| E1 | SmolVLA | 8 | No | continuous |
| E2 | LLaDA-V | 8 | No | language |
| E4 | SmolVLA | 0 | No | continuous |
| E5 | SmolVLA | 8 | Yes | continuous |
| E6 | LLaDA-V | 0 | No | language |
| E7 | LLaDA-V | 8 | Yes | language |

（E3 discrete 见 X2。编号保留 `new_milestone.md` 原案以便对照。）

**Ablation 轴**：

1. **History**：H0 / H4 / H8
2. **State**：有 / 无
3. **Action representation**：continuous vs language（X2 完成后加 discrete）—— 主研究问题

三轴，一次只改一个变量。全部 ablation 跑 dev100；只有进主对比表的模型跑 full1077。

决策频率**不是** ablation 轴（§3.3）：5 Hz 对所有模型固定。实验预算集中在 action representation 上。

验收：

- [ ] 所有模型共享同一 evaluator、同一 episode、同一相机、同一 Go2、同一低层控制器、同一成功定义
- [ ] 三个 baseline 完成，Image-Free 与视觉模型的差距已量化
- [ ] 主表模型在 full1077 上完成
- [ ] 每个实验在 `reports/EXPERIMENTS.md` 有完整条目（Hypothesis / Method / Control Variables / Metrics / Result / Failure Cases / Conclusion）
- [ ] 失败实验同样保留

停下，等 `CONTINUE P9`。

---

### P9 — Robustness 与 Failure Analysis

**扰动**（各挑少量，不做大规模 benchmark）：

视觉：brightness、camera noise、minor blur
控制：velocity tracking noise、execution delay、small odometry drift

**Failure 分类**（对每个模型分别统计）：

```
F1 Instruction misunderstanding   F2 Wrong turn        F3 Missed landmark
F4 Oscillation                    F5 Collision         F6 Premature stop
F7 Failure to stop                F8 Parser failure    F9 Locomotion failure
F10 Timeout
```

F5 的统计来源依 X1 状态标注。F8 只对 language 线适用。

这部分通常比多跑几个指标更有实验价值——**先看真实 failure，再决定研究方向**。

验收：

- [ ] 每个模型每类 failure 有计数与代表性视频
- [ ] 扰动前后指标对比完成
- [ ] 下一步研究假设由 measured failure 支撑，而非猜测

停下，等 `CONTINUE P10`。

---

### P10 — 平台交付

`git tag platform-v1`。产出：

- `reports/BASELINE_RESULTS.md`、`outputs/results.csv`
- 复现说明：config + random seed 即可重跑任一实验
- 新模型接入指引：实现 `BaseVLNPolicy` → 直接进 benchmark，不动 simulator / evaluator

**Definition of Done**（不以 SPL 数值为标准）：

```
✓ R2R episode 稳定加载到 Isaac
✓ Go2 执行统一 NavCommand
✓ 自动生成 Isaac-Go2 VLN imitation dataset
✓ SmolVLA 完成闭环 VLN
✓ LLaDA-V language-action 完成闭环 VLN
✓ 两个模型共享完全相同 evaluator
✓ 支持 VLN-CE-Isaac full1077 评测
✓ 自动输出 SR / SPL / nDTW / NE
✓ 自动输出 collision / fall / latency 等 robot metrics（collision 来源已标注）
✓ 每条 episode 保存 trajectory 与 video
✓ config + seed 可重现
✓ 不修改 simulator / evaluator 即可插入新 policy
```

---

## 6. 旁支里程碑（不阻塞主线）

### X1 — 接触力修复（可选）

**现状**：19 个 body 接触力恒为零。已排除：官方默认 TensorAPI 设置、`disable_contact_processing` 开关对照、强制刷新传感器、直接 PhysX 接口、200 Hz 物理子步取值。原因未定位。

触发条件：主线空闲或 collision 成为结论的关键瓶颈。

成功条件：正常脚部支撑产生合理非零力，且能与真实碰撞区分（正常支撑力不可简单标为碰撞）。

修复后：开放 F5 与 collision 消融，回填历史轨迹的碰撞标注，并在报告中更新指标来源说明。

不修复的后果：主线继续用代理指标，论文明写局限。**不构成主线阻塞。**

### X2 — LLaDA-V discrete head（后排）

`MLP(h) → action logits`，预测 `STOP / FORWARD_{S,M,L} / LEFT_{S,M,L} / RIGHT_{S,M,L}`，再经 `ActionAdapter` 转 `NavCommand`。

Codebook（P4 数据已含 `discrete_action` 标签，无需重采）：

```
FORWARD_S = 0.25 m    TURN_S = 15°
FORWARD_M = 0.50 m    TURN_M = 30°
FORWARD_L = 0.75 m    TURN_L = 45°
```

注意：`wz ≤ 0.5 rad/s` 下 `TURN_L` 需 1.57 s ≈ 8 个 0.2 s 命令，adapter 必须处理多命令展开，且这是**执行假设而非监督**，与 P3 的映射问题同源。

完成后补 E3 进主对比表，形成 continuous / language / discrete 三点对比。

---

## 7. 全程风险清单

沿用 `new_milestone.md §21`，按本项目实际状况标注：

| 风险 | 状态 | 防线 |
|---|---|---|
| 坐标映射错误 | **已缓解** —— ep1 实测走完 7.09 m 到 0.216 m | P2 回归验证保持 |
| high-level command duration 不明确 | **已决定不引入 duration**，改用固定 5 Hz + ZOH，时间语义明确写在 `action.yaml` | 这是本计划接受的已知局限：单命令时长不可变。若 P9 的 failure taxonomy 显示 F4 Oscillation 或 F7 Failure-to-stop 集中且归因到时间尺度，再作为后续研究重开 duration 议题 |
| 不同模型信息不一致 | **活跃风险** —— 当前三条线三套协议 | P1 冻结 Observation API + 每 episode 无泄漏校验 |
| point-agent 数据 vs Go2 测试 | **活跃风险** —— P3 正是为它设的 | P3 量化 gap；选 A 方案时必须有 embodiment 适配数据 |
| 平台没跑通就烧 GPU | **已发生过** —— M7 四轮修复 | P5 强制 overfit 阶梯；P4 分级门禁；dual_target 先行 debug |

追加两条本项目特有的：

| 风险 | 防线 |
|---|---|
| 共享 GPU 抢占 / 队列阻塞 | 复用 `gpu_wait.py` 的共享锁、显存准入、运行余量、只清自有进程组；每阶段记录 VRAM |
| 实时率 0.203 导致规模估算失真 | 一切规模估算引用 §2 实测基线，禁止用理论吞吐排期 |

---

## 附录 A — 历史证据（不进新主线）

以下工作真实存在且有完整证据，但因接口不兼容或外部阻塞不进入本计划主线。保留在此，避免日后无法回答"那些工作去哪了"。

### A.1 M0–M7 短程 VLN 线

`reports/MILESTONES.md`。M0–M6 全部 PASS；M7 FAIL。

- M7 首次闭环在 `short_vln_v1_0004` 上官方 success=1 / SPL=1 / 距目标 0.279893 m，但 raw action 37/2100 向量超界。
- M7-R3 引入 bounded action codec（`vx=0.25(tanh+1)`、`wz=0.5·tanh`）后 raw range 零违例（0/7500），但在 1500 帧上限处超时，末端误差 1.030050 m。
- M7 离线 replay（847 训练观测、三种噪声种子）：首步 `vx` MSE/MAE/r/R² = 0.002506 / 0.037369 / 0.861076 / 0.728022；`wz` = 0.032434 / 0.118126 / 0.856329 / 0.675405。在失败路线上 `wz` R² = -0.493863。

**结论（对新计划的输入）**：失败机制是 yaw 随机性、distribution shift 与 rollout 误差累积，**不是**协议、动作边界或 codec。P5 必须以闭环证据而非训练集拟合作为通过条件。

不复用原因：30-D state 协议、无 stop 头、bounded codec 与新契约不兼容。

### A.2 M7-N 语义 short-route

`reports/M7_N_N0_DESIGN.md`、`reports/M7_N_N1_GEOMETRY_PREVIEW.md`。

扫描 1077 个官方 NaVILA parent（commit `e9d2db12ce5788c0f987d734c0094100b6bc0d3a`），产出 3842 条 1.5–2.8 m 几何预候选，跨 11 scenes，manifest 54 MiB，SHA-256 `d17c259178d94ecf1595c3a8fc32b48facf44883e6f779694bd2c3e2afcbe340`。

**阻塞**：NaVILA 树含 mpcat40 映射但无 `.ply` 语义 mesh，语义预览 fail-closed。全部候选停留在 `pending_surface_and_semantic_preview`，均非训练数据。

### A.3 官方 NaVILA velocity benchmark（已停止）

`reports/OFFICIAL_NAVILA_VELOCITY_0911.md`。

使用 checkpoint `dual_target_v2/zoh_no_state_expert10k_autodl_0909_v1/checkpoint_010000`，5 Hz、预测 5 步执行首步、hold 10 个官方控制步、constant-zero 3 维 state、STOP 用 raw 阈值法。

官方 index 0 / episode ID 1 完成：2501 控制步、2502 视频帧，terminated by `official_time_limit`，success=0、SPL=0、path_length=1.5353069606、distance_to_goal=8.1353857885、oracle_navigation_error=8.1274131539。

于 P0 停止。已完成 episode 与 `progress.json` / `plan.json` / `results.json` 全部保留。

不复用原因：3 维无 stop 头接口 + constant-zero state，与新契约不兼容，结果不可与新结果并列。

### A.4 dual_target 红蓝箱（**保留使用**）

`CODEX_SMOLVLA_GO2_DUAL_TARGET_MILESTONES.md`、`config/dual_target_v1/task_contract.lock.json`。

固定指令 `Go to the red box and stop in front of it.` / `Go to the blue box and stop in front of it.`。DT0–DT7 里程碑体系，产出了目前最好的 checkpoint。

**在新计划中的角色**：极小可控 sanity-check 场景。任何新模型、新接口在上 R2R 之前先在这里跑通——比在 R2R 上 debug 快一个量级。P5 已把它写进验收。

### A.5 D5 / M6.2 诊断资产

`reports/M6_2_DIAGNOSIS.md`、`reports/D5_R*.md`、`src/smolvla/analyze_temporal_alignment.py`、`analyze_train_actions.py`、`evaluate_action_codec.py`。

诊断**方法**可复用（时间对齐分析、动作分布分析、codec 评估），诊断产出的 checkpoint 与数据不复用。P5 的 sanity check 清单直接来自这批诊断的教训。

---

## 附录 B — 与旧文档的关系

| 文档 | 状态 |
|---|---|
| `CODEX_GO2_SMOLVLA_SHORT_VLN_RUNBOOK.md` | 审计参考。M0–M8 记录见 `reports/MILESTONES.md` |
| `CODEX_SMOLVLA_GO2_DUAL_TARGET_MILESTONES.md` | 审计参考；红蓝箱场景本身按 A.4 继续使用 |
| `M6_2_DIAGNOSTIC_MILESTONES.md` | 审计参考 |
| `new_milestone.md` | 已合并入本文档，差异见 §0 |
| **本文档** | **唯一执行入口** |

冲突时以真实源码与实测证据为准：先查当前源码 / README，明确说明差异，以真实支持的 API 为准，禁止编造不存在的文件、字段或 CLI 参数。

实验诚信要求不变：所有数字必须保留 command、config、checkpoint、result file、相关日志；不稳定结果必须说明；失败实验同样保留。
