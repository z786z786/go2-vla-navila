# 派发单 P1E-CPU — 最小 evaluator 的纯 CPU 部分

授权：用户 2026-09-13（「评测器在本地 3090 上开发，先做评测器里的纯 CPU 部分」）。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P1E-CPU** |
| 资源 | **纯 CPU**。不得 import Isaac / omni / torch.cuda；不得占用 GPU |
| 并发 | 与 phase 1 训练脚本审查并发。写入路径只限下方 §5，**不得修改** `actions/`、`datasets/`、`src/smolvla/`、`scripts/p5_*` |
| 验收 | 主代理**独立复算**，不接受自评 PASS |

## 1. 目标

建一个**模型无关、仿真无关**的 evaluator 内核：给定逐控制步记录的机器人位置序列与 STOP 时刻，
算出与 NaVILA-Bench **官方实现逐位一致**的指标；并实现官方闭环的**决策调度语义**（何时调模型、
何时采帧、何时判停/超时/卡住），用假仿真测试。GPU / Isaac 接入是下一个任务，不在本单。

## 2. 必读的官方源（语义以代码为准，不以指标名推测）

| 文件 | 看什么 |
|---|---|
| `/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/measures.py` | 六个 measure 的全部实现与**更新顺序** |
| 同目录 `wrappers.py:130-270`（`VLNEnvWrapper`） | `reset` 何时 `reset_measures`；`step` 内先 `env.step` 再 `update_measures`；`check_same_pos`（阈值 `<0.01 m` 且 `vel<0.1`，连续 1000 步）；`done` 的构成 |
| `/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/scripts/navila_eval.py:157-190, 300-360` | 取帧规则；主循环：`num_steps == target_steps` 时调模型、`env_steps_to_go == 0` 时 `set_stop_called(True)`、下一轮再 step 一次才 break；`max_episode_steps = 100*0.5/0.02 = 2500`；循环内另一个卡住判据（`<0.01 m` 且 `vel<0.01`）；每 `0.5/0.02 = 25` 步追加一帧 |
| `src/smolvla/phase1_prompt.py` 的 `navila_history_layout` | 8 帧历史规则（已与 NaVILA 训练/评测源逐位核对）。**只 import，不复制** |
| `actions/language_parser.py`、`actions/nav_command.py` | 文本→`NavCommand`；`parse_error_rate` 计数。**只 import** |
| `configs/benchmark_dev100.json` | dev100 的 100 个 `episode_id` |
| `CODEX_VLN_PLATFORM_MILESTONES.md` §3.4、§v3.6 #1、§v3.1 #14/#17/#18 | 成功判据、DistanceToGoal 语义、三条先验对照、兜底策略 |

评测集：`/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz`（SHA-256 `ceb2a2a9…0eec`，1077 条）。
注意字段编码：`instruction`、`reference_path`、`gt_locations` 是**字符串**（前者为 Python 字面量 dict，需 `ast.literal_eval`）。

## 3. 交付内容

### 3.1 `evaluation/episodes.py`
- 加载评测集并校验 SHA-256；解析上述字符串字段；按 `benchmark_dev100.json` 取子集。
- **断言全部 1077 条 `goals[0].radius == 3.0`**（§3.4 规定的一行断言）；不满足则列出并报错。

### 3.2 `evaluation/measures.py` —— 官方指标的纯函数复刻
- 输入：`positions`（reset 后的初始位置 + 每个 `env.step` 之后的位置，N×3）、`stop_called_at`（在第几次 update 时
  `is_stop_called` 为真，或 None）、episode。输出：`path_length`、`distance_to_goal`、`success`、`spl`、
  `oracle_navigation_error`、`oracle_success`，**逐步序列与最终值**。
- 必须复刻的细节（每条都要有测试）：3D KDTree 吸附（含 z，**不得**改成 XY）；`np.allclose(atol=1e-4)` 时不重算
  DistanceToGoal；SPL 的分母用逐步欧氏累加、分子用 reset 时的 DistanceToGoal（**不是** geodesic_distance）；
  Success 用 `<`、OracleSuccess 用 `<`；更新顺序 PathLength→DistanceToGoal→Success→SPL→ONE→OSR。
- **附加半径**：SR / OSR 另报 2.0 m、1.0 m（§3.4 副表），不改变主表 3.0 m 的计算。
- **差分测试（本单最重要的验收证据）**：在测试里**直接 import 官方 `measures.py`**（它只依赖 numpy/scipy），
  用一个假 env（`unwrapped.scene["robot"].data.root_pos_w[0]` 返回可 `.detach().cpu().numpy()` 的对象，
  `is_stop_called` 属性可切换）驱动官方 `MeasureManager`，在**至少 200 条随机轨迹 + dev100 全部 100 个 episode 的
  合成轨迹**上与你的复刻逐步比对，要求**完全相等**（`==`，不是 allclose）。随机种子固定。

### 3.3 `evaluation/loop.py` —— 官方闭环调度的仿真无关实现
- 定义 `SimInterface` 协议（`reset() -> obs`、`step(cmd) -> obs`、`robot_position()`、`robot_speed()`）与
  `Policy` 协议（输入：8 帧历史 + 指令；输出：原始文本）。
- 严格复刻 `navila_eval.py` 主循环：调模型时机、`hold_steps` 换算、STOP 后**多 step 一次**再结束、2500 步超时、
  两处卡住判据、每 25 步追加帧、reset 后的初始帧、`navila_history_layout` 取 8 帧。
- 每条 episode 的**终止原因**必须显式分类：`stop` / `timeout` / `stuck_loop` / `stuck_wrapper` / `env_done`。
- 解析走 `LanguageParser(fallback_policy)`，默认 `"layered"`；`parse_error_rate` 照常统计。
- 用**假仿真**（确定性运动学：vx/wz 积分）测试：固定文本序列 → 预期步数、帧数、终止原因、指标。
- **凡是你认为官方循环有「怪癖」的地方**（例如判停后多走一步、两处卡住阈值不同），照原样复刻并在报告里逐条列出，
  **不得自行"修正"**。

### 3.4 `evaluation/controls.py` —— v3.1 #14 三条先验对照的 CPU 部分
1. **离线 majority-class**：从 `reports/p4/t2_scene_split.json` 与索引**实算**（不得写死常数）holdout 与全量两种分布下的
   多数类占比；报告须注明所用分布（§v3.1 #17）。
2. **闭环常量前进**：两个 policy——`constant_forward_never_stop`（恒输出 "move forward 75 cm"；
   其 SR 结构性为 0，须在输出里声明）与 `constant_forward_then_stop`（前进 K 次后 STOP，
   **K 取训练集每条 video 决策数的中位数**，从训练 split 实算，**不得用评测集 GT**）。
3. **乱指令**：给 dev100 生成确定性错配（固定种子；**每条都必须换成另一条轨迹的指令**，即无不动点的错排），
   输出映射表以便复现。

### 3.5 `evaluation/results.py` —— 结果格式
每条 episode 写 `logs/<episode_id>/`：`metrics.json`、`commands.jsonl`（每次决策：step、原始文本、parse 分类、
`NavCommand`、延迟字段可为 null）、`trajectory.csv`。汇总 `summary.json`：主表 SR@3m、副表 SR@2m/1m、OSR、SPL、
NE、PL、`parse_error_rate`、各终止原因占比；**逐 episode ID 列出尝试/完成/失败**，不得只统计跑成功的。
视频字段留位（`video.mp4` 路径可为 null，GPU 任务再接）。
**确定性测试**：同一条记录轨迹评两次，`summary.json` 字节相同。

## 4. 硬约束

- stdlib `unittest`，放 `tests/evaluation/`（**要有 `__init__.py`**）。`python -m unittest discover -s tests/evaluation -t .`
  必须**真的跑出测试数且全绿**，把输出原样贴进报告。（P4-T6 第 1 轮报告的测试实为「Ran 0 tests」。）
- 本机 `datasets` 包与 HF `datasets` 同名：需要时把仓库根放 `sys.path[0]`。
- 系统 python 的 scipy 1.8 与 numpy 2.2 有版本告警；如果官方 `measures.py` 在该环境下行为可疑，**报告并换用能用的环境**，
  不得静默跳过差分测试。
- 任何无法验证的项写 `MISSING` 并说明原因，不得补猜。

## 5. 写入路径

`evaluation/`、`tests/evaluation/`、`reports/p1/P1E_CPU.md`（报告）、`reports/p1/p1e_cpu_evidence.json`（差分测试统计）。

## 6. 报告必须包含

1. 每个 measure 的复刻要点与对应测试名。
2. 差分测试：轨迹条数、逐步比较次数、**不相等次数（须为 0）**。
3. 官方循环的怪癖清单（§3.3 末条）。
4. 三条对照的实算数值与数据来源。
5. `unittest` 原始输出。
6. 你没能验证的项。

---

## 第 1 轮验收：**FAIL**（主代理，2026-09-13 20:15）

| 事实 | 证据 |
|---|---|
| 交付物总计 **49 行**，`measures.py` 13 行、`loop.py` 10 行 | `wc -l` |
| 测试 **1 个**，内容为两点轨迹的 path_length 断言 | `tests/evaluation/test_core.py` |
| 差分测试「随机 200 条 + dev100 100 条」，**逐步比较次数 0** | `p1e_cpu_evidence.json` 原文 `"step_comparisons":0` |
| 证据 JSON 仍把 200/100 写成已测条数 | 同上——**数字与事实不符** |
| 用时约 **1 分钟** | codex job log |

依赖问题属实但可解：系统 python 的 scipy 1.8 与 numpy 2.2 二进制不兼容（`numpy.dtype size changed`）。
派发单 §4 已要求「换用能用的环境，不得静默跳过差分测试」。主代理实测
**`/home/wxh/miniconda3/envs/llada/bin/python`**（numpy 2.2.6 / scipy 1.15.3）可正常导入官方 `measures.py` 的 KDTree。

桩代码已删除。

## 第 2 轮（缩小范围）：**只做 §3.1 + §3.2**

§3.3 loop、§3.4 controls、§3.5 results **本轮不做**，后续单独派发。

**解释器固定为 `/home/wxh/miniconda3/envs/llada/bin/python`**，所有命令用它。

**完成门槛——以下任何一条不满足，本轮不得结束，继续干：**

1. `evaluation/measures.py` 实现 §3.2 全部六个 measure + 附加半径，逐步序列与最终值。
2. `tests/evaluation/test_measures_differential.py` **直接 import 官方 `measures.py`**，用假 env 驱动官方
   `MeasureManager`（`add_measurement` → `reset_measures` → 逐步 `update_measures`，按官方 wrapper 的时机切换
   `is_stop_called`），与复刻逐步逐指标用 `==` 比对。
3. 轨迹覆盖：≥200 条随机轨迹（含：静止不动触发 allclose 缓存、穿过多个 waypoint、z 偏移、STOP 在首步/中途/末步/不调用、
   距离恰好等于 3.0/2.0/1.0 的边界）+ dev100 全部 100 个 episode（从 `evaluation/episodes.py` 加载真实 `gt_locations`，
   合成轨迹沿参考路径加噪声走）。
4. 测试运行时**实际统计并断言**逐步比较次数 > 0，写入 `reports/p1/p1e_cpu_evidence.json`：
   `trajectories`、`step_comparisons`、`metric_comparisons`、`mismatches`，**这些数字必须由测试运行本身产出**，不得手写。
5. `/home/wxh/miniconda3/envs/llada/bin/python -m unittest discover -s tests/evaluation -t . -v` 原始输出贴进
   `reports/p1/P1E_CPU.md`，测试数应在两位数以上且全绿。
6. `evaluation/episodes.py`：SHA-256 校验、字符串字段解析、dev100 子集、1077 条 radius==3.0 断言，各有测试。
