# 派发单 P4-T3 — duration 制 NavCommand adapter + 确定性 parser

授权：用户 2026-09-13 `CONTINUE P4′`。波次：W1（与 P4-T1、P4-T2 并发）。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P4-T3** |
| 资源 | **CPU only。零 GPU，零网络，零数据依赖。** 纯代码 + 单元测试 |
| 并发 | 与 T1 / T3 并发。只写 `actions/`、`tests/actions/`、`reports/p4/t3_*` |

## 1. 目标

实现 phase 1 的动作接口：模型文本 → `NavCommand`。这是长期代码资产，不是一次性脚本。
**禁止用另一个 LLM 做 parser**——必须是 regex / grammar / 有限状态机。

## 2. 契约（里程碑文档 v3 修订后的冻结值，逐条照此实现）

```python
NavCommand:
    vx:         float   # [0.0, 0.5] m/s
    vy:         float   # 固定 0.0
    wz:         float   # [-pi/6, +pi/6] rad/s   (pi/6 = 0.5235987755982988)
    hold_steps: int     # {25, 50, 75}，或 0（stop）
    stop:       bool
```

- **时序**：`hold_steps = duration / (sim.dt * decimation) = duration / 0.02`，
  `duration ∈ {0.5, 1.0, 1.5} s` → `hold_steps ∈ {25, 50, 75}`。低层 50 Hz、`sim.dt=0.005`、`decimation=4`。
- **v3 已撤回**「固定 5 Hz / 取消 duration」与「wz ∈ ±0.5」，现按 NaVILA 原生：`wz = ±pi/6`、`vx = 0.5`。
- 原生参考实现：`/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py:57-85`
  （SHA-256 `c42b832a6e37b786f8c42e42f9ba790ca88fbb4fcf1cad8822d9695abc1e5bb0`），函数 `get_vel_command`。

### 2.1 词表映射（数据实测恰好 10 条，须写死）

| # | 动作文本（原样） | vx | wz | hold_steps | stop |
|---|---|---|---|---|---|
| 0 | `The next action is move forward 25 cm.` | 0.5 | 0 | 25 | F |
| 1 | `The next action is move forward 50 cm.` | 0.5 | 0 | 50 | F |
| 2 | `The next action is move forward 75 cm.` | 0.5 | 0 | 75 | F |
| 3 | `The next action is turn left 15 degree.` | 0 | +pi/6 | 25 | F |
| 4 | `The next action is turn left 30 degree.` | 0 | +pi/6 | 50 | F |
| 5 | `The next action is turn left 45 degree.` | 0 | +pi/6 | 75 | F |
| 6 | `The next action is turn right 15 degree.` | 0 | -pi/6 | 25 | F |
| 7 | `The next action is turn right 30 degree.` | 0 | -pi/6 | 50 | F |
| 8 | `The next action is turn right 45 degree.` | 0 | -pi/6 | 75 | F |
| 9 | `I think I should stop because I have finished the instruction.` | 0 | 0 | 0 | **T** |

### 2.2 解析失败兜底：三策略 + 配置开关（用户 2026-09-13 裁决）

原生 `get_vel_command` 有**四层**兜底，不是一层：

| 情形 | 原生行为 |
|---|---|
| 命中 `turn left` 但无 15/30/45 | `+pi/6, 0.5 s`（15°） |
| 命中 `turn right` 但无数字 | `-pi/6, 0.5 s` |
| 命中 `move forward` 或仅含 `move`，但无数字 | `vx=0.5, 0.5 s`（前进 25 cm） |
| **一个关键词都不命中** | `vx=0.5, 0.5 s`（**前进 25 cm**） |

实现三种策略，由配置项 `fallback_policy` 切换：

| 值 | 语义 |
|---|---|
| **`layered`**（**默认值**） | 部分命中按原生取最小档；**完全不命中 → 零速度意图 + 交由 limiter 制动**（`vx=0, wz=0, hold_steps=25, stop=False`，并标记 `parse_failure`） |
| `native` | 四层全照搬原生，完全不命中也前进 25 cm |
| `brake` | 四层全部改为零速度 + 制动 |

默认值由主代理裁定为 `layered`，理由见里程碑文档 §v3.10（服务于「看视频归因」的能力门禁）。
**不得更改默认值**；三种都必须实现且可切换。

### 2.3 `parse_error_rate` 分层计数（强制）

parser 必须返回/累计结构化的解析结果分类，**兜底不得吞掉错误**：

- `exact`：命中 §2.1 十条之一
- `partial_match`：命中关键词但缺数字或数字不在 {15,30,45}/{25,50,75}
- `total_miss`：无关键词命中
- 并区分子类计数，供报告输出 `parse_error_rate = (partial_match + total_miss) / total`

## 3. 必须覆盖的单元测试（至少这些，可加）

1. §2.1 十条**逐字**输入 → 逐字段断言（含 `hold_steps` 与 `wz` 符号）。
2. 大小写变体、首尾空白、多余空格、句末缺句点。
3. **原生子串匹配的已知怪癖**：`"turn left 145 degree"` 在原生下因 `"45" in text` 会被判为 45°。
   三种策略下的行为必须**各自被断言并在文档中写明**，不得悄悄"修正"原生行为——
   若实现选择比原生更严格的数字提取，必须显式记录该偏离及理由。
4. 关键词顺序：原生先判 `turn left`，再 `turn right`，再 `move`。同时含多个关键词的文本走哪条，须断言。
5. 仅含 `"move"` 而无 `"forward"` 的文本（原生也会命中 forward 分支）。
6. 空字符串、纯空白、超长乱码、非 ASCII → `total_miss`，三种策略行为各自断言。
7. 数字越界：`turn left 90 degree`、`move forward 200 cm` → 分类与各策略行为。
8. `stop` 的变体：包含 `stop` 但非第 9 条原文者，分类与行为。
9. `hold_steps` 换算的边界：断言 `duration/0.02` 对 0.5/1.0/1.5 精确得 25/50/75（不得出现浮点误差导致 24 或 74）。
10. `NavCommand` 的取值域断言：`vx ∈ [0, 0.5]`、`vy == 0.0`、`|wz| <= pi/6 + 1e-12`、`hold_steps ∈ {0,25,50,75}`。

## 4. 输出

| 路径 | 内容 |
|---|---|
| `/home/wxh/go2_short_vln/actions/__init__.py` | — |
| `/home/wxh/go2_short_vln/actions/nav_command.py` | `NavCommand` 数据类 + 取值域校验 |
| `/home/wxh/go2_short_vln/actions/language_parser.py` | parser + 三种 `fallback_policy` + 分层计数 |
| `/home/wxh/go2_short_vln/tests/actions/__init__.py` | — |
| `/home/wxh/go2_short_vln/tests/actions/test_language_parser.py` | §3 全部用例 |
| `/home/wxh/go2_short_vln/reports/p4/t3_parser_contract.json` | 机器可读契约：词表映射表、三策略语义、分类定义、已知偏离原生之处 |
| `/home/wxh/go2_short_vln/reports/p4/T3_PARSER.md` | 人读摘要，含与原生的逐条差异表 |

测试须能用 `python3 -m pytest tests/actions -q` 跑通（若环境无 pytest，改用 `unittest` 并在文档注明运行命令）。

## 5. 验收标准（主代理如何独立复算）

1. 主代理**独立重跑**测试套件，须全绿；并记录用例数。
2. 主代理**自己另写**一组对抗性输入（不看子代理的测试），逐条核对三种策略的输出与 `t3_parser_contract.json` 声明一致。
3. 主代理断言默认 `fallback_policy == "layered"`（读代码默认参数，非读文档）。
4. 主代理对 §2.1 十条逐字输入独立断言 `hold_steps` 与 `wz` 符号。
5. 主代理断言 parser 中**不存在**任何 LLM / 网络调用。
6. 主代理核对 `t3_parser_contract.json` 中「偏离原生之处」一节非空或显式声明为空，且与代码行为一致。

**子代理的 `passed` 字段、测试通过数不构成 PASS 依据。**

## 6. 禁止事项

1. **不得用 LLM 做 parser**，不得联网，不得占 GPU。
2. 不得更改 §2.2 的默认值 `layered`。
3. 不得改动 `configs/`、`.gitignore`、`CODEX_VLN_PLATFORM_MILESTONES.md`、`reports/p3/`。
4. 不得触碰 P4-T1 / P4-T2 的输出路径（解包目录、`reports/p4/t1_*`、`reports/p4/t2_*`、
   `/mnt/.../data/navila_dataset/`、`scripts/p4_extract_navila.*`、`scripts/p4_build_navila_index.py`）。
5. 不得改动既有 `src/` 下任何文件（本任务新建 `actions/`，与旧代码并存，见 v3.1 #13 最小垂直切片）。
6. 不得 `git add` / `git commit` / `git tag`。
7. 对原生行为的任何"修正"必须显式记录为偏离，不得静默改掉。

## 7. 验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| — | — | 待派发 | — |
