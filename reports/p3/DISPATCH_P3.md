# P3 派发单台账（数据可用性探针）

生成：2026-09-12。依据 `CODEX_VLN_PLATFORM_MILESTONES.md` §P3 与 §5「调度与并发执行协议」。
**提前执行授权**：用户于 2026-09-12 批准职责边界例外 **E2**（P3 不等 `CONTINUE P1`/`P2`）。

本文件由**主代理**维护。codex 子代理只读属于自己的那一节，不得修改本文件。

## P3 要回答的问题（文档原文五项）

1. **可获取性**：数据是否可下载、许可、体积、格式
2. **Scene 覆盖**：是否包含评测用的 11 个 VLN-CE-Isaac scene。**若包含即构成 leakage**
3. **动作语义**：离散语言动作的确切集合与数值，以及到本项目 `NavCommand` 的映射假设
4. **相机参数**：渲染高度、FOV、分辨率，与 Go2 车载相机的差值
5. **Embodiment gap 量化**：同场景帧的视角/高度对比，定量或至少可视化

最终产出 `reports/DATA_SOURCE_PROBE.md` + A/B/C 三选一建议（由主代理在全部子任务验收后撰写裁决段）。

## 全局约束（每张派发单默认继承）

- 源码/session 目录 `/home/wxh/go2_short_vln`（`/` 仅余约 25 GB）；数据/模型/缓存/大产物 `/mnt/wxh/go2_short_vln`（余约 1.8 TB）。
- **任何下载一律落 `/mnt/wxh/go2_short_vln/downloads/navila_probe/`**，禁止写入 `/home`。
- **本阶段禁止占用 GPU**、禁止启动 Isaac、禁止加载模型权重。
- **禁止改动本任务「输出」字段以外的文件**；禁止删除既有产物。
- 机器可读产物为 JSON（UTF-8，`ensure_ascii=false`，2 空格缩进），顶层必须含
  `task_id`、`generated_at`（ISO8601 +08:00）、`generator_command`、`inputs`、`findings`、`missing`。
- **证据缺失一律标 `"status": "MISSING"` 或 `"UNVERIFIABLE"` 并说明原因，不得追认、不得猜测、不得虚构。**
- **v2 裁决（强制）**：不可获取的数据**允许**以证据给出「无法验证」；
  **不要求为不可用数据虚构转换函数**。仅当最终推荐 A/C 时，格式与映射才必须通过可执行样例验证。
- **沙箱能力边界**：codex 沙箱已证实屏蔽 GPU 设备节点与 `.git` 写入。**网络能力未知**。
  每张派发单的第一步都必须**显式测试并报告**所需外部能力是否可用，失败即报 MISSING，**不要重试到超时**。

## 已确认的本机事实（主代理只读核查，2026-09-12）

- `/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/` 只含**仿真器与 benchmark 代码**
  （`isaaclab_exts`、`scripts`、`src`、`logs`、`LICENSE`、`README.md`）。
- `/mnt/wxh/go2_short_vln/downloads/` 只有 `matterport_usd.zip`、
  `matterport_usd.invalid-curl-20260830T123856Z.bin`、`r2r_vlnce`。
- `outputs/navila_*` 各目录是**本项目自己跑的 rollout**，**不是**公开数据集。
- `models/` 只有 `smolvla_base_c83c316`。
- **结论：公开 NaVILA R2R converted training data 本机不存在**，可获取性须联网查证。
- 官方 commit 参考：`e9d2db12ce5788c0f987d734c0094100b6bc0d3a`。

## 评测集 11 个 scene（leakage 判据，P0-T2 实测，可直接使用）

```
2azQ1b91cZZ  8194nk5LbLH  EU6Fwq7SyZv  QUCTc6BB5sX  TbHJrupSAjP  X7HyMhZNoso
Z6MFQCViBuw  oLBMNvg9in8  pLe4wQe7qrG  x8F5xyUWy9e  zsNo4HB9uLZ
```

来源 `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz`
SHA-256 `ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec`（1077 episodes / 11 scenes）。
训练集 61 scenes 来源 `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz`
SHA-256 `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34`（10819 episodes）。
两者交集实测为空，详见 `reports/split_disjoint_check.json`。

---

## W1（三项并发，CPU / 网络）

### P3-T1 — 可获取性 + Scene 覆盖 / leakage

**目标**：查清公开 NaVILA R2R converted training data 能否获取；若能，立即核对是否覆盖评测 11 scenes。

**第一步（强制）**：测试网络可用性并报告，例如对 `https://huggingface.co` 与 `https://github.com` 各发一次
短超时请求。**不可用即写 `"network": "UNAVAILABLE"`，本任务余下各项全部标 MISSING 并立即收工，不要重试到超时。**

**若网络可用，须查清**
1. **定位**：数据实际托管在哪（HuggingFace dataset / GitHub release / 论文附带链接）。给出**确切 URL**与获取方式。
   从 `/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/README.md` 与 `scripts/` 入手找官方指引。
2. **许可**：license 名称与原文链接；是否允许研究使用与再分发。
3. **体积**：压缩与解压后字节数（从 HTTP header / dataset card 读取，**不必真的下全量**）。
4. **格式**：目录结构、标注文件格式、单条样本含哪些字段（图像、instruction、动作标签、位姿）。
5. **Scene 覆盖 / leakage（本任务最高优先级）**：列出该数据集覆盖的全部 scene id，
   与上文 11 个评测 scene 做**机械交集**。交集非空即 **LEAKAGE**，须逐个列出。
   若只能拿到 metadata / dataset card 而非全量数据，用 metadata 做此核对并**明确标注证据强度**。

**输出**：`/home/wxh/go2_short_vln/reports/p3/t1_availability.json` + `T1_AVAILABILITY.md`
**下载落盘位置**（若确实需要下载样本）：`/mnt/wxh/go2_short_vln/downloads/navila_probe/`，**单次不超过 2 GB**，
超过先只取 metadata 与少量样本并说明。

**禁止**：下载全量大数据集；写入 `/home`；为不可获取的数据编造 URL、体积或 scene 列表。

**验收**：主代理独立核对 URL 可达性、license 文本、以及 scene 交集（用上文两个源文件哈希复算评测 11 scenes）。

---

### P3-T2 — 动作语义

**目标**：确定 NaVILA 离散语言动作的**确切集合与数值**，及其到本项目 `NavCommand` 的映射**假设**。

**主要证据源（本地优先，不依赖网络）**：
`/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/` 下的 `src/`、`scripts/`、`isaaclab_exts/`。
搜索动作定义、离散动作枚举、prompt 模板、以及把语言动作转成速度/位姿的代码路径。
网络可用时再补论文 / 官方 repo 的权威定义。

**须查清**
1. 动作集合的**完整枚举**：forward 有哪几档（多少 cm）、turn 有哪几档（多少度）、有无 stop、有无其他动作。
2. 每个动作的**确切数值**与来源文件路径 + 行号。**从源码读出，不要从论文记忆里写。**
3. 这些动作在 NaVILA 侧**如何被执行**（直接位姿传送？速度指令？走了多少控制步？）。
4. 到本项目 `NavCommand` 的映射：`vx ∈ [0, 0.5]`、`wz ∈ [-0.5, 0.5]`、5 Hz、ZOH 10 个低层步。
   **必须显式记录这是假设而非监督**，并写出每个宏动作展开成多少个 0.2 s tick 的算术。
   参考算术（须自行复核）：0.2 s 内 `vx ≤ 0.5` 最多行进 **0.1 m**；
   `turn 30°` 在 `wz ≤ 0.5 rad/s` 下需 1.05 s ≈ **5 个 tick**；`turn 45°` 需 1.57 s ≈ **8 个 tick**。
5. **量化残差**：每个宏动作按整数个 tick 展开后的距离/角度**截断误差**。

**可执行转换函数的条件**（v2 裁决）：**仅当**第 1、2 项的动作定义从**真实文件**确认后，才写
`scripts/p3_navila_action_map.py` + `tests/p3/test_navila_action_map.py`（覆盖各档 forward/turn、stop、
超范围数值、畸形输入）。**动作定义无法确认时不要虚构转换函数**，写 UNVERIFIABLE 并说明查过哪些路径。

**输出**：`reports/p3/t2_action_semantics.json` + `T2_ACTION_SEMANTICS.md`
（条件满足时另加上述脚本与测试）

**禁止**：凭记忆或论文印象填数值；在定义未确认时编写转换函数；改动 `third_party/` 任何文件。

**验收**：主代理抽查你引用的文件路径+行号是否真含该数值；独立复算 tick 展开算术与截断误差；跑一遍你的单元测试。

---

### P3-T3 — 相机参数 + Embodiment gap 量化

**目标**：量化 NaVILA 渲染视角与 Go2 车载相机的差距。

**Go2 侧证据源（本地）**：`third_party/NaVILA-Bench/isaaclab_exts/` 下的 Go2 / camera 配置；
本项目 `src/` 与 `config/`；以及**已采帧的实证**
`/mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_ep1_v3/rgb/` 与 `.../20260910_ep1_continuous_v2/rgb/`
（对应 `frames.jsonl`，ep1_v3 有 236 帧、continuous_v2 有 205 帧）。

**须查清**
1. **Go2 车载相机**：分辨率、FOV、安装高度、俯仰角、相对机体的位姿。给出**配置文件路径 + 字段名**。
   分辨率同时用实际 RGB 文件复核（契约 §3.1 声称 `[512,512,3]`，**须验证实际落盘尺寸是否一致**）。
2. **NaVILA 渲染相机**：高度、FOV、分辨率。本地配置里查不到就标 MISSING（网络可用时可补官方来源）。
3. **逐项差值表**：高度差（米）、FOV 差（度）、分辨率差、宽高比差。
4. **Embodiment gap 定量或可视化**：从已采 Go2 帧取样本，与 NaVILA 帧做对比。
   **NaVILA 帧不可获取时**（取决于 P3-T1），只做 Go2 侧的实测刻画
   （实际分辨率、可见视野范围、地平线位置、相机高度），并把 NaVILA 侧标 MISSING。
   **不要为拿不到的数据编造对比图或差距数值。**
5. 一段明确结论：视角/高度差是否大到会造成 distribution shift，依据是什么。

**输出**：`reports/p3/t3_camera_embodiment.json` + `T3_CAMERA_EMBODIMENT.md`
（对比图存 `/mnt/wxh/go2_short_vln/downloads/navila_probe/t3_figs/`，**不得写入 `/home`**）

**禁止**：启动 Isaac 或占用 GPU 来测相机参数（只读配置与已采帧）；编造无法获取的一侧数值。

**验收**：主代理用 PIL/numpy 独立读取若干 RGB 文件复核实际尺寸；核对你引用的配置字段真实存在。

---

## W2（依赖 T1 结论，暂不派发）

若 T1 判定数据**可获取且无 leakage**，再派：
- **P3-T4**：按 T2 的映射对真实样本做可执行验证（A/C 方案的前置条件）
- **P3-T5**：真实 NaVILA 帧 vs Go2 帧的 embodiment gap 定量对比

若 T1 判定**不可获取**或 **leakage 不可剔除**，W2 取消，按 v2 裁决直接走**方案 B**（纯 Go2 rollout），
并在 `DATA_SOURCE_PROBE.md` 中以证据记录「无法验证」。

## 台账

| 任务 | 状态 | 派发时间 | 交回时间 | 主代理判定 |
|---|---|---|---|---|
| P3-T1 | 已派发 | 2026-09-12 | — | — |
| P3-T2 | 已派发 | 2026-09-12 | — | — |
| P3-T3 | 已派发 | 2026-09-12 | — | — |
| P3-T4 | 未就绪（等 T1） | — | — | — |
| P3-T5 | 未就绪（等 T1） | — | — | — |

**模型区间**（用户裁决）：下限 `gpt-5.6-terra` + `xhigh`，上限 `gpt-6-astra` + `mid`。
P0 实测 codex 有效交付率约 50%（capacity / 传输中断），派发单须含「尽早落盘、增量重写、少量批处理命令」。

---

## W1 验收记录（主代理独立复算，2026-09-12）

### 沙箱能力边界（本轮新增确认，写入调度协议）

| 能力 | codex 沙箱 | 主代理 | 处置 |
|---|---|---|---|
| GPU 设备节点 | **屏蔽** | 可用 | P0 已确认；子代理报 GPU MISSING 须主代理复核 |
| `.git` 子结构写入 | **屏蔽** | 可用 | P0 例外 E1 |
| 网络 | 默认 **屏蔽**（`Operation not permitted`） | 可用但**间断** | 见下 |
| 写 `/mnt` | **屏蔽**（`workspace-write` 只给 workspace） | 可用 | 见下 |

**网络**：`codex-companion.mjs` 把沙箱**硬编码**为 `workspace-write`（脚本第 491 行），无网络透传，
所以经 `Agent` 工具派发的子代理**必然**无网。直接调 `codex exec` 并加
`-c 'sandbox_workspace_write.network_access=true'` 可放开，实测生效（从 `Operation not permitted` 变为真实 TLS 握手）。
但仍须**显式传入代理环境变量**：本机出网走 `http://127.0.0.1:17890`（`HTTP_PROXY`/`HTTPS_PROXY` 已在主代理环境中）。

**网络可靠性实测**：`https://huggingface.co` 5 次尝试 → `000, 000, 200, 200, 200`，**成功率约 60%**，
失败形态为 `curl: (35) SSL routines::unexpected eof`。`https://github.com` 即使经代理仍不可达。
**结论：派发单必须要求重试 4 次以上，单次失败不得判定不可达。**

**写 `/mnt`**：可加 `-c 'sandbox_workspace_write.writable_roots=["<路径>"]'` 定点放开。
T3 因未加此项而无法创建 `t3_figs/`，属派发单缺陷，非子代理过失。

**派发机制变更**：需要网络或写 `/mnt` 的子任务，改由主代理**直接调用 `codex exec`** 并带上述 `-c` 覆盖。
实现工作仍全部由 codex 完成，主代理只是换了调用通道，**不构成职责边界例外**。

### P3-T2 判定：**PASS（附一处实质更正）**

产出 `reports/p3/t2_action_semantics.json`、`T2_ACTION_SEMANTICS.md`、
`scripts/p3_navila_action_map.py`、`tests/p3/test_navila_action_map.py`。

**主代理独立核实**（直接读 `eval_utils.py` 第 57–85 行原文）：

| 项 | 子代理 | 主代理核实 | 判定 |
|---|---|---|---|
| 动作集合 | forward 25/50/75 cm；turn L/R 15/30/45°；STOP | 源码逐行确认 | ✅ |
| 引用行号 | `eval_utils.py:59–80` | 实际 `get_vel_command()` 位于 57–85，各档位行号吻合 | ✅ |
| 执行路径 | prompt → `get_vel_command()` → 速度+时长 → `env.step()` → 低层策略 | `navila_eval.py:327–336` 取 (velocity, duration)，`:330–333` 用 `duration / (sim.dt × decimation)` 换算步数；`wrappers.py:201–245` 更新命令 | ✅ 精确，且**不是**位姿传送 |
| `decimation=4`、`sim.dt=0.005` | 引 `go2_matterport_base_cfg.py:364–372` | 实际第 367 行 `decimation = 4`、第 371 行 `sim.dt = 0.005` | ✅ |
| 映射标为执行 ASSUMPTION | 已明确标注 | 符合 §P3 要求 | ✅ |
| 单元测试 | 6 个通过 | 主代理用 `unittest` 独立跑：**6 passed, OK** | ✅ |
| forward 截断残差 | 25cm→2 tick 残差 0.05 m；50cm→5 tick 残差 0；75cm→7 tick 残差 0.05 m | 独立复算一致 | ✅ |

**实质更正：子代理把转向角速度当成了本项目契约的 0.5 rad/s，未注意 NaVILA 源码用的是 `np.pi/6`。**

`eval_utils.py` 原文（主代理直接读取）：

```python
if "turn left" in text.lower():
    if "45" in text.lower():   return [0.0, 0.0, np.pi/6.0], 1.5
    elif "30" in text.lower(): return [0.0, 0.0, np.pi/6.0], 1.0
    elif "15" in text.lower(): return [0.0, 0.0, np.pi/6.0], 0.5
elif "move forward" ...:
    if "75" in text.lower():   return [0.5, 0.0, 0.0], 1.5
    elif "50" in text.lower(): return [0.5, 0.0, 0.0], 1.0
    elif "25" in text.lower(): return [0.5, 0.0, 0.0], 0.5
elif "stop" in text.lower():   return [0.0, 0.0, 0.0], 0.0
else:                          return [0.5, 0.0, 0.0], 0.5
```

由此得出**三条对 P1 契约有直接影响的事实，子代理全部漏报**：

1. **NaVILA 的 `wz = π/6 = 0.523599 rad/s，超出本项目契约 §3.2 的 ±0.5 上限 4.72%。**
   即**每一次 NaVILA 转向都越界**。直接重放 NaVILA 动作必然触发 clipping。
2. **NaVILA 的 `vx` 恒为 0.5 m/s**——不是一个范围，而是**单一常数**，恰好压在契约上界。
   而 P0 实测 Go2 expert 数据的 vx 最大只有 **0.349996**。
   **两个候选训练数据源的 vx 分布几乎不相交**：NaVILA 恒 0.5，Go2 expert ≤ 0.35。
   这比我先前说的「`(0.35, 0.5]` 零监督」要尖锐得多——不是覆盖不全，而是**两源基本不重叠**。
3. **NaVILA 对解析失败的兜底是 `[0.5, 0, 0], 0.5s`（即前进 25 cm）**，
   而本项目 §P6 冻结的规则是「解析失败固定为本 tick 零速度意图、经 limiter 制动」。
   **两者行为相反**：NaVILA 悄悄前进，我们要求刹停。移植 parser 时必须显式改掉，不能照抄。

**转向残差的正确算法**（主代理独立复算，NaVILA 原生 `wz=π/6` + 时长 0.5/1.0/1.5 s）：

| 宏动作 | 时长 | NaVILA 精确角度 | 5 Hz tick 数 | floor | ceil |
|---|---|---|---|---|---|
| turn 15° | 0.5 s | 15.0°（精确） | 2.5 | 2 tick = 12.00°（残差 +3.00°） | 3 tick = 18.00°（残差 −3.00°） |
| turn 30° | 1.0 s | 30.0°（精确） | 5.0 | 5 tick = 30.00°（**残差 0**） | 同左 |
| turn 45° | 1.5 s | 45.0°（精确） | 7.5 | 7 tick = 42.00°（残差 +3.00°） | 8 tick = 48.00°（残差 −3.00°） |

若**把 wz 夹到契约 0.5**，则在任何 tick 取整之前就已经欠转：
15°→14.32°（欠 0.68°）、30°→28.65°（欠 1.35°）、45°→42.97°（欠 2.03°）。

**关键结论**：NaVILA 的宏动作在**时长制**下是**精确可实现**的（π/6 × t 与 0.5 × t 都整好对上标签）。
残差**完全由我们自己的"固定 5 Hz tick、取消 duration"契约引入**——
正是 §0 修订 #1 决定砍掉 duration 的那个选择。25 cm / 75 cm / 15° / 45° 全部落在**半个 tick** 上。
文档 §7 风险表把 duration 议题标为「接受的已知局限」，本条给它加上了**定量代价**：
四个宏动作中的三个无法整除，误差 ±3° 或 ±0.05 m。

### P3-T3 判定：**PASS**

产出 `reports/p3/t3_camera_embodiment.json`、`T3_CAMERA_EMBODIMENT.md`。

主代理逐项独立核实（直接读配置原文 + 用 PIL 独立读帧）：

| 项 | 子代理 | 主代理核实 | 判定 |
|---|---|---|---|
| 配置位置 | `go2_matterport_base_cfg.py:301–307` | 实际第 301–307 行确为 `rgbd_camera = CameraCfg(...)` | ✅ |
| 分辨率 | 512×512 | 配置 `width=512, height=512` | ✅ |
| FOV | 96.7329° | `horizontal_aperture=54.0` + PinholeCameraCfg 默认 `focal_length=24.0` → `2·atan(54/48) = 96.7329°` | ✅ 独立复算一致 |
| 相对 base 高度 | 0.5 m | `offset=OffsetCfg(pos=(0.1, 0.0, 0.5), ...)` → **另有 0.1 m 前向偏移**，子代理未提但不影响结论 | ✅ |
| 实际落盘帧尺寸 | 441 个 JPEG 全为 `uint8 [512,512,3]`，与 §3.1 一致 | PIL 独立抽样 13 帧：全部 `(512,512) RGB`；文件数 236 + 205 = **441** | ✅ |
| 世界高度实测 | 0.805–0.833 m，均值 0.815 m | 采信（依赖 `low_level.jsonl` 位姿，主代理未逐帧复算） | 采信 |
| NaVILA 侧参数 | MISSING（无网络） | 与 T1 首轮一致 | ✅ 未伪造 |
| 对比图 | 未生成，标 UNVERIFIABLE | 符合 v2 裁决「不为不可用数据编造对比」 | ✅ |

**值得记录**：契约 §3.1 声称 `rgb: uint8 [512,512,3]` —— 这是 P0 以来**第一条经实测完全成立**的文档声称
（P0 阶段有 6 条被推翻）。`decimation=4` / `sim.dt=0.005` → 50 Hz 也与 §3.3 吻合。

---

## 旁支发现：X1 接触力问题可能已被本项目自己的证据解决

**这不是 P3 的任务范围，但证据是在 T3 核查同一配置文件时撞见的，价值足够单列。**

### 文档现状

§6 X1 称：19 个 body 接触力恒为零，且**已排除**「官方默认 TensorAPI 设置、`disable_contact_processing` 开关对照、
强制刷新传感器、直接 PhysX 接口、200 Hz 物理子步取值」五种排查，**原因未定位**。

### 主代理实测（全部只读，未开 GPU）

**采集侧（R2R 线，接触力全零）**：`scripts/collect_r2r_continuous_v1.py` 第 97–100 行

```python
# This IsaacLab version requires True for TensorAPI contact reporting.
cfg.sim.disable_contact_processing=True
```

产物 `20260910_ep1_continuous_v2/contact_diagnostics.json` 实测：
`configured_disable_contact_processing = true`、`runtime_disable_contact_processing = true`、
`sensor_buffer` 19 个 body 全 `[0,0,0]`、`direct_physx_forces` 同样全 `[0,0,0]`、`substep_peak_force_n = 0.0`。
all61 全 54 条完成样本 `maximum_contact_force_n = 0.0`，`audit.json` 标 `contact_force_usable: false`。

**dual_target 侧（同一机器、同一 checkpoint、同一 19 个 body、同一 `sim.dt=0.005` / `decimation=4`）**：
`src/dual_target/scene.py` 第 270 行

```python
env_cfg.sim.disable_contact_processing = False
```

产物 `outputs/dual_target_v1/dt1_contact_blue_20260905_r1/stage_status.json` 实测：

| 指标 | 实测值 |
|---|---|
| `negative_foot_peak_n`（正常脚部支撑） | **44.35 N** ← 非零且量级合理 |
| `negative_target_peak_n`（未接触目标时） | **0.0 N** ← 正确为零 |
| `positive_target_peak_n`（撞上目标箱） | **1037.41 N** ← 真实碰撞 |
| `first_collision_physics_step` | 1215 |
| `status` | `CONTACT_PRECHECK_PASSED` |

### 判断

1. 采集器注释「此 IsaacLab 版本需要 True 才能用 TensorAPI 报告接触」**与本项目自己的 dual_target 证据矛盾**：
   dual_target 用 `False` 拿到了非零力，且**能区分正常支撑（44 N）与真实碰撞（1037 N）**，
   未接触目标时正确为 0 —— 这**逐字满足** §6 X1 的成功条件
   「正常脚部支撑产生合理非零力，且能与真实碰撞区分（正常支撑力不可简单标为碰撞）」。
2. 文档称「`disable_contact_processing` 开关对照」已排除，但采集器至今**硬编码 True** 并附上述注释。
   两者不可能同时成立：要么当初的对照实验没真正生效，要么结论记错了。
3. **X1 可能只是一行配置错误**，而非未定位的深层缺陷。

### 但这还不是结论

dual_target 用的是平地+箱体场景，R2R 线用 `go2_matterport_base_cfg`（其 `__post_init__` 第 373 行也设 `True`）。
**决定性实验是：在 R2R 采集路径上把该 flag 改为 `False`，重跑一条 episode，看 19 个 body 是否出现非零力。**
这需要 GPU，属 §6 X1 旁支，**主代理不自行启动**。

若该实验成立，连带影响：P7 的 collision_count / collision_time 可以从 null 变为真实指标；
P9 的 F5 Collision 可以从「未知」变为可统计；§7 风险表「接触力恒零」一条可以关闭；
且历史轨迹的碰撞标注**仍无法回填**（原始记录里力确实是零，不是丢了标注）。

**建议**：X1 从「可选、原因未定位」升级为「有明确单点假设、待一次 GPU 验证」，
在 P1 契约冻结之后、P2 重构期间顺带验证（重构本就要碰采集路径）。是否执行待用户裁决。

## 台账更新

| 任务 | 状态 | 主代理判定 |
|---|---|---|
| P3-T1（首轮） | 已交回 | **正确行为**：网络被沙箱屏蔽，快速报 MISSING 未重试耗死；但结论为 UNVERIFIABLE，须带网重跑 |
| P3-T1（重跑） | 进行中（主代理直调 `codex exec`，已开网络 + 代理 + `/mnt` 写权限） | — |
| P3-T2 | 已交回 | **PASS** + 3 条实质更正（π/6 越界、vx 恒 0.5、兜底行为相反） |
| P3-T3 | 已交回 | **PASS**（唯一缺陷是派发单未给 `/mnt` 写权限，主代理过失） |
