# 派发单 P4-T2 — NaVILA 逐记录索引 + scene 分层留出集

授权：用户 2026-09-13 `CONTINUE P4′`。波次：W1（与 P4-T1、P4-T3 并发）。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P4-T2** |
| 资源 | **CPU only。零 GPU，零网络。** |
| 并发 | 与 T1 / T3 并发。**不读解包后的帧文件**（那是 T1 的产出），只读 annotations 与 R2R split |

## 1. 目标

产出 phase 1 训练/评测要长期依赖的**逐记录索引**，以及**按 scene 分层的留出集**。
这是数据契约，不是一次性报告——后续 loader、离线指标、先验对照都建在它上面。

## 2. 输入

| 用途 | 路径 | SHA-256 |
|---|---|---|
| NaVILA R2R 标注**完整版** | `/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_annotations.json` | `3587c020cc03807cfc03c454bcbb37a3d409139e125e1ea047fa1fbcf8c4cc1a` |
| R2R train split | `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz` | `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34` |
| 既有全量 join 产物（参考，不得覆盖） | `/home/wxh/go2_short_vln/reports/p3/t1e_leakage_full.json` 等 | — |

**禁止使用** `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/annotations.json`（184,991,744 字节截断副本；
P4-T1 会把它改名隔离，若已改名则不必理会）。开工前 `sha256sum` 核对上表两项。

## 3. 已确立的事实（主代理实测，作为实现依据与自检基准）

| 事实 | 值 |
|---|---|
| 记录数 | **353,894** |
| 不同 `video_id` | **288,594**（`video_id` **不是唯一键**） |
| 重数分布 | `{1: 234113, 2: 43662, 3: 10819}` |
| ×3 的 10,819 条 | 全部是 STOP，且全部是各 video 的最后一步 |
| ×2 的 43,662 条 | 全部是 30°/45° 转向（L45 15755 + R45 13015 + L30 7674 + R30 7218） |
| 不同 video 数 | **10,819** |
| **`video_id` 前缀 ↔ R2R train `episode_id`** | **完美双射 10,819/10,819，指令逐字吻合，零例外** |
| 动作词表 | 恰好 **10** 条字符串 |
| majority-class | **29.96%**（`The next action is move forward 75 cm.`，106,026 条） |
| stop 占比 | **9.17%**（32,457 条） |
| `frames` | 恒为连续 `0..N-1`；帧率与决策率**解耦**（`n_frames - step` 非常数） |

**过采样一律保留**（用户 2026-09-13「对齐原生」，见里程碑文档 §v3.1 #17）：
索引须覆盖全部 **353,894** 条记录，**不得去重**到 288,594。但**必须为每条记录标注其重数**
（`multiplicity` 1/2/3）与是否为过采样副本，以便日后需要时能无损去重。

## 4. 任务

### 4.1 逐记录索引

对全部 353,894 条记录，每条产出：

- `video_id`（原样）、`video`（前缀）、`step`（后缀整数）
- **`scene_id`**（由 `video` == train `episode_id` 的双射查得；须同时校验该 episode 的指令与本记录 `q` 逐字吻合，
  **不吻合即报错退出**，不得静默跳过）
- `instruction_raw`、`instruction_normalized`（`' '.join(s.split()).strip().lower()`）
- `action_text`（原样）、`action_id`（0–9，映射表须写死并在输出中声明）
- `n_frames`、`frames_first`、`frames_last`（**不要**把 601,125 条帧路径全量塞进索引；
  帧路径可由 `video` + 索引区间确定性重建，须在输出中写明重建规则）
- `multiplicity`（1/2/3）、`is_oversampled_copy`（bool）

内存约束：**不得 `json.load` 整个 317 MB 文件**，用 `json.JSONDecoder().raw_decode` 增量解析，常驻 < 1.5 GB。

### 4.2 scene 分层留出集

phase 1 需要在 NaVILA 数据内部切出留出集算离线指标（动作准确率、先验对照）。
**必须按 scene 切，不得按记录随机切**——随机切会让同一 scene 同时出现在两侧，造成场景泄漏。

**已知约束：scene 分布极不均衡。** 每 scene 的 video 数最多 **279**
（`8WUmhLawc2A` / `ur6pFq6Qu1A` / `r47D5H71a5s` / `JeFG25nYj2p`），最少 **3**（`gZ6f7yhEvPG`），
其上为 9（`YmJkqBEsHnH`）/ 15（`HxpKQynjfin`）/ 18（`XcA2TqTSSAj`）/ 21（`GdvgFV5R1Z5`）。

要求：

1. 固定 seed，写入输出；切分**可复现**。
2. 目标留出比例 **10%**（按 **video** 计，不是按记录计）。
3. **稀疏 scene 显式处置**：video 数 < 25 的 scene 必须在报告中逐个列出并说明归属决定
   （建议整体归入训练侧以免留出集被单一稀疏 scene 主导；但**须显式说明并给出理由**，不得默默按比例切）。
4. 输出两侧的 scene 集合，断言**交集为空**。
5. 输出两侧的记录数、video 数、scene 数，以及**两侧各自的动作分布与 majority-class 占比**
   （留出集的先验基线与训练集可能不同，报告须同时给出）。

## 5. 输出

| 路径 | 内容 | 备注 |
|---|---|---|
| `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.jsonl` | 逐记录索引，一行一条 JSON | 大产物放 `/mnt`，**不进 git** |
| `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.sha256` | 上述文件的 SHA-256 | — |
| `/home/wxh/go2_short_vln/reports/p4/t2_index_summary.json` | 计数、动作分布、scene 分布、重数分布、帧路径重建规则、`action_id` 映射表 | 小，入库 |
| `/home/wxh/go2_short_vln/reports/p4/t2_scene_split.json` | 留出集清单：seed、两侧 scene 列表、video 列表、计数、两侧动作分布、稀疏 scene 处置说明 | 小，入库 |
| `/home/wxh/go2_short_vln/reports/p4/T2_INDEX_SPLIT.md` | 人读摘要 | 入库 |
| `/home/wxh/go2_short_vln/scripts/p4_build_navila_index.py` | 可复算脚本，**无参数重跑且确定性** | 入库 |

注意 `.gitignore` 排除 `*.jsonl.gz`、`*.npz` 等；`reports/p4/` 下的四个产物须是小体积可入库文件。

## 6. 验收标准（主代理如何独立复算）

1. 主代理独立重算索引行数，须为 **353,894**；`video` 去重后须为 **10,819**。
2. 主代理独立重建 `video` → `scene_id` 映射（由 train split 的 `episode_id`），与索引**逐条比对**，零不符。
3. 主代理独立重算全量与两侧的动作分布及 majority-class，须与 `t2_index_summary.json` / `t2_scene_split.json` 一致到小数点后两位。
4. 主代理断言留出集与训练集的 **scene 交集为空**、**video 交集为空**。
5. 主代理核对稀疏 scene（3/9/15/18/21）在 `t2_scene_split.json` 中**逐个有显式归属与理由**。
6. 主代理无参数重跑 `scripts/p4_build_navila_index.py`，产物关键字段一致（确定性）。
7. 主代理按输出声明的「帧路径重建规则」随机重建 20 条记录的帧路径，与标注原文逐条比对。

## 7. 禁止事项

1. 不得去重记录（过采样保留，见 §3）。
2. 不得改动 `reports/p3/` 下任何既有产物；不得动 `CODEX_VLN_PLATFORM_MILESTONES.md`、`configs/`、`.gitignore`。
3. 不得触碰 P4-T1 的输出（解包目录、`reports/p4/t1_*`、`scripts/p4_extract_navila.*`）
   与 P4-T3 的输出（`actions/`、`tests/`、`reports/p4/t3_*`）。
4. 不得读取解包后的帧文件——T1 可能正在并发写入。帧的存在性由 T1 负责。
5. 不得 `git add` / `git commit` / `git tag`；不得联网；不得占 GPU。
6. 大产物不得写入 `/home`。
7. `scene_id` 查不到或指令不吻合时**报错退出并交诊断**，不得静默跳过或猜测。

## 8. 验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| — | — | 待派发 | — |

---

## 第 2 轮重派（2026-09-13）— codex 沙箱写权限已放行

**第 1 轮结果：BLOCKED，零产出/部分产出。** 子代理报
`OSError: [Errno 30] Read-only file system ... (mount is /dev/sdb1 on /mnt with ro flag)`。

**主代理独立核查推翻该归因**：`/mnt` 挂载标志实为 **`rw,relatime`**，主代理向同一目录写测试文件成功，
且本会话的 19.7 GB 下载就落在 `/mnt`。真实原因是 **codex 默认 `workspace-write` 沙箱**只放行项目根与 `/tmp`，
拒绝时以 `EROFS` 形式呈现，子代理误读为挂载只读。**这是 E1 同类的结构性阻塞，不是子代理失职**；
子代理将各项标记 `MISSING` 并交回诊断、未自行追认 PASS，行为正确。

**已采取的修复（用户 2026-09-13 裁定）**：在 `~/.codex/config.toml` 追加

```toml
[sandbox_workspace_write]
writable_roots = ["/mnt/wxh/go2_short_vln"]
```

作用域仅本项目的 `/mnt` 路径，**非整个 `/mnt`**；`network_access` 保持默认不放行。
备份见 `~/.codex/config.toml.bak-before-mnt-writable-20260913-101526`。TOML 已校验可解析且既有节点无丢失。

**本轮强制第一步**：先做写权限探针——向 `/mnt/wxh/go2_short_vln/.probe_<task>_<pid>` 写入并删除。
**探针失败即立刻中止并交回诊断，不得继续**（避免在 19.7 GB 级任务上空耗）。探针结果须写入产物 JSON。

### 本轮验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| 1 | 2026-09-13 | **PARTIAL** | 计算全部完成且经独立复算吻合；仅 `/mnt` 大索引因 codex 沙箱阻塞未落盘，已暂存 `/tmp` 并如实标 MISSING，行为正确 |
| 2 | 2026-09-13 | **PASS** | 索引落 `/mnt`（哈希与行数吻合）、`/tmp` 暂存已清、6 个稀疏 scene 逐个有归属、重跑逐字节确定性。11 项独立复算全通过，详见 `CODEX_VLN_PLATFORM_MILESTONES.md` §v3.14-A |

**主代理自身更正 2 条**（记录以免重犯）：曾用「不同 `video_id` 的重数分布」误断言索引的**记录级** multiplicity；
曾只列出 5 个稀疏 scene，遗漏 `Pm6F8kyY3z2`（24 videos，卡在 `<25` 阈值内侧）。两处均为主代理错、子代理对。

### T2 本轮的特殊情况：计算已完成，只差落盘

第 1 轮 T2 **计算全部完成且经主代理独立复算吻合**：`record_count` 353,894、`distinct_video_count` 10,819、
`scene_count` 61、majority-class 29.95982% —— 四项与主代理独立确立的事实逐项一致。
`reports/p4/` 下的四个小产物已落盘。

唯一缺失是 `/mnt` 上的大索引，它被**暂存在 `/tmp/p4_t2_navila_index/`**：
`t2_records.jsonl` 222,781,783 字节 / **353,894 行**，`t2_records.sha256` =
`d894855183543d230a4802701a84aa3d430be9ad55b52d6dc5b6827829b7a09f`（主代理已实算核对，自洽）。

**因此本轮 T2 的工作量很小**：把暂存产物移到
`/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/`，移动后重算 SHA-256 并核对与上述值一致，
更新产物 JSON 中的路径与状态，然后**删除 `/tmp` 暂存副本**（根分区仅余 27 GB，97% 已满，不要长期占用）。
若暂存副本已不存在，则按原派发单完整重跑。

**同时请复核并补齐**：`t2_scene_split.json` 的稀疏 scene 逐个归属说明是否完整
（`gZ6f7yhEvPG`=3、`YmJkqBEsHnH`=9、`HxpKQynjfin`=15、`XcA2TqTSSAj`=18、`GdvgFV5R1Z5`=21 个 video）。
