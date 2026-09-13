# 派发单 P3-T1E — NaVILA R2R annotations 全量 leakage join + 全量动作分布

生成：2026-09-13，主代理编排。上游裁决：用户于 2026-09-13 批准重验（先前接受 8.3% 部分覆盖的决定作废，
理由是 `R2R_annotations.json` 已全量落盘，100% join 的成本降为零下载 / 纯 CPU / 几分钟）。

本任务取代 `reports/p3/t1c_leakage.json` 的 `status: UNVERIFIABLE`，并补齐其 `missing` 列出的全部 6 项。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P3-T1E** |
| 资源 | **CPU only。不需要 GPU，不需要网络。** 不得走 `gpu_wait`，不得占用 GPU |
| 并发 | 可与其他 CPU 任务并发；**但不得干扰正在运行的下载进程 pid 348921** |

---

## 1. 目标

用**完整**的 NaVILA R2R annotations 文件，机械回答两个问题并留下可复算证据：

1. **NaVILA 的 R2R 训练数据里有没有 R2R `val_unseen` 的 episode？**
   （等价于：有没有泄漏进本项目的 11 个评测 scene。前置事实：评测 11 scenes 与 R2R `val_unseen` 的 scene 集合**相等**，见 `reports/split_disjoint_check.json`。）
2. **全量动作分布是什么？** 含 majority-class 基线——该数字将作为 phase 1 的离线先验对照门槛写入里程碑文档。

---

## 2. 输入（确切路径 + SHA-256，逐条核对后再开工）

| 用途 | 路径 | SHA-256 | 字节 |
|---|---|---|---|
| NaVILA R2R 标注**完整版** | `/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_annotations.json` | `3587c020cc03807cfc03c454bcbb37a3d409139e125e1ea047fa1fbcf8c4cc1a` | 317267756 |
| R2R train split | `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz` | `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34` | — |
| R2R val_seen split | `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/val_seen/val_seen.json.gz` | `7fc94841ebbd2eac0d398e020a2f638426948beaf4a561f6ee310dd67cddce55` | — |
| R2R val_unseen split | `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz` | `d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8` | — |
| R2R test split | `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/test/test.json.gz` | `69ed5962fac658d34be5739c1f8f15805cf5ce9bfa28f64a338da338244578b7` | — |
| 评测集（eval11 / 1077 ep） | `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz` | `ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec` | — |

### 陷阱：同名的不完整副本，**禁止使用**

`/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/annotations.json` 只有 **184,991,744** 字节（截断的第二次下载尝试）。
开工前必须先 `stat -c%s` + `sha256sum` 核对，只接受上表那份 317,267,756 字节 / `3587c020…` 的文件。
若哈希不符，**停止并交回诊断，不得改用副本**。

---

## 3. 方法（必须照此实现，偏离须在报告中说明理由）

### 3.1 记录格式（已由主代理实测确认）

```json
{"video_id":"914-23",
 "q":"Turn and walk away from the bathroom into the bedroom area. ...",
 "a":"The next action is move forward 75 cm.",
 "frames":["914/frame_0.jpg", ..., "914/frame_48.jpg"]}
```

- `video_id` 形如 `<video>-<step>`，`<step>` 为该 video 的第几个决策点。
- **无 `scene_id` 字段**，scene 身份必须 join 恢复。
- **不得按 `video_id` / 数字 id join**：数字 id 跨 split 碰撞（P3 实测 9471 个前缀在 train 命中 9471、在 val_unseen 也命中 1588）。
- 正确键是 **instruction 原文归一化**：`' '.join(s.split()).strip().lower()`，对 NaVILA 的 `q` 与 R2R split 的
  `episodes[*].instruction.instruction_text` 施加**同一个**归一化函数。

### 3.2 内存约束

317 MB 单行 JSON 数组。**不得 `json.load` 整文件后再处理**（会吃 2–3 GB 常驻）。
用 `json.JSONDecoder().raw_decode` 增量解析、或分块流式读取，保持常驻内存 < 1.5 GB。
解析必须覆盖**全部**记录，不得只取前 N MB —— 这正是本任务存在的原因。

### 3.3 必须产出的数字

**A. leakage**
1. 全量记录数、不同归一化指令数。
2. 每个 split（train / val_seen / val_unseen / test）的命中指令数与命中记录数。
3. **`val_unseen − train` 的差集大小**（决定性数字：NaVILA 里独有于 val_unseen 的指令条数）。
4. 匹配到 train 的指令所蕴含的 scene 集合（去重、排序、计数）。
5. 该 scene 集合与 eval11 的**交集**（必须打印实际集合，不能只打印长度）。
6. **无法解释的记录数**：既不命中任何 split 的归一化指令，连同其记录数与 3 条样例。

**B. 歧义条处置（用户已裁决的规则，照此执行，不得自行改判）**
- 凡一条归一化指令**同时**命中 train 与 val_unseen（或 test），无论其真实归属，**一律从训练集剔除**。
- 产出完整剔除清单（归一化指令、命中的 split 集合、对应 `video_id` 列表、受影响记录数）。

**C. 全量动作分布**
7. `a` 字段的完整词表与每条计数（主代理在 6712 条抽样上见到**恰好 10 条**字符串，全量须验证是否仍为 10）。
8. **majority-class 占比**（抽样值 29.74%，须给出全量值）。
9. stop 动作占比（抽样值 9.5%）。
10. 每 video 的决策点数分布（min / p25 / p50 / p75 / p90 / max）。
11. 每记录 `frames` 数分布（同上分位），以及 `frames` 是否恒为连续 `0..N-1`（抽样 6712/6712 成立，须全量验证）。
12. `n_frames` 在同一 video 内是否随 `<step>` 单调不减（抽样 1385 videos / 0 反例，须全量验证）。

---

## 4. 输出（确切路径；机器可读产物必须是 JSON）

| 路径 | 内容 |
|---|---|
| `/home/wxh/go2_short_vln/reports/p3/t1e_leakage_full.json` | §3.3 A + B 的全部数字，含 `inputs` 段落回写实测 SHA-256 与字节数 |
| `/home/wxh/go2_short_vln/reports/p3/t1e_action_distribution.json` | §3.3 C 的全部数字 |
| `/home/wxh/go2_short_vln/reports/p3/t1e_excluded_instructions.json` | §3.3 B 的完整剔除清单 |
| `/home/wxh/go2_short_vln/reports/p3/T1E_LEAKAGE_FULL.md` | 人读摘要：结论一句话 + 上述数字的表格 + 方法学局限 |
| `/home/wxh/go2_short_vln/scripts/p3_navila_leakage_join.py` | 可复算脚本。**必须无参数即可重跑**并产出上述 JSON；路径写死为上表输入 |

每份 JSON 必含 `task_id: "P3-T1E"`、`generated_at`、`generator_command`、`inputs`（路径 + 实测 sha256 + 字节数）。
产物体积须小：剔除清单若超过 5 MB，改为只存归一化指令与 split 集合，`video_id` 列表另存为
`reports/p3/t1e_excluded_video_ids.json.gz`（并在 JSON 里注明）。

---

## 5. 验收标准（主代理将如何独立复算，逐条可执行）

1. 主代理用**自己的**流式解析器独立重算「全量记录数」与「不同归一化指令数」，须与 JSON 一致。
2. 主代理独立重算 `a` 字段词表大小与 **majority-class 占比**，须与 `t1e_action_distribution.json` 一致到小数点后两位。
3. 主代理从 `val_unseen.json.gz` 随机抽 50 条指令，独立查它们在 NaVILA 全量里的命中情况，须与报告的 `val_unseen − train` 结论相容。
4. 主代理核对报告给出的 eval11 scene 集合与 `reports/split_disjoint_check.json` 中已冻结的 11 scenes **逐名一致**。
5. 主代理无参数重跑 `scripts/p3_navila_leakage_join.py`，产出的 JSON 关键字段须与提交版一致（确定性）。
6. 三份 JSON 的 `inputs.sha256` 须与本派发单 §2 表格逐字一致。

**子代理返回的 `passed` 字段、自测通过数、自我结论均不构成 PASS 依据。**

---

## 6. 禁止事项

1. 不得改动本任务 §4 输出以外的任何文件。**特别不得修改** `reports/DATA_SOURCE_PROBE.md`、
   `CODEX_VLN_PLATFORM_MILESTONES.md`、`configs/`、`.gitignore` —— 文档收口由主代理做。
2. 不得删除或覆盖任何既有产物，包括 `reports/p3/t1c_leakage.json`（保留为历史证据）。
3. 不得把大文件写入 `/home`。中间产物一律放 `/mnt/wxh/go2_short_vln/tmp/p3_t1e/`，任务结束自行清理。
4. **不得触碰正在运行的下载**：进程 pid 348921（`navila_dl_mirror.sh`）及其 curl 子进程、
   文件 `/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_train.tar.gz`、日志 `download_mirror.log`。
   不得 kill、不得读写该 tar.gz、不得改动 `navila_dl_mirror.sh`。
5. 不得使用 §2 点明的 184,991,744 字节不完整副本。
6. 不得联网。本任务全部输入已在本地。
7. 不得占用 GPU。
8. **证据缺失一律标记 `MISSING`，不得自行追认 PASS**；无法完成的项交回诊断，不得更换方案。
9. 不得 `git add` / `git commit` / `git tag`（提交由主代理负责）。

---

## 7. 失败与返工

单项失败时交回**诊断与证据**，不自行更换方案。同一失败原因经两次有证据的修复尝试仍存在，升级给主代理。
按 P0 实况校准：codex 派发有效交付率约 50%，故本派发单要求**尽早落盘、增量重写、少量批处理命令**。

---

## 8. 验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| 1 | 2026-09-13 | **PASS** | 主代理独立复算 15 项全部吻合（双算记录数、自实现统计、确定性重跑 4/4 规范化哈希 MATCH、输入哈希 6/6、eval11 stems 11/11、val_unseen 50 抽样相容）。逐项台账见 `CODEX_VLN_PLATFORM_MILESTONES.md` §v3.8 |

**验收结论**：`leakage = NO-LEAKAGE，覆盖率 100%`，取代 P3 §2 的 8.3% 部分覆盖与 `t1c_leakage.json` 的 `UNVERIFIABLE`。

**本派发单的缺口（主代理自评）**：§3.3 的 12 项未要求核查 `video_id` 是否唯一。
主代理在验收中发现 `video_id` **不唯一**（353,894 记录 / 288,594 不同 id / 18.45% 重复），
且重复是 NaVILA 对 STOP ×3、30°/45° 转向 ×2 的**有意类别过采样**。
这会改变离线先验基线（过采样态 majority-class 29.96%，去重态 **36.74%**），
并浮现一个 loader 决策。**属派发单设计缺口，不是子代理失职。** 详见 §v3.9-A。
后续涉及数据集统计的派发单须加入「主键唯一性」与「是否存在重采样」两项。
