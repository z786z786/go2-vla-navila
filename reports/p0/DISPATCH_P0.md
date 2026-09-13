# P0 派发单台账（主代理编排）

生成：2026-09-12。依据 `CODEX_VLN_PLATFORM_MILESTONES.md`（SHA-256 见下）§5「调度与并发执行协议」与「P0 波次拆解」。

本文件由**主代理**维护。codex 子代理只读本文件中属于自己的那一节，不得修改本文件。

## 全局约束（每张派发单默认继承）

- 源码 / session 工作目录：`/home/wxh/go2_short_vln`（根分区仅余约 25 GB）。
- 数据 / 模型 / 环境 / 缓存 / 大产物：`/mnt/wxh/go2_short_vln`（余约 1.8 TB）。`outputs` 已是指向后者的符号链接。
- **禁止**把任何大文件（>10 MB）写入 `/home`。中间产物写 `/mnt/wxh/go2_short_vln/tmp/`。
- **禁止**删除、移动或改写任何既有产物；本阶段是「冻结」，不是清理。
- **禁止**改动本任务「输出」字段以外的文件。
- **禁止**在证据缺失时自行追认 PASS；缺失项一律写 `"status": "MISSING"` 并说明为什么缺。
- 所有机器可读产物为 JSON（UTF-8，`ensure_ascii=false`，2 空格缩进），且必须含顶层字段：
  `task_id`、`generated_at`（ISO8601 +08:00）、`generator_command`、`inputs`（路径→SHA-256）、`findings`、`missing`。
- 报告中任何数字必须可由 `generator_command` 复算；不得引用本文档的结论当作证据。
- GPU：P0 全部为 CPU / 只读任务，**不得**启动 Isaac、不得加载模型、不得占用 GPU。

## 输入基线哈希（2026-09-12 实测）

| 路径 | SHA-256 |
|---|---|
| `/home/wxh/go2_short_vln/CODEX_VLN_PLATFORM_MILESTONES.md` | `1b4bd244be5fbec7f83c22a7511bd5ca77eebeb6f300773be459f788b25c5a0e`（本次编辑前基线；编辑后哈希由 P0-T5 重新记录） |
| `scripts/collect_r2r_ablation.py` | `71eea6d5f24d70ac334192f4a632d8781be1fce255ca4efc0e38b585f377f67e` |
| `scripts/audit_r2r_ablation.py` | `d235a679e97efd767cfc81b2f9508322b06ef469dba869f91be2f764dbdb8f63` |
| `scripts/validate_r2r_episode_load.py` | `d0b5c2787cf089113afc85e7726679d09c925a0323ac1f85038ed072e9a1b1ce` |
| `src/dual_target/r2r_path_expert.py` | `7b06e36379d41cc35b531ccaa6910a1a635783c1fea3d343efebcff011e5dfc9` |
| `src/dual_target/gpu_wait.py` | `cd82d3dceb4ed9d4a9234e10ec7104fce6b9f4a7c671e73237f73258842c2f14` |
| `config/dual_target_v1/task_contract.lock.json` | `ae040c5b66c7405197874d9d24408f03736606704c5d3cd278ad5827af5e0e14` |
| `/mnt/.../r2r_ablation/20260910_ep1_v3/summary.json` | `8c2806bb490fbcbd534b05e030b540bb1a4866d3a589407952feca945b2692aa` |
| `/mnt/.../r2r_ablation/20260910_ep1_v3/audit.json` | `2e1706a07761807cdaa029482e41f85e136a3ad85fc13a974ee69c43e6d3fc7a` |
| `/mnt/.../r2r_ablation/20260910_ep1_v3/manifest.json` | `6bd32d965d881233a967330157a07faadbc8329187e22a7f580586406040505c` |
| `/mnt/.../r2r_ablation/20260910_ep1_continuous_v2/summary.json` | `77a157269718adeabfb901652bb6585d208bce17274c088c66bdb938bdf1b180` |
| `/mnt/.../r2r_ablation/20260910_ep1_continuous_v2/audit.json` | `52fed06a4831b7d2f65f0185f1f135b186842bfcda58bab16d343dfb31a95e8a` |

主代理已核查的环境事实（子代理须自行复核，不得直接引用）：
工作目录**尚非 Git 仓库**；PID 779548 不存在；GPU 空闲（38 MiB / 24576 MiB）；
`/` 余 25 GB、`/mnt` 余 1.8 TB；仓库源码（排除 `outputs`）仅 14 MB；
`outputs/r2r_ablation` 共 53 GB，含 61 个 `20260910_all61_*` 目录、82 份 `summary.json`。

---

## W1（五项并发，CPU 只读）

### P0-T1 — 旧队列与历史进程核查

**目标**：用真实命令、启动时间、进程组与产物，判定「旧 1077-episode 官方队列及其它历史采集/评测进程」当前是否存在、是否需要停止、产物是否完整。

**输入**
- `/tmp/official_navila_velocity_0911_full_v1.log`（若不存在，记 MISSING）
- `/mnt/wxh/go2_short_vln/outputs/official_navila_velocity_0911_full_v1/`（含 `progress.json` / `plan.json` / `results.json`，逐一 SHA-256）
- `/mnt/wxh/go2_short_vln/outputs/official_navila_velocity_0911_v1/`
- `src/dual_target/gpu_wait.py`（锁与进程组约定）
- 系统：`ps -eo pid,ppid,pgid,lstart,etime,rss,cmd`、`nvidia-smi`、`ls /proc/*/cmdline`

**输出**：`/home/wxh/go2_short_vln/reports/p0/queue_audit.json`、`/home/wxh/go2_short_vln/reports/p0/QUEUE_AUDIT.md`

**必须回答**
1. PID 779548 是否存在；不存在则明确记 `"action": "no_action_required"`，**不得**把「不存在」写成「已停止」。
2. 当前是否还有本项目自有的采集 / 训练 / 评测进程在跑（按 cmdline 匹配 `collect_r2r`、`zoh_`、`navila`、`isaac`、`train_`、`evaluate`）。列出每一个：PID、PGID、启动时刻、完整 cmdline、是否属于本项目。
3. GPU 当前占用者（`nvidia-smi --query-compute-apps`）；是否有非本项目进程。
4. `official_navila_velocity_0911_full_v1` 的 `progress.json` / `plan.json` / `results.json` 是否齐全、内部计数是否自洽（计划条数 vs 已完成 episode 目录数 vs results 条数），列出差值。
5. 是否有任何进程需要停止；若需要，**只列出停止方案，不执行**，等主代理批准。

**禁止**：停止任何进程；清理任何缓存；写 `/tmp` 以外的临时文件。

**验收（主代理独立复算）**：重跑 `ps` 与 `nvidia-smi` 比对；对三份 JSON 重算 SHA-256 与计数；抽查 3 个 episode 目录的存在性。

---

### P0-T2 — scene 交集机械核对

**目标**：用一条可复算的机械核对，证明 R2R-VLNCE TRAIN 的 scene 集合与 VLN-CE-Isaac 评测集的 scene 集合交集为空。

**输入**
- `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz`
- `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz`
- 如上述文件不足以覆盖 10819/1077 条，自行在 `/mnt/wxh/go2_short_vln/data/r2r_vlnce/` 与 `/mnt/wxh/go2_short_vln/assets/` 下定位真正的来源并记录；**不得**猜测。

**输出**：`/home/wxh/go2_short_vln/reports/split_disjoint_check.json`，附生成脚本 `/home/wxh/go2_short_vln/scripts/p0_split_disjoint_check.py`

**必须包含**
1. 两个源文件的绝对路径与 SHA-256。
2. episode 计数：训练集实际条数、评测集实际条数（期望 10819 / 1077；不符必须如实报告并给出实际值）。
3. 归一化规则**显式写出**（按 scene 文件 stem 归一化，例如 `.../17DRP5sb8fy/17DRP5sb8fy.glb` → `17DRP5sb8fy`），并同时报告归一化前后的集合。
4. 两个 scene 集合（排序后全量列出）、大小（期望 61 / 11）、交集（期望空集）。
5. 断言 `intersection == []`；不成立则 `"assertion": "FAILED"` 并列出交集元素，**不得**修改归一化规则去凑空集。
6. 顺带断言：评测集全部 episode 的 `goals[0].radius == 3.0`，输出唯一值集合与反例列表（这是 P1 的输入，本阶段只取证不下结论）。

**验收**：主代理用自己的一行 python 重算两个集合与交集，并抽查 radius 分布。

---

### P0-T3 — §2 八项基础能力证据核对

**目标**：对里程碑文档 §2 表格的每一项能力，定位真实证据、算哈希、还原产生命令、标记缺失。

**输入**：`CODEX_VLN_PLATFORM_MILESTONES.md` §2 表格（八行）；证据根 `/mnt/wxh/go2_short_vln/outputs/`、`/home/wxh/go2_short_vln/reports/`、`scripts/`、`src/`。

**输出**：`/home/wxh/go2_short_vln/reports/p0/foundation_evidence.json`

**逐项要求**（八项：低层 locomotion、R2R-VLNCE 全量部署、坐标转换、路径 Expert、采集链路+审计、消融数组接口、GPU 共享治理、极小可控场景）
- `capability`、`claimed_evidence_path`（文档所写）、`actual_path`（实测；`/home` 与 `/mnt` 双根映射都要写）、`sha256`（目录给 manifest 文件哈希 + 文件数 + 总字节）、`producing_command`（从日志 / manifest / 脚本 argparse 还原；还原不出写 MISSING）、`status`（`VERIFIED` / `PARTIAL` / `MISSING`）、`notes`。
- 文档声称但远端不存在的三条必须单列并标 MISSING：
  `reports/r2r_ablation/20260910_ep1_v3/`、`reports/r2r_vlnce/audit_report.md`、`reports/OFFICIAL_NAVILA_VELOCITY_0911.md`
  （注意：`/mnt/.../outputs/r2r_ablation/20260910_ep1_v3/` 存在，属于路径映射问题，须写清楚是"路径记错"还是"证据缺失"）。
- 复算并记录 `20260910_ep1_v3` 与 `20260910_ep1_continuous_v2` 的关键数字：高层区间数、有效 RGB 数、停止意图区间数、末端 XY 距离、`ablation_arrays.npz` 的全部 key 与 shape/dtype。
  文档参考值：ep1_v3 = 196 区间 / 236 RGB（数组仅 195 个完整区间）/ 末端 0.2164048 m；continuous_v2 = 170 / 205 / 末端 0.2364878 m；ep1_v3 停止意图 13 个区间。**实测与参考不符时以实测为准并标注差异**。
- 记录 ep1_v3 `command_raw` 的 `wz` 最大绝对值（文档称约 2.536 rad/s）与 `command_applied` 的实际范围，验证「±0.5 只适用于 applied」这一结论。

**禁止**：修改任何 `outputs/` 下文件；重跑采集。

**验收**：主代理抽 3 项独立重算 SHA-256，并用 numpy 重算 npz 的区间数与 wz 极值。

---

### P0-T4 — all61 多场景结果追溯

**目标**：把「34 success / 18 timeout / 6 failed / 2 environment termination」这个文件汇总还原成带任务清单、缺失项与 scene 覆盖的可审计结论。

**输入**：`/mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_all61_*`（61 个目录、82 份 `summary.json`、各自的 `.log` / `_guard.json` / `_gpu.jsonl`）

**输出**：`/home/wxh/go2_short_vln/reports/p0/all61_audit.json`、`/home/wxh/go2_short_vln/reports/p0/ALL61_AUDIT.md`

**必须回答**
1. 原始任务清单是什么（从 `.log` / `_guard.json` / 启动命令还原）；计划多少条、实际产出多少条。
2. 61 个目录 vs 82 份 summary 的差异从何而来（重试？子目录？`_cache`？），逐一归类。
3. 逐 episode 表：`episode_id`、`scene`、`status`、`steps`、`final_distance_to_goal`、`terminated_by`、`summary.json` 的 SHA-256。
4. 真实 scene 覆盖：涉及多少个不同 scene，哪些 scene 缺失或只有失败样本。
5. 状态分布的**实测**计数（不得直接抄 34/18/6/2；不符必须报告实际值）。
6. ep1065 的 warmup 失败：出现几次、日志中的报错原文、失败阶段、是否有修复版本重跑。
7. 明确结论：这批数据**不构成**经独立 QC 的 61-scene 成功率，并说明要变成可用结论还缺什么。

**禁止**：重跑任何 episode；删除 `_cache` 目录。

**验收**：主代理独立统计 `summary.json` 的 status 字段分布与 scene 去重计数。

---

### P0-T5 — 源码哈希清单 + `.gitignore` 草案

**目标**：为 Git 基线做准备：产出「只含源码、配置、必要文档」的完整清单与哈希，以及排除大产物 / 凭据的 `.gitignore`。

**输入**：`/home/wxh/go2_short_vln` 全树（`outputs` 是指向 `/mnt` 的符号链接，**不进 Git**）。

**输出**
- `/home/wxh/go2_short_vln/reports/p0/source_manifest.json`
- `/home/wxh/go2_short_vln/.gitignore`
- `/home/wxh/go2_short_vln/reports/p0/SOURCE_MANIFEST.md`（人读摘要）

**要求**
1. `.gitignore` 必须排除：`outputs`（符号链接）、`__pycache__/`、`*.pyc`、`.DS_Store` 与 macOS 资源叉文件（仓库内大量 `._*` 文件，逐条确认是否为资源叉；是则排除并在报告中列出数量）、`*.npz`、`*.mp4`、`*.pt`/`*.pth`/`*.safetensors`、`*.log`、`*.jsonl.gz`、`data/`（若其下为大数据；先核实体积再决定，结论写进报告）、`third_party/` 下的体积大项、任何 `*.env` / `*token*` / `*credential*` / `*.key`。
2. 逐文件清单：相对路径、字节数、SHA-256、是否入 Git、排除理由。
3. 断言：拟入 Git 的文件总体积 **< 50 MB**，且**零个** >10 MB 的文件。不成立则列出违例文件，**不得**擅自删除。
4. 凭据扫描：对拟入 Git 的全部文本文件 grep 常见密钥模式（`sk-`、`ghp_`、`AKIA`、`BEGIN PRIVATE KEY`、`password=`、`token=`），命中项列出文件与行号（**不要把密钥原文写进报告**，只写位置与匹配模式）。
5. 重新记录 `CODEX_VLN_PLATFORM_MILESTONES.md` 的当前 SHA-256（主代理已于本次编辑改动过该文件）。

**禁止**：执行 `git init` / `git add` / `git commit`（属 W2 的 P0-T6）；删除任何文件。

**验收**：主代理抽样 10 个文件重算哈希；独立跑一次凭据 grep；核对总体积与最大文件。

---

## W2（W1 全部 PASS 后单独执行，独占写）

### P0-T6 — FOUNDATION_ARCHIVE.md + Git 基线 + `foundation-v0` tag

**目标**：把 W1 的四份证据汇总成不可变基线文档，并建立 Git 基线。

**输入**：W1 五项的全部输出（路径与 SHA-256 由主代理在派发时填入）。

**输出**
- `/home/wxh/go2_short_vln/reports/FOUNDATION_ARCHIVE.md`
- Git 仓库（`git init` + 首次提交 + `git tag foundation-v0`）

**要求**
1. `FOUNDATION_ARCHIVE.md` 覆盖 §2 全部八项能力，每项含：证据路径（`/home` 与 `/mnt` 双根映射）、SHA-256、产生命令、状态（VERIFIED/PARTIAL/MISSING）。
2. 单列一节「已知缺失与不可追认项」，含文档声称但不存在的三条报告、all61 的 QC 局限、ep1065 warmup 失败。
3. 单列一节「附录 A 三块历史资产去向」：A.1 M0–M7 短程 VLN 线、A.2 M7-N 语义 short-route、A.3 官方 NaVILA velocity benchmark（各自：保留位置、是否复用、不复用理由）。A.4 dual_target 标明**保留使用**，A.5 诊断资产标明**方法复用**。
4. Git 提交信息写明这是 P0 存量冻结基线，列出 tag 含义。提交后打印 `git log --stat | head -50` 与 `git tag` 存档到 `reports/p0/git_baseline.json`。

**验收**：主代理核对 tag 存在、`git ls-files | wc -l` 与 T5 清单一致、仓库体积、`.gitignore` 生效（`git status --porcelain` 无大文件待提交）。

---

## 台账

| 任务 | 状态 | 派发时间 | 交回时间 | 主代理判定 |
|---|---|---|---|---|
| P0-T1 | 重派中（terra xhigh） | 2026-09-12 | — | — |
| P0-T2 | **已交回** | 2026-09-12 | 2026-09-12 19:28 | **PASS**（主代理独立复算全部吻合，见下） |
| P0-T3 | 重派中（astra mid） | 2026-09-12 | — | — |
| P0-T4 | 重派中（terra xhigh） | 2026-09-12 | — | — |
| P0-T5 | 重派中（astra mid） | 2026-09-12 | — | — |
| P0-T6 | 未就绪（等 W1 全部 PASS） | — | — | — |

**Codex 运行时事实（2026-09-12 实测）**：该 ChatGPT 账号拒绝全部 `*-codex` 系列模型 ID
（`gpt-5.4`、`gpt-5.3-codex`、`gpt-5.2/5.1/5-codex`、`gpt-5.5-codex`、`codex-mini-latest`、`gpt-5.4-mini`、`gpt-5.1`、`gpt-5`
均返回 400 `not supported when using Codex with a ChatGPT account`）。
实测可用：`gpt-5.6-terra`（`~/.codex/config.toml` 默认）、`gpt-5.5`、`gpt-6-astra`、`gpt-5.3-codex-spark`。
派发时**不覆盖** `--model`，走配置默认。

### P0-T2 验收记录（主代理独立复算，2026-09-12）

判定 **PASS**。子代理产出：
- `reports/split_disjoint_check.json` SHA-256 `274c634868fe5efe92f35509a5e010a8bdbde2ca4d0dc1ab14478c614486c449`
- `scripts/p0_split_disjoint_check.py` SHA-256 `1353a88dd336c50b1d557ea7b552d088b28c4f81d7273ff70789d16e3905e8b2`

主代理**不使用子代理脚本**，另写一段 python 直接从源 `.json.gz` 复算，结果逐项吻合：

| 项 | 子代理报告 | 主代理独立复算 | 一致 |
|---|---|---|---|
| train `train.json.gz` SHA-256 | `f411066b…56a34` | `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34` | ✅ |
| eval `vln_ce_isaac_v1.json.gz` SHA-256 | `ceb2a2a9…0b0eec` | `ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec` | ✅ |
| train episodes | 10819 | 10819 | ✅ |
| eval episodes | 1077 | 1077 | ✅ |
| train scenes（stem 归一化） | 61 | 61 | ✅ |
| eval scenes | 11 | 11 | ✅ |
| 交集 | `[]` | `[]` | ✅ |
| eval `goals[0].radius` 分布 | `{3.0}`，无反例 | `{3.0: 1077}` | ✅ |

eval 11 scenes 实测清单：`2azQ1b91cZZ`、`8194nk5LbLH`、`EU6Fwq7SyZv`、`QUCTc6BB5sX`、`TbHJrupSAjP`、
`X7HyMhZNoso`、`Z6MFQCViBuw`、`oLBMNvg9in8`、`pLe4wQe7qrG`、`x8F5xyUWy9e`、`zsNo4HB9uLZ`。

Schema 合规：六个强制顶层字段齐全，`missing` 为空，归一化规则已显式写出（`PurePosixPath(scene_id).stem`，无 case folding / 别名映射）并同时保留归一化前的 raw scene_id 全量列表。

**结论**：里程碑 §4「两者 scene 集合不相交，评测天然为 unseen」由此条取得机械证据。
§3.4 要求的「全部 1077 条 `goals[0].radius == 3.0`」也已取证（1077/1077），可直接作为 P1 的输入。

### 运行时事故记录（2026-09-12）

W1 首轮五路并发中四路被 codex 运行时打断，零产物：
- `Selected model is at capacity`（`gpt-5.6-terra`）击中 T3、T5
- `Reconnecting... 2/5` → 进程 terminated（exit 143）击中 T1、T4

已按用户裁决把模型区间收敛为 **下限 `gpt-5.6-terra` + `xhigh`，上限 `gpt-6-astra` + `mid`**，
并分散到两个模型降低 capacity 争抢。重派派发单额外加一条**鲁棒性要求**：
尽早落盘、增量重写、用少量批处理命令替代大量小命令，避免传输中断吞掉全部进度。
`gpt-5.5` 与 astra `xhigh` 的两个作业已 cancel（前者低于下限，后者高于上限）。

---

## W1 验收记录（主代理独立复算，2026-09-12）

复算环境：`/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python`（numpy 2.2.6）。
系统 `python3` 无 numpy，凡涉及 `.npz` 的复算必须用该解释器。

### P0-T3 判定：**PARTIAL**（核心数字正确，但有一处错误的 MISSING + 一处漏报）

子代理产出 `reports/p0/foundation_evidence.json` SHA-256 `f6c19818a3b768f8d9a043110949fd367c9d53bc848232ed113fc6a1946f4390`。

**主代理确认正确的部分**

| 项 | 文档声称 | 独立复算 | 判定 |
|---|---|---|---|
| ep1_v3 高层区间数 | 196 | `actions.jsonl` 196 行 | ✅ |
| ep1_v3 有效 RGB | 236 | `frames.jsonl` 236 行 | ✅ |
| ep1_v3 停止意图区间 | 13 | `actions.jsonl` `stop_intent` 正例 **13** | ✅ |
| ep1_v3 npz 行数 | 195（文档已注明"数组仅 195 个完整区间"） | `npz` 195 行，`stop_intent` 正例 **12** | ✅ 二者不矛盾 |
| continuous_v2 区间 / RGB | 170 / 205 | `actions.jsonl` 170、`frames.jsonl` 205 | ✅ |
| continuous_v2 停止意图 | 文档未声称 | `actions.jsonl` 12、`npz` 12 | 实测补齐 |

**12 vs 13 的来源已钉死**：ep1_v3 第 196 个（最后一个）区间携带第 13 个停止意图，该区间不完整因而被排除在 `npz` 之外。
因此「13」的唯一正确数据源是 `actions.jsonl`，「12」的唯一正确数据源是 `ablation_arrays.npz`。
**任何引用这两个数字的地方必须标注数据源**，否则必然被读成矛盾。

**主代理推翻的 MISSING 判定**

子代理称「`wz` 原始/应用范围未能从现有 JSONL 结构可靠解析」。这是**错的**：数据不在 JSONL，而在
`ablation_arrays.npz` 的 `command_raw` / `command_applied`（均为 `(N,3)` float32，列序 `[vx, vy, wz]`）。
主代理直接复算：

| 数组 | 列 | ep1_v3 min / max / maxabs | continuous_v2 min / max / maxabs |
|---|---|---|---|
| `command_raw` | vx | +0.000000 / +0.349996 / **0.349996** | +0.000000 / +0.349925 / **0.349925** |
| `command_raw` | vy | 0 / 0 / 0 | 0 / 0 / 0 |
| `command_raw` | wz | −2.536474 / +1.036563 / **2.536474** | −2.504359 / +0.541811 / **2.504359** |
| `command_applied` | vx | +0.000000 / +0.349996 / 0.349996 | +0.000000 / +0.349925 / 0.349925 |
| `command_applied` | wz | −0.500000 / +0.500000 / **0.500000** | −0.500000 / +0.500000 / **0.500000** |

**结论**：§v2 执行裁决对 §0 修订 #2 的纠正**取得实证**——「expert 全部在 ±0.5 内」只适用于 `command_applied`；
`command_raw` 的 wz 最大绝对值实测 **2.536474 rad/s**，与文档「约 2.536」一致到小数第三位。
`command_applied` 的 wz 恰好落在 ±0.500000 边界，证明存在一次绝对 clipping。

**主代理发现的漏报（新事实，文档未记录）**

`command_raw` 与 `command_applied` 的 **vx 最大值仅 0.349996 / 0.349925**，即现有 expert 数据只覆盖
`vx ∈ [0, 0.35]`，而 §3.2 冻结的契约是 `vx ∈ [0.0, 0.5]`。
**`vx ∈ (0.35, 0.5]` 是零监督区间**——这与 §0 修订 #2 处理的 wz 问题属于同一类缺陷（契约允许模型在无监督区域自由输出）。
**此项必须作为 P1 的显式决策条目**：要么把 `vx` 上界收到 0.35，要么补采覆盖 0.35–0.5 的数据，要么显式接受并在报告中写明局限。不得默认沿用 0.5。

**MISSING 项已核对**：文档声称的三条 `/home` 路径确实不存在。其中
`reports/r2r_ablation/20260910_ep1_v3/` 属**路径记错**（真实证据在 `/mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_ep1_v3/`，内容完好）；
`reports/r2r_vlnce/audit_report.md`、`reports/OFFICIAL_NAVILA_VELOCITY_0911.md` 属**证据缺失**。三者性质不同，不可合并表述。

### P0-T4 判定：**PASS**（附三条主代理补充事实）

子代理产出：
- `reports/p0/all61_audit.json` SHA-256 `6c24ea47e8ceeec5f388412f394a509fc8968ee7c3110a0ab1fc3973ec322373`
- `reports/p0/ALL61_AUDIT.md` SHA-256 `06e5ad2dd2a6ab7919bebbd036a9bbccd9d95880a6204128bb7f4656cf847e23`

主代理独立复算全部吻合：

| 项 | 子代理 | 主代理独立复算 | 一致 |
|---|---|---|---|
| all61 目录（排除 `_cache`） | 61 = 60 episode + 1 `scene_probe_v1` | 61，非 episode 者确为 `20260910_all61_scene_probe_v1` | ✅ |
| 含 `summary.json` 的目录 | 60 | 60 | ✅ |
| 树内 `summary.json` 总数 | 82 = 60 + 22 其他实验 | 82，其中 all61 占 60，其他 22 | ✅ |
| `termination_reason` 分布 | 34 success / 18 timeout / 6 failed / 2 env termination | `{success:34, timeout:18, None:6, environment_termination:2}` = 60 | ✅ |
| `success` 布尔 | 34 | `True:34 / False:26` | ✅ |
| 覆盖 scene 数 | 60 | 60（从各 episode 的 `original_episode.json` 提取，零 UNKNOWN） | ✅ |
| 缺失场景 | `XcA2TqTSSAj` | 计划 61 − 实际 60 = 1，与之相符 | ✅ |

**主代理补充的三条事实（子代理未报）**

1. **`training_eligible` 在全部 60 条上均为 `False`**，无一例外。`summary.json` 自带备注
   `training eligibility requires separate audit`。
   意味着 all61 批次**当前零条具备训练资格**，这不是成功率问题而是资格问题，必须写进 `FOUNDATION_ARCHIVE.md`，
   并作为 P4「首批 50 条 QC 合格数据」的起点前提——不能把 34 条 success 误当作 34 条可训练数据。
2. **`maximum_contact_force_n` 在 54 条完成样本上全为 `0.0`，6 条失败样本为 `None`**，即整批零例外。
   这把 §6 X1 的「19 个 body 接触力恒零」从单条证据**放大为 60 条批量证据**，
   直接支撑 §7 / P7 的「collision 指标必须用代理来源并标注局限」的裁决。
3. **`stop_positive_actions` 存在一个离群值 `113`**（其余分布为 0 ×20、12 ×4、13 ×14、14 ×9、15 ×3、16 ×2、18 ×1、`None` ×6）。
   113 比次高值高出一个量级，**疑为停止意图锁存或分段缺陷**，列为 P4 数据 QC 的必查项。

### P0-T5 判定：**PASS**（附一条必须更正的结论 + 两条隐患）

子代理产出：
- `reports/p0/source_manifest.json` SHA-256 `679efa9d857885cfc744e014966124ac70618e6672ec88fa13ffa93061215c4c`
- `reports/p0/SOURCE_MANIFEST.md` SHA-256 `018c83a7c1044b2ea66e4a7f9cf2d90b0b8236a0c9b1adf77c7b94eb9b4d764d`
- `.gitignore` SHA-256 `7cdfbd0302cf9865bb2abde1ba0ebb8617c852141952c4b773562eef47161395`

**复算方法**：主代理不使用子代理清单，改用 scratchpad 内的独立 `GIT_DIR` 套 `--work-tree` 指向项目，
以 `git status --porcelain -uall` 真实施加 `.gitignore` 规则（**不在项目内 `git init`**，避免侵占 P0-T6 的范围）。

| 项 | 子代理 | 主代理独立复算 | 判定 |
|---|---|---|---|
| 拟入 Git 文件数 | 445 | 448 | 差异已解释 ↓ |
| 拟入 Git 总字节 | 3,068,192 | 3,340,183 | 差异已解释 ↓ |
| >10 MB 文件 | 0 | 0 | ✅ |
| 总量 < 50 MB | 成立 | 成立（3.3 MB） | ✅ |
| `._*` 文件 | 142，AppleDouble 资源叉，全部排除 | 模式生效，零 `._*` 进入清单 | ✅ |

**差异原因**：并发任务在 T5 采样之后落盘。主代理侧最大文件是 `reports/p0/source_manifest.json`（260,402 B，T5 自己的产出），
其次 `src/dual_target/runner.py`（88,423 B，与 T5 报告的"最大文件"一致）。
T4 的两份产出与 T3 的 `foundation_evidence.json` 也在 T5 之后出现。**非缺陷**。

**必须更正的结论：19 条凭据扫描命中全部是假阳性。**

主代理逐条取出命中行的实际文本，全部为字符串 `task-` 中的 `sk-` 子串：

| 命中位置 | 实际文本片段 |
|---|---|
| `config/dual_target_v1/dt0_contract.json:2` | `"go2-dual-target-ta`**`sk-`**`contract-v1-draft"` |
| `src/dual_target/runner.py:5` | `marker, or ta`**`sk-`**`derived control input` |
| `src/collector/collect_expert.py:726` | `# ta`**`sk-`**`config import.` |
| `scripts/dual_target_dt2_episode.sh:37` | `--tiny-plan "$plan" --ta`**`sk-`**`index "$index"` |
| `src/dual_target/tiny_runtime.py:17` | `parser.add_argument('--ta`**`sk-`**`index', ...)` |

`reports/p0/DISPATCH_P0.md:156` 的 6 条命中是**本派发单自己列出的扫描模式清单**，属自指命中。
**真实凭据泄露数 = 0**。子代理原样罗列命中而未判定假阳性，是报告质量缺陷；
`FOUNDATION_ARCHIVE.md` 与 P0-T6 的提交必须采用「零真实凭据」这一更正后结论。

**隐患一（latent trap，非当前缺陷）**：`.gitignore` 含 `*token*`。当前树内匹配 `*token*` / `*credential*` / `*.key` 的文件数为 **0**，
故无过度排除。但将来任何 `tokenizer.py` / `token_utils.py` 会被**静默忽略**。
建议 P0-T6 收紧为 `*.token`、`*credentials*.json`、`*.key`，或对 `*token*` 加 `!**/tokenizer*.py` 例外。

**隐患二**：`.gitignore` 排除整个 `data/`，而仓库 `data/` 实际只含**一个文件** `data/short_vln_v1.json`（4.8 MB，低于 10 MB 上限）。
排除它是子代理的判断，派发单原文要求「先核实体积再决定，结论写进报告」。
P0-T6 须在 `FOUNDATION_ARCHIVE.md` 中给出保留或排除的明确理由；若排除，须记录该文件的 SHA-256 以保可追溯。
`third_party/` 仅含两个指向 `/mnt` 的符号链接（`IsaacLab`、`NaVILA-Bench`），排除合理，但链接目标本身是有价值的溯源信息，应记入归档文档。

### 事故记录：游离空 `.git` 目录

复算过程中发现 `/home/wxh/go2_short_vln/.git` 存在，创建时间 `2026-09-12 19:30:45`，
但 `git log` 报 `not a git repository`。递归检查确认**整目录仅 1 个条目（即自身），无 `refs/`、无 `index`、无 `objects`**，
即空目录而非仓库。已由主代理 `rmdir` 移除（无任何提交或对象损失风险）。
产生原因未定位（派发单明确禁止子代理执行 git 写操作）。P0-T6 执行 `git init` 前须确认工作目录干净，
并在 `git_baseline.json` 中记录此次事故。

### W1 汇总

| 任务 | 判定 | 遗留动作 |
|---|---|---|
| P0-T1 | 进行中（前两次被 capacity / 传输中断打断，第三次改派 astra mid） | — |
| P0-T2 | **PASS** | 无 |
| P0-T3 | **PARTIAL** | wz 范围已由主代理补齐；vx 零监督区间须进 P1 决策条目 |
| P0-T4 | **PASS** | 三条补充事实须进归档文档 |
| P0-T5 | **PASS** | 凭据结论须更正为零；`*token*` 与 `data/` 两项须在 P0-T6 定论 |

**W1 尚未整体 PASS**：P0-T1 未交回。P0-T6 在 T1 验收通过后方可派发，
且其派发单必须把上述全部更正与补充事实作为强制输入。

---

## P0-T1 验收记录（主代理独立复算，2026-09-12）

### 判定：**PASS**（附一条必须更正的 MISSING + 一条重要新事实）

子代理产出：
- `reports/p0/queue_audit.json` SHA-256 `497435e56d436292c90ee654f47194d8cfc82276ac1f60fb7ef0ace8a126dc4d`
- `reports/p0/QUEUE_AUDIT.md` SHA-256 `96daaba9f2809f87be594e845ebb8399912ee6b246f0224c25b7b1569d829643`

| 项 | 子代理 | 主代理独立复算 | 判定 |
|---|---|---|---|
| PID 779548 | 不存在，`action: no_action_required` | `ps -p 779548` 无输出 | ✅ 且措辞正确（未写成"已停止"） |
| 项目自有进程 | 无 | 按 `collect_r2r\|zoh_\|navila\|isaac\|train_\|evaluate` 匹配，计数 **0** | ✅ |
| `/tmp/official_navila_velocity_0911_full_v1.log` | MISSING | 确实不存在 | ✅ |
| `progress.json` | `completed=4`、`planned=1077`、`FAILED_NEEDS_REVIEW` | 完全一致 | ✅ |
| `plan.json` | 1077 episodes | 1077 | ✅ |
| `results.json` | 4 条，`complete=false` | 4 条，`complete=False` | ✅ |
| episode 目录数 | 4 | 4（`episode_000001`–`episode_000004`） | ✅ |
| 需停止的进程 | 无，未执行停止 | 确认无，未发生任何停止操作 | ✅ |

三份 JSON 内部自洽：`completed=4` = `results` 条数 4 = 产物目录数 4。

**必须更正的 MISSING：GPU 事实并不缺失。**

子代理称「`nvidia-smi` 无法连接 NVIDIA driver，因此 GPU 占用者标记为 MISSING」。
主代理在同一时刻同一台机器上复算，`nvidia-smi` **正常工作**：

```
NVIDIA GeForce RTX 3090, 38 MiB / 24576 MiB
nvidia-smi --query-compute-apps=pid,used_memory --format=csv  →  表头之外零行
```

即 **GPU 空闲、零 compute 进程、无非本项目占用**。
子代理的 `nvidia-smi` 失败是 **codex 执行沙箱屏蔽了 GPU 设备节点**所致，
**不是**关于本机的事实。`FOUNDATION_ARCHIVE.md` 必须采用主代理的实测结论。

**方法论教训（写入调度协议）**：codex 子代理的沙箱能力边界与主代理不同。
凡涉及**设备节点、驱动、内核接口**（GPU、`/dev`、`/proc` 特权项）的核查，
子代理报 MISSING 时主代理必须自行复核，不得直接采信；反之涉及大规模文件遍历与哈希时子代理更高效。

**重要新事实：官方 1077 队列不是"被 P0 停止"的，而是自行失败在 4/1077。**

`progress.json` 的 `status` 为 **`FAILED_NEEDS_REVIEW`**，`completed=4`、`planned=1077`。
这与两处文档表述冲突，必须更正：

1. **§P0 任务 1** 写「停止正在运行的官方 1077-episode 队列（PID 779548）」——
   该队列早已不在运行，且从未接近全量；P0 无需也无法执行此停止动作。
2. **§附录 A.3** 写「于 P0 停止。已完成 episode 与 progress / plan / results 全部保留」——
   前半句不成立：队列在第 4 条后自行进入失败状态，**不是** P0 的人工停止。
   后半句成立：4 条 episode 产物与三份 JSON 完整保留。

A.3 同时称「官方 index 0 / episode ID 1 完成」，与 `completed=4` 并不矛盾（1 是 4 之一），
但把单条成功读成队列健康是错的——**真实状态是 4/1077 + `FAILED_NEEDS_REVIEW`**。
该队列失败的**根因尚未调查**（`/tmp` 日志已不存在），列为 P0 的 MISSING 项；
是否值得追查由用户在 `CONTINUE P1` 时裁决——它与新契约不兼容（§A.3 不复用理由成立），根因可能无需追查。

---

## W1 整体判定：**PASS**（5/5 交回，其中 T3 为 PARTIAL）

| 任务 | 判定 | 产出 |
|---|---|---|
| P0-T1 | **PASS** + 1 更正 + 1 新事实 | `queue_audit.json`、`QUEUE_AUDIT.md` |
| P0-T2 | **PASS** | `split_disjoint_check.json`、`scripts/p0_split_disjoint_check.py` |
| P0-T3 | **PARTIAL** | `foundation_evidence.json`（wz 由主代理补齐） |
| P0-T4 | **PASS** + 3 补充事实 | `all61_audit.json`、`ALL61_AUDIT.md` |
| P0-T5 | **PASS** + 1 更正 + 2 隐患 | `source_manifest.json`、`SOURCE_MANIFEST.md`、`.gitignore` |

**零项 FAIL。可以派发 W2。**

---

## W2 派发单

### P0-T6 — FOUNDATION_ARCHIVE.md + Git 基线 + `foundation-v0` tag

**目标**：把 W1 的证据（**含主代理的全部更正与补充**）汇总成不可变基线文档，并建立 Git 基线。

**资源**：CPU，独占写。W1 全部任务已收工，无并发冲突。

**输入**（全部已就位）

| 路径 | SHA-256 |
|---|---|
| `reports/p0/queue_audit.json` | `497435e56d436292c90ee654f47194d8cfc82276ac1f60fb7ef0ace8a126dc4d` |
| `reports/p0/QUEUE_AUDIT.md` | `96daaba9f2809f87be594e845ebb8399912ee6b246f0224c25b7b1569d829643` |
| `reports/split_disjoint_check.json` | `274c634868fe5efe92f35509a5e010a8bdbde2ca4d0dc1ab14478c614486c449` |
| `scripts/p0_split_disjoint_check.py` | `1353a88dd336c50b1d557ea7b552d088b28c4f81d7273ff70789d16e3905e8b2` |
| `reports/p0/foundation_evidence.json` | `f6c19818a3b768f8d9a043110949fd367c9d53bc848232ed113fc6a1946f4390` |
| `reports/p0/all61_audit.json` | `6c24ea47e8ceeec5f388412f394a509fc8968ee7c3110a0ab1fc3973ec322373` |
| `reports/p0/ALL61_AUDIT.md` | `06e5ad2dd2a6ab7919bebbd036a9bbccd9d95880a6204128bb7f4656cf847e23` |
| `reports/p0/source_manifest.json` | `679efa9d857885cfc744e014966124ac70618e6672ec88fa13ffa93061215c4c` |
| `reports/p0/SOURCE_MANIFEST.md` | `018c83a7c1044b2ea66e4a7f9cf2d90b0b8236a0c9b1adf77c7b94eb9b4d764d` |
| `.gitignore` | `7cdfbd0302cf9865bb2abde1ba0ebb8617c852141952c4b773562eef47161395` |
| 本派发单（含全部验收记录） | 执行时自行计算并记录 |

**输出**：`reports/FOUNDATION_ARCHIVE.md`、`reports/p0/git_baseline.json`、Git 仓库 + `foundation-v0` tag

**强制输入：主代理的更正与补充（优先于子代理报告原文）**

以下 9 条**必须**体现在 `FOUNDATION_ARCHIVE.md` 中。凡与子代理原始报告冲突处，以本清单为准：

1. **wz 范围不是 MISSING**。数据源为 `ablation_arrays.npz` 的 `command_raw` / `command_applied`（`(N,3)` float32，列序 `[vx,vy,wz]`）。
   ep1_v3：raw wz ∈ [−2.536474, +1.036563]，maxabs **2.536474**；applied wz ∈ [−0.500000, +0.500000]。
   continuous_v2：raw wz maxabs **2.504359**；applied wz ±0.500000。
   结论：「±0.5」只适用于 applied，存在一次绝对 clipping。
2. **vx 零监督区间**：raw/applied vx 最大值仅 **0.349996**（ep1_v3）/ **0.349925**（continuous_v2），
   而 §3.2 契约为 `vx ∈ [0.0, 0.5]`；`(0.35, 0.5]` 无任何监督数据。列为 **P1 必决条目**。
3. **停止意图 12 vs 13 必须标注数据源**：`actions.jsonl` = 13（196 区间），`ablation_arrays.npz` = 12（195 区间，
   第 196 个不完整区间被排除）。continuous_v2 两者均为 12。
4. **all61 的 `training_eligible` 全 60 条为 `False`**，无一例外。34 条 success **不等于** 34 条可训练数据。
5. **all61 的 `maximum_contact_force_n`**：54 条完成样本全为 `0.0`，6 条失败样本为 `None`，整批零例外。
   这是 X1 接触力问题的 60 条批量证据。
6. **all61 `stop_positive_actions` 离群值 113**（其余 0–18），疑为停止意图锁存或分段缺陷，列为 P4 QC 必查项。
7. **凭据扫描真实泄露数 = 0**。T5 报告的 19 条命中全为假阳性：13 条源于字符串 `task-` 内含 `sk-` 子串，
   6 条源于本派发单自列的扫描模式（自指命中）。归档文档须采用"零真实凭据"结论。
8. **GPU 事实不缺失**：主代理实测 `nvidia-smi` 正常，RTX 3090 用 38 MiB / 24576 MiB，零 compute 进程，无非项目占用。
   T1 的 MISSING 系 codex 沙箱屏蔽设备节点所致，不是本机事实。
9. **官方 1077 队列并非"被 P0 停止"**：`progress.json` 为 `completed=4`、`planned=1077`、`status=FAILED_NEEDS_REVIEW`，
   队列自行失败于 4/1077。须同时更正 §P0 任务 1 与 §附录 A.3 的表述。失败根因未调查（`/tmp` 日志已不存在），记为 MISSING。

**任务**

1. `FOUNDATION_ARCHIVE.md` 覆盖 §2 全部八项能力，每项含：证据路径（**`/home` 与 `/mnt` 双根映射**）、SHA-256、
   产生命令、状态（VERIFIED / PARTIAL / MISSING）。数据取自上表四份 JSON，冲突处以「强制输入」清单为准。
2. 单列「已知缺失与不可追认项」：文档声称但不存在的三条报告（区分**路径记错** vs **证据缺失**）、
   all61 的 QC 局限（缺 `XcA2TqTSSAj`、6 个 scene 仅 failed warmup、`training_eligible` 全 False）、
   ep1065 warmup 失败、官方队列失败根因未调查。
3. 单列「附录 A 三块历史资产去向」：A.1 M0–M7 短程 VLN 线、A.2 M7-N 语义 short-route、A.3 官方 NaVILA velocity benchmark
   （各自：保留位置、是否复用、不复用理由）。A.4 dual_target 标 **保留使用**，A.5 诊断资产标 **方法复用**。
4. **`.gitignore` 两项定论**（派发单要求的"结论写进报告"）：
   (a) `data/` 实际只含一个文件 `data/short_vln_v1.json`（4.8 MB，低于 10 MB 上限）。给出保留或排除的**明确理由**；
       若排除，**必须记录该文件的 SHA-256** 以保可追溯。
   (b) `*token*` 是 latent trap（当前零命中，但将来会静默忽略 `tokenizer.py` 一类文件）。
       收紧为 `*.token` / `*credentials*.json` / `*.key`，或加 `!**/tokenizer*.py` 例外。**改动 `.gitignore` 后重算其 SHA-256**。
   (c) `third_party/` 仅含两个指向 `/mnt` 的符号链接（`IsaacLab`、`NaVILA-Bench`），排除合理，但**链接目标须记入归档文档**。
5. 建立 Git 基线：`git init` → 首次提交 → `git tag foundation-v0`。
   **`git init` 前先确认 `/home/wxh/go2_short_vln/.git` 不存在**——主代理已移除一个创建于 `19:30:45` 的游离空 `.git` 目录
   （仅 1 个条目，无 refs / index / objects，无损失），此事故须记入 `git_baseline.json`。
6. 提交信息写明这是 P0 存量冻结基线并说明 tag 含义。
   提交后把 `git log --stat | head -50`、`git tag`、`git ls-files | wc -l`、仓库体积、`git status --porcelain`
   存档到 `reports/p0/git_baseline.json`。

**禁止**：改动 W1 任何产出的 JSON（`.gitignore` 除外，见任务 4）；删除任何文件；
把大文件加入 Git；提交前不检查 `.gitignore` 生效情况；在证据缺失处自行追认 PASS。

**验收（主代理独立复算）**：核对 `foundation-v0` tag 存在；
`git ls-files | wc -l` 与主代理自己的 `GIT_DIR` 模拟计数一致；仓库体积 < 50 MB 且零 >10 MB 文件；
`git status --porcelain` 无大文件待提交；`FOUNDATION_ARCHIVE.md` 的 9 条强制输入逐条到位；
`data/short_vln_v1.json` 的处置有理由且（若排除）有 SHA-256。

**台账**

| 任务 | 状态 | 派发时间 | 交回时间 | 主代理判定 |
|---|---|---|---|---|
| P0-T6 | 已派发 | 2026-09-12 | — | — |

---

## P0-T6 验收记录（主代理独立复算，2026-09-12）

### 判定：**PARTIAL — 需返工（P0-T6R）**；Git 基线**受结构性阻塞**

子代理产出（主代理重算哈希）：

| 文件 | 子代理报告 | 主代理重算 | 判定 |
|---|---|---|---|
| `reports/FOUNDATION_ARCHIVE.md` | `1ba472eaa42526a4e8534603b933fc27bfc5c3d83`（**41 字符，非法**） | `1ba472eaa42526a4e8534603b933fc27bfc5c3d2a0f0d1af62060053c58e3d83` | 报告串码，文件本身正常 |
| `reports/p0/git_baseline.json` | `07ff32c7…c9c7d4c` | 一致 | ✅ |
| `.gitignore`（已按任务 4 修订） | `317bbca4…e05e1474` | 一致 | ✅ |
| `data/short_vln_v1.json` | `88255ffb…0b34b6` | 一致 | ✅ |

**做对的部分**

- 9 条强制更正**逐条落位**，措辞准确，未回退到子代理原始错误结论。
- `.gitignore` 两项隐患均已处置：`data/` 整体忽略已移除（该文件 4.8 MB < 10 MB，纳入基线并记录 SHA-256）；
  `*token*` latent trap 已收紧为 `*.token` / `*credentials*.json` / `*.key`；`third_party/` 两个 `/mnt` 符号链接已记入文档。
- 附录 A 五块资产（A.1–A.5）去向齐全，A.3 已按更正写明「自行失败于 4/1077，非 P0 停止」。

**缺陷一（严重）：存在主代理无法复算的哈希。**

「Go2 低层 locomotion」行写 `manifest 00f188c7213f8cb329cf738409133cd70e6521d55808c57f4de8c5f52280f28d`。

- 该值**不是** `file_manifest.json` 自身的 SHA-256（实测 `7d2d9fc83294590563e10c66853f3979a1fd3a48f7c8220821175cfa6427405e`）。
- 主代理尝试三种自然聚合方式复算，全部不匹配：
  按序拼接条目哈希 → `ea82eda474872a0b…`；按 path 排序拼接 → `ea82eda474872a0b…`；`sha256  path` 行拼接 → `42498d33ec7484f7…`。

同行的**计数与字节数经核实是正确的**：`file_manifest.json` 含 **248** 条条目、合计 **17,961,385** 字节；
加上 `file_manifest.json` 自身 34,731 字节 = **249 个文件 / 17,996,116 字节**，与报告完全吻合。
即缺陷仅在哈希一项。

**P0 的全部意义就是可复算的哈希**。返工必须：改用 `file_manifest.json` 自身的 SHA-256，
或保留聚合哈希但**写出精确算法**（输入序列、分隔符、编码），使第三方能一行复现。

**缺陷二：7/8 能力的 `Producing command` 标为 MISSING，不成立。**

主代理实测证据就在产物里：

- `20260910_all61_ep1023_guard.json` 的 `command` 字段存**完整命令行**：
  `/mnt/wxh/go2_short_vln/envs/conda/navila-isaac/bin/python /home/wxh/go2_short_vln/scripts/collect_r2r_continuous_v1.py --expert continuous --output /mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_all61_ep1023 --episode-id 1023 --seed 20260910 --max-seconds 120 --load_run 2024-09-25_23-22-02 --headless --enable_cameras`
  （同文件另有 `status`、`pid`、`pgid`、`exit_code`、`min_free_mib`、`timeout_s`）
- `20260910_ep1_v3/manifest.json` 存完整溯源：`source` + `source_sha256`、`gt_source`、
  `collector_sha256 = e734e0b1…f3b82e`、`checkpoint = …/2024-09-25_23-22-02/model_26499.pt` + `checkpoint_sha256 = 1e210971…f2e76c`、
  `seed = 20260910`、`zoh = {physics_dt_s:0.005, decimation:4, hold_control_steps:10, chunk_size:5, execute_steps:1, …}`、
  `numeric_hz = 50`、`primary_rgb_hz = 5`、`navila_keyframe_interval_s = 0.5`、`command_boundary_sha256`。
- `20260910_ep1_v3.log` 存在于 `r2r_ablation/` 下（与目录同级）。

派发单原文允许「还原不出写 MISSING」，但**此处还原得出**。返工须逐项填入实际命令或溯源字段组合，
仅在确实无迹可循处保留 MISSING 并说明查过哪些来源。

**缺陷三：「R2R-VLNCE 全量部署」整行标 MISSING，过度。**

缺失的是**审计报告文件** `reports/r2r_vlnce/audit_report.md`，而**能力本身已被 P0-T2 机械验证**：
`train.json.gz`（SHA-256 `f411066b…56a34`）实测 **10819 episodes / 61 scenes**，与 §2 声称一致。
正确状态应为 **PARTIAL**：能力 VERIFIED（证据 = `reports/split_disjoint_check.json`），原审计报告 MISSING。
「6744 文件校验」与「两版本 SHA-256 齐全」两项子声明仍无证据，须单独标 MISSING。

**缺陷四（结构性阻塞，非子代理过失）：Git 基线无法由 codex 完成。**

子代理报「沙箱拒绝写入 `.git/branches`」，`git init` 失败，无提交、无 tag，`git ls-files` = 0。
主代理复核：`/home/wxh/go2_short_vln/.git` 再次出现，且**仅 1 个条目、无 refs / index / objects**，
与 `19:30:45` 那个游离空目录**完全同形**。

**根因由此确认**：先前的游离空 `.git` 不是来源不明，而是 **codex 沙箱执行 `git init` 的半成品**——
沙箱允许创建 `.git` 目录但拒绝写入其子结构。该失败已发生**两次，原因相同**，
按 §5 调度协议第 9 条「同一失败原因两次有证据的修复尝试后仍存在，升级给主代理」，**升级**。

改派 codex 第三次必然同因失败。主代理**不自行实现**（§5 职责边界第 1 条），
故 Git 基线一项**上报用户裁决**，不在本轮自行推进。
`FOUNDATION_ARCHIVE.md`、`.gitignore`、`git_baseline.json` 三项产物**不受此阻塞影响，予以保留**。

### P0-T6R 返工派发单（archive 修订，不含 Git）

**目标**：修掉缺陷一至三。**不触碰 Git**。

**输入**：`reports/FOUNDATION_ARCHIVE.md`（SHA-256 `1ba472eaa42526a4e8534603b933fc27bfc5c3d2a0f0d1af62060053c58e3d83`）
及本节列出的全部实测证据。

**输出**：原地修订 `reports/FOUNDATION_ARCHIVE.md`；修订说明追加到 `reports/p0/git_baseline.json` 的 `remediation` 字段。

**必办三项**

1. 「Go2 低层 locomotion」行的哈希改为可复算形式（`file_manifest.json` 自身 SHA-256
   `7d2d9fc83294590563e10c66853f3979a1fd3a48f7c8220821175cfa6427405e`，或保留聚合哈希并写出精确算法）。
   **保留**已核实正确的 249 文件 / 17,996,116 字节，并注明其推导（248 条条目 + `file_manifest.json` 自身 34,731 字节）。
2. 逐项填写 `Producing command`，数据源见缺陷二。确实无迹可循处保留 MISSING 并列出已查来源。
3. 「R2R-VLNCE 全量部署」改为 **PARTIAL**：能力 VERIFIED（证据 `reports/split_disjoint_check.json`，实测 10819 ep / 61 scenes），
   审计报告文件 MISSING，「6744 文件校验」与「两版本 SHA-256」单独标 MISSING。

**禁止**：执行任何 git 命令（含 `git init`）；改动 `.gitignore`；改动 W1 产出的 JSON；
删除任何文件；改写已核实正确的 9 条强制更正与附录 A。

**台账**

| 任务 | 状态 | 派发时间 | 交回时间 | 主代理判定 |
|---|---|---|---|---|
| P0-T6 | 已交回 | 2026-09-12 | 2026-09-12 | **PARTIAL**（4 缺陷，其中缺陷四为结构性阻塞） |
| P0-T6R | 已派发 | 2026-09-12 | — | — |
| P0-T6G（Git 基线） | **阻塞：待用户裁决** | — | — | — |

---

## P0-T6R 验收记录 + 主代理自我更正（2026-09-12）

### T6R 判定：**缺陷一、三已修复；缺陷二修成了回归，须再返工（P0-T6R2）**

产出（主代理重算，与子代理报告一致）：
`reports/FOUNDATION_ARCHIVE.md` SHA-256 `4d1fcbfe35df8e5ae337fd88a4835d5a4e336ea2c83b1c47a092fcb10ec05935`；
`reports/p0/git_baseline.json` SHA-256 `54fbb8bca7a7046d94db5779b620c6601f779a908cb0d60e23f6bde9032011f4`。

**已修复**

- 缺陷一：locomotion 行哈希改为 `file_manifest.json` 自身 SHA-256 `7d2d9fc8…27405e`，并写出 249 / 17,996,116 的推导（248 条 + 34,731）。**可复算，通过。**
- 缺陷三：R2R-VLNCE 行改为 `PARTIAL`，能力 VERIFIED（10819 ep / 61 scenes），审计报告、「6744 file checks」、「two-version SHA-256」三项分别标 MISSING。**通过。**

**缺陷二修成了回归（比原状更差）**

子代理把**同一条命令**填进了全部八行：
`… collect_r2r_continuous_v1.py --expert continuous --output …/20260910_all61_ep1023 --episode-id 1023 …`

该命令是 **all61 批次 ep1023** 的，不是 §2 任何一行的产生命令。逐行看都错：
`gpu_wait.py`、`task_contract.lock.json`、`validate_r2r_episode_load.py`、`collect_r2r_ablation.py`、`audit_r2r_ablation.py`
是**源码/配置，不是生成产物**，本就没有"产生命令"；`train.json.gz` 是**外部数据集**；
而 locomotion 行的证据是 **ep1_v3**，被归因到了**另一条 episode（ep1023）**的命令上。

在一份以"冻结可复算证据"为唯一目的的文档里，**错误归因比诚实的 MISSING 更有害**。

### 主代理自我更正（重要）

**我在 P0-T6 验收记录「缺陷二」中的判断部分是错的。** 原话称「产生命令 ARE 可恢复」，
依据是 `20260910_all61_ep1023_guard.json` 的 `command` 字段。该依据只对 **all61 批次**成立，
**对 ep1_v3 不成立**。实测：

1. `20260910_ep1_v3.log` **不含调用命令行**，只有 Isaac kit 的启动参数（`simulation_app.py` + `.kit` 文件 + `--/app/...` 开关）。
2. ep1_v3 的 `manifest.json` 记 `collector_sha256 = e734e0b1b7079da0d27b634b30f0dd9dcfd4b9644c31b132d522b95749f3b82e`，
   **与树内任何现存脚本都不匹配**：
   - `scripts/collect_r2r_ablation.py` = `71eea6d5f24d70ac334192f4a632d8781be1fce255ca4efc0e38b585f377f67e`
   - `scripts/collect_r2r_continuous_v1.py` = `71eea6d5…`（**与上者字节完全相同**）
   - `scripts/collect_r2r_ablation.py.pre_platform_sync_20260912` = `147c68686597221196a7aa2ec4157f13ee840c5b5cfc9dd9d2889f30a1f7f440`

**结论：产生 ep1_v3 的 collector 源码已不在仓库树内。** 子代理最初把该行标 MISSING 是**正确的**，
是我的派发单措辞（引用 ep1023 的命令作范例而未限定其归属）诱发了这次错误归因。

**顺带确认的两条事实**

- **`collect_r2r_ablation.py` 与 `collect_r2r_continuous_v1.py` 字节完全相同**（同一 SHA-256）。
  即 §P2「legacy / continuous 分开，默认 `--expert legacy` 不能被当作 continuous」所指的区分
  **是运行时 `--expert` flag，不是两个不同脚本**。P2 回归设计必须据此调整：
  不能靠"跑哪个脚本"区分两条 expert 线，只能靠命令行 flag + 产物 manifest 记录。
- §P2 预设的「旧 collector 哈希无法恢复时保留为历史证据，另建当前代码基线」**已被证实为现实**，
  不再是假设：ep1_v3 的 collector 源码不可恢复。这直接影响 P2 的 ep1 回归可行性——
  **无法用原始 collector 重跑 ep1**，只能用当前代码建新基线并接受差异，与 §P2 的 v2 替代回归门槛一致。

### P0-T6R2 返工派发单（仅修 producing command 一列）

**目标**：把「Producing command」列改成逐行准确。**只改这一列，其余一字不动。**

**输入**：`reports/FOUNDATION_ARCHIVE.md` SHA-256 `4d1fcbfe35df8e5ae337fd88a4835d5a4e336ea2c83b1c47a092fcb10ec05935`

**逐行规定值**（不得跨行复制，不得引用 ep1023 命令）

| 行 | Producing command 应填 |
|---|---|
| Go2 locomotion（ep1_v3） | `MISSING — 日志无 argv；manifest collector_sha256 e734e0b1…f3b82e 与树内任何现存脚本均不匹配（已查 collect_r2r_ablation.py 71eea6d5…、collect_r2r_continuous_v1.py 71eea6d5…（与前者字节相同）、collect_r2r_ablation.py.pre_platform_sync_20260912 147c6868…），产生该证据的 collector 源码已不在仓库内` |
| R2R-VLNCE 全量部署 | `n/a — 外部数据集 R2R-VLNCE v1-3，非本仓库生成；溯源为 train.json.gz SHA-256 f411066b…56a34` |
| Habitat ↔ Isaac 坐标转换 | `n/a — 源码文件，非生成产物` |
| 路径 Expert | `n/a — 源码文件，非生成产物` |
| 采集链路 + 审计 | `n/a — 源码文件，非生成产物` |
| 消融数组接口（`ablation_arrays.npz`） | 与 locomedtion 行同源（ep1_v3），填同一 `MISSING` 说明 |
| GPU 共享治理 | `n/a — 源码文件，非生成产物` |
| 极小可控场景 | `n/a — 配置文件，非生成产物` |

**另须补一小节**「Producing-command provenance」，写明：
(a) all61 批次的命令**确实**可从 `20260910_all61_*_guard.json` 的 `command` 字段完整恢复（给出 ep1023 那条作示例，并明确标注它属于 all61 ep1023，**不属于** §2 任何一行）；
(b) `collect_r2r_ablation.py` 与 `collect_r2r_continuous_v1.py` 字节相同，legacy/continuous 由运行时 `--expert` flag 区分；
(c) ep1_v3 的 collector 源码不可恢复，P2 无法用原始 collector 重跑 ep1。

**禁止**：执行任何 git 命令；改动 `.gitignore`；改动 9 条强制更正、附录 A、Known missing 各节；
改动已修复的 locomotion 哈希与 R2R-VLNCE 状态；跨行复制同一命令。

**台账**

| 任务 | 状态 | 主代理判定 |
|---|---|---|
| P0-T6R | 已交回 | 缺陷一、三 **PASS**；缺陷二 **回归**，转 T6R2 |
| P0-T6R2 | 已派发 | — |
| P0-T6G（Git 基线） | **已获用户破例授权（E1）**，待 T6R2 交回后由主代理执行 | — |

---

## P0-T6R2 验收 + Git 基线执行记录（2026-09-12）

### P0-T6R2 判定：**PASS**

`reports/FOUNDATION_ARCHIVE.md` SHA-256 `a01feab5ef9ee01718beb9b4635787ff499fd116bc959482cfb07a67d1ea5025`（64 字符，与子代理报告一致）。

| 核查项 | 结果 |
|---|---|
| `all61_ep1023` 出现次数 | **1 次**，且仅在 provenance 小节，附「does not belong to any row in §2」 |
| locomotion 哈希 `7d2d9fc8…` | 保留 ✅ |
| R2R-VLNCE `PARTIAL (capability VERIFIED…)` | 保留 ✅ |
| 「6744 file checks」单独 MISSING | 保留 ✅ |
| 9 条强制更正编号 1–9 | 齐全 ✅ |
| 附录 A（A.1–A.5） | 完好 ✅ |
| `data/short_vln_v1.json` SHA-256 `88255ffb…` | 保留 ✅ |
| 新增 `Producing-command provenance` (a)(b)(c) | 齐全 ✅ |

八行 Producing command 逐行核对与派发表一致：ep1_v3 两行为带查证依据的 MISSING，
外部数据集行为 `n/a — 外部数据集`，五个源码/配置行为 `n/a — 源码/配置文件，非生成产物`。**零跨行复制。**

### P0-T6G Git 基线：**由主代理执行（用户授权例外 E1）**

执行前清理：移除 codex 沙箱第二次留下的游离空 `.git`（仅 1 条目）。
身份仅设在**仓库本地**（`git config` 无 `--global`）：`user.name=wxh`、`user.email=8s9zbg98t7@privaterelay.appleid.com`。

提交前门禁（全部先查后提交）：

| 门禁 | 结果 |
|---|---|
| 暂存文件数 | 455 |
| 暂存总字节 | 8,719,883（8.32 MB） |
| >10 MB 文件 | **0** |
| 最大暂存文件 | `data/short_vln_v1.json` 5,018,835 B |
| `outputs/` 入库 | **0**（符号链接已被忽略） |
| `third_party/` 入库 | **0** |
| 真实凭据 | **0**。仅 3 条自指命中（`DISPATCH_P0.md:156` 的模式清单本身，及 `SOURCE_MANIFEST.md` / `source_manifest.json` 对它的转述） |

结果：commit `9992ce1`（完整 `9992ce12c2a88bdaae96e4f55535102d546ab1b7`），
annotated tag **`foundation-v0`** 指向该 commit，工作树干净（`git status --porcelain` 为空），`.git` 4.1 MB。

### 主代理独立验收（E1 不豁免验收）

用 scratchpad 内**全新的独立 `GIT_DIR`**（绕过仓库自身 index）按当前 `.gitignore` 重新枚举应入库文件：

| 项 | 结果 |
|---|---|
| 独立模拟计数 | **455** |
| `git ls-files` 实际 | **455** |
| 两份排序清单 `diff` | **零差异**（逐文件完全一致） |
| `foundation-v0` 类型 | `tag`（annotated），指向 `9992ce1` |
| 仓库体积 | 4.1 MB，远低于 50 MB 门槛 |
| `git status --porcelain` | 空，无大文件待提交 |

**Git 基线验收 PASS。**

---

## P0 结项判定：**PASS**

| 里程碑验收项（§P0 v2） | 结果 |
|---|---|
| 旧队列按真实命令/进程/产物核对，不存在者记为无需操作 | ✅ PID 779548 不存在，记 `no_action_required`；无自有进程在跑；**未停止任何进程**；4 条已完成 episode 与 progress/plan/results 全部保留 |
| 先记录源码哈希清单，再建只跟踪源码/配置/必要文档的 Git 基线并提交、打 tag | ✅ `source_manifest.json` 先行；455 文件 / 8.32 MB；模型、数据、环境、缓存、大输出、凭据均排除 |
| `FOUNDATION_ARCHIVE.md` 覆盖 §2 八项，含双根映射、真实证据、命令、哈希、缺失项、多场景失败 | ✅ 八项齐全，7 VERIFIED + 1 PARTIAL；缺失项显式标注 |
| 缺失证据不直接追认 PASS | ✅ 三条声称报告、ep1_v3 collector 源码、队列失败根因均标 MISSING |
| scene 交集结果带源文件哈希存档 | ✅ 交集为空，双源哈希齐全，主代理独立复算吻合 |
| `foundation-v0` tag 存在 | ✅ annotated tag → `9992ce1` |

**执行统计**：6 个里程碑子任务、11 次 codex 派发（5 次因 capacity / 传输中断零产物重派，1 次因沙箱结构性阻塞升级为用户授权例外）、
1 轮返工 + 1 轮返工修回归。主代理独立复算推翻或补充子代理结论 **13 条**，并更正自身判断 **1 条**。

**P0 到此停止，等用户显式 `CONTINUE P1`。**

### 移交 P1 的待裁决条目（主代理不自行决定）

1. **vx 上界**：契约 §3.2 冻结 `vx ∈ [0.0, 0.5]`，但现有 expert 数据 raw/applied 最大仅 **0.349996 / 0.349925**，
   `(0.35, 0.5]` 零监督。三选一：收上界到 0.35 / 补采覆盖该区间 / 显式接受并写明局限。
2. **P2 回归门槛降级**：ep1_v3 的 collector 源码不可恢复（`collector_sha256 e734e0b1…` 与树内任何脚本均不匹配），
   **无法用原始 collector 重跑 ep1**。§P2 的「196/236/13/0.216 逐值一致」从一开始就不可能满足。
   须确认改为 v2 已写明的「对归档记录做确定性离线回放 + 在线仿真先测重跑波动再冻结容差」。
3. **all61 训练资格**：60 条全部 `training_eligible=false`。P4「首批 50 条 QC 合格数据」的起点是**零**，
   不能把 34 条 success 当作 34 条可训练数据。需确认 P4 是否必须全新采集。
4. **队列失败根因**：官方 1077 队列自行失败于 4/1077，`/tmp` 日志已丢失。是否值得追查（§A.3 已判定该线不复用，可能无需追查）。

---

## 主代理事后更正 2（2026-09-12）：队列失败原因的精确化

先前 P0-T1 验收记录与 `FOUNDATION_ARCHIVE.md` 第 9 条更正均表述为队列
「**自行失败**于 4/1077（`FAILED_NEEDS_REVIEW`）」。该措辞**不够准确**，据 `progress.json` 实测精确化为：

```
status    = FAILED_NEEDS_REVIEW
pid       = 779548
completed = 4
planned   = 1077
detail    = RuntimeError: GPU free below 2048 MiB
```

**确切原因：被 `gpu_wait` 的显存准入门禁主动中止**——共享 3090 上其他进程占用显存，
可用显存低于 2048 MiB 阈值，守护逻辑抛错退出。

这不是算法或模型失败，而是**资源争抢导致的中止**，性质上原则可恢复。
但 §A.3 已判定该线不复用（3 维无 stop 头 + constant-zero state 与新契约不兼容），恢复价值近零。

补充事实：`results.json` 的 4 条 `status` 均为 `COMPLETE`，episode ID 为 **1、2、3、7**（**非连续**）；
顶层 `success`/`spl`/`distance_to_goal` 字段均为 `null`（逐条指标在各 episode 目录内）。
ID 4/5/6 的去向无记录，`/tmp` 日志已丢失，记 MISSING。

**不变的结论**：P0 无可停止对象，未执行任何停止操作，4 条产物与三份 JSON 完整保留。

**待办**：`FOUNDATION_ARCHIVE.md` 第 9 条与 A.3 节的「failed itself / 自行失败」措辞须改为
「aborted by the gpu_wait VRAM admission gate (`GPU free below 2048 MiB`)」。
此项与 P1 的契约冻结一并派发，不单独为一句措辞再起一轮 codex。
