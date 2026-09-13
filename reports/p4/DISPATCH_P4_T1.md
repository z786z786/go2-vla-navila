# 派发单 P4-T1 — 解包 NaVILA R2R train.tar.gz

授权：用户 2026-09-13 给出 `CONTINUE P4′`，并裁定采用**解包**方案（替代方案「对 tar 建偏移索引」未采纳）。
波次：W1（与 P4-T2、P4-T3 并发）。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P4-T1** |
| 资源 | **CPU + 磁盘 I/O。零 GPU，零网络。** 不得走 `gpu_wait` |
| 并发 | 与 P4-T2 / P4-T3 并发。三者不写同一批文件、不依赖彼此产出 |

## 1. 目标

把已下载校验通过的 `R2R_train.tar.gz` 解包到本地数据目录，并产出**可独立复算**的解包清单。

## 2. 输入

| 路径 | SHA-256 | 字节 |
|---|---|---|
| `/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_train.tar.gz` | `49ffc4f88e15ae4bdd2a8f01b16a8685deee0bd4d05711998d9ae88a99d4d228` | 19703787520 |

**开工前必须 `sha256sum` + `stat -c%s` 核对，不符即停并交诊断。**

**该文件是未压缩的纯 POSIX tar（GNU）**，尽管命名 `.tar.gz`。用 `tar -xf`，**不要**加 `-z`；
任何 gzip 相关校验都会失败且不构成损坏证据。

已知事实（主代理实测，可用于自检）：`tar -tf` 给出 **611,945** 条目
= **601,125** 个 `.jpg` + **10,819** 个 video 目录 + 1 个根目录 `train/`；帧为 **512×512** JPEG，均值约 32.0 KB。

## 3. 任务

1. 解包到 `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/`（即最终路径形如
   `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg`）。
   目标分区 `/mnt` 余 1.8 TB、2.42 亿 inode，容量与 inode 均充足。
2. **隔离同目录下的陷阱文件**：`/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/annotations.json`
   是 **184,991,744 字节的截断副本**（完整版是 `downloads/navila_probe/R2R_annotations.json`，317,267,756 字节）。
   将其**重命名**为 `annotations.INCOMPLETE-184991744B.json.quarantine` 并在清单中记录此操作。
   **不得删除**，不得覆盖，不得用它做任何统计。
3. 产出解包清单 JSON：每个 video 目录的帧数与帧索引连续性、总文件数、总字节数、
   以及**确定性抽样的 64 个文件的 SHA-256**（抽样规则须写死且可复现，例如按排序后的文件列表等距取样）。
4. 自检：解包后的文件集合与 `tar -tf` 列表**完全一致**（无缺失、无多余）。

## 4. 输出

| 路径 | 内容 |
|---|---|
| `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/` | 解包后的 601,125 帧 / 10,819 目录 |
| `/home/wxh/go2_short_vln/reports/p4/t1_extract_manifest.json` | §3.3 的清单；含 `task_id`/`generated_at`/`generator_command`/`inputs`（实测 sha256 + 字节） |
| `/home/wxh/go2_short_vln/reports/p4/T1_EXTRACT.md` | 人读摘要：计数、耗时、占用、抽样哈希表、隔离操作记录 |
| `/home/wxh/go2_short_vln/scripts/p4_extract_navila.py` 或 `.sh` | 可复算脚本，无参数重跑（对已存在的完整解包应为幂等或显式跳过） |

**清单 JSON 必须小**（< 1 MB）。逐文件哈希**不要**全量入库；只存 64 个抽样。
逐 video 帧数表若过大，改存分位数 + 异常项，并把完整表写到
`/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t1_per_video_frames.json`。

## 5. 验收标准（主代理如何独立复算）

1. 主代理独立 `find` 计数解包目录的 `.jpg` 数与子目录数，须为 **601,125 / 10,819**。
2. 主代理独立生成 `tar -tf` 列表，与解包结果做**双向差集**，须均为空。
3. 主代理按清单写死的抽样规则重算那 64 个 SHA-256，须逐条一致。
4. 主代理抽 5 个文件，用 `tar -xO` 从原始 tar 直接读出对应字节流，与磁盘文件 `cmp`，须一致。
5. 主代理核对 `annotations.INCOMPLETE-184991744B.json.quarantine` 存在且大小仍为 184,991,744。
6. 清单 `inputs.sha256` 与 §2 逐字一致。

**子代理的 `passed` 字段与自我结论不构成 PASS 依据。**

## 6. 禁止事项

1. 不得删除或改名 `R2R_train.tar.gz`（解包后仍保留，它是可复算的源）。
2. 不得删除 `annotations.json` 截断副本——只能按 §3.2 改名隔离。
3. 不得写入 `/home` 下除 §4 三个路径以外的任何文件。**特别不得动**
   `CODEX_VLN_PLATFORM_MILESTONES.md`、`reports/p3/`、`configs/`、`.gitignore`、`actions/`、`tests/`。
4. 不得 `git add` / `git commit` / `git tag`。
5. 不得占用 GPU，不得联网。
6. 不得触碰 P4-T2 / P4-T3 的输出路径（`reports/p4/t2_*`、`reports/p4/t3_*`、`actions/`、`tests/`）。
7. 证据缺失标记 `MISSING`，不得自行追认 PASS。

## 7. 验收台账

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
| 1 | 2026-09-13 | **BLOCKED** | codex 沙箱结构性阻塞；归因错误已由主代理更正；子代理未自我追认，行为正确 |
| 2 | 2026-09-13 | **PASS** | 子代理于约 10:42–10:50 完成解包（帧 mtime 保留为归档原值 2024-07-12 可证）。主代理在例外 E4 下的 `tar -xf` 为冗余重跑。验收：601,125 jpg / 10,819 目录、与 `tar -tf` 双向差集 0/0、64 抽样哈希、`tar -xOf` 逐字节 5/5、隔离文件 184,991,744 B。详见 `CODEX_VLN_PLATFORM_MILESTONES.md` §v3.13 |

**本轮的调度教训（已写入调度协议，见 §v3.13-D）**：纯 I/O 零决策步骤不再派发；
不得催问运行中的 turn（主代理 10:40 的状态查询很可能干扰了执行）；
监视器的 10 分钟停滞阈值对 xhigh 推理过于敏感，不构成异常信号。
