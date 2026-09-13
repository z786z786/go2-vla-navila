# P4-T2 — NaVILA R2R 逐记录索引与 scene 留出集

## 结果摘要

| 项目 | 值 |
|---|---:|
| 索引记录行数 | 353894 |
| 不同 video | 10819 |
| 不同 `video_id`（含 step） | 288594 |
| scene | 61 |
| 索引状态 | 已落盘 |
| 索引路径 | `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.jsonl` |
| 索引字节数 | 222781783 |
| 索引行数 | 353894 |
| 索引 SHA-256 | `d894855183543d230a4802701a84aa3d430be9ad55b52d6dc5b6827829b7a09f` |
| SHA-256 sidecar | `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.sha256` |
| `/mnt` 写权限探针 | PASS（写入并删除已核验） |
| 全量 majority action | `The next action is move forward 75 cm.` |
| 全量 majority 占比 | 29.959818% |
| 全量 stop 占比 | 9.171390% |
| 过采样 `video_id` 重数（重复记录组） | `{'1': 234113, '2': 43662, '3': 10819}` |

输入在解析前已核对 SHA-256；未读取禁止使用的截断 annotations 副本，也未读取 T1 解包帧文件。

## 全量动作分布

| action_text | count |
|---|---:|
| `I think I should stop because I have finished the instruction.` | 32457 |
| `The next action is move forward 25 cm.` | 37376 |
| `The next action is move forward 50 cm.` | 28047 |
| `The next action is move forward 75 cm.` | 106026 |
| `The next action is turn left 15 degree.` | 31465 |
| `The next action is turn left 30 degree.` | 15348 |
| `The next action is turn left 45 degree.` | 31510 |
| `The next action is turn right 15 degree.` | 31199 |
| `The next action is turn right 30 degree.` | 14436 |
| `The next action is turn right 45 degree.` | 26030 |

`action_id` 映射在 `t2_index_summary.json` 中显式声明（0–9，固定词典顺序）。

## Scene 切分

固定 seed 为 `20260913`。按整 scene 切分，目标留出 video 数为
`1082`（全量 10% 的最近整数）；使用 seeded whole-scene subset-sum，
实际留出 `1082` 个 video（10.000924%）。
训练/留出 scene 与 video 交集均为空。

| 侧 | records | videos | scenes | majority 占比 |
|---|---:|---:|---:|---:|
| train | 318331 | 9737 | 56 | 30.118650% |
| holdout | 35563 | 1082 | 5 | 28.538087% |

### 稀疏 scene（<25 videos）逐项处置

| scene stem | videos | 归属 | 理由 |
|---|---:|---|---|
| `gZ6f7yhEvPG` | 3 | `train` | fewer than 25 videos; keep the entire tiny scene in training so a single sparse scene does not dominate the holdout baseline |
| `YmJkqBEsHnH` | 9 | `train` | fewer than 25 videos; keep the entire tiny scene in training so a single sparse scene does not dominate the holdout baseline |
| `HxpKQynjfin` | 15 | `train` | fewer than 25 videos; keep the entire tiny scene in training so a single sparse scene does not dominate the holdout baseline |
| `XcA2TqTSSAj` | 18 | `train` | fewer than 25 videos; keep the entire tiny scene in training so a single sparse scene does not dominate the holdout baseline |
| `GdvgFV5R1Z5` | 21 | `train` | fewer than 25 videos; keep the entire tiny scene in training so a single sparse scene does not dominate the holdout baseline |
| `Pm6F8kyY3z2` | 24 | `train` | fewer than 25 videos; keep the entire tiny scene in training so a single sparse scene does not dominate the holdout baseline |

六个按字面 `<25 videos` 判定的稀疏 scene 全部固定在训练侧（其中派发单重点列出的五个最稀疏 scene 均包含在内），避免留出侧被单一小 scene 主导；机器可读的完整 scene ID、两侧 scene/video 清单和动作分布见 `t2_scene_split.json`。

## 帧路径重建

索引只保存 `n_frames`、`frames_first`、`frames_last`，不保存 601,125 条完整帧路径。对任意记录，原始标注路径为
`<video>/frame_<i>.jpg`，其中 `i = 0, ..., n_frames-1`；解包后的确定性绝对路径为
`/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg`。

## 复算与缺口

无参数重跑命令：`python3 /home/wxh/go2_short_vln/scripts/p4_build_navila_index.py`。

MISSING: /mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg existence — MISSING (not checked by P4-T2 because the dispatch assigns frame existence to P4-T1 and forbids reading extracted frames)
