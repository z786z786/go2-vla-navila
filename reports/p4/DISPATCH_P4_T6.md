# 派发单 P4-T6 — phase 1 训练吞吐探针

授权：用户 2026-09-13 `CONTINUE P4′` 并确认派发本任务。波次：**W3（GPU，串行独占，全局唯一 GPU 任务）**。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P4-T6** |
| 资源 | **GPU 独占**，必须走 `gpu_wait` 准入与项目锁。CPU/内存不限。**无需联网**（见 §3.2） |
| 并发 | **不得与任何其他 GPU 任务并行**。当前无其他 GPU 任务在跑 |

## 1. 目标

为 v3.1 #12「先跑吞吐探针再定规模」提供实测依据。**这是测量任务，不是训练任务**——
不追求 loss 下降，不追求收敛，不产出可用 checkpoint。只回答：

1. 在 phase 1 的真实配置下，**一个训练 step 要多久、吃多少显存**？
2. **最大可用 batch size** 是多少（24 GB 卡，且须为共享场景留余量）？
3. **预计算并缓存冻结视觉特征**能带来多少加速？缓存要多大？
4. 由此推出：**353,894 条样本跑 1 个 epoch 需要多少墙钟小时**？

## 2. 已确立的事实（主代理实测，作为实现依据与自检基准）

### 2.1 硬件与环境

| 项 | 值 |
|---|---|
| GPU | RTX 3090，24,576 MiB，compute capability **8.6**，驱动 580.178.04 |
| 实测峰值 | **fp16 71.5 / bf16 72.8 / tf32 38.4 TFLOP/s**（8192³ matmul，30 次，含 warmup） |
| Python | `/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python` |
| 关键包 | torch 2.11.0+cu130、transformers 5.5.4、lerobot 0.6.0、**peft 0.20.0**、accelerate 1.14.0 |

### 2.2 **必须离线运行**

`huggingface.co` 从本机网段（202.199.13.0/24）**不可达**。运行前必须导出：

```bash
export HF_HOME=/mnt/wxh/go2_short_vln/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

本地缓存已有 `models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct` 与 `models--lerobot--smolvla_base`。
**不得尝试下载任何模型**；若报 `We couldn't connect to huggingface.co`，是上述变量没设对，不是缺文件。

### 2.3 **warmup 是强制要求**（否则吞吐数会低约 18 倍）

3090 空闲时处于 **P8 / 210 MHz**。主代理实测：无 warmup 的 fp16 matmul 量到 **3.9 TFLOP/s**，
加 warmup 后为 **71.5 TFLOP/s**，跑起来后 GPU 进入 P2 / 1890 MHz / 259 W。

**所有计时必须**：先跑 ≥10 个丢弃的 warmup step，每次计时前后 `torch.cuda.synchronize()`，
报告中给出 warmup step 数、计时 step 数、以及计时期间的 `nvidia-smi` 时钟/功耗采样。

### 2.4 模型结构（主代理实测）

| 项 | 值 |
|---|---|
| SmolVLM2-500M 基座 | **507.48 M** 参数 |
| LoRA（`q_proj`/`v_proj`, r=16, α=32） | **2.228 M 可训练（0.439%）** |
| `lm_head` | **47.31 M**，且 **`tie_word_embeddings = False`**（不与词嵌入共享，须单独计入可训练参数） |
| 模型上 GPU（bf16） | 1.05 GB |
| 每帧图像 token | **64**（vision `position_embedding [1024,768]` = 32×32 patch；`connector` 输入 12288 = 768×16 → 16 patch 合 1 token） |

### 2.5 数据

| 项 | 值 |
|---|---|
| 索引 | `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.jsonl`，**353,894 行**，SHA-256 `d894855183543d230a4802701a84aa3d430be9ad55b52d6dc5b6827829b7a09f` |
| 帧 | `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg`，**601,125 张，全部 512×512 RGB**，均值约 32 KB |
| 留出集 | `reports/p4/t2_scene_split.json`，按 scene 切，holdout 1,082 videos（10.0009%），scene/video 交集均为空 |
| 索引字段 | `video_id, video, step, scene_id, instruction_raw, instruction_normalized, action_text, action_id(0-9), n_frames, frames_first, frames_last, multiplicity, is_oversampled_copy` |
| 帧路径重建 | 每行的帧为 `<video>/frame_<i>.jpg`，`i ∈ [0, n_frames-1]`；索引**不存**完整帧列表 |

**过采样一律保留**（v3.1 #17）：353,894 条全用，不去重。

## 3. phase 1 的训练配置（按 v3 冻结值，照此测量）

| 项 | 值 | 依据 |
|---|---|---|
| 骨干 | SmolVLM2-500M（即 SmolVLA 的 VLM 部分） | v3.1 #10 单线 |
| 输出头 | **`lm_head` 自回归生成文本宏动作** | v3.1 #5 |
| 微调范围 | **LoRA 微调 VLM + 训 `lm_head`**，视觉编码器冻结 | v3.1 #6 |
| `tokenizer_max_length` | **160** | v3.1 #11（**这一项才解除截断**） |
| `pad_language_to` | **`longest`** | v3.1 #11（省算力，与截断无关） |
| history | **8 帧** → 512 image token | §3.1 |
| 监督目标 | 对 `action_text` 的 next-token 交叉熵（目标串约 10–12 token） | v3.1 #5 |
| 精度 | bf16（3090 上 bf16 与 fp16 峰值相当，且数值更稳） | §2.1 实测 |

**8 帧的抽取规则本任务不冻结**：history 规则的冻结属 P1′/P5。本任务只需固定一种做法、写明它，
并保证 8 帧这个数量正确——吞吐只取决于帧数，不取决于选哪 8 帧。

## 4. 任务

### 4.1 最小 Dataset

在 `datasets/navila_r2r.py`（新建）实现一个 PyTorch `Dataset`，读 §2.5 的索引，
按 `video` + `n_frames` 重建帧路径，返回 8 帧图像 + instruction + 目标 `action_text`。
要求：可被 `DataLoader` 正常 batch；`num_workers` 可配；**不得把 601,125 条帧路径全量驻留内存**。

### 4.2 吞吐测量（核心）

对下列配置各测一组，**每组都要 warmup**：

| 变量 | 取值 |
|---|---|
| batch size | 从 1 开始倍增直到 OOM，报告**最大可用值**与各档的 step 时间 |
| 视觉特征 | **(a) 在线编码**（每 step 前向视觉编码器）vs **(b) 预计算缓存**（视觉编码器冻结，特征可离线算一次） |
| `num_workers` | 至少测 2 档，报告数据加载是否为瓶颈 |

每组报告：step 时间（均值 / p50 / p95）、**显存峰值**（`torch.cuda.max_memory_allocated` 与 `nvidia-smi` 两者都报）、
samples/s、以及 GPU 利用率采样。

### 4.3 vision 缓存的可行性与代价

- 缓存一帧的特征体积（64 token × 960 dim × 2 字节 ≈ 123 KB，**须实测确认**）
- 601,125 帧全量缓存的总体积（`/mnt` 余 1.8 TB）
- 预计算一遍需要多少墙钟
- **明确记录一个限制**：缓存与 phase 2 的视觉侧微调不兼容（v3.1 #6 的 LoRA 只动 text 塔，故 phase 1 可用）

### 4.4 规模推算

由实测 samples/s 推出：353,894 条样本 × {1, 2, 3} epoch 各需多少墙钟小时，两种视觉路径分列。
**推算必须给出算式与所用实测值**，不得只给结论数字。

## 5. GPU 准入协议（强制）

```bash
# 1) 先等准入（默认 EXCLUSIVE 策略：20 GB 空闲 / 连续 3 次采样 / 30 s 间隔）
/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python src/dual_target/gpu_wait.py \
    --project-root /home/wxh/go2_short_vln --run-id p4_t6_throughput
# 2) runner 内部再调 acquire_live_admission(project_root, run_id="p4_t6_throughput", actual_argv=...)
```

- **不得**用 `--required-free-mib` / `--required-samples` / `--interval-s` 覆盖阈值（那些是 test-only）。
- **不得**用 `--test-snapshots`（注入快照永远不能准入真实运行）。
- 运行期间若 `nvidia-smi` 空闲显存跌破 **2,048 MiB**，立即停止并记录——这正是官方 1077 队列当初的死因。
- 任务结束**必须释放锁**（`lease.close()`），不得留下悬挂锁。

## 6. 输出

| 路径 | 内容 |
|---|---|
| `datasets/navila_r2r.py` | §4.1 的 Dataset |
| `scripts/p4_throughput_probe.py` | 可复算探针，**无参数可重跑** |
| `reports/p4/t6_throughput.json` | 全部实测数字；含 `task_id`/`generated_at`/`generator_command`/`inputs`(路径+sha256)/`gpu_admission`(run_id、准入快照)/`warmup`(step 数、时钟采样) |
| `reports/p4/T6_THROUGHPUT.md` | 人读摘要 + §4.4 的推算算式 |
| `tests/datasets/test_navila_r2r.py` | Dataset 的最小单测（CPU 可跑：索引解析、帧路径重建、batch 形状） |

## 7. 验收标准（主代理如何独立复算）

1. 主代理读代码断言：`tokenizer_max_length == 160`、`pad_language_to == "longest"`、LoRA 目标模块与 r 值、视觉编码器确为冻结、`lm_head` 在可训练参数中。
2. 主代理独立重跑 `scripts/p4_throughput_probe.py` 的**最小配置**（batch=1，少量 step），核对 step 时间与显存量级一致（±30%，因 GPU 状态有波动）。
3. 主代理独立重跑 `tests/datasets/test_navila_r2r.py`，并**自己另抽 20 条记录**核对帧路径重建与索引一致。
4. 主代理核对 §4.4 的推算：用报告给出的 samples/s 独立重算小时数，须一致。
5. 主代理断言报告含 warmup 证据（step 数 + 时钟采样），且**未 warmup 的数字没有被当作结论**。
6. 主代理核对 `gpu_admission` 段：`run_id` 正确、准入来源为 `live_nvidia_smi`、策略为 EXCLUSIVE、锁已释放。
7. 主代理核对每帧缓存体积的**实测值**与 123 KB 估算的偏差，并核对全量缓存体积推算。

**子代理的 `passed`、测试通过数、以及任何未 warmup 的计时均不构成 PASS 依据。**

## 8. 禁止事项

1. **不得联网**；不得尝试下载模型或数据。
2. 不得绕过 `gpu_wait`；不得覆盖其阈值；不得使用注入快照。
3. 不得与其他 GPU 任务并行；结束必须释放锁。
4. **不得报告未经 warmup 的吞吐数字**作为结论（可作为对照列出，须标明）。
5. 不得去重索引记录（过采样保留）。
6. 不得改动 `actions/`、`reports/p3/`、`reports/p4/t1_*`、`reports/p4/t2_*`、`reports/p4/t4_*`、
   `configs/`、`.gitignore`、`CODEX_VLN_PLATFORM_MILESTONES.md`、既有 `src/`。
7. 不得 `git add` / `git commit` / `git tag`。
8. 不得训练到收敛或产出声称可用的 checkpoint——**本任务只测吞吐**。
9. 证据缺失标记 `MISSING`，不得自行追认 PASS。
10. 大产物（vision 缓存等）放 `/mnt`，不得写入 `/home`。

## 9. 验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| — | — | 待派发 | — |
