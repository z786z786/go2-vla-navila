# 派发单 P4-T6v2 — phase 1 训练吞吐（**真实训练**，非合成探针）

授权：用户 2026-09-13。取代 P4-T6 第 1 轮（主代理判 **FAIL**，见 `CODEX_VLN_PLATFORM_MILESTONES.md` §v3.17-A）。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P4-T6v2** |
| 资源 | 你（子代理）**只做 CPU 部分**；GPU 部分由主代理执行，见 §2 |
| 并发 | 与 P1-DEV100 并发，写入路径不重叠 |

## 1. 上一轮为何 FAIL —— 这四条是本轮的核心约束

主代理读代码发现 `scripts/p4_throughput_probe.py` 存在四处实质缺陷，**本轮必须全部避免**：

| # | 上一轮的错误 | 本轮要求 |
|---|---|---|
| 1 | **全脚本 `backward` 出现 0 次**，计时段整体在 `torch.no_grad()` 内——**训练**探针只测了**前向** | **必须**是完整训练 step：forward → loss → `backward()` → `optimizer.step()` → `zero_grad()`。**计时段内禁止出现 `torch.no_grad()`** |
| 2 | `text_embeds = torch.randn(...)` 冒充文本嵌入；全脚本**无任何 tokenizer 调用** | **必须**用真实 tokenizer 对 `instruction_raw` 编码。**禁止任何合成/随机张量替代真实输入** |
| 3 | 序列长度硬编码 `512+160`，`pad_language_to="longest"` 等于未测 | 序列长度必须由**真实 batch** 决定。须报告实测的文本 token 长度分布 |
| 4 | 视觉侧取 `pooler_output`，非真正喂入 connector 的**每帧 64 token 序列** | 必须走真实的视觉→connector 路径 |

另：上一轮 `tests/datasets/` 缺 `__init__.py`、单测用 pytest fixture 而环境无 pytest，
**`unittest discover` 实测「Ran 0 tests」——报告的测试从未执行过**。本轮单测**必须真的跑起来并全绿**。

## 2. 分工（**与上一轮不同，务必看清**）

codex 子代理的沙箱**没有 GPU**（`cudaGetDeviceCount` 返回 rc=100；子代理位于独立 PID 命名空间的容器内，
`Pid: 3`）。已尝试放行设备节点并失败回滚（§v3.18）。因此：

| 部分 | 归属 | 要求 |
|---|---|---|
| `datasets/navila_r2r.py` + 单测 | **你** | **纯 CPU，你必须真的跑通并自验**。不得交付未执行过的测试 |
| 训练脚本（含 GPU 计时段） | **你写，主代理跑** | 你无法执行它。**必须显式标记为未测试**，并在报告中列出你无法验证的每一项 |
| 执行与验收 | **主代理** | 主代理逐行审代码后在真实 GPU 上运行 |

**因此代码的可读性与自证性至关重要**：主代理要能只靠读代码就确认它测的是对的东西。

## 3. 核心设计：跑一小段**真实训练**，不做合成探针

用户 2026-09-13 裁定：**不再做合成探针，直接跑一小段真实训练来测吞吐。**

理由（记录在案）：合成探针可以在不报错的情况下测错东西——`torch.randn` 冒充嵌入、`no_grad` 里测「训练」吞吐，
两者都不会失败，只会给出漂亮的错数字；而**让一个跑不了自己代码的作者写 GPU 代码，必然产出这种东西**。
真实训练成本相近，但：step 时间与显存是**真的**；白送一条 loss 曲线；**loss 不动即说明有东西坏了**——
合成探针没有这种自检。

要求：

1. 默认 **200 个真实训练 step**（可配），在 `t2_records.jsonl` 的真实数据上。
2. **记录并输出 loss 曲线**（每 step 或每 N step）。报告须明确写出 loss 是否下降。
   **loss 不下降必须如实报告为异常信号，不得隐去。**
3. 计时：**前 ≥10 step 作为 warmup 丢弃**，之后计时。每次计时前后 `torch.cuda.synchronize()`。
   （3090 空闲于 P8/210 MHz；主代理实测无 warmup 的 fp16 matmul 为 **3.9 TFLOP/s**、有 warmup 为 **71.5**，**差 18 倍**。）
4. batch size 从 1 倍增至 OOM，报告**最大可用值**与各档 step 时间/显存。
5. 两条视觉路径对比：**在线编码** vs **预计算冻结特征缓存**。
6. `num_workers` 至少两档，报告数据加载是否为瓶颈。

## 4. 训练配置（v3 冻结值，照此实现）

| 项 | 值 |
|---|---|
| 骨干 | SmolVLM2-500M（**507.48 M** 参数，主代理实测） |
| 微调 | **LoRA**(`q_proj`/`v_proj`, r=16, α=32) 套在 **text 塔** → **1.638 M 可训练**；**+ 训 `lm_head`**（**47.31 M**，`tie_word_embeddings=False`，须单独计入）。**合计 48.947 M** |

> **数值更正（2026-09-13）**：本表原写 LoRA「2.228 M」，**是主代理的错误**。该值来自对**整个模型**套 LoRA 的冒烟测试——而视觉塔里**也有** `q_proj`/`v_proj`，套上去等于训练本该冻结的部分。LoRA 只应套在 text 塔，实测为 **1.638 M**，可训练总计 **48.947 M**。由代码审查发现脚本与本表不一致后追查确认；**错的是本表，不是脚本**。
| 视觉编码器 | **冻结** |
| `tokenizer_max_length` | **160** |
| `pad_language_to` | **`longest`** |
| history | **8 帧**，每帧 **64** image token |
| 精度 | **bf16** |
| 目标 | 对 `action_text` 的 next-token 交叉熵 |
| 优化器 | AdamW（lr 等参数自定并写明——本任务不调参，只需稳定可跑） |

**训练目标的构造建议复用 `src/smolvla/p5_contract.py`**：它委托 `actions.language_parser` 并强制
`classification == "exact"`，主代理已验证 100/100 通过。

## 5. 环境（**必须离线**）

```bash
export HF_HOME=/mnt/wxh/go2_short_vln/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

`huggingface.co` 从本机不可达；本地缓存已有 SmolVLM2-500M 与 smolvla_base。
报 `couldn't connect to huggingface.co` 是变量没设对，不是缺文件。
Python：`/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python`（peft 0.20.0、torch 2.11.0+cu130 已装）。

## 6. 数据

| 项 | 值 |
|---|---|
| 索引 | `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t2_records.jsonl`，**353,894 行**，sha256 `d894855183543d230a4802701a84aa3d430be9ad55b52d6dc5b6827829b7a09f` |
| 帧 | `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<video>/frame_<i>.jpg`，**601,125 张，全 512×512 RGB** |
| 留出集 | `reports/p4/t2_scene_split.json`（按 scene 切，holdout 1,082 videos） |
| 过采样 | **保留，不得去重**（v3.1 #17） |

## 7. 输出

| 路径 | 内容 |
|---|---|
| `datasets/navila_r2r.py` | Dataset（**你必须跑通**） |
| `tests/datasets/test_navila_r2r.py` + `tests/datasets/__init__.py` | 单测，**必须真的执行并全绿**，报告中给出运行命令与实际输出 |
| `scripts/p4_train_throughput.py` | 真实训练脚本，**无参数可重跑**；显式标记 GPU 段未经你测试 |
| `reports/p4/T6V2_PLAN.md` | 你能验证什么、**不能**验证什么，逐条列出；主代理执行前的审查要点 |

**`reports/p4/t6_throughput.json` 由主代理执行后产出，你不要创建它。**

## 8. 验收标准（主代理如何独立复算）

1. **逐行审 GPU 段**：断言计时段内**无 `torch.no_grad()`**、**有 `backward()` 与 `optimizer.step()`**、
   **有真实 tokenizer 调用**、**无随机张量替代真实输入**、序列长度来自真实 batch、视觉走 connector 路径。
   **任一条不满足即 FAIL。**
2. 主代理独立重跑你的单测，须真的执行且全绿（不接受 `Ran 0 tests`）。
3. 主代理另抽 20 条记录核对帧路径重建与索引一致。
4. 主代理读代码断言 §4 的每一个冻结配置值。
5. 主代理在真实 GPU 上执行训练脚本，核对 **loss 是否下降**；loss 不动则判定为仪器或数据管线有问题。
6. 主代理核对 warmup 证据与时钟采样。

## 9. 禁止事项

1. **计时段内禁止 `torch.no_grad()`**；**禁止用随机/合成张量替代真实输入**；**禁止硬编码序列长度**。
2. **禁止交付未执行过的测试**。你跑不了的部分必须显式标注，不得含糊带过。
3. 不得尝试访问 GPU、不得跑 `gpu_wait`、不得创建 `reports/p4/t6_throughput.json`。
4. 不得改动 `actions/`、`src/smolvla/p5_contract.py`、`reports/p3/`、`reports/p4/t1_*`/`t2_*`/`t4_*`、
   `configs/`、`.gitignore`、里程碑文档、既有 `src/`。
5. 不得去重索引记录。不得 `git` 任何操作。不得联网。
6. 证据缺失标记 `MISSING`，不得自行追认 PASS。

## 10. 验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| 1（P4-T6） | 2026-09-13 | **FAIL** | 仪器缺陷四处；单测从未执行。见 §v3.17-A |
| 2（本单） | — | 待验收 | — |
