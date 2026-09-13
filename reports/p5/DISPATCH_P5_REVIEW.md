# 派发单 P5-REVIEW — phase 1 正式训练前的整体代码审查

授权：用户 2026-09-13（「开跑前用 codex 的 gpt6 mid 整体 review 一下」）。模型/强度取 `~/.codex/config.toml` 默认
（`gpt-6-astra` / `medium`），不另行覆盖。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P5-REVIEW** |
| 性质 | **只读审查**。可以读任何文件、可以在 CPU 上运行单测和小脚本做核实；**不得修改仓库文件**，不得占用 GPU |
| 产出 | 仅写 `reports/p5/P5_REVIEW.md` |
| 背景 | 正式训练约 6–7 h/epoch，跑错一次代价很大。审查的价值在于**找出会让训练白跑的问题**，不是风格意见 |

## 1. 审查范围

| 文件 | 角色 |
|---|---|
| `scripts/p5_phase1_train.py` | 正式训练脚本（由已验收的 `scripts/p4_train_throughput.py` 演化而来） |
| `src/smolvla/phase1_prompt.py` | **训练与评测共用**的输入契约：prompt 格式、答案 token、8 帧历史规则 |
| `datasets/navila_r2r.py` | 数据集；本轮把历史取帧规则改为 NaVILA 原生 |
| `tests/smolvla/test_phase1_prompt.py`、`tests/datasets/test_navila_r2r.py` | 单测 |
| `reports/p4/t2_scene_split.json` | 训练 / 留出 split（只读，核对用法） |

## 2. 主代理在改写中已自行发现并修正的问题（请**核实修正是否正确**，不要只复述）

1. **prompt 格式**：吞吐探针直接拼 `[512 个视觉嵌入][指令]\n[动作]`，无图像分隔符、无 chat 模板、无结束 token。
   现改为 SmolVLM2 自身 chat 模板 + 每帧单图块 + NaVILA 原生问句（`NaVILA-Bench/scripts/vlm_server.py:126-131`），
   答案以 `<end_of_utterance>` 结尾。
2. **历史取帧规则**：原数据集用四舍五入均匀采样、短历史重复帧；NaVILA 训练（`llava/mm_utils.py`
   `vlnce_frame_sampling`，GitHub AnjieCheng/NaVILA main）与评测（`NaVILA-Bench/scripts/navila_eval.py:157-173`）
   都是**前补黑帧 + 7 帧向下取整 + 最后一帧**。全量索引上 87.8% 记录的取帧集合因此改变。
3. **精度**：探针以 bf16 存放 `lm_head` 并直接用 AdamW 更新（bf16 仅 7 位尾数），更新大量被舍入。现改为可训练参数
   fp32 主权重 + bf16 autocast。
4. **远端索引**：远端曾是缺 `scene_id`/`action_id` 的临时重建版，已换成权威索引（SHA-256 `d8948551…a09f`）。

## 3. 必须逐项给出结论的检查点

每项给 **成立 / 不成立 / 无法判断**，附代码行号与证据（能在 CPU 上跑的就跑，贴输出）。

A. **监督对象**：loss 是否恰好覆盖答案 token（含 `<end_of_utterance>`）、不含 prompt 与 padding；
   `sup[:, 1:]` 与 `last_hidden_state[:, :-1]` 的错位是否正确。
B. **视觉特征对齐**：`masked_scatter` 的填充顺序是否与 `<image>` 占位的顺序（batch→帧→token）一致；
   与 transformers 5.5.4 `SmolVLMModel.inputs_merger` 语义是否等价；像素归一化 `x/127.5-1` 是否等于处理器的
   rescale+normalize；不走图像切分是否合理。
C. **训练/评测一致性**：训练 prompt 与将来评测时构造的 prompt 是否可能出现任何一个 token 的差异（空白、引号、
   BPE 边界、截断）；`navila_history_layout` 与 NaVILA 两处源代码是否逐位一致（含 n<8 与大 n）。
D. **精度**：autocast 下 PEFT LoRA（fp32）与冻结 bf16 权重的混合是否有隐性 dtype 问题；fp32 主权重是否真的生效；
   `clip_grad_norm_` 返回的是否为裁剪前范数。
E. **优化器/调度**：两组学习率（LoRA 1e-4、lm_head 2e-5）、warmup 3% + cosine 到 0、AdamW wd=0 的实现是否正确；
   `LambdaLR` 的步进时机。
F. **断点续训**：采样器续跑是否既不重复也不遗漏样本；optimizer/scheduler/RNG 状态是否完整恢复；
   参数顺序校验是否足以防止 optimizer state 错位；原子写入是否真正原子；SIGTERM 路径是否会存下一致的 checkpoint；
   `eval_history.json` 与 TensorBoard `purge_step` 在续跑时的行为。
G. **留出集评测**：teacher-forced exact match 与贪心解码复现目标的等价性论证是否成立；乱指令对照（批内滚动 + 跳过同指令行）
   是否真的测到了「模型是否使用指令」；per-class 统计是否正确。
H. **数据**：train/holdout 由 `exclude_videos` 构造是否与 split 文件一致；过采样是否保留；DataLoader worker 中
   tokenizer 的 fork 安全。
I. **会让 6 小时训练白跑或静默出错的任何其他问题**（显存峰值、NaN、磁盘、日志丢失……）。

## 4. 需要用户决策、请给出**你的独立意见**的设计问题（主代理不预设答案）

1. **32 层 vs 16 层**：phase 1 训练完整 32 层 SmolVLM2 + `lm_head`；而 `smolvla_base` 的 VLM 只用前 16 层
   （`models/smolvla_base_c83c316/config.json` `num_vlm_layers: 16`）。这对 v3.1 #6「保护 phase 2 expert 兼容性」意味着什么？
2. **学习率与 epoch 数**：在 318,331 条（含过采样）、bs16、约 19.9k step/epoch 的条件下，1 epoch 是否足够或过多。
3. **prompt 中保留 NaVILA 问句里的 `<image>\n.` 这类原生怪异写法**是否合理。

## 5. 报告格式（`reports/p5/P5_REVIEW.md`）

1. 结论先行：**阻塞正式训练的问题**（若无，写「无」）。
2. §3 A–I 逐项结论表。
3. §4 三个设计问题的独立意见。
4. 非阻塞建议（简短）。
5. 你实际运行过的命令与输出；你没能验证的项标 `MISSING`。
