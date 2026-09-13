# P5 phase 1 pre-flight review

## 结论先行：阻塞项

有一个会使“保护 phase 2 expert 兼容性”目标失效的设计阻塞：脚本明确训练完整 VLM，而 `smolvla_base` 只保留 16 层（见设计问题 1）。若 phase 2 仍按该 base 结构加载，后 16 层参数没有对应位置，不能把该 run 当作兼容的 phase-2 初始化。除此之外，没有发现已由代码证据确认会让本次 1 epoch 训练白跑的实现错误；视觉预处理与真实 `transformers` 语义、以及完整 GPU 续训状态仍有 MISSING 项，正式开跑前应补证。

## §3 检查点

|项|结论|证据|
|---|---|---|
|A 监督对象|成立|`scripts/p5_phase1_train.py:151-165` 将序列分成 prompt 与 target，并只将 `sup[st:len(s)]` 置真；`132-135` 用 `sup[:,1:]` 配 `last_hidden_state[:,:-1]`，labels 为 `ids[:,1:]`，因此覆盖首个答案 token 至含 EOU，不覆盖 prompt/pad。对应单测 `tests/smolvla/test_phase1_prompt.py:93-106` 明确比较 target。|
|B 视觉特征对齐|无法判断|`scripts/p5_phase1_train.py:123-130` 以 batch-major、frame-major flatten 后 `masked_scatter`；`phase1_prompt.py:120-131` 生成每帧 64 个连续 `<image>` 槽，顺序在本地逻辑上相符。`preprocessor_config.json` 显示 `do_image_splitting:true`、resize/max-size 规则；脚本却在 `123` 直接做 `uint8/127.5-1`，不执行 processor resize/splitting。当前环境没有 transformers，未能逐位执行 `SmolVLMModel.inputs_merger` 或确认 vision 输入尺寸/位置编码等价性，标 MISSING。|
|C 训练/评测一致性|成立（共享构造）；历史规则逐位逻辑成立|训练 collate `scripts/p5_phase1_train.py:153-155` 与 greedy `240` 都调用同一 `Phase1PromptBuilder`；builder `139-152` 统一 strip、160-token 截断、EOU。`navila_history_layout` `phase1_prompt.py:83-88` 是前补黑帧、7 个 floor 索引加末帧；数据集 `datasets/navila_r2r.py:75-90` 直接调用它。外部 NaVILA 源码逐位复核未在本机运行，故“与外部实现”的独立复现为 MISSING；共享实现本身无分叉。|
|D 精度|成立（实现层面）|`scripts/p5_phase1_train.py:80-101` 先以 bf16 加载、仅打开 LoRA/lm_head，再将所有可训练参数转 fp32并断言；`485-486` bf16 autocast。`491` 的 `clip_grad_norm_` 返回值写入 `gradient_norm_before_clip`，PyTorch API 返回裁剪前总范数。未用 GradScaler 对 bf16 是合理的。实际远端 dtype/权重变化采用题设已给证据（lm_head fp32，99.75% 改变）。|
|E 优化器/调度|成立|默认值 `351-355` 为 LoRA 1e-4、head 2e-5、wd=0、warmup 3%；`390-393` 两 param groups + AdamW + LambdaLR。`319-324` warmup 后 cosine 至 0；`492` 明确 `opt.step(); sched.step()`，每个 optimizer step 后推进。注意首步实际使用 LambdaLR 初始化后的 warmup 缩放，单测 `tests/smolvla/test_phase1_prompt.py:123-130` 覆盖函数边界。|
|F 断点续训|成立（代码路径）；完整运行证据部分 MISSING|`172-185` 每 epoch 固定 seed+epoch permutation，`476-480` 以 global step 计算 epoch/offset，避免重复/遗漏（drop_last 的尾部按设计不训练）。`281-313` 保存 trainable weights、optimizer、scheduler、torch/CUDA/python RNG，并在 `304` 校验 trainable name 顺序；`295` 目录 rename 提供目录级原子切换。SIGTERM 在 `66-69` 设置 STOP，当前 batch 完成后 `512-518` 保存并退出。`414` 使用 `purge_step`；`427-429` 恢复并截断 eval history，修复了题设所述丢历史问题。MISSING：未在本机执行 CUDA/SIGTERM/optimizer-state 恢复；不能独立证明跨进程 CUDA RNG 与远端实际一致。|
|G 留出集评测|部分成立|`203-227` teacher-forced exact match 的定义正确：每个答案位置 argmax 全对等价于在正确前缀条件下 greedy 复现；`230-272` 又做真实 autoregressive 检查并报告 agreement。乱指令在 `147-150` 批内 roll，并跳过相同字符串行（`keep`），统计 `214-227` 使用 keep；per-class 用 action_id index_add，逻辑正确。限制：greedy 只抽 `greedy-records`（默认 32），不能证明全 holdout greedy；此外 shuffle control 改的是指令但不改变视觉/答案配对，作为“使用指令”的检验是合理但统计功效有限。|
|H 数据|成立|`load_splits` `189-197` 从 split 文件取 video 集，互斥断言并断言记录数 318331/35563；`datasets/navila_r2r.py:36-45` 按 video 排除，保留 JSONL 每行（含过采样）。`40-42` 设置 `TOKENIZERS_PARALLELISM=false`，数据集 `57-73` fork 后按 pid 重开句柄。测试文件覆盖重复、过滤、pickle。实际权威索引 SHA 与远端文件未在本机重新 hash，标该独立核验 MISSING。|
|I 其他静默/白跑风险|成立（无已证阻塞）；有待监控项|loss 非有限直接在 `487-489` 抛错且保留上一 checkpoint；SIGTERM 路径有 checkpoint；保存滚动保留 `keep_last` (`296-297`)。日志/评估写入不是原子写，`460` 中断可能留下半个 JSON；不过下次 resume 只在 `step and hist_path.exists()` 读取，损坏文件会直接失败而非静默继续。显存峰值只记录 `502`，没有 CPU 磁盘空间预检；这些是运维风险，需运行时监控，不能从静态代码判定会发生。|

## §2 四项修正的独立核验

1. **Prompt 修正：正确且已应用。** `phase1_prompt.py:123-129` 明确构造 `IM_START`、每帧 fake/global/64 image block、EOU 与 Assistant；`scripts/p5_phase1_train.py:153-155` 和 `240` 都经 builder，故训练/greedy 不再走探针的裸 embedding 拼接。单测 `tests/smolvla/test_phase1_prompt.py:47-71` 检查 8 个块、chat 起始、用户 EOU 和答案 EOU。
2. **历史取帧修正：正确且已应用。** `phase1_prompt.py:83-88` 对 n<8 返回前置 None，对 n>=8 使用 `i*(m-1)//7` 加末帧；`datasets/navila_r2r.py:85-90,113-118` 将 None 变成黑帧并从最后一帧取当前观测。测试 `tests/datasets/test_navila_r2r.py:20-23,52-57,68-74` 覆盖 1、3、8、9、49 帧和 padding。
3. **精度修正：正确且已应用。** `p5_phase1_train.py:80-101` 的可训练参数全部 fp32 断言，冻结权重保持 bf16；`485-486` 只在 forward 使用 bf16 autocast；optimizer 在 `390-392` 更新 fp32 master 参数。题设远端验证进一步观测到 lm_head fp32 且 99.75% 元素变化，独立支持修正生效。
4. **权威索引修正：代码已应用，hash 独立核验 MISSING。** `scripts/p5_phase1_train.py:50-56,189-197` 默认指向 `data/navila_dataset/R2R/index/t2_records.jsonl`，并按 `t2_scene_split.json` 的 video 集过滤、按期望计数断言；`code_sha256` 仅记录运行时 hash（`335-340`），仓库未提供该远端 SHA 的现场输出，因此不能声称已独立复核 `d8948551…a09f`。

## §4 设计问题独立意见

1. **32 层 vs 16 层：当前方案不满足严格的 phase-2 expert 兼容性。** 脚本在 `80` 加载完整模型且只冻结 vision/connector，LoRA 挂在完整 text tower（`83-91`）；而题设 base 配置只有 16 层。若 phase 2 依赖 16 层 VLM，必须明确只导出/映射前 16 层，或把 base 架构升级为 32 层并验证 expert 接口；否则“训练成功”与“可接 phase 2”是两个不同结论。
2. **1 epoch 的量级合理，但不能由现有结果证明足够。** 318,331/16 约 19,895 个 optimizer steps；1 epoch 已让 holdout answer loss 从 2.25 降至 0.176，但 exact match 28.9% 仅略高于 majority 28.5%、低于 shuffled 29.4%，说明拟合 token loss 尚未转成指令条件的决策收益。继续增加 epoch 可能过拟合 scene/action prior；建议把 1 epoch 作为上限候选，以固定 holdout exact/greedy 指标决定是否延长。
3. **保留 `<image>\n.` 是合理的兼容选择。** builder `61` 保留换行和句点，`phase1_prompt.py:20-24` 记录其来源；由于训练和评测都共享同一 token 序列，怪异本身不会产生 train/eval drift。代价是可读性和迁移性较差，若改写必须同时重训/重新验证，不能在评测侧单独清理。

## 非阻塞建议

- 在 GPU 环境用实际 transformers 版本跑一次 processor/`inputs_merger` 对照，确认不 resize 的原始帧尺寸与视觉位置编码兼容，并记录输出 shape。
- 将 `eval_history.json` 写入临时文件后 rename，避免磁盘/进程中断留下不可解析 JSON。
- 训练前检查输出盘剩余空间并把 checkpoint、TensorBoard 和 history 的写入路径统一到数据盘；当前 TensorBoard 默认值为 `/root/tf-logs`（`57`）。

## 实际运行的命令与输出

- `wc -l scripts/p5_phase1_train.py src/smolvla/phase1_prompt.py datasets/navila_r2r.py tests/smolvla/test_phase1_prompt.py tests/datasets/test_navila_r2r.py`：531、167、122、134、109 行。
- `rg -n "..." scripts/p5_phase1_train.py src/smolvla/phase1_prompt.py datasets/navila_r2r.py`：定位上述关键实现行。
- `pytest -q tests/smolvla/test_phase1_prompt.py tests/datasets/test_navila_r2r.py`：**收集失败**，两测试均因本机 `ModuleNotFoundError: No module named 'torch'`；不能据此声称测试通过，相关动态检查标为 MISSING。
- 读取模型 `preprocessor_config.json`：观察到 `do_image_splitting:true`、`do_resize:true`、mean/std `[0.5,0.5,0.5]`、`rescale_factor=1/255`；这正是 B 项保留 MISSING 的依据。
