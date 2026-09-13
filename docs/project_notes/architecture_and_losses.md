# 架构与损失设计子笔记

## 用途

这份子笔记专门沉淀：

- 主架构演化
- 关键模块实现
- 各阶段 loss / target 设计
- 训练目标语义变化

与主索引的关系是：

- `docs/llada_vla_go2_project_notes.md` 保留全局叙事
- 本文件保留“架构和损失”这条线的细节

## 1. 架构演化总览

### 1.1 早期 `SigLIP` 简化结构

最早并不是直接上当前这套 robotics lane，而是先用更简单的连续控制结构做模态归因实验：

- 视觉 backbone：`SigLIP`
- 结构：视觉特征 + 文本条件 + 状态输入 + 小型融合头
- 输出：连续动作回归头

这阶段的核心目标不是追求最强模型，而是回答：

- `image_only` 到底有没有信息
- `instruction_only` 到底能不能直接主导动作
- `state_only` 会不会已经足够

### 1.2 早期 `LLaDA-V` 连续控制设想

之后开始尝试把 `LLaDA-V` 当更强的多模态 backbone，但当时路线仍然偏连续控制：

- `image + instruction + state`
- `-> LLaDA-V backbone`
- `-> pooled/fused feature`
- `-> action head`
- `-> [vx, vy, wz]`

这一步的关键意义是：项目开始从“轻量视觉模型做 policy”过渡到“让 LLaDA-V 直接参与策略学习”。

### 1.3 robotics lane 初版

真正定型后，项目没有继续往“连续回归头”方向走，而是单独开出了 robotics lane：

- 目录：`LLaDA-V/train/robotics/`
- 与共享 llava 主训练路径隔离
- 复用：
  - vision tower
  - projector
  - multimodal prepare
  - `LLaDA-V` backbone

新建：

- tokenizer
- prompt builder
- dataset
- collator
- localized action heads
- decoder
- eval

### 1.4 real-only Stage 1 / 1.6

这一步开始，项目从“能不能训练”过渡到“是不是假成功”。

架构变化主要体现在：

- Stage 1：`projector_only`
- Stage 1.6：delta target + full-mask + LoRA last-4

### 1.5 当前 anchor-residual 主线

当前最成熟的主线可以概括为：

- `LLaDA-V` backbone
- robotics prompt
- localized `vx / vy / wz` heads
- step-level remask decoder
- anchor-residual target
- weighted / targeted validation

这条线的核心思想不是“把未来动作全都学出来”，而是：

- anchor 吸收惯性延续
- residual 负责真正的决策修正

## 2. 当前主架构

### 2.1 总体结构

当前主模型不是从零写的 transformer，而是一个 wrapper：

- 外层：`LladaVLAForVelocityControl`
- 内层：`LlavaLLaDAModelLM`

输入路径：

- image
- instruction
- state（当前以 prompt text 形式进入）
- optional previous_action condition

输出路径：

- action token logits
- action token predictions
- decoded continuous target chunk
- recovered absolute action chunk

### 2.2 关键模块

#### `VelocityControlConfig`

文件：

- `LLaDA-V/train/robotics/config.py`

职责：

- 统一管理动作空间语义，而不只是训练超参

关键字段：

- `horizon_k`
- `action_target_mode`
- `use_residual_action`
- `anchor_mode`
- `loss_w_vx / vy / wz`
- `use_step_level_remask`
- `prev_action_dropout_*`

最重要的理解是：

- 这个类决定了“模型在预测什么”
- 不是简单的 YAML 映射器

#### `VelocityTokenizer`

文件：

- `LLaDA-V/train/robotics/tokenization/velocity_tokenizer.py`

职责：

- 连续 `vx / vy / wz` 与离散 bins 之间的往返

关键意义：

- 它把连续控制问题嵌进了 LLaDA 风格 masked token 训练框架

#### `AnchorChunkBuilder`

文件：

- `LLaDA-V/train/robotics/anchors.py`

职责：

- 根据 `previous_action` 构造未来动作 anchor

支持模式：

- `repeat_prev`
- `decay_prev`
- `constant_velocity_extrapolation`

关键意义：

- 让模型不用把“惯性延续”也全靠自己记住

#### `ActionHeads`

文件：

- `LLaDA-V/train/robotics/modeling/action_heads.py`

职责：

- 把 action token hidden states 映射到每维局部动作 bins

关键实现：

```python
slot_ids = torch.arange(action_tokens) % 3
vx_hidden = hidden[:, slot_ids == 0, :]
vy_hidden = hidden[:, slot_ids == 1, :]
wz_hidden = hidden[:, slot_ids == 2, :]

vx_logits = vx_head(vx_hidden)
vy_logits = vy_head(vy_hidden)
wz_logits = wz_head(wz_hidden)
```

这意味着：

- `vx` token 只走 `vx_head`
- `vy` token 只走 `vy_head`
- `wz` token 只走 `wz_head`

也就是说，当前项目明确拒绝把动作监督退回 full-vocab LM head。

#### `LladaVLAForVelocityControl`

文件：

- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py`

职责：

- 定义训练和推理的真正语义

最关键的函数：

1. `actions_to_targets()`
   - 把 absolute action chunk 映射到当前 target space

2. `targets_to_actions()`
   - 把 delta / residual 空间重新恢复成 absolute action

3. `_inject_action_embeddings()`
   - 往 prompt 序列中的 action slots 注入专用 action embeddings

4. `_corrupt_action_embeddings()`
   - 只对 action 区做 mask corruption

5. `_compute_llada_diffusion_loss()`
   - 当前训练主损失

#### `VelocityDiffusionDecoder`

文件：

- `LLaDA-V/train/robotics/inference/velocity_diffusion_decoder.py`

职责：

- 通过 iterative denoising 解出整段动作

推理主循环可以概括成：

```python
for step in range(decode_steps):
    head_output = forward_action_predictions(inputs_embeds)
    current_tokens = head_output.action_predictions
    current_probabilities = head_output.action_probabilities
    step_confidence = aggregate(current_probabilities)
    resolve_some_steps_or_tokens(step_confidence)
    overwrite_resolved_slots_with_action_embeddings()
    keep_unresolved_slots_as_mask()
```

本质不是 AR 解码，而是“整段动作逐步冻结高置信位置”。

## 3. 当前完整数据流

### 3.1 dataset

文件：

- `LLaDA-V/train/robotics/data/dataset_vla.py`

`VelocityControlDataset.__getitem__()` 里当前真实做了这些事：

1. 读 record
2. 规范化 state
3. 提取 previous_action
4. 解析 image
5. pad / trim `action_chunk`
6. previous_action dropout
7. 构造 `anchor_chunk`
8. 构造 `target_action_chunk`
9. 量化成 `action_labels`
10. 构造 prompt + `action_positions`
11. 生成 sequence labels

关键结论：

- 这个 dataset 已经承担了大量“语义构造”工作
- 它不是简单的数据读取器

### 3.2 collator

文件：

- `LLaDA-V/train/robotics/data/collator_vla.py`

collator 主要工作是打包，不改语义。核心 stack 的张量包括：

- `action_labels`
- `action_positions`
- `action_mask`
- `action_valid_mask`
- `continuous_actions`
- `target_actions`
- `anchor_actions`
- `residual_actions`
- `previous_actions`

### 3.3 prompt builder

文件：

- `LLaDA-V/train/robotics/data/prompt_builder.py`

当前 prompt 由这些段组成：

- `<image>`
- `<instruction>`
- `<state>`
- optional `<previous_action>`
- `<action_plan>`
- `<action_tokens> MASK ... MASK </action_tokens>`

当前 state 不是单独 MLP 分支，而是 text-serialized state。

### 3.4 forward

主函数：

- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py::forward`

核心链路：

1. 准备 multimodal `inputs_embeds`
2. 解析 `action_positions`
3. 注入干净 action embeddings
4. 对 action 区做 corruption
5. backbone forward
6. gather action hidden states
7. localized heads 产出 logits
8. 计算 masked diffusion loss

### 3.5 decoder

主函数：

- `predict_actions()`
- `VelocityDiffusionDecoder.decode()`

核心链路：

1. mask 初始化 action slots
2. iterative denoise
3. token -> continuous target chunk
4. target chunk -> absolute action chunk
5. 输出 anchor / residual / execute actions

### 3.6 validation

主文件：

- `LLaDA-V/train/robotics/eval/eval_velocity.py`
- `LLaDA-V/train/robotics/eval/stage1_5_policy_validation.py`

核心思路：

- 不只算 model 本身
- 还要跟：
  - `previous_action_hold`
  - `anchor_zero_residual`
  - image / instruction shuffle
  - previous_action ablation
 进行比较

## 4. 各阶段损失函数

### 4.1 `SigLIP` 简化结构阶段

损失形式：

```text
L_reg = mean(||a_pred - a_gt||^2)
```

语义：

- 连续回归
- 单步或短 chunk 行为克隆

### 4.2 早期 `LLaDA-V` 连续控制设想阶段

损失形式：

```text
L_chunk = mean_t mean_d (a_pred[t, d] - a_gt[t, d])^2
```

语义：

- 连续 chunk 回归
- 还未切到 tokenized action

### 4.3 robotics lane Stage 0

损失形式：

```text
L_smoke = mean_{i in masked action tokens} CE(logits_i, label_i)
```

语义：

- 最小 smoke
- 只验证 action-only masked supervision 是否成立

### 4.4 Stage 1: projector-only absolute token 路线

损失形式：

```text
L_stage1 = sum_i CE(logits_i, label_abs_i) / p_mask_i * w_dim(i)
```

其中：

- `label_abs_i` 是 absolute action bin
- `p_mask_i` 是该位置被 mask 的概率
- `w_dim(i)` 是 `vx / vy / wz` 维度权重

语义：

- 图像、文本、状态只是条件
- 真正恢复的是 absolute action token

### 4.5 Stage 1.6: delta token 路线

损失形式：

```text
L_stage1.6 = sum_i CE(logits_i, label_delta_i) / p_mask_i * w_dim(i)
```

其中标签语义变成：

```text
delta[0] = action[0] - previous_action
delta[t] = action[t] - action[t-1]
```

语义：

- 换的是监督对象
- 不是换了 loss 外形

### 4.6 当前 anchor-residual 路线

损失形式：

```text
label_residual = quantize(action_gt - anchor(previous_action, state))
L_anchor_res = sum_i CE(logits_i, label_residual_i) / p_mask_i * w_dim(i)
```

推理恢复：

```text
action_pred = anchor + decode(residual_pred)
```

语义：

- 把惯性延续交给 anchor
- 把真正策略修正交给 residual

### 4.7 当前主损失的完整工程化版本

可以写成：

```text
L = (1 / B) * sum_b sum_i [
    valid_i * masked_i *
    CE(logits_i, label_i) / p_mask_i *
    w_dim(i) *
    w_mag(i)
] / noisy_data_length_b
```

其中：

- `valid_i`：是否在有效 action step 中
- `masked_i`：是否本轮被 mask
- `p_mask_i`：mask 概率校正
- `w_dim(i)`：维度权重
- `w_mag(i)`：可选的大改变量权重
- `noisy_data_length_b`：该样本有效 action-token 数

这说明当前主损失不是普通 CE，而是：

- action-only
- masked-only
- localized
- 按维度加权
- 可按幅值再加权
- 样本内归一

## 5. 当前判断

从架构和损失设计角度看，项目当前最清晰的主线是：

- 不再回退到纯连续回归 baseline
- 不退回 full-vocab 动作监督
- 继续沿着：
  - localized heads
  - anchor-residual
  - step-level remask
  - weighted / targeted validation
 这条线细化

## 6. 关键源码索引

- `LLaDA-V/train/robotics/config.py`
- `LLaDA-V/train/robotics/tokenization/velocity_tokenizer.py`
- `LLaDA-V/train/robotics/anchors.py`
- `LLaDA-V/train/robotics/data/dataset_vla.py`
- `LLaDA-V/train/robotics/data/collator_vla.py`
- `LLaDA-V/train/robotics/data/prompt_builder.py`
- `LLaDA-V/train/robotics/modeling/action_heads.py`
- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py`
- `LLaDA-V/train/robotics/inference/velocity_diffusion_decoder.py`
- `LLaDA-V/train/robotics/eval/eval_velocity.py`
- `LLaDA-V/train/robotics/eval/stage1_5_policy_validation.py`
