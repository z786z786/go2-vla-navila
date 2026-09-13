# LLaDA-VLA Go2 模型架构图（Markdown / CLI 可读版）

> HTML 无法查看时使用本文件。重点解释 `action_positions` 和 `action_labels` 在模型里的位置。

## 1. 核心概念

### action_positions

`action_positions` 是动作 token 在整条 prompt token 序列中的绝对位置索引。

- 形状：单样本 `[3K]`，batch 后 `[B, 3K]`
- 当前训练 K=4，所以 `3K=12`
- 顺序固定为：
  - `vx_0, vy_0, wz_0`
  - `vx_1, vy_1, wz_1`
  - `vx_2, vy_2, wz_2`
  - `vx_3, vy_3, wz_3`

它回答的问题是：

```text
整条 input_ids 里，哪些 token 位置是 action slots？
```

代码位置：

- `LLaDA-V/train/robotics/data/prompt_builder.py::PromptTokenizationResult`
- `LLaDA-V/train/robotics/data/prompt_builder.py::tokenize_prompt_segments()`

生成逻辑：

```python
input_ids = concat(prefix_ids, action_placeholder_ids, suffix_ids)
action_positions = arange(
    prefix_ids.numel(),
    prefix_ids.numel() + action_placeholder_ids.numel(),
)
```

如果 prefix 有 137 个 token，K=4，那么：

```text
action_positions = [137, 138, 139, ..., 148]
```

这 12 个位置就是 action 区。

### action_labels

`action_labels` 是每个 action slot 对应的动作监督标签，即离散 bin id。

- 形状：单样本 `[3K]`，batch 后 `[B, 3K]`
- 它不是位置，而是类别标签。
- 当前 bins 通常是 256 类。

它回答的问题是：

```text
每个 action slot 应该预测哪个动作 bin？
```

代码位置：

- `LLaDA-V/train/robotics/data/dataset_vla.py::__getitem__()`
- `LLaDA-V/train/robotics/tokenization/velocity_tokenizer.py::encode_action_chunk()`

生成逻辑：

```python
action_chunk = sample["action_chunk"]          # continuous [K,3]
anchor_chunk = AnchorChunkBuilder(previous_action)
target_action_chunk = action_chunk - anchor_chunk   # residual target
action_labels = VelocityTokenizer.encode_action_chunk(target_action_chunk)
```

当前主线是 anchored residual，所以 `action_labels` 对应的是 residual 动作，不是 absolute 动作。

## 2. 二者如何绑定

在 dataset 里构造 sequence labels：

```python
labels = torch.full_like(input_ids, IGNORE_INDEX)
labels[action_positions] = action_labels
```

含义：

```text
action_positions 决定监督发生在哪里。
action_labels 决定这些位置监督成什么类别。
```

一句话：

```text
action_positions = 坐标
action_labels    = 答案
```

## 3. 完整训练架构 ASCII 图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                               Processed Dataset                              │
│                                                                              │
│  instruction                                                                  │
│  image / image_path                                                           │
│  state: vx, vy, yaw_speed, roll, pitch, yaw, mode, gait_type                  │
│  previous_action: vx, vy, wz                                                  │
│  action_chunk: continuous [K, 3] = [vx, vy, wz] * K                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Dataset Semantic Build                              │
│                                                                              │
│  1. normalize state                                                           │
│  2. extract previous_action                                                   │
│  3. pad / trim action_chunk                                                   │
│  4. anchor_chunk = AnchorChunkBuilder(previous_action)                        │
│  5. target_action_chunk = action_chunk - anchor_chunk                         │
│     当前主线：anchored residual target                                         │
│  6. action_labels = VelocityTokenizer.encode_action_chunk(target_action_chunk) │
│                                                                              │
│  action_labels shape: [3K]                                                     │
│  K=4 时：[vx0_bin, vy0_bin, wz0_bin, ..., vx3_bin, vy3_bin, wz3_bin]          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Prompt Builder                                   │
│                                                                              │
│  prompt =                                                                     │
│    <image>                                                                    │
│    <instruction> ... </instruction>                                           │
│    <state> vx=... ; vy=... ; yaw_speed=... ; ... </state>                     │
│    optional <previous_action> ... </previous_action>                          │
│    <action_plan> [step 0] vx_0 vy_0 wz_0 ... </action_plan>                   │
│    <action_tokens> MASK MASK ... MASK </action_tokens>                        │
│                                                                              │
│  当前 V2.3 主配置：use_prev_action_as_condition=false                         │
│  所以 previous_action 不进 prompt，但仍用于 anchor。                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         tokenize_prompt_segments                              │
│                                                                              │
│  input_ids = prefix_ids + action_placeholder_ids + suffix_ids                 │
│                                                                              │
│  action_positions =                                                           │
│    [prefix_len, prefix_len+1, ..., prefix_len+3K-1]                            │
│                                                                              │
│  action_positions shape: [3K]                                                  │
│  含义：action slots 在 input_ids 里的绝对 token 下标。                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Build Sequence Labels                              │
│                                                                              │
│  labels = IGNORE_INDEX everywhere                                             │
│  labels[action_positions] = action_labels                                     │
│                                                                              │
│  也就是：                                                                     │
│  - 非 action token 不算 loss                                                   │
│  - 只有 action_positions 指定的位置有动作监督                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LladaVLAForVelocityControl.forward                         │
│                                                                              │
│  input_ids                                                                    │
│  attention_mask                                                               │
│  labels                                                                       │
│  action_labels                                                                │
│  action_positions                                                             │
│  images                                                                       │
│  previous_actions / anchor_actions / target_actions                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           _prepare_inputs                                     │
│                                                                              │
│  调用 LLaDA-V 原始 multimodal prepare：                                        │
│                                                                              │
│  image -> vision tower -> mm_projector                                        │
│  text/state/action placeholder -> token embeddings                            │
│                                                                              │
│  输出：inputs_embeds [B, L, H]                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         _resolve_action_positions                             │
│                                                                              │
│  优先级：                                                                     │
│  1. 从 labels != IGNORE_INDEX 找 action positions                             │
│  2. 如果显式传了 action_positions，就直接用                                  │
│  3. 从 input_ids 里的 mask_token 找                                           │
│  4. 根据 action_labels 长度 fallback 到序列末尾                               │
│                                                                              │
│  输出：resolved_action_positions [B, 3K]                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         _inject_action_embeddings                             │
│                                                                              │
│  对每个 action slot：                                                         │
│                                                                              │
│  for offset, position in enumerate(action_positions):                         │
│      dim = ("vx", "vy", "wz")[offset % 3]                                  │
│      label = action_labels[offset]                                            │
│      inputs_embeds[position] = action_embeddings[dim](label)                  │
│                                                                              │
│  这里 action_positions 决定替换哪个 token embedding。                         │
│  这里 action_labels 决定替换成哪个动作 bin embedding。                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         _corrupt_action_embeddings                            │
│                                                                              │
│  只在 action_positions 对应的区域做 mask corruption。                         │
│                                                                              │
│  clean action embeddings -> noisy action embeddings                           │
│                                                                              │
│  输出：                                                                       │
│  - noisy_embeds [B, L, H]                                                     │
│  - masked_action_mask [B, 3K]                                                 │
│  - p_mask [B, 3K]                                                             │
│                                                                              │
│  当前 masking：high-mask curriculum + full-mask probability                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              LLaDA-V Backbone                                 │
│                                                                              │
│  noisy_embeds -> transformer / diffusion backbone                             │
│                                                                              │
│  输出：hidden_states [B, L, H]                                                 │
│                                                                              │
│  注意：backbone 看到的是完整序列：image/text/state/action slots。              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       _gather_action_hidden_states                             │
│                                                                              │
│  action_hidden_states = hidden_states[batch, action_positions]                │
│                                                                              │
│  [B, L, H] + [B, 3K] -> [B, 3K, H]                                            │
│                                                                              │
│  这里 action_positions 是 backbone 输出和 action head 之间的桥。              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                ActionHeads                                    │
│                                                                              │
│  输入：action_hidden_states [B, 3K, H]                                        │
│                                                                              │
│  slot % 3 == 0 -> vx_head                                                     │
│  slot % 3 == 1 -> vy_head                                                     │
│  slot % 3 == 2 -> wz_head                                                     │
│                                                                              │
│  输出：                                                                       │
│  action_logits [B, 3K, n_bins]                                                │
│  action_predictions [B, 3K]                                                   │
│  action_probabilities [B, 3K]                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         LLaDA Diffusion CE Loss                               │
│                                                                              │
│  loss = CE(action_logits, action_labels)                                      │
│                                                                              │
│  只在以下位置算：                                                             │
│  masked_action_mask & action_valid_mask                                       │
│                                                                              │
│  并做：                                                                       │
│  - / p_mask                                                                  │
│  - vx/vy/wz slot weight                                                       │
│                                                                              │
│  所以 action_labels 是最终分类监督答案。                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 4. 推理架构 ASCII 图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Inference Prompt                                 │
│                                                                              │
│  image + instruction + state + action MASK slots                              │
│                                                                              │
│  推理时没有 action_labels。                                                   │
│  但仍然有 action_positions，用来定位 action slots。                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         prepare_generation_inputs                             │
│                                                                              │
│  action slots 初始化为 mask embedding                                         │
│  resolved_action_positions [B, 3K]                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VelocityDiffusionDecoder                              │
│                                                                              │
│  for decode step:                                                             │
│      backbone(noisy/masked action slots)                                      │
│      gather hidden_states[action_positions]                                   │
│      ActionHeads predict action_tokens                                        │
│      compute confidence                                                       │
│      freeze high-confidence step or token                                     │
│      unresolved slots remain mask                                             │
│                                                                              │
│  当前主线：step-level remask                                                  │
│  一次冻结 [vx_t, vy_t, wz_t] 三个 token。                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Predicted Action Tokens                              │
│                                                                              │
│  action_tokens [B, 3K]                                                        │
│  action_probabilities [B, 3K]                                                 │
│  step_confidence [B, K]                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VelocityTokenizer.decode                              │
│                                                                              │
│  bins -> continuous target chunk                                              │
│                                                                              │
│  target_actions [B, K, 3]                                                     │
│  当前主线中这是 residual target actions。                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         targets_to_actions                                    │
│                                                                              │
│  anchor_actions = Anchor(previous_action)                                     │
│  continuous_actions = anchor_actions + residual_actions                       │
│                                                                              │
│  输出 absolute velocity chunk: [B, K, 3]                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Go2 Execute Actions                                │
│                                                                              │
│  execute_actions = continuous_actions[:, :execute_steps]                      │
│                                                                              │
│  每步是 [vx, vy, wz] 高层速度控制命令。                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 5. 最短总结

```text
action_positions:
  是 action token 在 input_ids 里的位置。
  负责定位：在哪里注入 action embedding，在哪里 gather hidden states。

action_labels:
  是 action token 的分类监督标签。
  负责告诉模型：每个 vx/vy/wz slot 应该预测哪个 bin。

关系：
  labels[action_positions] = action_labels

训练：
  action_positions + action_labels 都用。

推理：
  只用 action_positions，不用 action_labels；action_labels 由模型预测成 action_tokens。
```
