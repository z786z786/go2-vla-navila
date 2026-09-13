# LLaDA-V on Go2 项目笔记

> 维护规则
>
> - 这是当前工作空间的项目笔记主索引。
> - 后续如果单文件过长，可以拆分到 `docs/project_notes/` 下的多个子笔记。
> - 一旦拆分，必须保留本文件作为总索引，并在这里登记子笔记入口，避免后续 Codex 会话丢失上下文。
>
> 当前子笔记索引：
>
> - `docs/project_notes/architecture_and_losses.md`: 主架构、关键模块、loss / target 设计的后续细化入口。
> - `docs/project_notes/experiment_timeline.md`: 逐轮实验时间线和结果沉淀入口。
> - `docs/project_notes/data_and_validation.md`: 数据合同、previous_action、validation / targeted-v2 的细化入口。
> - `docs/project_notes/turnpilot_targeted_v2_experiment_log.md`: TurnPilot / Targeted V2 / policy validation 的近期计划、实验结论与后续记录入口。

## 0. 笔记组织方式

当前文档是总索引，负责：

- 给出全局主线
- 记录稳定结论
- 提供子笔记跳转入口

如果后续信息继续增长，建议按下面方式拆分：

- 架构与损失细节写入 `docs/project_notes/architecture_and_losses.md`
- 逐轮实验记录写入 `docs/project_notes/experiment_timeline.md`
- 数据合同与验证细节写入 `docs/project_notes/data_and_validation.md`
- TurnPilot / targeted-v2 近期专项继续写入 `docs/project_notes/turnpilot_targeted_v2_experiment_log.md`

主规则：

- 新信息可以先写子笔记
- 但只要它会改变全局理解，就必须在本主索引里补一段摘要或更新跳转说明

## 子笔记入口

- [架构与损失设计](project_notes/architecture_and_losses.md)
- [实验时间线](project_notes/experiment_timeline.md)
- [数据与验证](project_notes/data_and_validation.md)
- [TurnPilot / Targeted V2 专项](project_notes/turnpilot_targeted_v2_experiment_log.md)
  最新摘要（2026-05-06）：V2.3 `turn-reacquire` 训练前准备已完成，包含 manifest、weighted dataset、校验、config 与启动脚本；当前按用户要求停在 GPU 训练前，等待远端重启/启动显卡后再跑完整 GPU preflight 和训练。
  最新摘要（2026-05-18）：新 5090 云端主机 `ssh -p 14363 root@connect.westd.seetacloud.com` 已接入，远端 repo 路径使用 `/root/autodl-tmp/llada-vla-go2`，训练 Python 使用 `/root/miniconda3/envs/llada-vla/bin/python`。该机器已验证 `torch 2.9.1+cu128` 可正常识别 `RTX 5090`，`scripts/check_train_env.sh` 与 `scripts/train_llada_vla_go2.sh --help` 均通过；当前阻塞从环境转为模型权重和 processed dataset 尚未落盘。
  最新摘要（2026-05-23）：在线闭环 `execute_steps > 1` 已修正为真正逐步执行 `model.predict_actions()` 返回的多个 `execute_actions`，每步仍经过 safety filter 并下发 command；单步返回 shape `(3,)`，多步返回 shape `(execute_steps, 3)`。当前 turnpilot sim 下每步仍是 `control_dt=0.05s`，建议下一轮先试 `execute_steps=2`。
  调试结论（2026-05-18）：本地工作区此前存在宿主级属主错误，导致 `neu-rt` 只能读不能写；已通过宿主侧 `sudo chown -R neu-rt:neu-rt /home/wxh/llada-vla-go2` 修复。后续若再次出现 `touch` / `apply_patch` 对仓库根目录报 `Permission denied`，优先检查属主是否被重置。

## 1. 项目概览

这条项目线的核心目标一直比较稳定：不是做一个“全能 VLA”，而是在 `LLaDA-V` 的多模态骨架上，尽快落出一条能用于 Go2 高层速度控制的 VLA 路线。输入是图像、文本指令、机器人状态，输出是未来若干步的高层速度动作 `vx / vy / wz`。早期目标更偏“先打通链路”，后期目标逐步收紧为“证明模型不只是复读 previous action，而是真的用上视觉和指令做决策”。

这份笔记的主线证据来自当前工作仓 `llada-vla-go2`，辅助证据来自：

- `docs/training_plan_current_v1.md`
- `LLaDA-V/train/robotics/*`
- `configs/*.yaml`
- `outputs/*`
- `.omx/specs/*`、`.omx/plans/*`、`.omx/logs/turns-*.jsonl`
- `.codex/history.jsonl`、`.codex/sessions/2026/04/*`
- `trash/LLaDA-V`、`trash/go2_vla_collector_cleanup_20260419_211924`

文中所有带 `推断：` 的段落都是基于这些痕迹做的合理补全，不视为硬事实。

## 1.1 当前工作空间的稳定模块边界

按当前仓库结构和入口脚本，这个工作空间可以稳定拆成 7 个一层模块：

1. `LLaDA-V/`
   - 模型与训练核心子系统。
   - 其中 `LLaDA-V/train/robotics/` 已形成独立的 Go2 velocity-control lane，内部再分为配置/动作表示、prompt 与 dataset、localized action heads、decoder、eval、train 入口等子模块。

2. `collectors/`
   - real / sim 数据采集、标签整理、processed dataset 构建逻辑。
   - `collectors/common/processed_dataset.py` 是 raw session -> processed dataset 的核心转换点。

3. `data/` 与 `datasets/`
   - `data/` 更偏原始或修复后的 session 数据。
   - `datasets/` 更偏训练消费的派生数据包，标准接口是 `train.jsonl`、`val.jsonl` 与 `images/`。

4. `deploy/`
   - 在线控制执行面。
   - 负责 observation -> prompt -> model prediction -> safety filter -> velocity command 的闭环。

5. `scripts/` 与 `tools/`
   - 仓库编排层。
   - `scripts/` 负责采集、准备、审计、训练、评估、同步、发布；`tools/` 提供可复用的流水线辅助逻辑。

6. `configs/` 与 `workspace/`
   - 运行控制面。
   - `configs/workspace*.yaml`、`workspace/config.py`、release 配置共同定义路径解析、环境切换、发布目标与实验参数入口。

7. `tests/` 与 `docs/`
   - 治理与验证面。
   - `tests/` 负责数据合同、lane metadata、wrapper、mock deploy、evaluation smoke 等回归校验；`docs/` 负责 runbook、数据合同、实验结论与长期项目记忆。

补充：`outputs/` 主要承担训练、评估、审计、sanity 可视化等运行工件目录，不宜视为源码模块。

从接口关系看，仓库当前最稳定的跨模块边界有两条：

- 数据边界：raw session contract -> processed dataset contract
- 策略边界：`previous_action + state + instruction + image` -> `predict_actions()` -> `execute_actions/continuous_actions`

这两个边界同时连接了 collectors、training lane、eval 与 deploy，是后续多 agent 拆分任务时最适合按接口并行的切分线。

## 2. 时间线总览

### Phase 0: 最初设想与路线定型

#### 目标

最早的任务定义非常明确：在不破坏已有 collector / baseline2 / deploy 流程的前提下，把 `LLaDA-V` 接成一个新的 Go2 policy 路线。最初设想不是 chunk diffusion，而是更保守的一版：

- `instruction + RGB + state`
- `-> LLaDA-V backbone`
- `-> pooled feature / fusion`
- `-> Go2 action head`
- `-> [vx, vy, wz]`

证据来自 `.codex/history.jsonl` 里最早的大段任务说明，其中多次强调：

- 第一版先冻结 backbone
- 第一版不要 diffusion action decoder
- 第一版不要长时序 chunk
- 第一版优先验证“LLaDA-V feature + 小动作头”是否能学出动作映射

在这之前，你还做过一批比 `LLaDA-V robotics lane` 更早的简化结构实验，核心形态是：

- `SigLIP` 或 `siglip + 简单融合头`
- 连续控制小网络
- modality ablation：`instruction_only`、`image_only`、`state_only`

这些痕迹没有完整实验报告保存在当前仓里，但在会话和历史任务里有明确残留，说明你不是一上来就直接走 tokenized action / diffusion，而是先用更简单的视觉骨干和连续控制头试过模态贡献。

#### 当时的模型/配置

当时更像是“把 LLaDA-V 包成视觉 backbone”的思路，模型结构偏经典：特征抽取、状态编码、fusion、动作头。重点是最小侵入，不覆盖原 baseline 路线。

更早的简化结构可以理解为：

- 视觉 backbone：`SigLIP`
- 指令：简单文本条件或弱文本接口
- 状态：拼接或单独 MLP 编码
- 输出：连续动作回归头

推断：这批实验本质上是在回答“到底哪种模态对当前 Go2 任务真正有用”，也就是先做 modality attribution，再决定是否值得上更重的 `LLaDA-V`。

#### loss / target

最早要求是 MSE 单步回归，不是 tokenized action，也不是 diffusion denoising。

更具体地说，最早的几条简单结构路线的 loss 可以整理为：

- `SigLIP + 连续动作头`：连续回归 loss，主流可能是 `MSE` 或等价逐维 L2
- `instruction_only`：同样是连续动作回归 loss，只是输入里拿掉图像
- `image_only`：连续动作回归 loss，只是输入里拿掉文本和大部分条件
- `state_only`：连续动作回归 loss，只保留状态相关输入

推断：这几条实验阶段的动作输出大概率还是单步或短 horizon 的连续控制，不涉及离散 bin、mask token、localized classification。

#### 数据来源

数据侧强调复用已有 manifest / jsonl / collector / bridge / safety shell，说明项目从一开始就是在现有 Go2 数据链路上嫁接模型，而不是另起炉灶。

#### 预期

预期非常工程化：

- 先验证 feature 导出稳定
- 冻结 backbone 时 backward 正常
- 离线 `vx, vy, wz` 数量级合理
- 在线仍保留 safety shell

#### 结果

这一阶段你后来明确回忆出来的几个结论非常重要：

- `image_only` 几乎没用
- `instruction_only` 对“前后左右旋转”等单一动作强依赖，说明在非常受限的动作模板里，文本是有效的
- `state_only` 结果你记得不完全，但总体也“不太理想”

这三条信息虽然不是精确数值，但它们对项目演化的解释力很强：

- `image_only` 几乎没用，说明早期图像信号要么视觉骨干太弱、要么数据里的视觉决策点不足、要么图像与动作之间的监督关系太稀
- `instruction_only` 在单一旋转类动作上有效，说明文本在“动作模板非常强约束”的 setting 里可以直接主导策略
- `state_only` 也不理想，说明当前任务并不能靠状态做纯闭环控制，至少在你当时的数据与任务定义下不是这样

这条最初设想后来没有直接按“单步 MSE 回归”继续到底。项目很快转向了 robotics 独立 lane、离散动作 token、chunk 预测、localized action heads、diffusion-style masking。

#### 结论 / 下一步

最早的路线起到了“把问题空间缩小”的作用：先定义清楚这不是通用 VLA，而是 Go2 高层速度策略。后续大部分设计虽然变了，但这个边界一直没变。

同时，早期 ablation 还给出了一个后来被不断验证的隐含结论：

- “单独某个模态并不够”
- 真正的问题不是“有没有某个神奇模态”，而是“模型能否在合适的数据分布和 target 设计下，把视觉、文本、状态组织成有用的决策信号”

### Phase 1: 在 LLaDA-V 内建立 robotics lane

#### 目标

在不高风险改动 `LLaDA-V` 原始 conversation / llava 主路径的前提下，把 robotics 逻辑隔离到独立目录和入口，形成可训练、可离线评估、可 mock 部署的 Go2 velocity-control lane。

#### 当时的模型/配置

这一步的关键决策被完整记录在：

- `.omx/specs/deep-interview-llada-vla-go2-velocity-control.md`
- `.omx/plans/ralplan-llada-vla-go2-velocity-control.md`

定下来的架构是：

- 不直接改共享 `train.py` / `modeling_llada.py` 主路径
- 新建 `LLaDA-V/train/robotics/`
- 复用 vision tower / projector / multimodal prepare / LLaDA backbone
- 新建 velocity tokenizer、prompt builder、dataset、collator、action heads、decoder、eval、deploy

当前代码里，这条结构已经落成：

- `LLaDA-V/train/robotics/tokenization/velocity_tokenizer.py`
- `LLaDA-V/train/robotics/data/*`
- `LLaDA-V/train/robotics/modeling/*`
- `LLaDA-V/train/robotics/inference/*`
- `LLaDA-V/train/robotics/eval/*`
- `LLaDA-V/train/robotics/train_velocity.py`

#### loss / target

这一阶段的核心转折是：项目从“直接回归连续动作”切到了“动作 token 化 + localized classification + diffusion-style masking”。这说明团队判断 `LLaDA-V` 最适合被当作条件去噪 backbone，而不是普通 pooled feature encoder。

具体来说，这个转折意味着 loss 从：

- 连续值回归 loss

切到了：

- 动作量化后的离散分类 loss
- 但不是 full-vocab CE
- 而是 action-only、localized、masked 的 CE / diffusion-style denoising loss

#### 数据来源

数据合同从一开始就强调兼容旧 collector，同时逐步收紧到推荐的新合同。当前 `LLaDA-V/train/robotics/README.md` 已经把推荐格式写死为：

- `dict-form state`
- 明确的 `previous_action`
- 明确的 `action_chunk`
- 明确的 `action_mask`

#### 预期

这一步预期不是立刻拿到好指标，而是搭好可持续演进的框架：

- 保留 localized special-token classification
- 三个独立 head：`vx_head / vy_head / wz_head`
- 训练只在 action 区做监督
- 推理做简化 step-level remasking
- 部署默认安全优先，zero-command fallback

#### 结果

这些设计已经真实体现在当前代码中，而不是只停留在文档里。例如：

- `VelocityControlConfig` 当前已参数化 `horizon_k`、ranges、target mode、remask 方式
- `ActionHeads` 已实现 per-dim localized logits
- `LladaVLAForVelocityControl` 已把 anchor / delta / residual 三种 target 统一起来
- `VelocityDiffusionDecoder` 已实现 token-level 和 step-level remasking

#### 结论 / 下一步

这一步是整个项目最成功的一步：不是实验结果成功，而是工程架构落地成功。后面所有训练、验证、数据修正都围绕这条 robotics lane 演进。

## 3. 当前主架构关键模块详解

这一节专门解释“现在主架构到底是怎么实现的”，并给出注释式说明。

### 3.1 `VelocityControlConfig`

文件：

- `LLaDA-V/train/robotics/config.py`

作用：

- 定义整个 velocity-control lane 的全局语义配置
- 统一管理 `horizon_k`、`action_bins`、`vx/vy/wz` 范围、target mode、anchor mode、dropout、loss weights、fallback 参数

关键实现含义：

- `horizon_k` / `chunk_size`：动作 chunk 长度，当前主线为 `4`
- `action_target_mode`：决定训练目标是 `absolute`、`delta` 还是 `residual`
- `use_residual_action`：把 `anchored_residual` 语义收敛为当前主线 target
- `loss_w_vx / vy / wz`：给不同动作维度不同损失权重
- `use_step_level_remask`：推理时是否按 step 而不是单 token 重遮罩

注释式理解：

- 这个类不是普通超参容器，而是把“动作语义”本身编码成程序约束
- 也就是说，很多实验变化不是改模型结构，而是先改这个配置层

### 3.2 `VelocityTokenizer`

文件：

- `LLaDA-V/train/robotics/tokenization/velocity_tokenizer.py`

作用：

- 把连续 `vx / vy / wz` 映射成离散 bins
- 支持 chunk 编码 / 解码
- 支持 absolute、delta、residual 三类 target 的 range

注释式理解：

- 它解决的是“连续控制动作如何放进 LLaDA 风格 masked token 训练框架”
- 这一步是项目从连续控制 baseline 切到 tokenized action 路线的关键桥梁

### 3.3 `AnchorChunkBuilder`

文件：

- `LLaDA-V/train/robotics/anchors.py`

作用：

- 根据 `previous_action` 和部分状态，构造未来动作 anchor chunk

支持模式：

- `repeat_prev`
- `decay_prev`
- `constant_velocity_extrapolation`

注释式理解：

- anchor 的角色不是替代模型，而是把“惯性延续”这部分低难度预测先吸收掉
- 这样 residual 只需要学习“为什么这一段不该继续沿着 previous action 走”

### 3.4 `ActionHeads`

文件：

- `LLaDA-V/train/robotics/modeling/action_heads.py`

作用：

- 对 action token 的 hidden states 做维度专属分类

实现细节：

- token 位置按 `% 3` 路由到 `vx / vy / wz`
- 每个 head 只预测自己的 local bins
- 最终重新拼回 flattened action-token logits

注释式理解：

- 这部分是当前项目里“最不愿意退让的原则”之一
- 你们明确不希望动作监督退回 full LM vocab，因为那会让动作语义和语言 token 混在一起

关键实现可以直接压成下面这段伪代码：

```python
slot_ids = torch.arange(action_tokens, device=device) % 3
vx_mask = slot_ids == 0
vy_mask = slot_ids == 1
wz_mask = slot_ids == 2

vx_logits = vx_head(action_hidden_states[:, vx_mask, :])
vy_logits = vy_head(action_hidden_states[:, vy_mask, :])
wz_logits = wz_head(action_hidden_states[:, wz_mask, :])
```

然后再把三路 logits 按原 token 顺序重新拼回 `action_token_logits`。这意味着：

- `vx` 位置永远不会去预测 `wz bins`
- `wz` 位置也不会退化成 full vocab 分类
- 每一个 action token 的语义在 head 级别就是固定的

### 3.5 `LladaVLAForVelocityControl`

文件：

- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py`

作用：

- 这是当前主模型壳层
- 它把 LLaDA backbone 变成一个面向速度控制的条件去噪器

关键逻辑分为 5 段：

1. `actions_to_targets()` / `targets_to_actions()`
   - 统一 absolute / delta / residual 的前后变换

2. `_prepare_inputs()`
   - 复用 llava 的 multimodal prepare，把图像和文本条件拼进 backbone 输入

3. `_inject_action_embeddings()`
   - 把动作标签对应的 action embeddings 注入到 action slots

4. `_corrupt_action_embeddings()`
   - 只在 action 区施加 mask corruption

5. `_compute_llada_diffusion_loss()`
   - 在 masked action positions 上计算 weighted CE / diffusion-style loss

注释式理解：

- 这个类没有改写 LLaDA 主 backbone 的语言能力定义
- 它做的是“在 backbone 外围加一个机器人动作语义层”
- 因此它本质是 wrapper，不是从零写了一个新 transformer

最关键的代码实现点有 4 个：

1. `actions_to_targets()`
   - absolute：`target = action`
   - delta：
     - `target[0] = action[0] - previous_action`
     - `target[t] = action[t] - action[t-1]`
   - residual：
     - `target = action - anchor_chunk`

2. `targets_to_actions()`
   - 把 delta / residual target 空间重新还原到 absolute action 空间

3. `_inject_action_embeddings()`
   - 在 prompt 序列里，action slots 不直接放普通文本 token embedding
   - 而是替换成专门的 `action_embeddings["vx"|"vy"|"wz"]`

4. `_corrupt_action_embeddings()`
   - 只在 action 区采样 mask，不碰 image / instruction / state 这类条件部分

### 3.6 `VelocityDiffusionDecoder`

文件：

- `LLaDA-V/train/robotics/inference/velocity_diffusion_decoder.py`

作用：

- 实现推理时的 iterative denoising

当前逻辑：

- 初始 action slots 全 mask
- 每轮预测 token
- 根据 token confidence 或 step confidence 决定哪些位置保留
- 未解决位置继续 remask

注释式理解：

- 这部分对应训练时的 masked denoising objective
- 也保证了训练和推理的语义是一致的，而不是 train 一套、infer 一套

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

这说明它不是自回归逐 token 往后吐，而是“整段动作反复去噪”。

### 3.7 `stage1_5_policy_validation.py`

文件：

- `LLaDA-V/train/robotics/eval/stage1_5_policy_validation.py`

作用：

- 把模型输出和多种 baseline、ablation、slice metrics 放到同一验证框架里

关键点：

- 比较 `model`、`previous_action_hold`、`anchor_zero_residual`
- 支持 image shuffle / instruction shuffle / prev-action ablation
- 统计 decision slices

注释式理解：

- 它不是普通 eval 脚本，而是“证明模型是否真的在做决策”的闸门脚本
- 后面 targeted-v2 也是从这里的失败样本反推出来的

## 4. 当前主架构完整数据流

这一节按真实代码路径，把当前主线从样本读取到验证输出完整走一遍。

### 4.1 `dataset`：从原始 record 到训练样本

主文件：

- `LLaDA-V/train/robotics/data/dataset_vla.py`

`VelocityControlDataset.__getitem__()` 里当前真实发生的事情可以拆成 10 步：

1. 读取原始 record
   - 输入源一般是 `train.jsonl` / `val.jsonl`
   - 要求至少存在：
     - `instruction`
     - `state`
     - `action_chunk`

2. 标准化 `state`
   - `_normalize_state()` 会把 dict-form state 统一成当前 prompt / deploy 可消费的字段
   - 其中会显式移除旧的 `vx_prev / vy_prev / wz_prev` 等 legacy previous-action state keys
   - 同时补出 `yaw_speed <- wz` 这类别名

3. 提取 `previous_action`
   - 优先取 top-level `prev_action` / `previous_action`
   - 否则回退到 legacy state previous fields
   - 再不行才回退到 `control_action` / `raw_action`

4. 构造图像输入
   - `_resolve_image_path()` 解析 `image` / `image_path` / `source_image_path`
   - `_process_image()` 用 image processor 变成模型输入 tensor

5. 规范化 `action_chunk`
   - `_coerce_action_chunk()` 把 action steps 统一成 `K x 3`
   - 支持 legacy 2D `[vx, wz]` 兼容路径，会补成 `[vx, 0.0, wz]`
   - 同时生成 step-level `action_mask`

6. 对 `previous_action` 做 dropout / mask / noise
   - `_apply_previous_action_dropout()` 只影响 prompt condition
   - 这一步输出 `previous_action_dropout_mode`

7. 构造 `anchor_chunk` 与 `target_action_chunk`
   - `anchor_chunk = AnchorChunkBuilder.build(...)`
   - `_build_target_chunk()` 决定当前训练目标是：
     - absolute
     - delta
     - residual

8. 将连续 target 量化成 `action_labels`
   - `action_labels = velocity_tokenizer.encode_action_chunk(target_action_chunk)`
   - 同时生成 token-level `action_valid_mask = action_mask.repeat_interleave(3)`

9. 构造 prompt
   - `RoboticsPromptBuilder.build(...)`
   - 这里会把：
     - instruction
     - serialized state
     - optional previous_action block
     - action plan
     - `<action_tokens>` placeholder
     拼成单个 prompt

10. tokenization 并显式标记 action slots
   - `tokenize_prompt_segments(...)`
   - 生成：
     - `input_ids`
     - `action_positions`
     - `action_token_start`
     - `action_token_end`
   - 然后 `_build_labels()` 只在 action positions 上填入 `action_labels`，其它位置全是 `IGNORE_INDEX`

注释式理解：

- 当前 dataset 不只是“读数据”
- 它已经把训练 target semantics、previous-action condition、anchor 语义、prompt 对齐、action-slot 定位全部做完了
- 因此后面的 train loop 其实非常轻，复杂度大部分都在 dataset 里

### 4.2 `collator`：把样本打成一个批

主文件：

- `LLaDA-V/train/robotics/data/collator_vla.py`

`DataCollatorForVelocityControl.__call__()` 做的事相对直接：

- pad `input_ids`
- pad `labels`
- 生成 `attention_mask`
- stack 以下核心张量：
  - `action_labels`
  - `action_positions`
  - `action_mask`
  - `action_valid_mask`
  - `continuous_actions`
  - `target_actions`
  - `anchor_actions`
  - `residual_actions`
  - `previous_actions`
  - `anchor_previous_actions`
  - `target_clip_mask`
- 保留以下非张量元信息：
  - `states`
  - `instructions`
  - `prompts`
  - `raw_records`
- 如果有图像，还会展开：
  - `images`
  - `image_sizes`
  - `modalities`

注释式理解：

- collator 基本不改语义
- 它的工作只是“把 dataset 已经算好的结果搬进 batch”
- 这也是为什么当前项目很多调试都要从 dataset 开始看，而不是先看 collator

### 4.3 `prompt builder`：把机器人观测翻译成 LLaDA 可消费上下文

主文件：

- `LLaDA-V/train/robotics/data/prompt_builder.py`

当前 prompt 由几段组成：

1. `<image>`
2. `<instruction> ... </instruction>`
3. `<state> ... </state>`
4. 可选 `<previous_action> ... </previous_action>`
5. `<action_plan> ... </action_plan>`
6. `<action_tokens> MASK MASK MASK ... </action_tokens>`

这里有两个关键点：

- `state` 以文本序列化方式注入，不是结构化 state encoder
- action region 的长度严格等于 `3K`

注释式理解：

- 当前 V1 仍然是 text-serialized state
- 所以所谓“image + instruction + state”里，state 目前是通过 prompt 文字串进 backbone，不是额外 MLP 融合
- 这也解释了你后来为什么反复担心 state / previous_action shortcut：因为它们直接进了统一 prompt 语义空间

### 4.4 `forward`：从条件序列到 masked action loss

主文件：

- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py`

`forward()` 里的计算图可以按下面理解：

1. `_prepare_inputs()`
   - 调用 backbone 的 `prepare_inputs_labels_for_multimodal()`
   - 把图像 embedding 和文本 token 对齐成 `inputs_embeds`

2. `_resolve_action_positions()`
   - 找到 action slots 在整条序列里的绝对位置

3. `_inject_action_embeddings()`
   - 把 ground-truth action labels 对应的 embedding 注入 action slots
   - 这一步得到“干净的 action token embeddings”

4. `_corrupt_action_embeddings()`
   - 按 masking schedule 对 action slots 随机 corruption
   - 输出：
     - `noisy_embeds`
     - `masked_action_mask`
     - `p_mask`

5. backbone forward
   - 用 `noisy_embeds` 跑 `self.backbone.get_model()`
   - 得到全序列 hidden states

6. `_gather_action_hidden_states()`
   - 只抽取 action positions 的 hidden states

7. `ActionHeads.forward()`
   - 分别过 `vx_head / vy_head / wz_head`
   - 得到 localized logits 和 argmax predictions

8. `_compute_llada_diffusion_loss()`
   - 只在 masked action positions 上算 CE
   - 并做：
     - `1 / p_mask` 校正
     - 维度权重 `loss_w_vx / vy / wz`
     - 可选 `delta_magnitude_reweighting`

注释式理解：

- 当前训练并不是“直接拿 whole sequence logits 做语言建模”
- 而是“把 backbone 当条件去噪器，只在动作区恢复动作 token”

当前主损失可以直接写成：

```text
L = (1 / B) * sum_b sum_i [
    valid_i * masked_i *
    CE(logits_i, label_i) / p_mask_i *
    w_dim(i) *
    w_mag(i)
] / noisy_data_length_b
```

其中：

- `valid_i`：该 token 是否落在有效 action step 中
- `masked_i`：该 token 是否本轮被 mask，只有 mask 位置参与 loss
- `p_mask_i`：该位置被 mask 的概率，用于重要性校正
- `w_dim(i)`：维度权重，来自 `loss_w_vx / loss_w_vy / loss_w_wz`
- `w_mag(i)`：可选的大改变量权重，对大 delta / residual 样本增重
- `noisy_data_length_b`：该样本有效 action-token 数，用于样本内归一化

所以当前训练 loss 不是普通平均 CE，而是：

- action-only
- masked-only
- 维度加权
- 可选幅值加权
- 按样本有效长度归一

### 4.5 `predict_actions` / `decoder`：从 mask slots 到连续 chunk

主入口：

- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py::predict_actions()`
- `LLaDA-V/train/robotics/inference/velocity_diffusion_decoder.py`

推理链条如下：

1. `prepare_generation_inputs()`
   - 构造条件 prompt
   - 把 action slots 初始化成 mask embeddings

2. `VelocityDiffusionDecoder.decode()`
   - 迭代预测 token
   - 根据 `remask_mode` 决定：
     - token-level 保留
     - 或 step-level 保留

3. token -> continuous target chunk
   - `velocity_tokenizer.decode_action_chunk(...)`

4. target chunk -> absolute action chunk
   - `build_anchor_chunk(...)`
   - `targets_to_actions(...)`

5. 输出调试友好的中间量
   - `anchor_actions`
   - `residual_actions`
   - `target_actions`
   - `continuous_actions`
   - `execute_actions`

注释式理解：

- 训练预测的是 target-space token
- 最后部署和评估真正用的是 absolute action chunk
- 因此“target semantics”和“执行语义”被明确分开了

### 4.6 `validation`：如何判断模型是不是在做真实决策

主文件：

- `LLaDA-V/train/robotics/eval/eval_velocity.py`
- `LLaDA-V/train/robotics/eval/stage1_5_policy_validation.py`

验证链条如下：

1. `build_model()`
   - 加载 backbone
   - 加载 robotics wrapper
   - 必要时重建 LoRA 结构

2. `evaluate()`
   - 调 `model.predict_actions()`
   - 收集：
     - token accuracy
     - per-dim accuracy
     - absolute MAE
     - residual MAE
     - step MAE

3. `stage1_5_policy_validation.py`
   - 额外构造 baseline：
     - mean action
     - previous action hold
     - first-step hold
     - anchor-zero-residual
   - 再做 ablation：
     - image shuffle
     - instruction shuffle
     - previous_action ablation

4. 失败样本再回流到 manifest 工具
   - 这一步连接后来的：
     - normalized metrics
     - compare script
     - targeted-v2 manifest

注释式理解：

- 当前 validation 的目标不是“模型数值多低”
- 而是“模型是否比惯性 baseline 更有附加价值”

这对应一个很关键的项目转变：

- 训练时优化的是 masked localized CE
- 但实验成功标准逐渐变成：
  - absolute-space MAE
  - residual-space MAE
  - relative improvement vs hold / anchor baseline
  - conditioning ablation gain

### Phase 2: real-only 训练主线

#### 目标

先用真实数据建立一个最保守、最可信的 multimodal baseline，而不是一开始就混 sim。

这条主线的目标在 `docs/training_plan_current_v1.md` 里写得很清楚：

- input：前视图像、机器人状态、文本指令
- output：未来 `K` 步 velocity chunk
- 控制假设：10Hz、receding horizon
- 不是 general-purpose VLA，只是 high-level velocity control

#### 当时的模型/配置

文档把训练拆成多个 stage：

- Stage 0：real-only smoke
- Stage 1：real-only projector + action heads
- Stage 1.5：完整 checkpoint validation gate
- Stage 1.6：delta + full-mask policy training
- Stage 1.7A：event-heavy real data prep

早期 real projector-only 配置对应：

- `configs/llada_vla_go2_real_projector_only.yaml`

其中当前可见的关键字段：

- `horizon_k: 4`
- `train_mode: projector_only`
- `freeze_vision_tower: true`
- `freeze_llada_backbone: true`
- `lora.enabled: false`

#### loss / target

Stage 1 时的核心设计是：

- backbone 冻结
- 训练 projector + action heads
- 用 diffusion-mask noise 训练 action token 预测

推断：这一步的目标是验证多模态投影和动作头能否学到基本动作条件关系，而不是直接解决 closed-loop policy collapse。

#### 数据来源

`docs/training_plan_current_v1.md` 记录了当时的主数据集规模：

- `current_v1` 总计 `83,927` 样本
- real `37,868`
- sim `46,059`

real-only 调试集：

- `datasets/real_small_current_v1`
- 真实 session `20260418_204747`

`outputs/data_audit/current_v1_real_vs_sim/audit_summary.json` 进一步给出了 real 数据分布：

- real `vx` 范围约 `0.0..0.3`
- real `wz` 范围约 `-0.12..0.12`
- `vy` 基本恒为 `0`

#### 预期

文档里的预期非常克制：

- Stage 0 只要求 loss finite
- Stage 1 要求 train loss downward、eval 不发散、`vx/wz` MAE 改善
- `vy` 维度预计会接近零，因为数据里没有 lateral command

#### 结果

Stage 1 有明确完成证据，来自 `docs/training_plan_current_v1.md`：

- 训练在 5090 云 GPU 上完成
- 输出目录：`/root/autodl-tmp/outputs/train_current_v1_real_projector_only_main`
- 最佳 checkpoint：`checkpoint-1023`
- `best_eval_loss=0.627869`
- 训练时长约 `3h14m`，评估约 `24m`

这是整个项目第一批硬结果里最“正面”的一个：至少说明数据合同、模型前向、训练链路、保存恢复全都跑通了。

#### 结论 / 下一步

这一阶段的结论不是“模型已经可用”，而是“可以训练出一个 stable multimodal baseline”。后续马上引入 Stage 1.5 来确认它是不是伪成功。

### Phase 3: Stage 1.5 / 1.6 与 previous_action 问题暴露

#### 目标

这一步开始，目标从“loss 会降”转向“模型是不是真的在看视觉和指令，而不是靠 previous action 或状态先验硬撑”。

#### 当时的模型/配置

Stage 1.5 validation gate 的设计非常严苛，要求做：

- 完整 real-val 评估
- constant mean / hold baselines 对比
- 图像打乱 ablation
- 指令打乱 ablation
- session leakage 检查

Stage 1.6 再引入：

- delta target
- full-mask policy training
- high-mask curriculum
- previous-action dropout
- LoRA last-4

当前 Stage 1.6 配置在：

- `configs/llada_vla_go2_real_stage1_6_delta.yaml`

关键字段包括：

- `action_target_mode: delta`
- `use_prev_action_as_condition: true`
- `use_step_level_remask: true`
- `per_dim_action_heads: true`
- `loss_w_wz: 1.2`
- `lora.enabled: true`
- `last_n_layers: 4`

#### loss / target

这一阶段的目标设计明显更复杂：

- 训练标签从 absolute chunk 转向 delta chunk
- 推理时再从 delta 恢复 absolute action
- 希望弱化 absolute-action dominant mode collapse
- 用 previous_action dropout 打击 shortcut

#### 数据来源

仍然以 real-only 为主，但重点开始转向 event-heavy data，而不是平均意义上的大盘指标。

`outputs/data_audit/current_v1_real_vs_sim/audit_summary.json` 显示 real-only 数据里：

- `first_step_change` 非常少
- `vx_change` 很少
- turning 虽然不少，但很多是平滑转向而非强决策点

这对模型学“真正的视觉决策修正”并不友好。

#### 预期

Stage 1.6 的预期在文档里写得很直接：

- full-mask policy decoding 不再塌缩
- prediction std 上升
- turning / first-step-change / wz-change 子集改善
- image / instruction / no-prev-action ablation 要能显著变差

#### 结果

从 `.omx/logs/turns-2026-04-21.jsonl` 可看出，这一步之后的核心判断变成了：

- “从完全塌缩，进步到了主要靠 previous action 的弱条件策略”

这是当前项目中最关键的一句中期结论。它说明：

1. Stage 1.6 不是完全失败
2. 但它也没有过线
3. 主要问题从“完全不工作”变成了“工作，但靠 shortcut”

推断：这大概率对应这样一种现象：teacher-forced 或 denoising 指标可能改善，但一旦 full-mask rollout，模型仍然倾向于沿着 previous_action / anchor 走，视觉和指令的边际作用不够强。

#### 结论 / 下一步

这一步的结论有两个：

- 模型结构本身不是完全错误，因为 Stage 1.6 确实比完全塌缩好
- 真实数据分布、previous_action 合同、event-rich 样本占比、验证方法，比“再多训一会”更关键

### Phase 4: 数据合同、previous_action 与 event-heavy real data

#### 目标

修正最容易造成 shortcut 泄漏和错误归因的数据路径，尤其是 `previous_action` 的多路径问题。

#### 当时的模型/配置

这一阶段不是改主网络，而是大量修数据合同、审计逻辑、验证逻辑、manifest 构建逻辑。

当前仓里能看到这一串工作痕迹：

- `scripts/repair_previous_action_contract.py`
- `scripts/audit_velocity_dataset.py`
- `scripts/build_priority_train_manifest.py`
- `scripts/build_weighted_train_jsonl.py`
- `scripts/build_targeted_v2_manifest_from_failures.py`
- `scripts/validate_v2_weighted_manifest.py`
- `tests/unit/test_processed_dataset_contract.py`
- `tests/unit/test_targeted_v2_policy_validation_tools.py`

#### loss / target

这一阶段虽然不直接改 loss，但它实质上在修“loss 到底在监督什么”。如果 previous_action 同时从 top-level、state、prompt、anchor 多路进模型，那么任何指标都可能被污染。

#### 数据来源

文档和 audit 都指向一个结论：下一步 real 数据不该继续无差别扩充，而是要优先补：

- `recovery_correction`
- `turning_onset`
- `start/stop/restart`

`docs/training_plan_current_v1.md` 已经把建议采集配比写出来：

- `35%` recovery / correction
- `30%` turning onset / offset
- `20%` start / stop / restart
- `15%` generic overflow

`outputs/data_audit/event_rich_candidates/train/summary.json` 也表明已经开始显式筛 event-heavy 候选。

#### 预期

预期是让模型必须面对“previous_action 不能解释一切”的样本，尤其是在：

- 起步
- 转向修正
- 纠偏
- 暂停后再启动

这些地方视觉和指令的必要性更高。

#### 结果

从 `.omx/logs/turns-2026-04-21.jsonl` 可以看到大量围绕 turning onset、平滑转向、停走、速度分布、是否去掉 state 的讨论。最终倾向很明确：

- 不建议直接去掉 state
- 更应该打击 previous_action shortcut
- turning onset 定义和采集习惯需要与训练目标对齐

#### 结论 / 下一步

这一步的本质结论是：问题不只是模型，更多是“样本是否迫使模型做视觉决策”。这也为后面的 sim-only / turnpilot 路线铺路，因为仿真更容易系统性生成这种样本。

### Phase 5: sim-only、turnpilot、anchor-residual 主线

#### 目标

在真实数据主线之外，单独开一条 sim-only / sim-first 侧线，回答两个问题：

1. 当前 VLA 合同在仿真域里能不能闭环成立
2. 能不能用更受控、更事件丰富的 sim 数据解决真实主线暴露出的 previous-action 依赖问题

#### 当时的模型/配置

这条线最终演化成当前最活跃的一组配置：

- `configs/llada_vla_go2_sim_today_main_turnpilot_absolute.yaml`
- `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual.yaml`
- `configs/llada_vla_go2_sim_office_clean_absolute.yaml`
- `configs/llada_vla_go2_sim_office_clean_anchor_residual.yaml`

其中当前主线配置 `llada_vla_go2_sim_today_main_turnpilot_anchor_residual.yaml` 明确了：

- `horizon_k: 4`
- `action_target_mode: anchored_residual`
- `use_residual_action: true`
- `anchor_mode: repeat_prev`
- `use_prev_action_as_condition: true`
- previous-action dropout / mask / noise
- `use_step_level_remask: true`
- `loss_w_wz: 1.2`
- `lora.enabled: true`
- `last_n_layers: 4`

这标志着项目从 real Stage 1.6 的 delta 路线，进一步走到了 sim 主线的 anchor-residual 路线。

#### loss / target

这是整个项目 loss 设计上最成熟的一版：

- target 不是 absolute，也不是简单 delta，而是 anchored residual
- anchor chunk 用 `previous_action` 构造
- 模型只学 residual
- 推理时 `absolute = anchor + residual`
- 再 clamp 到 absolute action range

这套设计的隐含动机很明确：把“惯性延续”交给 anchor，把“真正的决策修正”交给 residual。

#### 数据来源

当前 sim 主数据集是：

- `datasets/current_v2_sim_today_main_turnpilot_20260425`

`outputs/data_audit/current_v2_sim_today_main_turnpilot_20260425/audit_summary.json` 给出的关键事实：

- 总样本 `203,042`
- 全部为 sim
- train `169,241`
- val `33,801`
- 主要 session 既有 `main`，也有 `turnpilot`
- `turnpilot` 相关 session 包括：
  - `dual_target_room_turnpilot_s0800_b00_gpu0`
  - `dual_target_room_turnpilot_s0801_b01_gpu0`

同一份 audit 还显示了 sim 数据特征：

- `vx` 范围约 `0.0..0.7`
- `wz` 范围约 `-0.8..1.0`
- `prev_action_sources` 全是 `logged`
- `turning`、`vx_change`、`first_step_change` 数量明显比早期 real 主线更丰富

这说明 sim 数据已经被故意设计为更适合训练“决策型 correction”。

#### 预期

这条 sim 线的预期有两层：

- 近目标：在仿真域里证明当前 VLA contract 闭环可用
- 中目标：形成比旧 projector-only baseline 更强的视觉/指令敏感性

后期又进一步加入：

- turnpilot subset
- main-only subset
- priority-weighted subset
- targeted-v2 failure-driven manifest

说明目标已经从“只是采 sim 数据”转向“根据 validation failure 反推训练样本”。

#### 结果

1. 旧 sim baseline 的结果并不好。

`outputs/sim_policy_validation/train_current_v1_sim_office_clean_projector_eval10_noplot/metrics.json`：

- `model_mae = 0.156959`
- `previous_action_hold_mae = 0.055213`
- `anchor_zero_residual_mae = 0.055213`
- `relative_improvement_vs_anchor_zero_residual = -1.8427`

`eval200` 结果也类似：

- `model_mae = 0.126111`
- `previous_action_hold_mae = 0.033469`
- `anchor_zero_residual_mae = 0.033469`
- `relative_improvement_vs_anchor_zero_residual = -2.7680`

这说明旧 sim baseline 明显输给 hold / anchor-zero-residual，问题和 real 主线的“弱条件策略”判断是一致的。

2. turnpilot / priority-weighted / targeted-v2 已经成为当前修复主线。

从 `scripts/prepare_sim_today_main_only_subset_priority_weighted_data.sh`、`scripts/build_targeted_v2_manifest_from_failures.py`、`scripts/normalize_policy_validation_metrics.py`、`scripts/compare_normalized_policy_validation.py` 可以看出，最近阶段已经不再满足于看一个 `metrics.json`，而是开始：

- 统一 normalized policy validation schema
- 显式比对 baseline / candidate
- 根据失败样本构 targeted v2 manifest
- 做 weighted retraining

#### 结论 / 下一步

当前最接近“项目现在在做什么”的描述是：

不是再证明这条路线能不能训练，而是在围绕 anchor-residual + turnpilot + weighted subset + failure-driven manifest 这一套，追求让模型终于在 policy validation 上赢过 `previous_action_hold` 和 `anchor_zero_residual`。

## 5. 当前最准模型架构

这一节描述的是“现在仓里实际存在的版本”，不是最初想法。

### 4.1 Backbone 与 lane 划分

当前主架构不是把 `LLaDA-V` 当普通 encoder，而是保留它的多模态 backbone 身份，把 robotics 逻辑独立包在外面。

关键模块：

- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py`
- `LLaDA-V/train/robotics/modeling/action_heads.py`
- `LLaDA-V/train/robotics/inference/velocity_diffusion_decoder.py`

`LladaVLAForVelocityControl` 做的事是：

- 复用 `LlavaLLaDAModelLM` backbone
- 单独维护 action embeddings
- 单独维护 `vx / vy / wz` action heads
- 单独处理 anchor / delta / residual target 转换
- 单独处理 action-only corruption 与 loss

### 4.2 动作表示

当前是离散 chunk 预测，不是单步回归。

- `horizon_k = 4`
- `action_dim = 3`
- 总 action tokens = `12`
- token 顺序固定为 `[vx, vy, wz] * K`

`VelocityControlConfig` 已把这套逻辑写死为可配置参数。

### 4.3 localized action heads

动作预测不是 full vocab CE，而是 localized per-dim classification：

- `vx_head`
- `vy_head`
- `wz_head`

`ActionHeads` 里按 token 位置 `% 3` 路由到对应 head。这个设计对应项目的一个长期坚持：动作监督不能退回 full LM vocab。

### 4.4 decoder

推理不是传统 autoregressive decode，而是 diffusion-style iterative denoising：

- action slots 初始为 mask
- 每轮预测 action token
- 按 token-level 或 step-level confidence 决定哪些位置保留、哪些继续 remask
- 当前默认 `step_level_remask`

这说明项目把“动作 chunk 生成”视为 masked denoising 问题，而不是 next-token generation。

## 6. 各阶段架构与 loss 设计

这一节按阶段明确写“架构是什么，loss 是什么”，避免只写目标不写监督方式。

### 5.1 早期 `SigLIP` 简化结构阶段

#### 架构

- 视觉骨干：`SigLIP` 或类似轻量视觉 encoder
- 融合方式：简单拼接 / MLP / 小型 fusion head
- 输出：连续动作头

#### loss

- 以连续动作回归为主
- 大概率是逐维 `MSE`

可以写成：

```text
L_reg = mean(||a_pred - a_gt||^2)
```

#### 注释

- 这一阶段的 loss 简单，但它已经暴露出模态问题：
  - `image_only` 没有效果
  - `instruction_only` 只在模板动作上有效
  - `state_only` 也不理想

### 5.2 早期 `LLaDA-V` 连续控制设想阶段

#### 架构

- `LLaDA-V / llada-v backbone`
- image / instruction / state 融合
- 连续动作 chunk regression head

#### loss

- 连续 chunk regression loss
- 不是分类
- 不是 tokenized action

可以写成：

```text
L_chunk = mean_t mean_d (a_pred[t, d] - a_gt[t, d])^2
```

#### 注释

- 这一步是从简单 `SigLIP` 往更强多模态骨干迁移的过渡方案
- 但后续真正落地时，团队判断 tokenized action 更适合 LLaDA 的 denoising 训练范式

### 5.3 robotics lane Stage 0: action-head smoke

#### 架构

- `LLaDA-V` backbone 冻结
- 只训练 `action_embeddings + action_heads`

#### loss

- masked action-token classification loss
- 至少保证每个 batch 有一个 masked action token 被监督

可以写成：

```text
L_smoke = mean_{i in masked action tokens} CE(logits_i, label_i)
```

#### 注释

- 这是最小可运行 smoke，不追求性能，只验证链路和有限监督成立

### 5.4 Stage 1: projector-only real baseline

#### 架构

- 冻结 vision tower
- 冻结 llada backbone
- 训练 `mm_projector + action_embeddings + action_heads`

#### loss

- diffusion-mask noise 下的 masked action-token CE
- 当前代码里对应 `_compute_llada_diffusion_loss()`

更细的实现是：

- 只在 masked action positions 上算 loss
- 用 `1 / p_mask` 做重要性校正
- 再按 `loss_w_vx / vy / wz` 做维度加权
- 如果启用 `delta_magnitude_reweighting`，还会对大改变量加权

因此这一阶段更准确的形式是：

```text
L_stage1 = sum_i CE(logits_i, label_abs_i) / p_mask_i * w_dim(i)
```

其中 `label_abs_i` 是 absolute action bin。

#### 注释

- Stage 1 的 loss 已经不再是纯回归
- 核心思想是：图像、文本、状态都只是条件；真正被恢复的是 masked action tokens

### 5.5 Stage 1.6: delta + full-mask

#### 架构

- backbone 仍以 `projector_only` 为主
- 加 LoRA 到最后 4 层 language backbone
- 引入 high-mask curriculum

#### loss

- target 改成 `delta chunk`
- 仍使用 masked token CE / diffusion-style denoising loss

标签语义：

- `delta[0] = action[0] - previous_action`
- `delta[i] = action[i] - action[i-1]`

因此这一步的 loss 外形不变，但标签语义变成：

```text
L_stage1.6 = sum_i CE(logits_i, label_delta_i) / p_mask_i * w_dim(i)
```

#### 注释

- 这一步不是换损失函数形式，而是换“监督对象”
- loss 仍是 token CE，但 token 现在表示 delta，不表示 absolute action

### 5.6 anchor-residual 当前主线

#### 架构

- 保留 robotics lane + localized action heads + step-level remask
- 增加 anchor builder
- 主 target 是 anchored residual

#### loss

- target 变为 residual token
- residual = future action - anchor chunk
- loss 仍是 masked localized CE

更明确地说：

```text
label_residual = quantize(action_gt - anchor(previous_action, state))
L_anchor_res = sum_i CE(logits_i, label_residual_i) / p_mask_i * w_dim(i)
```

推理时再恢复：

```text
action_pred = anchor + decode(residual_pred)
```

#### 注释

- 当前主线的关键不在 loss 数学形式有多新，而在 target semantics 改了
- 这等于把“惯性延续”和“真正修正”分开建模

### 5.7 policy validation / normalized validation 阶段

#### 架构

- 不改训练主模型
- 强化验证与失败回流工具链

#### loss / 指标关注点

- 训练 loss 仍是上面的 masked localized CE
- 但决策判断不再只看 `eval_loss`

主要看：

- `model_mae`
- `previous_action_hold_mae`
- `anchor_zero_residual_mae`
- image / instruction conditioning gain
- decision slices

#### 注释

- 这一步的核心是“训练 loss 已经不够说明问题”
- 必须让 validation 直接回答：模型有没有超过惯性基线

## 7. loss / target 设计演化

### 6.1 最初设想：MSE 单步回归

最早是最朴素的行为克隆：输出 `[vx, vy, wz]` 连续值，MSE 监督。

### 6.2 过渡到 action token + diffusion masking

进入 robotics lane 后，loss 逻辑改成：

- 连续动作先量化成 bins
- 只在 action token 位置做监督
- 只对 masked action positions 计算 diffusion-style loss

这一步的意义是把 `LLaDA-V` 的 masked denoising 训练方式和机器人动作预测对齐。

### 6.3 delta target

Stage 1.6 引入 delta：

- `delta[0] = action[0] - previous_action`
- `delta[i] = action[i] - action[i-1]`

目标是降低 absolute dominant mode collapse，逼模型学变化量。

### 6.4 anchored residual target

当前 sim 主线又进一步走到 anchored residual：

- anchor 来自 `previous_action` 展开出的 chunk
- residual = future - anchor
- 模型学 residual
- 推理再恢复 absolute chunk

这是目前最能体现“决策修正”思想的一版 target。

### 6.5 权重与 curriculum

当前还能看到这些 loss 细节：

- `loss_w_vx = 1.0`
- `loss_w_vy = 1.0`
- `loss_w_wz = 1.2`
- high-mask curriculum
- full-mask probability 分阶段提高
- previous_action dropout / noise / mask

推断：`wz` 加权偏高，是因为转向决策是当前最难、也是最关键的维度。

## 8. 数据与实验路线演化

### 7.1 real-only 先行

一开始明确不混 sim，原因很直接：

- real `vx` 低
- real `wz` 小
- sim 更激进
- 一上来混合会把基线判断搞乱

文档甚至明确写了：

- real-only baseline 先立住
- sim mixing 只能在 real-only baseline 之后

### 7.2 真实数据问题逐渐暴露

真实数据主线暴露出来的问题包括：

- `vy` 恒为零
- `mode/gait_type` 没信息量
- turning onset 稀缺
- 起停片段不足
- previous_action 很容易成为强 shortcut

### 7.3 sim-only side lane

于是项目开了 sim-only 侧线，最开始只是为了验证 contract 在仿真域可闭环。

### 7.4 turnpilot 与对照采集

后来的 sim 数据采集已经不只是“多采一点”，而是显式地做：

- main route
- turnpilot route
- door / box 对照
- 同场景不同指令对照
- 更平缓起停
- 减少初始化坏帧、穿模、动态加载干扰

这部分的大量设计讨论在 `.omx/logs/turns-2026-04-24.jsonl` 和 `turns-2026-04-25.jsonl` 中有持续痕迹。

### 7.5 subset、weighted、targeted-v2

最近一阶段，数据路线已经进入“从 validation failure 反推训练集”的模式：

- main-only subset
- turnpilot subset
- priority-weighted subset
- targeted-v2 manifest

这说明项目已从粗放式扩数据，转向 error-driven curriculum。

## 9. 关键实验记录

### 8.1 实验 A：real projector-only baseline

#### 实验目标

验证在真实数据上，冻结 vision tower 和 llada backbone，仅训练 projector + action heads，是否能形成 stable multimodal baseline。

#### 预期结果

- loss 可稳定下降
- eval 不发散
- 至少比随机或纯坏链路强

#### 实际结果

- 完整跑通
- `checkpoint-1023`
- `best_eval_loss = 0.627869`

#### 实验结论

链路可训练，multimodal baseline 成立，但这还不能证明模型真的在用视觉和指令。

### 8.2 实验 B：Stage 1.5 验证思想

#### 实验目标

区分“真正多模态策略”与“塌缩成 action prior”的伪成功。

#### 预期结果

- image shuffle / instruction shuffle 退化
- 优于 constant / hold baselines

#### 实际结果

仓内没有直接留下完整 Stage 1.5 指标 JSON，但后续多轮日志已经明确表述：模型主要还是依赖 previous action。

#### 实验结论

这一步虽然没有一份干净的最终数字留在本地，但从后续策略调整能反推出：Stage 1 baseline 没有真正过 Stage 1.5 的精神门槛。

### 8.3 实验 C：Stage 1.6 delta/full-mask 修复

#### 实验目标

缓解 full-mask rollout collapse，把模型从“直接塌掉”拉到能做一点修正。

#### 预期结果

- rollout 不再近零方差
- turning / wz-change 子集改善

#### 实际结果

中期判断是：

- “从完全塌缩，进步到了主要靠 previous action 的弱条件策略”

#### 实验结论

有效，但没过线。说明结构方向可行，数据与 shortcut 控制仍是瓶颈。

### 8.4 实验 D：旧 sim projector baseline policy validation

#### 实验目标

验证 sim-only baseline 是否至少能在仿真验证上超过简单 hold。

#### 预期结果

至少应该优于 `previous_action_hold` 或 `anchor_zero_residual`，否则模型没有提供额外价值。

#### 实际结果

两个现成结果都输得很明显：

- eval10：`0.156959` vs hold `0.055213`
- eval200：`0.126111` vs hold `0.033469`

#### 实验结论

旧 sim baseline 没有过 policy validation 这道线。

### 8.5 实验 E：turnpilot / weighted / targeted-v2 当前路线

#### 实验目标

让当前主线 finally 在 policy validation 上体现：

- 优于 hold / anchor baseline
- 对视觉 / 指令更敏感
- 在 failure slice 上改善

#### 预期结果

至少在 turning、decision-heavy、targeted failure subset 上领先简单基线。

#### 实际结果

当前仓内已经有：

- normalized policy validation tooling
- baseline registry schema
- targeted-v2 manifest builder
- weighted manifest validator

但最新一轮“是否已经赢过 baseline”的最终结论，本地还没有沉淀成单一最终报告。

#### 实验结论

项目已经进入“精修验证指标与样本闭环”的阶段，而不是“还在搭链路”。这是一种成熟信号，但也说明终局结果尚未完全收口。

## 10. 当前结论与判断

### 9.1 现在最可信的结论

1. 这条 `LLaDA-V -> robotics lane -> Go2 velocity control` 的工程链路已经打通。
2. 当前模型不是完全不会学，真实主线和仿真主线都证明它能训练、能出非平凡输出。
3. 真正的瓶颈不是“能不能跑”，而是“能不能在 policy validation 上稳定赢过 previous-action baseline”。

### 9.2 现在最大的问题

当前最大问题一直很一致：

- previous_action 太强
- 数据里必须做视觉决策的样本不够密
- 某些 validation 结果容易被平均指标掩盖

换句话说，现在的问题是“辨识度”，不是“可训练性”。

### 9.3 当前最靠谱的主线

从现有痕迹看，当前最靠谱的路线不是再回头做 simple regression，而是继续沿着这条线：

- anchor-residual target
- step-level remask
- per-dim localized heads
- LoRA last-4
- turnpilot / event-heavy data
- priority-weighted / targeted-v2 manifest
- normalized policy validation

## 11. 改进方法

### 10.1 数据层

- 继续补 event-heavy 样本，而不是无差别加数据
- 强化真正需要视觉纠偏的 turning / recovery / restart 样本
- 继续把 sim 采集设计成能制造“anchor 不够用”的场景

### 10.2 模型层

- 保持 anchor-residual 主线，不建议回退到 absolute-only
- previous_action condition 继续严格控制，不让它从多路径泄漏
- LoRA last-4 是当前比较稳的折中，不建议贸然 full finetune

### 10.3 验证层

- 统一 normalized policy validation schema
- 明确 baseline registry
- 固定 sample-id 子集，防止不同轮比较时样本不等价
- 更多看 failure slices，不只看 global MAE

## 12. 你的思考与方法论

从会话痕迹看，你这条项目线里有几条很稳定的思考方式。

### 11.1 强烈偏好最小侵入

你从一开始就不想为了新路线破坏已有 collector、bridge、safety shell、baseline 流程。后续 robotics lane 独立出来，正好吻合这个偏好。

### 11.2 非常在意“是不是假成功”

你不满足于看 loss 下降，而是反复追问：

- 模型有没有用上视觉和指令
- 是否只是比 previous_action 更差
- validation 配置到底对不对

这使得项目没有停在一个好看的 train loss 上。

### 11.3 对数据质量的敏感度很高

从真实数据的 turning onset，到 sim 数据的穿模、静止帧、动态加载、目标对比度、停车过晚、起停平滑，你一直在盯“标签是不是值得学”。这对这个项目是正确的，因为当前最核心的问题正是数据是否迫使模型做决策。

### 11.4 你接受逐步演化，而不是一次性追求论文形态

最初从回归头开始想，后来切到 robotics lane，再切到 delta，再切到 anchor-residual，再到 targeted-v2。这说明你更看重“让每一阶段的错误暴露出来”，而不是一开始就写一个看起来最先进的方案。

推断：这也是为什么这条项目线虽然还没完全收口，但已经积累了相当扎实的工程与实验痕迹。

## 13. 当前空缺与合理推断

### 12.1 本地缺失的地方

- 没有在当前仓里找到一份完整的、最终定稿的 Stage 1.5 实验报告 JSON
- 没有找到一份已经明确宣告“最新 weighted / targeted-v2 模型正式赢过 baseline”的最终总结
- 真实主线某些 checkpoint 细节只存在于日志与文档，不在当前本地 outputs 里

### 12.2 推断

推断：如果没有这些额外修正，项目大概率会停在“loss 很好看，但 rollout 还是 hold-policy”的状态。你后面把大量时间花在 data contract、previous_action、event-heavy、targeted-v2 上，说明你已经意识到模型层改动的收益开始小于数据与验证层修正。

推断：当前这条 sim turnpilot + anchor-residual + weighted subset 的路线，本质上是在试图制造一种训练分布，使 residual 预测真正必要，而不是总能靠 anchor 混过去。

## 14. 当前建议

如果把“从开始到现在”的脉络压成一句话，这个项目的演化是：

从“把 `LLaDA-V` 接进 Go2 高层速度控制”出发，经过 real-only baseline、Stage 1.5/1.6 暴露 previous-action shortcut、再到 sim turnpilot / anchor-residual / targeted-v2 这条更受控的修正路线，项目已经从搭链路阶段进入了“如何让模型在真正的决策样本和 policy validation 上胜过简单惯性基线”的阶段。

当前最稳妥的判断是：

- 工程架构已经基本对了
- baseline 问题已经被识别得很清楚
- 成败关键现在在数据设计与验证闭环，而不是再发明一套全新模型

## 15. 附录：关键证据索引

### 14.1 主仓与文档

- `README.md`
- `docs/training_plan_current_v1.md`
- `docs/runbooks/processed_dataset_contract.md`
- `docs/runbooks/raw_session_contract.md`

### 14.2 当前模型与训练实现

- `LLaDA-V/train/robotics/README.md`
- `LLaDA-V/train/robotics/config.py`
- `LLaDA-V/train/robotics/anchors.py`
- `LLaDA-V/train/robotics/modeling/llada_vla_velocity.py`
- `LLaDA-V/train/robotics/modeling/action_heads.py`
- `LLaDA-V/train/robotics/inference/velocity_diffusion_decoder.py`
- `LLaDA-V/train/robotics/eval/stage1_5_policy_validation.py`
- `LLaDA-V/train/robotics/eval/eval_velocity.py`
- `LLaDA-V/train/robotics/train_velocity.py`

### 14.3 关键配置

- `configs/llada_vla_go2_real_projector_only.yaml`
- `configs/llada_vla_go2_real_stage1_6_delta.yaml`
- `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual.yaml`
- `configs/workspace.yaml`

### 14.4 关键实验产物

- `outputs/data_audit/current_v1_real_vs_sim/audit_summary.json`
- `outputs/data_audit/event_rich_candidates/train/summary.json`
- `outputs/data_audit/current_v2_sim_today_main_turnpilot_20260425/audit_summary.json`
- `outputs/sim_policy_validation/train_current_v1_sim_office_clean_projector_eval10_noplot/metrics.json`
- `outputs/sim_policy_validation/train_current_v1_sim_office_clean_projector_eval200_noplot/metrics.json`
- `outputs/local_val4_eval_smoke.json`

### 14.5 计划、访谈、日志

- `.omx/specs/deep-interview-llada-vla-go2-velocity-control.md`
- `.omx/plans/ralplan-llada-vla-go2-velocity-control.md`
- `.omx/context/llada-vla-go2-velocity-control-20260419T112023Z.md`
- `.omx/interviews/llada-vla-go2-velocity-control-20260419T113204Z.md`
- `.omx/logs/turns-2026-04-19.jsonl`
- `.omx/logs/turns-2026-04-21.jsonl`
- `.omx/logs/turns-2026-04-24.jsonl`
- `.omx/logs/turns-2026-04-25.jsonl`
- `.codex/history.jsonl`

### 14.6 归档与旧痕迹

- `trash/LLaDA-V`
- `trash/go2_vla_collector_cleanup_20260419_211924`

## 16. 近期实验记录入口与维护约定

从 2026-04-26 起，TurnPilot / Targeted V2 / policy validation 的大计划、关键实验结论和后续关键节点记录到：

- `docs/project_notes/turnpilot_targeted_v2_experiment_log.md`

后续每次出现重要实验选择、关键指标结论、失败归因、路径或配置变更、用户确认过的实验决策，都必须追加到该笔记，避免只停留在会话上下文中。
