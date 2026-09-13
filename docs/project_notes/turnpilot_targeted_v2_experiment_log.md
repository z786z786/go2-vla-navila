# TurnPilot Targeted V2 近期计划与实验结论记录

> 维护规则
>
> - 这是 `llada-vla-go2` 项目中 TurnPilot / policy-validation / targeted training 相关的关键计划与实验结论笔记。
> - 后续每次出现重要实验选择、关键节点、指标结论、失败归因、路径变更、验证 gate 变更，都要追加到本文件。
> - 记录必须包含：日期、计划/动作、关键路径、验证样本或数据来源、核心指标、结论、下一步。
> - 不覆盖旧记录；新结论以追加方式写入。

## 2026-04-26: Targeted V2 总计划

### 背景

当前 weighted checkpoint-450 已固化为可追溯 baseline。下一轮目标不是继续盲训，而是把 policy validation 的失败样本归因转成可复现的 manifest 规则，并用统一 normalized JSON 比较 baseline 与 V2 结果。

### 核心原则

- 固定 validation split 和 fixed validation sample ids，不把 validation 失败样本直接加入训练，避免指标污染。
- V2 默认只使用已批准 train pool。
- baseline 与 V2 结果统一保存为 normalized JSON，后续 diff 只比较 normalized JSON，不直接读不同格式的 raw metrics。
- 所有输出新建目录，绝不覆盖旧目录。
- 训练/评估/数据处理前必须 review 代码、配置、路径，并 debug 关键 IO。
- checkpoint、dataset、lane/config、entrypoint、output dir、resume/overwrite 等实验语义选择，执行前需要确认。

### Baseline 固化

baseline 使用当前 weighted checkpoint-450 的父目录作为 inference artifact，checkpoint-450 子目录只作为 resume 来源。

- resume checkpoint: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_2gpu_priority_weighted/checkpoint-450`
- inference artifact: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_2gpu_priority_weighted`
- baseline normalized: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/baseline_checkpoint_450.metrics.normalized.json`
- baseline registry: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/baseline_checkpoint_450.registry.json`
- fixed validation ids: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/validation_sample_ids.txt`

重要发现：`checkpoint-450` 子目录包含 optimizer/scheduler/rng/trainer_state，适合 resume；但缺少 `robotics_velocity_config.json` / `robotics_velocity_head.bin`，不适合作 inference `MODEL_PATH`。父目录才是正确 inference artifact。

### 数据与配置

- 原始已批准 train pool: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_20260425`
- 训练配置: `/root/autodl-tmp/llada-vla-go2/configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual.yaml`
- 关键配置: `horizon_k=4`, `chunk_size=4`, `action_target_mode=anchored_residual`, `anchor_mode=repeat_prev`, `use_prev_action_as_condition=true`
- 数据 raw chunks 为 K=6，训练配置裁剪到 K=4；这是为了 checkpoint 兼容而有意保留。

### Manifest 规则

失败归因从 policy validation 的 `metrics.json` 和 `predictions.jsonl` 读取，只用固定 validation sample ids 做归因，不把这些样本加入 V2 训练。

默认 bucket 与权重：

- high 3.0: `stop_should_hold`, `stop_after_motion`, `micro_yaw_at_stop`, `instruction_forward_vs_stop`, `instruction_left_vs_right`, `instruction_stop_vs_turn`
- medium-high 2.0: `low_vx_but_should_move`, `wz_change`
- medium 1.5: `first_step_commit`, `residual_recovery`
- low-medium 1.15: `normal_cruise`

多标签样本默认取最大权重，不叠乘，避免权重爆炸。

matched counterexample 规则：优先找同 `contrast_group_id`、同/近 `trajectory_step_index` 的配对，构造 `forward_vs_stop`、`left_vs_right`、`stop_vs_turn`。如果 train pool 找不到配对，只记录 unmatched，不合成伪样本，不使用 validation 样本补进去。

### V2 manifest 与 weighted dataset 产物

- manifest: `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_parent_artifact_20260426_143200/targeted_v2_priority_manifest_train.jsonl`
- manifest summary: `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_parent_artifact_20260426_143200/targeted_v2_priority_manifest_summary_train.json`
- weighted dataset: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_weighted_parent_artifact_20260426_143200`
- rows: train `35129`, val `1985`
- validation leak count: `0`

## 2026-04-26: Targeted V2 短训计划与训练结果

### 训练计划

从 weighted best checkpoint-450 继续训练，保存并评估三个短训点：

- `plus025`: +0.25 epoch
- `plus050`: +0.5 epoch
- `plus100`: +1.0 epoch

模型选择使用 policy validation，不使用 train loss。

### 训练实现与修正

训练入口和 wrapper 已支持：

- `--resume-checkpoint`
- `--num-train-epochs`
- `--max-steps`
- wrapper 透传 `RESUME_CHECKPOINT`, `MAX_STEPS`, `NUM_TRAIN_EPOCHS`

PyTorch 2.7 resume 遇到 `_pickle.UnpicklingError: Weights only load failed`。已验证 workaround：

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`

### 训练产物

训练链脚本：

- `/root/autodl-tmp/llada-vla-go2/scripts/run_targeted_v2_training_chain_20260426_143200_retry1.sh`
- master log: `/root/autodl-tmp/llada-vla-go2/outputs/run_targeted_v2_training_chain_20260426_143200_retry1.master.log`

三个 checkpoint 均完成，且 inference artifacts 完整：

- `plus025`: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus025_20260426_143200_retry1`
  - `eval_loss@550 = 0.7910719513893127`
  - final train loss approx `0.169`
- `plus050`: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus050_20260426_143200_retry1`
  - `eval_loss@700 = 0.7018317580223083`
  - final train loss approx `0.2545`
- `plus100`: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus100_20260426_143200_retry1`
  - `eval_loss@800 = 0.5612765550613403`
  - final train loss approx `0.3106`

训练 retry1 后未见 OOM/CUDA error。

## 2026-04-26: Fixed-ID Policy Validation 计划与执行

### 验证计划

对 `plus025` / `plus050` / `plus100` 使用同一组 fixed validation sample ids 跑 policy validation，输出 normalized metrics 和 diff：

- `metrics.json`
- `predictions.jsonl`
- `v2_checkpoint_<label>.metrics.normalized.json`
- `diff_vs_baseline_checkpoint_450.normalized.json`

固定 validation ids：

- `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/validation_sample_ids.txt`

### 执行策略

最初准备顺序链：

- `/root/autodl-tmp/llada-vla-go2/scripts/run_targeted_v2_fixed_id_policy_validation_20260426_143200_retry1.sh`

后续为了利用两张 GPU，改为并行：

- GPU0: `plus025` 完成后继续 `plus100`
- GPU1: 单独运行 `plus050`

原顺序链在 `plus025` 后检测到 `plus050` 输出/日志已存在，因此按“绝不覆盖”保护拒绝继续。这是预期行为，不是验证失败。

### 验证输出

- `plus025`: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus025_fixed_ids_20260426_143200_retry1`
- `plus050`: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus050_fixed_ids_20260426_143200_retry1`
- `plus100`: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_fixed_ids_20260426_143200_retry1`

## 2026-04-26: Targeted V2 Policy Validation 结果与结论

### 总体指标

| checkpoint | model_mae lower | improvement_vs_prev_hold higher | relative_improvement | image_gain | instruction_gain | decision_heavy | low_vx_or_stop | turning | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline ckpt450 | 0.015920 | 0.006388 | 28.64% | 0.008842 | -0.000448 | 0.013597 | -0.001614 | 0.015128 | fail |
| plus025 | 0.015255 | 0.007054 | 31.62% | 0.011463 | -0.000930 | 0.014524 | -0.000818 | 0.016671 | fail |
| plus050 | 0.013184 | 0.009124 | 40.90% | 0.014797 | -0.000120 | 0.018690 | -0.000596 | 0.020398 | fail |
| plus100 | 0.011901 | 0.010407 | 46.65% | 0.016868 | -0.000812 | 0.020807 | 0.000303 | 0.022909 | fail |

### 相对 baseline 的主要变化

- `plus025`: `model_mae` 降低 `0.000665`，整体 hold improvement 增加 `0.000665`。
- `plus050`: `model_mae` 降低 `0.002736`，整体 hold improvement 增加 `0.002736`。
- `plus100`: `model_mae` 降低 `0.004019`，整体 hold improvement 增加 `0.004019`，三者最好。

### 关键 subset

| subset | baseline | plus025 | plus050 | plus100 | 当前最好 |
|---|---:|---:|---:|---:|---|
| decision_heavy | 0.013597 | 0.014524 | 0.018690 | 0.020807 | plus100 |
| low_vx_or_stop | -0.001614 | -0.000818 | -0.000596 | 0.000303 | plus100 |
| turning | 0.015128 | 0.016671 | 0.020398 | 0.022909 | plus100 |
| wz_change | 0.002495 | 0.005704 | 0.015650 | 0.020258 | plus100 |
| first_step_change | 0.028040 | 0.031058 | 0.037801 | 0.040945 | plus100 |
| vx_change | 0.021925 | 0.023202 | 0.030452 | 0.033195 | plus100 |
| large_delta_wz | -0.007247 | -0.004344 | 0.004245 | 0.004117 | plus050 slightly |
| large_residual_correction | 0.019190 | 0.020271 | 0.026381 | 0.029055 | plus100 |

### Gate 结论

通过项：

- `improvement_vs_previous_action_hold > 0`: 三个 V2 checkpoint 都通过，`plus100` 最强。
- `image_conditioning_gain >= 0.005`: 三个 V2 checkpoint 都通过，`plus100` 最强。
- `decision_heavy_improvement_vs_previous_action_hold > 0`: 三个 V2 checkpoint 都通过，`plus100` 最强。
- `turning_improvement_vs_previous_action_hold > 0`: 三个 V2 checkpoint 都通过，`plus100` 最强。
- `low_vx_or_stop >= -0.002`: 三个 V2 checkpoint 都通过，`plus100` 转正到 `0.000303`。

失败项：

- `instruction_conditioning_gain > 0`: 三个 V2 checkpoint 都失败。
- `plus050` 最接近通过：`-0.000120`。
- `plus100` action 指标最好，但 instruction gain 为 `-0.000812`，比 `plus050` 差。

### 结论

- 三个 V2 checkpoint 都明显优于 baseline ckpt450。
- 严格 gate 都没有完全通过，唯一失败项是 `instruction_conditioning_gain > 0`。
- 如果按 policy validation 主指标选模型，`plus100` 是当前最佳候选。
- 如果按 strict gate pass 定义，V2 还不能宣布通过。
- `plus050` 是 instruction gate 最接近的候选；`plus100` 是主 policy/action 指标最强的候选。

## 2026-04-26: 下一步计划

### 不建议立即继续盲训

当前主要瓶颈不是 global MAE，而是 `instruction_conditioning_gain` 为负。继续训练可能继续提升主指标，但不一定解决 instruction gate。

### 优先做 prev_action prompt ablation

A/B 设计：

- A: anchor 使用 prev_action，prompt 含 `<previous_action>`。
- B: anchor 仍使用 prev_action，但 prompt 不含 `<previous_action>`。

固定项：dataset、validation ids、seed、decode steps、horizon_k、model checkpoint、output schema。

建议先跑：

- `plus100`: 主指标最好。
- `plus050`: `instruction_conditioning_gain = -0.000120`，最接近过 gate。

判定规则：

- 如果 `plus100 + no_prev_action_prompt` 让 `instruction_conditioning_gain > 0`，且主指标、turning、decision_heavy 不明显回退，则采用 `plus100+B`。
- 如果 `plus050+B` 过 instruction gate，但主指标弱于 `plus100+B`，再比较是否值得折中。
- 如果 B 仍不行，说明问题不只是 prompt previous_action，需要进入 V2.1 数据/权重调整。

### Instruction failure slice 分析

重点切片：

- `forward_vs_stop`
- `left_vs_right`
- `stop_vs_turn`
- `instruction_forward_vs_stop`
- `instruction_left_vs_right`
- `instruction_stop_vs_turn`
- low-vx / stop 决策样本
- wz / turning 样本

输出目标：

- 哪些 instruction bucket 拉低 gain。
- 模型在 A/B 中是否更依赖 previous_action 而不是 instruction。
- prediction vs GT vs previous_action 的三方对比。
- 是否存在“模型更像 previous_action hold”的失败模式。

### 如果进入 V2.1

仍不加入 validation 失败样本到训练。优先从 train pool 中加强 matched counterexample：

- stop vs forward
- stop vs turn
- left vs right

可能的权重调整：

- `instruction_forward_vs_stop`: `3.0 -> 3.5`
- `instruction_left_vs_right`: `3.0 -> 3.5`
- `instruction_stop_vs_turn`: `3.0 -> 3.5`
- `normal_cruise`: `1.15 -> 1.0` 或下采样
- 增加 `previous_action_hold would be wrong` 类 bucket

V2.1 短训建议从 `plus100` 继续，而不是回到 ckpt450：

- `plus100 + 0.25 epoch`
- `plus100 + 0.5 epoch`
- 可选 `plus100 + 1.0 epoch`，仅当前两个点 instruction gain 有改善趋势时再跑。

## 未来记录约定

以后本项目每次出现以下情况，都要追加到本文件：

- 新的大计划或训练/验证方案。
- 实验语义选择变更：checkpoint、dataset、lane/config、entrypoint、output dir、resume/overwrite。
- 新 manifest、weighted dataset、registry、normalized metrics、diff 产物。
- policy validation、ablation、failure slice 的关键结果。
- 训练失败、评估失败、路径错误、数据泄漏、schema 不一致等关键问题及修正。
- 用户确认过的重要实验决策。

记录格式建议：

```markdown
## YYYY-MM-DD: 标题

### 背景
### 计划/动作
### 关键路径
### 核心指标/结果
### 结论
### 下一步
```

## 2026-04-26: no-prev-action-prompt ablation 启动记录

### 背景

Targeted V2 的 `plus025` / `plus050` / `plus100` 在主 policy 指标上均优于 baseline，但 strict gate 唯一失败项是 `instruction_conditioning_gain > 0`。因此按计划优先做 prompt ablation，而不是继续盲训。

### 计划/动作

执行 B 方案：保留 prev_action anchor，但从 prompt 中移除 `<previous_action>`。当前 5090 服务器只剩一张 GPU，因此顺序跑：

- `plus100` no-prevprompt fixed-id policy validation
- `plus050` no-prevprompt fixed-id policy validation

### 关键路径

远端服务器：`ssh -p 47766 root@connect.bjb2.seetacloud.com`

新增 no-prevprompt eval config：

- `/root/autodl-tmp/llada-vla-go2/configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_eval_20260426_205133.yaml`

新增 eval-only artifact，使用 symlink 指向原 `model.safetensors` / `robotics_velocity_head.bin`，只修改 `robotics_velocity_config.json` 中 `use_prev_action_as_condition=false`：

- `/root/autodl-tmp/llada-vla-go2/outputs/eval_artifact_targeted_v2_plus100_no_prevprompt_20260426_205133`
- `/root/autodl-tmp/llada-vla-go2/outputs/eval_artifact_targeted_v2_plus050_no_prevprompt_20260426_205133`

固定 validation ids 不变：

- `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/validation_sample_ids.txt`

计划输出：

- `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_no_prevprompt_fixed_ids_20260426_205133_retry1`
- `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus050_no_prevprompt_fixed_ids_20260426_205133_retry1`

运行脚本：

- `/root/autodl-tmp/llada-vla-go2/scripts/run_targeted_v2_no_prevprompt_fixed_id_policy_validation_20260426_205133_retry1.sh`
- master log: `/root/autodl-tmp/llada-vla-go2/outputs/run_targeted_v2_no_prevprompt_fixed_id_policy_validation_20260426_205133_retry1.launch2.master.log`

### Preflight / Debug 结果

- GPU precheck：当前只有 GPU0，空闲可用。
- fixed validation ids: `200`，val rows: `1985`，missing: `0`。
- sample ids hash: `4726f80ac96ebe559ca603846e589a00f90e8d67fe98ea85c5fe8be420507303`。
- debug sample: `dual_target_room_main_s0100_b00_gpu0:ep_000111:000009`。
- A prompt 含 `<previous_action>`，B prompt 不含 `<previous_action>`。
- anchor 保持相同，`anchor_mode=repeat_prev`，`action_target_mode=residual`，`horizon_k=4`，action tokens `12`。

### 注意事项

第一次启动脚本时 pgrep precheck 误匹配到了正在写入 heredoc 的父 shell 命令，触发安全拒绝；未产生 validation 输出。随后用 `launch2.master.log` 重新启动，当前运行 `plus100` no-prevprompt validation。

### 下一步

等待 `plus100` 与 `plus050` 生成 normalized JSON 后，对比：

- B vs baseline ckpt450
- B vs 对应 prompt-A checkpoint

重点看 `instruction_conditioning_gain` 是否转正，以及主指标、turning、decision_heavy 是否明显回退。

## 2026-04-26: Sim 近目标收尾阶段视觉退化观察

### 用户观察

在 sim 采集数据中，Go2 快到达目标时指令通常保持不变；但由于目标物体/区域过大，机器人最后收尾阶段的整个视野几乎都被目标颜色块占满，图像在连续帧之间基本没有变化。

### 影响判断

这个观察会影响对 image/instruction conditioning 指标的解释：近目标收尾样本里，即使图像输入存在，图像也可能缺乏可区分的几何/位姿变化信息；模型更容易依赖 previous_action、anchor、速度状态或数据先验，而不是从图像中判断“是否该减速/停止/转向”。这类样本如果占比高，可能让 image ablation / instruction ablation 的指标表现与真实决策需求不完全一致。

### 后续处理建议

- failure slice 中单独标记 near-target / target-color-dominant / low-image-change 样本。
- 对收尾阶段不要只看 global image conditioning gain，要分开看 approach 阶段和 terminal/near-goal 阶段。
- 后续采集可考虑让目标视觉更有边界/尺度变化，或补充目标边缘、相对位置、距离/深度、终止条件相关信号。
- V2.1 manifest 若继续做，应避免把“整帧都是目标色块且指令不变”的样本简单当成高质量视觉决策样本；这类样本更适合归入 terminal-control / stop-deceleration 专门 bucket。

## 2026-04-26: no-prev-action-prompt ablation 完成初步结果

### 结果

`plus100` 和 `plus050` 的 no-prevprompt fixed-id policy validation 已完成，均生成 `metrics.json`、`predictions.jsonl`、normalized JSON 和 diff JSON。

关键对比：

| checkpoint | prompt | model_mae | improvement_vs_prev_hold | image_gain | instruction_gain | gate |
|---|---|---:|---:|---:|---:|---|
| plus100 | A 含 previous_action | 0.011901 | 0.010407 | 0.016868 | -0.000812 | fail |
| plus100 | B 去 previous_action | 0.012755 | 0.009553 | 0.017990 | 0.000816 | pass |
| plus050 | A 含 previous_action | 0.013184 | 0.009124 | 0.014797 | -0.000120 | fail |
| plus050 | B 去 previous_action | 0.014124 | 0.008185 | 0.014778 | 0.001150 | pass |

### 结论

去掉 prompt 里的 `<previous_action>` 后，`instruction_conditioning_gain` 转正，两个候选都通过 strict gates；代价是主 MAE 和 previous-action-hold improvement 有小幅回退。当前更推荐 `plus100+B`，因为它在通过 gate 的同时主 policy 指标仍强于 `plus050+B`。


## 2026-04-26: near-target static-image 观察与新对话交接信息

### 用户观察

在 sim 采集数据中，Go2 快到达目标时指令通常不变；由于目标过大，Go2 的整个视野可能几乎都是目标颜色块。也就是说，在最后收尾阶段 image 基本没有变化，instruction 也没有变化，但动作标签仍需要从前进切到减速、停止或微调。

### 对当前结果的解释

这个观察能解释为什么 Targeted V2 A 方案主 policy 指标变好，但 `instruction_conditioning_gain` 仍为负：

- near-target / terminal 阶段中，instruction 的边际信息很低，因为指令仍是同一句，例如 `go to the door`。
- 如果画面被目标颜色块填满，image 的边际变化也很低。
- 监督要求动作变化，但变化更可能由 previous_action、状态、隐含距离/接近程度决定，而不是由文本变化决定。
- 因此模型容易依赖 previous_action anchor 或动作/状态先验，导致 policy MAE 变好，但 instruction shuffle/ablation 未必变差。

这说明 instruction gate 的失败不能简单解释为“模型完全不听指令”，还可能是 validation 分布里包含大量 instruction/image 边际信息很低的 terminal saturated frames。

### 与 no-prevprompt ablation 的关系

no-prevprompt B 方案已经证明 prompt 里的 `<previous_action>` 会压制 instruction gate：

- `plus100 A`: `instruction_conditioning_gain=-0.000812`, strict gate fail。
- `plus100 B`: `instruction_conditioning_gain=0.000816`, strict gate pass。
- `plus050 A`: `instruction_conditioning_gain=-0.000120`, strict gate fail。
- `plus050 B`: `instruction_conditioning_gain=0.001150`, strict gate pass。

但 B 方案也让主 action 指标小幅回退：

- `plus100 model_mae`: `0.011901 -> 0.012755`。
- `plus050 model_mae`: `0.013184 -> 0.014124`。

当前推荐模型/配置仍是 `plus100 + no-prevprompt`，因为它通过 strict gate，且主 policy 指标强于 `plus050 + no-prevprompt`。

### 下一步建议

优先做 lightweight slice analysis，不重训：

- 构造或近似 `near_target_color_saturation` / `large_target_fill_ratio` / `late_approach_static_image` / `same_instruction_low_image_delta` / `terminal_settle_phase` 切片。
- 读取 fixed-id validation 的 `predictions.jsonl`，回查 val jsonl 和 image。
- 计算目标颜色块/主色块占比、RGB histogram 或 image embedding 相邻变化、trajectory step index、GT vx/wz 是否从运动转停止。
- 分别报告 terminal/static-image 与 non-terminal 两组的 instruction gain、image gain、prev-action-hold improvement。
- 判断 instruction gate 是整体问题，还是主要被 terminal saturated frames 稀释。

如果后续进入 V2.1 数据策略，应避免 terminal saturated frames 在 weighted manifest 中过度放大；同时增加真正需要 instruction 区分的 counterexamples，例如同画面不同 instruction、多目标同时可见、同 instruction 但目标相对位置不同。

## 2026-04-26: 新 Codex 对话交接摘要

### 开始位置

新 Codex 对话应从阅读以下文件开始：

- `docs/llada_vla_go2_project_notes.md`
- `docs/project_notes/turnpilot_targeted_v2_experiment_log.md`
- `/home/wxh/.codex/memories/PROFILE.md`
- `/home/wxh/.codex/memories/ACTIVE.md`

### 必须知道的硬规则

- 默认中文回复。
- 任何训练、评估、数据处理前必须 review 相关代码/配置/路径，并 debug 关键 IO。
- 涉及 checkpoint、dataset、lane/config、entrypoint、output dir、resume/overwrite 等实验语义选择时，执行前必须确认。
- 永远不要覆盖旧结果；必须新建 timestamp/suffix 输出目录。
- 当前 5090 远端：`ssh -p 47766 root@connect.bjb2.seetacloud.com`。
- 当前远端 repo：`/root/autodl-tmp/llada-vla-go2`。
- 当前远端 Python：`/root/miniconda3/envs/llada-vla/bin/python`。
- 目前 5090 只有一张 GPU0 可用。

### 当前最重要结论

- Targeted V2 三个 checkpoint 都明显优于 baseline ckpt450，但 A 方案 strict gate 因 `instruction_conditioning_gain` 为负而失败。
- 去掉 prompt 中 `<previous_action>` 的 B 方案让 `plus100` 和 `plus050` 都通过 strict gate。
- 推荐候选：`plus100 + no-prevprompt`。
- `plus100+B` 指标：`model_mae=0.012755`, `improvement_vs_previous_action_hold=0.009553`, `image_conditioning_gain=0.017990`, `instruction_conditioning_gain=0.000816`, strict gate pass。
- `plus050+B` 指标：`model_mae=0.014124`, `improvement_vs_previous_action_hold=0.008185`, `image_conditioning_gain=0.014778`, `instruction_conditioning_gain=0.001150`, strict gate pass。
- `plus100+A` 仍有最好 MAE `0.011901`，但 instruction gate fail；因此不作为 strict-pass 默认配置。

### 当前关键路径

Baseline / fixed ids：

- baseline normalized: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/baseline_checkpoint_450.metrics.normalized.json`
- fixed validation ids: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/validation_sample_ids.txt`

Prompt-A results：

- plus100 A: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_fixed_ids_20260426_143200_retry1/v2_checkpoint_plus100.metrics.normalized.json`
- plus050 A: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus050_fixed_ids_20260426_143200_retry1/v2_checkpoint_plus050.metrics.normalized.json`

No-prevprompt B artifacts/results：

- config: `/root/autodl-tmp/llada-vla-go2/configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_eval_20260426_205133.yaml`
- plus100 B artifact: `/root/autodl-tmp/llada-vla-go2/outputs/eval_artifact_targeted_v2_plus100_no_prevprompt_20260426_205133`
- plus050 B artifact: `/root/autodl-tmp/llada-vla-go2/outputs/eval_artifact_targeted_v2_plus050_no_prevprompt_20260426_205133`
- plus100 B normalized: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_no_prevprompt_fixed_ids_20260426_205133_retry1/v2_checkpoint_plus100_no_prevprompt.metrics.normalized.json`
- plus050 B normalized: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus050_no_prevprompt_fixed_ids_20260426_205133_retry1/v2_checkpoint_plus050_no_prevprompt.metrics.normalized.json`

### 建议下一个任务

从这里开始：实现并运行 near-target / static-image slice analysis。目标不是重训，而是解释 instruction gate 与 terminal saturated frames 的关系。

最小执行路线：

1. Review `stage1_5_policy_validation.py`、normalized metrics schema、val jsonl 字段和 image 路径。
2. 写一个只读分析脚本或 notebook-style script，输入 B/A 的 `predictions.jsonl` 和 val jsonl。
3. 计算 image 主色块/低变化/terminal step proxy。
4. 输出每个 slice 的 model MAE、instruction gain、image gain、prev-hold improvement。
5. 把结果追加到本笔记。

## 2026-04-26: near-target / static-image slice analysis 完成

### 背景

用户观察到 sim 快到目标时 instruction 通常不变，且画面可能被目标颜色块填满、相邻帧变化很低。该分析用于判断 `instruction_conditioning_gain` 失败/变弱是否主要被 terminal saturated/static frames 稀释；本轮只读 fixed-id predictions、val jsonl 和 image，不重训。

### Preflight / Debug

- 已 review `LLaDA-V/train/robotics/eval/stage1_5_policy_validation.py`：`image_conditioning_gain = image_shuffle_mae - model_mae`，`instruction_conditioning_gain = instruction_shuffle_mae - model_mae`，`improvement_vs_previous_action_hold = previous_action_hold_mae - model_mae`。
- 已 review normalized schema：strict gate 依赖全局 `summary` 与 `subsets`，本分析保持相同的 gain/improvement 语义，但按 terminal/static-image proxy 重新切片。
- 输入 val jsonl：`/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_weighted_parent_artifact_20260426_143200/val.jsonl`。
- 输入 image folder：`/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_weighted_parent_artifact_20260426_143200/images`，该路径是 symlink，实际指向 `current_v2_sim_today_main_turnpilot_20260425`；固定 ids 的 image missing count 为 `0`。
- fixed validation ids：`/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/validation_sample_ids.txt`，样本数 `200`，valid action tokens `2343`。
- 新增只读脚本：`scripts/analyze_near_target_static_image_slices.py`，远端同路径也已同步；脚本检查 output dir 非空时拒绝写入，避免覆盖旧结果。
- 小样本 debug 输出：`/root/autodl-tmp/llada-vla-go2/outputs/near_target_static_image_slice_analysis_debug_20260426_212605`。

### 关键路径

最终输出目录：

- `/root/autodl-tmp/llada-vla-go2/outputs/near_target_static_image_slice_analysis_fixed_ids_20260426_212708_globalmean`

核心输出文件：

- `near_target_static_image_slice_metrics.json`
- `sample_features.jsonl`

输入 predictions：

- baseline ckpt450: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/predictions.jsonl`
- plus100 A: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_fixed_ids_20260426_143200_retry1/predictions.jsonl`
- plus100 B: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_no_prevprompt_fixed_ids_20260426_205133_retry1/predictions.jsonl`
- plus050 A: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus050_fixed_ids_20260426_143200_retry1/predictions.jsonl`
- plus050 B: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus050_no_prevprompt_fixed_ids_20260426_205133_retry1/predictions.jsonl`

### Slice 定义

本轮是 proxy 分析，不把这些阈值写成最终 gate：

- `terminal_progress_ge_75`: `trajectory_step_index / (trajectory_length - 1) >= 0.75`。
- `static_image_low_delta_q25`: 当前帧与相邻帧 RGB L1 mean delta 位于 fixed-id 样本最低 25%，阈值 `0.0012716115`；图像下采样到 `96px` 计算。
- `target_color_dominant_q75`: 量化主色块占比位于最高 25%，阈值 `0.9842664931`；这类样本通常非常接近“整帧被单一目标/背景色块占满”。
- `terminal_action_settle`: 未来 chunk 最后一帧 `abs(vx) <= 0.06` 且 `abs(wz) <= 0.08`。
- `terminal_static_proxy`: `terminal_progress_ge_75`，或 `static_image_low_delta_q25 && target_color_dominant_q75`，或 `terminal_action_settle && static_image_low_delta_q25`。

Slice counts：

| slice | count |
|---|---:|
| all | 200 |
| terminal_static_proxy | 71 |
| non_terminal_static_proxy | 129 |
| terminal_progress_ge_75 | 50 |
| static_image_low_delta_q25 | 50 |
| target_color_dominant_q75 | 50 |
| terminal_action_settle | 53 |

### 核心指标：terminal/static vs non-terminal/static

| run | slice | count | model_mae | hold_improve | image_gain | instruction_gain |
|---|---|---:|---:|---:|---:|---:|
| baseline | terminal_static | 71 | 0.006509 | 0.000012 | 0.008085 | -0.000907 |
| baseline | non_terminal_static | 129 | 0.020753 | 0.009663 | 0.009231 | -0.000212 |
| plus050 A | terminal_static | 71 | 0.005243 | 0.001278 | 0.017340 | -0.001078 |
| plus050 A | non_terminal_static | 129 | 0.017262 | 0.013153 | 0.013490 | 0.000372 |
| plus050 B | terminal_static | 71 | 0.005480 | 0.001041 | 0.018802 | -0.000266 |
| plus050 B | non_terminal_static | 129 | 0.018563 | 0.011853 | 0.012712 | 0.001877 |
| plus100 A | terminal_static | 71 | 0.004304 | 0.002217 | 0.018689 | -0.000105 |
| plus100 A | non_terminal_static | 129 | 0.015803 | 0.014613 | 0.015933 | -0.001176 |
| plus100 B | terminal_static | 71 | 0.004336 | 0.002185 | 0.025795 | -0.000258 |
| plus100 B | non_terminal_static | 129 | 0.017079 | 0.013337 | 0.013982 | 0.001368 |

### 重要分解

`plus100+B` 的更细切片：

| slice | count | model_mae | hold_improve | image_gain | instruction_gain |
|---|---:|---:|---:|---:|---:|
| terminal_progress_ge_75 | 50 | 0.003013 | -0.000484 | 0.026377 | -0.000202 |
| terminal_progress_lt_75 | 150 | 0.015694 | 0.012580 | 0.015461 | 0.001124 |
| static_image_low_delta_q25 | 50 | 0.004977 | 0.003275 | 0.025499 | -0.000208 |
| non_static_image | 150 | 0.015102 | 0.011447 | 0.015725 | 0.001125 |
| target_color_dominant_q75 | 50 | 0.005574 | 0.003562 | 0.025669 | -0.000200 |
| non_target_color_dominant | 150 | 0.015032 | 0.011452 | 0.015556 | 0.001138 |
| terminal_action_settle | 53 | 0.002945 | -0.000367 | 0.025994 | -0.000195 |
| non_terminal_action_settle | 147 | 0.015975 | 0.012809 | 0.015363 | 0.001148 |

### 结论

- terminal/static proxy 样本确实会稀释 instruction gain：`plus100+B` 在 non-terminal/static 组 instruction gain 为 `+0.001368`，但 terminal/static 组为 `-0.000258`；`plus050+B` 也类似，non-terminal/static 为 `+0.001877`，terminal/static 为 `-0.000266`。
- 纯 terminal progress / terminal action settle 样本的 hold improvement 接近 0 或为负。例如 `plus100+B` 在 `terminal_progress_ge_75` 上为 `-0.000484`，在 `terminal_action_settle` 上为 `-0.000367`。这说明收尾/停止阶段 previous-action hold 已经很强，模型很难在这些样本上表现出额外 policy value。
- static/dominant slices 的 instruction gain 为负，但 non-static / non-dominant slices 为正。对 `plus100+B`：`static_image_low_delta_q25=-0.000208` vs `non_static_image=+0.001125`；`target_color_dominant_q75=-0.000200` vs `non_target_color_dominant=+0.001138`。
- 这不能完全解释 `plus100+A` 的 instruction gate fail：`plus100+A` 在 terminal/static 为 `-0.000105`，non-terminal/static 也为 `-0.001176`。因此 A 方案失败不只是 terminal saturated frames，prompt 中 `<previous_action>` 对 instruction 的压制仍是主要因素之一。
- image gain 没有被 terminal/static 稀释，反而在 terminal/static 和 saturated-image slices 上更高。可能原因是 image shuffle 换到其他目标/episode 后会破坏目标颜色/场景身份；因此 image gain 不能简单解释为“相邻帧几何变化有效”，它也可能包含目标颜色/场景辨识信号。

### 下一步

- 推荐继续把 `plus100+B` 作为 strict-pass 默认候选，但在报告 gate 时补充 terminal/static slice 说明：global instruction gain 是由 non-terminal/static 的正 gain 与 terminal/static 的负/近零 gain 混合得到。
- V2.1 如果调整 validation gate，建议至少分开报告 terminal/static 与 non-terminal/static，不要让 terminal saturated frames 独立决定 instruction gate。
- V2.1 如果调整数据权重，建议不要继续放大 terminal saturated/static frames；应增加真正需要 instruction 区分的 counterexamples，例如同画面不同 instruction、多目标同时可见、同 instruction 但目标相对位置不同。

## 2026-04-26: V2.1 前置流程确认

### 用户确认

用户同意当前 near-target/static-image slice analysis 的结论与下一步方向，并再次强调：后续任何训练、评估、数据处理或 V2.1 方案执行前，必须先做好代码 review、配置文件检查、路径/IO debug，并持续把关键过程与结果记录到本实验笔记。

### 执行约束

- 继续以 `plus100 + no-prevprompt` 作为 strict-pass 默认候选，除非后续有新的验证结果推翻。
- 后续进入 V2.1 前必须先确认实验语义选择：checkpoint、dataset、lane/config、entrypoint、output dir、resume/overwrite 策略。
- 输出目录必须新建 timestamp/suffix，不覆盖旧结果。
- validation gate / 数据权重调整前，优先复核 `stage1_5_policy_validation.py`、相关 config、manifest schema、fixed-id 样本、image 路径和 predictions schema。

## 2026-04-26: V2.1 preflight review 与待确认方案

### 动作

继续 V2.1 前只读 preflight：review 代码、配置、路径、现有产物，并做关键 IO/schema debug。未启动训练、评估或新数据处理。

### 代码与配置 review 结果

- `stage1_5_policy_validation.py` 的 metric 语义已确认，slice analysis 已按全局 valid action token mean 对齐原始 `metrics.json`，`all` slice 可复现 normalized 指标。
- `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual.yaml` 当前 `use_prev_action_as_condition=true`；no-prevprompt eval config `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_eval_20260426_205133.yaml` 为 `false`。
- 当前推荐候选 `plus100+B` 是 eval-only no-prevprompt：原 plus100 训练产物的 `robotics_velocity_config.json` 仍为 `use_prev_action_as_condition=true`。
- 若 V2.1 改成 no-prevprompt 继续训练，需要显式创建训练 config 并确认是否使用 `use_prev_action_as_condition=false`。
- `train_velocity.py` 支持 `--resume-checkpoint`、`--init-checkpoint-path`、`--num-train-epochs`、`--max-steps`。若从 plus100 parent artifact 用 `--init-checkpoint-path` 初始化，必须注意 base config 里 `reset_action_modules_after_load=true`，这会重置 action modules；V2.1 若采用 init 方式，应把该项改为 `false`。若用 `--resume-checkpoint checkpoint-950`，Trainer 会恢复 optimizer/scheduler/trainer_state，但这是在旧 prompt/data setting 上继续，是否适合 V2.1 需要确认。
- 发现并修复安全问题：`scripts/build_weighted_train_jsonl.py` 原先在 `output_root.exists()` 时直接 `shutil.rmtree(output_root)`；已改为默认拒绝已存在 output root，只有显式 `--overwrite` 才删除。`scripts/build_targeted_v2_manifest_from_failures.py` 也已改为拒绝覆盖既有 manifest/summary/validation ids 输出。两脚本本地与远端均 `py_compile` 通过，远端已同步。

### IO / schema debug 结果

当前 V2 train pool：

- path: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_20260425/train.jsonl`
- rows: `18846`
- target label: door `9452`, dark gray suitcase `9394`
- instructions: `navigate/move/approach/go to the door` 与 `navigate/move/approach/go to the dark gray suitcase`，共 8 类 phrasing。
- contrast groups: `82` 个，全部是 door vs suitcase 多目标/多指令组。

Fixed-id validation：

- val fixed rows: `200`
- target label: dark gray suitcase `103`, door `97`
- contrast groups: `9` 个，全部有 door vs suitcase 配对。

当前 V2 manifest 关键发现：

- `instruction_forward_vs_stop` / `instruction_left_vs_right` / `instruction_stop_vs_turn` 均未激活，train hit count 为 `0`。
- 原因：当前 `instruction_class()` 把 `go/navigate/move/approach to X` 都归成 `forward`，无法表达 door vs suitcase 目标对象差异。因此 V2 原 manifest 实际没有加强“同画面不同目标指令”的训练信号。
- 但从数据看，同 `contrast_group_id`、近 step 的 door vs suitcase counterexample 很充足：train 中 window=1 的 object counterexample candidate unique ids 为 `18475`，说明 V2.1 可以做 object-target-aware manifest，而不是继续沿用 forward/stop/left/right 规则。

当前 V2 weighted dataset 终端样本风险：

| priority_bucket | rows | terminal_progress | terminal_settle | terminal_or_settle |
|---|---:|---:|---:|---:|
| normal_cruise | 6326 | 1011 | 1178 | 1211 |
| low_vx_but_should_move | 5467 | 0 | 0 | 0 |
| stop_should_hold | 3775 | 3775 | 3775 | 3775 |
| wz_change | 1356 | 0 | 0 | 0 |
| first_step_commit | 1287 | 0 | 0 | 0 |
| residual_recovery | 635 | 0 | 0 | 0 |

`stop_should_hold` 在 V2 中权重为 `3.0`，weighted 后贡献 `11325` 行，且全部是 terminal/settle 样本。这与 near-target/static-image 分析结论一致：V2.1 不宜继续高权重放大 terminal saturated/stop-hold 样本。

### 建议的 V2.1 方向

建议 V2.1 不是盲目继续训练，而是先生成一个新 manifest / weighted dataset，核心变化：

1. 增加 object-target contrast bucket，例如 `instruction_target_door_vs_suitcase` 或 `target_object_contrast_near_step`，基于同 `contrast_group_id`、近 `trajectory_step_index` 且 `target_label` 不同的配对。
2. 对 object-target contrast 不要无差别给 3.0 高权重，因为候选几乎覆盖全 train pool；建议先用温和权重 `1.25-1.5`，并优先叠加到 non-terminal / decision-heavy 样本。
3. 降低或封顶 terminal stop-hold 权重：`stop_should_hold 3.0 -> 1.0/1.15`，避免 V2.1 继续把 terminal saturated frames 扩到 1.1 万行。
4. `normal_cruise 1.15 -> 1.0`，把容量让给 object contrast + non-terminal decision-heavy。
5. 保留有效的 decision/action buckets：`low_vx_but_should_move=2.0`、`wz_change=2.0`、`first_step_commit=1.5`、`residual_recovery=1.5`。
6. 训练 config 建议采用 no-prevprompt，即 `use_prev_action_as_condition=false`，anchor 仍保留 prev_action；这样与当前 strict-pass 推理配置一致。

### 必须确认的实验语义

执行 V2.1 manifest / weighted dataset / training 前需要用户确认：

- checkpoint 初始化：从 `plus100` parent artifact 初始化，还是从 `checkpoint-950` resume optimizer/trainer state。
- training prompt：是否正式用 no-prevprompt 训练 config，即 `use_prev_action_as_condition=false`。
- V2.1 manifest 权重：object contrast 权重、stop_should_hold 降权幅度、normal_cruise 是否降到 `1.0`。
- 输出命名：建议统一使用新 timestamp/suffix `targeted_v2_1_object_contrast_no_prevprompt_<TS>`，所有 output dir 新建且不覆盖。
- 短训点：是否仍采用 `+0.25 epoch` / `+0.5 epoch`，并只在趋势好时再跑 `+1.0 epoch`。

## 2026-04-26: 术语澄清记录

### plus100 parent artifact

`plus100 parent artifact` 指训练输出父目录 `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus100_20260426_143200_retry1`，不是某个 `checkpoint-*` 子目录。父目录包含最终 inference 需要的 `model.safetensors`、`robotics_velocity_head.bin`、`robotics_velocity_config.json`、tokenizer 文件等，因此 policy validation 默认应使用父目录作为 model artifact。

### checkpoint-950

`checkpoint-950` 是上述 plus100 训练 run 内部由 HuggingFace Trainer 按 `save_steps=50` 自动保存的 resume checkpoint，路径为 `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus100_20260426_143200_retry1/checkpoint-950`。它属于 2026-04-26 Targeted V2 plus100 短训过程，不是单独的新实验；之前笔记主要记录了 plus100 输出目录和 policy validation 结果，没有展开列出每个内部 checkpoint。

### manifest

本项目这里的 manifest 指 priority manifest JSONL，不是新采集数据。它按 train sample_id 记录 `priority_bucket`、`matched_buckets`、`sample_weight`、`priority_reason` 等，用于构造 weighted train jsonl；validation sample ids 会被排除，避免把 fixed-id validation 失败样本加入训练。

### 数据采集判断

当前 V2.1 建议先不采集新数据，而是先修正 manifest 逻辑：现有 train pool 已有充足 door-vs-suitcase object contrast，但旧 `instruction_class()` 把所有 `go/navigate/move/approach to X` 都归为 `forward`，没有利用目标对象差异。若 V2.1 后仍不足，再考虑定向采集更多目标/场景。

## 2026-04-26: 3090 定向采集计划草案

### 背景

用户希望如果 V2.1 后 `instruction_gain` 仍不稳，可以让 3090 先进行定向采集。目标不是无差别扩数据，而是补充“同一画面多个目标同时可见，不同 instruction 指向不同目标”的 object/instruction contrast 数据，并避免 terminal saturated frames 继续主导训练。

### 代码/配置 review 摘要

- 3090 wrapper：`scripts/collect_sim_dual_target_contrast_room_3090.sh`，可通过 `CONFIG_PATH` 指向不同采集 yaml。
- 3090 环境：`scripts/use_3090_env.sh`，默认 IsaacLab 路径 `/home/zxq/zxq/IsaacLab`，raw 输出可放 `/mnt/simdata/llada-vla-go2/collectors/data/sim_sessions`。
- 当前无代码改动即可跑的 config：`collectors/sim_go2/configs/collection_dual_target_contrast_main.yaml`，比 `room_local` 覆盖更多 `visible_early / visible_after_small_turn / search` 可见性桶。
- scene target slot 上限：`collectors/sim_go2/envs/scene_builder.py` 中 `TARGET_SLOT_NAMES=(target_0..target_3)`，最多 4 个目标。
- 当前 target spawn 默认是 `CuboidCfg`；真实 cone/cylinder/chair/table mesh 或 primitive 形状需要额外实现 shape spawn 支持。若不改代码，新增目标可以先用不同颜色/尺寸的 cuboid proxy 表示。

### 采集目标

- 第一优先级：object contrast，而非单目标导航。
- 每个 layout 中至少 2 个目标同时可见；理想是 3-4 个目标同时在视野中，instruction 指向其中一个。
- 对同一 `contrast_group_id`，保留同/近 `trajectory_step_index` 的不同 target instruction 配对，用于后续 manifest 的 object contrast bucket。
- 控制 terminal saturated：减少“最后整帧被目标色块填满”的样本权重；采集端尽量 stop 更早、目标不要过大，处理端后续对 terminal/static slice 降权。

### 3090 立即可跑 lane（无需新代码）

先跑当前 2-target main config，作为 object contrast + visibility variation 的补充 raw 数据：

- config: `collectors/sim_go2/configs/collection_dual_target_contrast_main.yaml`
- 推荐 batch: `240` episodes / seed，先 smoke `12` episodes。
- 建议 run name: `v2_1_collect_2target_main_seed3000_gpu0`
- 目的：补充 reveal/search 与非 local-only 的 door/suitcase contrast，不改变当前 V2.1 训练语义，后续是否纳入训练另行确认。

建议命令：

```bash
source scripts/use_3090_env.sh
SIM_RAW_ROOT_BASE=/mnt/simdata/llada-vla-go2/collectors/data/sim_sessions \
CONFIG_PATH=$REPO_ROOT/collectors/sim_go2/configs/collection_dual_target_contrast_main.yaml \
GPU_ID=0 SEED=3000 NUM_EPISODES=12 HEADLESS=true \
RUN_NAME=v2_1_collect_2target_main_smoke_seed3000_gpu0 \
bash scripts/collect_sim_dual_target_contrast_room_3090.sh
```

smoke 通过 `quality_gate_summary.json` 后再跑：

```bash
source scripts/use_3090_env.sh
SIM_RAW_ROOT_BASE=/mnt/simdata/llada-vla-go2/collectors/data/sim_sessions \
CONFIG_PATH=$REPO_ROOT/collectors/sim_go2/configs/collection_dual_target_contrast_main.yaml \
GPU_ID=0 SEED=3000 NUM_EPISODES=240 HEADLESS=true \
RUN_NAME=v2_1_collect_2target_main_seed3000_gpu0 \
bash scripts/collect_sim_dual_target_contrast_room_3090.sh
```

### 需要新 config 的 4-target lane

建议新增 `collection_multi_target_object_contrast_v2_1.yaml`，先不改 collector 代码，使用 4 个 cuboid proxy target：

- `door_target`: orange door，延续现有。
- `suitcase_target`: dark gray suitcase/box，延续现有。
- `chair_target`: blue chair proxy，较矮中等尺寸 cuboid。
- `table_target`: green table proxy，低矮扁平 cuboid。

如果要真实 cone/cylinder，需先改 `scene_builder.py` 让 target visual 根据 `shape` 选择 `ConeCfg`/`CylinderCfg` 等 primitive；这应作为单独代码任务并先 smoke。

4-target 采集配比建议：

- `visible_multi`: 50%，3-4 个目标同时可见。
- `partial_reveal`: 30%，目标初始部分可见或小转向后可见。
- `search_light`: 20%，目标不在初始中心，但不做长搜索，避免复杂任务污染。

推荐首轮规模：

- smoke: `12` episodes。
- pilot: `120` episodes。
- full: `360-480` episodes，按 2-3 个 seed 分批，每批 `120-160`。

### 暂不建议的采集

- 暂不混入复杂任务：绕障碍、多阶段任务、搜索式长程任务、动态物体任务。
- 暂不采集只有单目标可见的数据；它对 instruction/object contrast 贡献低。
- 暂不扩大 terminal hold/stop 数据；已有 V2 的 `stop_should_hold` 已过度放大。

### 后续使用规则

3090 新 raw 数据即使采完，也不默认进入 V2.1 训练。必须先做 raw quality gate、pack/audit、contrast group 检查、terminal/static slice 检查，并由用户确认 dataset 语义后，才可构造新 dataset 或训练。

## 2026-04-26: V2.1 object-contrast manifest dry-run 完成

### 背景

用户确认 V2.1 前置方案：以 `plus100 + no-prevprompt` 为默认候选，新增 object-target contrast，降低 terminal stop-hold 放大，不直接开训，先做 manifest/config patch 与 dry-run + schema/debug 检查。

### 代码 / 配置更新

新增 V2.1 manifest builder：

- 本地/远端：`scripts/build_targeted_v2_1_object_contrast_manifest.py`

新增 no-prevprompt 训练配置：

- 本地/远端：`configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_1.yaml`
- `velocity_control.use_prev_action_as_condition=false`
- `model.reset_action_modules_after_load=false`
- `model.model_name=llada-vla-go2-sim-today-main-turnpilot-anchor-residual-no-prevprompt-v2-1`

配置语义：训练 prompt 不包含 `<previous_action>`，但 anchor 仍由真实 `previous_action` 构造。`reset_action_modules_after_load=false` 是为了后续如从 plus100 parent artifact 初始化时不重置已训练 action modules。

### Dry-run 输入

使用 `plus100+B` fixed-id policy validation 作为 failure reference：

- metrics: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_no_prevprompt_fixed_ids_20260426_205133_retry1/metrics.json`
- predictions: `/root/autodl-tmp/llada-vla-go2/outputs/policy_validation_targeted_v2_plus100_no_prevprompt_fixed_ids_20260426_205133_retry1/predictions.jsonl`
- train pool: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_20260425/train.jsonl`

### Dry-run 迭代说明

第一次 dry-run 输出：

- `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215437`
- `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215437`

该版发现 `low_vx_but_should_move` 过度泛化：由于 V2.1 builder 最初把所有 future moves_forward 都标成 low-vx bucket，weighted 后 `low_vx_but_should_move` 贡献 `23390` 行，object contrast 被压制。该输出保留为失败 dry-run，不推荐用于训练。

修正：`low_vx_but_should_move` 仅在 future should move 且 previous action 接近低速时触发：`prev_abs_vx <= 0.10` 且 `prev_abs_wz <= 0.08`。修正后重新生成新 timestamp 输出。

### 推荐 dry-run 输出

推荐 V2.1 dry-run 输出目录：

- audit: `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev`
- weighted dataset: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev`

核心文件：

- manifest: `outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/targeted_v2_1_object_contrast_manifest_train.jsonl`
- manifest summary: `outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/targeted_v2_1_object_contrast_manifest_summary_train.json`
- validation report: `outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/targeted_v2_1_weighted_manifest_validation.json`
- fixed validation ids copy: `outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/validation_sample_ids.txt`

### Manifest / weighted dataset 结果

Manifest rows: `18846`，validation leak removed count: `0`。

Object contrast:

- bucket: `target_object_contrast_near_step`
- groups: `82`
- labeled samples: `13860`
- terminal skipped: `4986`
- pair events: `13860`
- same-step pair events: `376`
- include terminal: `false`
- step window: `1`

Manifest top bucket counts:

| bucket | rows |
|---|---:|
| target_object_contrast_near_step | 9252 |
| stop_should_hold | 3775 |
| first_step_commit | 2445 |
| wz_change | 1356 |
| normal_cruise | 1211 |
| residual_recovery | 807 |

Weight counts:

| weight | rows |
|---:|---:|
| 1.00 | 1211 |
| 1.15 | 3775 |
| 1.35 | 9252 |
| 1.50 | 3252 |
| 2.00 | 1356 |

Weighted dataset:

- train rows: `25651`
- source train rows: `18846`
- expansion ratio: `1.3611`
- unique weighted sample ids: `18846`
- copy histogram: one-copy `12041`, two-copy `6805`
- image folder symlink valid: points to `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_turnpilot_20260425`
- validation leak count after explicit debug: `0`
- validator pass: `true`, errors `0`

Weighted bucket rows:

| bucket | weighted rows |
|---|---:|
| target_object_contrast_near_step | 12482 |
| stop_should_hold | 4352 |
| first_step_commit | 3672 |
| wz_change | 2712 |
| residual_recovery | 1222 |
| normal_cruise | 1211 |

### 对比 V2 的改善

- V2 weighted train rows: `35129`；V2.1 dry-run train rows: `25651`，总体扩增从 `1.864x` 降到 `1.361x`。
- V2 `stop_should_hold` weighted rows: `11325`；V2.1 降到 `4352`，明显减少 terminal stop-hold 放大。
- V2.1 引入 `target_object_contrast_near_step`，且默认跳过 terminal progress / terminal settle 样本，避免把 near-target saturated frames 当成 object contrast 主样本。

### 当前推荐

推荐使用 `20260426_215525_lowprev` 这一版作为 V2.1 数据候选。下一步如开训，建议明确使用：

- config: `/root/autodl-tmp/llada-vla-go2/configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_1.yaml`
- train data: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/train.jsonl`
- eval data: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/val.jsonl`
- image folder: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/images`
- initialization: prefer plus100 parent artifact init, not `checkpoint-950` trainer resume, unless explicitly要恢复 optimizer/scheduler。

尚未启动训练。

## 2026-04-26: V2.1 training preflight、LoRA init 修复与短训启动

### 用户确认

用户确认可以进入 V2.1 下一步，并再次强调训练前必须 review 代码和配置、debug 关键模块输入输出是否符合逻辑。本节记录训练启动前检查和启动过程。

### Preflight / Debug 结果

已 review：

- `scripts/train_current_v1_sim_today_main_only_subset_anchor_residual_1gpu.sh`
- `LLaDA-V/train/robotics/train_velocity.py`
- `LLaDA-V/train/robotics/checkpointing.py`
- `LLaDA-V/train/robotics/data/dataset_vla.py`
- `LLaDA-V/train/robotics/data/prompt_builder.py`
- V2.1 config: `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_1.yaml`

关键 IO debug：

- config exists，`use_prev_action_as_condition=false`，`reset_action_modules_after_load=false`，`horizon_k=4`，`anchor_mode=repeat_prev`。
- V2.1 train rows `25651`，val rows `1985`。
- image folder symlink 有效，sample image exists。
- fixed validation ids `200`，train unique ids `18846`，train/fixed intersection `0`。
- prompt debug：sample prompt 不含 `<previous_action>`，`previous_action_text_len=0`，action token count `12`，包含 `<image>`。
- anchor debug：anchor 使用真实 previous action，repeat_prev 与 residual target 构造正常。
- plus100 init artifact 文件完整：`model.safetensors`、`robotics_velocity_head.bin`、`robotics_velocity_config.json` 均存在。

### 关键问题与修复

第一次启动脚本 `20260426_220004` 后，日志显示 `Loaded adaptation checkpoint` 发生在 `Applied LoRA` 之前。这意味着 `init_checkpoint_path` 加载 plus100 parent artifact 时，artifact 中的 LoRA 权重 key 可能因为 LoRA adapter 尚未注入而没有正确加载。已立即停止该训练链，停止时进度刚到 `0/201`，只产生了 partial output metadata：

- partial output: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus025_20260426_220004`
- 不推荐使用该 partial output。

修复：调整 `LLaDA-V/train/robotics/train_velocity.py` 的加载顺序：先 `_configure_trainable_params()` 和 `_maybe_apply_lora()`，再 `load_full_checkpoint_for_adaptation()`。这样 plus100 parent artifact 中的 LoRA keys 已有对应 module，可由 `model.safetensors` 加载。修复已本地/远端 `py_compile` 通过。

### 正式短训链启动

修复后重新生成新 timestamp 训练链：

- script: `/root/autodl-tmp/llada-vla-go2/scripts/run_targeted_v2_1_no_prevprompt_training_chain_20260426_220225.sh`
- master log: `/root/autodl-tmp/llada-vla-go2/outputs/run_targeted_v2_1_no_prevprompt_training_chain_20260426_220225.master.log`

训练语义：

- init checkpoint/artifact: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus100_20260426_143200_retry1`
- 不 resume optimizer/scheduler；使用 parent artifact 初始化。
- train config: `/root/autodl-tmp/llada-vla-go2/configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_1.yaml`
- train data: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/train.jsonl`
- eval data: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/val.jsonl`
- image folder: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_1_object_contrast_no_prevprompt_20260426_215525_lowprev/images`
- GPU: 5090 GPU0

计划输出：

- plus025: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus025_20260426_220225`
- plus050: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus050_20260426_220225`

启动后日志确认修复生效：

- `Applied LoRA ...` 出现在 `Loaded adaptation checkpoint type=full ...` 之前。
- train clip stats: `wz target_clip_rate=0.009498`，无 clip warning。
- eval clip stats: `wz target_clip_rate=0.007545`，无 clip warning。
- `plus025` 计划 steps `201`；step 5 日志：`loss=0.7331`, `grad_norm=6.54297`, `learning_rate=7.142857e-05`。
- GPU memory about `30105 / 32607 MiB`，utilization about `93%`。

### 注意事项

- 远端无 `tmux`，因此本次使用 `setsid/nohup` 后台运行并写 master log。
- 期间出现一次 SSH DNS 临时失败 `Temporary failure in name resolution`，重试后正常；训练进程未受影响。

## 2026-04-26: TensorBoard output 可见性纠正

### 用户纠正

用户反馈当前训练在 TensorBoard 上看不到，并要求后续每次训练都必须把 output/logging 导向用户能看到 TensorBoard 的位置。

### 新执行规则

- 后续训练前必须确认用户当前 TensorBoard `--logdir` 或可见输出目录。
- 训练 `output_dir` / `logging_dir` 不应只写到远端临时路径；必须写到或同步到用户能在 TensorBoard 看到的位置。
- 若训练已经启动且不适合中断，应至少把当前 `runs/events...` 同步或软链到用户 TensorBoard logdir，并记录同步方式。

## 2026-04-26: TensorBoard symlink 修正

### 用户说明

用户指出之前 Targeted V2 的 TensorBoard runs 放在 `/root/tf-logs/llada-vla-go2-targeted-v2/{plus025,plus050,plus100}`，后续训练也应放到同一 TensorBoard 可见位置。

### 修正动作

已为当前 V2.1 训练新增 TensorBoard 可见 symlink：

- `/root/tf-logs/llada-vla-go2-targeted-v2/v2_1_plus025` -> `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus025_20260426_220225/runs`
- `/root/tf-logs/llada-vla-go2-targeted-v2/v2_1_plus050` -> `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus050_20260426_220225/runs`

已确认 `v2_1_plus025` 下可以看到当前 event 文件。`v2_1_plus050` 会在 plus050 开始并创建 runs 后生效。

### 后续规则

远端 5090 训练的 TensorBoard 可见目录固定使用 `/root/tf-logs/llada-vla-go2-targeted-v2`。每次训练启动后必须创建并验证对应 symlink，不再只依赖 output_dir 内部 `runs/`。

## 2026-04-26: plus100 本地 3090 turnpilot 闭环根因排查与修复

### 发现的链路问题

1. `deploy/go2/controller_loop.py` 在在线闭环时没有把 `previous_action` 写入 prompt 文本，导致部署时的 prompt 条件与训练/validation 不一致。
2. `collectors/sim_go2/envs/task_generator.py` 在 `collection_dual_target_contrast_room_local_turn_pilot.yaml` 下对 contrast pair 施加了“所有 target 同时满足 visibility bucket”的过严约束，结果 `shared layout` 采样 100% 失败并静默退化到 `fallback` task。
3. `fallback` task 本身还存在几何不一致：`robot_yaw=0` 时目标被放在 `(0, 2)` 方向，但 root-kinematic / visibility 约定里 `yaw=0` 朝 `+x`，因此 fallback 元数据宣称 `visible_first_frame=true`，实际 sim 中目标并不在正前方。

### 已做修复

- `deploy/go2/controller_loop.py`
  - 在线 prompt 现在显式注入 `previous_action`。
  - 闭环 tokenization 改为复用 `tokenize_prompt_segments(...)`，与训练/validation 保持一致。
- `collectors/sim_go2/envs/task_generator.py`
  - contrast 共享采样失败时，不再直接退化到硬编码 fallback；改为按 active target 分别做有效 layout 采样，避免闭环评估落到无效 fallback 任务。
  - fallback task 改为几何自洽的直行目标，并由 `rollout_visibility(...)` 真实回填 visibility 元数据，而不是硬编码 `visible_first_frame=true`。
- 新增/通过测试：
  - `tests/integration/test_mock_deploy.py`
  - `tests/unit/test_safety_filter.py`
  - `tests/unit/test_dual_target_contrast_task_generator.py`

### 关键验证

- 纯 Python 复查 `GoalNavigationTaskGenerator`：修复前该 turnpilot config 连续采样只会产出 `turn_bucket=fallback`；修复后可稳定采出真实 `visible_after_small_turn` 任务，例如：
  - `door_target / right_large / visible_after_small_turn / target_visible_first_frame=false / target_visible_within_10f=true`
  - `suitcase_target / left_large / visible_after_small_turn / target_visible_first_frame=false / target_visible_within_10f=true`
- 本地 3090 GPU 预检：`torch.cuda.is_available() == True`，IsaacSim Vulkan/GPU foundation 正常启动。

### 修复后真实闭环结果

#### A. `execute_steps=1`

输出目录：`outputs/closed_loop_plus100_local_turnpilot_postfix_20260426_234054`

- episode 0: `approach the door`
  - `timeout`, `goal_distance=2.9031m`, `goal_clearance=2.5631m`
  - mean `|vx| ~= 0.0311`, mean `|wz| ~= 0.3368`, displacement `~= 0.1849m`
- episode 1: `go to the dark gray suitcase`
  - `timeout`, `goal_distance=2.5718m`, `goal_clearance=2.1818m`
  - mean `|vx| ~= 0.0311`, mean `|wz| ~= 0.3016`, displacement `~= 0.2065m`

结论：修复后闭环确实进入了有效 turnpilot 任务分布，不再是 fallback；但 plus100 仍然没有成功完成任务，且 180 steps 内目标始终未进入 `target_visible=true`。

#### B. `execute_steps=4`

输出目录：`outputs/closed_loop_plus100_local_turnpilot_exec4_20260426_235649`

- episode 0: `approach the door`
  - `timeout`, `goal_distance=2.9240m`, `goal_clearance=2.5840m`
  - mean `|vx| ~= 0.0311`, mean `|wz| ~= 0.3645`, displacement `~= 0.1663m`

结论：把 `execute_steps` 从 `1` 提到 `4` 没有带来明显改善；剩余问题不像是 receding-horizon 执行步数配置，而更像是模型在真实闭环 turnpilot 上本身会持续把目标转丢。

### 当前判断

- 前两个“代码/评估链路”问题已经修复并验证。
- 修复后结果依然 `timeout`，因此当前主要剩余问题更偏向模型策略本身，而不是 prompt/sampler/fallback 这类基础设施 bug。
- 下一步更值得做的是：
  1. 记录在线 rollout 中每步 `target_visible / goal_heading / predicted chunk / chosen execute action`，确认模型是否系统性朝错误方向积分；
  2. 对比当前 plus100 与 `no-prevprompt B` 候选在相同本地 turnpilot 闭环上的真实表现；
  3. 做 near-target / static-image / target-not-visible slice，确认是否是 instruction-conditioned search/turn 行为仍未学到。

## 2026-04-27: plus100 在线闭环逐步 trace 诊断

### 诊断动作

- 为 `deploy/go2/controller_loop.py` 增加在线 `model_debug` 记录：
  - `previous_action`
  - `target_actions`
  - `anchor_actions`
  - `residual_actions`
  - `continuous_actions`
  - `execute_actions`
  - `selected_execute_action`
  - `safe_command`
  - `step_confidence`
  - `safety_mode` / `safety_reason`
- `collectors/sim_go2/scripts/eval_closed_loop_policy.py` 的 episode trace 现在会把上述 `model_debug` 写进每个 step。
- 新增集成测试覆盖 debug payload 注入，当前相关测试通过：
  - `tests/integration/test_mock_deploy.py`
  - `tests/unit/test_safety_filter.py`
  - `tests/unit/test_dual_target_contrast_task_generator.py`

### trace-rich 本地闭环结果

输出目录：`outputs/closed_loop_plus100_local_turnpilot_trace_20260427_001301`

- episode 0: `approach the door`
  - `timeout`
  - `goal_distance=2.9524m`
  - `goal_clearance=2.6124m`
  - `turn_bucket=right_large`
  - `visibility_bucket=visible_after_small_turn`

### 关键逐步观察

- step 1:
  - `goal_heading=-0.4558`
  - `selected_execute_action=[0.00137, 0.00078, -0.02549]`
  - `safe_command=[0.00034, 0.00020, -0.00637]`
  - `anchor_actions[0]=[0.0, 0.0, 0.0]`
- step 30:
  - `goal_heading=-0.3498`
  - `target_visible=true`
  - `safe_command=[0.01029, 0.00588, -0.13235]`
- step 60:
  - `goal_heading=-0.0712`
  - `target_visible=true`
  - `safe_command=[0.02059, 0.01176, -0.23039]`
- step 120:
  - `goal_heading=1.1460`
  - `target_visible=false`
  - `safe_command=[0.04118, 0.02353, -0.54020]`
- step 180:
  - `goal_heading=3.1015`
  - `target_visible=false`
  - `safe_command=[0.06177, 0.03529, -0.72745]`

### 结论

- `safety_filter` 不是当前主因：`safe_command` 与 `selected_execute_action` 方向一致，只做了符合配置的 rate-limit / EMA 平滑，并没有把合理前进动作压成零。
- 模型在整个 rollout 中基本持续输出“微小正 `vx` + 持续同向 `wz`”的动作；随着 anchor 积分，`|wz|` 单调放大，最终把目标重新转丢。
- `residual_actions` 很小且高度稳定，例如 step 30 / 60 / 120 的第一步 residual 几乎固定在：
  - `vx ~= +0.00137`
  - `vy ~= +0.00078`
  - `wz ~= -0.01373`
- 这说明当前 plus100 在线闭环更像是在“重复并轻微修正上一步动作”，而不是在目标短暂进入视野后做有效 re-plan。
- 将 `execute_steps` 从 `1` 提到 `4` 的单轮检查也没有明显改善，因此剩余问题目前更像是模型策略本身（尤其是 turn/search/reacquire 机制不足），不是 prompt/sampler/safety filter 或 receding-horizon 步数配置 bug。

## 2026-04-27: V2.2 三路实验改向、preflight review 与顺序排队

### 决策

基于 turnpilot 闭环 trace，当前问题定性为“模型没有学到 turn/search/reacquire”，而不是基础设施链路问题。因此停止正在跑的 V2.1，对实验方向改为三条 V2.2 对照：

- `residual + reacquire proxy data`
- `residual + weak anchor`
- `absolute + no-prevprompt`

### 停止的旧实验

已停止：

- `run_targeted_v2_1_no_prevprompt_training_chain_20260426_220225.sh`
- `run_targeted_v2_1_no_prevprompt_fixed_id_policy_validation_20260426_220225.sh`

GPU 已确认释放后才开始新实验。

### V2.2 preflight review / check

已 review 并修正：

- `scripts/run_targeted_v2_2_experiment_chain.sh`
  - 修复 config 路径 preflight 检查逻辑。
  - 删除提前创建 output dir 的行为，避免与 lane guard 冲突。
  - 新增 `PRECHECK_ONLY=1`，支持只检查不启动训练。
- `scripts/run_targeted_v2_2_queue_20260427_005148.sh`
  - 顺序等待单卡空闲，自动接 `weak_anchor` 和 `absolute`。
- `scripts/build_targeted_v2_2_reacquire_proxy_manifest.py`
  - 语法与 dry-run 已通过。
  - 由于当前 packed train jsonl 不再保留 `visibility_bucket / turn_bucket / target_visible_within_10f`，V2.2 的 “reacquire” 只能先做 proxy 版，而不是精确回填版。

配置 review 通过：

- `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_2_reacquire.yaml`
- `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_2_weak_anchor.yaml`
- `configs/llada_vla_go2_sim_today_main_turnpilot_absolute_no_prevprompt_train_v2_2.yaml`

远端三条 wrapper 均已执行 `PRECHECK_ONLY=1`，确认：

- checkpoint 路径存在
- train/val/image 路径存在
- `no-prevprompt` 语义正确
- `weak_anchor` 使用 `decay_prev + anchor_use_dropped_prev_action=true`
- `absolute` 使用 `action_target_mode=absolute` 且 `reset_action_modules_after_load=true`

### V2.2 reacquire proxy 数据

新增 manifest builder：

- `scripts/build_targeted_v2_2_reacquire_proxy_manifest.py`

dry-run 产物：

- audit: `outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_2_reacquire_proxy_20260427_005148`
- dataset: `datasets/current_v2_sim_today_main_only_subset_targeted_v2_2_reacquire_proxy_20260427_005148`

结果：

- manifest rows: `18846`
- weighted train rows: `27489`
- validator pass: `true`
- validation leak: `0`

bucket counts:

- `target_object_contrast_near_step`: `8373`
- `reacquire_turn_proxy`: `3823`
- `stop_should_hold`: `3775`
- `normal_cruise`: `1211`
- `first_step_commit`: `857`
- `residual_recovery`: `807`

这版 proxy 规则只依赖 packed train jsonl 里仍然保留的字段：`action_chunk / previous_action / contrast_group_id / target_label / trajectory_step_index / trajectory_length`，不再依赖已经丢失的 privileged visibility 标签。

### 已启动 / 已排队

当前已启动第一条：

- script: `scripts/run_targeted_v2_2_reacquire_20260427_005148.sh`
- log: `outputs/run_targeted_v2_2_reacquire_20260427_005148.master.log`

已排队等待：

- `scripts/run_targeted_v2_2_weak_anchor_20260427_005148.sh`
- `scripts/run_targeted_v2_2_absolute_20260427_005148.sh`
- queue watcher: `scripts/run_targeted_v2_2_queue_20260427_005148.sh`
- queue log: `outputs/run_targeted_v2_2_queue_20260427_005148.master.log`

## 2026-04-27: plus100 vs plus100+B 本地真实 turnpilot 闭环对比

### 对比设置

- 平台：本地 3090 + IsaacSim Vulkan/camera 正常环境
- 配置：`collectors/sim_go2/configs/collection_dual_target_contrast_room_local_turn_pilot.yaml`
- 闭环入口：`collectors/sim_go2/scripts/eval_closed_loop_policy.py`
- 共同条件：`execute_steps=1`，`seed=0`，`headless=true`
- A 模型：`outputs/closed_loop_plus100_local_turnpilot_trace_20260427_001301`
- B 模型：`outputs/plus100_b_local_model_root`
  - 语义：沿用 plus100 原权重，但把 `robotics_velocity_config.json` 里的 `use_prev_action_as_condition` 改为 `false`
  - 本地比对输出：`outputs/closed_loop_plus100B_local_turnpilot_trace_20260427_003254`

### 结果（同一 door / right_large / visible_after_small_turn 任务）

| model | success | goal_distance | goal_clearance | mean |vx| | mean |wz| | displacement | first_visible_step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plus100 A | timeout | 2.9524 | 2.6124 | 0.03105 | 0.38885 | 0.15250 | 26 |
| plus100 B | timeout | 2.9207 | 2.5807 | 0.02362 | 0.33029 | 0.14013 | 28 |

### 逐步行为差异

- `plus100+A`：
  - step 1 `selected_execute_action=[0.00137, 0.00078, -0.02549]`
  - step 30 visible 后仍持续增加同向转动，step 180 `safe_command=[0.06177, 0.03529, -0.72745]`
- `plus100+B`：
  - step 1 `selected_execute_action=[0.00137, 0.00078, -0.01373]`
  - step 30 visible 后也继续积累同向转动，但整体 `|wz|` 比 A 更小；step 180 `safe_command=[0.04941, 0.03529, -0.63627]`

### 结论

- `plus100+B` 没有把本地真实 turnpilot 闭环从 timeout 拉成 success。
- 相比 `plus100+A`，`plus100+B` 在同一任务上略好：
  - 最终 `goal_distance` 更小（`2.9207 < 2.9524`）
  - 最终 `goal_clearance` 更小（`2.5807 < 2.6124`）
  - 整体转动更保守（`mean |wz|` 更低）
- 但改进幅度很小，仍然没有解决“目标短暂进入视野后无法稳定 reacquire / 收敛”的核心闭环失败模式。
- 因此当前判断：`plus100+B` 在 offline gate 上优于 `plus100+A`，在真实闭环上也可能略优，但两者都还不足以完成该 turnpilot 任务。

## 2026-04-27: 5090 远端数据盘保守清理

### 已删除目录（释放旧训练产物空间）

- `outputs/train_current_v1_sim_office_clean_absolute_1gpu_sdpa_20260425_1035`
- `outputs/train_current_v1_sim_office_clean_anchor_residual_1gpu_sdpa_20260425_1035`
- `outputs/train_current_v1_sim_today_main_only_subset_absolute_1gpu_20260425_131149`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_1gpu_20260425_131149`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_1gpu_k4`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_1gpu_k6`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_2gpu`

### 清理效果

- `/root/autodl-tmp` 使用率：`451G / 500G (91%) -> 324G / 500G (65%)`
- 可用空间：`50G -> 177G`

### 当前仍保留的大目录

- `48G` `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_2gpu_priority_weighted`
- `48G` `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus025_20260426_143200_retry1`
- `48G` `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus050_20260426_143200_retry1`
- `48G` `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus100_20260426_143200_retry1`
- `48G` `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus025_20260426_220225`
- `32G` `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus050_20260426_220225`

这些目录对应当前 weighted / targeted_v2 / no-prevprompt 主线 checkpoint，本轮先未删除。

## 2026-04-27: 5090 远端数据盘第二层保守清理

### 已删除目录

- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus025_20260426_143200_retry1`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus050_20260426_143200_retry1`

### 清理后预期语义

- 保留当前原始 `plus100` 训练大权重，便于继续做 A / B 真实闭环对照。
- 保留 no-prevprompt 主线大权重（V2.1 object contrast），便于后续继续对比 `plus100+B` / `plus050+B` 方向。
- 保留所有 targeted_v2 fixed-id policy validation、no-prevprompt policy validation、eval artifact、proxy rollout 等小结果目录。

## 2026-05-04: V2.2 reacquire 已恢复结果，weak-anchor / absolute 重新启动

### 背景

- 之前主笔记只记录到 `V2.2` 三路实验“已启动 / 已排队”，但没有把 `reacquire` 这一路后续真实产物和当前 `weak_anchor` / `absolute` 的重启状态写回文档。
- 本次在远端 `5090` 恢复可连后，对 `outputs/`、master log、checkpoint 内 `trainer_state.json` 做了补查。

### Reacquire 路线恢复结果

- `reacquire` 不是只跑了半截；实际保留产物包括：
  - `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus025_20260427_011001`
  - `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus050_20260427_011001`
- 更早还有一版残留：
  - `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus025_20260427_005148`
  - 推断：这是第一次启动后的早期产物，后续主线结果以 `20260427_011001` 为准。
- 对应 master log：
  - `outputs/run_targeted_v2_2_reacquire_20260427_011001.master.log`

### 核心训练结果

- `plus025`:
  - `global_step=100`
  - `best_metric / eval_loss = 0.5234111547470093`
  - `best_model_checkpoint = .../plus025_20260427_011001/checkpoint-100`
- `plus050`:
  - `global_step=200`
  - `best_metric / eval_loss = 0.44823047518730164`
  - `best_model_checkpoint = .../plus050_20260427_011001/checkpoint-200`
- 就 trainer 内置 `eval_loss` 而言，`plus050` 明显优于 `plus025`。

### 当前能确认的限制

- 本次补查没有发现单独的 `reacquire` fixed-id policy validation / normalized metrics 输出目录。
- 也没有发现命名上明确对应 `reacquire` 的真实闭环评估结果目录。
- 因此当前只能确认：`reacquire` 训练本身完成到了 `plus050`，且 trainer 内置 `eval_loss` 有改善；但还不能仅凭这点判断它是否优于主线候选，或是否真正改善了闭环 `reacquire` 行为。

### 2026-05-04 当前执行状态

- 远端恢复后，已重新做 `weak_anchor` / `absolute` 的 preflight：
  - 两张 `5090` 空闲
  - 无残留 `train_velocity.py`
  - 目标输出目录不存在，因此不会覆盖旧结果
  - config / dataset / init checkpoint 均存在
- 已重新启动：
  - `weak_anchor` 当前正在运行：
    - script: `scripts/run_targeted_v2_2_weak_anchor_20260427_005148.sh`
    - log: `outputs/run_targeted_v2_2_weak_anchor_20260427_005148.master.log`
  - `absolute` 已挂等待队列，待 `weak_anchor` 结束后自动启动：
    - queued log: `outputs/run_targeted_v2_2_absolute_queue_20260427_005148.master.log`
    - run log: `outputs/run_targeted_v2_2_absolute_20260427_005148.master.log`

### 结论

- `V2.2` 第一条 `reacquire` 之前已经真实跑完到 `plus050`，不是只有“已启动”。
- 但它缺少后续统一 policy validation / 闭环结果沉淀，所以目前只能算“训练结果已知、最终策略结论未闭环”。
- 当前优先级应保持为：
  1. 让 `weak_anchor` 和 `absolute` 正常跑完。
  2. 之后统一补这三条 V2.2 的 fixed-id policy validation 与闭环对照，再决定哪条真正值得继续。

## 2026-05-04: V2.2 weak-anchor / absolute 训练完成与阶段性判断

### 背景

- 在恢复 `reacquire` 结果后，继续完成剩余两条 `V2.2` 对照：
  - `weak_anchor`
  - `absolute`
- 本次仅确认 trainer 内置 `eval_loss` 与训练产物，不代表已完成 fixed-id policy validation 或真实闭环优劣结论。

### Weak-anchor 结果

- 训练产物：
  - `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_weak_anchor_no_prevprompt_plus025_20260427_005148`
  - `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_weak_anchor_no_prevprompt_plus050_20260427_005148`
- 从 master log 可确认：
  - `plus025` 最佳 `eval_loss` 约 `2.2994`
  - `plus050` 最佳 `eval_loss` 约 `2.1766`
- 这一路明显劣于 `reacquire`，也明显差于当前主线历史候选；单看 trainer 内置评估，基本可以判定该方向不值得优先继续。

### Absolute 结果

- 训练产物：
  - `outputs/train_current_v1_sim_today_main_turnpilot_absolute_targeted_v2_2_no_prevprompt_plus025_20260427_005148`
  - `outputs/train_current_v1_sim_today_main_turnpilot_absolute_targeted_v2_2_no_prevprompt_plus050_20260427_005148`
- checkpoint 内 `trainer_state.json` 显示：
  - `plus025`:
    - `global_step=100`
    - `best_metric / eval_loss = 0.8210591673851013`
    - `best_model_checkpoint = .../plus025_20260427_005148/checkpoint-100`
  - `plus050`:
    - `global_step=200`
    - `best_metric / eval_loss = 0.6520998477935791`
    - `best_model_checkpoint = .../plus050_20260427_005148/checkpoint-200`
- 与 `reacquire` 比较：
  - `absolute` 明显好于 `weak_anchor`
  - 但仍落后于 `reacquire plus050 = 0.44823047518730164`

### 运行链路补充

- `absolute` 第一次没有自动接上，不是 GPU 或路径问题，而是临时队列命令把自身也匹配进等待条件，导致永远不放行。
- 已通过手动重新 preflight 并直接启动 `scripts/run_targeted_v2_2_absolute_20260427_005148.sh` 修正；最终训练正常完成。

### 当前阶段性结论

- 按 trainer 内置 `eval_loss` 排序：
  1. `reacquire plus050 = 0.4482`
  2. `reacquire plus025 = 0.5234`
  3. `absolute plus050 = 0.6521`
  4. `absolute plus025 = 0.8211`
  5. `weak_anchor plus050 = 2.1766`
  6. `weak_anchor plus025 = 2.2994`
- 更准确地说，当前最值得继续验证的是：
  - `reacquire plus050`
  - `absolute plus050`
- `weak_anchor` 方向当前可视为低优先级或直接淘汰候选，除非后续有非常强的新证据。

### 下一步建议

- 不建议继续盲目加训。
- 下一步应做统一验证闭环：
  1. 对 `reacquire plus050` 与 `absolute plus050` 跑同一组 fixed-id policy validation。
  2. 再把这两个候选放到同一组本地/远端真实 `turnpilot` 闭环任务上对照。
  3. 用“offline fixed-id + 真实闭环”双重结果决定是否保留 `absolute` 作为强对照，还是回到 `reacquire` 主线继续做更细的数据策略。

## 2026-05-05: V2.2 fixed-id policy validation 完成结果（reacquire plus050 vs absolute plus050）

### 背景

- 按上一阶段计划，对 `reacquire plus050` 与 `absolute plus050` 跑同一组 fixed-id policy validation。
- 目标不是继续看 trainer 内置 `eval_loss`，而是直接判断：
  1. 谁在当前 strict gate 下真正通过；
  2. 谁相对当前默认候选 `plus100+B`（`plus100 no-prevprompt`）更强；
  3. `absolute` 是否值得保留为后续真实闭环强对照。

### 计划/动作

- 远端脚本：`scripts/run_targeted_v2_2_fixed_id_policy_validation_20260504_235000.sh`
- 远端输出目录：
  - `outputs/policy_validation_targeted_v2_2_reacquire_plus050_fixed_ids_20260504_235000`
  - `outputs/policy_validation_targeted_v2_2_absolute_plus050_fixed_ids_20260504_235000`
- 对照基线：
  - baseline：`outputs/policy_validation_baseline_ckpt450_parent_artifact_main_only_subset_targeted_v2_20260426_143200/baseline_checkpoint_450.metrics.normalized.json`
  - 当前默认候选：`outputs/policy_validation_targeted_v2_plus100_no_prevprompt_fixed_ids_20260426_205133_retry1/v2_checkpoint_plus100_no_prevprompt.metrics.normalized.json`

### 关键路径

- `reacquire plus050` 远端已正常产出 normalized / diff：
  - `v2_2_checkpoint_reacquire_plus050.metrics.normalized.json`
  - `diff_vs_baseline_checkpoint_450.normalized.json`
  - `diff_vs_plus100_no_prevprompt.normalized.json`
- `absolute plus050` 远端 `metrics.json` 已完成，但目录内未看到预期的 normalized / diff；其 `.post.log` 为空，主 `.log` 末尾停在 raw metrics JSON。当前只能确认原始离线验证本体完成，后处理产物未完整落盘。
- 本轮分析对 `absolute plus050` 采用远端 `metrics.json` 按同一口径人工读取 summary/subset 指标；因此下面涉及 `absolute` 的结论是基于 raw metrics 复原，不是直接引用远端 normalized 文件。

### 核心指标/结果

| checkpoint | model_mae lower | improvement_vs_prev_hold higher | relative_improvement | image_gain | instruction_gain | decision_heavy | low_vx_or_stop | turning | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline ckpt450 | 0.015920 | 0.006388 | 28.64% | 0.008842 | -0.000448 | 0.013597 | -0.001614 | 0.015128 | fail |
| plus100+B | 0.012755 | 0.009553 | 42.82% | 0.017990 | 0.000816 | 0.019217 | 0.000001 | 0.021818 | pass |
| reacquire plus050 | 0.009156 | 0.013152 | 58.96% | 0.018834 | 0.000475 | 0.026112 | 0.002308 | 0.028972 | pass |
| absolute plus050 | 0.018735 | 0.003573 | 16.02% | 0.025173 | -0.000036 | 0.009318 | -0.004684 | 0.008861 | fail |

相对 `plus100+B` 的关键变化：

- `reacquire plus050`
  - `model_mae`: `-0.003599`
  - `improvement_vs_previous_action_hold`: `+0.003599`
  - `image_conditioning_gain`: `+0.000844`
  - `instruction_conditioning_gain`: `-0.000342`
  - subset improvement:
    - `decision_heavy`: `+0.006894`
    - `low_vx_or_stop`: `+0.002307`
    - `turning`: `+0.007154`
    - `wz_change`: `+0.014942`
    - `first_step_change`: `+0.009495`
    - `vx_change`: `+0.011264`
    - `large_delta_wz`: `+0.016233`
    - `large_residual_correction`: `+0.009772`

- `absolute plus050`
  - `model_mae`: `+0.005979`
  - `improvement_vs_previous_action_hold`: `-0.005979`
  - `image_conditioning_gain`: `+0.007183`
  - `instruction_conditioning_gain`: `-0.000852`
  - subset improvement:
    - `decision_heavy`: `-0.009899`
    - `low_vx_or_stop`: `-0.004685`
    - `turning`: `-0.012957`
    - `wz_change`: `-0.027043`
    - `first_step_change`: `-0.021774`
    - `vx_change`: `-0.015506`
    - `large_delta_wz`: `-0.020270`
    - `large_residual_correction`: `-0.013696`

strict gate 判定：

- `reacquire plus050`
  - `improvement_vs_previous_action_hold > 0`: pass
  - `instruction_conditioning_gain > 0`: pass
  - `image_conditioning_gain >= 0.005`: pass
  - `decision_heavy_improvement_vs_previous_action_hold > 0`: pass
  - `turning_improvement_vs_previous_action_hold > 0`: pass
  - `low_vx_or_stop >= -0.002`: pass

- `absolute plus050`
  - `improvement_vs_previous_action_hold > 0`: pass
  - `instruction_conditioning_gain > 0`: fail
  - `image_conditioning_gain >= 0.005`: pass
  - `decision_heavy_improvement_vs_previous_action_hold > 0`: pass
  - `turning_improvement_vs_previous_action_hold > 0`: pass
  - `low_vx_or_stop >= -0.002`: fail

### 结论

- `reacquire plus050` 已经不是“仅 trainer eval_loss 更好”，而是在 fixed-id offline 上也全面超过当前默认候选 `plus100+B`，成为新的主线首选候选。
- `reacquire plus050` 的唯一相对 `plus100+B` 退步是 `instruction_conditioning_gain` 从 `0.000816` 降到 `0.000475`，但它仍保持为正，且 action / turning / stop-related / first-step / residual correction 各关键 slice 全部明显更强。
- `absolute plus050` 虽然 `image_conditioning_gain` 最高，但主 policy/action 指标普遍退化，且 `instruction_conditioning_gain` 再次为负、`low_vx_or_stop` 跌到 `-0.004684`，明显不适合作为当前主线。
- 因此本轮结果把“是否保留 `absolute` 作为强对照”的优先级明显下调：它最多只保留为后续真实闭环的次要对照，不应再与 `reacquire` 争夺主线候选位置。

### 下一步

- 默认候选从 `plus100+B` 切换为 `reacquire plus050`。
- 优先恢复并完成 `reacquire plus050` 的本地/远端真实 `turnpilot` 在线闭环。
- `absolute plus050` 只在资源允许时保留一次同条件真实闭环对照，用于判断是否存在明显的 offline-online gap；若闭环也不占优，可进一步下调其优先级。
- 需要补查 `absolute` 这轮为什么只落了 raw `metrics.json` 而没有落 normalized / diff，避免后续自动化把“原始验证完成”和“后处理完成”混为一谈。

## 2026-05-05: 本地 3090 在线闭环阻塞点定位

### 背景

- 在远端两条 `V2.2` fixed-id policy validation 并行运行的同时，尝试把：
  - `reacquire plus050`
  - `absolute plus050`
  拉到本地 3090，尽快启动同条件 `turnpilot` 在线闭环。

### 预检与修复

- 本地 `3090` GPU 访问、Isaac Python 路径、闭环入口脚本都可用。
- 另发现一个本地导入链问题：`deploy/go2/adapters/__init__.py` 硬编码导入了仓库内不存在的 `bridge_adapter.py`。
- 已改为 optional import，使 `collectors/sim_go2/scripts/eval_closed_loop_policy.py` 可以正常过导入阶段。

### 当前阻塞

- 本地前台重跑 `reacquire plus050` 在线闭环后，真实失败点已定位为：
  - `safetensors_rust.SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered`
- 触发位置：
  - `LLaDA-V/train/robotics/inference/predict.py`
  - `LLaDA-V/train/robotics/checkpointing.py`
  - `load_full_or_robotics_checkpoint(...)` 读取本地复制后的 `model.safetensors`

### 结论

- 当前本地在线闭环没有真正开始 rollout。
- 根因不是 Isaac/3090/闭环逻辑本身，而是本地复制下来的 checkpoint 根目录 `model.safetensors` 不完整或损坏，导致模型加载阶段直接失败。
- 在重新同步并校验本地 checkpoint 完整性之前，`reacquire` / `absolute` 的本地在线闭环都不能继续。

### 下一步

- 重新从远端同步两个候选的 `model.safetensors`，并在本地做文件完整性校验后，再恢复：
  1. `reacquire plus050` 本地在线闭环
  2. `absolute plus050` 本地在线闭环

## 2026-05-05: 本地 checkpoint root 重新同步并校验完成

### 背景

- 用户要求复查并重新拉取本地 `3090` 上要用的两个 checkpoint root：
  - `outputs/reacquire_plus050_local_model_root`
  - `outputs/absolute_plus050_local_model_root`
- 上一节已经确认本地闭环阻塞点来自 `model.safetensors` 不完整。

### 复查结果

- 远端源目录根 artifact 完整：
  - `reacquire plus050`: `model.safetensors = 16939227640` bytes
  - `absolute plus050`: `model.safetensors = 16939227640` bytes
- 本地旧副本确认是半截文件：
  - `outputs/reacquire_plus050_local_model_root/model.safetensors = 9534554112` bytes
  - `outputs/absolute_plus050_local_model_root/model.safetensors = 10068033536` bytes
- 两边 `robotics_velocity_head.bin` 均为 `72378417` bytes；`robotics_velocity_config.json` 分别为 `1872` / `1873` bytes。

### 处理

- 采用 `rsync --append-verify` 直接在现有本地 root 上断点续传，只补齐根 artifact：
  - `model.safetensors`
  - `robotics_velocity_head.bin`
  - `robotics_velocity_config.json`
- 未继续拉完整 `48G` 训练目录；本轮只修复本地在线闭环实际依赖的 checkpoint root。

### 校验

- 续传后两个本地 `model.safetensors` 都已变为 `16939227640` bytes。
- 使用本地 `llada` 环境直接 `safe_open(..., framework='pt', device='cpu')` 校验：
  - `outputs/reacquire_plus050_local_model_root/model.safetensors`: `OK`, `784` tensors
  - `outputs/absolute_plus050_local_model_root/model.safetensors`: `OK`, `784` tensors

### 结论

- 当前 `reacquire plus050` / `absolute plus050` 的本地 checkpoint root 已恢复到可加载状态。
- 之前的 `safetensors_rust.SafetensorError: incomplete metadata, file not fully covered` 已不再成立；后续若在线闭环仍失败，应继续排查 rollout / policy 行为本身，而不是 checkpoint 传输完整性。

## 2026-05-06: 本地 3090 在线闭环恢复与 `reacquire plus050` / `absolute plus050` 对照

### 本地兼容修复

- 为恢复本地 Isaac 在线闭环，本轮先修了两个本地兼容点：
  1. `collectors/sim_go2/envs/scene_builder.py`
     - 对 `kind: plane` 的场景，临时跳过 `TerrainImporterCfg` 的 ground-plane physics material bind。
     - 原因：当前本地 IsaacLab 5.1 / Isaac Sim 5.1 组合下，ground plane 路径会在 `bind_physics_material()` 里触发 `Stage.GetPrimAtPath(... NoneType)`；不跳过会在 scene 初始化阶段直接崩。
  2. `collectors/sim_go2/scripts/eval_closed_loop_policy.py`
     - 调整收尾顺序：先写 `summary.json`，再 `env.close()`，并补 `simulation_app.close()`。
     - 原因：旧逻辑会在 episode 已经跑完、`episode_000.json` 已落盘后卡在关闭阶段，导致 `summary.json` 不落盘。
- 说明：ground-plane 的 `ChangePropertyCommand` 仍会打印 Isaac warning，但本轮验证显示它不是致命错误；修复后流程已经可以进入真实 rollout 并稳定产出 `summary.json`。

### 运行设置

- 本地设备：`3090`
- config：`collectors/sim_go2/configs/collection_dual_target_contrast_room_local_turn_pilot.yaml`
- `num_episodes = 1`
- `seed = 0`
- `headless = true`
- `execute_steps = 1`
- 两条路线都使用同一任务：
  - `instruction = "approach the door"`
  - `active_target_id = door_target`
  - `turn_bucket = right_large`
  - `visibility_bucket = visible_after_small_turn`
  - `layout_id = 7842d35a9f8f69a5`

### 输出目录

- `reacquire plus050`
  - `outputs/closed_loop_reacquire_plus050_local_turnpilot_trace_20260506_0001`
- `absolute plus050`
  - `outputs/closed_loop_absolute_plus050_local_turnpilot_trace_20260506_0001`

### 结果

| route | success | collision | timeout | goal_distance | goal_clearance | steps |
| --- | --- | --- | --- | --- | --- | --- |
| `reacquire plus050` | false | false | true | `2.886449` | `2.546449` | `180` |
| `absolute plus050` | false | false | true | `2.000903` | `1.660903` | `180` |

### 结论

- 在这组本地真实 `turnpilot` 在线闭环上，两条路线都没有完成任务，且都以 `timeout` 结束。
- 但在完全相同的单任务条件下，`absolute plus050` 明显比 `reacquire plus050` 更接近目标：
  - `goal_distance`: `2.0009` vs `2.8864`
  - `goal_clearance`: `1.6609` vs `2.5464`
- 因此当前可以确认：
  - `reacquire plus050` 虽然在 fixed-id offline 上明显强于 `absolute plus050`，但这条本地在线闭环样本上并没有占优。
  - `absolute plus050` 虽然同样失败，但至少在这一个真实 rollout 上比 `reacquire plus050` 更接近目标。
- 这里的结论只基于 `1` 个 seed / `1` 个 layout / `1` 条 instruction；它足以说明“当前 offline 最优候选不自动等于在线更优”，但还不足以单独推翻主线排序。

### 启动阶段行为判读

- `reacquire plus050`
  - 起步阶段 `goal_heading` 初始为负（目标在右侧），模型输出也是负 `wz`，初始转向符号基本正确。
  - 到约 `step 32` 目标首次重新进入可见；到 `step 76` 左右 `goal_heading` 已接近 `0`。
  - 但模型没有在接近对准时减小/反转 `wz`，而是继续沿同一方向积分，导致：
    - `goal_heading` 从接近 `0` 继续跨到正值；
    - 约 `step 100` 后重新失去目标；
    - 后续一路把目标转到身后，最终 `final goal_heading ≈ 2.4793`。
- `absolute plus050`
  - 起步阶段 `goal_heading` 同样为负，但模型从 `step 1` 开始就输出正 `wz`，即起步转向符号直接错误。
  - 同时伴随较大的前向 `vx`（前几步迅速抬到 `0.35+`），属于“边前冲边往反方向转”。
  - 因此它虽然这次最终 `goal_distance` 比 `reacquire` 小，但本质上不是学会了正确 startup turn，而更像是靠前向推进偶然更接近了目标。
- 推断：
  - 两条路线都还没有学到可靠的 `turn/search/reacquire` 策略；
  - 只是错误模式不同：
    - `reacquire`: 初始 turn sign 基本对，但不会在接近对准后停转/回正；
    - `absolute`: 连 startup turn sign 都可能直接做反。

### 下一步

- 优先补做更小规模但更有统计性的本地在线对照，例如：
  1. 固定一组 `turnpilot` seeds / layouts，对 `reacquire plus050` 和 `absolute plus050` 各跑多条 episode。
  2. 对比两者在 `target_visible`、首次 re-acquire、turn direction consistency、near-goal stop 行为上的 trace 差异。
  3. 再决定是否需要调整“offline 首选候选 = online 首选候选”的当前假设。

## 2026-05-06: `reacquire plus050` 本地 5-episode 在线闭环复查（转向方向一致性 + 推理耗时）

### 背景

- 用户要求继续多做几次 `reacquire plus050` 本地在线闭环，重点看：
  1. 是否总是往正确方向旋转；
  2. 模型推理时间大概是多少。
- 为避免每次单独重启 Isaac 带来的显存残留 / OOM / 卡收尾问题，本轮改成单进程连续跑 `5` 个 episode。

### 本轮本地修复

- `deploy/go2/controller_loop.py`
  - 在 `model.predict_actions(...)` 前后加入计时，把
    - `predict_ms`
    - `safety_ms`
    - `control_loop_ms`
    写入每步 `model_debug`。
- `collectors/sim_go2/scripts/eval_closed_loop_policy.py`
  - 在 episode summary 与总 summary 中汇总以上耗时统计，输出 `mean / p50 / p95 / max`。

### 运行设置

- model：`outputs/reacquire_plus050_local_model_root`
- backbone：`/home/wxh/models/LLaDA-V`
- config：`collectors/sim_go2/configs/collection_dual_target_contrast_room_local_turn_pilot.yaml`
- output：
  - `outputs/closed_loop_reacquire_plus050_local_turnpilot_multi5_20260506`
- 参数：
  - `num_episodes = 5`
  - `seed = 0`
  - `headless = true`
  - `execute_steps = 1`

### 总结果

- `5 / 5` 全部 `timeout`
- `0 / 5` `success`
- `0 / 5` `collision`
- `mean_goal_clearance = 2.302842`
- `mean_predict_ms = 1062.789`
- `mean_control_loop_ms = 1079.776`

### 转向方向一致性判据

- 对每个 episode 记录：
  - `init_goal_heading`：reset 时目标相对机身的初始 heading
  - `mean_wz_first5`：前 `5` 步执行命令的平均 `wz`
  - `dyaw_first10`：前 `10` 步机体净转角
- 若 `mean_wz_first5` 与 `init_goal_heading` 同号，则记为“前 5 步转向方向正确”。
- 若 `dyaw_first10` 与 `init_goal_heading` 同号，则记为“前 10 步实际净转角方向正确”。

### 分 episode 结果

| episode | instruction | turn_bucket | init_goal_heading | mean_wz_first5 | dyaw_first10 | correct_wz_first5 | correct_dyaw_first10 | first_visible_step | goal_clearance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `000` | `approach the door` | `right_large` | `-0.4561` | `-0.00794` | `-0.00701` | yes | yes | `37` | `2.4694` |
| `001` | `go to the dark gray suitcase` | `left_large` | `0.4966` | `0.03225` | `0.04481` | yes | yes | `16` | `2.3560` |
| `002` | `approach the door` | `left_large` | `0.5413` | `-0.00735` | `-0.00819` | no | no | `null` | `2.1754` |
| `003` | `go to the dark gray suitcase` | `right_large` | `-0.5593` | `-0.01618` | `-0.01666` | yes | yes | `42` | `2.0947` |
| `004` | `move to the door` | `left_large` | `0.4220` | `-0.00794` | `-0.00810` | no | no | `null` | `2.4187` |

### 结论

- `reacquire plus050` 并不是“总是转向正确”。
- 按当前这 `5` 个 episode 的统计：
  - 前 `5` 步转向方向正确：`3 / 5`
  - 前 `10` 步实际净转角方向正确：`3 / 5`
- 两个明显失败的方向样本都是 `left_large`：
  - `episode_002`
  - `episode_004`
- 这两条失败样本的共同点：
  - `init_goal_heading > 0`（目标在左侧，需要左转）
  - 但前几步 `wz < 0`、净转角也 `< 0`，模型实际朝右转
  - `target_visible` 从未重新变成 `true`
- 另外三个“方向正确”的样本虽然能把目标重新带回视野（`first_visible_step = 16 / 37 / 42`），但最终仍然没完成任务，说明问题不只在初始转向符号，还包括后续 re-acquire / approach / stop 质量。

### 推理耗时

- 每步 `predict_actions` 推理耗时基本稳定在 `1.06s` 左右。
- `5` 个 episode 的 `predict_ms.mean` 分别为：
  - `1060.16`
  - `1066.33`
  - `1061.04`
  - `1062.56`
  - `1063.86`
- `control_loop_ms.mean` 分别为：
  - `1077.40`
  - `1083.09`
  - `1077.85`
  - `1080.29`
  - `1080.24`
- 说明当前本地闭环瓶颈几乎完全在模型推理本身；`safety_filter` 额外开销只有约 `0.18~0.19 ms`。

### 额外运行现象

- 若按“每个 seed 单独重启一个新 Isaac 进程”的方式跑，当前本地环境容易遇到：
  - 前一进程收尾不退出，残留约 `20 GiB` 显存占用；
  - 下一次 `load_model()` 直接 `CUDA OOM`；
  - 或单次 run 卡在长时间无产物状态。
- 单进程多 episode 路线明显更稳，因此后续本地批量闭环优先用这种方式。

### 决策建议

- 当前不建议在“完全相同的数据分布 + 完全相同训练目标”上继续盲目长训，期待模型自己把左转方向修正过来。
- 原因有两点：
  1. 问题已经不是纯随机波动，而是出现了可复现的方向性偏差：`left_large` 有 `2/2` 明显转错方向。
  2. 即使方向转对的 `3` 条 episode 也全部 `timeout`，说明问题不只在“先左转还是先右转”，还包括后续 `re-acquire / approach / stop` 质量。
- 因此更合理的下一步顺序是：
  1. 先用现有数据做一次定向 slice audit / rebalancing，不急着立刻大规模新采集。
  2. 如果 audit 证明 `left_large` 成功样本明显不足、或标签质量明显差，再做下一轮定向 sim 采集。

### 推荐执行顺序

1. 先做现有数据审计
   - 重点统计当前训练数据里：
     - `left_large` vs `right_large`
     - `target_visible_within_3f/10f`
     - 成功 re-acquire 的 episode 数
     - 首次转向方向与目标方位是否一致
   - 目标是先确认：现在的问题到底是“左转样本太少”，还是“左转样本有，但监督本身就不干净 / 不一致”。
2. 先做一版现有数据上的 targeted fine-tune
   - 不建议直接无脑继续全量长训。
   - 更建议基于当前 `reacquire plus050`，构一版 failure-driven / balanced manifest：
     - 提高 `left_large`
     - 提高“目标重新进入视野”的成功样本
     - 降低已经反复证明会把目标转丢的样本权重
   - 这是成本最低、信息增益最高的一步。
3. 若现有数据 audit 后发现 `left_large` 正样本不够，再做新采集
   - 新采集不应泛泛扩大数据量，而应明确针对：
     - `visible_after_small_turn`
     - `left_large`
     - 目标重新进入视野后继续 approach 成功
   - 最好做左右镜像配对采集，让 `left_large` / `right_large` 在布局、目标、指令模板上尽量对称。

### 当前判断

- 结论不是“立刻停训，只做新采集”，也不是“继续用旧数据硬训到转对”为止。
- 当前最佳路线是：
  - **先审计现有数据并做一版定向再平衡 fine-tune**
  - **只有当现有数据不足以覆盖 `left_large` 成功转向 / re-acquire 行为时，再进入下一步定向采集**

### 若只选一个下一步

- 首选：先做 `left_large / right_large / re-acquire 成功率 / 首次转向方向一致性` 的现有数据 audit，并据此构一版 targeted manifest。
- 这一步如果做完，我再决定要不要进入新的 sim 数据采集计划，会比现在直接拍脑袋扩数据更稳。

## 2026-05-06: 下一位 Codex 可直接实施的执行计划

### 目标

- 先验证当前问题到底是：
  1. `left_large` 样本量不足；
  2. `left_large` 标签/行为监督不一致；
  3. 即使方向转对，后续 `re-acquire / approach / stop` 仍然不足。
- 在此基础上决定：
  - 先用现有数据做定向再平衡训练；
  - 还是先进入下一轮定向 sim 采集。

### Phase A: 现有数据审计

1. 对当前主线训练数据做 `turnpilot` 切片审计
   - 数据范围：当前 `reacquire plus050` 所用训练数据与其 targeted manifest。
   - 至少统计：
     - `left_large` / `right_large` / 其他 `turn_bucket` 数量
     - `target_visible_first_frame`
     - `target_visible_within_3f`
     - `target_visible_within_10f`
     - `active_target_id`
     - instruction template 分布
2. 做行为层审计
   - 重点检查 `left_large` 样本中：
     - 前几步 `wz` 符号是否与目标方位一致
     - 目标是否真的在转向后重新进入视野
     - 成功 episode 与失败 episode 的比例
3. 产出一个 audit summary
   - 明确回答：
     - `left_large` 是否明显欠采样
     - `left_large` 是否存在监督不一致
     - `reacquire` 成功样本是否本身就太少

### Phase B: 基于现有数据的 targeted manifest

1. 若 audit 显示现有数据仍可用，先不急着新采集
2. 构一版新的训练 manifest，原则如下：
   - 提高 `left_large` 权重
   - 提高“目标在 3~10 frame 内重新进入视野”的样本权重
   - 提高最终成功接近目标的样本权重
   - 降低会把目标持续转丢的样本权重
3. 若可行，做左右对称平衡
   - 让 `left_large` 与 `right_large` 在目标类别、instruction template、layout 模式上尽量对称

### Phase C: 小步快跑训练

1. 从当前 `reacquire plus050` checkpoint 出发，不做全量长训
2. 先跑一版短程 targeted fine-tune
   - 目标不是整体 loss 更低，而是先看：
     - `left_large` 首次转向方向是否改善
     - `target_visible_within_3f/10f` 是否改善
     - fixed-id offline 是否仍保持不差
3. 训练完成后，先做：
   - fixed-id offline
   - 本地单进程多 episode 在线闭环

### Phase D: 在线闭环验收标准

1. 不再只看最终 success rate
2. 至少同时看下面几项：
   - 前 `5` 步 `wz` 与 `init_goal_heading` 同号比例
   - 前 `10` 步净转角方向正确比例
   - `first_visible_step`
   - `target_visible` 是否重新变 `true`
   - `goal_clearance`
   - `success / timeout / collision`
3. 若新模型在“转向方向正确率”上仍不过关，则不要继续盲目长训

### Phase E: 触发新采集的条件

- 只有满足下面任一条件，才进入下一轮定向 sim 采集：
  1. audit 证明 `left_large` 成功样本明显不足；
  2. 现有 `left_large` 样本监督明显不一致，难以靠 reweight 修正；
  3. 用现有数据做 targeted fine-tune 后，在线“转向方向正确率”仍无明显改善。

### Phase F: 若要新采集，采什么

1. 新采集不要泛化扩量，要定向针对：
   - `visible_after_small_turn`
   - `left_large`
   - reacquire 后继续 approach 成功
2. 采集时尽量做左右镜像配对
   - 同类 layout
   - 同类目标
   - 同类 instruction template
   - 仅改变目标初始位于左侧还是右侧
3. 新采集后的第一轮用途
   - 不直接替代主数据集
   - 先作为 targeted subset 验证是否真的修复 `left_large` 转向偏差

### 新 Codex 的推荐执行顺序

1. 先完成 Phase A audit
2. 再决定：
   - 若现有数据可救：走 Phase B + C
   - 若现有数据明显不够：进入 Phase F
3. 每做完一个阶段，都把结论继续追加到本文件

## 2026-05-04: 5090 远端数据盘保守清理与 absolute 队列修正

### 背景

- 远端 `outputs/` 再次累积了多份 32G~48G 级训练目录；清理前 `/root/autodl-tmp` 使用率回到 `436G / 500G (88%)`。
- 当天需要继续跑 `absolute`，因此清理策略仍保持保守：保留当前主线 `plus100`、`priority_weighted`、`reacquire` 和正在运行的 `absolute`，只删除已证伪或已停止的旧大目录。

### 删除目录

- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_weak_anchor_no_prevprompt_plus025_20260427_005148`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_weak_anchor_no_prevprompt_plus050_20260427_005148`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus025_20260426_220225`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_1_object_contrast_no_prevprompt_plus050_20260426_220225`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus025_20260427_005148`

### 清理效果

- `/root/autodl-tmp`: `436G / 500G (88%) -> 261G / 500G (53%)`
- 可用空间：`65G -> 240G`

### 保留的大目录

- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_plus100_20260426_143200_retry1`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_2gpu_priority_weighted`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus025_20260427_011001`
- `outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus050_20260427_011001`
- `outputs/train_current_v1_sim_today_main_turnpilot_absolute_targeted_v2_2_no_prevprompt_plus025_20260427_005148`（运行中）

### 队列问题与修正

- 发现 `absolute` 的等待队列脚本存在一个非预期行为：等待条件里把 `run_targeted_v2_2_absolute_20260427_005148.sh` 自己也算作“活跃任务”，导致队列永远不放行。
- 处理方式：
  - 终止这次错误挂起的 queue shell。
  - 重新做 `PRECHECK_ONLY=1` 确认 config / dataset / init checkpoint / output dir 均正常。
  - 手动直接启动 `scripts/run_targeted_v2_2_absolute_20260427_005148.sh`。

### 结论

- 这轮清理释放空间充足，当前后续训练不再受数据盘压力约束。
- `weak_anchor` 已有 trainer 结果且明显偏差，相关大目录已清理。
- `absolute` 已脱离错误等待队列，重新进入双卡训练。

## 2026-05-06: Phase A 现有数据审计已执行（left/right + reacquire proxy）

### 背景

基于 `reacquire plus050` 本地 5-episode 在线闭环，当前问题集中在 `left_large` 首次转向方向不稳定，以及方向转对后仍无法完成 re-acquire / approach / stop。上一节计划要求先做现有数据审计，再决定是否构造新的 targeted manifest 或进入定向采集。

### 动作

新增只读审计脚本：

- 本地/远端脚本：`scripts/audit_turnpilot_reacquire_distribution.py`

该脚本读取 train jsonl、可选 targeted manifest、可选 rollout trace，输出：

- `audit_summary.json`
- `audit_report.md`

重要口径：当前 packed train jsonl 不保留显式 `turn_bucket` / `visibility_bucket` / `target_visible_within_3f/10f`，因此本轮 left/right 和 re-acquire 分析使用 proxy：

- 用 `scene_targets` + `state.x/y/yaw` 计算 active target 相对 heading。
- 用监督动作 chunk 的 `wz` 符号判断动作是否朝目标方位旋转。
- 所有 `*_proxy` 结论不能当作 privileged simulator visibility 真值。

### 关键路径

远端 source train 审计：

- input train: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_20260425/train.jsonl`
- input manifest: `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_2_reacquire_proxy_20260427_005148/targeted_v2_2_reacquire_proxy_manifest_train.jsonl`
- output: `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/turnpilot_reacquire_distribution_audit_20260506_phaseA_source_train`

远端 weighted train 审计：

- input train: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_2_reacquire_proxy_20260427_005148/train.jsonl`
- input manifest: `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_2_reacquire_proxy_20260427_005148/targeted_v2_2_reacquire_proxy_manifest_train.jsonl`
- output: `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/turnpilot_reacquire_distribution_audit_20260506_phaseA_r2`

本地 smoke / rollout 复核：

- train: `datasets/current_v2_sim_today_main_turnpilot_20260425/train.jsonl`
- rollout: `outputs/closed_loop_reacquire_plus050_local_turnpilot_multi5_20260506`
- output: `/tmp/turnpilot_audit_local_smoke_20260506_r2`

### 核心结果

Source train（未加权）：

| metric | value |
| --- | ---: |
| rows | `18846` |
| episodes | `188` |
| explicit turn buckets | `{}` |
| explicit visibility buckets | `{}` |
| center_proxy | `16916` |
| left_small_proxy | `854` |
| right_small_proxy | `760` |
| left_large_proxy | `148` |
| right_large_proxy | `168` |
| row direction consistent | `2720` |
| row direction not_applicable_small_heading_or_wz | `16126` |

Weighted train（V2.2 reacquire proxy 加权后）：

| metric | value |
| --- | ---: |
| rows | `27489` |
| episodes | `188` |
| explicit turn buckets | `{}` |
| explicit visibility buckets | `{}` |
| center_proxy | `22694` |
| left_small_proxy | `2117` |
| right_small_proxy | `1900` |
| left_large_proxy | `362` |
| right_large_proxy | `416` |
| row direction consistent | `6783` |
| row direction not_applicable_small_heading_or_wz | `20706` |

Manifest overlay（weighted train）：

- `reacquire_turn_proxy`: `9557` weighted rows，其中 `left_large_proxy=362`，`right_large_proxy=416`，`left_small_proxy=2117`，`right_small_proxy=1900`，`center_proxy=4762`。
- `wz_change`: `3347` weighted rows，其中 `left_large_proxy=362`，`right_large_proxy=416`，`left_small_proxy=1327`，`right_small_proxy=1166`。
- `low_vx_but_should_move`: `926` weighted rows，其中 `left_large_proxy=144`，`right_large_proxy=164`。

本地 rollout smoke 复核：

- `reacquire plus050` 5 条在线闭环仍为 `0/5 success`，`5/5 timeout`。
- rollout 方向统计复现人工判读：`consistent=3`，`inconsistent=2`。
- `left_large` 中 `consistent=1`，`inconsistent=2`；`right_large` 中 `consistent=2`。

### 结论

- 当前计划“先 audit、不要直接盲训”是对的。
- 现有 packed train 不能直接回答 `target_visible_within_3f/10f` 或真实 re-acquire 成功率，因为这些 privileged visibility 字段没有进入训练 jsonl。
- 按 heading/action proxy 看，训练监督本身没有明显 left/right 符号反转问题；当前 source train 和 weighted train 的可判定行都是 `consistent`，没有发现 `inconsistent` 行。
- `left_large` 并不是相对 `right_large` 极端欠采样：source train 为 `148` vs `168`，weighted train 为 `362` vs `416`。但两者绝对数量都很小，large-turn/re-acquire 行为占比低。
- V2.2 reacquire manifest 已经覆盖了所有 large/small turn proxy，但同时仍包含大量 `center_proxy` re-acquire/object-contrast 样本；如果继续基于现有数据做 Phase B，不能只简单说“补左转”，更合理的是提高 large-turn / post-alignment / turn-stop 类样本权重，并降低不提供方向决策信号的 center/static 样本权重。

### 下一步

- 不建议直接长训。
- 若继续现有数据路线，Phase B 应构造一版 `large-turn + left_large focused + post-alignment turn-stop` 的新 targeted manifest，并在生成 dataset 前先做 dry-run audit。
- 由于当前 train jsonl 缺少真实 visibility 标签，Phase B 的 `target_visible_within_3f/10f` 只能用几何/动作 proxy 近似；如果需要真实 re-acquire 成功率，下一步必须让采集/packing 保留 `turn_bucket`、`visibility_bucket`、`target_visible_*` 或从 raw sim trace 重新回填。

## 2026-05-06: V2.3 turn-reacquire 训练前准备完成，等待用户重启/启动远端 GPU

### 用户流程要求

用户明确要求后续训练类任务采用固定流程：

- 先完成所有准备工作：代码 review、配置检查、路径/IO debug、manifest/dataset 校验、smoke/preflight。
- 真正需要占用 GPU 开训前必须停下汇报。
- 等用户重启远端服务器或启动显卡并确认后，再开始训练。
- 以后训练任务默认也按该流程执行。

该要求已写入全局 ACTIVE memory。

### 本轮动作

基于 Phase A 结果，新增 V2.3 turn-reacquire manifest builder：

- `scripts/build_targeted_v2_3_turn_reacquire_manifest.py`

新增训练配置：

- `configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_3_turn_reacquire.yaml`

新增训练启动脚本（只准备好，不自动运行训练）：

- `scripts/run_targeted_v2_3_turn_reacquire_20260506_ready.sh`

训练脚本默认语义：

- init checkpoint: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus050_20260427_011001`
- data root: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready`
- config: `/root/autodl-tmp/llada-vla-go2/configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_3_turn_reacquire.yaml`
- output prefix: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_3_turn_reacquire_no_prevprompt`
- default epochs: `0.25 0.50`
- default GPUs when user confirms training: `CUDA_VISIBLE_DEVICES=0,1`, `NPROC_PER_NODE=2`
- `RUN_TS=20260506_ready`
- `TB_PREFIX=v2_3_turn_reacquire`

### V2.3 manifest / dataset 工件

正式 manifest：

- `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready/targeted_v2_3_turn_reacquire_manifest_train.jsonl`
- `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready/targeted_v2_3_turn_reacquire_manifest_summary_train.json`

正式 weighted dataset：

- `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready`

校验报告：

- `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready/targeted_v2_3_turn_reacquire_weighted_manifest_validation.json`
- `/root/autodl-tmp/llada-vla-go2/outputs/data_audit/turnpilot_reacquire_distribution_audit_20260506_v2_3_ready`

### 核心检查结果

Manifest summary：

- manifest rows: `18846`
- leaked validation ids removed: `0`
- priority buckets:
  - `target_object_contrast_near_step`: `5440`
  - `post_alignment_turn_stop_proxy`: `3833`
  - `stop_should_hold`: `3775`
  - `first_step_commit`: `1818`
  - `reacquire_turn_proxy`: `1614`
  - `normal_cruise`: `1211`
  - `residual_recovery`: `807`
  - `large_turn_reacquire_proxy`: `168`
  - `left_large_reacquire_proxy`: `148`
  - `wz_change`: `32`

Weighted dataset summary：

- train rows: `28631`
- val rows: `1985`
- test rows: `0`
- weighted expansion ratio: `1.5192`
- dataset size: `217M`
- data audit size: `15M`
- `/root/autodl-tmp` free space after prep: `160G` free, usage `69%`

Weighted manifest validation:

- pass: `true`
- errors: `0`
- leaked validation ids count: `0`
- weighted rows: `28631`
- unique source manifest rows: `18846`

V2.3 weighted audit:

- explicit `turn_bucket`: still absent (`{}`), so left/right labels remain proxy-only.
- explicit `visibility_bucket`: still absent (`{}`), so true `target_visible_within_3f/10f` cannot be recovered from train jsonl.
- row turn proxy counts:
  - `center_proxy`: `23408`
  - `left_small_proxy`: `2118`
  - `right_small_proxy`: `1923`
  - `left_large_proxy`: `592`
  - `right_large_proxy`: `590`
- manifest overlay confirms large-turn balance:
  - `large_turn_reacquire_proxy`: `left_large_proxy=592`, `right_large_proxy=590`
  - `left_large_reacquire_proxy`: `left_large_proxy=592`
  - `post_alignment_turn_stop_proxy`: `center_proxy=7666`

Config / script / path checks:

- local `bash -n` passed for V2.3 launch script.
- remote `bash -n` passed for V2.3 launch script.
- YAML parsed successfully with local `/home/wxh/miniconda3/envs/llada/bin/python`.
- YAML parsed successfully with remote `/root/miniconda3/envs/llada-vla/bin/python`.
- remote non-GPU path checks passed:
  - config exists
  - `train.jsonl` / `val.jsonl` exist
  - `images` exists
  - init checkpoint `model.safetensors` / `robotics_velocity_config.json` exist
  - planned `plus025` / `plus050` output dirs do not exist

### 当前唯一未完成项

远端当前 `nvidia-smi` 不可用或 GPU 未启动，因此未运行完整 `PRECHECK_ONLY=1` GPU preflight。按用户要求，本轮不启动训练，等待用户重启/启动远端 GPU 后再执行：

```bash
cd /root/autodl-tmp/llada-vla-go2
PRECHECK_ONLY=1 bash scripts/run_targeted_v2_3_turn_reacquire_20260506_ready.sh
```

如果 GPU preflight 通过，再由用户确认后启动实际训练：

```bash
cd /root/autodl-tmp/llada-vla-go2
bash scripts/run_targeted_v2_3_turn_reacquire_20260506_ready.sh
```

### 结论

除远端 GPU 当前未启动导致无法做完整 GPU preflight 外，V2.3 turn-reacquire 的训练前准备已完成。当前应停在这里，不开始训练。

## 2026-05-07: V2.3 turn-reacquire 正式训练启动

### 用户确认

用户提供远端连接并要求启动正式训练。按既定流程，先执行完整 GPU preflight，通过后才启动训练。

### Preflight

远端：

- `ssh -p 47766 root@connect.bjb2.seetacloud.com`
- repo: `/root/autodl-tmp/llada-vla-go2`

GPU precheck：

- GPU0: RTX 5090, `0 MiB / 32607 MiB`, util `0%`
- GPU1: RTX 5090, `0 MiB / 32607 MiB`, util `0%`

完整脚本 preflight：

```bash
cd /root/autodl-tmp/llada-vla-go2
PRECHECK_ONLY=1 bash scripts/run_targeted_v2_3_turn_reacquire_20260506_ready.sh
```

通过项：

- `RUN_TS=20260506_ready`
- `CONFIG_PATH=/root/autodl-tmp/llada-vla-go2/configs/llada_vla_go2_sim_today_main_turnpilot_anchor_residual_no_prevprompt_train_v2_3_turn_reacquire.yaml`
- `DATA_ROOT=/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready`
- `CUDA_VISIBLE_DEVICES=0,1`
- `NPROC_PER_NODE=2`
- `action_target_mode=anchored_residual`
- `anchor_mode=repeat_prev`
- `use_prev_action_as_condition=false`
- planned `plus025` / `plus050` output dirs 不存在。

### 启动方式

远端当前没有 `tmux`（`command -v tmux` 返回空），因此改用 `nohup` 后台启动：

```bash
cd /root/autodl-tmp/llada-vla-go2
nohup bash scripts/run_targeted_v2_3_turn_reacquire_20260506_ready.sh \
  > outputs/run_targeted_v2_3_turn_reacquire_20260506_ready.master.log 2>&1 < /dev/null &
```

master log：

- `/root/autodl-tmp/llada-vla-go2/outputs/run_targeted_v2_3_turn_reacquire_20260506_ready.master.log`

当前进程：

- wrapper PID: `1626`
- torchrun PID: `1686`
- worker PIDs: `1703`, `1704`

当前阶段：

- 正在跑 `plus025`
- output dir: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_3_turn_reacquire_no_prevprompt_plus025_20260506_ready`
- TensorBoard symlink: `/root/tf-logs/llada-vla-go2-targeted-v2/v2_3_turn_reacquire_plus025`

### 启动后状态

- 双卡已被训练占用：
  - GPU0: about `23097 MiB / 32607 MiB`, util about `95%`
  - GPU1: about `23093 MiB / 32607 MiB`, util about `94%`
- 训练已进入 step loop：
  - `plus025`: `8 / 112` steps 左右
  - step 5 trainer log: `loss=0.5607`, `grad_norm=2.9728`, `lr=9.9074e-05`
- checkpoint artifact 成功加载：
  - init checkpoint: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_2_reacquire_no_prevprompt_plus050_20260427_011001`
  - `Loaded adaptation checkpoint type=full`
- trainable params:
  - `trainable_params=49,816,576`
  - LoRA applied to language backbone layers `[28, 29, 30, 31]`

### 监控项

- `train target clip stats` 中 `wz target_clip_rate=0.013510`，触发 `target_clip_warning=true`。
- `eval target_clip_rate.wz=0.007545`，未触发 warning。
- 当前判断：这是数据 reweight 后 large-turn 样本增强带来的合理监控项，不是启动失败；后续看 plus025/plus050 eval 与 policy validation 再判断是否需要调 `delta_wz_range` 或权重。

## 2026-05-07: V2.3 turn-reacquire 训练完成

### 结果

训练脚本按 `EPOCHS_LIST="0.25 0.50"` 顺序执行，已确认 `plus025` 成功结束后自动进入 `plus050`，且 `plus050` 也正常结束。

Master log：

- `/root/autodl-tmp/llada-vla-go2/outputs/run_targeted_v2_3_turn_reacquire_20260506_ready.master.log`

输出目录：

- `plus025`: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_3_turn_reacquire_no_prevprompt_plus025_20260506_ready`
- `plus050`: `/root/autodl-tmp/llada-vla-go2/outputs/train_current_v1_sim_today_main_only_subset_anchor_residual_targeted_v2_3_turn_reacquire_no_prevprompt_plus050_20260506_ready`

Artifact 检查：

- 两个输出目录根部均有：
  - `model.safetensors`，大小 `16939227640` bytes
  - `robotics_velocity_head.bin`，大小 `72378417` bytes
  - `robotics_velocity_config.json`，大小 `1872` bytes
  - `training_args.bin`
- `plus025` 有 `checkpoint-50` / `checkpoint-100`。
- `plus050` 有 `checkpoint-150` / `checkpoint-200`。

### Trainer 指标

`plus025`:

- step `50` eval_loss: `0.4791106879711151`
- step `100` eval_loss: `0.4221739172935486`
- final train runtime: `2010.0677s`
- final train_loss: `0.5393994844385556`

`plus050`:

- step `50` eval_loss: `0.4965696930885315`
- step `100` eval_loss: `0.438001424074173`
- step `150` eval_loss: `0.3952696919441223`
- step `200` eval_loss: `0.37737929821014404`
- final train runtime: `4087.1304s`
- final train_loss: `0.49113679624029566`

### 结论

- 自动接续逻辑已经实证通过：`plus025` 完成后确实进入并完成了 `plus050`。
- 训练 artifact 完整，可进入下一步 fixed-id policy validation。
- fixed-id policy validation 尚未在本轮记录中执行；如果继续验证，需要占用 GPU。

## 2026-05-06: V2.3 训练数据 sanity web 已生成

### 动作

用户要求用现成 sanity web 可视化准备训练的数据。本轮复用 canonical packed VLA sanity web：

```bash
cd /root/autodl-tmp/llada-vla-go2
/root/miniconda3/envs/llada-vla/bin/python scripts/sanity_check_canonical_vla.py \
  --data-root datasets/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready \
  --output-dir datasets/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready/sanity_webui \
  --seed 0 \
  --num-samples 6
```

### 输出

- index: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready/sanity_webui/index.html`
- summary: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_only_subset_targeted_v2_3_turn_reacquire_20260506_ready/sanity_webui/summary.json`
- episode pages: `208`
- frame count: `30616`
- split counts: train `176` episodes, val `32` episodes
- image symlink target: `/root/autodl-tmp/llada-vla-go2/datasets/current_v2_sim_today_main_turnpilot_20260425`

### 注意

- `visibility_bucket_counts = {"unknown": 208}`，`visible3/visible10` ratio 为 `0.0`。这是因为当前 V2.3 packed dataset 没有保留显式 `visibility_bucket` / `target_visible_*` 字段，不是 sanity web 生成失败。
- 因为这是 weighted training dataset 的 `dataset.jsonl`，页面中可能体现 weighted copy 后的重复训练帧；这是为了可视化“实际将被训练消费的分布”。

## 2026-05-23: 闭环 `execute_steps > 1` 实际多步执行修正

### 背景

- 用户确认当前在线闭环每次模型预测 `horizon_k=4`，但实际只执行 `execute_steps=1`。
- 检查发现 `deploy/go2/controller_loop.py` 虽然把 `execute_steps` 传给 `model.predict_actions(...)`，但 `run_once()` 固定只发送 `outputs["execute_actions"][0, 0]`，因此 `--execute-steps 2/3/4` 之前不会真正多步执行。

### 修改

- `Go2ControllerLoop.run_once()` 现在会遍历 `outputs["execute_actions"][0]`：
  - 每个 predicted action step 都经过 `VelocitySafetyFilter.apply(...)`；
  - 每个安全后的 command 都通过 `Go2VelocityCommander.send(...)` 下发；
  - step `>0` 前会重新读取当前 observation/state，用于后续 safety filter；
  - 每个实际发送 step 都写入 `model_debug`，新增 `execute_step_index`。
- 返回值保持兼容：
  - 单步执行仍返回 shape `(3,)`；
  - 多步执行返回 shape `(execute_steps, 3)`。

### 验证

- 新增/更新测试：`tests/integration/test_mock_deploy.py`
- RED 已确认旧实现下 `execute_steps=3` 只返回/发送单步。
- GREEN 验证命令：

```bash
PYTHONPATH="$PWD:$PWD/LLaDA-V:$PWD/LLaDA-V/train" \
/home/wxh/miniconda3/envs/llada/bin/python -m pytest tests/integration/test_mock_deploy.py -q
```

结果：`4 passed, 1 warning`。

### 运行语义

- 对当前 Isaac Sim turnpilot 配置，单个 action step 仍对应 `control_dt=0.05s`。
- 因此若后续用 `--execute-steps 2`，每次推理后将 open-loop 执行约 `0.10s`；`--execute-steps 4` 则约 `0.20s`。
- 建议下一次真实闭环对照先跑 `execute_steps=2`，不要直接从 1 跳到 4。

## 2026-05-23: `execute_steps=2` 本地闭环 smoke 尝试

### 运行前预检

- 当前没有正在运行的 `eval_closed_loop_policy.py` / Isaac Sim 闭环进程。
- GPU 3090 预检时约 `789 MiB / 24576 MiB` 已用，其中一个无关 memory_world_model Python 进程约 `752 MiB`。
- 磁盘 `/home/wxh/llada-vla-go2` 可用约 `135G`。

### 尝试 1：CUDA sim，5 episodes

输出目录：`outputs/closed_loop_reacquire_plus050_local_turnpilot_exec2_20260523_231050`
日志：`outputs/closed_loop_reacquire_plus050_local_turnpilot_exec2_20260523_231050.log`

命令核心参数：

- model: `outputs/reacquire_plus050_local_model_root`
- backbone: `/home/wxh/models/LLaDA-V`
- config: `collectors/sim_go2/configs/collection_dual_target_contrast_room_local_turn_pilot.yaml`
- `num_episodes=5`
- `seed=0`
- `execute_steps=2`
- sim device: default `cuda:0`

结果：未生成 `summary.json` / episode trace。Isaac/PhysX 初始化阶段报 CUDA OOM：

- `Unable to allocate memory of size 671088640 for mGpuContactPairsDev`
- `Failed to fetch DOF velocity attribute`

推断：LLaDA-V 模型加载到 3090 后，Isaac/PhysX GPU contact pair buffer 再申请约 640MiB 时失败；这次不是 rollout 逻辑失败，而是 CUDA sim 初始化资源失败。

### 尝试 2：CPU sim，1 episode smoke

输出目录：`outputs/closed_loop_reacquire_plus050_local_turnpilot_exec2_smoke_cpu_sim_20260523_231831`
日志：`outputs/closed_loop_reacquire_plus050_local_turnpilot_exec2_smoke_cpu_sim_20260523_231831.log`

命令核心参数：

- `num_episodes=1`
- `execute_steps=2`
- `--sim-device cpu`

结果：已生成 `summary.json` 和 `episodes/episode_000.json`。Isaac 进程在输出 summary 后曾短暂卡在退出阶段并占用约 `19.8GiB` GPU，但最终自行正常退出（exit code 0）；退出后 GPU 只剩无关 memory_world_model 进程约 `752MiB`。

关键验证结论：

- episode trace 中 `step_records=180`，`command_count=180`。
- `model_debug.execute_step_index` 计数为 `{0: 90, 1: 90}`，说明 `execute_steps=2` 的多步发送逻辑确实生效：90 次模型推理，每次发送 2 个 action。
- `mean_predict_ms≈1070.8`，`mean_control_loop_ms≈1130.6`。

但 CPU sim 结果本身不可作为策略效果评估：

- summary 里 `goal_distance≈362.999`，trace 最后目标位置为 `(256, 256)`，即目标保持在 hidden target 位置附近。
- 这说明 `--sim-device cpu` 下当前 scene/target 写入或同步不符合原 CUDA sim 语义；该 smoke 只验证了 controller 多步执行路径，不验证导航性能。

### 下一步建议

- CPU sim smoke 已自行正常退出，无需再 kill 残留进程。
- 若继续真实闭环评估 `execute_steps=2`，优先解决 CUDA sim OOM，而不是继续用 CPU sim 做性能结论。
- 可选方向：降低 Isaac/PhysX GPU 内存需求、调整加载顺序/渲染配置，或用更大显存机器跑 `execute_steps=2` 对照。
