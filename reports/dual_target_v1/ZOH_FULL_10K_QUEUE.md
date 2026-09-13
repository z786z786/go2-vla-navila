# C全量微调10k执行记录

## 当前：从step 2000排队续训

不使用显存占位。续训仍需空闲13824 MiB连续3次并锁内复查，运行保护2048 MiB。v4的 `checkpoint_002000` 已深度验证；新输出 `zoh_full_10000_resume_0907_v1`，队列PID 2411033，当前因free约7710 MiB处于 `WAITING_GPU`，没有模型分配。恢复时载入完整Adam、scheduler、RNG与样本游标，并在step2010对照中断前已有loss证明连续性；正式新增checkpoint为4k/6k/8k/10k，2k由父checkpoint提供。TensorBoard run为 `zoh_full_10000_resume_v1`，仍只写五个训练指标。

## 当前运行：v4

用户明确授权本次C准入降到13824 MiB（13.5 GiB），2048 MiB运行保护不变。v2在0 step停止；v3在step 1被错误的expert-only首loss基线拦截，保留失败证据。修复后scope-aware门引用已完成full-5k基线，v4首步loss `0.8879846446216106` 与full基线完全相等；不是放宽数值容差。

v4队列PID 2360823，supervisor 2360981，trainer 2362857；输出 `outputs/dual_target_v2/zoh_full_10000_0907_v4`，队列根 `outputs/dual_target_v2/zoh_full_10000_queue_0907_v4`，TensorBoard run `zoh_full_10000_v4`。step-1的全量梯度、三组代表权重实际更新、scheduler cursor、固定预测变化均通过；实测峰值allocated约7760 MiB。TensorBoard事件实读恰好包含约定五项训练指标。任务继续到10k，随后最终重载与全部checkpoint深度校验并停止；不运行validation/eval/rollout。

## 历史：v2等待队列

从同一 `smolvla_base` fresh启动C：全量450,046,176参数可训练，包括vision/VLM；BF16 autocast、FP32 master/Adam，vision使用non-reentrant activation checkpointing。除训练范围外，与B保持同一已批准train-only数据、160000采样索引、seed、micro4/accum4、有效batch16、5 Hz/chunk5/execute1、loss、AdamW和官方warmup/cosine scheduler。

队列仅在GPU空闲至少14336 MiB连续3次（30秒间隔）并通过锁内复查后启动，运行中保留2048 MiB。正式checkpoint为2k/4k/6k/8k/10k，保存并验证模型、处理器、optimizer、scheduler、RNG、采样游标及文件hash。训练完成后重载最终模型/处理器并复核固定预测。TensorBoard run `zoh_full_10000_v2` 只写五个训练指标。不读取validation/test、不计算validation loss、不运行Isaac或rollout。

preflight已通过。队列PID 2331147，supervisor 2331245，输出 `outputs/dual_target_v2/zoh_full_10000_0907_v2`，队列根 `outputs/dual_target_v2/zoh_full_10000_queue_0907_v2`，日志 `/tmp/go2_zoh_full_10000_queue_0907_v2.log`。启动时free约7529 MiB，当前0 step等待中。
