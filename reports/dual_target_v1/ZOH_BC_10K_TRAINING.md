# B/C fresh 10k train-only 执行记录

## 范围

本轮只回答“BC loss 是否因 5k 预算不足而尚未收敛”这一训练问题，不做任何导航评估。旧 B/C train rollout 已中断并完整保留；validation/test 不被读取，训练期间和训练完成后均不启动 Isaac 或 rollout。完成 B 后自动串行开始 C，C 完成并核验后流水线停止，等待用户。

## 固定对照

| 项目 | B | C |
|---|---|---|
| 初始化 | 同一 smolvla_base，fresh | 同一 smolvla_base，fresh |
| 可训练范围 | action expert＋state/action/time projections | 全部 450,046,176 参数，含 vision/VLM |
| 数据 | 已批准 train 16 trajectories / 770 frames | 完全相同 |
| 样本顺序 | seed 20260906，160000 个有放回 episode-balanced 索引 | 与 B 逐项相同 |
| batch | micro4 × accum4 = effective16 | 相同 |
| 时间接口 | 5 Hz，chunk5，execute1 | 相同 |
| 优化器 | AdamW，peak LR 1e-4，clip10 | 相同 |
| scheduler | 官方 warmup＋cosine，按 10000 updates 构建 | 相同 |
| 精度 | BF16 autocast；可训练 master/Adam FP32 | BF16 autocast；全部 master/Adam FP32，vision non-reentrant checkpointing |

官方 scheduler 的配置参数仍为 warmup=1000、decay=30000，但 LeRobot builder 会按本次 10000 总步数缩放；实测 step1 使用约 2.994e-7，step334 达到约 9.973e-5，step10000 使用约 2.500e-6。日志记录的是本次 optimizer update 实际使用的 LR，不用 `lr_next` 冒充。

## Checkpoint 和门槛

只生成 `checkpoint_002000/004000/006000/008000/010000`。每个 checkpoint 原子写入并立刻检查全部文件哈希，以及 optimizer update、scheduler cursor、Adam step、RNG、数据/normalizer/sample hash 和 consumed-sample cursor。step1 只做有限梯度、权重实际更新和固定输入预测变化检查，不生成正式 checkpoint。step10000 后释放旧模型/optimizer，再从 final checkpoint 重载模型与处理器，要求固定输入/噪声预测一致。

TensorBoard 新 run：`zoh_scheduler_10000` 和 `zoh_full_10000`，只允许：

- `Loss/train`
- `Optimization/learning_rate`
- `Optimization/gradient_norm_before_clip`
- `Performance/step_time_s`
- `Memory/peak_allocated_MiB`

## 资源与失败语义

B 需连续三次空闲至少 7 GiB；C 至少 14 GiB；均保留运行中 2 GiB 保护，只终止自己创建的进程组，不处理外部 GPU 作业。B 最长 6 小时，C 最长 12 小时。资源不足时等待；任何训练、checkpoint、reload 或哈希门失败则保留现场并停止，不跳过到下一组。

远端 preflight 已通过：train=770、采样域 0–769、总样本 160000、前缀 80000 完全一致、10k LR 曲线数值一致、checkpoint 集合固定、三个新执行模块无 eval/Isaac 导入、TensorBoard 五标签精确匹配。远端 SmolVLA 环境不带 pytest，因此使用同环境直接运行的 `scripts.zoh_10k_preflight.py`；未为测试临时修改训练环境。

## 启动实况

流水线 PID 2232478，B supervisor 2232542、trainer 2232705、TensorBoard bridge 2232712。B 的 step1 loss=0.8877298534，与既有同 base/数据/采样/RNG 起点差为0；实际 LR=2.994011976e-7，action projection 最大更新约2.999e-7，固定预测最大变化约0.003052，有限梯度和 scheduler cursor 均通过。此时未创建 checkpoint。TensorBoard event accumulator 实读只含五个批准标签。该记录只批准训练基础设施与首步一致性，不表示 loss 收敛或导航成功。
