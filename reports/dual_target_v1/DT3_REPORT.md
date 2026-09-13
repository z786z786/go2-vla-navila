# DT3 tiny 微调：250更新完成，待闭环验收

## 2026-09-06继续执行（覆盖下文暂停状态）

用户明确指示准备好后进行DT3，主代理已启动固定观察诊断及首轮CUDA闭环，训练更新仍为250。

CPU诊断`dt3_diagnostic_250_0906_v2`（32次推理，固定噪声20260906）：16个起始指令案例全都平均向右转，仅8个符号与目标方向一致；不同指令有数值差异，但未随颜色交换正确改变转向。16个真实末帧的前10步raw动作均至少有一步超过停车阈值。它们来自CPU BF16，需CUDA闭环确认，且单噪声不能说明所有策略种子行为。

首次诊断入口因缺少SmolVLA配置注册导入失败，尚未推理；修复后在新的v2输出目录完整完成，旧记录保留。训练和checkpoint不变。

闭环`dt3_eval250_0906_v1`已启动：16固定tiny任务，首策略种子20260906。模型worker只接收RGB/3D体速度/固定指令；原始50×3动作记录且每10步重规划；GT只用于评分和审计。物理钟在等待推理期间保持一致。CPU接口/块消费测试2/2通过，本地84项80通过4依赖跳过。GPU资源共享8 GiB准入及2 GiB运行保护，队列PID1931071，03:19 UTC首任务开始。结果仍待实测。

## 最新执行结果

DT2已正式APPROVED。随后按用户授权运行`dt3_tiny_pilot_0906_v1`，完成250次optimizer update（4000个有放回采样anchor），首次更新计入这250次，不另计warmup训练。状态为`PILOT_ARTIFACT_CHECKS_PASS_NOT_DT3_APPROVAL`；已停止自动扩档，尚未执行模型导航闭环，不把loss当导航成功。

实测micro-batch 1/2/4/8/16的热身后forward/backward吞吐分别约5.94/11.39/21.08/33.68/42.51样本每秒。选择micro-batch=16、accumulation=1、effective batch=16，恒定LR=1e-4、BF16 autocast＋FP32可训练主权重/Adam状态。训练首个事件至最终checkpoint完成约192秒（含保存/重载）；吞吐试测数不含完整数据准备及checkpoint开销。

实际缓存7392帧uint8 RGB，共5,813,305,344字节（5.41 GiB），4个CPU加载进程、主进程8线程。含Adam两份moment预留的torch峰值allocated=4112.12 MiB；外部采样自有进程峰值4530 MiB（4.42 GiB），最低实际free=12957 MiB（12.65 GiB）。全程无OOM或非有限数值。

首次更新后模型/processor/optimizer/RNG实际保存重载，固定观察/噪声预测差异为0。另写CPU验收脚本核对6份checkpoint所有文件哈希；250步优化器状态、4000个抽样索引及消费位置一致。冻结的345个张量与原始base精确相同，155个可训练张量发生更新；action_out_proj中被忽略的vy及内部padding输出行未改变，vx/wz行最大更新约0.003038。

最后一次记录loss=0.21083，末5次记录为0.34528/0.16806/0.09219/0.18428/0.21083。不同batch/noise有波动，不能用单点首尾或下降曲线代替任务成功验证。

最终checkpoint远端路径：`/mnt/wxh/go2_short_vln/outputs/dual_target_v1/dt3_tiny_pilot_0906_v1/checkpoint_000250/`。checkpoint manifest SHA256 `4c96dd6150c4a1b8a79ce59499ad7ce3264668246784967815535062f7ec0870`；training_config SHA256 `a90dbdd3903374ce202224db5e4dfbb4e5fe54d0d9defd8f004f694951747bfd`。小型证据镜像于`remote_runs/dt3_tiny_pilot_0906_v1/`，模型与optimizer大文件留远端。

训练及监督进程已退出，项目锁可获取；GPU只剩外部PID1866676，实际free回到17493 MiB。没有操作外部任务。累计DT3实际训练预算已用250/5000；下一步是固定指令/颜色响应诊断及tiny闭环，只有诊断支持才进入500等后续档位，不进入DT4。

2026-09-06最新用户授权：DT2共享采集且数据完整验收后，直接按服务器资源调整参数开始DT3。该授权不等同于DT2已通过，也不授权DT4。主代理单独实现和复核。

## 训练候选

- 只读取获批的完整16条LeRobot tiny数据；启动前校验DT2批准绑定及所有数据文件哈希。固定基础SmolVLA与合同哈希不变。
- 新的三维state/action归一化来自完整tiny train；视觉IDENTITY。训练输入只有front RGB、body velocity、instruction，GT只用于审计。
- 冻结视觉语言模型，只训练expert和state/action/time projections，实际CPU载入确认99,880,992个参数可训练。可训练主权重及Adam状态FP32、计算BF16 autocast；在严格加载权重之前转换参数dtype，避免checkpoint重载经过BF16舍入。
- 每个anchor只平均其有效vx/wz目标，再对anchor平均；固定vy和padding梯度为零。保持episode均衡和effective batch=16，micro-batch候选1/2/4/8/16，累积数为16/micro。
- 资源试测保留真实Adam两份moment buffer空间，进行真实forward/backward但不做optimizer update。选择实测吞吐距最优5%内显存更省的micro-batch，而非机械填满显存。
- CPU约64逻辑核、约489 GiB可用RAM；候选使用4个加载进程、8个主进程CPU线程，将实际无损loader的RGB缓存为uint8（显式16 GiB上限）。不使用全服务器CPU。
- 共享准入实际空闲≥8 GiB；本次allocator上限为准入free减3 GiB；外部watchdog实际free<2 GiB则只终止本次隔离进程组。训练与Isaac分开。
- AdamW LR=1e-4、betas=(.9,.95)、eps=1e-8、weight_decay=1e-10、clip=10。250更新pilot采用恒定LR，不继承基础配置的1000 warmup/30000 decay。有效batch和累计5000更新上限不变。
- 先完成第1次真实更新、保存与重载模型/processor/optimizer/RNG、固定观察和噪声对比，才继续250更新。1/50/100/150/200/250保存不覆盖的原子完成checkpoint；保存模型、处理器、optimizer、全RNG、完整抽样序列和消费位置、数据/归一化哈希。异常保留证据，不自动重试。
- 当前没有通用resume CLI；checkpoint状态齐备，pilot内第1步实际重载并续训至250已验证，不能声称任意中断可bitwise恢复。250后停止自动扩档，先做固定指令/颜色诊断和tiny闭环。后续使用该checkpoint时保持BF16 autocast及FP32可训练参数的加载顺序。

## 已有验证

远端`/tmp/dt3_candidates_0906_v3/cpu_preflight_result.json`：禁用CUDA，4项原始单元测试通过；真实两条数据的处理器三维归一化、padding、无损RGB、保存/重载输出相同；固定base严格载入，全部可训练权重FP32。加入批准绑定拒绝测试后，远端完整82项回归通过且无跳过。上述预检在训练前完成，没有改变采集运行快照。

## 当前状态

DT2 APPROVED；DT3已完成首轮250更新及训练产物验收，尚欠闭环行为验证，NOT_APPROVED。无运行中的本次GPU任务。用户的“补齐数据、调整训练参数并开始训练”请求已完成；本次没有自动扩大到500更新或进入DT4。
