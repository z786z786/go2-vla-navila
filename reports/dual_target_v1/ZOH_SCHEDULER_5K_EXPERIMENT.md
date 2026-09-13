# Warmup + cosine 单变量对照

2026-09-07 用户明确授权“直接补 scheduler 单变量对照进行训练”。本轮从同一 smolvla_base 新训练 5000 步，不从常数 LR 的 5k checkpoint 继续，不全量解冻、不改采样权重/数据/动作接口。

## 对照与唯一训练变量

对照组：`zoh_train_1000_0906_v1` 前1000步 + `zoh_train_5000_0906_v2` 接续至5000步，恒定 LR=1e-4。本组：`zoh_scheduler_5000_0907_v1`。

- 原数据 `zoh_dataset_0906_v2/dataset`：16条770帧；train-only归一化、完整5Hz ZOH动作标签、逐anchor有效vx/wz损失、末尾padding规则不变。
- 直接加载常数组完整 `sampled_indices.npy`，80000索引逐项相同，并核验文件hash与前16000索引一致。已知常数组hash `ec4477a68f13a41229d839658bc630c227426b1d6781f2201f688cd516c7138e`。
- 固定micro=4，accum=4，effective16；即使显存更空闲，也不扩大batch。profile只验证micro4可运行，不改变训练配置。
- seed=20260906，初始化base文件hash、参数冻结/数量/dtype逐项与常数组比对；BF16 autocast、FP32 trainable master/Adam，VLM/vision frozen，expert及投影trainable不变。
- AdamW betas(.9,.95)、eps1e-8、weight_decay1e-10、clip10，RGB512、state3/action3、chunk5/execute1均不变。
- 5个复用核心文件 zoh_train/zoh_train_core/tiny_train_core/smolvla_probe_client/contracts 与常数组初次supervisor receipt hash一致。

唯一优化改动：使用服务器现有 LeRobot `CosineDecayWithWarmupSchedulerConfig`；配置warmup1000/decay30000/peak1e-4/min2.5e-6，build(total=5000)按其实现自动缩放warmup为166、decay为5000，不自行换另一种cosine公式。源码hash记录在training_config。

第一次更新使用LR=1e-4/167≈5.988e-7，记录更新实际使用lr及更新后的lr_next；先optimizer.step再scheduler.step，scheduler.last_epoch必须等于完成更新数。warmup边界166/167额外记录；最后使用的LR略高于2.5e-6、最后scheduler.step后的LR为2.5e-6，避免一步偏移混淆。

## 验证与观测

- 真实服务器9/9测试通过：完整5000步LR形状、scheduler/Adam恢复后LR及参数轨迹一致、实际lr解析、chunk边界/损失mask、采样恢复。当地9项中6项依赖跳过，不能用本地跳过替代远端结果。
- 第1次更新前loss须与常数组第1步loss在1e-4绝对容差内一致，辅助检测初始化/采样/RNG异常。
- 第1步检查FP32投影确实更新且有限，并真实保存重载；预处理/预测必须一致，optimizer与scheduler游标/LR必须恢复。因warmup首步极小，不要求BF16推理结果一定出现可见变化，不将量化分辨率误认为无更新。
- 保存1/250/500/750/1000/2000/3000/4000/5000 checkpoint，包含scheduler/Adam/RNG/采样游标。相比常数组，wall time和保存安排不作为训练变量；不宣称跨GPU负载运行逐bit完全确定。
- 新TensorBoard桥 `zoh_scheduled_training_to_tensorboard.py` 读取每步实际lr，不再用配置峰值覆盖；scheduled日志缺实际lr直接报错。旧桥和旧曲线不改。
- 训练结束停止，待验收后再决定固定输入多噪声对照与配对闭环；不自动测试或开启下一次训练。

## 资源与产物

输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_scheduler_5000_0907_v1`；日志 `/tmp/go2_zoh_scheduler_5000_0907_v1_supervisor.log`。CPU supervisor入口 `zoh_train_scheduler5k_launch`；训练 `zoh_train_scheduler5k`，公共对照助手 `zoh_scheduler_core`。

准入7168MiB连续检查及锁复查；实际free floor2048MiB，4小时单次运行上限，只终止自有组。分配模型前短暂掉至准入以下在CPU侧等待。保持原门槛，不沿用仿真1GiB例外。

上一轮train布局结果已全完成：slot0/3成功，slot1/2错误目标停稳，均停到左侧。仅此一组布局/单seed，不称训练集总体成功率或可靠语言跟随。独立validation四条均超时，二者保留作后续同条件比较。

## 启动实测

父PID2113786，训练子PID/PGID2113985，已RUNNING；真实首步loss=0.8877298534并通过常数组首步loss一致性检查。首步模型/预处理重载差0，scheduler/Adam恢复gate通过。更新1实际lr=5.988e-7，更新10=5.988e-6，warmup生效。

TensorBoard桥PID2114430；原地址 `http://localhost:16007` 下独立run `zoh_scheduler_5000`，读取日志实际lr。旧run `zoh_train_1000` / `zoh_train_5000_resume` 保留。最新更新数以远端events.jsonl为准。
