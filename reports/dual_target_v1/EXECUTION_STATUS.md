# 双目标导航执行状态

## 最新：C全量10k从step 2000排队精确续训

2026-09-07：按用户决定不做显存预留，保持空闲13824 MiB连续3次/30秒准入与2048 MiB运行保护。v4因外部PID动态增占显存，在step2120触发保护停止；正式 `checkpoint_002000` 已通过模型文件、Adam、scheduler、RNG、采样游标和hash验证，2001–2120未保存更新不沿用。

独立续训 preflight 通过，从 `zoh_full_10000_0907_v4/checkpoint_002000` 恢复到总step10000，严格复用同一160000采样顺序；输出 `zoh_full_10000_resume_0907_v1`，后台队列PID **2411033**。当前free约7710 MiB，状态 `WAITING_GPU`，未加载模型、未占训练显存。达到13.5 GiB后自动启动，先用step2010 loss与中断运行做轨迹连续性复核，再保存4k/6k/8k/10k并最终reload；不读取validation、不执行eval/rollout。

## 最新：C全量微调fresh 10k已在13.5 GiB例外下真实运行

2026-09-07：用户明确授权将本次C全量微调准入从14336 MiB降为13824 MiB，运行时2048 MiB硬保护不变。原v2队列在0 step时安全停止。首次v3启动完成了1次更新，但一致性门错误地将full-policy首步loss与expert-only基线比较，因差约0.000255主动退出；GPU、数据和全量梯度无异常，v3证据保留且不续跑。

已改为scope-aware基线：完成的full-5k与两次fresh full启动首步loss均为0.8879846446；新v4首步与该基线差0。v4输出 `outputs/dual_target_v2/zoh_full_10000_0907_v4`，队列PID **2360823**、supervisor **2360981**、trainer **2362857**。全量梯度门通过（vision/text/lm_expert均有非零有限梯度），vision patch、text q_proj与action projection均实际更新；峰值allocated约7760 MiB，观测free约5777 MiB，高于2048 MiB保护。TensorBoard run `zoh_full_10000_v4` 实读仅有五个批准训练标签。训练/验证仍按trajectory隔离，本轮不读取validation/test、不运行rollout；仅保存2k/4k/6k/8k/10k，完成最终reload与深度hash后自动停止。

## 最新：C全量微调fresh 10k已排队等待14 GiB

2026-09-07：用户要求排队C全量微调10k，显存满足必要条件后自动启动。独立C-only队列已通过preflight：旧 `zoh_full_10000_0907_v1` 确认为 `INTERRUPTED_RESUMABLE` 且0 step；新输出 `zoh_full_10000_0907_v2` fresh；与B使用完全相同的160000采样顺序、train-only数据、scheduler、micro4/accum4、loss和seed，仅训练范围扩大为全部参数（含vision/VLM，vision non-reentrant checkpointing）。

队列PID **2331147**，supervisor PGID **2331245**，日志 `/tmp/go2_zoh_full_10000_queue_0907_v2.log`。当前状态 `WAITING_GPU`：需要空闲14336 MiB连续3次、间隔30秒并锁内复查；当前free约7529 MiB，尚未加载模型或执行更新。运行保护2048 MiB；checkpoint只保存2k/4k/6k/8k/10k并逐份验收。TensorBoard run为 `zoh_full_10000_v2`，训练开始后自动创建，只写五个训练指标。禁止validation loss和任何rollout；10k及最终reload/深度hash通过后停止等待用户。

## 最新：B-10k validation rollout 8/8完成并停止

2026-09-07：`b10k_eval_rollout_0907_v1` 已正常完成，状态 `B_10K_VALIDATION_ROLLOUT_COMPLETE_AWAITING_USER`；8条结果为4 success、3 failed_wrong_target_stop、1 timeout、0 collision。进入正确目标区域5/8，进入错误目标区域3/8，末距均值1.2253 m。逐条：slot16停错、17成功、18成功、19停错、20成功、21成功、22超时、23停错。全部轨迹审计、视频、结果和artifact hash已生成；队列、Isaac及模型worker均已退出。该结果是validation布局单seed闭环表现，不使用独立test，也未回流训练或checkpoint选择。

## 最新：B-10k开始8条validation rollout

2026-09-07：用户在B-10k的16条train rollout完成后，明确要求在eval场景运行。此处eval固定解释为已冻结validation slots16–23，共8条；不使用独立test，不训练，结果不回流checkpoint选择。沿用相同B `checkpoint_010000`、policy seed、控制/评分/视频规则，以及用户刚批准的同部署链7 GiB准入/1 GiB运行保护。

preflight已核验checkpoint全哈希、8个slot边界及test禁用。队列PID **2301345**，输出 `outputs/dual_target_v2/b10k_eval_rollout_0907_v1`，日志 `/tmp/go2_b10k_eval_rollout_0907_v1_queue.log`；slot16 Isaac PGID **2301523**、模型worker **2302167** 已RUNNING并产生真实动作。启动阶段自有峰值5826 MiB、最低free 2377 MiB，高于1024 MiB保护线。当前0/8表示首条未结束，完成8条后自动停止。

## 最新：用户授权7 GiB/1 GiB例外，B-10k train rollout已真实运行

2026-09-07：原9 GiB队列 PID 2264616 在 `WAITING_GPU`、0/16、无Isaac子进程时停止，证据保留为 `b10k_train_rollout_0907_v1`。用户明确要求降低门槛强行启动；独立v2使用7 GiB连续准入和1 GiB运行保护，只适用于本次B-10k train rollout，其余模型/任务/checkpoint/seed/评分均不变。

v2 preflight通过，队列 PID **2265826**，输出 `outputs/dual_target_v2/b10k_train_rollout_0907_v2`，日志 `/tmp/go2_b10k_train_rollout_0907_v2_floor1_queue.log`。slot0 Isaac PGID **2266001** 已RUNNING，模型worker **2266647** 已产生真实动作chunk；最新资源复核自有峰值5826 MiB、最低实际free 2070 MiB，高于授权的1024 MiB保护线。当前0/16表示第一条尚未结束，不能计作失败或成功。validation/test仍禁用，完成16条后停止，不恢复C。

## 最新：暂停0-step C等待，启动B-10k全部train rollout

2026-09-07：按用户新优先级，B fresh-10k 已完成并通过 checkpoint/reload 验收后，先做其全部16个train任务的真实闭环 rollout。原 B→C 训练流水线 PID 2232478 已精确停止；C 当时仅 `WAITING_GPU`、0 step、无模型分配，现保存为 `INTERRUPTED_RESUMABLE`，不删除其空启动目录。B-10k 模型和五个checkpoint不受影响。

独立 B-10k train-only preflight 通过：`checkpoint_010000` 全文件哈希一致，16个任务严格为slots0–15，policy seed=20260906，运行时明确拒绝validation/test。末尾不足0.2秒的partial hold按真实50 Hz帧数保留，不补成完整动作。新队列 PID **2264616**，根目录 `outputs/dual_target_v2/b10k_train_rollout_0907_v1`，日志 `/tmp/go2_b10k_train_rollout_0907_v1_queue.log`；当前slot0处于 `WAITING_GPU`。沿用同部署链实测自有峰值约5.84 GiB后的9 GiB准入/2 GiB运行保护；启动时空闲7942 MiB，不足9216 MiB，尚未启动Isaac或产生rollout。资源满足后自动执行16条并停止，不启动C、validation或test。

## 最新：取消 eval，B/C 从 base 重训 10k，仅保存训练证据

2026-09-07：用户取消 eval，并授权直接执行新的 B/C 串行训练。旧 `two_ckpt_train_0907_v2` rollout 队列 PID 2186795 已安全停止；`queue_status=INTERRUPTED`，已有视频/轨迹不删除、不作为新训练结果。新流水线不读取 validation/test、不计算 validation loss、不运行中间或最终 rollout，也不启动 Isaac。

B 为 scheduler＋expert/projection-only，C 为相同 scheduler＋全量微调；二者均从同一 `smolvla_base` fresh 初始化，使用同一已批准 train-only 770 帧数据、相同 160000 样本顺序、micro4/accum4（有效 batch16）、5 Hz/chunk5/execute1、loss 和 seed。官方 warmup/cosine 配置按 10k 构建，实际峰值更新为 step334；正式 checkpoint 只保存 2000/4000/6000/8000/10000，含模型、处理器、optimizer、scheduler、RNG、采样游标和逐文件哈希。TensorBoard 每组只写五个训练标量。

新实现 `zoh_train_10k.py`、`zoh_train_10k_launch.py`、`zoh_train_bc_10k_pipeline.py` 及无 pytest 依赖的 preflight 已部署；远端 preflight 通过，确认前 80000 样本与已审 B-5k 完全一致、LR 数值和 train-only import 门成立。新输出路径启动前均不存在，磁盘可用约 1.9 TiB。

唯一流水线 PID **2232478**，日志 `/tmp/go2_zoh_bc_10000_0907_v1_pipeline.log`，根目录 `outputs/dual_target_v2/zoh_bc_10000_0907_v1`；B supervisor **2232542**、trainer **2232705**、TensorBoard bridge **2232712**。B 已通过 step1 门：首 loss 与既有同起点完全一致（差0）、action projection 实际更新、固定预测发生变化，且没有早期 checkpoint；实测 TensorBoard event 恰好只有五项约定标签。状态已进入 `TRAINING_B_scheduler_expert`，完成前不可写成10k已完成。详见 `ZOH_BC_10K_TRAINING.md`。

## 最新：改为先完成B/C两个checkpoint的train rollout，eval等待用户决定

2026-09-07：按用户最新指令安全停止三checkpoint train＋eval矩阵；停止前A/slot0、B/slot0均success，证据保留但不计入新结果。新范围仅B scheduler expert-only与C scheduler full，完整重跑train slots0–15，共32条；完成后自动停止，不启动validation/test。

新代码21项远端回归及真实checkpoint矩阵绑定通过。`two_ckpt_train_0907_v1` 因交付父目录初始化错误在GPU前退出、0 rollout，已补嵌套目录测试并改用 `two_ckpt_train_0907_v2`。远端PID **2186795**，日志 `/tmp/go2_two_ckpt_train_0907_v1_queue.log`；本地同步PID **23016**，目标 `reports/dual_target_v1/remote_runs/two_ckpt_train_0907_v2/`。启动后SSH暂时连续超时，后台session不依赖SSH；恢复先查queue_status，勿重复启动。详见 `ZOH_TWO_CKPT_TRAIN_ROLLOUT.md`。

## 最新：A/B/C 三个 5k checkpoint 已齐，闭环对比从 evaluation_v2 恢复运行

2026-09-07：B scheduler-only 与 C scheduler＋全量微调均完成5000步。C首步门通过：vision 197、text 139、lm_expert 145个参数张量有非零有限梯度；视觉patch embedding、首层文本q_proj、action_out_proj均实际更新；checkpoint重载预测差0。C训练峰值分配约7735MiB，末步loss 0.004069；B末步loss 0.008659。单个末步loss仅为采样噪声，不提前排序checkpoint。

首次对比 `evaluation/` 在0/72、尚未产生动作时因fresh Python进程先解析config、后导入SmolVLA注册模块而停止，证据完整保留，未计作模型失败。已修复统一loader导入顺序；远端19项回归、A/C两个真实checkpoint CPU fresh-process加载及GPU worker启动均通过。

恢复父PID **2181558**，队列PID **2181599**，新目录 `evaluation_v2/`；当前第一条A/train/slot0已RUNNING，模型worker在线，已写入连续模型chunk及50Hz物理轨迹。9GiB准入通过，运行时尚余约11.6GiB。结果仍为0/72，第一条尚未结束。第一次失败目录不删、不复用其空结果；后续72条全部在v2重新执行。本地同步PID **19513**，日志 `reports/dual_target_v1/three_ckpt_sync_v2.log`。

## 最新：三个5k checkpoint自动对照队列已启动

2026-09-07 用户授权 B scheduler 完成后从base训练 C全量微调至5000，再自动对 A/B/C 在全部16个train＋8个validation任务逐条闭环比较（72条）。本流水线覆盖此前“训练完成停下”的限制，不使用独立test、不混入新数据、不再加训。父PID **2121054**，状态 `WAITING_SCHEDULER_TRAINING`，输出 `outputs/dual_target_v2/three_ckpt_compare_0907_v1`，日志 `/tmp/go2_three_ckpt_compare_0907_v1_pipeline.log`。

全量450,046,176参数CPU真实加载/FP32scope/视觉重计算检查、A checkpoint全部hash核验和19项回归已通过；全量GPU首步gate将在B退出后执行。C micro4/accum4与B scheduler相同，仅扩大scope（含全量FP32master及视觉激活重计算），14GiB准入/2GiB保护/8h。闭环三组统一9GiB准入/2GiB保护；普通任务失败继续，基础设施异常停并保留证据。末尾partial hold视频实测正确。

本地自动同步PID **69972**，目标 `reports/dual_target_v1/remote_runs/three_ckpt_compare_0907_v1/`；报告/CSV/全72条视频及训练证据完成后自动下载并hash校验。详见 `ZOH_THREE_CHECKPOINT_COMPARISON.md`。仅队列就绪，不代表比较已经完成；恢复先查流水线状态，勿重复启动或修改冻结源。

## 最新：用户授权 scheduler 单变量对照，fresh base 5000步已启动排队

2026-09-07：父 PID **2113786**，入口 `src.dual_target.zoh_train_scheduler5k_launch`，输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_scheduler_5000_0907_v1`，日志 `/tmp/go2_zoh_scheduler_5000_0907_v1_supervisor.log`。从同一base重训5000，唯一优化变量为官方warmup+cosine；实际warmup166、衰减跨度5000，名义peak1e-4/final2.5e-6。固定micro4/accum4，复用常数组完整80000采样索引，不改数据/归一化/损失/冻结范围。远端9测试通过，5个共享核心源码hash与常数组一致。门槛7GiB/运行free2GiB/4h不变，完成后停止，不自动评估或再训。

上一轮训练布局4条完成：0/3成功、1/2错误目标停稳，均到左侧；验证布局4条全超时，均保留。新独立代码与正式记录见 `ZOH_SCHEDULER_5K_EXPERIMENT.md`。实际训练状态和首步gate以新supervisor_status/events/single_update_gate为准，不重复启动。

已实际 RUNNING：子 PID/PGID **2113985**；首步 loss=0.8877298534，通过与常数组初始loss一致性检查，模型/预处理/scheduler保存重载gate通过、预测差0。已记录更新1的实际lr=5.988e-7、更新10=5.988e-6。TensorBoard桥PID **2114430**，复用 `http://localhost:16007`，新run `zoh_scheduler_5000`；不覆盖常数LR曲线。

## 最新：用户要求训练布局 rollout，四条拟合检查已排队

固定 checkpoint5000，在 train slots0..3（dt1_dev_000，训练视频01–04）实时模型控制，非专家回放、非训练更新、非独立验证。入口 scripts/zoh_train_scene_queue.py / zoh_train_scene_runtime.py；不改当前验证代码。等待验证 PID2089283 完成最后一条后串行 GPU 准入；输出 `outputs/dual_target_v2/zoh_train_scene_5k_0906_v1`，日志 `/tmp/go2_zoh_train_scene_5k_0906_v1_queue.log`。7GiB准入/1GiB运行保护不变，完成四条后停。详见 ZOH_TRAIN_SCENE_ROLLOUT.md；实际状态以新 queue_status.json 为准。

## 最新：5k 训练验收通过，启动最终 checkpoint 四条配对验证

用户授权下一步。训练 `zoh_train_5000_0906_v2` 正常完成 5000；五 checkpoint 全文件 hash、数据/预处理绑定通过；末 loss=0.04304、末10记录均值=0.07374。固定噪声训练起点对照方向匹配 8/16，换红蓝指令首 wz 平均变化仅0.00308，尚不能说语言跟随改善。

当前验证父 PID **2089283**，输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_val_5k_pair_0906_v1`，日志 `/tmp/go2_zoh_val_5k_pair_0906_v1_queue.log`，入口 `zoh_eval_5k_pair_queue`。仅 checkpoint5000、validation slots16..19四条；不运行旧矩阵、不测试/训练。7GiB准入、1GiB运行保护，普通任务失败继续配对、资源或基础设施失败停。完成四条后停下验收，等用户决定是否扩展。详见 `ZOH_5K_NEXT_VALIDATION.md`。队列已启动，实际运行/排队以其 queue_status.json 为准，勿重复启动或改冻结源码。

## 最新：停止闭环评估，checkpoint 1000 审计后续训至累计 5000

用户要求停止后续 rollout。已有 3 条均 checkpoint 1000、超时无碰撞；第 4 条显存保护中断，旧队列 2054019 已退出，不再续跑。

独立数据/推理审计：数据与 checkpoint 哈希、实际预处理一致性通过；三条首 chunk 固定噪声重放误差均 0。专家起步左/右各 8，但模型固定噪声在 16 个训练起点全预测左转，换红/蓝指令的首动作 wz 平均差仅 0.00288。存在语言利用不足，不是仅停车阈值问题；数据仅 4 布局/770 帧有捷径/过拟合风险。详见 `ZOH_5K_DIAGNOSIS_AND_TRAINING.md`。

用户最新 5k 授权解释为从当前 1000 续至总 5000、新增 4000；保留 Adam/RNG/采样游标，LR 1e-4/effective16/5Hz/chunk5/execute1 不变，micro 重新 profile。源 `zoh_train_resume5k.py` 与独立 launcher；远端 16 测试通过。

当前排队/训练父 PID **2070600**，输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_train_5000_0906_v2`，日志 `/tmp/go2_zoh_train_5000_0906_v2_supervisor.log`。7 GiB 准入、2 GiB 运行保护；每 1k checkpoint，首新步 1001 重载 gate。完成 5000 后停下，不自动评估。首 v1 在缓存后 free<7GiB、模型分配前退出，0 新增更新，证据保留；v2 在此阶段可 CPU 等待，不降低门槛。实际状态以 v2 supervisor_status.json 为准，不重复启动。

续训实际启动已确认：子 PID/PGID **2070624**，micro=4、accum=4，第 1001 步 loss=0.0968426，真实保存重载 gate 通过、预测误差 0。TensorBoard bridge PID **2071474**，复用原 `http://localhost:16007`，新 run `zoh_train_5000_resume`（原 1k 曲线不改）。

## 最新：超时结果兼容修复，保留首条失败并从第2条续跑

首条真实1500步超时未缺轨迹；原timeout结果漏steps导致汇总KeyError。独立兼容层已按完整证据派生步数且不改旧结果，新循环所有正常退出写实际步数。本地/远端11项测试通过，真实首条审计通过结构一致性（任务仍FAILED_TIMEOUT、未进入停车区、末距0.45503m）。见 `ZOH_TIMEOUT_RESULT_FIX.md`。

续跑PID2054019，输出 `outputs/dual_target_v2/zoh_val_resume_0906_v1`，父日志 `/tmp/go2_zoh_val_resume_0906_v1_queue.log`；第1条从 `zoh_val_scored_0906_v1` 引用并记失败，不重跑，从slot17/checkpoint1000起继续其余31条。用户明确批准续跑，取消“必须首条导航成功才继续”的临时试验门槛，保留7GiB准入/1GiB保护。任务失败记账继续，基础设施异常仍停。不追加训练、不做test模型评估。恢复查新队列，勿改冻结源码或重复启动。

## 最新：评分映射兼容修复通过回归，验证已重新启动

详见 `ZOH_SCORING_FIX.md`。旧静态clip校验与部署slew冲突已用独立ZOH动作审计+新版评分frame修复，原停车/碰撞标准不改。8项本地/远端测试通过；受影响593步回放INVALID_SAMPLE从60降至0，32条专家仍全部成功。原2045780队列及子进程已停止，证据保留。新PID2048028，输出 `outputs/dual_target_v2/zoh_val_scored_0906_v1`，父日志 `/tmp/go2_zoh_val_scored_0906_v1_queue.log`，仍7GiB/1GiB资源策略、首条通过才续矩阵。回放通过非模型导航批准；恢复先查新队列，勿重启旧版本或修改冻结源码。

## 最新：用户明确授权运行保护降至1 GiB，重新尝试闭环

上次7GiB准入/2GiB保护试运行约35秒时free1953MiB触发保护，0/32；父2044453、Isaac2044615、模型2045252均退出，无模型任务结果。原代码/目录保留。用户随后授权降低显存保护并开始：独立 `zoh_eval_probe7_floor1_queue.py` / `zoh_closed_loop_probe7_floor1.py`，PID2045780，输出 `outputs/dual_target_v2/zoh_val_probe7_floor1_0906_v1`，父日志 `/tmp/go2_zoh_val_probe7_floor1_0906_v1_queue.log`。准入7168MiB连续3次/30秒不变，运行保护改1024MiB，单条1800秒限制不变；OOM风险已说明。先重试1000步checkpoint/validation slot16，任务成功且独立审计和资源合格才继续其余31条；否则停，不自动重试，不改任务判分，不加训或测试集评估。新源码全量hash冻结，队列运行/等待期间勿修改。

## 最新：用户授权7 GiB单次闭环资源试运行，成功才降低本矩阵准入

旧9GiB队列PID2044002在WAITING_GPU、0/32、无子进程时停止并确认退出，目录 `zoh_val_eval_0906_v1` 保留。独立新队列PID2044453，输出 `outputs/dual_target_v2/zoh_val_probe7_0906_v1`，父日志 `/tmp/go2_zoh_val_probe7_0906_v1_queue.log`；代码 `zoh_closed_loop_probe7.py` / `zoh_eval_probe7_queue.py`，旧代码不改。准入7168MiB、连续3次/30秒，运行2048MiB保护不变。先1000步checkpoint的validation slot16：仅任务成功、独立轨迹审计通过、资源采样合格后写 `resource_probe_acceptance.json`，继续同一32任务矩阵；首条任务失败或资源/运行异常即停，不自动重试，不推广为通用GPU规则。真实结果尚待队列，不将试运行授权写成低门槛已经验证。

## 最新：1000步训练产物验收通过，验证集闭环矩阵已排队

用户授权继续验收与闭环。全部五个checkpoint完整性/接口检查通过，单次更新重载门槛通过，训练已正常退出；不等于导航成功。独立新版chunk5/execute1部署适配完成，远端6项协议/训练测试通过。验证队列PID2044002，输出 `outputs/dual_target_v2/zoh_val_eval_0906_v1`；1000/750/500/250各8条validation任务，共32条，test禁用，无追加训练。9GiB连续准入、2GiB运行保护；冻结源码勿改。最新实际进度须查queue_status，尚无闭环成功批准。详见 `ZOH_V2_VALIDATION_EXECUTION.md`。

## 最新：用户授权直接启动新版训练，独立首阶段 1000 更新队列已启动

见 `ZOH_V2_TRAINING_EXECUTION.md`。新版训练入口已适配批准的 train-only 770 帧、5 Hz/chunk5/execute1；重新加载基础模型，保留 LR1e-4、有效batch16。远端9项训练测试全部通过。监督 PID2039433，输出 `outputs/dual_target_v2/zoh_train_1000_0906_v1`；实际是否开始更新须读取该目录状态/事件，勿凭队列启动宣称训练已更新。7 GiB准入、2 GiB保护；1000步后停下验收，无测试集自动评估或预算扩展。采集速度优化暂缓，候选不接入本次训练。

## 最新：32 条数据集主代理验收通过；提速尚未整体完成

32/32 自动采集/转换完成，父进程 1975859 已退出。主代理额外完成全部轨迹重放、末尾 hold、16 组 reset 配对、train-only normalizer 独立复算、17 个导出文件哈希核验，并实际查看全部 32 条的 160 个图像面板。正式批准见 `ZOH_V2_DATASET_FINAL_REVIEW.md` 与 `zoh_v2_dataset_approval.json`。train 770 帧、validation 412 帧，test 430 个高层观测仅存审计数据；无训练任务。

提速前期实现：独立 `fast_rgb.py` 有界异步 PNG 写入，保持像素和全部物理/相机时序不变；远端 4 项测试通过。770 张训练图像三组交错 CPU 基准，同步中位 2.629 s、异步 1.275 s，I/O 2.062 倍，全部像素一致；不代表仿真整体 2 倍，单条预计仅省约 1 秒，不作为主要提速验收。候选未接入正式采集，未做 Isaac 端到端验证，不批准为生产采集器。

高收益优化仍待完成：旧评分器要求 observation_seq 每个低层步严格递增，旧独立审计还要求 camera/render 每步递增。因此 5 Hz RGB 必须用新版本显式拆开 50 Hz 状态采样与图像帧证据，不能直接降低频率来规避旧检查。还需检查低层 depth 观测依赖、自动渲染与额外显式渲染，验证常驻环境的重置隔离与持续租约。保留用户已授权的优化计划；按每个里程碑通过后停止规则，本轮正式完成数据集里程碑，不启动训练。下次继续从采集提速开始，不重复验收或重采本批。

## 最新：用户授权完成本批后优化采集速度

2026-09-06 最新检查：22/32 完成自动审计，第 23 条运行中，父 PID 1975859 存活，无新增错误。用户明确要求先完成剩余采集，再优化采集速度。保持当前冻结源码、32 槽计划和证据不变；当前队列只会自动采集、转换并退出等待验收，不会自行启动优化或训练。后续主代理完成本批转换/图像/轨迹验收后执行独立版本速度优化，无需再次询问是否开展优化；如出现重要决策再询问用户。优化验收后停下，不启动训练。

优化优先级：先测量启动/仿真/显式渲染/PNG 写入/审计/资源等待分项耗时；将非必要 RGB 采集压至真实 5 Hz 决策点（保留必要的末帧及失败图像），50 Hz 状态、施加命令、碰撞、停车及真实动作段持续时间证据不减；验证 Isaac 常驻多 episode 重置及控制器状态清理，避免每条冷启动；常驻期间保持资源租约和连续显存监控，避免重复首次准入等待，释放/重建进程后仍须重新准入。不得简单删除安全检查或沿用过期准入凭据。保持 200/50/5 Hz、共享限幅/变化率、split、seed、chunk 5 和停稳标准不变；新审计格式独立版本，不覆写旧证据。用训练布局的少量固定配对任务验证新鲜 RGB 时序、重置隔离、边缘目标可见性、末尾完整段/partial 排除及自主停稳，报告同资源条件下实际提速与峰值显存；不能用图片数减少比例冒充实测提速。验证/测试集不得被用来调优化参数。

## 最新：授权资源中断恢复，保留前三条并重试第 4 条

2026-09-06：7 GiB 队列完成 3/32 后，slot 3 因 free=1955 MiB <2048 MiB 停止；本任务占用 4348 MiB，其他任务显存增长。原 PID 1967826 和子进程 1973266 已退出。用户批准保留前三条及中断证据，重试该资源中断并继续。独立 `zoh_dataset_resume.py` 启动 PID 1975859，日志 `/tmp/go2_zoh_dataset_0906_v2_resume01.log`，输出仍为 `zoh_dataset_0906_v2`。恢复前重审前三条；原失败状态、slot 3 日志/GPU 记录/运行目录移至同根 `resource_retry_slot03_attempt0/`，附原文件哈希、原计划哈希及恢复源码哈希。固定计划与五个冻结源码不改，未删除证据；仅本次重试获准，再次失败仍停。准入 7168 MiB、运行保护 2048 MiB 不变；完成后等待主代理验收，不训练。

## 最新：用户授权准入降至 7 GiB，重新启动尚无轨迹的采集队列

2026-09-06：旧队列 1966059 在 0/32、无子进程的等待状态下停止，原证据保留。新队列 PID 1967826，输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_dataset_0906_v2/`，准入 7168 MiB、连续 3 次/间隔 30 秒、运行余量 2048 MiB；只修改资源策略，冻结新清单，控制/数据/验收标准不变。9 项数据集测试重跑通过。完成采集与 CPU 转换后仍停在主代理验收点，不训练。恢复以该 v2 状态为准，不重复启动。

## 最新：用户批准执行 4/2/2 新数据集计划，队列已启动

2026-09-06 已新增 ZOH 数据集采集/转换代码，旧 gate 源码不变。完整真实终端 hold、partial 标签排除、逐高层帧边缘可见性审计、train/validation/test 隔离均已实现；本地 101 项测试 97 通过/4 依赖跳过。远端计划与运行入口预检通过。

新队列 PID 1966059，输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_dataset_0906_v1/`；固定 32 条新专家任务，失败即停，全部通过后 CPU 转换并停在主代理验收点，不训练。启动前空闲 7424 MiB，不足 8 GiB 时按原门槛等待。详见 `ZOH_V2_DATASET_EXECUTION.md`。恢复先查队列，不重复启动；数据集尚未批准。

## 最新：ZOH v2 物理前置验证已由主代理最终批准，停止等待下一阶段指令

2026-09-06 完成双端轨迹复算、源文件/证据哈希检查、260 个高层图像解码与 20 面板实际视觉抽查。一个探针及四组合专家任务全部通过；专家连续实际停稳约 1.04 秒、停车中心误差 4.8–5.4 cm。详见 `ZOH_V2_FINAL_REVIEW.md` 与 `zoh_v2_gate_approval.json`。

确认自有 ZOH 进程全部退出、项目锁可用，未启动正式数据采集或训练。批准范围仅物理前置 gate，不是 SmolVLA 成功或完整 DT3 通过。非阻断注意：右侧目标中途接近画面边缘；四条终止 hold 不完整，后续转换必须明确屏蔽/持续时间规则。旧队列状态与旧合同保持原始证据不改写。

## 最新：旧评估已按用户授权停止，新版开始 GPU 准入

后续确认新版已 `RUNNING`，probe/slot 0，子进程 PGID 1955423；尚无完成/验收结果。

用户明确要求停止旧评估、优先新版。旧 PID 1931071 与当前子进程 PGID 1953016 已核对身份并终止、确认退出；证据全部保留。旧 7 条完成，slot 7 中断，未完成 16 条矩阵。新版 PID 1954210 存活，状态 `WAITING_GPU`、probe/slot 0，GPU free 8242 MiB；按连续 3×30 秒与锁复查启动，不重开队列、不训练。详见 `ZOH_V2_EXECUTION.md`。

## 2026-09-06 用户授权执行 5 Hz ZOH v2；前置队列已启动等待旧评估

详见 `ZOH_V2_EXECUTION.md`。独立新增 zoh 模块和共享命令约束，默认 chunk 5 / execute 1；物理 200 Hz、低层 50 Hz 不变。候选变化率尚未物理批准。新增 8 项测试本地/远端通过，本地全 92 项 88 通过/4 依赖跳过，远端旧绑定保持一致。

新队列 PID 1954210，输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_gate_0906_v1/`；已确认 `WAITING_LEGACY_QUEUE`，先等待旧 PID 1931071（当前 slot 7）再按共享 8 GiB门槛准入。尚未收到停止旧评估的确认，未停止旧任务。新队列固定一个 14 秒运动/停车探针和四个 ZOH 专家任务，失败即停，全部检查通过后也停在主代理验收前；不采正式数据、不训练。

恢复先检查两队列/进程，禁止重复启动。新物理验证、正式 split 清单、新数据转换和新版模型训练均未验收，不得将 CPU 测试通过误写为 v2 或 DT3 完成。

## 2026-09-06 数据分布与多种子诊断完成；DT3 仍未通过

主代理完成只读数据分析与 96 次 CPU 多噪声推理，见 `DT3_DATA_DISTRIBUTION_REVIEW.md`。实际 4,000 抽样红/蓝 2015/1985，无明显任务采样失衡；动作右转帧约为左转两倍，近零目标名义损失权重 23.70%。没有证据需要推倒重采。

250 步模型在噪声 20260906 / 20260907 / 20260908 下起始平均动作分别全部右 / 全部左 / 全部右；24 对同图换指令均未改变转向符号。不能概括为模型恒定右转，应表述为本对照中噪声敏感而指令选择不可靠。48 个终点观测-噪声组合均无前 10 动作全满足 raw 停止阈值，不替代物理闭环。

最新远端队列 `dt3_eval250_0906_v1` 仍运行，已完成槽 0/1/2（超时/碰撞/超时），槽 3、PGID 1941559 在跑。不要重复启动。累计训练仍 250/5000，未改数据/训练参数/在跑源码，不进入 DT4。下方首槽运行记录为历史快照。

## 2026-09-06继续DT3：CPU诊断完成，CUDA闭环首任务运行中

用户明确要求准备好后进行DT3。主代理从250步checkpoint继续，尚未追加训练。CPU诊断`dt3_diagnostic_250_0906_v2`：固定噪声20260906，8个实际首帧各测试红/蓝两条指令，共16例；全部平均初始wz为负，仅8/16与目标侧相符，说明该固定噪声下存在右转偏置。16个专家末帧均未满足前10个raw动作全部落在停车阈值内。CPU诊断不替代CUDA或闭环结论。

新闭环批次`dt3_eval250_0906_v1`，队列PID1931071，首slot0隔离PGID1931134，03:19 UTC已获共享8 GiB门槛准入并启动。运行余量保护2 GiB；预声明16任务、种子20260906、50步chunk执行10步。原始模型3D动作保留，部署边界/停车评分沿用合同。任务失败正常留存继续矩阵，运行错误停止，不重试。恢复先检查queue_status.json和当前进程，不重复启动。

闭环代码为新文件，DT1/DT2 runner与合同未修改；新循环从已审查runner物理记录/评分派生，将专家命令替换为仅RGB、body velocity、instruction的管道模型调用，并校验推理等待期间物理钟不前进。专用CPU测试2/2本地远端均通过，本地全84项80通过/4依赖跳过。新文件采集时冻结。DT3 NOT_APPROVED；通过后停下，不进入DT4。

## 最新：DT2 APPROVED；DT3首轮250更新完成并停止扩档

本节覆盖下方全部旧状态。按用户最新授权已完成数据补齐、主代理DT2验收、资源参数实测与首轮训练。DT3 run=`dt3_tiny_pilot_0906_v1`，micro/effective batch=16、accumulation=1、LR=1e-4，BF16计算＋FP32可训练权重/Adam。250更新及6份checkpoint已完成并经CPU另脚本复核：冻结权重不变、忽略输出行不变、optimizer步数和采样位置正确，首步真实保存重载预测差异0。

峰值自有显存4530 MiB、最低实际free12957 MiB；进程已退出，锁可获取，GPU仅剩外部PID1866676。DT2最终报告DT2_FINAL_REPORT.md，DT3实际配置/结果DT3_REPORT.md。训练证据与小型checkpoint清单在remote_runs/dt3_tiny_pilot_0906_v1/；大权重/optimizer仅远端。累计预算250/5000，未自动进入500档或DT4。

DT3 NOT_APPROVED：接下来必须做固定观察/噪声的指令与颜色诊断、tiny闭环评估（再按计划扩到48次），不能用loss或训练文件验收代替模型导航成功。没有运行中的本次训练或采集队列。恢复前仍先读远端result/supervisor_status/events，避免重复启动。

## 最新：DT2 APPROVED；DT3资源试测/训练获准启动

2026-09-06主代理完成DT2验收：4组16条、7392真实训练帧，新增8/8一次成功；原始轨迹、8对复位、7408图像、转换器144＋另脚本144个loader anchor、fresh train normalizer和采样权重全部通过。最终报告DT2_FINAL_REPORT.md，批准dt2_approval.json。采集父进程及8个PGID均退出、锁可获取，外部PID1866676保留。训练代码采集完成后部署，远端82项CPU回归全通过，无跳过。

按最新用户授权直接进入DT3，不再等待下一条里程碑指令。候选配置及预检见DT3_REPORT.md，实际训练参数须GPU试测。DT3尚未批准，不进入DT4；以下全部进行中/排队记录为历史。

## 最新：DT2共享采集中，数据验收后获准直接启动DT3

2026-09-06用户明确授权DT2共享及更低准入，并要求数据补齐后按服务器资源调整参数开始训练。本节覆盖下文旧v1独占排队及DT2后暂停边界。旧PID1873188已取消且无episode启动，记录保留；新批次`dt2_tiny_0906_v2`父进程1873855，实际空闲≥8 GiB准入、运行余量≥2 GiB。18:42 UTC新增slot8成功，slot9等待，完整DT2尚未批准。77项远端回归76通过/1依赖跳过。源代码在采集期间冻结，DT3仅在本地准备，不提前训练。

恢复先查v2 collection_events/result和当前进程，禁止重复启动。DT2全部16条数据和主代理复核通过后，先训练资源实测及单batch更新/重载检查，再进行250更新短预算；effective batch=16、初始LR=1e-4、累计最多5000，后续扩档需闭环诊断，不授权DT4。

## 当前阶段：DT2进行中，默认独占GPU排队；DT3未开始

2026-09-06用户明确授权“进行DT2”。已冻结4组16条tiny计划，复用DT1 repeat=0的8条并全部按DT2条件重审通过，新增两组8条尚待采集。2条/954帧真实LeRobot转换试跑和18个loader anchor回查通过；它是接口试跑，不是完整DT2数据。76项回归本地74通过/2依赖跳过、远端75通过/1依赖跳过。详见DT2_REPORT.md。

唯一DT2父进程/等待器PID1873188、统一exec会话9310，批次`dt2_tiny_0906_v1`，当前slot8，默认20 GiB且无外部计算。当前共享授权问题已提出但未收到确认；不能将DT1单次共享例外自动扩展到DT2。恢复先读batch的collection_events/result与输出根dt1_gpu_wait_state，不重复启动。资源服务字段stage=DT1是旧实现格式，真正任务阶段见DT2 collection_receipt。

DT1保持历史APPROVED，其合同SHA不变。DT2只对runner作两行显式场景参数扩展，兼容性哈希记录在dt2_source_compatibility.json，其余锁定源文件不改。DT2 NOT_APPROVED，DT3 NOT_STARTED；DT2完成执行及主代理验收后停止。

## 最终状态：DT1 APPROVED；已停止，等待用户下一个里程碑指令

2026-09-06 主代理完成DT1执行及验收。本节覆盖以下全部进行中/排队/否决历史状态。16/16固定专家任务、8对任务等价复位、7444张图像审计、运动校准、红蓝接触正负例、暂停物理钟检查及真实SmolVLA共存均通过。合同锁`config/dual_target_v1/task_contract.lock.json`已生成；总报告`DT1_REPORT.md`，批准和证据哈希见`dt1_approval.json`、`dt1_final/gate.json`。

共存run `dt1_coexist_shared_0906_v2` 在用户授权的共享条件下通过：实际准入空闲17513 MiB，外部PID1866676未操作；本次采样峰值5618 MiB，最低实际空闲11850 MiB，有限[1,50,3]输出，单次推理886 ms。无模型动作执行或训练。71项最终回归本地69通过/2依赖跳过，远端70通过/1依赖跳过，SmolVLA专用处理器实测已通过。

旧等待器1867707、共享等待器1869427、隔离PGID1869449、Isaac1869451、模型1870651均已退出；实际输出目录`.dt1_gpu_wait.lock`可获取。没有运行中的DT1队列或本次GPU进程。DT0/DT1 APPROVED；DT2–DT7 NOT_STARTED，下一阶段必须用户明确指令。目标在少量中途画面仅剩边缘窄条的限制已记录；DT1通过不是模型导航正确性结论。

## 2026-09-06 当前结论：16/16 专家矩阵通过，共存排队中

本节覆盖以下历史状态。`dt1_m_0906_v2` 已完成预先指定的16项，无逐项重试；主代理对全部原始轨迹和8对复位另脚本重算通过，连续停稳均为1.0199999772 s。16条共7444张实际观察图像的离线检查通过：初始两目标可见、颜色交换正确、所选目标全程可见。远端 `dt1_m_0906_v2_visual/` 保存16个前视RGB＋审计专用俯视轨迹重建视频；右面板不是第二台Isaac相机，也不是策略输入。

共存探针新增7文件已在矩阵结束后部署，未改变矩阵的运行快照；5项专用CPU测试通过。唯一 run `dt1_coexist_0906_v1` 正按默认独占门槛等待GPU，外部用户作业不干预。本地完整69项测试：67通过，2项缺少远端依赖而跳过；相关真实OpenCV和SmolVLA处理器检查此前分别在远端通过。

DT1仍为NOT_APPROVED：尚需真实GPU共存结果、合同锁与最终验收。DT2 NOT_STARTED。仅主代理执行和复核，无Terra/独立代理复核；DT1完成验收后停止等待用户指令。这些专家成功不是SmolVLA导航成功。报告见DT1_REPORT.md。

最新远端部署后回归：Isaac环境68通过/1依赖跳过，跳过的SmolVLA处理器检查在专用环境单独通过，均禁用CUDA。人工抽查发现蓝目标中途最少可见帧仅剩左缘窄条；已在报告记录该可见性余量限制，不声称完整或始终清晰居中。

恢复入口：唯一等待器PID1867707，run_id=`dt1_coexist_0906_v1`；统一exec会话99138。它在GPU条件满足后自动衔接一次`dual_target_dt1_coexistence.sh`，不重复启动。结果位于远端输出根同名目录，launcher日志/receipt在根目录。恢复时先读`dt1_gpu_wait_state.json`和结果文件；若已有结果，用`dt1_coexistence_review.py`复算，并完成合同锁和总验收。不要将等待器的无变化当成代码失败。

## 2026-09-06 最新进度（以下历史暂停状态已被本节覆盖）

用户批准进一步简化并继续。配对检查已改为 task_equivalence_v2，2 cm/3°根位姿可比、每次正确清空history、静止起步；不要求关节值或RGB跨运行完全相同，仍验证各自文件哈希和真实图像。导航成功评分不改。取消等待器退出码修复已部署。

远端63项CPU测试实际通过（含真实OpenCV）；本地66项中65通过、1项缺OpenCV跳过。蓝探针 dt1_blue_20260906_v2 已通过：467步、重算连续停稳1.0199999772 s，无碰撞/外部停车；与红探针r4的新配对检查通过，仅为开发探针，不替代整批矩阵。

当前启动固定16任务矩阵 dt1_m_0906_v2，源码冻结，任何失败停止，不重试刷结果。部署备份在远端 dt1_equivalence_deploy_20260906/before.tar.gz；补丁归档SHA256 0c1c8348cc7fd4b5df8b0ab14c5fa79541b56733b9ba8b1c6470cc8c998622a0。SmolVLA共存、任务合同锁、整批验收尚未完成，DT1 NOT_APPROVED；DT2 NOT_STARTED。主代理单独负责，不安排Terra。

模式（最新用户指令，取代下文历史分工）：主代理单独负责 DT1 实现、远端调试、测试及验收；用户明确不需要 Terra xhigh 复核，不再派发子代理。验收使用原始证据与独立重算脚本，但属于同一代理复核，不称为独立代理验收。每个里程碑通过后停下，不自动进入 DT2。

当前恢复点（覆盖下文历史进度）：图像目录错误已由远端真实 OpenCV 旧版复现/修复版回读验证解决。r3 已完成完整导航记录，但末端专家在 .12 rad 朝向边界反复切换而超时；加入专家保持滞回后，r4 红任务 487 步成功，原始轨迹重算连续停稳 1.0199999772 s，487 张输入 RGB 均可见红目标。评分条件不变，远端 58 项 CPU 测试通过。

当前重要决策：同种子 r3/r4 的 reset 及预热后物理状态完全一致，但 reset RGB 哈希、首次动作 RGB 哈希不同；首次动作图像每通道平均差 0.017368/255、最大 5/255，reset 图像平均差 0.160034/255、最大 42/255。具体渲染差异源尚未定位。现有逐字节相同的配对图像检查不能通过，不自行放宽。蓝任务 dt1_pair_blue_20260905_r4 因外部 GPU 作业尚未启动，已请求取消该等待器，暂停等待用户决定图像等价验收口径。16 矩阵启动器仅本地准备，尚未部署或启动；模型共存、合同锁亦未完成。DT1 NOT_APPROVED，DT2 NOT_STARTED。

取消确认：15:54:51 UTC 等待器已退出，owned PID1848215/1848216 消失，仅外部 cc 的 GPU 作业仍在。发现 waiter 取消状态却返回退出码0，shell 因而尝试下一条 launcher；runner 的准入检查在 AppLauncher 前拒绝，留下蓝 run 的 FAILED/GpuWaitError 记录，未创建仿真或分配模型 GPU。此 CLI 退出码问题已用真实 subprocess 测试复现，并在本地修为取消退出130；64项本地测试中63通过、1项真实OpenCV测试因本地缺依赖跳过。gpu_wait.py 退出码修复和对应测试尚未部署。不可将蓝 run 的入口失败计为专家导航失败。

用户最新指令“继续DT1”：已恢复 Terra xhigh 执行剩余 DT1，主代理独立验收。先完成运动原语/零速校准与单条完整专家探针，再补齐 16 任务矩阵、配对复位、任务合同锁及 SmolVLA 共存推理；通过后停止，不进入 DT2。2026-09-05 13:55 UTC 远端核查无 GPU compute 作业、可用显存 24071 MiB、磁盘约 2 T 空闲，接触修复运行源码与验收快照一致。旧 TensorBoard/日志跟随进程不属本轮，不操作。

此前 Terra xhigh 已完成接触修复，主代理独立验收红蓝箱真实碰撞正例及足地负例：**CONTACT_FIX_APPROVED；完整 DT1 仍 NOT_APPROVED**。详见 `DT1_CONTACT_FIX_REVIEW.md`、`dt1_contact_fix_approval.json`。

本轮继续进展：完整单专家基线 `dt1_expert_baseline_20260905_r1` 原始 trace 独立通过（353 步、1.02 s 停稳）；运动校准 `dt1_motion_20260905_r1` 独立通过（600 步/2400 子步，前进/左右转部署端点有效）。零命令实测峰值 .012142 m/s / .009427 rad/s，保持 .03/.05 停车阈值不放宽。正在实施并审核配对初态记录与依据头部碰撞形体的 .75 m stand_off、末端朝向校准，尚未跑 16 矩阵或 SmolVLA 共存。详细本轮记录见 `DT1_CONTINUATION_REVIEW.md`。

本轮已部署 19 个刚体精确过滤及真实 view/count/matrix 启动验证；本地与远端双 Python 分别 42/42 CPU 回归通过。最新两个 run 为 `dt1_contact_red_20260905_r1` / `dt1_contact_blue_20260905_r1`：各 50 步足地负例（足底峰值 44.354 N、目标 0 N）和 153 步前向接近正例，首碰撞子步 1215，Head_lower 对本色目标 1037.414 N，评分 failed_collision。两轮独立 trace、实际传感器合同、代码哈希、四张 RGB、原始日志均通过主代理检查；旧 PhysX filter 错误不再出现。仅证明两路接触通道，不是模型选目标证据。

此前运动烟测 run：`dt1_shared_20260905T1245Z`。100 步零命令预热、50 对真实 pre/post、每步 4×0.005 s、约 1 s 物理动作；1.001 s 墙钟暂停时物理钟/位姿不变。主代理独立看过首尾图：初始红蓝箱可见，接近红箱时其仍可见；Go2 前进约 0.24 m、发生预期左转。raw episode 仍为 `FAILED_TIMEOUT`，符合 50 步限制，不能改写为导航成功。

已修复的历史否决项：旧 runner log 第 223–224 行，PhysX 对 `/World/envs/env_*/Robot/*` 报 `expected 1, found 19`。该旧 run 全零接触力仍不构成有效“无碰撞”证据；新修复及正负例不追认旧 smoke 通过。

资源独立重算：32 条外部采样，owned 峰值 4348 MiB（约 4.25 GiB），总 GPU 峰值 4436 MiB，最低空闲 20140 MiB。注意：完整 50 步那次外部作业已自行结束，不能声称是在“仅空闲 10 GiB 且有外部计算”的条件下完整验证；未加载 SmolVLA，未验证模型共存峰值。最终核查 owned PID 1831050 / PGID 1831048 已消失，GPU compute 列表为空、项目锁可获取。

源码/测试：接触修复后主代理本地及远端双 Python 各 42/42；真实 USD CPU 解析与三角面断言通过。地面、时间轴、history=9、异常留存及接触修复已部署。`scripts/dual_target_shared_smoke.sh` 仍有后续本地候选未部署：本地 `4f9eae7...`，远端及两轮 contact manifest `e266d913...`。不要在下次同步时混淆运行快照。

历史尝试/备份/失败均保留，见 DT1_REVIEW.md 与 `remote_runs/`。DT0 保持 APPROVED。DT1 仍欠：完整运动原语及零速校准、配对 reset、16 个专家成功、任务合同锁定、SmolVLA 新 3D 接口共存推理。没有模型任务成功证据。

| 阶段 | 状态 | 主代理验收 |
|---|---|---|
| DT0 | APPROVED | 最终验收见 DT0_REVIEW.md、dt0_approval.json |
| DT1 | PARTIAL_LIVE_EVIDENCE / NOT_APPROVED | 图像/运动/时序局部证据及本轮接触修复专项通过；其余门槛未齐 |
| DT2 | NOT_STARTED | — |
| DT3 | NOT_STARTED | — |
| DT4 | NOT_STARTED | — |
| DT5 | NOT_STARTED | — |
| DT6 | NOT_STARTED | — |
| DT7 | NOT_STARTED | — |

本文件由主代理维护；实施者的 gate.json 是自检证据，不等同于独立验收。
