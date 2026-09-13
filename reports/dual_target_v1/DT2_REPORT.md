# DT2 tiny 数据采集与转换（进行中）

**最终覆盖：DT2已由主代理验收APPROVED，见DT2_FINAL_REPORT.md及dt2_approval.json。以下内容保留为进行中历史；最新授权允许验收后直接开始DT3。**

最新状态（2026-09-06）：**COLLECTING_SHARED / NOT_APPROVED**。用户已明确授权把共享例外扩展到DT2、下调GPU门槛，并在DT2数据补齐且验收通过后直接调整参数启动DT3训练。主代理单独执行和复核，不安排Terra。以下v1排队信息属于保留的历史记录，由本段覆盖。

旧v1等待器1873188已取消并确认退出，未启动episode；保留其INTERRUPTED_RESUMABLE记录。新批次`dt2_tiny_0906_v2`父进程1873855，共享准入实际空闲8192 MiB、3×30 s及持锁复查，运行中仍保留2048 MiB余量，禁止操作外部作业。18:42 UTC检查新增slot8已成功，slot9等待启动；本批其余轨迹仍待实测。最新回归77项，远端76通过、1项处理器依赖跳过。

DT3只在完整16条、真实loader、归一化、采样及视觉审计通过后启动。训练与Isaac分开，依据显存实测选择micro-batch和累积数，effective batch=16，LR起始1e-4，先250更新检查点；后续档位仍须闭环诊断支持，不自动跑满5000，也不进入DT4。

## 已完成

- 固定4个train几何组、四组合各一次，共16条演示。计划`dt2_tiny_0906_v1/tiny_plan.json`，SHA256 `5aa10422f70d517f2a59c20b2919d0984f3f2fa4e8a50f586a53cb1845f07340`。
- 复用DT1固定矩阵中两个几何组的repeat=0四组合，共8条，不挑最好结果、不把重复试验计为新几何组。已全部按DT2原始轨迹条件重新检查；每条有51个真实停稳pre-action零命令观察，名义跨度1.0 s（Isaac float32时钟记录0.999999977648 s），无padding/复制补尾段。
- 新增组`dt2_train_002`：A=(2.55,1.35)、B=(2.25,-1.05)；`dt2_train_003`：A=(2.10,1.00)、B=(2.65,-1.60)。两者起点均(0,0)，沿用锁定的箱体/停车几何；四组几何签名不同。新8条尚未运行，不提前批准其可达性或可见性。
- 采集复用DT1完整物理控制/记录/评分函数，仅增加显式`scene_task`参数，不改控制或评分。DT1合同锁不变，runner历史/扩展SHA记录于`config/dual_target_v1/dt2_source_compatibility.json`；其余锁定源码逐项验证。
- 新3维LeRobot转换器只写front RGB、state、action、task及LeRobot自身索引字段。采用无损图像存储，不缩放/裁剪/增强；GT仅在独立审计文件，动作使用当帧执行前RGB/state与随后实际部署命令。
- 真实远端转换试跑`dt2_converter_probe_0906_v1`：2条完整轨迹、954帧，真实LeRobot loader抽查18个首/中/尾/边界anchor。RGB逐像素一致、state正确、50×3动作块和`action_is_pad`正确且不跨episode。输出明确为`PROBE_ONLY_NOT_DT2_DATASET`，不能替代完整16条数据。
- 归一化只使用所选train原始帧；记录raw_std，常量通道安全std=1。固定vy的mean=0/raw_std=0/safe_std=1；训练损失约定只作用于非padding的vx/wz，部署vy=0。正式DT2 normalizer须由完整16条重新生成，不能采用2条试跑统计。
- sampler默认episode均衡、episode内anchor均匀，有放回；保留全部实际帧及终端监督，无旧terminal≤10% cap，报告near-zero/terminal/padded chunk曝光比例。真实动作不加padding，padding仅发生在loader查询并有mask。
- 当前76项CPU回归：本地74通过、2项远端依赖跳过；远端Isaac环境75通过、1项SmolVLA处理器依赖跳过。真实转换试跑在SmolVLA环境运行，CUDA显式禁用。

## 当前队列及边界

唯一采集父进程/等待器PID1873188（统一exec会话9310），任务`dt2_tiny_0906_v1_s08`。使用既有资源服务的默认20 GiB且无外部计算作业门槛、3×30 s采样和持锁复查。资源服务JSON的stage=DT1是沿用的旧格式；真正阶段由DT2的collection_receipt/events/plan明确记录，不是重新启动DT1。

当前外部PID1866676占GPU、实际空闲17493 MiB。DT1共享例外明确仅适用于当时共存测试，不自动扩展到DT2。已非阻塞询问用户是否授权DT2共享16 GiB准入；尚未收到确认，故队列保持默认独占。共享实现仅为候选入口，没有启用。

队列文件：远端`/mnt/wxh/go2_short_vln/outputs/dual_target_v1/dt2_tiny_0906_v1/`内的`collection_receipt.json`、`collection_events.jsonl`及后续`collection_result.json`。资源状态仍在输出根`dt1_gpu_wait_state.json`。每次新episode有900 s墙钟上限和进程外实际空闲显存监测，低于2 GiB只清理本次隔离进程组；首个失败停止，不自动重试。

若用户后来授权共享：先通过既有取消接口取消当前WAITING_GPU，并确认父进程退出、没有episode启动；保留v1的取消记录，再生成新的批次根/receipt以共享模式启动，不覆盖现有不可重写记录。若已启动episode，则不能直接切换运行策略。

## 剩余验收

1. 完成新增8条，连同复用8条验证全部16条的时序、实际停稳、配对初态与全程可见性。
2. 转换完整16条，重新计算train normalizer与采样统计；用真实loader检查图像/状态/连续chunk，并由主代理另行重算数据与mask。
3. 生成最终数据清单、源码/合同/原始尝试哈希、审计视频和DT2批准记录；清理本次采集队列/进程，按最新授权进入DT3资源试测和训练。

## 部署与恢复

采集归档`dt2_collection_0906_v1.tar.gz` SHA256 `c230699b4971137fd50156e0b8ae8435e9a244b8bb32881d3c9a62ed1b41341e`。旧runner备份在远端`dt2_collection_deploy_0906_v1/before.tar.gz`。当前runner SHA256 `d506b4b2364f98b553ed79ded979f03567bbaacbefe5931fd29dedf0157745aa`。DT1批准及合同不改写。

本地计划镜像`dt2_tiny_plan_0906_v1.json`，试跑小型审计文件在`remote_runs/dt2_converter_probe_0906_v1/`；LeRobot实际数据及原始图像留远端。
