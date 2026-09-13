# DT1 继续执行：主代理验收工作记录

## 2026-09-06 最终验收完成

DT1 APPROVED，以下排队/进行中描述均为历史。共享共存 `dt1_coexist_shared_0906_v2` 实测及主代理原始证据复算通过：拥有进程采样峰值5618 MiB、最低实际空闲11850 MiB、有限[1,50,3]、推理886 ms，物理钟/位姿在模型等待期间不变。合同锁、总报告和批准记录已完成；本次全部进程退出且真实项目锁可获取。未执行模型动作/训练，未启动DT2，按用户指令停止。详见DT1_REPORT.md、dt1_approval.json。

## 当前完整矩阵结果（覆盖下文进行中状态）

`dt1_m_0906_v2` 的16项按预先固定顺序全部完成，launcher均退出0，collector退出0。主代理对真实远端文件运行 `dt1_matrix_independent_checks.py`：16条轨迹、8对task_equivalence_v2配对通过，单一冻结源码和单一运动校准哈希一致，停车窗均1.0199999772 s。复算结果与96项原始证据SHA256绑定保存于远端矩阵目录 `root_matrix_review.json`；此文件自身不代表完整DT1批准。

全程离线图像审计覆盖7444张实际观察图像，16条所选目标全程可见，初始红蓝两色及交换位置正确；16段审计视频已生成。视频右侧是GT轨迹俯视重建，不是额外仿真相机，未反馈给任何控制输入。

共存候选归档已部署（仅新增7文件，跳过macOS附属元数据，拒绝覆盖），5项专用CPU测试通过。`dt1_coexist_0906_v1` 等待默认GPU空闲门槛；目前无GPU共存通过结论。本地69项回归67通过、2依赖跳过。全部历史失败保留；不启动DT2。

最新用户授权：继续 DT1，由主代理单独实现、调试及复核，不再使用 Terra xhigh；不宣称独立代理验收。DT1 通过后停止，不启动 DT2。当前尚未完成全部门槛。下文较早的独立验收描述仅对应当时分工。

## 2026-09-06 简化后继续（本节覆盖旧暂停决定）

用户批准 task_equivalence_v2：任务配置、种子及复位协议相同，实际根位姿差≤2 cm/3°，每次history按wrapper约定清零且静止起步；保留完整关节/图像记录，不要求跨运行关节值或RGB哈希一致。无像素容差标定专项。每个RGB文件仍以自己的哈希验证完整性，真实颜色/可见性另做离线审计。

新蓝探针 dt1_blue_20260906_v2 已完成467步；原始轨迹重算停车1.0199999772 s，与红r4配对通过。离线真实RGB全程可见，审计视频已生成在远端 dt1_visual_probe_0906/00.mp4；右面板明确标注为GT轨迹俯视重建，非第二台仿真相机或策略输入。

固定矩阵 dt1_m_0906_v2 的16项计划已在GPU任务前写入，launcher SHA256 ca85b548680da94519abcbdeba81f32c99a9587ffdb4e7a8c6870ff68d4fae48；不逐项重试，不挑选最好结果。前8项完成，前6项已另脚本重算全部通过；当前进入第二组几何，尚未完整验收。运行源码保持同一快照，最新部署 reset_audit.py 0cfa88847bfb99b9c08593febb0a7e3ad03dc8a242d72a313aa1530e40ffe676，gpu_wait.py 6e448f8d8ac38d40bfbc853f5a2c0c6b815ba9b838858ae270d543f3fcf05c29。

SmolVLA共存候选尚未部署：真实远端SmolVLA环境CPU预处理检查通过；strict=True加载450046176参数全部匹配；真实红任务首帧完整CPU前向得到有限[1,50,3]，耗时5.625 s，torch.cuda.is_initialized=False，不执行动作/不训练。候选归档 /tmp/dt1_coexistence_final_0906.tar.gz SHA256 2ddec319c9d55dfb820c94f0bbc98776f876bbd82ee2860d8b019c6370059687，待矩阵结束后部署。CPU通过不代表GPU共存通过。

## 单代理接手后的集成修复（2026-09-05）

1. r1 reset 捕获不应读取首次 step 才生成的 terminal buffers；r2 writer 漏建 rgb_reset 目录。真实 OpenCV 测试在远端旧 records.py 上重现相同 Dt1RunnerError，修复后 reset/pre/post PNG 写入、解码尺寸和 RGB/BGR 顺序全部通过。远端当时 57 项 CPU 检查全部通过，无 skip；本地无 cv2，此集成测试明确 skip，不冒充通过。部署备份 `dt1_rgb_verified_deploy_20260905/before.tar.gz`；补丁 SHA256 `217f1440e3969d1fe5186efa2bc4c177a408d07152c545b0e4281f8aed05e071`。
2. `dt1_pair_probe_20260905_r3` 完成 100 步预热与 1500 步实际导航，但 FAILED_TIMEOUT。原始记录显示已到正确停车区，无碰撞，末端朝向围绕 .12 rad 切换阈值形成极限环；末段每约 15 步仅有 3 步零命令，不能覆盖 1 s 物理停稳。不是文件记录错误再次出现，也不是 SmolVLA 失败。
3. 使用实际 .119 -> .127 rad 回弹写出失败测试后，专家加入末端保持滞回：仍以 .12 rad 进入保持，以 .24 rad 退出；离开 .12 m 专家中心范围亦清除保持。评分器、.30 m 成功半径、.03/.05 实际速度阈值、1 s 连续停稳均不改。远端 58 项 CPU 检查通过。备份 `dt1_expert_hold_deploy_20260905/before.tar.gz`；补丁 SHA256 `de36d7e22c9be61e4afc71623a46eae6bf2b9fff812eff48ffcb84bf1ed4ffec`。新探针 `dt1_pair_probe_20260905_r4` 待真实结果，不提前宣称修复成功。

## 现有冻结版本完整专家基线

### 接手后新几何的 r4 结果

`dt1_pair_probe_20260905_r4` launcher exit 0 / result success。487 个 pre/post 动作对，主代理用不导入实现评分器的 trace 脚本重算通过，连续停车 1.0199999772 s。末机体速度 `[0.00070552,-0.00022631,-0.00164915]`，没有目标碰撞、跌倒、reset 或 evaluator_stop。首/中/终 RGB 已人工视觉检查，审计端（不参与控制）遍历 487 张输入图像，红目标颜色像素最少 4762，无完全出画帧。

新增未通过项：同种子、同任务 r3/r4 的 reset 根/关节/历史/目标位姿以及 learner-start 根位姿和机体速度完全一致；但图像不同。真实 OpenCV 对比：reset RGB 平均绝对通道差 0.160034/255、最大42、99百分位4；实际首次动作 RGB 平均差0.017368/255、最大5、99百分位1。不能从此推断具体随机源已定位。当前严格 RGB 哈希匹配会否决，未擅自改阈值。

蓝配对探针 `dt1_pair_blue_20260905_r4` 只启动了等待器 PID1848216，外部用户 cc 作业 PID1847848 占用 GPU，蓝任务未运行。15:54:29 UTC 已通过项目取消入口请求停止等待器，待确认退出。16 矩阵未启动。下一步需用户决定：优先追求逐字节确定性，或批准以严格物理一致和预先冻结的图像等价容差验证配对；不能用复制图像制造一致。

run：`dt1_expert_baseline_20260905_r1`，旧已审部署代码未改；任务 `dt1_dev_000 / A_red_B_blue / red / repeat=0 / seed=3401`。独占 waiter 3×30 s + fresh recheck，1500 env 步/30 s 仿真上限、900 s 墙钟 timeout。源码 runner `dfa6c2d...`、scene `39c3b449...`。返回 exit 0，stage 内 episode_result=success；stage 的 RUNNING 仅表示 DT1 未完成，不代表本次进程仍运行。

主代理从原始 pre_action/post_step_events JSONL 独立复算：353 步、每步 4×0.005 s，停车窗 1.0199999772 s，首次成功索引 352；所有真实目标接触力低于阈值，无碰撞、跌倒、外部停车或 reset。末位置 `(1.24254,1.14695)` m，停车中心 `(1.3,1.2)` m；末机体速度 `[0.001264,-0.000361,-0.000742]`。raw/applied 零命令和实际速度均满足现有草案 `.03 m/s / .05 rad/s` 门槛；这些门槛仍待独立零速校准确认。

暂停记录：墙钟 1.001086 s 内 physics step=405、sim time=2.0249999547 s 及 root pose 均未变化。

主代理已看初始 `rgb_pre/000001.png`（红蓝均清晰可见）、中段 `rgb_post/000178.png`（红目标右侧可见）及停车 `rgb_post/000354.png`（红目标仍在右下，但被画幅明显裁切）。不能将“可见”夸大成“完整居中”。目标可见性余量与停车区域安全间隙在 DT1 几何锁定前继续核查。

结论：单条专家基线数值通过，支持沿用现有专家，不必无依据重写。不是 16/16 配对矩阵、不是已校准任务合同、更不是 SmolVLA 导航成功。原始证据存 `remote_runs/dt1_expert_baseline_20260905_r1/`。

## 后续验收队列

1. 真正静止、前进、左转、右转及回零；校准只用零命令稳态，不根据模型失败倒推。
2. 记录配对 reset 的根/关节/速度/历史缓冲区、箱体物理状态与第一张 RGB；配对种子不含指令颜色。
3. 两组四组合两次重复共 16 个专家全部成功，独立重算停车；保留失败尝试和实际命令。
4. 停车几何、实际速度阈值、动作边界与最长任务时间形成输入可追溯的合同锁。
5. Isaac + 缓存 SmolVLA 新 3D state/action 接口真实共存推理，测双进程显存峰值；不训练、不执行未适配模型随机动作。
6. 俯视视频、视角/泄漏核查及 owned 进程/锁清理；完整验收通过后停止。

## 运动探针部署记录

主代理核查并备份旧 runner/test 后仅部署 5 文件，归档 `/tmp/dt1_motion_patch_20260905_r1.tar.gz` SHA-256 `ef0b4bb412a057fc31a5ab30ca6292b4bcaf0508659b32d7784a38c0fc3d8750`；可恢复副本位于远端 outputs/dual_target_v1/dt1_motion_deploy_20260905_r1/backup。地面/contact/旧 shared supervisor 不改。

| 文件 | SHA-256 |
|---|---|
| calibration.py | d1cb81c5b5fdce795b2c19840328f07437af4833630470820501ed5aa131508a |
| motion_precheck.py | 5463cc0e3dc286ce8b9c5b98b2d9ad82b36774c8220d836473159b42e6a29d28 |
| runner.py | b55f19a3044b0252b8fbafda2a4dba60fa17dd9aa09e63bf931d84131216dd72 |
| dual_target_dt1_expert.sh | 72b770ffefe9007272351ecd57e6a2ab1e24c3ad4cbbac7489b008460509f575 |
| test_dt1_cpu_components.py | aba79429a70671d713d805ad8a757ebcfddeafb1631c240fac81040dd3d41797 |

主代理本地及远端两个 Python 各 45/45 CPU 测试通过，shell 语法通过。独立 `dt1_motion_independent_checks.py` 合成自测接纳 1 个有效例、拒绝 7 个无效变体（不算实测）。真实 run `dt1_motion_20260905_r1` 在 2026-09-05 14:12:57 UTC 完成 3×30 s 独占准入后启动。固定 100 步预热 + 600 步运动/回零；前进 `.5 m/s`、转动 `±.5 rad/s`，900 s 墙钟上限。每个回零段末 51 个真实观察用于校准，公式 max(原草案阈值, 实测峰值×1.2+.002) 向上取到千分位；超出 `.09 m/s / .15 rad/s` 包络即失败，不自动改包络。

真实运动专项已通过主代理独立复算：600 步/2400 子步；沿初始朝向前移 0.716922 m（平面位移 0.717235 m），mean body vx=0.481640 m/s；左/右转 +0.710142 / -0.736158 rad。四个零命令末窗 planar 峰值 0.0121420091 m/s、yaw 峰值 0.0094267167 rad/s，公式保持原草案 `.03 m/s / .05 rad/s`，无需放宽。目标未移动、无碰撞/reset，原始运行日志无 Error/Traceback/Filter pattern。launcher exit 0，完成时 GPU compute 列表为空。

## 停车几何与配对补丁早审

主代理读取真实 Go2 USD 隐藏碰撞体（CPU，无 Kit/CUDA）。必须用 `UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy','guide'], False, True)`；普通可见性过滤会给碰撞体空范围，旧 USD 22 构造器不支持 `ignoreVisibility` 关键字，只支持位置参数。Head_lower sphere 静态 world x=[0.2459999844,0.3399999812]，y=±0.0469999984；静态平面包络约 0.344 m。该读数与前轮真实触箱位置相符，但不是完整步态包络。

开发任务 stand_off 拟校准为 .75 m、parking radius 保持 .30 m；最近盘边缘 root 到箱面 .45 m，相对上述静态包络约 .106 m 余量。专家在进入中心后增加固定箱体前向对准，再发零命令；动态接触仍一票否决，不能声称所有姿态碰撞自由。原 contact 撞箱诊断保留自己的 .45/.10 非任务停车参数。

配对记录需覆盖 reset 根/关节/速度/9×45 历史、两箱位姿、同场景 hash，以及 reset 和 warmup 后实际第一观察两次 RGB/状态。主代理在 GPU 启动前发现 sidecar 的 learner_state 读取 `robot_root_pose_w`，而 `_default_runtime_snapshot` 实际返回 `robot_pose_w`；退回修正并要求直接调用 sidecar writer 的集成单测，不能只靠手工 ResetAudit 对象测试。
