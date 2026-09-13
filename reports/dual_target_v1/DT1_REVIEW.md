# DT1 主代理验收（含历史独立分工记录）

## 最新最终结论：APPROVED，2026-09-06

以下NOT_APPROVED/接触错误/排队状态均为保留的开发历史，不是当前结论。主代理按用户最新单代理工作方式完成16/16矩阵原始轨迹重算、8对task_equivalence_v2配对、全程图像审计、运动及接触证据复验，以及真实Isaac＋SmolVLA共享GPU共存验收。完整报告DT1_REPORT.md，合同`config/dual_target_v1/task_contract.lock.json`，机器记录dt1_approval.json及dt1_final/gate.json。已清理本次进程并释放真实项目锁；DT1通过后停止，DT2未启动。模型导航成功仍未经验证。

状态：PARTIAL_LIVE_EVIDENCE / DT1_NOT_APPROVED。DT0 已批准；最新真实 50 步图像/运动/时序已独立核查，但 PhysX 接触过滤初始化错误否决完整烟测，DT1 未通过。此前“尚无真实结果”等段落为开发历史；本轮最终结论见文末。按用户最新指令，每个里程碑通过后必须停下，等待用户下一里程碑指令；本轮仅授权的烟测已执行完，未进入 DT2。

## 已发现的低层历史更新风险

主代理只读核对远端 `RslRlVecEnvHistoryWrapper.step`：`torch.cat([obs, proprio_obs_history], dim=1)` 创建了新的拼接张量。原 `VLNEnvWrapper.update_command` 写当前 policy 的 `6:9` 和独立历史 buffer 的最新 `6:9`，未回写拼接张量的历史尾部。新 adapter 必须显式重建或同步尾部；保留旧 checkpoint 的观测顺序、尺度、45 维 proprio 和 9 帧 history，通过逐帧断言证明当前命令与最新历史命令一致。

## 独立验收重点

1. 真实 RGB：初始红蓝都可见、颜色交换正确、接近和停车时目标可见；无 expert marker。首帧和动作前画面有真实相机更新证据，不用像素必须不同来判断新帧。
2. 物理场景：平地 mesh 与 ray-caster 绑定正确，实体箱碰撞；普通足地接触不误报，受控撞箱确实被捕获。ground-only 地形观测的范围须明示。
3. 执行时序：实测 0.005 s 物理子步、4 次 decimation、0.02 s 高层步；动作前观察与执行后评分不混淆；等待推理的 1 s 墙钟内真实物理时间和位姿不变。
4. 终止：自动 reset 之前保存碰撞/跌倒与位姿，终止失败不能被 reset 或后续正常帧擦除；外部零速保护只记失败，不能生成自主成功。
5. 配对：两几何、每组两颜色配置×两指令×两重复，共 16 个指定专家任务全部真实成功。相同几何/颜色/重复的两指令使用相同 seed、reset 状态和首帧条件；保留全部失败尝试。
6. 停稳：独立从 raw/applied 命令、实际机体平移/转动速度、停车区和连续物理时间重算。至少 1 s 有效新观察，warmup 不计，评分器不负责发停车命令。实际机体阈值来自零命令稳态校准，不从学习器失败倒推。
7. 资源：无外部计算作业且连续空闲才启动，锁后重查；Isaac 与缓存 SmolVLA 真实共存并完成新 3D 接口推理，峰值及两个进程证据齐全。此预检不是未训练模型的导航能力测试。
8. 交付：合同锁、checkpoint 与源码 hash、全部实际命令/退出码/日志/图像/视频和任务记录可复核；仅拥有的作业已清理；不越级采 DT2 数据。

尚无真实 DT1 仿真结果，不得据此报告模型正确性。

## 开发中首批审查

- `low_level.py` 已显式重建拼接历史，方案符合对旧封装问题的修正；仍待真实低层输入维度和运动验证。
- 主代理实测：`SubstepContactLatch.capture({"red": NaN, "blue": 0})` 在首批版本返回 `False`，记录最大力均为 0。已退回修复：坏接触数据必须报错并保留失败，不能变成“无碰撞”。同时要求负力拒绝与严格 JSON 序列化。此为开发中发现，尚不构成 DT1 最终验收结论。
- 主代理已编写 `dt1_independent_checks.py`，不导入运行时 scorer。离线合成 self-test 接纳 1 个有效记录、拒绝 9 个伪成功变体；真实 trace 尚未输入。它只审动作、物理时间、位姿/区域和停稳，不代替视觉、资源和全套实验验收。
- 接触力修复复验：主代理分别输入 NaN、Inf、负数和布尔，均被拒绝；四子步 `[0,2,0,0] N` 的瞬时接触保留为 `[False,True,False,False]`、最大力 2 N。开发期纯 CPU 缺陷已修复，真实传感器接触仍待验证。
- GPU 等待首审：主代理注入“显存查询正常、计算进程查询退出 1 且输出为空”，首版 `probe_gpu` 将其判为 eligible。已要求非零退出/未知输出 fail-closed、有界查询超时；取消操作须改为项目内取消文件或严格核验 PID 所属项目/启动时间，不能仅凭模块名 signal。同样须保证测试注入不成为真实启动授权。
- live runner 首审：待修缺失的 latch 窗口方法、RSL `done` 整型张量解析、warmup 重置失败未处理、自动 reset 优先级、硬编码 `fallen=False`、由预期 step 计数冒充实测物理时钟、观察编号语义及直接 `--live` 的资源准入。均已退回实施者；第一版仅为开发代码，尚不批准启动真实作业。
- 后续开发复验：31 项单测和 runner CPU self-check 通过。主代理独立确认查询退出异常、超时、N/A/none/错误文本均被拒绝；取消已改为项目内请求文件。runner 已兼容 TensorDict 与严格 0/1 done、处理 warmup 失败与 post 终止优先级、复用上次 post 作为下一 pre。仍要求底层物理时钟/暂停验证、错误目标持续停稳语义及资源就绪时效绑定；DT1 继续实现中。
- 本地冻结点复验：33/33 单测及 DT1 CPU self-check 通过。新增底层 SimulationContext step/time 与 manager 固定 offset/增量核对、真实暂停观测、错误目标独立 1 s 停稳窗；GPU 就绪要求同 run_id、≤90 s、新鲜持锁复查及一次性消费；DT1 当前源码 manifest 单独绑定历史 DT0 approval，不把入口版本变化误认为 DT0 失效。允许准备限定同步及远端 CPU 复验；首次真实启动仍须资源条件和主代理对远端复验结果确认，完整 DT1 仍未通过。

## 远端 CPU 部署与复验

主代理已完成，而非仅依赖实施者自报：

- 同步归档 `/tmp/dt1_source_sync_20260905T1900Z.tar.gz`，SHA-256 `8da746e19614a3cb7d5b1d04a36953371f2d61db8949053b8189761cc35f691c`。11 项限定文件，包含 7 个新增 DT1 源文件、阶段入口、DT1 CPU 测试、DT0 approval 和历史 gate；不含模型、数据或凭据。
- 主代理传输后复核 9 个目标 ABSENT、历史 gate 同 hash、阶段入口仍为批准的 DT0 基线。解包到新暂存目录，仅复制新增文件及明确更新阶段脚本；历史 gate 仅 cmp，没有覆盖。
- 旧阶段脚本可恢复副本：`/mnt/wxh/go2_short_vln/outputs/dual_target_v1/dt1_cpu_deploy_20260905T1135Z/backup/dual_target_run_stage.sh`，原 hash `60dcae8c4a1c4c567031795b055a5d3eeef1cb96f37bb5ce6aebe5f22f3a5794`。新脚本 hash `953f1861af6b18da86833eefe3932521041f1618566b4309b03167d52f2d33cc`。
- 远端两个项目环境设置 `CUDA_VISIBLE_DEVICES=`，分别运行 `python -m unittest discover -s tests/dual_target -v`，Python 3.10 / 3.12 均 33/33，退出 0；仿真 Python 执行 `python -m src.dual_target.runner --cpu-self-check` 退出 0，无 GPU allocation。
- 主代理逐项比对所有双目标源码、阶段脚本、DT1 测试、DT0 approval 的本地/远端 SHA-256，全部一致。当前 runner hash `f492174c68f292ed8b841fe8a99237f01935b8232811dbd111e4411b83b8f32a`；gpu_wait hash `5eb0343c877d57f14765ce0ce68a693bfcb32638af323b9c0dd0e2fa6cd0afab`。

允许进入首次有界真实烟测的资源等待：仍需同 run 的 3×30 s 空闲准入与持锁复查，首次结果交主代理检查再继续 DT1。此许可不是 DT1 APPROVED；16 个专家任务、运动原语/碰撞实测、视觉检查、合同校准及 SmolVLA 共存均仍待完成。

## 用户同意共享 GPU 烟测（2026-09-05）

资源例外仅用于一次 Isaac + Go2 低层烟测：至少 10240 MiB 空闲、连续 3×30 s 并持锁复查，允许记录已有外部计算进程；100 步预热后最多 50 个高层物理步，不加载 SmolVLA、不训练。启动与执行须进程外监测显存，低于 2048 MiB 余量时终止本次拥有的子进程组，并设 900 s 总时限。该策略不能防止其他用户突然申请显存，不是共享 GPU 预留保证。其余 DT1 资源验收仍未完成。

旧独占 waiter `1824449` 已消费取消请求并退出；主代理复查该 PID 不存在，随后才审查新版本。

本轮启动兼容性审查：

- 旧 `src/inference/isaac_client.py:216` 已有真实历史使用的本地资产路径与首次 `SimulationContext.reset` 前延长 Kit timeline 的实现。新 runner 原先缺这两项；两秒 timeline 与 100×0.02 s 预热冲突，已要求仅移植必要的进程内初始化，不导入旧导航控制逻辑。
- 主代理读取远端 `NaVILA-Bench/scripts/cli_args.py`：旧入口显式设置 `rslrl_cfg.policy.history_length = args_cli.history_length`，旧 client 默认 9。新 runner 必须同时设置 wrapper 与 policy 的 9 帧 history，不能只设置 wrapper。
- 进程内 monitor 仅设置失败标志无法及时中断 AppLauncher 的阻塞调用；要求进程外 supervisor 只管理自己创建的子进程组。未验证前不启动。
- 主代理独立 trace 工具新增 `--smoke-only`，要求恰好 50 步，保留时钟、观察、命令和碰撞完整性检查，明确不批准任务成功或 DT1；合成自测通过，不是真实仿真证据。

共享烟测部署复验已完成：

- 主代理独立执行本地 35/35 CPU 单测，通过；远端仿真 Python 3.10 和 SmolVLA Python 3.12 各 35/35，CPU self-check 和 `bash -n` 通过，全程未分配 GPU。
- 限定 4 文件归档 `/tmp/dt1_shared_patch_20260905T1213Z.tar.gz`，SHA-256 `7fe14872278148cdd221204ae33d1e8e2a8e9c0d7e642cf467eb66ad40c98a47`；部署前原 3 文件哈希与已审基线一致，新 supervisor 不存在。
- 原文件备份及新暂存：`/mnt/wxh/go2_short_vln/outputs/dual_target_v1/dt1_shared_deploy_20260905T1213Z/{backup,incoming}`；未修改旧数据、checkpoint 或 DT0 approval。
- 新哈希：gpu_wait `207f426004edc7d2742b8059abacf541167de46f8f15e71229b58905ede336a1`；runner `c0ea3e941eac24216cf6eb140eb9d8d689a15c198f6b7263f796f4943774b5ed`；supervisor `d3931c676e8d7783b082703bc0c6ff43acd611f523321d15f049f029aba8d477`；DT1 CPU test `391d15c7d05ecb494b48dd94dfcd7c8631940070e1cc5866b14a14d6932e3ed2`。主代理核对远端输出全部一致。
- 进程外保护审查修复了查询无限阻塞、TERM 后父进程先退出而子进程残留、shell 异常退出漏清理三个边界问题；查询有 5 s timeout，已验证的拥有进程组用 TERM/10 s 宽限/KILL 清理，EXIT trap 兜底。

主代理已允许 Terra 启动唯一共享 waiter，满足固定准入后通过已审 supervisor 进行一次 50 步测试。仍不是 DT1 通过。

### 首次实际启动结果与限定修复

- `dt1_shared_20260905T1215Z`：waiter 就绪后交接超过 90 s，在 AppLauncher 前准入失败；未启动 Isaac、未分配本作业 GPU。
- `dt1_shared_20260905T1221Z`：同一远端命令衔接 waiter 与 supervisor，准入成功。Isaac App 约 11 s 启动完成，Kit timeline 从 1.6667 s 延长至 10000 s，随后因缺失 `Isaac/Environments/Grid/default_environment.usd` 在地面场景构建失败。stage_status=FAILED，`AttributeError: 'NoneType' object has no attribute 'GetPath'`；没有 episode、warmup 或 50 步记录。主代理判定烟测未通过。
- 主代理从下载原始 JSONL 独立重算：14 个外部采样、owned 峰值 1513 MiB、最低空闲 12525 MiB，无资源保护触发；拥有进程已退出，外部 PID 1805362 保留。该占用仅是启动阶段数据，不是完整 Go2 或模型共存峰值。
- 已发现 Kit 的关闭路径可能将异常的 shell 退出码变成 0：本次实际 stage FAILED，旧 supervisor 仍报告 exit 0。独立验收未据退出码误报成功；修复为关闭前持久化完整 traceback，supervisor 再读 stage_status。
- 修复改用项目内 `config/dual_target_v1/assets/dt1_flat_ground.usda`：20×20 m 静态碰撞 Mesh，无外部资产引用；显式物理材质 static/dynamic friction=1、restitution=0。仍仅 ground raycast，目标仍是真实碰撞实体。
- 修复部署归档 `/tmp/dt1_ground_fix_20260905T1230Z.tar.gz`，SHA `794415c2feb0c95314871584a1d437ec4b26aff51c16dc6172689345c4125cd1`；四个旧文件部署前核对并备份到 `dt1_ground_deploy_20260905T1230Z/backup`，仅新增上述地面资产。主代理本地及远端双 Python 各 37/37，shell 语法通过。
- 远端冻结 hash：scene `67278bea49da93b9b470c54d3c810c298113e2a92c12eddb3e0357317ac5e6df`；runner `39e86b28e3c9ef5ce2297b445514b5a06f1868b7d11dc65809a17a189c34c9c8`；supervisor `e266d913370bc06722febc009168a149a63761955e3bc717796acde6a634b7bc`；ground `eab0d591d7faa08a316da85d1415f6539a5631aa5757d192d99975bc15668bcb`；test `8d7e0ebf4c0539599d683f4eaf1d37981d340c70886d1bcbebfb5e6cb1d1e979`。
- 已安装 USD 解析器需要其自身 `omni.usd.libs` Python 和动态库目录；仅为 CPU 解析命令设置这两个进程环境变量，未改系统/安装依赖。离线真实 `Usd.Stage.Open` 与 Mesh/CollisionAPI/MaterialAPI/摩擦断言通过。随后同命令开始 run `dt1_shared_20260905T1233Z` 的准入与同范围一次复测。
- 备注：实施者随后本地 supervisor 候选 `4f9eae7...` 尚未部署；远端运行快照仍为 `e266d913...`，不得混报或运行中覆盖。

### 地面三角面兼容修复

`dt1_shared_20260905T1233Z` 已进入真实地形导入，但旧 IsaacLab `terrain_importer.py:272` 直接将 USD 顶点索引 `reshape(-1, 3)`，不接受四边形面，失败为 `ValueError: cannot reshape array of size 4 into shape (3)`。这次完整 traceback 已保存，supervisor 正确返回 1，未误报成功。主代理重算 13 条采样：owned 峰值 1513 MiB、最低空闲 22976 MiB；当时外部计算已自行结束，未干预其他用户。

限定两行修复：地面变为两个三角面 `[3,3]` / `[0,1,2,0,2,3]`；回归测试增加此旧版本契约断言。只更新 ground 和 CPU test，原文件备份到远端 `dt1_triangles_deploy_20260905T1244Z/backup`；归档 SHA `da53fd99af08276017d5ff39f8fb30ed45b9e15bb679127be9fbe819cdd536ae`。新 ground SHA `85c5d540d27f30a097e0df78ab5a2d957d5bfd358e3a67a233c7394012131e43`；test SHA `da9c25486fc2ffbdc73f1800abfdae6a0af07d894b09d9f7f04ff743ffb2f236`，主代理核对本地/远端一致。

主代理本地与远端两个 Python 各 37/37；远端真实 `UsdGeom.Mesh` CPU 读取 faceVertexCounts/Indices 并精确断言通过，同命令衔接 `dt1_shared_20260905T1245Z` 固定资源准入与同范围 50 步复测。不加载 SmolVLA，不扩大到全 DT1 批次。

## 本轮最终独立结论：局部链路有效，完整烟测与 DT1 不批准

最终 run `dt1_shared_20260905T1245Z` 已执行结束，原始文件完整下载到 `reports/dual_target_v1/remote_runs/dt1_shared_20260905T1245Z/`。source_manifest SHA `77724d1af8270b88c6ae05fe0bada6595953bb35636056857120c6446289819a`，与 partial gate 的绑定一致；24 项 manifest 中仅 supervisor 与后续未部署的本地候选不同，已明确区分。

主代理实际复验（不只依赖 Terra 自报）：

- 独立 `review_trace(..., smoke_only=True)` 通过：50 对 pre/post，每步 4 个子步，累计 0.9999999776 s；100 条 warmup 独立、连续、命令为零。该工具仅证实记录内部完整性，不能覆盖初始化日志中的传感器错误。
- camera frame 101→151，50 次有效更新。主代理查看真实首帧 `rgb_pre/000001.png`：左右红蓝箱均可见；尾帧 `rgb_post/000051.png` 红箱在前方可见，无专家路径 marker。
- Go2 从约 `(0,-0.004)` 前进至 `(0.233,0.068)` m，yaw 约 0.037→0.570 rad，实际前进/左转响应存在。这由真值专家发速度命令，**未加载 SmolVLA，不是模型视觉语言选目标证据**。
- pause 原始记录墙钟 1.001094 s，SimulationContext step=405、time=2.0249999547 s 和 root pose 前后完全一致。初始化 manager 0 / sim 5 固定 offset 被记录。
- 主代理重算 32 条外部显存样本：owned peak 4348 MiB、total peak 4436 MiB、min free 20140 MiB；准入 fresh check 距离 waiter 0.23423 s。完整 50 步运行时外部计算已经自行结束，因此不能声称验证了与其他任务共占 10 GiB 空闲的完整运行，更未验证 SmolVLA 共存。
- raw `FAILED_TIMEOUT` 与 shell exit 1 保留：50 步预算本来不足以到达并停稳，不能改写成导航成功。主代理最后只读确认 owned PID 1831050 / PGID 1831048 消失，GPU compute 列表为空、项目锁释放。

**否决项（主代理已复核原始日志第 223–224 行）：** PhysX 两次报告 `Filter pattern '/World/envs/env_*/Robot/*' did not match ... (expected 1, found 19)`。目标 contact sensor 的通配符初始化无效，本次全零目标接触数据不能作为碰撞链有效或“无碰撞”的证据。先前用户进度中“无碰撞”的表述已明确收窄/更正为：记录全零，但传感器有效性未通过。

因此本轮状态为 **PARTIAL_LIVE_EVIDENCE / DT1_NOT_APPROVED**，不签署完整 smoke approval，不推进 DT2。本轮授权的 50 步检查完成后停止执行；下一次继续 DT1 的第一项是精确刚体 filter 配置及受控撞箱正例、普通足地接触负例，不能只让错误日志消失。随后才补其余运动原语/零速校准、16 专家成功、配对 reset、合同锁及新 3D SmolVLA 共存验收。

## 2026-09-05 接触修复专项验收补记

用户随后授权“修复”。Terra xhigh 实施逐刚体精确过滤与真实 backend 验证，主代理独立核查 `dt1_contact_red_20260905_r1` / `dt1_contact_blue_20260905_r1`：各 50 步足地负例无目标误报、153 步前向接近后真实触箱（Head_lower，1037.414 N），首碰撞物理子步 1215、评分 failed_collision；原始 trace、传感器合同、源码哈希、四张 RGB 与日志均通过。原过滤错误已修复，专项 **CONTACT_FIX_APPROVED**；旧无效接触证据不追认，完整 DT1 仍 **NOT_APPROVED**。详情见 `DT1_CONTACT_FIX_REVIEW.md` 与 `dt1_contact_fix_approval.json`。本轮测试进程已退出、项目锁释放，停止等待用户指令。
