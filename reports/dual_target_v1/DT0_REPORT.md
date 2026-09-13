# DT0 — 环境审计与 CPU 合同准备

状态：`READY_FOR_REVIEW`（Terra 自检）。这不是主代理的 `APPROVED`，不得据此开始 DT1。

## 已完成的 CPU-only 产物

- `src/dual_target/contracts.py`：严格 allowlist 的 `RGB + [body_vx, body_vy, body_yaw_rate] + task` 输入、`50×3` 有限动作和 `[vx, vy, wz]` 部署限幅。审计 ID 只能放在 envelope 的 `audit`，不能进入 `policy_input`；task 仅接受规范空白后的两条冻结指令，拒绝把坐标/目标真值藏进 prompt。
- `src/dual_target/layouts.py`：基础几何组、两个颜色配置 × 两条指令的四组合；候选始终是 `pending_sim_validation`。除 lineage 外，5 cm bin 的平移/镜像不变 geometry signature 会拒绝重命名同一几何或镜像小扰动跨 split/重复计组；同组还验证冻结文本、颜色→槽位、停车区→槽位和停车区映射不变。拒绝空 split、起点位于停车区、停车区重叠、有效移动距离不足和停车圆边缘与箱体缺少静态间隙。
- `src/dual_target/scoring.py`：评分器不会发出任何命令。连续窗口要求严格递增的 `sim_time_s`、`physics_step` 和 `observation_seq`，相邻观察最大 50 ms；窗口从第一帧开始计时，故 50 个 20 ms 帧仅有 0.98 s，必须有第 51 个新帧才可成功。事件字段必须是 `bool`，`applied_action` 必须是 raw action 经固定映射所得且 `vy=0`。NaN/Inf、重复时刻/步号、warmup、raw 负 `vx` 被限幅为零均不能得分；类型有效的碰撞、跌倒、错误目标停稳、外部 `evaluator_stop` 在其他同帧遥测为 NaN 时也会锁存失败。
- `src/dual_target/stage.py` 和 `scripts/dual_target_run_stage.sh`：只实现 DT0；DT1+ 均返回 64。自检状态机仅提交 `NOT_STARTED → RUNNING → READY_FOR_REVIEW`，代码拒绝 Terra 标记 `APPROVED`。项目固定 `fcntl.flock` 防止靠换 `--output-dir` 绕过并发；run ID 唯一、拒绝覆盖，且创建 run 后异常会写 `FAILED`、PID 与原因。GPU wait 对**提供的 JSON 快照**计算连续三次准入/重置，而不查询或分配 GPU。

`config/dual_target_v1/` 内的值均为 DT0 draft，且明确要由 DT1 的相机、碰撞、停稳和物理时钟测量后冻结。

## 远端只读证据

完整机器快照在 [dt0_remote_audit.json](dt0_remote_audit.json)。远端存在 NaVILA、IsaacLab、两个隔离 Python 和缓存的 SmolVLA；GPU 为 RTX 3090 24 GiB，但审计时有外部计算进程使用约 20.2 GiB。DT0 未启动等待器、Isaac、CUDA 模型或训练。

远端工作树不是 Git repository。同步前已逐项确认本阶段拥有的六个目标均不存在，因此不会覆盖既有远端实验文件。

## 已定位的 DT1 接入合同

| 项目 | 已证实的源头 | DT1 必做项 |
|---|---|---|
| 物理网格 | `go2_matterport_base_cfg.py`: `sim.dt=0.005`, `decimation=4` | 以实测确认一个 wrapper `step` 对应 4 个物理子步/0.02 s，而不是将 50 Hz 误写为物理积分频率。 |
| 低层观察 | `go2_matterport_vision_cfg.py` 的 policy 顺序为 `base_ang_vel(3), base_rpy(3), velocity_commands(3), joint_pos(12), joint_vel(12), last_action(12), lidar height_map`；proprio 为前六项且 history 9 帧 | 保持已训练低层的维度与顺序；`height_map` 实际用 `lidar_sensor`，不可只改 `height_scanner`。 |
| 速度注入 | `wrappers.py: VLNEnvWrapper.update_command` 对 history wrapper 同时写 `low_level_obs[:,6:9]` 与 `proprio_obs_buf[:,-1,6:9]`；非 history 分支写 `9:12` | 真实实例确认走的 wrapper 分支、观测 shape、history 最新帧和在 `env.step` 后命令是否保持为当前值。 |
| reset | `VLNEnvWrapper.reset` 以零命令 warmup 100 步 | 所有 warmup 样本标成不可计分；数据的首个 policy observation 必须在 warmup 后。 |
| 旧目标命令 | `base_velocity` 是 `UniformVelocityCommandCfg`，三个随机速度范围均固定为零，但含 heading 配置 | 不将声明误当为运行时事实：DT1 记录 generated command 与 wrapper 覆盖后的低层输入，证明没有 goal 导引。 |
| 场景与 ray cast | 两个 ray caster 当前都绑 `/World/matterport`；IsaacLab ray caster 初始化要求一个 `mesh_prim_paths`，且只取其下第一个 Plane/Mesh | 自有 adapter 明确采用 **single ground-only mesh** 或一个预合并 static mesh；不能传 `ground + red box + blue box` 三路径。前者可用于无障碍近场任务，但须在报告中说明目标箱不进入低层 terrain 输入。 |
| 终止证据 | `ManagerBasedRLEnv.step` 在返回前可能自动 reset done env | DT1 在 auto-reset 前采集 contact、termination、位姿和评分事件；禁止用 reset 后位置当作终点。 |
| Go2 asset | 基础配置从 `UNITREE_GO2_CFG` 复制 | 在 `gym.make` 前对最终 `env_cfg.scene.robot.spawn.usd_path` 显式设置并记录已核验本地 USD，不仅修改资产全局路径。 |
| 低层 checkpoint | 旧 `demo_planner.py` 经 RSL `get_checkpoint_path` / `ppo_runner.load` / `get_inference_policy` 取得 policy | DT1 固定实际 `load_run`、checkpoint 文件、normalizer/runner 版本和 hash；DT0 只定位路径链路，未加载。 |
| 视频 | 已知 Kit close 可使 MP4 缺 `moov` | DT1 先落帧，再由独立编码步骤生成 MP4。 |

## 自检与命令

```text
PYTHONPYCACHEPREFIX=/tmp/vln_dual_target_pycache python3 -m unittest \
  tests.dual_target.test_contracts_and_layouts tests.dual_target.test_scoring_and_stage
# 18 tests, OK

scripts/dual_target_run_stage.sh DT0 --run-id dt0_terra_review_002 \
  --gpu-wait-dry-run --gpu-wait-snapshots tests/dual_target/gpu_wait_ready_snapshots.json
# exit 0; final self-check gate below; no GPU probe/allocation

scripts/dual_target_run_stage.sh DT1
# exit 64; refuses unimplemented/unapproved stage
```

最终自检 gate：[dt0_runs/dt0_terra_review_002/gate.json](dt0_runs/dt0_terra_review_002/gate.json)。它记录实际 argv、run ID、源/合同哈希和三份模拟资源快照（恰好连续三次满足），但不把这组模拟快照宣称为远端实时资源可用性。

低层 checkpoint 已具体定位：`/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/logs/rsl_rl/go2_vision/2024-09-25_23-22-02/` 下的 `model_26499.pt`（`1e210971…e76c`）、`model_26000.pt`（`b9b4b051…a446`）和 `model_25500.pt`（`73cb111f…d1c6`），各 13,873,706 bytes；完整 SHA-256 在审计 JSON。DT1 必须通过 `--load_run`/`--checkpoint` 解析出唯一候选并记录该 hash，不能默认“latest”。

本机 `/usr/bin/python3` 是 Python 3.9，故 stage 入口不使用 `zip(..., strict=True)`；远端计划运行解释器为 Python 3.10/3.12。编译缓存被限制在 `/tmp`，不影响源码或实验数据。

## 限制与交接

DT0 没有、也不声称有真实相机帧、碰撞、低层控制、物理时间、GPU 共存峰值或专家成功证据。这些均是 DT1 的明确验收项。下一动作只能是主代理审阅 `gate.json`、源码和本报告；主代理 `APPROVED` 后才可下发 DT1。
