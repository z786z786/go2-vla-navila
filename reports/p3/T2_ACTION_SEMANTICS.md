# P3-T2 动作语义

网络探测一次：`curl -I --connect-timeout 3 --max-time 5 https://github.com` 失败，exit 7（无法连接）。因此本报告只使用本地 `/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/`；论文专属或网络专属主张标记 MISSING。

## 完整枚举（本地真实源码）

证据文件为 [`eval_utils.py`]( /mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py ) 的 `get_vel_command`：

- `move forward 25/50/75 cm`：分别在 79–80、77–78、75–76 行返回 `vx=0.5 m/s`，持续 `0.5/1.0/1.5 s`。
- `turn left 15/30/45 degrees`：分别在 63–64、61–62、59–60 行返回 `wz=+pi/6 rad/s`，持续 `0.5/1.0/1.5 s`。
- `turn right 15/30/45 degrees`：分别在 71–72、69–70、67–68 行返回 `wz=-pi/6 rad/s`，持续 `0.5/1.0/1.5 s`。
- `stop`：82–83 行返回零速度、零持续时间。

该函数没有其它显式动作分支；未知文本会在 84–85 行默认落到前进 0.5 s，因而这个默认行为不是额外的已验证宏动作。

## NaVILA 执行路径

[`vlm_server.py:122-134`]( /mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/scripts/vlm_server.py ) 的 prompt 要求模型输出左/右转角度、前进距离或 stop。`navila_eval.py:327-336` 调用 `get_vel_command`，将速度和持续时间换算为 `env_steps_to_go = int(duration/(sim.dt*decimation))`，然后循环 `env.step(vlm_vel_commands)`。`wrappers.py:201-245` 把命令写入低层观测，调用低层 policy，再推进环境；Go2 配置在 `go2_matterport_base_cfg.py:364-372` 设置 `decimation=4`、`sim.dt=0.005`，即 50 Hz 环境控制步。动作是速度命令经低层策略执行，不是位姿瞬移。

## NavCommand 映射与 tick 算术

这是**执行 ASSUMPTION，不是监督**：用 `vx=0.5 m/s`、`wz=±0.5 rad/s`、`vy=0` 的 raw NavCommand，再交给平台共享限制器；5 Hz 每 tick 为 0.2 s，ZOH 为 10 个低层步。转换函数位于 [`p3_navila_action_map.py`]( /home/wxh/go2_short_vln/scripts/p3_navila_action_map.py )，只实现这个明确的探针假设，不声称复现 NaVILA 的监督标签。

采用满速整数 tick **向下截断**，所以残差是目标减去已执行积分：

| 宏动作 | 理想时长 | tick（低层步） | 截断后积分 | 残差 |
|---|---:|---:|---:|---:|
| forward 25 cm | 0.50 s | 2 (20) | 0.20 m | 0.05 m |
| forward 50 cm | 1.00 s | 5 (50) | 0.50 m | 0 m |
| forward 75 cm | 1.50 s | 7 (70) | 0.70 m | 0.05 m |
| turn ±15° | 0.524 s | 2 (20) | ±11.459° | 3.541° |
| turn ±30° | 1.047 s | 5 (50) | ±28.648° | 1.352° |
| turn ±45° | 1.571 s | 7 (70) | ±40.107° | 4.893° |
| stop | — | 1 (10) zero tick | 0 | 0 |

复核：0.2 s × 0.5 m/s = 0.1 m；30°/0.5 = 1.047 s，向下取 5 tick；45°/0.5 = 1.571 s，向下取 7 tick。派发单的“约 8 tick”是四舍五入近似；8 个完整 tick 会执行 45.837°，即超转 0.837°，所以本实现明确记录了向下截断残差。

## 门禁与验证

动作集合和每个数值均已由真实源码路径与行号确认，因此满足写 mapping function 的 precondition；未修改 `third_party/`。运行 `python -m unittest tests/p3/test_navila_action_map.py`：6 tests，全部通过。测试覆盖各档 forward/turn、stop、超范围数值及畸形输入，并 AST 读取上述真实 parser 验证其实际返回值。

论文专属定义、官方训练标签序列化以及网络补充均为 MISSING；本地树中查过 `src/`、`scripts/`、`isaaclab_exts/`，未发现独立训练数据枚举文件。
