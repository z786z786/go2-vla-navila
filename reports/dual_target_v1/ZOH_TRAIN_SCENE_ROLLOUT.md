# 训练布局闭环拟合检查

2026-09-06 用户请求用 train 数据 rollout。范围为固定 checkpoint 5000 在第一组已见训练布局的四条配对闭环；并非专家动作回放、并非训练更新，成功也不计入独立验证/测试成功率。

- 布局 dt1_dev_000，A=(2,1.2)、B=(2,-1.2)，机器人原点朝 +X；对应训练视频 01–04。
- slots 0..3：A红B蓝去红/去蓝，A蓝B红去红/去蓝；固定 policy seed 20260906，配对重置规则与采集相同，不声称初态逐像素完全一致。
- 原闭环模型/RGB/实测速度/语言输入、chunk5/execute1、5Hz ZOH、控制边界和停车评分均保持不变。
- 独立 `scripts/zoh_train_scene_runtime.py` 与 `scripts/zoh_train_scene_queue.py`，不改仍在运行的验证源码；plan 明确 evaluation_split=train、scope=seen-training-layout-fit-check，并绑定新脚本哈希。
- 首先等待验证父 PID2089283 退出，再进入 GPU 准入。7GiB准入、1GiB运行保护、自有进程组、1800s单条上限；普通任务失败继续四条配对，资源/基础设施异常停止，不自动重试。
- 输出 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_train_scene_5k_0906_v1`；父日志 `/tmp/go2_zoh_train_scene_5k_0906_v1_queue.log`。完成四条后停下，按选目标/到达/停稳分别检查，再决定后续，不扩到全部16条。
- 入口导入检查通过，沿用的模型合同/结果/评分11项回归通过。实际运行状态以 queue_status.json 为准，勿重复启动或改绑定源码。
