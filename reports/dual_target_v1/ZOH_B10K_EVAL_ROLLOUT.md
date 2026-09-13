# B fresh-10k validation rollout执行记录

固定B scheduler＋expert-only `checkpoint_010000`，在全部8个冻结validation任务（slots16–23）执行真实Isaac闭环。policy seed为20260906；不训练、不使用独立test，validation结果不回流优化或checkpoint选择。

运行保持5 Hz重新观察、chunk5但每次只执行首动作，50 Hz低层控制，以及冻结的限幅、变化率、碰撞和连续自主停稳标准。每条保留原始50 Hz轨迹、模型chunk、配对reset、独立轨迹复算与视频；末尾不足0.2秒的动作段按实际执行帧保留。

沿用用户对相同B rollout栈批准的7 GiB准入/1 GiB运行保护。preflight通过；队列PID 2301345，slot16 PGID 2301523、模型worker 2302167 已产生动作。启动阶段自有峰值5826 MiB、最低free 2377 MiB。输出为 `outputs/dual_target_v2/b10k_eval_rollout_0907_v1`；8条完成后停止。

## 最终结果

8/8完成：4成功、3次停错目标、1次超时、0碰撞；进入正确目标区域5/8，进入错误目标区域3/8，末距均值1.2253 m。slot16停错，17/18成功，19停错，20/21成功，22超时，23停错。最终报告、结果JSON、逐条视频和artifact hash齐全，所有自有进程已退出。该单seed validation结果不等同独立test结论。
