# B fresh-10k train rollout 执行记录

用户要求先暂停尚未开始的 C 全量微调，运行 B scheduler＋expert-only 的 train rollout。C等待器已停止在0 step；本轮固定 B `checkpoint_010000`，不追加训练。

- 任务：全部16个train slots 0–15，固定policy seed 20260906。
- 禁止：validation、test、基于结果选任务或重试刷结果。
- 每条最长1500个50 Hz物理步；模型每0.2秒重新观察并只执行chunk首动作。
- 评分：沿用冻结的正确目标、错误目标、碰撞和连续自主停稳标准。
- 证据：原始pre/post轨迹、model chunks、配对reset、独立轨迹重算、逐条MP4。
- 视频：保留初始帧和每个实际执行后帧；终止时不足0.2秒的partial hold按实际时长保留。
- 资源：9 GiB连续三次准入、2 GiB运行保护；只终止本队列拥有的进程组。

远端 preflight 已验证 B-10k训练结果、最终reload门、sample binding及checkpoint全文件哈希；checkpoint manifest SHA256 为 `cbb97d891b2d1ff1aec34232de409df044eefd02d82c800ca91ae0d0d0a5c582`。队列 PID 2264616，输出 `outputs/dual_target_v2/b10k_train_rollout_0907_v1`。启动时GPU空闲7942 MiB，故slot0正在等待9216 MiB准入，尚无模型结果。

## 低显存单次例外

用户随后明确要求降低门槛强行启动。v1在0/16且未创建Isaac时停止并保留；独立v2只把准入改为7 GiB、运行保护改为1 GiB。v2队列PID 2265826，输出 `outputs/dual_target_v2/b10k_train_rollout_0907_v2`，slot0 PGID 2266001、模型worker 2266647 已产生真实动作。启动后实测自有峰值5826 MiB、最低free 2070 MiB，当前未越过1 GiB保护线。该资源例外不推广到C训练或其他任务。
