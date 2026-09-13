# 超时结果兼容修复与验证续跑

原循环的 timeout 分支缺少 steps 字段，队列审计直接索引导致 KeyError。仿真已完整运行1500低层步/150高层命令，无评分invalid，无碰撞；模型未进入正确/错误停车区域，末尾距正确停车中心0.45503m，真实结果为FAILED_TIMEOUT。

## 修复范围

- 新 `zoh_result_contract.normalized_result`：仅对原结果是FAILED_TIMEOUT、pre/post完整且均1500步的旧记录，由证据派生steps并明确标注来源；不改原JSON。不完整记录、非超时缺步数、已声明步数不一致均拒绝。
- 新 `zoh_eval_review_v2` 接纳上述兼容结果，并拒绝任何包含INVALID_SAMPLE的轨迹作为正常模型失败归档。
- 新 `zoh_result_loop` 在所有正常退出结果补充实际ZOH计数，不改指令、控制、评分或图像时序。
- 旧 `zoh_val_scored_0906_v1` 原状态、结果、轨迹及源码保持不变。重审通过后，新队列 `recovered_first_result.json` 记录原stage、plan与三份轨迹文件哈希，以及首条真实FAILED_TIMEOUT，不转成功、不重跑。

本地/远端11项测试通过；现有真实首条独立审计通过结构一致性检查（任务结果仍失败），确认1500低层步、150高层命令、无partial尾段。

## 续跑授权

用户要求“补齐并继续”，因此从原32任务矩阵第2条继续，资源可运行与模型导航成功不再混为一个准入条件。保留7GiB准入、1GiB运行保护、1800秒单条上限。正常任务超时/碰撞等失败计入结果继续；资源/进程/时序/文件异常仍停止，无自动重试。模型验证仍仅validation，不使用test、不追加训练。

续跑根 `/mnt/wxh/go2_short_vln/outputs/dual_target_v2/zoh_val_resume_0906_v1`，日志 `/tmp/go2_zoh_val_resume_0906_v1_queue.log`。新plan包含首条旧run的明确引用与其余31条新run，检查checkpoint/slot/seed与旧矩阵一致，再冻结源码。
