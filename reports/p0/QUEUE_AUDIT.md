# P0-T1 旧队列与历史进程核查

生成时间：2026-09-12T19:47:25.223459+08:00

1. **PID 779548**：未找到。`action: no_action_required`；不能表述为“已停止”。
2. **项目采集/训练/评测进程**：按 `collect_r2r|zoh_|navila|isaac|train_|evaluate` 检索未发现项目自有运行进程（仅审计命令自身被匹配并排除）。
3. **GPU**：`nvidia-smi` 查询失败（无法与 NVIDIA driver 通信），GPU 占用者及是否存在非项目进程记为 **MISSING**，未作空闲推断。
4. **官方队列产物**：`progress.json`、`plan.json`、`results.json` 均存在并已记录 SHA-256；progress 为 `completed=4` / `planned=1077`，plan episode 条数为 `1077`，results 条数为 `4`，episode 目录数为 `4`。差值见 JSON；状态为不完整/不自洽（progress FAILED_NEEDS_REVIEW，complete=false）。
5. **是否需要停止进程**：当前无可识别的项目进程，`NO_ACTION_REQUIRED`；不执行任何停止操作。

缺失：`/tmp/official_navila_velocity_0911_full_v1.log` 不存在；GPU 事实因驱动查询失败无法取得。

三份 JSON SHA-256：
- progress.json: `30b561d2dc8d1b94ba528e36d279e7777a1ac0c9e98aafb52f55848b8d8205df`
- plan.json: `5223ab9d830fd634cf78187e0951f752610558f66cee9469634655a8a2e02373`
- results.json: `1e295ee2829096b0270b02bedf7f471d3dcddb6b41091cf2deb15d6510286ec8`
