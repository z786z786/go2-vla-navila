# P4-T6 吞吐探针报告

生成时间：2026-09-13。探针脚本：[scripts/p4_throughput_probe.py](../../scripts/p4_throughput_probe.py)。

## 结果摘要

本次按派发单执行了 GPU 准入命令，但 `nvidia-smi` 返回退出码 9（无法连接 NVIDIA 驱动），所以真实 CUDA 吞吐、显存、时钟/功耗和最大 batch 均为 **MISSING**。脚本仍会在有 GPU 的机器上自动等待、调用 `acquire_live_admission`、执行 10 步 warmup 后再计时。

| batch | workers | 在线 step 均值/p50/p95 | 在线显存（torch / nvidia-smi） | 缓存 step 均值/p50/p95 | 缓存显存（torch / nvidia-smi） |
|---:|---:|---|---|---|---|
| 1 | 0, 2 | MISSING | MISSING / MISSING | MISSING | MISSING / MISSING |
| 2 | 0, 2 | MISSING | MISSING / MISSING | MISSING | MISSING / MISSING |
| 4 | 0, 2 | MISSING | MISSING / MISSING | MISSING | MISSING / MISSING |
| 8 | 0, 2 | MISSING | MISSING / MISSING | MISSING | MISSING / MISSING |
| 16 | 0, 2 | MISSING | MISSING / MISSING | MISSING | MISSING / MISSING |
| 32 | 0, 2 | MISSING | MISSING / MISSING | MISSING | MISSING / MISSING |
| 64 | 0, 2 | MISSING | MISSING / MISSING | MISSING | MISSING / MISSING |

最大可用 batch size：**MISSING**。在线编码 vs 预计算缓存 samples/s：**MISSING / MISSING**。`num_workers=0,2` 两档已写入脚本的测量矩阵，但本次无 CUDA，未产生数字。

## 视觉缓存体积

缓存张量形状固定为 `64 × 960`、bf16（2 bytes）：

```
每帧字节数 = 64 × 960 × 2 = 122,880 bytes = 120 KiB
全量字节数 = 601,125 × 122,880 = 73,866,240,000 bytes
全量 GiB = 73,866,240,000 ÷ 1024³ = 68.7933 GiB
```

这是张量表示体积的实测字节计数；视觉编码器实际预计算耗时为 **MISSING**。该缓存仅适用于 phase 1（视觉编码器冻结）；phase 2 视觉侧微调时缓存不再有效。

## 规模推算

报告 JSON 保留代入式。对于实测 samples/s 为 `S_online` 与 `S_cache`：

```
1 epoch 在线 = 353,894 × 1 ÷ S_online ÷ 3,600 小时
2 epoch 在线 = 353,894 × 2 ÷ S_online ÷ 3,600 小时
3 epoch 在线 = 353,894 × 3 ÷ S_online ÷ 3,600 小时
1 epoch 缓存 = 353,894 × 1 ÷ S_cache ÷ 3,600 小时
2 epoch 缓存 = 353,894 × 2 ÷ S_cache ÷ 3,600 小时
3 epoch 缓存 = 353,894 × 3 ÷ S_cache ÷ 3,600 小时
```

本次 `S_online`、`S_cache` 均为 **MISSING**，故六个小时数不能验证，未填入伪造数字；过采样记录按 353,894 行保留、不去重。

## 准入与 warmup 证据

- run_id：`p4_t6_throughput`
- 准入来源字段：`live_nvidia_smi`；策略：`dt1_exclusive_v1`（20,480 MiB、连续 3 次、30 s；未覆盖阈值）。
- 最终状态：`FAILED`，原因是驱动不可用，因此未获得 live lease；锁释放检查：`true`。
- warmup 配置：10 个丢弃 step，计时 10 step；每次计时前后调用 `torch.cuda.synchronize()`。本次 GPU 时钟/功耗采样序列：**MISSING**（无驱动）。

## 交付与 MISSING

交付文件：

- `datasets/navila_r2r.py`
- `scripts/p4_throughput_probe.py`
- `reports/p4/t6_throughput.json`
- `reports/p4/T6_THROUGHPUT.md`
- `tests/datasets/test_navila_r2r.py`

逐条 MISSING：真实 GPU step 时间/显存、最大可用 batch、在线与缓存 samples/s、GPU 利用率/时钟/功耗序列、视觉缓存预计算墙钟时间、基于实测 samples/s 的 epoch 小时数。CPU Dataset 单测已在 smolvla 环境直接运行通过；pytest 命令本机不可用（环境未安装 pytest）。
