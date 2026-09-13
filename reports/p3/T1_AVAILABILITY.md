# P3-T1 可获取性与 Scene 覆盖（retry 2）

## 结论

两个 Hugging Face 数据集端点均可访问。`a8cheng/NaVILA-Dataset` 的 R2R 标注和训练归档可定位，但 R2R `annotations.json` 为 317,267,756 字节，按派单上限未下载。`ML-GOD/navila_r2r` 的 `meta/info.json` 已获取，显示 101 episodes、5,666 frames、10 Hz、Parquet/LeRobot 格式和 512×512×3 图像字段。

Scene 覆盖与 11 个 VLN-CE-Isaac 场景的交集为 **UNVERIFIABLE**。本次没有获得 R2R 每 episode 的 scene id，因此不能声称无 leakage。

## 托管、发现与许可

- a8cheng: https://huggingface.co/datasets/a8cheng/NaVILA-Dataset
- ML-GOD: https://huggingface.co/datasets/ML-GOD/navila_r2r

定位依据是用户指定的 API/tree/resolve 请求、HF API 返回的 dataset card/tree，以及本地 `third_party/NaVILA-Bench/README.md` 第 12 行的 NaVILA Hugging Face collection 链接。ML-GOD API card 明确为 Apache-2.0。a8cheng API/tree 没有返回 dataset license；本地 benchmark README 的 MIT 是 benchmark 代码许可证，不能推定为数据集许可证。

## 体积与结构

a8cheng API 报 `usedStorage=46,160,544,850` bytes；R2R `annotations.json` 317,267,756 bytes，`train.tar.gz` 19,703,787,520 bytes；RxR 对应 1,458,888,437 和 23,570,935,028 bytes。目录包含 `R2R/annotations.json`、`R2R/train.tar.gz`、`RxR/...`、`EnvDrop/...`、`Human/...` 与 `ScanQA/...`。

ML-GOD API 报 used storage 1,497,184,017 bytes，目录为 `data/chunk-000/episode_*.parquet` 和 `meta/{info,episodes,episodes_stats,tasks}` 文件。

## 格式与字段

a8cheng R2R 标注格式只能确认是 JSON 文件名和大小；内容字段未读取。ML-GOD `meta/info.json` 给出的 per-sample 字段是：`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`、`observation.image`（512×512×3）、`observation.state`（4 float32）、`observation.dummy_state`（4 float32）、`action`（4 float32）。

## Scene 与 leakage

本次使用派单给定的 11 个 scene id。已获取的 R2R tree 只显示文件级元数据，没有 episode scene identity；ML-GOD info 也只有 tensor schema。故 scene id 列表不可得，交集标记 **UNVERIFIABLE**，不是 NO-LEAKAGE。完整证据与每次成功 fetch 后的增量记录在 `t1_availability.json`。
