# P3-T3 — 相机参数 + Embodiment gap 量化

生成时间：2026-09-12 20:35:59 +08:00  
机器可读证据：`reports/p3/t3_camera_embodiment.json`

## 结论先行

Go2 车载/策略相机的配置和已采帧均可核验：策略使用的 `rgbd_camera` 安装在 `Robot/base` 前方 0.1 m、上方 0.5 m，名义光轴俯仰为 0°，由配置和实际内参得到水平/垂直 FOV 96.7329°。两次指定 rollout 的 441 个 JPEG 全部解码为 `uint8 [512, 512, 3]`，与契约 §3.1 一致，没有发现实际分辨率不匹配。

NaVILA 独立渲染相机（训练/源数据侧）的高度、FOV、分辨率及帧在本地不可得；HuggingFace 和 GitHub 的短时网络探测也均连接失败。因此本任务不提供 NaVILA 数值、差值数字或伪造比较图，跨视角/高度 gap 的最终大小标记为 `UNVERIFIABLE`。这不是 Go2 侧缺测：Go2 的实测相机世界高度约 0.815 m，光轴保持近水平，几何地平线在中心列平均约 263.58 px（中心 256 px 下方 7.58 px）。

## 1. 强制外部能力探测

派发单要求的短时网络探测在任务开始即执行：

| URL | 结果 |
|---|---|
| `https://huggingface.co` | `curl` code 7，HTTP 000，5 s 内无法连接 |
| `https://github.com` | `curl` code 7，HTTP 000，5 s 内无法连接 |

失败后未重试，也未下载任何数据。

## 2. Go2 相机配置

证据文件：`/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/config/go2/go2_matterport_base_cfg.py`，配置行 301–307。采集 manifest 的 `camera_intrinsics` 与 `camera_offset` 也与该配置一致。

| 项目 | 核验值 | 配置路径 + 字段 |
|---|---:|---|
| 分辨率 | 512 × 512 px | `scene.rgbd_camera.width`, `scene.rgbd_camera.height`（`go2_matterport_base_cfg.py:305-306`） |
| 水平 aperture | 54.0 mm | `scene.rgbd_camera.spawn.horizontal_aperture`（`:304`） |
| 焦距 | 24.0 cm（继承默认值） | `PinholeCameraCfg.focal_length`，`third_party/IsaacLab/.../sensors_cfg.py:40-43` |
| 水平 FOV | 96.7329° | 由 54.0 mm = 5.4 cm、24.0 cm 代入 `2 atan(aperture/(2f))`；并由 manifest `fx=227.555557`、图宽 512 独立复核 |
| 垂直 FOV | 96.7329°（由 K 推导） | 配置未显式给 `vertical_aperture`；manifest `fy=227.555557` 且图高 512，故同值 |
| 安装位置（相对机体） | (+0.1, 0, +0.5) m | `scene.rgbd_camera.offset.pos`（`:303`），parent 为 `scene.rgbd_camera.prim_path={ENV_REGEX_NS}/Robot/base/rgbd_camera` |
| 安装高度 | +0.5 m above `Robot/base` | `scene.rgbd_camera.offset.pos[2]`（`:303`） |
| 姿态/俯仰 | 四元数 (-0.5, +0.5, -0.5, +0.5)，名义 pitch 0° | `scene.rgbd_camera.offset.rot`（`:303`）；`offset.convention` 未写出，`CameraCfg.OffsetCfg` 默认 `ros`（`camera_cfg.py:27-35`）。ROS +Z 光轴映射到 parent +X，故相对机体前向无仰俯 |

`viz_rgb_camera` 是另一只可视化相机（配置行 309–315，aperture 100 mm、offset z 0.8 m），不是 manifest 中 `camera_intrinsics=227.555557` 的策略 RGB 来源；本报告只把采集的 `rgbd_camera` 作为 Go2 车载相机。

相机坐标语义由本地 IsaacLab 源码确认：ROS 相机为 +Z forward、−Y up（`camera_data.py:72-80`），传感器生成时把 offset 转成 OpenGL（`camera.py:114-123`）。

## 3. 实际落盘 RGB 核验（契约 §3.1）

使用 `/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python`，PIL 解码每个 JPEG，再用 numpy 检查 `shape` 和 `dtype`：

| 目录 | JPEG 数 | 解码结果 | 解码错误 | 是否匹配 `uint8 [512,512,3]` |
|---|---:|---|---:|---|
| `/mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_ep1_v3/rgb/` | 236 | `[512, 512, 3]`, `uint8`（236/236） | 0 | 是 |
| `/mnt/wxh/go2_short_vln/outputs/r2r_ablation/20260910_ep1_continuous_v2/rgb/` | 205 | `[512, 512, 3]`, `uint8`（205/205） | 0 | 是 |
| 合计 | 441 | 全部一致 | 0 | 是 |

JPEG 是压缩存储；表中的 `uint8` 是按契约读取后的 RGB 数组类型，而非声称 JPEG 文件本身是未压缩数组。

## 4. Go2 帧的经验视角/高度刻画

输入为两份 `frames.jsonl` 的 `camera_position_w`、`camera_quaternion_ros_wxyz` 以及各自 manifest 的 K；没有启动 Isaac，也没有加载模型。

| 统计量 | `ep1_v3`（236 帧） | `continuous_v2`（205 帧） | 合并保存帧（441，含重叠轨迹） |
|---|---:|---:|---:|
| 相机世界 Z 高度 min–max (m) | 0.805232–0.833156 | 0.805054–0.833156 | 0.805054–0.833156 |
| 相机世界 Z 平均 (m) | 0.815057 | 0.815197 | 0.815122 |
| 光轴世界仰角 min–max (°) | −2.145688–+4.218963 | −2.145688–+4.055028 | −2.145688–+4.218963 |
| 光轴世界仰角平均 (°) | +1.828678 | +1.997298 | +1.907062 |
| 几何地平线 y（x=256）平均 (px) | 263.267994 | 263.938772 | 263.579807 |
| 几何地平线 y（x=256）min–max (px) | 247.474210–272.788005 | 247.474210–272.133867 | 247.474210–272.788005 |

这里的“几何地平线”是用保存的 ROS 相机姿态，将世界水平射线平面投影到图像得到的 y 坐标；Matterport 室内 RGB 不保证存在可见的天空/语义地平线线条。相对图像中心 256 px，合并帧平均下移 7.58 px，完整范围 25.31 px，反映步态/机体姿态扰动而不是另一个相机的差值。

## 5. NaVILA 侧与逐项差值

本地检查了 NaVILA-Bench 的任务/config 和 `scripts/navila_eval.py`。其中 `navila_eval.py` 选择的是 Go2 `go2_matterport_vision` 任务并读取其 `camera_obs`；脚本中的 `sim.set_camera_view` 只是 viewer 相机，不是 NaVILA 训练源数据的渲染参数。允许的本地目录没有独立的 NaVILA 源帧/相机 metadata，网络又不可用，故 NaVILA 侧三项均为 `MISSING`。

| 项目 | Go2 | NaVILA render | 差值 |
|---|---|---|---|
| 高度 | offset +0.500 m（实际 world-z 平均 0.815122 m） | `MISSING` | `UNVERIFIABLE` |
| 水平 FOV | 96.7329° | `MISSING` | `UNVERIFIABLE` |
| 垂直 FOV | 96.7329° | `MISSING` | `UNVERIFIABLE` |
| 宽度 | 512 px（全量解码确认） | `MISSING` | `UNVERIFIABLE` |
| 高度（像素） | 512 px（全量解码确认） | `MISSING` | `UNVERIFIABLE` |
| 宽高比 | 1.000 | `MISSING` | `UNVERIFIABLE` |
| 名义 pitch / viewpoint | 0°；实测光轴仰角均值 +1.907° | `MISSING` | `UNVERIFIABLE` |

不把 Go2 配置本身重复当作 NaVILA render 参数，也不对缺失侧猜测高度、FOV 或分辨率。

## 6. Embodiment gap 裁决

状态：**UNVERIFIABLE**。NaVILA 侧没有可核验的 render camera 或帧，因此不能给出跨侧高度/FOV/分辨率差，也不能声称已有的 Go2 帧是“同场景 NaVILA-vs-Go2”对比。

证据支持的有限结论是：Go2 的部署视角确实是低安装（相对 `Robot/base` +0.5 m，保存 world-z 约 0.815 m）且近水平（仰角范围 −2.15° 到 +4.22°）。如果未来取得的 NaVILA 源渲染相机明显更高或俯仰不同，这种 viewpoint/height 差有造成 distribution shift 的合理机制；但在当前证据下，**是否“大到足以造成 shift”及其数值阈值均不可验证**。

未生成比较图。派发单指定的 `/mnt/wxh/go2_short_vln/downloads/navila_probe/t3_figs/` 当时不存在，沙箱拒绝创建该目录（`Read-only file system`）；没有把图件写到别处来冒充指定产物，也没有伪造 NaVILA 对比图。
