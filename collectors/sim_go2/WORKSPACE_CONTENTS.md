# 当前工作目录内容说明

生成时间：2026-05-04

工作目录：`/home/wxh/llada-vla-go2/collectors/sim_go2`

## 目录用途概览

这是一个用于 Go2 模拟数据采集的代码目录，主要承载 Isaac Sim / Isaac Lab 场景下的原始轨迹采集、回放、打包、校验与统计分析逻辑。

根据 [README.md](/home/wxh/llada-vla-go2/collectors/sim_go2/README.md)，该目录的核心目标是维护 sim raw-data collection 代码，并让输出数据契约尽量与真实机器人采集链路保持兼容。

## 根目录顶层内容

### 隐藏目录

- `.agents`
  代理或编排相关目录。

- `.codex`
  Codex 本地代理/技能/运行时相关目录。

- `.git`
  Git 仓库元数据目录。

- `.omx`
  oh-my-codex 运行状态目录。
  当前可见子目录：
  `logs`
  `state`

### 文档与包标记

- `README.md`
  当前模块说明文档，介绍 sim 采集代码的定位、入口脚本、环境变量提示、depth probe 工作流和 dual-target contrast 工作流。

- `__init__.py`
  Python 包初始化文件。

- `__pycache__/`
  Python 编译缓存目录。
  当前可见内容：
  `__init__.cpython-310.pyc`
  `__init__.cpython-311.pyc`
  `__init__.cpython-313.pyc`

## 业务代码目录

### `backends/`

用途：后端动作或运动接口封装。

当前可见内容：
`__init__.py`
`motion_backends.py`
`__pycache__/`

### `configs/`

用途：采集任务、场景、随机化、打包、相机和机器人参数配置。

当前可见配置文件：
`collection_base_nav.yaml`
`collection_box_curriculum.yaml`
`collection_box_curriculum_depth_probe.yaml`
`collection_door_curriculum.yaml`
`collection_door_curriculum_depth_probe.yaml`
`collection_dual_target_contrast_conference.yaml`
`collection_dual_target_contrast_full.yaml`
`collection_dual_target_contrast_main.yaml`
`collection_dual_target_contrast_room.yaml`
`collection_dual_target_contrast_room_local.yaml`
`collection_dual_target_contrast_room_local_smoke.yaml`
`collection_dual_target_contrast_room_local_turn_pilot.yaml`
`collection_full.yaml`
`collection_semantic_variation.yaml`
`collection_turn_heavy.yaml`
`domain_randomization.yaml`
`packing.yaml`
`robot_go2_camera.yaml`
`scene_local_office_room.yaml`
`scene_office.yaml`
`scene_office_conference.yaml`
`scene_office_full.yaml`
`scene_office_main.yaml`
`scene_office_room.yaml`
`scene_plane.yaml`
`scene_warehouse_forklifts.yaml`
`scene_warehouse_shelves.yaml`
`scene_warehouse_standard.yaml`

### `controllers/`

用途：控制器逻辑。

当前可见内容：
`__init__.py`
`heuristic_nav_controller.py`
`__pycache__/`

### `envs/`

用途：环境、场景构建和任务生成。

当前可见内容：
`__init__.py`
`go2_nav_env.py`
`scene_builder.py`
`task_generator.py`
`__pycache__/`

### `outputs/`

用途：运行输出目录。

当前可见内容：
`asset_cache/`

### `packers/`

用途：数据集打包逻辑。

当前可见内容：
`__init__.py`
`dataset_packer.py`

### `recorders/`

用途：采集过程中的图像、日志和原始 episode 记录。

当前可见内容：
`__init__.py`
`image_writer.py`
`jsonl_writer.py`
`raw_episode_logger.py`
`__pycache__/`

### `scripts/`

用途：主要可执行脚本入口，覆盖采集、回放、统计、校验和可视化。

当前可见内容：
`__init__.py`
`collect_box_curriculum.py`
`collect_dataset.py`
`collect_door_curriculum.py`
`collect_raw_trajectories.py`
`compute_dataset_stats.py`
`compute_depth_stats.py`
`eval_closed_loop_policy.py`
`image_writer.py`
`jsonl_writer.py`
`raw_episode_logger.py`
`replay_episode.py`
`replay_sample.py`
`stats_dataset.py`
`validate_dataset.py`
`visualize_samples.py`
`__pycache__/`

README 中特别提到的核心入口包括：
`collectors/sim_go2/scripts/collect_raw_trajectories.py`

### `utils/`

用途：通用工具函数与数据结构辅助逻辑。

当前可见内容：
`__init__.py`
`action_utils.py`
`io.py`
`randomization.py`
`schema.py`
`state_utils.py`
`target_stop.py`
`visibility.py`
`__pycache__/`

## 从 README 提炼出的重点

- 当前目录服务于 Go2 的模拟原始数据采集。
- 支持 depth probe 工作流，即采集 RGB + depth 原始会话并统计深度分布。
- 支持 dual-target contrast 工作流，即在相同场景布局下对不同目标采集对比数据。
- 需要时会依赖 `ISAACLAB_ROOT` 和 `UNITREE_RL_LAB_ROOT` 等环境变量。
- 输出目标是尽量对齐真实采集链路的数据契约。

## 简短结论

这个目录本质上是一个完整的模拟采集子系统，包含：
配置层、环境层、控制层、记录层、打包层、统计校验脚本，以及运行时输出目录。
