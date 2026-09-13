# 数据与验证子笔记

## 用途

这份子笔记专门记录：

- 数据合同
- previous_action 路径
- manifest / weighted / targeted-v2
- policy validation 设计
- baseline / ablation / slice 指标

当以下内容变化时，优先更新本文件：

- 数据 schema
- dataset / collator / prompt 数据流
- previous_action 重建 / dropout / fallback 规则
- validation baseline 定义
- normalized metrics / comparison / manifest 构造逻辑

## 建议结构

### 数据合同

- 原始 collector 合同
- processed dataset 合同
- robotics lane 推荐合同

### previous_action 路径

- top-level 来源
- legacy fallback
- prompt condition
- anchor builder

### 数据集分支

- `current_v1`
- `current_v1_real_only`
- `current_v1_sim_only`
- `current_v2_sim_today_main_turnpilot_20260425`
- weighted / targeted-v2 子集

### 验证设计

- `eval_velocity.py`
- `stage1_5_policy_validation.py`
- baseline registry
- normalized metrics
- targeted-v2 manifest

## 当前状态

- 当前主内容仍在 `docs/llada_vla_go2_project_notes.md`。
- 后续如果数据与验证部分继续膨胀，应把细节迁到本文件。
