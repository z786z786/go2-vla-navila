# DT2 最终验收：APPROVED

日期：2026-09-06；实施与验收均为主代理，使用另写脚本重算原始证据，不称为独立代理验收。DT2通过仅证明tiny专家数据可用，不是SmolVLA导航成功。

## 数据与原始行为

固定4个train几何组×2种颜色配置×2种指令，共16条、7,392个真实训练帧。两个DT1组固定复用repeat=0的8条，两个新增组预先声明的8条全部一次成功，无逐slot重试、无挑选最好结果。

批次`dt2_tiny_0906_v2`，计划SHA256 `a92b33720f250f86dc0c35a45fc794ddc5b551b4a7b250e7a73635a57dbc953d`。最终数据目录：`/mnt/wxh/go2_short_vln/outputs/dual_target_v1/dt2_tiny_dataset_0906_v1/`；dataset_manifest SHA256 `d6d58876dd35eb655723c05e69f7e5d12476b9053f5c6e161445cfd257b07606`。

16条原始pre/post时序、位姿重算停车区域、碰撞/终止条件及评分全部通过；每条连续实际停稳1.0199999772秒，另有51个真实零命令停稳pre-action尾帧（名义跨度1秒，float32物理钟漂移容差1e-6秒）。没有复制或padding补停稳段。8对复位通过已获用户批准的task_equivalence_v2，不要求跨运行RGB/关节逐字节相同。

## 图像、转换和训练统计

实际检查7,408张图像（7,392个pre-action输入加16张最终post图），16个前视RGB＋审计轨迹视频可回读，初始两目标与颜色交换正确，所选目标没有完全离开画面。右侧俯视图是GT轨迹重建，不是第二台Isaac相机，也不进入策略输入。

主代理另行查看新增场景4种配置首帧、全部8条最小目标可见面积帧及4条末端原图，共16张未经修改的RGB。新增组最小目标面积仍能辨认。保留DT1已知限制：部分旧组中途只有图像边缘窄条（最小2,064颜色像素），可见不等于始终完整/居中；近距离目标底部可超出画幅。

LeRobot采用无损image存储；只含front RGB、3维body velocity、3维action及task/框架索引，无GT字段。动作标签是当前观察之后实际执行的命令，50步chunk只在本episode内查询。转换器检查144个anchor；主代理另写脚本再检查144个不同分布的首中尾/边界anchor，RGB逐像素一致、state正确、动作与padding mask正确。

全7392帧fresh train normalizer经NumPy重新计算一致；vy mean=0/raw_std=0、安全std=1。normalizer SHA256 `75832fc724a78287100f9feb0fc7d02bfbe2a2efcde67080775aa72b3cc10777`。训练必须使用该统计，损失排除vy及所有padding。

默认episode均匀、episode内anchor均匀、有放回，权重逐条重算且和为1。保留全部真实帧，无旧terminal cap：terminal anchor曝光11.13%、near-zero/all-valid-chunk-near-zero18.55%、any-valid-chunk-near-zero29.24%、含padding的anchor10.69%。这些是采样曝光率，不是模型成功率。

## 资源、代码及清理

用户明确授权DT2共享GPU：实际free≥8 GiB准入，3×30秒及持锁复查，运行free保护线2 GiB。1,590条外部采样中，本次进程峰值4,380 MiB，最低实际free13,073 MiB。外部PID1866676保留且未发送信号。

采集父进程1873855及8个隔离PGID均退出，实际项目锁可获取。旧v1独占等待已取消且无episode启动，取消记录保留。DT1合同SHA `ae040c5b66c7405197874d9d24408f03736606704c5d3cd278ad5827af5e0e14`不变；只通过记录过的两行runner场景参数兼容扩展用于DT2，采集期间源码冻结。训练新增代码是在最后一条采集结束后部署。

最终82项回归：本地78通过/4依赖跳过；远端SmolVLA环境82通过、无跳过。真实训练候选CPU预检确认三维处理器、mask、保存重载及FP32可训练参数，未提前训练。

## 验收证据与下一步

- 数值重算：`dt2_root_dataset_review.json`，SHA `709d2238001c9310a7da788d338ab1e55b374607688ef3fa3cc60ae9f521e7cc`。
- 图像审计：`dt2_tiny_0906_v2_visual/visual_review.json`，SHA `b754bbe042368a73d52c62946f4a805b9d181fd9c0ba1df07cdf586a4ac57f56`。
- 进程/锁：`remote_runs/dt2_tiny_0906_v2/root_cleanup_review.json`，SHA `9b8b112398367ab7036c7b50e19348c0e84d45c5f86d73058230b4ab827e2790`。
- 原始尝试、完整数据、源码快照均保留远端；小型证据和视频已镜像本地。

根据用户最新授权，DT2通过后直接进入DT3：先资源试测与单batch更新/重载门槛，再启动250更新pilot。effective batch=16、LR起始1e-4、累计预算≤5000不扩大；后续档位须闭环诊断支持，不自动开始DT4。
