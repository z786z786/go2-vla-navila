# Phase 1 模型输入输出示例（由训练代码路径实际生成）

生成命令：`python scripts/p5_render_example.py`。模板 token 数（不含指令）：**622**，其中图像占位 512。

## 记录 `914-23`（scene `mp3d/8WUmhLawc2A/8WUmhLawc2A.glb`，n_frames=49）

**8 帧历史**（NaVILA 规则 `navila_history_layout(49)` = `[0, 6, 13, 20, 27, 34, 41, 48]`，None=黑帧）：见 `example_frames_914-23.jpg`

**模型看到的完整输入**（每个 `<image>x64` 会被该帧的 64 个视觉 token 替换）：

```text
<|im_start|>User: Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations <fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
, and current observation <fake_token_around_image><global-img><image>x64<fake_token_around_image>
. Your assigned task is: "Turn and walk away from the bathroom into the bedroom area. Walk through the open door on the other side of the room and into the dining room area at the end of the hallway. Stop near the couches in the next room passed the kitchen." Analyze this series of images to decide your next action, which could be turning left or right by a specific degree, moving forward a certain distance, or stop if the task is completed.<end_of_utterance>
Assistant:
```

- prompt 总 token：**673**（指令 51 token，截断=False）
- 监督目标（loss 只算这些）：` The next action is move forward 75 cm.<end_of_utterance>`
- 目标 token ids（12 个）：`[378, 1867, 2353, 314, 1485, 3703, 216, 39, 37, 5973, 30, 49279]`
- 训练序列长度：685

**模型应输出**（贪心解码到 `<end_of_utterance>` 停止）→ parser → 控制命令：

```text
The next action is move forward 75 cm.
```

`NavCommand(vx=0.5, vy=0.0, wz=0.000000, hold_steps=75, stop=False)`，parse 分类 `exact`

## 记录 `7656-1`（scene `mp3d/EDJbREhghzL/EDJbREhghzL.glb`，n_frames=4）

**8 帧历史**（NaVILA 规则 `navila_history_layout(4)` = `[None, None, None, None, 0, 1, 2, 3]`，None=黑帧）：见 `example_frames_7656-1.jpg`

**模型看到的完整输入**（每个 `<image>x64` 会被该帧的 64 个视觉 token 替换）：

```text
<|im_start|>User: Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations <fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
<fake_token_around_image><global-img><image>x64<fake_token_around_image>
, and current observation <fake_token_around_image><global-img><image>x64<fake_token_around_image>
. Your assigned task is: "Go towards the stairs and past them. Veer slightly left at the mirrored door and go down the hall. Wait in the nook with two chairs and the fireplace." Analyze this series of images to decide your next action, which could be turning left or right by a specific degree, moving forward a certain distance, or stop if the task is completed.<end_of_utterance>
Assistant:
```

- prompt 总 token：**656**（指令 34 token，截断=False）
- 监督目标（loss 只算这些）：` The next action is turn left 15 degree.<end_of_utterance>`
- 目标 token ids（12 个）：`[378, 1867, 2353, 314, 1607, 2049, 216, 33, 37, 3815, 30, 49279]`
- 训练序列长度：668

**模型应输出**（贪心解码到 `<end_of_utterance>` 停止）→ parser → 控制命令：

```text
The next action is turn left 15 degree.
```

`NavCommand(vx=0.0, vy=0.0, wz=0.523599, hold_steps=25, stop=False)`，parse 分类 `exact`

