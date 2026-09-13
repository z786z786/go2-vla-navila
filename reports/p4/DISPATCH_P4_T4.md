# 派发单 P4-T4 — NaVILA 帧属性普查 + habitat↔Isaac 渲染差距量化

授权：用户 2026-09-13 `CONTINUE P4′`。波次：W2（P4-T1 已完成，帧已在磁盘上）。
本任务旨在收窄 **OPEN #5「NaVILA 渲染侧相机参数 MISSING」**，该项自 P3 起标为 `UNVERIFIABLE`。

| 字段 | 内容 |
|---|---|
| 任务 ID | **P4-T4** |
| 资源 | **CPU only。零 GPU，零网络。** |
| Python 环境 | 系统 `python3` **无** PIL/numpy/cv2。请用 `/mnt/wxh/go2_short_vln/envs/conda/smolvla/bin/python`（PIL / numpy / cv2 均可用）。脚本须在文件头注明所用解释器 |

---

## 1. 为什么这次能做，而 P3 当时不能

P3 §5 把 embodiment gap 记为 `UNVERIFIABLE`，理由是**当时没有取到任何 NaVILA 帧**，并明确拒绝伪造对比图。
现在两侧素材都在本地：

| 侧 | 路径 | 规模 |
|---|---|---|
| NaVILA（habitat 渲染，训练数据） | `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/train/<episode_id>/frame_<i>.jpg` | **601,125** 帧 / 10,819 episode |
| Go2（Isaac 渲染，本项目实采） | `/mnt/wxh/go2_short_vln/outputs/r2r_ablation/<run_dir>/rgb/*.jpg` | **24,277** 帧 / 62 episode |

**关键事实（主代理已实测）**：这 62 个 Go2 rollout 的 `episode_id` **全部**也存在于 NaVILA 训练帧目录，
覆盖 **57 / 61** 个 R2R train scene。即存在 62 组「**同一 episode、同一场景、同一起点，两套渲染管线**」的配对。
这是本任务的核心素材，不要只做单侧统计。

---

## 2. 已知的 Go2 侧相机参数（勿重新推定，直接引用）

源：`/mnt/wxh/go2_short_vln/third_party/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/config/go2/go2_matterport_base_cfg.py:301-307`
（SHA-256 `a9bfa20ede39d6c837e64420353e608572260a336e60b6cf9ff8864fbb4366b7`，NaVILA-Bench 自带未改动配置）

| 项 | 值 |
|---|---|
| 分辨率 | 512 × 512 |
| HFOV | **96.7329°**（`horizontal_aperture=54.0`，PinholeCameraCfg 默认 `focal_length=24.0`） |
| 相对 `Robot/base` 位姿 | `pos=(0.1, 0.0, 0.5)`，`rot=(-0.5, 0.5, -0.5, 0.5)` |
| 实测世界高度 | 0.805–0.833 m，均值 **0.815 m** |
| 实测光轴仰角 | **−2.15° ~ +4.22°**（名义 pitch 0°） |

**NaVILA 渲染侧的相机参数是未知量，正是本任务要界定的对象。**
注意：NaVILA-Bench 仓库内**不含**训练数据的渲染配置（主代理已搜索确认），所以不要指望从该仓库读出答案；
只能从图像本身推断，或明确报告无法推断。

---

## 3. 任务

### 3.1 全量帧属性普查（NaVILA 侧 601,125 帧）

逐帧读取**头部元数据即可**（不必解码全图，用 PIL 的惰性 `Image.open` + `.size` / `.mode`）：

1. 分辨率分布（主代理抽 4 张均为 512×512，须全量验证是否有例外，例外逐个列出）
2. 色彩模式分布（RGB / L / CMYK 等）
3. 文件大小分位（min / p25 / p50 / p75 / p90 / max）
4. **损坏或不可解码的文件清单**（用 `Image.verify()` 或解码首块；若全量解码过慢，可对全量做 `verify()`、
   仅对抽样做完整解码，并注明覆盖率）

### 3.2 配对对比（62 组，本任务的核心）

对全部 62 个配对 episode，产出**可复算的定量对比**。至少覆盖：

1. **视野范围差异**：两侧渲染同一场景。给出任一可量化的视野/尺度差指标及其计算方法。
2. **地平线/光轴指向**：估计 NaVILA 帧的地平线所在像素行，与 Go2 帧对比。
   Go2 侧名义 pitch 0°（实测 −2.15°~+4.22°），可作为参考基准。
3. **相机高度的界定**：尝试给出 NaVILA 相机高度的估计或**区间上下界**。
   R2R-CE 的常规 agent 相机高度约 1.25 m，而 Go2 实测 0.815 m——若你的估计支持或推翻这个量级，明确说明依据。
   **若无法给出有依据的估计，就写 `UNVERIFIABLE` 并说明为何不能**，不得给出无支撑的数字。
4. **低阶图像统计**：亮度、对比度、饱和度、边缘密度的分布对比（两侧各自聚合，并做配对差）。

### 3.3 视觉证据

为至少 **8 组**配对生成并排对比图（NaVILA 帧 | Go2 帧），选取规则须写死可复现。
存为 PNG 到 `/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/index/t4_pairs/`（大产物放 `/mnt`）。
**每张图须标注 episode_id、scene、两侧帧索引。**

---

## 4. 输出

| 路径 | 内容 | 备注 |
|---|---|---|
| `reports/p4/t4_frame_survey.json` | §3.1 全量普查结果 | 小，入库 |
| `reports/p4/t4_render_gap.json` | §3.2 配对对比的全部定量结果 + 方法说明 | 小，入库 |
| `reports/p4/T4_RENDER_GAP.md` | 人读摘要：结论、方法、**明确列出仍 `UNVERIFIABLE` 的部分** | 入库 |
| `scripts/p4_frame_survey_and_gap.py` | 可复算脚本，**无参数重跑且确定性** | 入库 |
| `/mnt/.../R2R/index/t4_pairs/*.png` | §3.3 对比图 | 大产物，不入库 |

每份 JSON 必含 `task_id`、`generated_at`、`generator_command`、`inputs`（路径 + 实测 sha256 或目录指纹）、`missing`。

---

## 5. 验收标准（主代理如何独立复算）

1. 主代理独立重算 NaVILA 帧总数（应为 601,125）与分辨率分布，与 `t4_frame_survey.json` 一致。
2. 主代理独立重建 62 组配对清单（按 `episode_id` 交集），与报告的配对集合**逐条一致**。
3. 主代理对报告中的每个定量指标，抽 3 组配对按报告声明的方法重算，须落在声明的容差内。
4. 主代理核对每个「估计值」都有方法与依据；**无依据的数字一律判 FAIL**。
5. 主代理核对 `UNVERIFIABLE` 项均有「为何不能」的说明，而非空置。
6. 主代理无参数重跑脚本，关键字段一致（确定性）。
7. 主代理打开至少 2 张对比图确认标注完整、左右两侧未混淆。

**子代理的 `passed` 字段与自我结论不构成 PASS 依据。**

---

## 6. 禁止事项

1. **不得伪造或"合理推测"任何相机参数**。P3 在无素材时宁可标 `UNVERIFIABLE` 也不编造，本任务延续该标准。
   能测则测，不能测则明写不能测及原因。
2. 不得修改 NaVILA 帧或 Go2 帧本身（只读）。
3. 不得删除或改动 `/mnt/.../R2R/train/`、`/mnt/.../R2R/index/t2_*`、`/mnt/.../outputs/r2r_ablation/` 下任何既有文件。
4. 不得动 `reports/p3/`、`reports/p4/t1_*`、`reports/p4/t2_*`、`reports/p4/t3_*`、
   `CODEX_VLN_PLATFORM_MILESTONES.md`、`configs/`、`.gitignore`、`actions/`、`tests/`。
5. 不得 `git add` / `git commit` / `git tag`。
6. 不得联网，不得占 GPU。
7. 大产物（PNG）不得写入 `/home`；根分区仅余约 27 GB。
8. 证据缺失标记 `MISSING`，不得自行追认 PASS。

---

## 7. 验收台账

| 轮次 | 时间 | 结果 | 备注 |
|---|---|---|---|
| 1 | 2026-09-13 | **PASS** | 七项独立复算全通过：601,125 帧全 512×512 RGB / 0 损坏、62 组配对逐条一致、15 项指标吻合到小数点后 5 位、4 项 UNVERIFIABLE 均有「为何不能」、重跑确定性、2 张对比图人工核对。详见 `CODEX_VLN_PLATFORM_MILESTONES.md` §v3.15 |

**主代理新增限定（报告低估之处）**：配对帧**并非位姿对齐**——Go2 首帧在 `sim_time_s=2.0`（warmup 后），
两侧帧密度差 5.5 倍，轨迹分别来自 GT 路径与 RouteExpert 实走路径。报告称「起点微偏移」低估了此项。
低阶统计因此混合渲染、视角、相机参数三者，引用时须同时引用该限定。详见 §v3.15-C。

**主代理自身更正 1 条**：非空 RGB 目录数主代理算 65、报告 67；差额为两个 `contact_probe` 目录（名中无 `ep<数字>`）。报告正确。
