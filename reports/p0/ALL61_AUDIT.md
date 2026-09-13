# P0-T4 all61 多场景结果追溯

- 生成时间：`2026-09-12T19:43:31.400386+08:00`
- 证据根：`/mnt/wxh/go2_short_vln/outputs/r2r_ablation`
- 生成命令：`python3 - <<'PY' (batched read-only pathlib audit of /mnt/wxh/go2_short_vln/outputs/r2r_ablation; writes reports/p0/all61_audit.json and reports/p0/ALL61_AUDIT.md) PY`

## 结论摘要

实测状态分布（由 60 份 `20260910_all61_ep*/summary.json` 直接读取）：**success 34 / timeout 18 / failed 6 / environment termination 2**。表内计数合计 60。

这批数据**不构成经独立 QC 的 61-scene success rate**：scene probe 的 61-scene 口径是 60 个可执行计划项加上不可用场景 `XcA2TqTSSAj`；当前只有 60 个 episode 结果，且 6 个结果在 warmup 阶段失败。

## 原始任务清单与实际产出

`20260910_all61_scene_probe_v1/plan.json` 有 60 条可执行任务，`unavailable_scenes.json` 另列 1 个不可用场景，因此计划口径为 **61**。60 条 guard 启动命令均从对应 `*_guard.json` 还原；实际 episode 输出目录和 all61 episode summary 均为 **60**。

不可用任务：`XcA2TqTSSAj`（没有分配到可运行 episode_id；不是成功或失败样本）。

## 61 个目录 vs 82 份 summary.json：实际 listing/diff

严格按前缀 `20260910_all61_*` 列目录得到 **121** 个：其中 `_cache` **60** 个、非 cache **61** 个（60 个 episode 目录 + 1 个 `scene_probe_v1` 目录）。因此台账所说的“61 个目录”对应非 cache 命名空间；严格 glob 若不排除 `_cache` 会得到 121。

整棵 evidence tree 的 `summary.json` 实测 **82** 份 = all61 episode 目录内 **60** 份 + 非 all61 的 **22** 份。60 个 `_cache` 目录贡献 **0** 份 summary；差额不是 retry 子目录，而是下面列出的 22 个其它实验输出：

- `20260910_contact_probe/summary.json`
- `20260910_contact_probe_v2/summary.json`
- `20260910_ep1_continuous_v2/summary.json`
- `20260910_ep1_v1/summary.json`
- `20260910_ep1_v2/summary.json`
- `20260910_ep1_v3/summary.json`
- `20260910_ep7601_cache_v3/summary.json`
- `20260910_pilot6_ep1254/summary.json`
- `20260910_pilot6_ep2672/summary.json`
- `20260910_pilot6_ep6908_long_v2/summary.json`
- `20260910_pilot6_ep7247/summary.json`
- `20260910_pilot6_ep8483/summary.json`
- `20260911_ep1065_h028_verified_v1/summary.json`
- `20260911_ep1065_h036_wd_v1/summary.json`
- `20260911_ep1065_h040_wd_v1/summary.json`
- `20260911_ep1065_settle_v1/summary.json`
- `20260911_ep1065_sync_v1/summary.json`
- `20260911_ep1065_sync_verified_v4/summary.json`
- `20260911_height_ep2213_v2/summary.json`
- `20260911_spawn_ep1065_corrected_v4/summary.json`
- `20260911_spawn_ep1065_v1/summary.json`
- `20260911_warmup_ep1065_v4/summary.json`

## Scene 覆盖

实测覆盖 **60 个不同 scene**（每个产生的 episode 对应一个 scene）。缺失 scene：`XcA2TqTSSAj`。

仅有失败样本的 scene（`status == failed`）：

- `29hnd4uzFmX`
- `7y3sRwLe3Va`
- `EDJbREhghzL`
- `SN83YJsR3w2`
- `V2XKFyX4ASd`
- `e9zR4mvMWw7`

没有 success 的 scene 共 26 个；其中 timeout/environment termination 与上述 warmup-only failed 的区分见逐 episode 表。

## ep1065 warmup 失败

all61 首次样本 `20260910_all61_ep1065`（scene `SN83YJsR3w2`）只写入 40 条 warmup 记录，未产生 scored low-level/action/frame 轨迹。`summary.json` 的 verbatim error：

```text
RuntimeError('warmup terminated; retained pre-reset evidence')
```

对应 guard 的 verbatim error：`RuntimeError('collector did not report successful trajectory')`。`20260910_all61_ep1065.log` 中没有 `[Error]`、`error:` 或 `Traceback` 行（实测提取为空），所以不能伪造一个不存在的 log traceback；失败信息由 summary/guard 记录。

围绕 ep1065 的 guard 尝试共 16 次：10 次有 trajectory summary 但 warmup rejected/terminated，6 次没有 trajectory summary，另有 1 次仅做 episode-load validation。所有 trajectory 版本均未得到成功 summary；validation 成功不等于轨迹重跑成功。

其它 ep1065 尝试日志中出现的 verbatim error 行：

```text
2026-09-11 09:14:43 [548,851ms] [Error] [omni.physx.plugin] Subscribtion cannot be changed during the event call.
2026-09-11 09:14:44 [549,890ms] [Error] [omni.physx.plugin] Subscribtion cannot be changed during the event call.
collect_r2r_continuous_v1_warmup_v3.py: error: unrecognized arguments: --spawn-height-offset 0.32
```

## 逐 episode 审计表

`steps` 优先取 `summary.low_level_records`；warmup 失败样本无该字段且 `low_level.jsonl` 为 0 行，因此记为 0，并在 JSON 中标注 provenance。`final_distance_to_goal` 对 warmup 失败为缺失。

| episode_id | scene | status | steps | final_distance_to_goal (m) | terminated_by | summary.json SHA-256 |
|---:|---|---|---:|---:|---|---|
| 55 | GdvgFV5R1Z5 | success | 1911 | 0.296499035 | success | 1e95f20ae7b945ad72011eec7b9c74eb9131580e85efa4d21691899da01d2339 |
| 127 | 1LXtFkjw3qL | success | 1889 | 0.272808679 | success | bbf501d9926a7478748850040cddf8f8dcfe8944bc90ac63e037aed45efb1865 |
| 181 | pRbA3pwrgk9 | timeout | 6000 | 4.74297512 | timeout | 829695b9448b6ee288b479c7402813bb90b418f1962141ef306bd69a7452dc14 |
| 575 | 17DRP5sb8fy | timeout | 6000 | 6.23875911 | timeout | e1a8333450996bbd36326edfd1ab1b479939f477200077018e23e4a09fc5338d |
| 1023 | JmbYfDe2QKZ | timeout | 6000 | 6.41074791 | timeout | dbe3526428b917517c9001840ed2dcdedc627307161c6f7aa8593a18db701d65 |
| 1065 | SN83YJsR3w2 | failed | 0 | — | warmup | 265a0bef8c0c3be3470dba43e00fe39261dad999406bd36c46ccb5f84733d14f |
| 1110 | gTV8FGcVJC9 | success | 2034 | 0.238025711 | success | 17760835bea15017828ef63ddebb25aeba17c8529ecbe170d52a31837dce131c |
| 1201 | jh4fc5c5qoQ | success | 1570 | 0.236243205 | success | 5d4db182a2a6817791517bbb8007d12338f9ce9961900bb7588c1e1828a89154 |
| 1415 | YmJkqBEsHnH | success | 1551 | 0.297802018 | success | 3a706efeb567109dec42623ca9e317868c831e1c679dc395d250861dfb1ad127 |
| 1919 | 2n8kARJN3HM | success | 1900 | 0.247392542 | success | 5f09994feac56ffff0726a69016069a5a190efc812e3c7c7278918c7d31046c3 |
| 2021 | gZ6f7yhEvPG | timeout | 6000 | 3.63991779 | timeout | 0e7d60adb2b74488ad294ed9704294897e5a34ab7e5cba878b66431b8a1c6c45 |
| 2567 | sKLMLpTHeUy | success | 2091 | 0.261912502 | success | 4b359bcd98351c67d75eb234e3a838a377967792d9c07cc0dc50e3829677aa82 |
| 2681 | JeFG25nYj2p | success | 2138 | 0.297491549 | success | 444f1088720c3347d5ace256ff699f0ed1ad16e58fe350cb0c3efea96a4683c2 |
| 2798 | Vvot9Ly1tCj | success | 1812 | 0.248348244 | success | 193e045dda762f97a003f7c5bebb542a0449428220b99ebfde386f47caf8c433 |
| 2915 | 5q7pvUzZiYa | timeout | 6000 | 4.80877322 | timeout | 3434d6fd4b4c62b1c3548e674fee45779ebf27c714c92873ba11d0c11a28fb8e |
| 2924 | sT4fr6TAbpF | success | 1983 | 0.228311308 | success | 0e9645f3d3bd6ebe354e878e90f9e06b93ca3a1a6453f7c670b9be8731ef9cfd |
| 2954 | HxpKQynjfin | environment_termination | 72 | 4.72775032 | environment_termination | afef921e97bf44ef3c165e5891a0792958ac1013ca65e9a60aad100f292a3f16 |
| 3266 | Uxmj2M2itWa | success | 2339 | 0.249528362 | success | e5e776af3c9caae0c1f7ca04e496104333216ff14e2d2431bb071a4065d09741 |
| 3302 | B6ByNegPMKs | success | 1922 | 0.296591587 | success | edaf80dc7762754dfa5f9996c042440773d1a8304f133d400ab03eef7848c503 |
| 3431 | qoiz87JEwZ2 | success | 2098 | 0.236578213 | success | 5eafdf0b7626c04bca9d5c4a95057cfa6f45eb9e03d84bccd01b9f3f9823e6e1 |
| 3506 | cV4RVeZvu5T | timeout | 6000 | 8.27916021 | timeout | 51c2303a6f759e42a72cf901f979b892d3d1cc90e06139a93d3a650fd00d432f |
| 3572 | EDJbREhghzL | failed | 0 | — | warmup | 265a0bef8c0c3be3470dba43e00fe39261dad999406bd36c46ccb5f84733d14f |
| 3827 | 759xd9YjKW5 | timeout | 6000 | 5.53727215 | timeout | d27f1096c881f19ee09e3d2e9d2cfccd5974b5df2d403a2aac66a9feea11df5b |
| 4385 | Pm6F8kyY3z2 | timeout | 6000 | 5.37709102 | timeout | c0dc7f620fe555f05b807359a799a0773c5e914530133407771c284b9c4ab5a6 |
| 4451 | 82sE5b5pLXE | success | 2002 | 0.248849708 | success | 844e44475b59c9e5caa674a0d9ccc33ccc2ed92b9cb3cd3599d903edc6755a42 |
| 4469 | 29hnd4uzFmX | failed | 0 | — | warmup | 265a0bef8c0c3be3470dba43e00fe39261dad999406bd36c46ccb5f84733d14f |
| 4499 | VVfe2KiqLaN | timeout | 6000 | 8.61604456 | timeout | b9809acd58452fd5b881c1b00193079aa86abf2bda20c87550bba74f6fd39fb8 |
| 4802 | V2XKFyX4ASd | failed | 0 | — | warmup | 265a0bef8c0c3be3470dba43e00fe39261dad999406bd36c46ccb5f84733d14f |
| 4961 | r1Q1Z4BcV1o | success | 1849 | 0.222326831 | success | 658da09893cda06d6ed69e457b8362dda60c0859cd4ad4fbcc2d81e9fef8e504 |
| 5267 | ac26ZMwG7aT | success | 2119 | 0.218817807 | success | 1bcb55612c5f65a6d1461b0bc5c2a17440406b9033a591c3333b36c864cd6a19 |
| 5576 | 7y3sRwLe3Va | failed | 0 | — | warmup | 265a0bef8c0c3be3470dba43e00fe39261dad999406bd36c46ccb5f84733d14f |
| 5861 | mJXqzFtmKg4 | success | 2011 | 0.216728624 | success | 7d702f6d9df131ac4539f1aeed867c53dad636b7c0f03477b94e92e017bad4b0 |
| 5864 | S9hNv5qa7GM | success | 2004 | 0.264134818 | success | 5ee7d82024c15037da04db9556776a92eff426930b7063cdd867cd8ec4e41251 |
| 6011 | PuKPg4mmafe | success | 1781 | 0.230962179 | success | 681ecf39300e112429aaf45c679c76b7201689e84d4b263adcb307723804e2b0 |
| 6215 | r47D5H71a5s | success | 1817 | 0.226031221 | success | 6c46b37c54ef7f36deb1673ee0ae3185149eb3ff02f62f9a21e97031ecc3913f |
| 6398 | 8WUmhLawc2A | timeout | 6000 | 7.13846755 | timeout | b1f1ebe6745939cd0db118900d93b5ff39e1f614241a72c14f13f33ab418d00c |
| 6716 | VzqfbhrpDEA | success | 1920 | 0.224446944 | success | 9e00a8e90308edd91b6f58a272ce7aae538ca72f16f72cd647e58fa0b4cc8ef4 |
| 6791 | rPc6DW4iMge | success | 1911 | 0.28831288 | success | d6ab29f242bb2b0d76c41b751e4987f3985fbaff30acb2dd53124c5ce71c530b |
| 7166 | aayBHfsNo7d | success | 4173 | 0.293376036 | success | de7488943b5b346d1353872b206c713ba77ce275aa1e0a31d3ec44b07ba61e48 |
| 7205 | p5wJjkQkbXX | timeout | 6000 | 6.12813892 | timeout | fbcfde57204a079a8a34c70d748c600aa4aef852f92fdc49ad47cb5190946175 |
| 7223 | i5noydFURQK | timeout | 6000 | 7.07955839 | timeout | df9ffa20a35ce8105487fcd41d8a676b54af738b4ec70028516ce8b8b9893b0a |
| 7376 | VLzqgDo317F | success | 2227 | 0.263486375 | success | 9150f10dd3858e9585534626299083c0ca83d05bfbba35d743b9cd09b3255811 |
| 8078 | PX4nDJXEHrG | timeout | 6000 | 6.06891058 | timeout | 84df8fc272c07aba1e49777a55bb7e50cc4eb25c2e65ebd661e1db3d8e05aeb6 |
| 8432 | D7G3Y4RVNrH | environment_termination | 448 | 8.04285781 | environment_termination | 800b5112c470099ece2b2e4810ffd560cfb32942cefc26c5aef1e573659957b8 |
| 8483 | E9uDoFAP3SH | success | 1686 | 0.202690129 | success | 07bcecce6683d704e785ce38467ffc1e9e1de984e1ed2f69b139eb16d9d84f4d |
| 8624 | ur6pFq6Qu1A | success | 2160 | 0.284042691 | success | 5eb93bbeda90008e2f548965bcd00b449675dd062ad5fc0d2578fdc69f3f88ba |
| 8987 | s8pcmisQ38h | timeout | 6000 | 6.46125274 | timeout | 8e29a55f99555c9964d9977e21a768651635854c6003c0970ce55abcf5af815f |
| 9221 | ULsKaCPVFJR | success | 1826 | 0.262108978 | success | e40ab785dcd7ad08d8a5add68e6f06b8858b7141879b1761c80d5adab5e3fded |
| 9293 | e9zR4mvMWw7 | failed | 0 | — | warmup | 265a0bef8c0c3be3470dba43e00fe39261dad999406bd36c46ccb5f84733d14f |
| 9305 | dhjEzFoUFzH | success | 2017 | 0.257832175 | success | ee3bc45f98d6f3fe133a0a09de9a2b5cb8cacbdf935ccb807a3a5576b9b5624a |
| 9512 | JF19kD82Mey | timeout | 6000 | 3.48661935 | timeout | 7f49454b07af7cf861a0ee43aeaeff30ce41f6ec955c30605e34fcff1d9c19a3 |
| 9554 | ZMojNkEp431 | timeout | 6000 | 6.00652722 | timeout | 6f73d01f3005761f975dbbae1a4bf080dedb67dff69c71259ad4b054cf43282a |
| 9776 | uNb9QFRL6hY | success | 2220 | 0.23113733 | success | cb0ce1dc77468bbaa12fa6b8343ff0a1ae120c1d28fdc1748664273bbcabea94 |
| 9893 | b8cTxDM8gDG | success | 2076 | 0.231788421 | success | e718a9f1846f3f5f417ea7c666ccb43ddb758bb7f22b089d3a2577b6fe42a933 |
| 9956 | VFuaQ6m2Qom | timeout | 6000 | 7.981468 | timeout | 66f743d4945513420393e8bb55e873a3c87f0fc64d5310be125e452d639fd712 |
| 10235 | 5LpN3gDmAk7 | success | 2157 | 0.272016159 | success | 6c585efbe84ce49f6994e958998db591c4c386660dc78e6d2af557428914a988 |
| 10238 | vyrNrziPKCB | success | 1813 | 0.282013895 | success | 4994b1ef63d01b2b47e9c2d3ed1f0650e86987cc11b33944e0376f98f179a359 |
| 10301 | kEZ7cmS4wCh | timeout | 6000 | 7.68180249 | timeout | 892b4b35ad4e8be1d954a590940735b6180831d3d4ee70e9549b2b01e04f22fc |
| 10610 | D7N2EKCX4Sj | success | 2003 | 0.259757962 | success | 283cc5c7dcbedbe18a99b9ab80c898f2a8d25e96add4d441612b40d842c7ca81 |
| 10721 | 1pXnuDYAj8r | success | 1822 | 0.275470565 | success | fa2b7d960278d5c7d685682c550fda875f54a2b0845c158065eca9e11b093d8e |

## 仍缺少的 QC 前置条件

- Expected scene XcA2TqTSSAj is listed unavailable by scene_probe; no all61 episode output exists for it.
- Six produced scenes have only warmup-failed samples (no scored trajectory).
- An independently QC'd 61-scene success-rate denominator is therefore unavailable; one scene is missing and six samples never entered scored rollout.
- Existing per-episode audit records warn that contact-force data are all zero/unverified; independent QC must validate terminal holds, resets, source/task identity, and evidence completeness before reporting a success rate.

因此当前只能报告“60 个已产出样本的实测分布”，不能把它升级为 61-scene 的 QC 成功率。
