# VLN-CE-Isaac Episode Statistics (M2)

## Dataset and method

- Dataset: `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz`
- Dataset episodes: **1077**
- Dataset scenes: **11**
- Deterministic scene-stratified random sample: **100 episodes**, seed `20260831`
- Reference path length: sum of consecutive 3D Euclidean waypoint distances.
- Major turn: absolute wrapped change between consecutive non-zero **XY** segment headings, threshold **45 degrees**. Height changes are not turns.
- `reference_path` is the sparse task/reference route. `gt_locations` is the dense expert route consumed by the official PD planner and distance metric; they are reported separately.

## Observed schema

The gzip JSON top level contains only `episodes`. Fields below are the union observed over all 1077 episodes, not guessed names.

- Episode: `episode_id, episode_new_id, goals, gt_actions, gt_forward_steps, gt_locations, info, instruction, reference_path, scene_id, start_position, start_rotation, trajectory_id`
- Instruction: `instruction_text, instruction_tokens`
- Goal: `position, radius`
- Info: `geodesic_distance`

## Dataset integrity checks

| Check | Result |
|---|---:|
| Unique `episode_id` values | 1077 / 1077 |
| Distinct episode key sets | 1 |
| Distinct instruction key sets | 1 |
| Distinct goal key sets | 1 |
| Empty instruction text | 0 |
| Reference paths with fewer than 2 points | 0 |
| Expert paths with fewer than 2 points | 0 |
| Start vs `reference_path[0]` mismatch (>1e-6 m) | 0 |
| `goals[0]` vs `reference_path[-1]` mismatch (>1e-6 m) | 0 |

## Sample distributions

| Metric | Min | P25 | Median | Mean | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Reference path length, 3D (m) | 5.19 | 7.46 | 9.17 | 9.45 | 10.84 | 18.07 |
| Reference waypoint count | 4.0 | 5.0 | 6.0 | 5.8 | 7.0 | 7.0 |
| Major turn count | 0.0 | 1.0 | 1.0 | 1.2 | 2.0 | 4.0 |
| Expert waypoint count | 19.0 | 28.8 | 35.0 | 36.5 | 42.0 | 69.0 |
| Start-goal distance, 3D (m) | 2.29 | 5.77 | 7.02 | 7.45 | 8.95 | 15.40 |

## Scene coverage in sample

The sample covers **11 scenes**.

| Scene | Episodes |
|---|---:|
| `2azQ1b91cZZ` | 17 |
| `8194nk5LbLH` | 1 |
| `EU6Fwq7SyZv` | 6 |
| `QUCTc6BB5sX` | 18 |
| `TbHJrupSAjP` | 16 |
| `X7HyMhZNoso` | 6 |
| `Z6MFQCViBuw` | 6 |
| `oLBMNvg9in8` | 2 |
| `pLe4wQe7qrG` | 2 |
| `x8F5xyUWy9e` | 3 |
| `zsNo4HB9uLZ` | 23 |

## Fields needed by short-VLN M3

- Stable provenance: `episode_id`, `episode_new_id`, `trajectory_id`, `scene_id`.
- Start pose: `start_position`, `start_rotation` (official code passes it as WXYZ).
- Goal: `goals[0].position`; the official demo also places its goal marker at `reference_path[-1]`, so both must be retained and consistency-checked.
- Sparse geometry: `reference_path` for segment length and turn filtering.
- Dense expert geometry: `gt_locations`, plus `gt_actions` and `gt_forward_steps` for audit only.
- Language: `instruction.instruction_text`; it describes the full original route and must not be reused unchanged after arbitrary path cropping.
- Navigation metadata: `info.geodesic_distance`, goal `radius`.

## Deterministic sample IDs

52, 71, 78, 81, 82, 129, 136, 146, 165, 166, 190, 207, 245, 248, 256, 260, 261, 265, 310, 312, 321, 366, 367, 382, 407, 420, 448, 506, 507, 577, 611, 636, 646, 656, 677, 719, 774, 782, 783, 785, 786, 802, 869, 918, 932, 954, 955, 976, 1077, 1078, 1095, 1106, 1135, 1140, 1150, 1156, 1167, 1171, 1179, 1184, 1191, 1202, 1280, 1284, 1289, 1314, 1335, 1337, 1374, 1388, 1407, 1436, 1471, 1474, 1478, 1488, 1513, 1548, 1561, 1564, 1590, 1600, 1608, 1613, 1635, 1638, 1658, 1684, 1710, 1713, 1757, 1777, 1779, 1791, 1797, 1801, 1804, 1808, 1820, 1833
