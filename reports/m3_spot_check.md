# M3 Short-Episode Spot Check

These 20 deterministic checks cover every split, instruction category, and source scene. `PASS` means the path is an exact contiguous source slice, length is 1–4 m, major turns are ≤1, the instruction category and distances agree with recomputed geometry, and the original long instruction was not reused. This is a geometry/language audit, not a claim of landmark visibility.

| Short ID | Source ID | Scene | Split | Source indices | Length (m) | Turns | Category | Short instruction | Audit |
|---|---:|---|---|---|---:|---:|---|---|---|
| short_vln_v1_0019 | 32 | `x8F5xyUWy9e` | seen-val | `[10, 22]` | 3.026 | 1 | left_turn | Move forward about 1.6 meters, turn left, then continue about 1.4 meters and stop. | PASS |
| short_vln_v1_0095 | 141 | `2azQ1b91cZZ` | unseen-test | `[42, 53]` | 2.750 | 0 | straight | Move forward along the available route for about 2.8 meters, then stop. | PASS |
| short_vln_v1_0129 | 190 | `QUCTc6BB5sX` | train | `[20, 28]` | 2.061 | 1 | left_turn | Move forward about 1.1 meters, turn left, then continue about 1.0 meters and stop. | PASS |
| short_vln_v1_0132 | 193 | `2azQ1b91cZZ` | unseen-test | `[7, 19]` | 3.000 | 0 | straight | Move forward along the available route for about 3.0 meters, then stop. | PASS |
| short_vln_v1_0214 | 335 | `QUCTc6BB5sX` | train | `[19, 29]` | 2.517 | 0 | straight | Move forward along the available route for about 2.5 meters, then stop. | PASS |
| short_vln_v1_0363 | 598 | `X7HyMhZNoso` | train | `[13, 27]` | 3.500 | 0 | straight | Move forward along the available route for about 3.5 meters, then stop. | PASS |
| short_vln_v1_0366 | 601 | `X7HyMhZNoso` | train | `[30, 41]` | 2.750 | 1 | right_turn | Move forward about 1.5 meters, turn right, then continue about 1.2 meters and stop. | PASS |
| short_vln_v1_0434 | 732 | `8194nk5LbLH` | unseen-test | `[30, 43]` | 3.258 | 1 | left_turn | Move forward about 0.9 meters, turn left, then continue about 2.3 meters and stop. | PASS |
| short_vln_v1_0463 | 785 | `QUCTc6BB5sX` | train | `[12, 25]` | 3.259 | 1 | right_turn | Move forward about 1.5 meters, turn right, then continue about 1.8 meters and stop. | PASS |
| short_vln_v1_0512 | 888 | `EU6Fwq7SyZv` | seen-val | `[12, 25]` | 3.276 | 1 | right_turn | Move forward about 1.5 meters, turn right, then continue about 1.8 meters and stop. | PASS |
| short_vln_v1_0595 | 1028 | `Z6MFQCViBuw` | train | `[18, 29]` | 2.750 | 0 | straight | Move forward along the available route for about 2.8 meters, then stop. | PASS |
| short_vln_v1_0781 | 1334 | `zsNo4HB9uLZ` | seen-val | `[26, 41]` | 3.750 | 1 | left_turn | Move forward about 1.9 meters, turn left, then continue about 1.9 meters and stop. | PASS |
| short_vln_v1_0816 | 1381 | `2azQ1b91cZZ` | unseen-test | `[3, 10]` | 1.750 | 1 | left_turn | Move forward about 0.9 meters, turn left, then continue about 0.9 meters and stop. | PASS |
| short_vln_v1_0840 | 1435 | `zsNo4HB9uLZ` | train | `[5, 18]` | 3.250 | 1 | right_turn | Move forward about 1.7 meters, turn right, then continue about 1.6 meters and stop. | PASS |
| short_vln_v1_0919 | 1562 | `QUCTc6BB5sX` | seen-val | `[10, 22]` | 3.000 | 1 | left_turn | Move forward about 1.5 meters, turn left, then continue about 1.5 meters and stop. | PASS |
| short_vln_v1_0963 | 1645 | `zsNo4HB9uLZ` | train | `[0, 6]` | 1.500 | 0 | straight | Move forward along the available route for about 1.5 meters, then stop. | PASS |
| short_vln_v1_0973 | 1655 | `TbHJrupSAjP` | train | `[29, 39]` | 2.500 | 1 | right_turn | Move forward about 1.2 meters, turn right, then continue about 1.3 meters and stop. | PASS |
| short_vln_v1_1049 | 1773 | `zsNo4HB9uLZ` | seen-val | `[6, 13]` | 1.763 | 1 | right_turn | Move forward about 1.0 meters, turn right, then continue about 0.8 meters and stop. | PASS |
| short_vln_v1_1050 | 1777 | `pLe4wQe7qrG` | train | `[9, 16]` | 1.756 | 1 | right_turn | Move forward about 0.9 meters, turn right, then continue about 0.9 meters and stop. | PASS |
| short_vln_v1_1074 | 1831 | `oLBMNvg9in8` | train | `[18, 28]` | 2.547 | 1 | right_turn | Move forward about 1.5 meters, turn right, then continue about 1.1 meters and stop. | PASS |
