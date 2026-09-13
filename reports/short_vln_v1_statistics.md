# Short-VLN V1 Statistics (M3)

## Generation contract

- Source episodes: **1077**
- Generated short episodes: **1077** (one per source episode)
- Source geometry: contiguous slices of official `gt_locations`
- Path constraint: **1–4 m** (3D polyline length)
- Major-turn constraint: **≤1**, threshold 45°, same-sign components merged within 0.75 m
- Turn instruction legs: each at least 0.75 m
- Language: deterministic geometry templates; no external LLM and no landmark labels
- Start rotation: synthesized WXYZ yaw aligned to the outgoing path tangent
- Seed: `20260831`
- Unseen selection: 2 whole scenes whose source-episode count is closest to the 20% target

## Acceptance summary

- Validation: **PASS**
- JSON reload: PASS
- Unique source usage: **1077 / 1077**
- Train/unseen scene leakage: **0**
- Seen-val scenes absent from train: **0**
- Short instructions identical to original full-route instruction: **0**
- Automatically audited spot checks: **20**

## Dataset distributions

| Metric | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| Path length (m) | 1.500 | 2.750 | 2.720 | 4.000 |
| Major turn count | 0 | 0 | 0.40 | 1 |

## Split counts

| Split | Episodes | Scenes |
|---|---:|---:|
| train | 735 | 9 |
| seen-val | 129 | 9 |
| unseen-test | 213 | 2 |

## Instruction categories

| Category | Episodes |
|---|---:|
| straight | 647 |
| left_turn | 171 |
| right_turn | 259 |

## Per-scene split audit

| Scene | Train | Seen-val | Unseen-test |
|---|---:|---:|---:|
| `2azQ1b91cZZ` | 0 | 0 | 204 |
| `8194nk5LbLH` | 0 | 0 | 9 |
| `EU6Fwq7SyZv` | 46 | 8 | 0 |
| `QUCTc6BB5sX` | 204 | 36 | 0 |
| `TbHJrupSAjP` | 92 | 16 | 0 |
| `X7HyMhZNoso` | 59 | 10 | 0 |
| `Z6MFQCViBuw` | 56 | 10 | 0 |
| `oLBMNvg9in8` | 20 | 4 | 0 |
| `pLe4wQe7qrG` | 2 | 1 | 0 |
| `x8F5xyUWy9e` | 26 | 4 | 0 |
| `zsNo4HB9uLZ` | 230 | 40 | 0 |

## Known semantic boundary

The generated instructions are fully supported by path geometry, but M3 does not claim a landmark is visible in RGB. The original full-route instruction is retained only as `source_metadata.original_instruction_text_not_for_training`. M4 must use the short `instruction` field and should visually verify rollout starts before collection.
