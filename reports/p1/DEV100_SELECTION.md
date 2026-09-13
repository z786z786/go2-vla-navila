# P1-DEV100 selection

Frozen 2026-09-13 from the verified input SHA-256 `ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec` using seed **20260913**. Each physical trajectory contributes at most one episode. The fixed episode rule is: choose the numerically smallest original `episode_id` among its three episodes. Routes are ranked by SHA-256 of `f"{seed}:{scene}:{trajectory_id}"`, with numeric trajectory tie-break; output is sorted by `(scene, episode_id)`.

| scene | episodes | routes | allocated |
|---|---:|---:|---:|
| zsNo4HB9uLZ | 270 | 90 | 26 |
| QUCTc6BB5sX | 240 | 80 | 23 |
| 2azQ1b91cZZ | 204 | 68 | 20 |
| TbHJrupSAjP | 108 | 36 | 11 |
| X7HyMhZNoso | 69 | 23 | 7 |
| Z6MFQCViBuw | 66 | 22 | 7 |
| EU6Fwq7SyZv | 54 | 18 | 6 |
| x8F5xyUWy9e | 30 | 10 | 3 |
| oLBMNvg9in8 | 24 | 8 | 2 |
| 8194nk5LbLH | 9 | 3 | 1 |
| pLe4wQe7qrG | 3 | 1 | 1 |
| **Total** | **1077** | **359** | **100** |

Every scene receives one route. The other 89 seats use route-count proportional quotas and largest remainder; `pLe4wQe7qrG` has one route and therefore contributes exactly one. The choice maximizes distinct physical routes for this screening subset; instruction variants are covered by the separately approved misleading-instruction control.

Independent checks: raw input yielded 1077 episodes, 359 trajectories, three episodes per trajectory, and the row counts above. `len(episode_ids)==100`; `len(set(trajectory_ids))==100`; all selected episode IDs exist. Allocation was recomputed independently and matches. A zero-argument rerun was byte-identical. Evaluation scenes have empty intersection with the 61-scene training metadata list. One source trajectory (`4668`) has differing `start_rotation` across its three episodes; the physical-route invariant used by the dispatch (`scene_id`, start position, goals, `gt_locations`) remains identical.

MISSING: none.
