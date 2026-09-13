# P3-T1E — NaVILA R2R 全量 leakage join 与动作分布

The verified full annotation file is joined to R2R by normalized raw instruction text; the train-implied scene/eval11 intersection is `[]`, and `val_unseen − train` contains 0 matched normalized instructions.

## Input verification

| Input path | Measured SHA-256 | Measured bytes |
|---|---|---:|
| `/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_annotations.json` | `3587c020cc03807cfc03c454bcbb37a3d409139e125e1ea047fa1fbcf8c4cc1a` | 317267756 |
| `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/train/train.json.gz` | `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34` | 1772932 |
| `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/val_seen/val_seen.json.gz` | `7fc94841ebbd2eac0d398e020a2f638426948beaf4a561f6ee310dd67cddce55` | 172881 |
| `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz` | `d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8` | 325775 |
| `/mnt/wxh/go2_short_vln/data/r2r_vlnce/R2R_VLNCE_v1-3/test/test.json.gz` | `69ed5962fac658d34be5739c1f8f15805cf5ce9bfa28f64a338da338244578b7` | 335501 |
| `/mnt/wxh/go2_short_vln/assets/vln_ce_isaac/vln_ce_isaac_v1.json.gz` | `ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec` | 501265 |

Every prescribed input was SHA-256 verified before any split or annotations parsing. The full annotations source is the 317,267,756-byte file at the dispatch-prescribed `downloads/navila_probe` path; the similarly named truncated copy was not read.

## Leakage join

| Quantity | Value |
|---|---:|
| Full annotation records | 353894 |
| Distinct normalized instructions | 10815 |
| `val_unseen − train` matched instructions | 0 |
| `val_unseen − train` affected records | 0 |
| Train-implied raw scenes | 61 |
| Train-implied normalized scene stems | 61 |
| eval11 scenes | 11 |
| Train/eval11 scene-stem intersection | 0 (`[]`) |
| Unexplained records | 0 |
| Unexplained normalized instructions | 0 |
| Excluded ambiguous instructions | 3 |
| Excluded affected records | 84 |

| R2R split | Matched normalized instructions | Matched annotation records |
|---|---:|---:|
| train | 10815 | 353894 |
| val_seen | 1 | 28 |
| val_unseen | 2 | 49 |
| test | 1 | 35 |

Matched train raw scene IDs (sorted): `['mp3d/17DRP5sb8fy/17DRP5sb8fy.glb', 'mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb', 'mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb', 'mp3d/29hnd4uzFmX/29hnd4uzFmX.glb', 'mp3d/2n8kARJN3HM/2n8kARJN3HM.glb', 'mp3d/5LpN3gDmAk7/5LpN3gDmAk7.glb', 'mp3d/5q7pvUzZiYa/5q7pvUzZiYa.glb', 'mp3d/759xd9YjKW5/759xd9YjKW5.glb', 'mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb', 'mp3d/82sE5b5pLXE/82sE5b5pLXE.glb', 'mp3d/8WUmhLawc2A/8WUmhLawc2A.glb', 'mp3d/B6ByNegPMKs/B6ByNegPMKs.glb', 'mp3d/D7G3Y4RVNrH/D7G3Y4RVNrH.glb', 'mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.glb', 'mp3d/E9uDoFAP3SH/E9uDoFAP3SH.glb', 'mp3d/EDJbREhghzL/EDJbREhghzL.glb', 'mp3d/GdvgFV5R1Z5/GdvgFV5R1Z5.glb', 'mp3d/HxpKQynjfin/HxpKQynjfin.glb', 'mp3d/JF19kD82Mey/JF19kD82Mey.glb', 'mp3d/JeFG25nYj2p/JeFG25nYj2p.glb', 'mp3d/JmbYfDe2QKZ/JmbYfDe2QKZ.glb', 'mp3d/PX4nDJXEHrG/PX4nDJXEHrG.glb', 'mp3d/Pm6F8kyY3z2/Pm6F8kyY3z2.glb', 'mp3d/PuKPg4mmafe/PuKPg4mmafe.glb', 'mp3d/S9hNv5qa7GM/S9hNv5qa7GM.glb', 'mp3d/SN83YJsR3w2/SN83YJsR3w2.glb', 'mp3d/ULsKaCPVFJR/ULsKaCPVFJR.glb', 'mp3d/Uxmj2M2itWa/Uxmj2M2itWa.glb', 'mp3d/V2XKFyX4ASd/V2XKFyX4ASd.glb', 'mp3d/VFuaQ6m2Qom/VFuaQ6m2Qom.glb', 'mp3d/VLzqgDo317F/VLzqgDo317F.glb', 'mp3d/VVfe2KiqLaN/VVfe2KiqLaN.glb', 'mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.glb', 'mp3d/VzqfbhrpDEA/VzqfbhrpDEA.glb', 'mp3d/XcA2TqTSSAj/XcA2TqTSSAj.glb', 'mp3d/YmJkqBEsHnH/YmJkqBEsHnH.glb', 'mp3d/ZMojNkEp431/ZMojNkEp431.glb', 'mp3d/aayBHfsNo7d/aayBHfsNo7d.glb', 'mp3d/ac26ZMwG7aT/ac26ZMwG7aT.glb', 'mp3d/b8cTxDM8gDG/b8cTxDM8gDG.glb', 'mp3d/cV4RVeZvu5T/cV4RVeZvu5T.glb', 'mp3d/dhjEzFoUFzH/dhjEzFoUFzH.glb', 'mp3d/e9zR4mvMWw7/e9zR4mvMWw7.glb', 'mp3d/gTV8FGcVJC9/gTV8FGcVJC9.glb', 'mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb', 'mp3d/i5noydFURQK/i5noydFURQK.glb', 'mp3d/jh4fc5c5qoQ/jh4fc5c5qoQ.glb', 'mp3d/kEZ7cmS4wCh/kEZ7cmS4wCh.glb', 'mp3d/mJXqzFtmKg4/mJXqzFtmKg4.glb', 'mp3d/p5wJjkQkbXX/p5wJjkQkbXX.glb', 'mp3d/pRbA3pwrgk9/pRbA3pwrgk9.glb', 'mp3d/qoiz87JEwZ2/qoiz87JEwZ2.glb', 'mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.glb', 'mp3d/r47D5H71a5s/r47D5H71a5s.glb', 'mp3d/rPc6DW4iMge/rPc6DW4iMge.glb', 'mp3d/s8pcmisQ38h/s8pcmisQ38h.glb', 'mp3d/sKLMLpTHeUy/sKLMLpTHeUy.glb', 'mp3d/sT4fr6TAbpF/sT4fr6TAbpF.glb', 'mp3d/uNb9QFRL6hY/uNb9QFRL6hY.glb', 'mp3d/ur6pFq6Qu1A/ur6pFq6Qu1A.glb', 'mp3d/vyrNrziPKCB/vyrNrziPKCB.glb']`

Matched train normalized scene stems (sorted): `['17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '2n8kARJN3HM', '5LpN3gDmAk7', '5q7pvUzZiYa', '759xd9YjKW5', '7y3sRwLe3Va', '82sE5b5pLXE', '8WUmhLawc2A', 'B6ByNegPMKs', 'D7G3Y4RVNrH', 'D7N2EKCX4Sj', 'E9uDoFAP3SH', 'EDJbREhghzL', 'GdvgFV5R1Z5', 'HxpKQynjfin', 'JF19kD82Mey', 'JeFG25nYj2p', 'JmbYfDe2QKZ', 'PX4nDJXEHrG', 'Pm6F8kyY3z2', 'PuKPg4mmafe', 'S9hNv5qa7GM', 'SN83YJsR3w2', 'ULsKaCPVFJR', 'Uxmj2M2itWa', 'V2XKFyX4ASd', 'VFuaQ6m2Qom', 'VLzqgDo317F', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'VzqfbhrpDEA', 'XcA2TqTSSAj', 'YmJkqBEsHnH', 'ZMojNkEp431', 'aayBHfsNo7d', 'ac26ZMwG7aT', 'b8cTxDM8gDG', 'cV4RVeZvu5T', 'dhjEzFoUFzH', 'e9zR4mvMWw7', 'gTV8FGcVJC9', 'gZ6f7yhEvPG', 'i5noydFURQK', 'jh4fc5c5qoQ', 'kEZ7cmS4wCh', 'mJXqzFtmKg4', 'p5wJjkQkbXX', 'pRbA3pwrgk9', 'qoiz87JEwZ2', 'r1Q1Z4BcV1o', 'r47D5H71a5s', 'rPc6DW4iMge', 's8pcmisQ38h', 'sKLMLpTHeUy', 'sT4fr6TAbpF', 'uNb9QFRL6hY', 'ur6pFq6Qu1A', 'vyrNrziPKCB']`

eval11 normalized scene stems (sorted): `['2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'QUCTc6BB5sX', 'TbHJrupSAjP', 'X7HyMhZNoso', 'Z6MFQCViBuw', 'oLBMNvg9in8', 'pLe4wQe7qrG', 'x8F5xyUWy9e', 'zsNo4HB9uLZ']`

## Full action distribution

| Quantity | Value |
|---|---:|
| Action vocabulary size | 10 |
| Majority action | `The next action is move forward 75 cm.` |
| Majority count | 106026 |
| Majority share | 29.959818% |
| Stop action | `I think I should stop because I have finished the instruction.` |
| Stop count | 32457 |
| Stop share | 9.171390% |

Full-dataset rates differ from the prior 6,712-record reference sample: majority share is 29.959818% (reference 29.74%, delta +0.219818 percentage points) and stop share is 9.171390% (reference 9.5%, delta -0.328610 percentage points). The full action vocabulary remains 10 entries. These values were obtained by counting every decoded record in the verified 317,267,756-byte array; no sampling or agreement-forcing adjustment was used.

| Exact `a` value | Count |
|---|---:|
| `I think I should stop because I have finished the instruction.` | 32457 |
| `The next action is move forward 25 cm.` | 37376 |
| `The next action is move forward 50 cm.` | 28047 |
| `The next action is move forward 75 cm.` | 106026 |
| `The next action is turn left 15 degree.` | 31465 |
| `The next action is turn left 30 degree.` | 15348 |
| `The next action is turn left 45 degree.` | 31510 |
| `The next action is turn right 15 degree.` | 31199 |
| `The next action is turn right 30 degree.` | 14436 |
| `The next action is turn right 45 degree.` | 26030 |

## Video and frame structure

The percentile convention is `nearest rank: sorted[ceil(p*n)-1]`. Decision-point counts use unique numeric `<step>` values per video prefix; repeated rows at a step are instruction variants and remain included in the record-level action/frame counts.

| Distribution | Count | Min | P25 | P50 | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision points per video | 10819 | 8 | 20 | 25 | 31 | 38 | 289 |
| Frames per record | 353894 | 1 | 14 | 29 | 46 | 62 | 501 |

| Check | Value |
|---|---:|
| Frames contiguous `0..N-1` records | 353894 |
| Frames non-contiguous records | 0 |
| Videos with non-decreasing `n_frames` by numeric step | 10819 |
| Videos with a monotonicity counterexample | 0 |

## Method and limits

The parser uses `json.JSONDecoder().raw_decode` on 1 MiB text chunks and discards the processed prefix after each record; it never calls `json.load()` on the 317 MB annotations array. All split membership uses exactly `' '.join(s.split()).strip().lower()` on both sides. Scene identity is recovered only from the matching R2R split episodes, not from `video_id` or numeric IDs. A normalized instruction found in train and either val_unseen or test is listed in the complete exclusion artifact and excluded under the dispatch rule. Text normalization can reveal identical text but cannot independently prove an original episode's provenance when R2R split text is duplicated; that ambiguity is why the specified exclusion rule is applied. The `generated_at` field is the newest verified input file mtime (rather than wall-clock time) so no-argument reruns reproduce the JSON bytes deterministically.

MISSING items: none.
